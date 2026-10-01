# SPDX-License-Identifier: Apache-2.0
"""SM75 (Turing) Triton paged cross-attention for NHD KV cache layout.

Raw kv_cache: NHD 4D [num_blocks, nkv, block_size, 2*head_dim].
K = kv_cache[:, :, :, :head_dim], V = kv_cache[:, :, :, head_dim:].

Design: one program per (query_token, query_head) — avoids tl.dot
SMEM pressure by using tl.sum instead of tensor-core matmuls. Each
instance holds HEAD_DIM-sized register vectors, well within SM75's
64 KB SMEM limit. CUDA-graph-safe: all shapes are fixed; variable
context is read from in-place-updated GPU tensors.
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _nhd_paged_cross_attn_fwd(
    # Query:  [num_reqs * query_len, nq, hd]
    Q, q_stride_t, q_stride_h, q_stride_d,
    # Output: [num_reqs * query_len, nq, hd]
    Out, o_stride_t, o_stride_h, o_stride_d,
    # KV cache NHD 4D: [num_blocks, nkv, block_size, 2*hd]
    KV, kv_stride_b, kv_stride_h, kv_stride_l, kv_stride_c,
    # Paged-KV metadata — GPU tensors updated in-place each step
    kv_indptr,       # [num_reqs + 1]  int32 GPU
    kv_indices,      # [total_pages]   int32 GPU (full pre-allocated buffer)
    kv_last_len,     # [num_reqs]      int32 GPU
    # Scaling
    scale,
    # Compile-time constants
    QUERY_LEN: tl.constexpr,    # K+1, typically 8
    BLOCK_SIZE: tl.constexpr,   # page_size, 16
    HEAD_DIM: tl.constexpr,     # head_dim, 128
    GQA_RATIO: tl.constexpr,    # nq // nkv
):
    """One program per (query_token_index, query_head_index).

    Uses tl.sum instead of tl.dot to avoid tl.dot's MMA SMEM tiles,
    which exceed SM75's 64 KB shared-memory limit when HEAD_DIM=128.
    """
    tok_idx = tl.program_id(0)   # 0 .. num_reqs * QUERY_LEN - 1
    q_head  = tl.program_id(1)   # 0 .. nq - 1

    req_id  = tok_idx // QUERY_LEN
    kv_head = q_head // GQA_RATIO

    offs_d = tl.arange(0, HEAD_DIM)
    offs_l = tl.arange(0, BLOCK_SIZE)

    # ── Load query vector: [HEAD_DIM] ─────────────────────────────────────────
    q = tl.load(Q
                + tok_idx * q_stride_t
                + q_head  * q_stride_h
                + offs_d  * q_stride_d).to(tl.float32)

    # ── Online softmax state (scalars) ────────────────────────────────────────
    m   = float("-inf")
    l   = 0.0
    acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    # ── Page range for this request ───────────────────────────────────────────
    page_start = tl.load(kv_indptr + req_id)
    page_end   = tl.load(kv_indptr + req_id + 1)
    last_len   = tl.load(kv_last_len + req_id)
    n_pages    = page_end - page_start

    # ── Iterate over KV pages ─────────────────────────────────────────────────
    for page_i in range(n_pages):
        block_idx = tl.load(kv_indices + page_start + page_i)
        valid     = tl.where(page_i < n_pages - 1, BLOCK_SIZE, last_len)

        kv_base = KV + block_idx * kv_stride_b + kv_head * kv_stride_h

        # K: [BLOCK_SIZE, HEAD_DIM]
        k = tl.load(kv_base
                    + offs_l[:, None] * kv_stride_l
                    + offs_d[None, :] * kv_stride_c,
                    mask=offs_l[:, None] < valid,
                    other=0.0).to(tl.float32)

        # V: [BLOCK_SIZE, HEAD_DIM]
        v = tl.load(kv_base
                    + offs_l[:, None] * kv_stride_l
                    + (HEAD_DIM + offs_d[None, :]) * kv_stride_c,
                    mask=offs_l[:, None] < valid,
                    other=0.0).to(tl.float32)

        # Q·K^T via element-wise + sum — avoids tl.dot MMA SMEM tiles
        # q: [HEAD_DIM], k: [BLOCK_SIZE, HEAD_DIM]
        # scores[l] = sum_d(q[d] * k[l, d])
        scores = tl.sum(q[None, :] * k, axis=1) * scale  # [BLOCK_SIZE]
        scores = tl.where(offs_l < valid, scores, float("-inf"))

        # Online softmax update
        m_new = tl.max(tl.maximum(m, scores), axis=0)   # scalar
        alpha = tl.exp(m - m_new)
        exp_s = tl.exp(scores - m_new)                  # [BLOCK_SIZE]

        # acc += sum_l(exp_s[l] * v[l, :]) — element-wise, no tl.dot
        acc = acc * alpha + tl.sum(exp_s[:, None] * v, axis=0)  # [HEAD_DIM]
        l   = l   * alpha + tl.sum(exp_s,            axis=0)    # scalar
        m   = m_new

    # ── Normalize and store ───────────────────────────────────────────────────
    out = (acc / l).to(Out.dtype.element_ty)
    tl.store(Out
             + tok_idx * o_stride_t
             + q_head  * o_stride_h
             + offs_d  * o_stride_d,
             out)


def sm75_paged_cross_attn(
    query:       torch.Tensor,   # [B*Q, nq, hd]
    kv_cache:    torch.Tensor,   # [num_blocks, nkv, block_size, 2*hd]  NHD 4D
    kv_indptr:   torch.Tensor,   # [B+1]  int32 GPU  (pre-allocated, updated in-place)
    kv_indices:  torch.Tensor,   # [total_pages]  int32 GPU
    kv_last_len: torch.Tensor,   # [B]  int32 GPU
    num_reqs:    int,
    query_len:   int,
    scale:       float,
    output:      torch.Tensor,   # [B*Q, nq, hd]
) -> None:
    """Triton paged cross-attention, CUDA-graph-safe.

    kv_cache must be NHD 4D: [num_blocks, nkv, block_size, 2*head_dim].
    kv_indptr / kv_indices / kv_last_len are in-place-updated GPU tensors.
    """
    assert kv_cache.ndim == 4, (
        f"Expected NHD 4D kv_cache, got ndim={kv_cache.ndim} shape={kv_cache.shape}"
    )
    nkv        = kv_cache.shape[1]
    block_size = kv_cache.shape[2]
    hd         = kv_cache.shape[3] // 2
    nq         = query.shape[1]
    gqa_ratio  = nq // nkv

    # One program per (query_token, query_head)
    grid = (num_reqs * query_len, nq)

    _nhd_paged_cross_attn_fwd[grid](
        query,    query.stride(0),    query.stride(1),    query.stride(2),
        output,   output.stride(0),   output.stride(1),   output.stride(2),
        kv_cache, kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2), kv_cache.stride(3),
        kv_indptr, kv_indices, kv_last_len,
        scale,
        QUERY_LEN=query_len,
        BLOCK_SIZE=block_size,
        HEAD_DIM=hd,
        GQA_RATIO=gqa_ratio,
        num_stages=1,
        num_warps=4,
    )
