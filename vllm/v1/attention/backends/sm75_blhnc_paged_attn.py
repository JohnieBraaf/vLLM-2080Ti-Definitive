# SPDX-License-Identifier: Apache-2.0
"""SM75 (Turing) Triton paged cross-attention for NHD KV cache layout.

Raw kv_cache passed to FlashInfer forward() is always NHD 4D:
  [num_blocks, num_kv_heads, block_size, 2*head_dim]
  K = kv_cache[:, :, :, :head_dim]
  V = kv_cache[:, :, :, head_dim:]

The BLHNC env var controls only the permuted FlashInfer view, not raw storage.

This kernel is CUDA-graph-compatible: all tensor shapes are fixed, variable
context length is read from in-place-updated GPU scalars.
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
    kv_indices,      # [total_pages]   int32 GPU  (full pre-allocated buffer)
    kv_last_len,     # [num_reqs]      int32 GPU
    # Scaling
    scale,
    # Compile-time constants
    QUERY_LEN: tl.constexpr,    # K+1, typically 8
    BLOCK_SIZE: tl.constexpr,   # page_size, 16
    HEAD_DIM: tl.constexpr,     # head_dim, 128
    GQA_RATIO: tl.constexpr,    # nq // nkv
):
    """One program per (request_id, query_head_id)."""
    req_id = tl.program_id(0)
    q_head = tl.program_id(1)
    kv_head = q_head // GQA_RATIO

    # ── Load query tokens: [QUERY_LEN, HEAD_DIM] ─────────────────────────────
    q_base = req_id * QUERY_LEN
    offs_t = tl.arange(0, QUERY_LEN)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = (Q
              + (q_base + offs_t[:, None]) * q_stride_t
              + q_head * q_stride_h
              + offs_d[None, :] * q_stride_d)
    q = tl.load(q_ptrs).to(tl.float32)

    # ── Online softmax accumulators ───────────────────────────────────────────
    m = tl.full((QUERY_LEN,), float("-inf"), dtype=tl.float32)
    l = tl.zeros((QUERY_LEN,), dtype=tl.float32)
    acc = tl.zeros((QUERY_LEN, HEAD_DIM), dtype=tl.float32)

    # ── Page range for this request ───────────────────────────────────────────
    page_start = tl.load(kv_indptr + req_id)
    page_end   = tl.load(kv_indptr + req_id + 1)
    last_len   = tl.load(kv_last_len + req_id)
    n_pages    = page_end - page_start

    offs_l = tl.arange(0, BLOCK_SIZE)

    # ── Iterate over KV pages ─────────────────────────────────────────────────
    for page_i in range(n_pages):
        block_idx = tl.load(kv_indices + page_start + page_i)
        valid = tl.where(page_i < n_pages - 1, BLOCK_SIZE, last_len)

        kv_base = KV + block_idx * kv_stride_b + kv_head * kv_stride_h

        # K: kv_cache[block_idx, kv_head, l, :HEAD_DIM]
        k_ptrs = (kv_base
                  + offs_l[:, None] * kv_stride_l
                  + offs_d[None, :] * kv_stride_c)
        k = tl.load(k_ptrs, mask=offs_l[:, None] < valid, other=0.0).to(tl.float32)

        # V: kv_cache[block_idx, kv_head, l, HEAD_DIM:]
        v_ptrs = (kv_base
                  + offs_l[:, None] * kv_stride_l
                  + (HEAD_DIM + offs_d[None, :]) * kv_stride_c)
        v = tl.load(v_ptrs, mask=offs_l[:, None] < valid, other=0.0).to(tl.float32)

        # Scores: [QUERY_LEN, BLOCK_SIZE]
        scores = tl.dot(q, tl.trans(k)) * scale
        scores = tl.where(offs_l[None, :] < valid, scores, float("-inf"))

        # Online softmax update
        m_new  = tl.maximum(m, tl.max(scores, axis=1))
        alpha  = tl.exp(m - m_new)
        exp_s  = tl.exp(scores - m_new[:, None])

        acc = acc * alpha[:, None] + tl.dot(exp_s, v)
        l   = l   * alpha          + tl.sum(exp_s, axis=1)
        m   = m_new

    # ── Normalize and store ───────────────────────────────────────────────────
    out = (acc / l[:, None]).to(Out.dtype.element_ty)
    out_ptrs = (Out
                + (q_base + offs_t[:, None]) * o_stride_t
                + q_head * o_stride_h
                + offs_d[None, :] * o_stride_d)
    tl.store(out_ptrs, out)


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
    kv_indptr / kv_indices / kv_last_len are in-place-updated GPU tensors;
    the CUDA graph captures their pointers, not their values.
    """
    assert kv_cache.ndim == 4, (
        f"Expected NHD 4D kv_cache, got ndim={kv_cache.ndim} shape={kv_cache.shape}"
    )
    nkv       = kv_cache.shape[1]
    block_size = kv_cache.shape[2]
    hd        = kv_cache.shape[3] // 2
    nq        = query.shape[1]
    gqa_ratio = nq // nkv

    grid = (num_reqs, nq)

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
