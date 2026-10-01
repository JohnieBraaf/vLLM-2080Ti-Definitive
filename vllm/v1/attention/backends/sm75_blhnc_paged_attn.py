# SPDX-License-Identifier: Apache-2.0
"""SM75 (Turing) Triton paged cross-attention for NHD KV cache layout.

Raw kv_cache: NHD 4D [num_blocks, nkv, block_size, 2*head_dim].

Design: one program per (request, query_head), batch over QUERY_LEN tokens.
HEAD_DIM=128 is split into two 64-wide chunks so each tl.dot tile fits within
SM75's 64 KB SMEM limit. Uses FP16 tensor cores for compute.
"""

import torch
from vllm.triton_utils import tl, triton

# Half of HEAD_DIM — must divide HEAD_DIM evenly.
_CHUNK_D = 64


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
    QUERY_LEN:  tl.constexpr,   # K+1, typically 8
    BLOCK_SIZE: tl.constexpr,   # page_size, 16
    HEAD_DIM:   tl.constexpr,   # head_dim, 128
    CHUNK_D:    tl.constexpr,   # HEAD_DIM // 2 = 64
    GQA_RATIO:  tl.constexpr,   # nq // nkv
):
    """Batch over QUERY_LEN; HEAD_DIM split into 2×CHUNK_D to fit 64 KB SMEM."""
    req_id  = tl.program_id(0)
    q_head  = tl.program_id(1)
    kv_head = q_head // GQA_RATIO

    q_base = req_id * QUERY_LEN
    offs_t = tl.arange(0, QUERY_LEN)
    offs_l = tl.arange(0, BLOCK_SIZE)
    offs_c = tl.arange(0, CHUNK_D)

    # ── Load q in two halves: each [QUERY_LEN, CHUNK_D] ──────────────────────
    q0 = tl.load(Q
                 + (q_base + offs_t[:, None]) * q_stride_t
                 + q_head * q_stride_h
                 + offs_c[None, :] * q_stride_d).to(tl.float16)        # chunk 0
    q1 = tl.load(Q
                 + (q_base + offs_t[:, None]) * q_stride_t
                 + q_head * q_stride_h
                 + (CHUNK_D + offs_c[None, :]) * q_stride_d).to(tl.float16)  # chunk 1

    # ── Online softmax accumulators ───────────────────────────────────────────
    m    = tl.full((QUERY_LEN,), float("-inf"), dtype=tl.float32)
    l    = tl.zeros((QUERY_LEN,), dtype=tl.float32)
    acc0 = tl.zeros((QUERY_LEN, CHUNK_D), dtype=tl.float32)
    acc1 = tl.zeros((QUERY_LEN, CHUNK_D), dtype=tl.float32)

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

        # K chunk 0: [BLOCK_SIZE, CHUNK_D]
        k0 = tl.load(kv_base
                     + offs_l[:, None] * kv_stride_l
                     + offs_c[None, :] * kv_stride_c,
                     mask=offs_l[:, None] < valid,
                     other=0.0).to(tl.float16)
        # K chunk 1: [BLOCK_SIZE, CHUNK_D]
        k1 = tl.load(kv_base
                     + offs_l[:, None] * kv_stride_l
                     + (CHUNK_D + offs_c[None, :]) * kv_stride_c,
                     mask=offs_l[:, None] < valid,
                     other=0.0).to(tl.float16)

        # Scores: q0@k0.T + q1@k1.T → [QUERY_LEN, BLOCK_SIZE]
        scores = (tl.dot(q0, tl.trans(k0), out_dtype=tl.float32)
                + tl.dot(q1, tl.trans(k1), out_dtype=tl.float32)) * scale
        scores = tl.where(offs_l[None, :] < valid, scores, float("-inf"))

        # Online softmax
        m_new = tl.maximum(m, tl.max(scores, axis=1))
        alpha = tl.exp(m - m_new)
        exp_s = tl.exp(scores - m_new[:, None]).to(tl.float16)   # [QUERY_LEN, BLOCK_SIZE]

        # V chunk 0: [BLOCK_SIZE, CHUNK_D]
        v0 = tl.load(kv_base
                     + offs_l[:, None] * kv_stride_l
                     + (HEAD_DIM + offs_c[None, :]) * kv_stride_c,
                     mask=offs_l[:, None] < valid,
                     other=0.0).to(tl.float16)
        # V chunk 1: [BLOCK_SIZE, CHUNK_D]
        v1 = tl.load(kv_base
                     + offs_l[:, None] * kv_stride_l
                     + (HEAD_DIM + CHUNK_D + offs_c[None, :]) * kv_stride_c,
                     mask=offs_l[:, None] < valid,
                     other=0.0).to(tl.float16)

        acc0 = acc0 * alpha[:, None] + tl.dot(exp_s, v0, out_dtype=tl.float32)
        acc1 = acc1 * alpha[:, None] + tl.dot(exp_s, v1, out_dtype=tl.float32)
        l    = l    * alpha          + tl.sum(exp_s.to(tl.float32), axis=1)
        m    = m_new

    # ── Normalize and store ───────────────────────────────────────────────────
    inv_l = (1.0 / l).to(tl.float32)
    out0 = (acc0 * inv_l[:, None]).to(Out.dtype.element_ty)
    out1 = (acc1 * inv_l[:, None]).to(Out.dtype.element_ty)

    tl.store(Out
             + (q_base + offs_t[:, None]) * o_stride_t
             + q_head * o_stride_h
             + offs_c[None, :] * o_stride_d,
             out0)
    tl.store(Out
             + (q_base + offs_t[:, None]) * o_stride_t
             + q_head * o_stride_h
             + (CHUNK_D + offs_c[None, :]) * o_stride_d,
             out1)


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
    """
    assert kv_cache.ndim == 4, (
        f"Expected NHD 4D kv_cache, got ndim={kv_cache.ndim} shape={kv_cache.shape}"
    )
    nkv        = kv_cache.shape[1]
    block_size = kv_cache.shape[2]
    hd         = kv_cache.shape[3] // 2
    nq         = query.shape[1]
    gqa_ratio  = nq // nkv
    chunk_d    = hd // 2   # Split HEAD_DIM into two equal halves

    assert hd % 2 == 0, f"HEAD_DIM {hd} must be even for 2-chunk split"

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
        CHUNK_D=chunk_d,
        GQA_RATIO=gqa_ratio,
        num_stages=1,
        num_warps=4,
    )
