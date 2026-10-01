# SPDX-License-Identifier: Apache-2.0
"""SM75 (Turing) Triton paged cross-attention for BLHNC KV cache layout.

Replaces the Python-loop SDPA bypass with a single Triton kernel that is
CUDA-graph-compatible: all tensor shapes are fixed at capture time, variable
context length is read from an in-place-updated GPU tensor.

KV cache layout: BLHNC = [num_blocks, block_size, num_kv_heads, 2, head_dim]
  K is at dimension index 2 (n=0), V at n=1.

Kernel algorithm: FlashAttention-2 online softmax, one program per
(request, query_head), iterating over KV pages.
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _blhnc_paged_cross_attn_fwd(
    # Query:  [num_reqs * query_len, nq, hd]
    Q,
    q_stride_t, q_stride_h, q_stride_d,
    # Output: [num_reqs * query_len, nq, hd]
    Out,
    o_stride_t, o_stride_h, o_stride_d,
    # KV cache: [num_blocks, block_size, nkv, 2, hd]  (BLHNC)
    KV,
    kv_stride_b, kv_stride_l, kv_stride_h, kv_stride_n, kv_stride_d,
    # Paged-KV metadata (GPU tensors, in-place updated each step)
    kv_indptr,       # [num_reqs + 1]   int32 GPU
    kv_indices,      # [total_pages]    int32 GPU  (full pre-allocated buffer)
    kv_last_len,     # [num_reqs]       int32 GPU
    # Scaling
    scale,
    # Compile-time constants
    QUERY_LEN: tl.constexpr,   # K+1, typically 8
    BLOCK_SIZE: tl.constexpr,  # page_size, 16
    HEAD_DIM: tl.constexpr,    # head_dim, 128
    GQA_RATIO: tl.constexpr,   # nq // nkv
):
    """One program per (request_id, query_head_id)."""
    req_id = tl.program_id(0)
    q_head = tl.program_id(1)
    kv_head = q_head // GQA_RATIO

    # ── Load query tokens for this (req, head): [QUERY_LEN, HEAD_DIM] ────────
    q_base = req_id * QUERY_LEN
    offs_t = tl.arange(0, QUERY_LEN)   # [QUERY_LEN]
    offs_d = tl.arange(0, HEAD_DIM)    # [HEAD_DIM]

    q_ptrs = (Q
              + (q_base + offs_t[:, None]) * q_stride_t
              + q_head * q_stride_h
              + offs_d[None, :] * q_stride_d)
    q = tl.load(q_ptrs).to(tl.float32)  # [QUERY_LEN, HEAD_DIM]

    # ── Online softmax accumulators ───────────────────────────────────────────
    m = tl.full((QUERY_LEN,), float("-inf"), dtype=tl.float32)
    l = tl.zeros((QUERY_LEN,), dtype=tl.float32)
    acc = tl.zeros((QUERY_LEN, HEAD_DIM), dtype=tl.float32)

    # ── Paging metadata for this request ─────────────────────────────────────
    page_start = tl.load(kv_indptr + req_id)
    page_end   = tl.load(kv_indptr + req_id + 1)
    last_len   = tl.load(kv_last_len + req_id)
    n_pages    = page_end - page_start

    offs_l = tl.arange(0, BLOCK_SIZE)  # [BLOCK_SIZE]

    # ── Iterate over KV pages ─────────────────────────────────────────────────
    for page_i in range(n_pages):
        block_idx = tl.load(kv_indices + page_start + page_i)

        # Valid tokens in this page (last page may be partial)
        valid = tl.where(page_i < n_pages - 1, BLOCK_SIZE, last_len)

        # ── Load K: [BLOCK_SIZE, HEAD_DIM] ────────────────────────────────────
        k_ptrs = (KV
                  + block_idx * kv_stride_b
                  + offs_l[:, None] * kv_stride_l
                  + kv_head * kv_stride_h
                  + 0 * kv_stride_n          # n=0 → K
                  + offs_d[None, :] * kv_stride_d)
        k = tl.load(k_ptrs, mask=offs_l[:, None] < valid, other=0.0).to(tl.float32)

        # ── Load V: [BLOCK_SIZE, HEAD_DIM] ────────────────────────────────────
        v_ptrs = (KV
                  + block_idx * kv_stride_b
                  + offs_l[:, None] * kv_stride_l
                  + kv_head * kv_stride_h
                  + 1 * kv_stride_n          # n=1 → V
                  + offs_d[None, :] * kv_stride_d)
        v = tl.load(v_ptrs, mask=offs_l[:, None] < valid, other=0.0).to(tl.float32)

        # ── Attention scores: [QUERY_LEN, BLOCK_SIZE] ─────────────────────────
        scores = tl.dot(q, tl.trans(k)) * scale  # [QUERY_LEN, BLOCK_SIZE]

        # Mask out padding positions in the last page
        scores = tl.where(offs_l[None, :] < valid, scores, float("-inf"))

        # ── Online softmax update ─────────────────────────────────────────────
        m_new = tl.maximum(m, tl.max(scores, axis=1))          # [QUERY_LEN]
        alpha  = tl.exp(m - m_new)                              # [QUERY_LEN]
        exp_s  = tl.exp(scores - m_new[:, None])                # [QUERY_LEN, BLOCK_SIZE]

        acc = acc * alpha[:, None] + tl.dot(exp_s, v)
        l   = l   * alpha          + tl.sum(exp_s, axis=1)
        m   = m_new

    # ── Normalise and store ───────────────────────────────────────────────────
    out = (acc / l[:, None]).to(Out.dtype.element_ty)

    out_ptrs = (Out
                + (q_base + offs_t[:, None]) * o_stride_t
                + q_head * o_stride_h
                + offs_d[None, :] * o_stride_d)
    tl.store(out_ptrs, out)


def sm75_blhnc_paged_cross_attn(
    query:          torch.Tensor,  # [B*Q, nq, hd]
    kv_cache:       torch.Tensor,  # [num_blocks, block_size, nkv, 2, hd]
    kv_indptr:      torch.Tensor,  # [B+1]  int32  GPU
    kv_indices:     torch.Tensor,  # [total_pages] int32 GPU  (full buffer)
    kv_last_len:    torch.Tensor,  # [B]    int32  GPU
    num_reqs:       int,
    query_len:      int,           # K+1
    scale:          float,
    output:         torch.Tensor,  # [B*Q, nq, hd]
) -> None:
    """Launch the Triton paged cross-attention kernel.

    All tensor shapes are fixed for a given spec-decode configuration,
    making this CUDA-graph-safe as long as the caller updates kv_indptr,
    kv_indices, and kv_last_len in-place before graph replay.
    """
    nq = query.shape[1]
    nkv = kv_cache.shape[2]
    hd = kv_cache.shape[4]
    block_size = kv_cache.shape[1]
    gqa_ratio = nq // nkv

    grid = (num_reqs, nq)

    _blhnc_paged_cross_attn_fwd[grid](
        query,
        query.stride(0), query.stride(1), query.stride(2),
        output,
        output.stride(0), output.stride(1), output.stride(2),
        kv_cache,
        kv_cache.stride(0), kv_cache.stride(1),
        kv_cache.stride(2), kv_cache.stride(3), kv_cache.stride(4),
        kv_indptr,
        kv_indices,
        kv_last_len,
        scale,
        QUERY_LEN=query_len,
        BLOCK_SIZE=block_size,
        HEAD_DIM=hd,
        GQA_RATIO=gqa_ratio,
    )
