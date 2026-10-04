# SPDX-License-Identifier: Apache-2.0
"""SM75 (Turing) Triton paged cross-attention for NHD KV cache layout.

Raw kv_cache: NHD 4D [num_blocks, nkv, block_size, 2*head_dim].

Design: one program per (request, query_head), batch over QUERY_LEN tokens.
HEAD_DIM is split into 64-wide chunks (NUM_CHUNKS = HEAD_DIM // 64) so each
tl.dot tile fits within SM75's 64 KB SMEM limit. Supports hd=128 (2 chunks)
and hd=256 (4 chunks). Uses FP16 tensor cores for compute.
"""

import torch
from vllm.triton_utils import tl, triton

# Half of HEAD_DIM — must divide HEAD_DIM evenly.
_CHUNK_D = 64


@triton.jit
def _nhd_paged_attn_fwd(
    # Query:  [num_reqs * query_len, nq, hd]
    Q, q_stride_t, q_stride_h, q_stride_d,
    # Output: [num_reqs * query_len, nq, hd]
    Out, o_stride_t, o_stride_h, o_stride_d,
    # KV cache NHD 4D: [num_blocks, nkv, block_size, 2*hd]
    KV, kv_stride_b, kv_stride_h, kv_stride_l, kv_stride_c,
    # Paged-KV metadata — GPU tensors updated in-place each step
    kv_indptr,           # [num_reqs + 1]  int32 GPU
    kv_indices,          # [total_pages]   int32 GPU (full pre-allocated buffer)
    kv_last_len,         # [num_reqs]      int32 GPU
    # Scaling
    scale,
    # Causal masking: pointer to a [num_reqs] int64 GPU tensor holding the
    # absolute sequence position of the first query token for each request.
    # Stored as a GPU tensor (not a scalar) so CUDA-graph replay works:
    # the address is stable and the caller writes T_bonus before each replay.
    # Pages are traversed in sequence order so slot (page_i, offs_l) has
    # position page_i * BLOCK_SIZE + offs_l.  Only used when IS_CAUSAL.
    query_start_pos_ptr,
    # Compile-time constants
    QUERY_LEN:  tl.constexpr,   # real query length, K+1
    QUERY_PAD:  tl.constexpr,   # next_power_of_2(QUERY_LEN): tl.arange needs pow2
    BLOCK_SIZE: tl.constexpr,   # page_size, 16
    HEAD_DIM:   tl.constexpr,   # head_dim, 128
    CHUNK_D:    tl.constexpr,   # always 64: fits within SM75 64 KB SMEM
    NUM_CHUNKS: tl.constexpr,   # HEAD_DIM // 64 — 2 for hd=128, 4 for hd=256
    GQA_RATIO:  tl.constexpr,   # nq // nkv
    IS_CAUSAL:  tl.constexpr,   # True for self-attn, False for cross-attn
):
    """Batch over QUERY_LEN; HEAD_DIM split into NUM_CHUNKS×64 chunks to fit SM75 64 KB SMEM."""
    req_id  = tl.program_id(0)
    q_head  = tl.program_id(1)
    kv_head = q_head // GQA_RATIO

    q_base = req_id * QUERY_LEN
    # Triton requires arange bounds to be a power of two, but the query length
    # is K+1: 8 for DFlash2 K=7, 3 or 5 for MTP.  Pad the row dimension and mask
    # the padding rows on both the q load and the store.
    offs_t  = tl.arange(0, QUERY_PAD)
    t_valid = offs_t < QUERY_LEN
    offs_l = tl.arange(0, BLOCK_SIZE)
    offs_c = tl.arange(0, CHUNK_D)

    # ── Load q in NUM_CHUNKS halves: each [QUERY_LEN, CHUNK_D] ─────────────────
    q0 = tl.load(Q
                 + (q_base + offs_t[:, None]) * q_stride_t
                 + q_head * q_stride_h
                 + offs_c[None, :] * q_stride_d,
                 mask=t_valid[:, None], other=0.0).to(tl.float16)      # chunk 0
    q1 = tl.load(Q
                 + (q_base + offs_t[:, None]) * q_stride_t
                 + q_head * q_stride_h
                 + (CHUNK_D + offs_c[None, :]) * q_stride_d,
                 mask=t_valid[:, None], other=0.0).to(tl.float16)      # chunk 1
    if NUM_CHUNKS >= 4:
        q2 = tl.load(Q
                     + (q_base + offs_t[:, None]) * q_stride_t
                     + q_head * q_stride_h
                     + (2 * CHUNK_D + offs_c[None, :]) * q_stride_d,
                     mask=t_valid[:, None], other=0.0).to(tl.float16)  # chunk 2
        q3 = tl.load(Q
                     + (q_base + offs_t[:, None]) * q_stride_t
                     + q_head * q_stride_h
                     + (3 * CHUNK_D + offs_c[None, :]) * q_stride_d,
                     mask=t_valid[:, None], other=0.0).to(tl.float16)  # chunk 3

    # ── Online softmax accumulators ───────────────────────────────────────────
    m    = tl.full((QUERY_PAD,), float("-inf"), dtype=tl.float32)
    l    = tl.zeros((QUERY_PAD,), dtype=tl.float32)
    acc0 = tl.zeros((QUERY_PAD, CHUNK_D), dtype=tl.float32)
    acc1 = tl.zeros((QUERY_PAD, CHUNK_D), dtype=tl.float32)
    if NUM_CHUNKS >= 4:
        acc2 = tl.zeros((QUERY_PAD, CHUNK_D), dtype=tl.float32)
        acc3 = tl.zeros((QUERY_PAD, CHUNK_D), dtype=tl.float32)

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
        if NUM_CHUNKS >= 4:
            k2 = tl.load(kv_base
                         + offs_l[:, None] * kv_stride_l
                         + (2 * CHUNK_D + offs_c[None, :]) * kv_stride_c,
                         mask=offs_l[:, None] < valid,
                         other=0.0).to(tl.float16)
            k3 = tl.load(kv_base
                         + offs_l[:, None] * kv_stride_l
                         + (3 * CHUNK_D + offs_c[None, :]) * kv_stride_c,
                         mask=offs_l[:, None] < valid,
                         other=0.0).to(tl.float16)

        # Scores: sum_c(qc @ kc.T) → [QUERY_LEN, BLOCK_SIZE]
        scores = tl.dot(q0, tl.trans(k0), out_dtype=tl.float32) + tl.dot(q1, tl.trans(k1), out_dtype=tl.float32)
        if NUM_CHUNKS >= 4:
            scores = scores + tl.dot(q2, tl.trans(k2), out_dtype=tl.float32) + tl.dot(q3, tl.trans(k3), out_dtype=tl.float32)
        scores = scores * scale
        scores = tl.where(offs_l[None, :] < valid, scores, float("-inf"))

        # Causal mask: query token at offset offs_t can only attend to KV
        # slots at sequence positions <= query_start_pos + offs_t.
        # Pages are traversed in order so position = page_i * BLOCK_SIZE + offs_l.
        # query_start_pos is loaded from a stable GPU tensor so CUDA-graph
        # replay works: the caller writes T_bonus before each replay.
        if IS_CAUSAL:
            query_start_pos  = tl.load(query_start_pos_ptr + req_id).to(tl.int64)
            kv_seq_pos       = page_i * BLOCK_SIZE + offs_l          # [BLOCK_SIZE]
            q_causal_limit   = query_start_pos + offs_t              # [QUERY_LEN]
            scores = tl.where(
                kv_seq_pos[None, :] > q_causal_limit[:, None],
                float("-inf"), scores)

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
        if NUM_CHUNKS >= 4:
            v2 = tl.load(kv_base
                         + offs_l[:, None] * kv_stride_l
                         + (HEAD_DIM + 2 * CHUNK_D + offs_c[None, :]) * kv_stride_c,
                         mask=offs_l[:, None] < valid,
                         other=0.0).to(tl.float16)
            v3 = tl.load(kv_base
                         + offs_l[:, None] * kv_stride_l
                         + (HEAD_DIM + 3 * CHUNK_D + offs_c[None, :]) * kv_stride_c,
                         mask=offs_l[:, None] < valid,
                         other=0.0).to(tl.float16)

        acc0 = acc0 * alpha[:, None] + tl.dot(exp_s, v0, out_dtype=tl.float32)
        acc1 = acc1 * alpha[:, None] + tl.dot(exp_s, v1, out_dtype=tl.float32)
        if NUM_CHUNKS >= 4:
            acc2 = acc2 * alpha[:, None] + tl.dot(exp_s, v2, out_dtype=tl.float32)
            acc3 = acc3 * alpha[:, None] + tl.dot(exp_s, v3, out_dtype=tl.float32)
        l    = l    * alpha          + tl.sum(exp_s.to(tl.float32), axis=1)
        m    = m_new

    # ── Normalize and store ───────────────────────────────────────────────────
    inv_l = (1.0 / l).to(tl.float32)
    out0 = (acc0 * inv_l[:, None]).to(Out.dtype.element_ty)
    out1 = (acc1 * inv_l[:, None]).to(Out.dtype.element_ty)
    if NUM_CHUNKS >= 4:
        out2 = (acc2 * inv_l[:, None]).to(Out.dtype.element_ty)
        out3 = (acc3 * inv_l[:, None]).to(Out.dtype.element_ty)

    tl.store(Out
             + (q_base + offs_t[:, None]) * o_stride_t
             + q_head * o_stride_h
             + offs_c[None, :] * o_stride_d,
             out0, mask=t_valid[:, None])
    tl.store(Out
             + (q_base + offs_t[:, None]) * o_stride_t
             + q_head * o_stride_h
             + (CHUNK_D + offs_c[None, :]) * o_stride_d,
             out1, mask=t_valid[:, None])
    if NUM_CHUNKS >= 4:
        tl.store(Out
                 + (q_base + offs_t[:, None]) * o_stride_t
                 + q_head * o_stride_h
                 + (2 * CHUNK_D + offs_c[None, :]) * o_stride_d,
                 out2, mask=t_valid[:, None])
        tl.store(Out
                 + (q_base + offs_t[:, None]) * o_stride_t
                 + q_head * o_stride_h
                 + (3 * CHUNK_D + offs_c[None, :]) * o_stride_d,
                 out3, mask=t_valid[:, None])


def sm75_paged_attn(
    query:           torch.Tensor,              # [B*Q, nq, hd]
    kv_cache:        torch.Tensor,              # NHD 4D: [num_blocks, nkv, block_size, 2*hd]
    kv_indptr:       torch.Tensor,              # [B+1]  int32 GPU
    kv_indices:      torch.Tensor,              # [total_pages]  int32 GPU
    kv_last_len:     torch.Tensor,              # [B]  int32 GPU
    num_reqs:        int,
    query_len:       int,
    scale:           float,
    output:          torch.Tensor,              # [B*Q, nq, hd]
    causal:          bool         = False,
    query_start_pos: torch.Tensor = None,       # [B] int64 GPU — T_bonus per request
) -> None:
    """Triton paged attention for SM75/Turing, CUDA-graph-safe.

    Supports causal (self-attention) and non-causal (cross-attention) modes.
    kv_cache must be NHD 4D: [num_blocks, nkv, block_size, 2*head_dim].

    When causal=True, query_start_pos must be a [num_reqs] int64 GPU tensor
    whose i-th element is the absolute sequence position of the first query
    token for request i (T_bonus).  Storing T_bonus in a GPU tensor (rather
    than passing it as a Python scalar) keeps the kernel-launch arguments
    stable across steps so torch.cuda.CUDAGraph replay works correctly.
    The caller writes the current T_bonus into the tensor before each replay.
    Pages must be in sequence order (guaranteed by DraftModelRunner).
    """
    assert kv_cache.ndim == 4, (
        f"Expected NHD 4D kv_cache, got ndim={kv_cache.ndim} shape={kv_cache.shape}"
    )
    nkv        = kv_cache.shape[1]
    block_size = kv_cache.shape[2]
    hd         = kv_cache.shape[3] // 2
    nq         = query.shape[1]
    gqa_ratio  = nq // nkv
    chunk_d    = 64
    num_chunks = hd // 64

    assert hd % 64 == 0, f"HEAD_DIM {hd} must be a multiple of 64"

    if causal and query_start_pos is None:
        raise ValueError("sm75_paged_attn: causal=True requires query_start_pos tensor")
    if not causal:
        if query_start_pos is None:
            query_start_pos = torch.zeros(num_reqs, dtype=torch.int64,
                                          device=query.device)

    grid = (num_reqs, nq)

    _nhd_paged_attn_fwd[grid](
        query,    query.stride(0),    query.stride(1),    query.stride(2),
        output,   output.stride(0),   output.stride(1),   output.stride(2),
        kv_cache, kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2), kv_cache.stride(3),
        kv_indptr, kv_indices, kv_last_len,
        scale,
        query_start_pos,
        QUERY_LEN=query_len,
        QUERY_PAD=1 << (query_len - 1).bit_length(),
        BLOCK_SIZE=block_size,
        HEAD_DIM=hd,
        CHUNK_D=chunk_d,
        NUM_CHUNKS=num_chunks,
        GQA_RATIO=gqa_ratio,
        IS_CAUSAL=causal,
        num_stages=1,
        num_warps=4,
    )


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
    """Non-causal alias kept for backward compatibility."""
    sm75_paged_attn(
        query, kv_cache, kv_indptr, kv_indices, kv_last_len,
        num_reqs, query_len, scale, output,
        causal=False,
    )

def _do_sm75_triton_warmup(ql,bs,hd,nkv,nq):
    try:
        import torch as _t;d=_t.cuda.current_device();_q=_t.zeros(ql,nq,hd,dtype=_t.float16,device=d);_kv=_t.zeros(1,nkv,bs,2*hd,dtype=_t.float16,device=d);_ip=_t.tensor([0,1],dtype=_t.int32,device=d);_ix=_t.tensor([0],dtype=_t.int32,device=d);_ll=_t.tensor([1],dtype=_t.int32,device=d);_out=_t.zeros_like(_q);sm75_paged_cross_attn(_q,_kv,_ip,_ix,_ll,1,ql,1.0,_out);_t.cuda.synchronize()
    except Exception:pass
