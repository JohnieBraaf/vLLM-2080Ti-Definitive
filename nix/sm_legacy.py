from __future__ import annotations
import importlib.util
from pathlib import Path
import torch

_EXT = None

def _load_ext():
    global _EXT
    if _EXT is not None:
        return _EXT
    sofiles = list(Path(__file__).parent.glob("flash_qla_legacy_gdn*.so"))
    if not sofiles:
        raise RuntimeError(
            "gdn_prefill_backend=flashqla_legacy requires the rebuilt SM70/SM75 "
            "FlashQLA extension with gdn_forward_varlen"
        )
    so = sofiles[0]
    spec = importlib.util.spec_from_file_location("flash_qla_legacy_gdn", so)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _EXT = mod
    return _EXT

def _check_inputs(q, k, v, g, beta, initial_state):
    tensors = [q, k, v, g, beta]
    if initial_state is not None:
        tensors.append(initial_state)
    if any(not t.is_cuda for t in tensors):
        raise ValueError("legacy GDN tensors must be CUDA tensors")
    if any(t.dtype != torch.float32 for t in tensors):
        raise ValueError("legacy GDN backend currently supports float32 tensors only")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("legacy GDN tensors must be contiguous")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [B, T, H, D]")
    if g.ndim != 3 or beta.ndim != 3:
        raise ValueError("g and beta must have shape [B, T, Hv]")
    if q.shape != k.shape:
        raise ValueError("q and k must have the same shape")
    batch, tokens, q_heads, dim = q.shape
    if v.shape[0] != batch or v.shape[1] != tokens or v.shape[3] != dim:
        raise ValueError("v must have shape [B, T, Hv, D] matching q/k")
    if g.shape != beta.shape or g.shape != v.shape[:3]:
        raise ValueError("g and beta must have shape [B, T, Hv]")
    if v.shape[2] % q_heads != 0:
        raise ValueError("Hv must be divisible by Hq")
    if dim not in (16, 32, 64, 128):
        raise ValueError("legacy GDN backend supports D in {16, 32, 64, 128}")
    if initial_state is not None and initial_state.shape != (batch, v.shape[2], dim, dim):
        raise ValueError("initial_state must have shape [B, Hv, D, D]")

def chunk_gated_delta_rule_fwd_legacy(q, k, v, g, beta, scale=None, initial_state=None):
    _check_inputs(q, k, v, g, beta, initial_state)
    if scale is None:
        scale = q.shape[-1] ** -0.5
    ext = _load_ext()
    return ext.gdn_forward(q, k, v, g, beta, initial_state, float(scale))

def chunk_gated_delta_rule_fwd_legacy_varlen(q, k, v, g, beta, cu_seqlens, scale=None, initial_state=None):
    if scale is None:
        scale = q.shape[-1] ** -0.5
    ext = _load_ext()
    return ext.gdn_forward_varlen(q, k, v, g, beta, initial_state, float(scale), cu_seqlens)
