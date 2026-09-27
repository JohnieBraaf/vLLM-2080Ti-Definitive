# SPDX-License-Identifier: Apache-2.0
"""Proxy speculator that routes DFlash2 draft proposals to a remote draft server.

Runs on the main workers (GPU 0,2). Implements the BaseSpeculator interface so
the rest of the model runner is unaware that the draft model is on a different
GPU.

Only TP rank 0 communicates with the draft server via ZMQ.  After getting draft
tokens, rank 0 broadcasts them to other TP ranks via NCCL so every worker
returns the same result (required for CUDA-graph correctness).
"""

from __future__ import annotations

import json
from typing import Any

import torch
import zmq

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.v1.spec_decode.disagg_dflash.protocol import (
    MSG_ACK,
    build_decode,
    build_free,
    build_ping,
    build_prefill,
    parse_draft_response,
    parse_header,
)
from vllm.v1.worker.gpu.dp_utils import DPSyncState
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator

logger = init_logger(__name__)

_TIMEOUT_MS = 10_000  # 10 s per request


class DisaggDFlashProposer(BaseSpeculator):
    """
    Drop-in replacement for DFlashSpeculator that offloads the draft model to a
    separate process on GPU1 via ZMQ.

    The main model's hidden states are the only data that need to cross the
    wire (≈112 KB per decode step at batch=8, Qwen3-27B hidden_size=7168).
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.device = device
        assert vllm_config.speculative_config is not None
        spec = vllm_config.speculative_config
        self.num_speculative_steps = spec.num_speculative_tokens
        self.draft_model_config    = spec.draft_model_config
        self.address               = spec.disagg_draft_address

        max_reqs   = vllm_config.scheduler_config.max_num_seqs
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.max_num_reqs   = max_reqs
        self.max_num_tokens = max_tokens
        self.dtype = vllm_config.model_config.dtype

        # Draft-token output buffer (returned to the model runner each step).
        # Must be on device so NCCL broadcast works.
        self.draft_tokens = torch.zeros(
            max_reqs,
            self.num_speculative_steps,
            dtype=torch.int64,
            device=device,
        )

        # Attributes expected by the model runner (DraftModelSpeculator contract).
        self.supports_mm_inputs     = False
        self.uses_mrope             = False
        self.needs_extra_input_slots = False
        self.parallel_drafting      = True
        self.draft_is_prefilling    = torch.zeros(max_reqs, dtype=torch.bool)
        self.idx_mapping            = torch.zeros(max_reqs, dtype=torch.int32, device=device)
        self.hidden_size            = spec.draft_model_config.get_hidden_size()
        self.vocab_size             = spec.draft_model_config.get_vocab_size()
        self.max_model_len          = vllm_config.model_config.max_model_len

        # Sequence tracking (rank 0 only).
        self._active_seqs: set[str] = set()

        # ── ZMQ: only TP rank 0 communicates with the draft server ─────────
        try:
            from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
            self._tp_rank = get_tensor_model_parallel_rank()
        except Exception:
            self._tp_rank = 0

        if self._tp_rank == 0:
            self._zmq_ctx = zmq.Context()
            self._sock    = self._zmq_ctx.socket(zmq.DEALER)
            self._sock.setsockopt(zmq.RCVTIMEO, _TIMEOUT_MS)
            self._sock.setsockopt(zmq.SNDTIMEO, _TIMEOUT_MS)
            self._sock.connect(self.address)
            logger.info(
                "DisaggDFlashProposer connected to draft server at %s", self.address
            )
            self._ping_server()
        else:
            self._zmq_ctx = None
            self._sock    = None

    # ──────────────────────────────────────────────────────────────────────────
    # BaseSpeculator interface
    # ──────────────────────────────────────────────────────────────────────────

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        pass

    def capture(self) -> None:
        pass

    def propose(
        self,
        input_batch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync: DPSyncState | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs=None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        # Skip ZMQ during warmup / profiling passes.
        if dummy_run or is_profile:
            return self.draft_tokens[: input_batch.num_reqs]

        num_reqs = input_batch.num_reqs
        tp_size  = self.vllm_config.parallel_config.tensor_parallel_size

        # Only rank 0 contacts the draft server.
        if self._tp_rank == 0:
            try:
                self._do_propose(input_batch, last_hidden_states, num_reqs,
                                 temperature, seeds)
            except Exception as exc:
                # Log and continue — draft_tokens stay at zero.
                # The broadcast below MUST still run so rank 1 doesn't deadlock.
                logger.warning("DisaggDFlashProposer: draft server failed: %s", exc)

        # Broadcast the result from rank 0 to all other TP ranks.
        if tp_size > 1:
            import torch.distributed as dist
            from vllm.distributed.parallel_state import get_tp_group
            dist.broadcast(
                self.draft_tokens,
                src=0,
                group=get_tp_group().device_group,
            )

        return self.draft_tokens[:num_reqs]

    def _do_propose(
        self,
        input_batch,
        last_hidden_states: torch.Tensor,
        num_reqs: int,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
    ) -> None:
        """Rank-0 only: contact the draft server and update self.draft_tokens."""
        req_ids = [input_batch.req_ids[i] for i in range(num_reqs)]

        # ── detect finished sequences, send FREE ─────────────────────────────
        current  = set(req_ids)
        finished = self._active_seqs - current
        for seq_id in finished:
            self._send_free(seq_id)

        # ── detect new (prefill) sequences, send PREFILL ──────────────────────
        # Check against the OLD _active_seqs BEFORE updating it.
        _qsl = getattr(input_batch, "query_start_loc_np", None)
        if _qsl is None:
            _qsl = getattr(input_batch, "query_start_loc", None)
        qsl = _qsl

        for i, seq_id in enumerate(req_ids):
            if seq_id not in self._active_seqs:   # new sequence
                tok_start = int(qsl[i])
                tok_end   = int(qsl[i + 1])
                hs_seq    = last_hidden_states[tok_start:tok_end].cpu()
                seq_len   = int(input_batch.seq_lens_cpu_upper_bound[i])
                T         = tok_end - tok_start
                pos       = torch.arange(seq_len - T, seq_len, dtype=torch.int64)
                self._send_prefill(seq_id, hs_seq, pos)

        # Update tracking AFTER sending PREFILLs.
        self._active_seqs = current

        # ── build DECODE payload ──────────────────────────────────────────────
        decode_hs    = torch.zeros(num_reqs, last_hidden_states.shape[-1], dtype=torch.float16)
        decode_pos   = torch.zeros(num_reqs, dtype=torch.int64)
        decode_temps = temperature[:num_reqs].cpu().float()
        decode_seeds = seeds[:num_reqs].cpu()

        for i in range(num_reqs):
            tok_end = int(qsl[i + 1]) - 1
            decode_hs[i]  = last_hidden_states[tok_end].cpu().to(torch.float16)
            decode_pos[i] = int(input_batch.seq_lens_cpu_upper_bound[i]) - 1

        draft_tokens_cpu = self._send_decode(
            req_ids, decode_hs, decode_pos, decode_temps, decode_seeds
        )
        self.draft_tokens[:num_reqs] = draft_tokens_cpu.to(
            device=self.device, dtype=torch.int64
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Lifecycle hooks called by the model runner
    # ──────────────────────────────────────────────────────────────────────────

    def set_attn(self, *args, **kwargs) -> None:
        pass

    def load_draft_model(self, *args, **kwargs):
        return None

    def set_eplb_state(self, eplb_state) -> None:
        pass

    def set_num_cached_tokens(self, num_cached_tokens) -> None:
        pass

    def __getattr__(self, name: str):
        """Catch-all for any DraftModelSpeculator attributes the model runner
        reads but the proxy doesn't need to implement."""
        _false = {
            "use_local_argmax_reduction", "use_heterogeneous_vocab",
            "use_fp64_gumbel", "_share_mtp_indices",
            "_enable_probabilistic_draft_probs", "constant_draft_positions",
        }
        if name in _false:
            return False
        _none = {
            "pcp_manager", "eplb_state", "model", "vocab_mapping",
            "draft_logits", "draft_watermarker", "allowed_attn_types",
            "_last_draft_probs", "backup_next_token_ids",
        }
        if name in _none:
            return None
        _empty = {"attn_groups", "draft_attn_groups"}
        if name in _empty:
            return []
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # ZMQ helpers (rank 0 only)
    # ──────────────────────────────────────────────────────────────────────────

    def _send(self, header: bytes, payload: bytes) -> tuple[bytes, bytes]:
        self._sock.send_multipart([b"", header, payload])
        parts = self._sock.recv_multipart()
        if len(parts) == 3:
            return parts[1], parts[2]
        if len(parts) == 2:
            return parts[0], parts[1]
        raise RuntimeError(f"Unexpected ZMQ response frames: {len(parts)}")

    def _ping_server(self) -> None:
        h, _ = build_ping()
        resp_h, _ = self._send(h, b"")
        hdr = parse_header(resp_h)
        if hdr.get("t") != MSG_ACK:
            raise RuntimeError(f"Draft server ping failed: {hdr}")
        logger.info("Draft server ping OK.")

    def _send_prefill(
        self,
        seq_id: str,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        h, p = build_prefill(seq_id, hidden_states, positions)
        resp_h, _ = self._send(h, p)
        hdr = parse_header(resp_h)
        if hdr.get("t") != MSG_ACK:
            raise RuntimeError(f"PREFILL failed for {seq_id}: {hdr}")

    def _send_decode(
        self,
        seq_ids: list[str],
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        temperatures: torch.Tensor,
        seeds: torch.Tensor,
    ) -> torch.Tensor:
        h, p = build_decode(seq_ids, hidden_states, positions, temperatures, seeds)
        resp_h, resp_p = self._send(h, p)
        hdr = parse_header(resp_h)
        if hdr.get("t") != MSG_ACK:
            raise RuntimeError(f"DECODE failed: {hdr}")
        return parse_draft_response(hdr, resp_p)

    def _send_free(self, seq_id: str) -> None:
        h, p = build_free(seq_id)
        try:
            resp_h, _ = self._send(h, p)
            hdr = parse_header(resp_h)
            if hdr.get("t") != MSG_ACK:
                logger.warning("FREE ack unexpected for %s: %s", seq_id, hdr)
        except Exception:
            logger.warning("FREE for %s failed (server may have restarted)", seq_id)

    def __del__(self):
        try:
            if self._sock is not None:
                self._sock.close(linger=0)
            if self._zmq_ctx is not None:
                self._zmq_ctx.destroy(linger=0)
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────────────────
    # Helper: write a config JSON for the draft server
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def write_draft_config(vllm_config: VllmConfig, path: str) -> None:
        """Serialise the config fields that DraftModelRunner needs."""
        spec = vllm_config.speculative_config
        assert spec is not None
        d = {
            "speculative_config": {
                "model": spec.draft_model_config.model,
                "num_speculative_tokens": spec.num_speculative_tokens,
                "kv_cache_dtype": str(spec.kv_cache_dtype) if spec.kv_cache_dtype else None,
            },
            "model_config": {
                "max_model_len": vllm_config.model_config.max_model_len,
                "dtype": str(vllm_config.model_config.dtype).replace("torch.", ""),
            },
            "cache_config": {
                "block_size": vllm_config.cache_config.block_size,
            },
            "scheduler_config": {
                "max_num_seqs": vllm_config.scheduler_config.max_num_seqs,
                "max_num_batched_tokens": vllm_config.scheduler_config.max_num_batched_tokens,
            },
        }
        with open(path, "w") as f:
            json.dump(d, f, indent=2)
        logger.info("Draft server config written to %s", path)
