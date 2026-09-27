# SPDX-License-Identifier: Apache-2.0
"""Standalone draft server for disaggregated DFlash2 speculative decoding.

Run on GPU1 (the non-NVLink GPU):
    python -m vllm.v1.spec_decode.disagg_dflash.draft_server \
        --device 1 \
        --address tcp://0.0.0.0:50052 \
        --config-json /tmp/dflash_draft_config.json

The config JSON is a serialised VllmConfig produced by the proxy at startup.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from typing import Any

import msgpack
import numpy as np
import torch
import zmq

from vllm.logger import init_logger
from vllm.v1.spec_decode.disagg_dflash.protocol import (
    MSG_ACK,
    MSG_DECODE,
    MSG_ERROR,
    MSG_FREE,
    MSG_PING,
    MSG_PREFILL,
    build_ack,
    build_draft_response,
    build_error,
    parse_decode_payload,
    parse_header,
    parse_prefill_payload,
)

logger = init_logger(__name__)


class SimpleBlockManager:
    """Trivial free-list block allocator for the draft KV cache on GPU1."""

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self._free: list[int] = list(range(num_blocks))

    def allocate(self, num_tokens: int) -> list[int]:
        needed = (num_tokens + self.block_size - 1) // self.block_size
        if needed > len(self._free):
            raise RuntimeError(
                f"Draft KV cache OOM: need {needed} blocks, "
                f"only {len(self._free)} free"
            )
        blocks = self._free[:needed]
        self._free = self._free[needed:]
        return blocks

    def free(self, blocks: list[int]) -> None:
        self._free.extend(blocks)

    def allocate_one(self) -> int:
        if not self._free:
            raise RuntimeError("Draft KV cache OOM: no free blocks")
        return self._free.pop(0)


class DraftModelRunner:
    """
    Loads the DFlash2 model on a single GPU and serves hidden-state-based
    draft proposals.

    Initialisation mirrors what GPUModelRunner does for the draft speculator,
    but only for the pieces needed by a standalone process:
      - model weights loaded via get_model()
      - KV cache allocated directly on the device
      - attention set up via vLLM's existing backend
    """

    def __init__(self, vllm_config_dict: dict, device: torch.device):
        self.device = device
        torch.cuda.set_device(device)

        self._build_vllm_config(vllm_config_dict)
        self._load_model()
        self._init_kv_cache()
        self._init_block_manager()

        # Per-sequence state: block_table and current length
        self.seq_block_tables: dict[str, list[int]] = {}
        self.seq_lengths:      dict[str, int]       = {}

        logger.info(
            "DraftModelRunner ready on %s: %d KV blocks (block_size=%d)",
            device, self.num_blocks, self.block_size,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Initialisation helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _build_vllm_config(self, d: dict) -> None:
        spec = d["speculative_config"]
        self.draft_model_path = spec["model"]
        self.num_speculative_tokens = spec["num_speculative_tokens"]
        self.kv_cache_dtype = spec.get("kv_cache_dtype") or "auto"
        self.block_size = d.get("cache_config", {}).get("block_size", 16)
        self.max_model_len = d["model_config"]["max_model_len"]
        self.dtype_str = d["model_config"].get("dtype", "float16")
        self.dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[
            self.dtype_str
        ]
        self._config_dict = d

    def _load_model(self) -> None:
        from vllm.engine.arg_utils import EngineArgs
        from vllm.model_executor.model_loader import get_model

        logger.info("Loading DFlash2 draft model: %s", self.draft_model_path)

        kv_dtype = self.kv_cache_dtype if self.kv_cache_dtype not in (None, "auto") else "auto"
        engine_args = EngineArgs(
            model=self.draft_model_path,
            max_model_len=self.max_model_len,
            dtype=self.dtype_str,
            gpu_memory_utilization=0.85,
            enforce_eager=True,
            trust_remote_code=True,
            tensor_parallel_size=1,
            kv_cache_dtype=kv_dtype,
            disable_log_stats=True,
        )
        self.vllm_config = engine_args.create_engine_config()
        self.draft_model_config = self.vllm_config.model_config

        self.model = get_model(
            vllm_config=self.vllm_config,
            model_config=self.draft_model_config,
        )
        self.model.to(self.device)
        self.model.eval()
        logger.info("Draft model loaded.")

    def _init_kv_cache(self) -> None:
        """Allocate per-layer KV cache tensors on GPU1 and inject them."""
        from vllm.config import ParallelConfig

        cfg = self.draft_model_config
        par = ParallelConfig(tensor_parallel_size=1)

        num_layers  = cfg.get_num_layers(par)
        num_kv_heads = cfg.get_num_kv_heads(par)
        head_size   = cfg.get_head_size()
        bs          = self.block_size

        # Estimate free memory after model weights
        torch.cuda.synchronize(self.device)
        free_mem, total_mem = torch.cuda.mem_get_info(self.device.index)
        # Reserve 1 GB for activations / overhead
        usable = max(0, free_mem - 1 * 1024**3)

        bytes_per_block = (
            2  # K and V
            * num_layers
            * num_kv_heads
            * head_size
            * bs
            * 2  # float16
        )
        self.num_blocks = max(1, int(usable // bytes_per_block))
        logger.info(
            "Allocating %d KV blocks (%d layers, %d heads, head_size=%d, "
            "block_size=%d) on %s",
            self.num_blocks, num_layers, num_kv_heads, head_size, bs, self.device,
        )

        # kv_cache[layer] = tensor of shape [2, num_blocks, block_size, num_kv_heads, head_size]
        kv_dtype = torch.float8_e4m3fn if self.kv_cache_dtype == "fp8" \
                   else torch.float16
        self.kv_cache: list[torch.Tensor] = []
        for _ in range(num_layers):
            t = torch.zeros(
                2, self.num_blocks, bs, num_kv_heads, head_size,
                dtype=kv_dtype,
                device=self.device,
            )
            self.kv_cache.append(t)

        # Inject into the model's attention layers so they can read/write KV.
        # vLLM v1 attention layers store kv_cache as a list attribute.
        layer_idx = 0
        for name, module in self.model.named_modules():
            if hasattr(module, "kv_cache"):
                if layer_idx < len(self.kv_cache):
                    module.kv_cache = self.kv_cache[layer_idx]
                    layer_idx += 1

        self.num_kv_heads = num_kv_heads
        self.head_size    = head_size
        self.num_layers   = num_layers

    def _init_block_manager(self) -> None:
        self.block_manager = SimpleBlockManager(self.num_blocks, self.block_size)

    # ──────────────────────────────────────────────────────────────────────────
    # Request handlers
    # ──────────────────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def handle_prefill(
        self,
        seq_id: str,
        hidden_states: torch.Tensor,  # [T, H]
        positions:     torch.Tensor,  # [T]
    ) -> None:
        T = hidden_states.shape[0]
        hidden_states = hidden_states.to(device=self.device, dtype=self.dtype)
        positions     = positions.to(device=self.device)

        # Allocate KV blocks for this sequence
        blocks = self.block_manager.allocate(T)
        self.seq_block_tables[seq_id] = blocks
        self.seq_lengths[seq_id]      = T

        # Compute slot mapping: token i → block[i // bs] * bs + (i % bs)
        slot_mapping = self._compute_slot_mapping_for_sequence(blocks, T)

        # Project context hidden states into KV cache
        self.model.precompute_and_store_context_kv(
            hidden_states,
            positions,
            slot_mapping,
        )
        logger.debug("PREFILL seq=%s T=%d blocks=%d", seq_id, T, len(blocks))

    @torch.inference_mode()
    def handle_decode(
        self,
        seq_ids:      list[str],
        hidden_states: torch.Tensor,  # [B, H] — one new token per sequence
        positions:     torch.Tensor,  # [B]
        temperatures:  torch.Tensor,  # [B]
        seeds:         torch.Tensor,  # [B]
    ) -> torch.Tensor:                # [B, K]
        B = len(seq_ids)
        K = self.num_speculative_tokens
        hidden_states = hidden_states.to(device=self.device, dtype=self.dtype)
        positions     = positions.to(device=self.device)
        temperatures  = temperatures.to(device=self.device)
        seeds         = seeds.to(device=self.device)

        # 1. Extend each sequence's KV cache with the new token's hidden state
        for i, seq_id in enumerate(seq_ids):
            T_old = self.seq_lengths[seq_id]
            T_new = T_old + 1
            blocks = self.seq_block_tables[seq_id]

            # Allocate a new block if the current one is full
            if T_new > len(blocks) * self.block_size:
                blocks.append(self.block_manager.allocate_one())

            new_slot = self._slot_for_position(blocks, T_old)
            hs_i = hidden_states[i].unsqueeze(0)    # [1, H]
            pos_i = positions[i].unsqueeze(0)        # [1]
            slot_i = torch.tensor([new_slot], device=self.device, dtype=torch.int64)

            self.model.precompute_and_store_context_kv(hs_i, pos_i, slot_i)
            self.seq_lengths[seq_id] = T_new

        # 2. Build query input: for each request, (bonus_pos, mask_pos × K)
        #    bonus token = the new token id (last sampled) → we use the hidden state
        #    mask tokens = the model's special mask_token_id
        draft_token_ids = self._run_draft_forward(seq_ids, positions, temperatures, seeds)
        return draft_token_ids  # [B, K]

    def handle_free(self, seq_id: str) -> None:
        blocks = self.seq_block_tables.pop(seq_id, [])
        self.block_manager.free(blocks)
        self.seq_lengths.pop(seq_id, None)
        logger.debug("FREE seq=%s released %d blocks", seq_id, len(blocks))

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_slot_mapping_for_sequence(
        self, blocks: list[int], num_tokens: int
    ) -> torch.Tensor:
        slots = []
        for i in range(num_tokens):
            slots.append(self._slot_for_position(blocks, i))
        return torch.tensor(slots, dtype=torch.int64, device=self.device)

    def _slot_for_position(self, blocks: list[int], pos: int) -> int:
        block_idx = pos // self.block_size
        offset    = pos  % self.block_size
        return blocks[block_idx] * self.block_size + offset

    def _build_block_table_tensor(
        self, seq_ids: list[str], max_blocks: int
    ) -> torch.Tensor:
        B = len(seq_ids)
        bt = torch.zeros(B, max_blocks, dtype=torch.int32, device=self.device)
        for i, seq_id in enumerate(seq_ids):
            blocks = self.seq_block_tables[seq_id]
            bt[i, : len(blocks)] = torch.tensor(
                blocks, dtype=torch.int32, device=self.device
            )
        return bt

    def _run_draft_forward(
        self,
        seq_ids:     list[str],
        positions:   torch.Tensor,   # [B] — positions of newly added tokens
        temperatures: torch.Tensor,  # [B]
        seeds:        torch.Tensor,  # [B]
    ) -> torch.Tensor:               # [B, K]
        from vllm.forward_context import set_forward_context

        B = len(seq_ids)
        K = self.num_speculative_tokens
        bs = self.block_size

        seq_lens  = torch.tensor(
            [self.seq_lengths[sid] for sid in seq_ids],
            dtype=torch.int32, device=self.device,
        )
        max_seq_len = int(seq_lens.max().item())
        max_blocks  = (max_seq_len + bs - 1) // bs

        block_tables = self._build_block_table_tensor(seq_ids, max_blocks)

        # Query layout per request: (bonus + K mask) tokens
        num_query_per_req = 1 + K
        num_query_total   = B * num_query_per_req

        # Input IDs: bonus position uses the hidden state already stored;
        # mask tokens use the model's mask_token_id
        mask_token_id = self._get_mask_token_id()
        input_ids = torch.full(
            (num_query_total,), mask_token_id,
            dtype=torch.int32, device=self.device,
        )
        # The bonus slot for each request is the first of the num_query_per_req
        # We leave it as mask_token_id; precompute_and_store_context_kv already
        # handled writing the bonus KV

        # Positions for query tokens
        query_positions = torch.zeros(num_query_total, dtype=torch.int64, device=self.device)
        for i in range(B):
            base_pos = int(positions[i].item())  # position of the new token
            for j in range(num_query_per_req):
                query_positions[i * num_query_per_req + j] = base_pos + j

        # Slot mapping for query tokens (write their KV into the cache)
        query_slots = torch.full(
            (num_query_total,), -1, dtype=torch.int64, device=self.device
        )
        for i, seq_id in enumerate(seq_ids):
            blocks = self.seq_block_tables[seq_id]
            T = self.seq_lengths[seq_id]
            for j in range(num_query_per_req):
                slot = self._slot_for_position(blocks, T - 1 + j)
                query_slots[i * num_query_per_req + j] = slot

        # query_start_loc: [B+1]
        query_start_loc = torch.arange(B + 1, dtype=torch.int32, device=self.device) \
                          * num_query_per_req

        attn_metadata = self._build_attn_metadata(
            seq_lens=seq_lens,
            block_tables=block_tables,
            query_start_loc=query_start_loc,
            slot_mapping=query_slots,
            num_query_total=num_query_total,
            num_query_per_req=num_query_per_req,
            max_seq_len=max_seq_len,
        )
        if attn_metadata is None:
            # Fallback: return zeros (will be wrong, but won't crash)
            return torch.zeros(B, K, dtype=torch.int32, device=self.device)

        # Build per-layer attn metadata dict
        per_layer_meta = {}
        for name, module in self.model.named_modules():
            if hasattr(module, "attn_layer_name"):
                per_layer_meta[module.attn_layer_name] = attn_metadata

        # If we couldn't build per-layer map via attn_layer_name, try layer names
        if not per_layer_meta:
            for name, module in self.model.named_modules():
                if "attn" in name and hasattr(module, "forward"):
                    per_layer_meta[name] = attn_metadata

        with set_forward_context(
            per_layer_meta,
            self.vllm_config,
            num_tokens=num_query_total,
        ):
            logits = self.model(
                input_ids=input_ids,
                positions=query_positions,
            )

        # logits: [num_query_total, vocab_size] or [B, K, vocab_size]
        # DFlash returns logits for mask positions only; shape [B*K, vocab]
        # The sample indices are the mask positions (skip the bonus slot)
        sample_indices = torch.tensor(
            [i * num_query_per_req + 1 + k for i in range(B) for k in range(K)],
            dtype=torch.int64, device=self.device,
        )
        if logits.dim() == 2:
            sampled_logits = logits[sample_indices]  # [B*K, vocab]
        else:
            sampled_logits = logits.view(-1, logits.shape[-1])[sample_indices]

        # Greedy sample (temperature handled by main verifier)
        draft_tokens = sampled_logits.argmax(dim=-1).int().view(B, K)
        return draft_tokens

    def _build_attn_metadata(
        self,
        seq_lens:         torch.Tensor,  # [B]
        block_tables:     torch.Tensor,  # [B, max_blocks]
        query_start_loc:  torch.Tensor,  # [B+1]
        slot_mapping:     torch.Tensor,  # [num_query_total]
        num_query_total:  int,
        num_query_per_req: int,
        max_seq_len:      int,
    ):
        """Try to build FlashAttentionMetadata; return None on import error."""
        try:
            from vllm.v1.attention.backends.flash_attn import (
                FlashAttentionMetadata,
            )
        except ImportError:
            logger.warning("FlashAttentionMetadata not available; skipping forward.")
            return None

        return FlashAttentionMetadata(
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            block_tables=block_tables,
            slot_mapping=slot_mapping,
            num_actual_tokens=num_query_total,
            max_query_len=num_query_per_req,
            max_seq_len=max_seq_len,
        )

    def _get_mask_token_id(self) -> int:
        hf_config = self.draft_model_config.hf_config
        dflash_config = getattr(hf_config, "dflash_config", None) or {}
        if "mask_token_id" in dflash_config:
            return dflash_config["mask_token_id"]
        if hasattr(hf_config, "mask_token_id"):
            return hf_config.mask_token_id
        return 0


# ── ZMQ server loop ────────────────────────────────────────────────────────────

def run_server(runner: DraftModelRunner, address: str) -> None:
    ctx = zmq.Context()
    sock = ctx.socket(zmq.ROUTER)
    sock.bind(address)
    logger.info("Draft server listening on %s", address)

    running = True

    def _stop(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    while running:
        try:
            parts = sock.recv_multipart(flags=zmq.NOBLOCK)
        except zmq.Again:
            time.sleep(0.0005)
            continue

        # ROUTER frame layout: [identity, empty, header_frame, payload_frame]
        if len(parts) < 4:
            continue
        identity, _, header_frame, payload_frame = parts[0], parts[1], parts[2], parts[3]

        try:
            header = parse_header(header_frame)
            t = header["t"]

            if t == MSG_PING:
                resp_h, resp_p = build_ack()

            elif t == MSG_PREFILL:
                hs, pos = parse_prefill_payload(header, payload_frame)
                runner.handle_prefill(header["seq_id"], hs, pos)
                resp_h, resp_p = build_ack()

            elif t == MSG_DECODE:
                hs, pos, temps, sds = parse_decode_payload(header, payload_frame)
                draft_tokens = runner.handle_decode(
                    header["seq_ids"], hs, pos, temps, sds
                )
                resp_h, resp_p = build_draft_response(draft_tokens)

            elif t == MSG_FREE:
                runner.handle_free(header["seq_id"])
                resp_h, resp_p = build_ack()

            else:
                resp_h, resp_p = build_error(f"unknown message type {t}")

        except Exception as exc:
            logger.exception("Error handling message type %s", header.get("t"))
            resp_h, resp_p = build_error(str(exc))

        sock.send_multipart([identity, b"", resp_h, resp_p])

    sock.close()
    ctx.destroy()
    logger.info("Draft server stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="DFlash2 disaggregated draft server")
    parser.add_argument("--device",   type=int,  default=1,
                        help="CUDA device index (default: 1)")
    parser.add_argument("--address",  type=str,  default="tcp://0.0.0.0:50052",
                        help="ZMQ ROUTER bind address")
    parser.add_argument("--config-json", type=str, required=True,
                        help="Path to JSON file containing serialised VllmConfig fields")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    device = torch.device("cuda:0")  # after remapping, device 0 is our GPU

    with open(args.config_json) as f:
        config_dict = json.load(f)

    runner = DraftModelRunner(config_dict, device)
    run_server(runner, args.address)


if __name__ == "__main__":
    main()
