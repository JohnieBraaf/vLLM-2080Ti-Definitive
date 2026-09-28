# SPDX-License-Identifier: Apache-2.0
"""Standalone draft server for disaggregated DFlash2 speculative decoding.

v2: uses vLLM's proper KV cache infrastructure (init_attn_backend,
allocate_kv_cache, bind_kv_cache_to_layers) so FlashInfer attention
works correctly for both the KV-write path (precompute_and_store_context_kv)
and the KV-read path (model forward producing draft tokens).

Run on GPU1:
    python -m vllm.v1.spec_decode.disagg_dflash.draft_server \
        --device 1 \
        --address tcp://0.0.0.0:50052 \
        --config-json /tmp/dflash_draft_config.json
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time

import torch
import zmq

from vllm.logger import init_logger
from vllm.v1.spec_decode.disagg_dflash.protocol import (
    MSG_ACK,
    MSG_DECODE,
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

    v2 uses vLLM's proper KV cache infrastructure:
      - get_kv_cache_spec() for per-layer specs
      - init_attn_backend() for FlashInfer attention groups
      - allocate_kv_cache() for properly-formatted KV tensors
      - bind_kv_cache_to_layers() to bind KV into attention layers
    """

    def __init__(self, vllm_config_dict: dict, device: torch.device):
        self.device = device
        torch.cuda.set_device(device)

        self._build_vllm_config(vllm_config_dict)
        self._load_model()
        self._init_kv_cache()
        self._init_block_manager()

        self.seq_block_tables: dict[str, list[int]] = {}
        self.seq_lengths:      dict[str, int]       = {}

        logger.info(
            "DraftModelRunner ready on %s: %d KV blocks (block_size=%d)",
            device, self.num_blocks, self.block_size,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Initialisation
    # ──────────────────────────────────────────────────────────────────────────

    def _build_vllm_config(self, d: dict) -> None:
        spec = d["speculative_config"]
        self.draft_model_path = spec["model"]
        self.num_speculative_tokens = spec["num_speculative_tokens"]
        self.kv_cache_dtype_str = spec.get("kv_cache_dtype") or "auto"
        self.max_model_len = d["model_config"]["max_model_len"]
        self.dtype_str = d["model_config"].get("dtype", "float16")
        self.dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[
            self.dtype_str
        ]
        self._config_dict = d

    def _init_distributed(self) -> None:
        import torch.distributed as dist
        from vllm.config import set_current_vllm_config
        from vllm.distributed.parallel_state import (
            init_distributed_environment,
            initialize_model_parallel,
        )

        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12356")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        init_distributed_environment(
            world_size=1, rank=0,
            local_rank=self.device.index or 0,
        )
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )
        # Temporarily set a minimal config so get_current_vllm_config() works
        # during initialize_model_parallel. We'll replace it after EngineArgs.
        # (Some internal vLLM calls read the global config during init.)

    def _load_model(self) -> None:
        from vllm.engine.arg_utils import EngineArgs
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.model_loader import get_model
        from vllm.model_executor.models.qwen3_dflash import dflash_has_any_non_causal

        logger.info("Loading DFlash2 draft model: %s", self.draft_model_path)

        kv_dtype = (
            self.kv_cache_dtype_str
            if self.kv_cache_dtype_str not in (None, "auto")
            else "auto"
        )
        engine_args = EngineArgs(
            model=self.draft_model_path,
            max_model_len=self.max_model_len,
            dtype=self.dtype_str,
            gpu_memory_utilization=0.01,  # we allocate KV cache ourselves
            enforce_eager=True,
            trust_remote_code=True,
            tensor_parallel_size=1,
            kv_cache_dtype=kv_dtype,
            disable_log_stats=True,
        )
        self.vllm_config = engine_args.create_engine_config()
        self.draft_model_config = self.vllm_config.model_config

        # DFlash2DraftModel reads vllm_config.speculative_config.draft_model_config
        # from its __init__. Inject a stub with __getattr__ fallback.
        class _StubSpecConfig:
            def __init__(self, mc, k):
                self.draft_model_config = mc
                self.num_speculative_tokens = k
                self.method = "dflash"
                self.parallel_drafting = True
                self.enable_adaptive_verification = False
                self.rejection_sample_method = "standard"
                self.draft_sample_method = "greedy"
                self.use_local_argmax_reduction = False
                self.use_heterogeneous_vocab = False
                self.use_fp64_gumbel = False
                self.disable_padded_drafter_batch = False
                self.kv_cache_dtype = None
                self.attention_backend = None

            def __getattr__(self, name):
                return None

        object.__setattr__(
            self.vllm_config,
            "speculative_config",
            _StubSpecConfig(self.draft_model_config, self.num_speculative_tokens),
        )

        # Keep the vllm config context active for the entire process lifetime.
        self._vllm_config_ctx = set_current_vllm_config(self.vllm_config)
        self._vllm_config_ctx.__enter__()

        self.requires_non_causal = dflash_has_any_non_causal(
            self.draft_model_config.hf_config
        )

        # Initialize distributed TP group (needed by VocabParallelEmbedding etc.)
        import torch.distributed as dist
        from vllm.distributed.parallel_state import (
            init_distributed_environment,
            initialize_model_parallel,
        )
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12356")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        init_distributed_environment(
            world_size=1, rank=0,
            local_rank=self.device.index or 0,
        )
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

        self.model = get_model(
            vllm_config=self.vllm_config,
            model_config=self.draft_model_config,
        )
        self.model.eval()
        logger.info("Draft model loaded.")

    def _init_kv_cache(self) -> None:
        """Initialise KV cache using vLLM's proper infrastructure.

        Uses get_kv_cache_spec → KVCacheConfig → init_attn_backend →
        allocate_kv_cache → bind_kv_cache_to_layers so FlashInfer
        attention is correctly wired for both writes and reads.
        """
        from vllm.v1.kv_cache_interface import (
            AttentionSpec,
            KVCacheConfig,
            KVCacheGroupSpec,
            KVCacheTensor,
            UniformTypeKVCacheSpecs,
        )
        from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
        from vllm.v1.kv_cache_layout import KVCacheLayout
        from vllm.v1.worker.gpu.attn_utils import init_attn_backend
        from vllm.v1.worker.utils import allocate_kv_cache, bind_kv_cache_to_layers
        from vllm.config import get_layers_from_vllm_config
        from vllm.model_executor.layers.attention import Attention

        # ── Step 1: per-layer KV specs from the loaded model's attention layers ─
        _all_attn = get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase)
        kv_specs = {
            _n: _s for _n, _layer in _all_attn.items()
            if (_s := _layer.get_kv_cache_spec(self.vllm_config)) is not None
        }

        # Filter to attention specs only (skip Mamba / GDN state).
        def _is_attn(spec) -> bool:
            if isinstance(spec, AttentionSpec):
                return True
            if isinstance(spec, UniformTypeKVCacheSpecs):
                return any(isinstance(s, AttentionSpec) for s in spec.kv_cache_specs.values())
            return False

        attn_specs: dict = {n: s for n, s in kv_specs.items() if _is_attn(s)}
        if not attn_specs:
            raise RuntimeError("No attention layers found in draft model")

        # Resolve first concrete AttentionSpec for sizing.
        first_spec = next(iter(attn_specs.values()))
        if isinstance(first_spec, UniformTypeKVCacheSpecs):
            first_spec = next(iter(first_spec.kv_cache_specs.values()))

        num_kv_heads  = first_spec.num_kv_heads
        head_size     = first_spec.head_size
        bs            = first_spec.block_size
        self.block_size = bs

        # ── Step 2: compute num_blocks from free memory ───────────────────────
        torch.cuda.synchronize(self.device)
        free_bytes, _ = torch.cuda.mem_get_info(self.device.index)
        usable_bytes  = max(0, free_bytes - 1 * 1024 ** 3)  # 1 GB overhead

        # k + v, float16 (2 bytes), [num_kv_heads, block_size, head_size]
        block_page_bytes = 2 * num_kv_heads * bs * head_size * 2
        num_attn_layers  = len(attn_specs)
        self.num_blocks  = max(1, int(usable_bytes // (num_attn_layers * block_page_bytes)))

        logger.info(
            "Allocating %d KV blocks (%d attn layers, %d heads, "
            "head_size=%d, block_size=%d) on %s",
            self.num_blocks, num_attn_layers, num_kv_heads,
            head_size, bs, self.device,
        )

        # ── Step 3: build KVCacheConfig ───────────────────────────────────────
        attn_names   = list(attn_specs.keys())
        layer_stride = self.num_blocks * block_page_bytes  # layer-outermost
        kv_tensor    = KVCacheTensor(
            size         = num_attn_layers * layer_stride,
            layers       = attn_names,
            layer_stride = layer_stride,
            block_stride = block_page_bytes,
        )
        kv_group = KVCacheGroupSpec(
            layer_names   = attn_names,
            kv_cache_spec = first_spec,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks        = self.num_blocks,
            kv_cache_tensors  = [kv_tensor],
            kv_cache_groups   = [kv_group],
        )
        self.kv_cache_config = kv_cache_config

        # ── Step 4: init_attn_backend → attn_groups with FlashInfer ──────────
        self.attn_groups, _, _ = init_attn_backend(
            kv_cache_config, self.vllm_config, self.device
        )

        # ── Step 5: allocate KV tensors in the correct physical layout ────────
        layout   = KVCacheLayout[os.environ.get("VLLM_KV_CACHE_LAYOUT", "BLHNC")]
        kv_caches = allocate_kv_cache(kv_cache_config, self.device, layout)
        self.kv_caches = kv_caches

        # ── Step 6: bind KV cache into attention layers ───────────────────────
        # bind_kv_cache_to_layers calls layer.bind_kv_cache(tensor) which is
        # the correct API — not module.kv_cache = tensor.
        fwd_ctx = get_layers_from_vllm_config(self.vllm_config, Attention)
        bind_kv_cache_to_layers(kv_caches, fwd_ctx)

        self.num_kv_heads = num_kv_heads
        self.head_size    = head_size

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

        blocks       = self.block_manager.allocate(T)
        self.seq_block_tables[seq_id] = blocks
        self.seq_lengths[seq_id]      = T

        slot_mapping = self._compute_slot_mapping_for_sequence(blocks, T)

        # precompute_and_store_context_kv is a direct KV write; no forward
        # context required — it writes via FlashInfer's do_kv_cache_update using
        # the slot_mapping, and the attention layers are already bound via
        # bind_kv_cache_to_layers.
        self.model.precompute_and_store_context_kv(
            hidden_states,
            positions,
            slot_mapping,
        )
        logger.debug("PREFILL seq=%s T=%d blocks=%d", seq_id, T, len(blocks))

    @torch.inference_mode()
    def handle_decode(
        self,
        seq_ids:       list[str],
        hidden_states: torch.Tensor,  # [B, H]
        positions:     torch.Tensor,  # [B]
        temperatures:  torch.Tensor,  # [B]
        seeds:         torch.Tensor,  # [B]
    ) -> torch.Tensor:                # [B, K]
        B = len(seq_ids)
        hidden_states = hidden_states.to(device=self.device, dtype=self.dtype)
        positions     = positions.to(device=self.device)

        # Extend each sequence with the newly generated token's KV.
        for i, seq_id in enumerate(seq_ids):
            T_old = self.seq_lengths[seq_id]
            T_new = T_old + 1
            blocks = self.seq_block_tables[seq_id]
            if T_new > len(blocks) * self.block_size:
                blocks.append(self.block_manager.allocate_one())

            new_slot = self._slot_for_position(blocks, T_old)
            hs_i     = hidden_states[i].unsqueeze(0)
            pos_i    = positions[i].unsqueeze(0)
            slot_i   = torch.tensor([new_slot], device=self.device, dtype=torch.int64)
            self.model.precompute_and_store_context_kv(hs_i, pos_i, slot_i)
            self.seq_lengths[seq_id] = T_new

        return self._run_draft_forward(seq_ids, positions)

    def handle_free(self, seq_id: str) -> None:
        blocks = self.seq_block_tables.pop(seq_id, [])
        self.block_manager.free(blocks)
        self.seq_lengths.pop(seq_id, None)
        logger.debug("FREE seq=%s released %d blocks", seq_id, len(blocks))

    # ──────────────────────────────────────────────────────────────────────────
    # Draft forward pass (v2: proper FlashInfer metadata via attn_groups)
    # ──────────────────────────────────────────────────────────────────────────

    def _run_draft_forward(
        self,
        seq_ids:   list[str],
        positions: torch.Tensor,   # [B] — positions of the newly appended token
    ) -> torch.Tensor:             # [B, K]
        from vllm.forward_context import set_forward_context
        from vllm.v1.attention.backend import CommonAttentionMetadata

        B  = len(seq_ids)
        K  = self.num_speculative_tokens
        bs = self.block_size

        # ── sequence metadata ─────────────────────────────────────────────────
        # After handle_decode, seq_lengths already include the new token.
        context_lens = torch.tensor(
            [self.seq_lengths[sid] for sid in seq_ids],
            dtype=torch.int32, device=self.device,
        )
        max_ctx = int(context_lens.max().item())
        # ── query layout: [bonus_token, mask_1, …, mask_K] per request ───────
        num_query_per_req = 1 + K
        num_query_total   = B * num_query_per_req

        mask_token_id = self._get_mask_token_id()
        input_ids = torch.full(
            (num_query_total,), mask_token_id,
            dtype=torch.int32, device=self.device,
        )

        # Positions: newly-appended token is at (context_lens[i] - 1),
        # query tokens at context_lens[i] - 1 + j for j in [0, num_query_per_req).
        query_positions = torch.zeros(num_query_total, dtype=torch.int64, device=self.device)
        for i in range(B):
            base = int(positions[i].item())
            for j in range(num_query_per_req):
                query_positions[i * num_query_per_req + j] = base + j

        # Slot mapping for query tokens (allocate fresh slots).
        query_slots = torch.zeros(num_query_total, dtype=torch.int64, device=self.device)
        for i, seq_id in enumerate(seq_ids):
            blocks = self.seq_block_tables[seq_id]
            T      = self.seq_lengths[seq_id]
            for j in range(num_query_per_req):
                pos = T + j  # slots for query tokens after the context
                while pos >= len(blocks) * bs:
                    blocks.append(self.block_manager.allocate_one())
                query_slots[i * num_query_per_req + j] = self._slot_for_position(blocks, pos)

        # Build block tables AFTER query slot allocation, which may extend block tables.
        max_blk = max(len(self.seq_block_tables[sid]) for sid in seq_ids)
        block_tables = self._build_block_table_tensor(seq_ids, max_blk)

        # query_start_loc [B+1]
        query_start_loc = (
            torch.arange(B + 1, dtype=torch.int32, device=self.device)
            * num_query_per_req
        )

        # ── CommonAttentionMetadata ───────────────────────────────────────────
        # DFlash2 cross-attention: seq_lens = context + query (full KV window).
        cad_seq_lens = context_lens + num_query_per_req

        cad = CommonAttentionMetadata(
            query_start_loc       = query_start_loc,
            seq_lens              = cad_seq_lens,
            query_start_loc_cpu   = query_start_loc.cpu(),
            seq_lens_cpu_upper_bound = int(cad_seq_lens.max().item()),
            num_reqs              = B,
            num_actual_tokens     = num_query_total,
            max_query_len         = num_query_per_req,
            max_seq_len           = int(cad_seq_lens.max().item()),
            block_table_tensor    = block_tables,
            slot_mapping          = query_slots,
            causal                = not self.requires_non_causal,
        )

        # ── per-layer FlashInfer metadata from attn_groups ────────────────────
        per_layer_meta: dict = {}
        for group_list in self.attn_groups:
            for group in group_list:
                try:
                    meta = group.get_metadata_builder().build_for_drafting(
                        common_attn_metadata=cad, draft_index=0
                    )
                    for ln in group.layer_names:
                        per_layer_meta[ln] = meta
                except Exception as exc:
                    logger.warning("build_for_drafting failed: %s", exc)

        if not per_layer_meta:
            logger.warning("No attention metadata built; returning zeros")
            return torch.zeros(B, K, dtype=torch.int32, device=self.device)

        # ── model forward ─────────────────────────────────────────────────────
        with set_forward_context(
            per_layer_meta,
            self.vllm_config,
            num_tokens=num_query_total,
        ):
            logits = self.model(
                input_ids=input_ids,
                positions=query_positions,
            )

        # logits: [num_query_total, vocab] or similar
        # Sample from mask positions (skip the bonus slot at each group start).
        sample_idx = torch.tensor(
            [i * num_query_per_req + 1 + k for i in range(B) for k in range(K)],
            dtype=torch.int64, device=self.device,
        )
        flat_logits = (
            logits if logits.dim() == 2
            else logits.view(-1, logits.shape[-1])
        )
        draft_tokens = flat_logits[sample_idx].argmax(dim=-1).int().view(B, K)
        return draft_tokens

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_slot_mapping_for_sequence(
        self, blocks: list[int], num_tokens: int
    ) -> torch.Tensor:
        slots = [self._slot_for_position(blocks, i) for i in range(num_tokens)]
        return torch.tensor(slots, dtype=torch.int64, device=self.device)

    def _slot_for_position(self, blocks: list[int], pos: int) -> int:
        return blocks[pos // self.block_size] * self.block_size + pos % self.block_size

    def _build_block_table_tensor(
        self, seq_ids: list[str], max_blocks: int
    ) -> torch.Tensor:
        B  = len(seq_ids)
        bt = torch.zeros(B, max_blocks, dtype=torch.int32, device=self.device)
        for i, seq_id in enumerate(seq_ids):
            blks = self.seq_block_tables[seq_id]
            bt[i, : len(blks)] = torch.tensor(blks, dtype=torch.int32, device=self.device)
        return bt

    def _get_mask_token_id(self) -> int:
        hf_config    = self.draft_model_config.hf_config
        dflash_config = getattr(hf_config, "dflash_config", None) or {}
        if "mask_token_id" in dflash_config:
            return dflash_config["mask_token_id"]
        if hasattr(hf_config, "mask_token_id"):
            return hf_config.mask_token_id
        return 0


# ── ZMQ server loop ────────────────────────────────────────────────────────────

def run_server(runner: DraftModelRunner, address: str) -> None:
    ctx  = zmq.Context()
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

        if len(parts) < 3:
            continue
        identity     = parts[0]
        header_frame = parts[2] if len(parts) >= 4 else parts[1]
        payload_frame = parts[3] if len(parts) >= 4 else parts[2]

        try:
            header = parse_header(header_frame)
            t      = header["t"]

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
    parser = argparse.ArgumentParser(description="DFlash2 disaggregated draft server v2")
    parser.add_argument("--device",     type=int, default=1)
    parser.add_argument("--address",    type=str, default="tcp://0.0.0.0:50052")
    parser.add_argument("--config-json", type=str, required=True)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    device = torch.device("cuda:0")

    with open(args.config_json) as f:
        config_dict = json.load(f)

    runner = DraftModelRunner(config_dict, device)
    run_server(runner, args.address)


if __name__ == "__main__":
    main()
