# SPDX-License-Identifier: Apache-2.0
# slot tensors use int64 (Long) throughout — FlashInfer do_kv_cache_update requires Long.
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

    def __init__(self, vllm_config_dict: dict, device: torch.device,
                 *, dist_master_port: int = 29600, kv_headroom_gb: float = 1.0):
        self.device = device
        self.dist_master_port = dist_master_port
        self.kv_headroom_bytes = int(kv_headroom_gb * 1024 ** 3)
        torch.cuda.set_device(device)

        self._build_vllm_config(vllm_config_dict)
        self._load_model()
        self._init_kv_cache()
        self._init_block_manager()

        self.seq_block_tables: dict[str, list[int]] = {}
        self.seq_lengths:      dict[str, int]       = {}
        self._cuda_graph = None
        self._paged_kv_builders: list = []
        self._cpu_block_idx_buf = None

        self._warmup_kernels()

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
        os.environ.setdefault("MASTER_PORT", str(self.dist_master_port))
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        if not dist.is_initialized():
            dist.init_process_group(backend="gloo")  # gloo=CPU-only, no GPU resource leak
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
            # Required for DFlash2’s GDN cross-attention to use the
            # FlashQLA legacy SM75-optimised kernel instead of falling back
            # to a generic implementation that produces wrong results.
            additional_config={"gdn_prefill_backend": "flashqla_legacy"},
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
        os.environ.setdefault("MASTER_PORT", str(self.dist_master_port))
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        if not dist.is_initialized():
            dist.init_process_group(backend="gloo")  # gloo=CPU-only, no GPU resource leak
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
        usable_bytes  = max(0, free_bytes - self.kv_headroom_bytes)

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

        # ── Step 4b: resolve KV cache layout (engine core normally does this) ─────
        # build_for_drafting compares layout against a list of strings like
        # ['BLHNC', 'LBHNC', ...].  Store the layout NAME (string), not the
        # KVCacheLayout enum, so the comparison succeeds.
        _layout_str = os.environ.get("VLLM_KV_CACHE_LAYOUT", "BLHNC")
        object.__setattr__(self.vllm_config.cache_config, "kv_cache_layout", _layout_str)
        if hasattr(kv_cache_config, "kv_cache_layout"):
            object.__setattr__(kv_cache_config, "kv_cache_layout", _layout_str)

        # ── Step 5: allocate KV tensors in the correct physical layout ────────
        layout    = KVCacheLayout[_layout_str]  # enum required by allocate_kv_cache
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

    def _warmup_kernels(self) -> None:
        """Pre-compile FlashInfer JIT kernels via a dummy forward pass.

        On SM75 (RTX 2080 Ti) first-time compilation is 60-120 s.  Running this
        during startup ensures actual requests hit the kernel cache.
        """
        logger.info("Warming up FlashInfer kernels (may take a few minutes on SM75)…")
        seq_id = "__warmup__"
        try:
            H        = self.draft_model_config.get_hidden_size()
            dummy_T  = self.block_size          # one full block of context
            dummy_hs = torch.zeros(dummy_T, H, dtype=self.dtype, device=self.device)
            dummy_pos = torch.arange(dummy_T,  dtype=torch.int64, device=self.device)
            logger.info("Warmup: running handle_prefill (precompute_and_store_context_kv)…")
            self.handle_prefill(seq_id, dummy_hs, dummy_pos)
            torch.cuda.synchronize(self.device)
            logger.info("Warmup: handle_prefill OK")

            dummy_new   = torch.zeros(1, H, dtype=self.dtype, device=self.device)
            dummy_pos_n = torch.tensor([dummy_T], dtype=torch.int64, device=self.device)
            dummy_temps = torch.ones(1,  dtype=torch.float32, device=self.device)
            dummy_seeds = torch.zeros(1, dtype=torch.int64,   device=self.device)
            dummy_bonus = torch.zeros(1, dtype=torch.int32,   device=self.device)
            logger.info("Warmup: running handle_decode (model forward pass)…")
            self.handle_decode([seq_id], dummy_new, dummy_pos_n, dummy_temps, dummy_seeds, dummy_bonus)
            torch.cuda.synchronize(self.device)
            logger.info("FlashInfer warmup complete.")
        except Exception as exc:
            logger.warning("Warmup failed (non-fatal): %s", exc)
        finally:
            self.handle_free(seq_id)
        try:
            self._init_persistent_metadata()
        except Exception as _e:
            logger.warning("Persistent metadata init failed: %s", _e)
        self._cuda_graph = None  # graph disabled: always use eager path

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
        logger.warning("PREFILL seq=%s T=%d pos0=%d posN=%d",
                       seq_id, T, int(positions[0].item()), int(positions[-1].item()))
        needed_blocks = -(-T // self.block_size)
        if needed_blocks > self.num_blocks:
            raise RuntimeError(
                f"PREFILL T={T} needs {needed_blocks} blocks but only "
                f"{self.num_blocks} available — dropping sequence"
            )
        hidden_states = hidden_states.to(device=self.device, dtype=self.dtype)
        positions     = positions.to(device=self.device)

        blocks       = self.block_manager.allocate(T)
        if seq_id in self.seq_block_tables:
            # Chunked prefill: extend existing context rather than overwriting.
            # Each chunk writes its own new blocks; accumulate them.
            self.seq_block_tables[seq_id].extend(blocks)
            self.seq_lengths[seq_id] += T
        else:
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
        bonus_ids:     torch.Tensor,  # [B] int32 — actual token IDs for j=0
    ) -> torch.Tensor:                # [B, K]
        B = len(seq_ids)
        hidden_states = hidden_states.to(device=self.device, dtype=self.dtype)
        positions     = positions.to(device=self.device)

        # Extend each sequence with the newly generated token’s KV.
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

        return self._run_draft_forward(seq_ids, positions, bonus_ids)

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
        positions: torch.Tensor,
        bonus_ids: torch.Tensor,
    ) -> torch.Tensor:
        if (getattr(self, "_cuda_graph", None) is not None
                and len(seq_ids) == 1
                and not torch.cuda.is_current_stream_capturing()):
            return self._run_draft_forward_graph(seq_ids, positions, bonus_ids)
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
        # Override j=0 (bonus position) with the actual next-token ID.
        # In co-located DFlash2, j=0 uses next_token_id not mask_token_id.
        # With mask_token_id at j=0 the forward pass writes wrong K/V to the
        # bonus slot (T_old), corrupting all mask-token attention.
        _bids = bonus_ids.to(device=self.device, dtype=torch.int32)
        for _i in range(B):
            input_ids[_i * num_query_per_req] = int(_bids[_i].item())

        # Positions: the bonus token is at context_lens[i] (the first slot after
        # the last valid context position); query tokens at context_lens[i] + j
        # for j in [0, num_query_per_req).  Matches the co-located reference
        # (query_pos = last_valid_pos + 1 + query_off); the previous T_old start
        # made the bonus slot collide with the context token's slot (~55% acc).
        query_positions = torch.zeros(num_query_total, dtype=torch.int64, device=self.device)
        for i in range(B):
            base = int(positions[i].item())
            for j in range(num_query_per_req):
                query_positions[i * num_query_per_req + j] = base + 1 + j

        # Slot mapping for query tokens.
        # The bonus slot (j=0) reuses position T_old = positions[i] — the same
        # Allocate query slots. Save the pre-query block count per sequence so
        # we can release the temporary query blocks after the forward pass.
        pre_query_counts: list[int] = []
        query_slots = torch.zeros(num_query_total, dtype=torch.int64, device=self.device)
        for i, seq_id in enumerate(seq_ids):
            blocks   = self.seq_block_tables[seq_id]
            pre_query_counts.append(len(blocks))   # save BEFORE extending
            T_bonus  = int(positions[i].item())   # = T_old (bonus position)
            for j in range(num_query_per_req):
                pos = T_bonus + 1 + j             # T_old+1, T_old+2, …, T_old+K+1
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
        cad_seq_lens = context_lens + num_query_per_req - 1

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

        # Release the temporary query blocks — they are not part of the committed
        # sequence and must be freed each step or the block pool exhausts.
        for _i, _sid in enumerate(seq_ids):
            _blks = self.seq_block_tables[_sid]
            _extra = _blks[pre_query_counts[_i]:]
            del _blks[pre_query_counts[_i]:]
            if _extra:
                self.block_manager.free(_extra)

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

    # ━━ DraftGPUWorker: persistent metadata + incremental paged_kv updates ━━━
    #
    # Problem: build_for_drafting() was called every decode step (~10 ms) to
    # rebuild paged_kv_indptr/indices/last_page_len from scratch.
    #
    # Solution: call it ONCE in _init_persistent_metadata(), cache the builder
    # references, then update only the 3 changing CpuGpuBuffer tensors per step
    # via _update_paged_kv_direct() (~0.1 ms).  The SM75 wrapper attributes
    # (_sm75_kv_indptr_gpu, _sm75_kv_last_len_gpu) are VIEWS into these same
    # pre-allocated tensors, so they auto-reflect any in-place update.

    @torch.inference_mode()
    def _init_persistent_metadata(self) -> None:
        from vllm.v1.attention.backend import CommonAttentionMetadata
        B = 1
        Q = 1 + self.num_speculative_tokens
        H = self.draft_model_config.get_hidden_size()
        bs = self.block_size
        max_blocks = self.num_blocks + Q // bs + 4

        # Pre-allocate CPU buffer for block indices (avoids per-step allocation)
        self._cpu_block_idx_buf = torch.zeros(max_blocks, dtype=torch.int32)

        # Dummy prefill to prime the builders with a valid block table
        seq_id = "__pminit__"
        self.handle_prefill(
            seq_id,
            torch.zeros(bs, H, dtype=self.dtype, device=self.device),
            torch.arange(bs, dtype=torch.int64, device=self.device),
        )
        T_ctx   = self.seq_lengths[seq_id]
        blocks  = self.seq_block_tables[seq_id]
        n_blks  = len(blocks)
        bt      = torch.zeros(B, n_blks, dtype=torch.int32, device=self.device)
        bt[0, :n_blks] = torch.tensor(blocks[:n_blks], dtype=torch.int32,
                                      device=self.device)
        cad_sl  = torch.tensor([T_ctx + Q - 1], dtype=torch.int32, device=self.device)
        cad = CommonAttentionMetadata(
            query_start_loc=torch.tensor([0, Q], dtype=torch.int32, device=self.device),
            seq_lens=cad_sl,
            query_start_loc_cpu=torch.tensor([0, Q], dtype=torch.int32),
            seq_lens_cpu_upper_bound=int(cad_sl[0].item()),
            num_reqs=B, num_actual_tokens=Q,
            max_query_len=Q, max_seq_len=int(cad_sl[0].item()),
            block_table_tensor=bt,
            slot_mapping=torch.zeros(Q, dtype=torch.int64, device=self.device),
            causal=not self.requires_non_causal,
        )
        for _gl in self.attn_groups:
            for _g in _gl:
                try:
                    _g.get_metadata_builder().build_for_drafting(
                        common_attn_metadata=cad, draft_index=0)
                    self._paged_kv_builders.append(_g.get_metadata_builder())
                except Exception:
                    pass
        self.handle_free(seq_id)
        logger.info("DraftGPUWorker: persistent metadata ready (%d builders).",
                    len(self._paged_kv_builders))

    def _update_paged_kv_direct(self, blocks: list, T_ctx: int) -> None:
        """
        Incremental paged_kv update — replaces build_for_drafting() each step.

        For B=1 this is 3 tiny operations per builder:
          1. paged_kv_indptr  [0, num_pages]  — H2D 2 int32s
          2. paged_kv_indices [block_ids]     — H2D num_pages int32s
          3. paged_kv_last_page_len [fill]    — H2D 1 int32
        """
        if not self._paged_kv_builders:
            return
        num_pages = len(blocks)
        last_fill = T_ctx % self.block_size or self.block_size
        # Fill CPU buffer (no allocation)
        for _i in range(num_pages):
            self._cpu_block_idx_buf[_i] = blocks[_i]
        for builder in self._paged_kv_builders:
            builder.paged_kv_indptr.np[0] = 0
            builder.paged_kv_indptr.np[1] = num_pages
            builder.paged_kv_indptr.gpu[0:2].copy_(
                builder.paged_kv_indptr.cpu[0:2], non_blocking=True)
            if num_pages > 0:
                builder.paged_kv_indices[0:num_pages].copy_(
                    self._cpu_block_idx_buf[0:num_pages].to(self.device),
                    non_blocking=True)
            builder.paged_kv_last_page_len.np[0] = last_fill
            builder.paged_kv_last_page_len.gpu[0:1].copy_(
                builder.paged_kv_last_page_len.cpu[0:1], non_blocking=True)

    @torch.inference_mode()
    def _try_capture_cuda_graph(self) -> None:
        from vllm.forward_context import set_forward_context
        from vllm.v1.attention.backend import CommonAttentionMetadata
        B = 1
        K = self.num_speculative_tokens
        Q = 1 + K
        H = self.draft_model_config.get_hidden_size()
        bs = self.block_size
        self._sg_input_ids       = torch.zeros(Q, dtype=torch.int32,  device=self.device)
        self._sg_positions       = torch.zeros(Q, dtype=torch.int64,  device=self.device)
        self._sg_slot_mapping    = torch.zeros(Q, dtype=torch.int64,  device=self.device)
        self._sg_query_start_pos = torch.zeros(B, dtype=torch.int64,  device=self.device)
        self._sg_Q = Q
        self._sg_K = K
        seq_id = "__cudagraph__"
        self.handle_prefill(
            seq_id,
            torch.zeros(bs, H, dtype=self.dtype, device=self.device),
            torch.arange(bs, dtype=torch.int64, device=self.device),
        )

        def _setup_step():
            T_old  = self.seq_lengths[seq_id]
            blocks = self.seq_block_tables[seq_id]
            if T_old >= len(blocks) * bs:
                blocks.append(self.block_manager.allocate_one())
            slot_b = self._slot_for_position(blocks, T_old)
            self.model.precompute_and_store_context_kv(
                torch.zeros(1, H, dtype=self.dtype, device=self.device),
                torch.tensor([T_old], dtype=torch.int64, device=self.device),
                torch.tensor([slot_b], dtype=torch.int64, device=self.device),
            )
            self.seq_lengths[seq_id] = T_old + 1
            T_ctx   = self.seq_lengths[seq_id]
            T_bonus = T_old
            self._sg_query_start_pos[0] = T_bonus + 1
            self._sg_input_ids.fill_(self._get_mask_token_id())
            self._sg_input_ids[0] = 0
            for _j in range(Q):
                self._sg_positions[_j] = T_bonus + 1 + _j
            pre_cnt = len(blocks)
            for _j in range(Q):
                _pos = T_bonus + 1 + _j
                while _pos >= len(blocks) * bs:
                    blocks.append(self.block_manager.allocate_one())
                self._sg_slot_mapping[_j] = self._slot_for_position(blocks, _pos)
            # Incremental update (fast path)
            self._update_paged_kv_direct(blocks[:pre_cnt], T_ctx)
            # Build per_layer_meta with updated tensors
            n_blks = len(blocks)
            bt = torch.zeros(B, n_blks, dtype=torch.int32, device=self.device)
            bt[0, :n_blks] = torch.tensor(
                blocks[:n_blks], dtype=torch.int32, device=self.device)
            cad_sl = torch.tensor([T_ctx + Q - 1], dtype=torch.int32, device=self.device)
            cad = CommonAttentionMetadata(
                query_start_loc=torch.tensor([0, Q], dtype=torch.int32, device=self.device),
                seq_lens=cad_sl,
                query_start_loc_cpu=torch.tensor([0, Q], dtype=torch.int32),
                seq_lens_cpu_upper_bound=int(cad_sl[0].item()),
                num_reqs=B, num_actual_tokens=Q,
                max_query_len=Q, max_seq_len=int(cad_sl[0].item()),
                block_table_tensor=bt,
                slot_mapping=self._sg_slot_mapping,
                causal=not self.requires_non_causal,
            )
            per_layer_meta = {}
            for _gl in self.attn_groups:
                for _g in _gl:
                    try:
                        _meta = _g.get_metadata_builder().build_for_drafting(
                            common_attn_metadata=cad, draft_index=0)
                        for _ln in _g.layer_names:
                            per_layer_meta[_ln] = _meta
                        if (_meta is not None
                                and hasattr(_meta, 'prefill')
                                and _meta.prefill is not None
                                and hasattr(_meta.prefill, 'wrapper')
                                and _meta.prefill.wrapper is not None
                                and hasattr(_meta.prefill.wrapper, '_sm75_kv_indptr_gpu')):
                            _meta.prefill.wrapper._sm75_qsp_tensor = self._sg_query_start_pos
                    except Exception:
                        pass
            extra = blocks[pre_cnt:]
            del blocks[pre_cnt:]
            if extra:
                self.block_manager.free(extra)
            return per_layer_meta

        try:
            for _ in range(3):
                _plm = _setup_step()
                with set_forward_context(_plm, self.vllm_config, num_tokens=Q):
                    self.model(input_ids=self._sg_input_ids, positions=self._sg_positions)
                torch.cuda.synchronize(self.device)
            _plm_cap = _setup_step()
            self._cuda_graph = torch.cuda.CUDAGraph()
            with set_forward_context(_plm_cap, self.vllm_config, num_tokens=Q):
                with torch.cuda.graph(self._cuda_graph):
                    self._sg_logits = self.model(
                        input_ids=self._sg_input_ids,
                        positions=self._sg_positions,
                    )
            torch.cuda.synchronize(self.device)
            logger.info("DraftGPUWorker: CUDA graph captured (Q=%d).", Q)
        finally:
            self.handle_free(seq_id)

    @torch.inference_mode()
    def _run_draft_forward_graph(
        self,
        seq_ids:   list,
        positions: torch.Tensor,
        bonus_ids: torch.Tensor,
    ) -> torch.Tensor:
        seq_id  = seq_ids[0]
        Q       = self._sg_Q
        K       = self._sg_K
        bs      = self.block_size
        T_bonus = int(positions[0].item())
        blocks  = self.seq_block_tables[seq_id]
        pre_cnt = len(blocks)
        self._sg_input_ids.fill_(self._get_mask_token_id())
        self._sg_input_ids[0] = int(bonus_ids[0].item())
        for _j in range(Q):
            self._sg_positions[_j] = T_bonus + 1 + _j
        for _j in range(Q):
            _pos = T_bonus + 1 + _j
            while _pos >= len(blocks) * bs:
                blocks.append(self.block_manager.allocate_one())
            self._sg_slot_mapping[_j] = self._slot_for_position(blocks, _pos)
        T_ctx = self.seq_lengths[seq_id]
        self._sg_query_start_pos[0] = T_bonus + 1
        # Incremental paged_kv update — the key DraftGPUWorker improvement
        self._update_paged_kv_direct(blocks[:pre_cnt], T_ctx)
        self._cuda_graph.replay()
        flat = (self._sg_logits if self._sg_logits.dim() == 2
                else self._sg_logits.view(-1, self._sg_logits.shape[-1]))
        draft_tokens = flat[1:Q].argmax(dim=-1).int().view(1, K)
        extra = blocks[pre_cnt:]
        del blocks[pre_cnt:]
        if extra:
            self.block_manager.free(extra)
        return draft_tokens


# ── ZMQ server loop ────────────────────────────────────────────────────────────

def run_server(runner: DraftModelRunner, address: str) -> None:
    ctx  = zmq.Context()
    sock = ctx.socket(zmq.ROUTER)
    sock.bind(address)
    logger.info("Draft server listening on %s", address)

    running = True

    def _stop(sig, frame):
        nonlocal running
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
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
                hs, pos, temps, sds, bonus_ids = parse_decode_payload(header, payload_frame)
                draft_tokens = runner.handle_decode(
                    header["seq_ids"], hs, pos, temps, sds, bonus_ids
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
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    ctx.destroy()
    logger.info("Draft server stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="DFlash2 disaggregated draft server")
    parser.add_argument("--device", type=int, default=None,
        help="CUDA device ordinal to use. Sets CUDA_VISIBLE_DEVICES if not already "
             "set in the environment. When CUDA_VISIBLE_DEVICES is already set this "
             "argument is ignored.")
    parser.add_argument("--address", type=str, default="tcp://0.0.0.0:50052",
        help="ZMQ ROUTER bind address (default: tcp://0.0.0.0:50052).")
    parser.add_argument("--config-json", type=str, required=True,
        help="Path to JSON config produced by DisaggDFlashProposer.write_draft_config.")
    parser.add_argument("--dist-master-port", type=int, default=29600,
        help="TCP port for the single-rank gloo process group (default: 29600).")
    parser.add_argument("--kv-headroom-gb", type=float, default=1.0,
        help="GPU memory (GiB) to reserve beyond the KV cache (default: 1.0).")
    args = parser.parse_args()

    try:
        from setproctitle import setproctitle
        setproctitle("VLLM::Worker_DFlash2")
    except ImportError:
        pass
    if args.device is not None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.device))
    device = torch.device("cuda:0")

    with open(args.config_json) as f:
        config_dict = json.load(f)

    runner = DraftModelRunner(config_dict, device,
                              dist_master_port=args.dist_master_port,
                              kv_headroom_gb=args.kv_headroom_gb)
    run_server(runner, args.address)


if __name__ == "__main__":
    main()
