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
    build_ack,
    build_draft_response,
    build_error,
    parse_decode_payload,
    parse_header,
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
        self._dbg = bool(os.environ.get("VLLM_DISAGG_DEBUG"))
        self._dbg_steps = 0
        self._dbg_real = False
        torch.cuda.set_device(device)

        self._build_vllm_config(vllm_config_dict)
        self._load_model()
        self._init_kv_cache()
        self._init_block_manager()

        self.seq_block_tables: dict[str, list[int]] = {}
        self.seq_lengths:      dict[str, int]       = {}
        self._cuda_graph: torch.cuda.CUDAGraph | None = None
        self._sample_idx: torch.Tensor | None = None
        self._graph_verify = bool(os.environ.get("VLLM_DRAFT_GRAPH_VERIFY"))

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
        self.target_model_path = spec.get("target_model")
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
        self._load_target_head_and_embed()
        logger.info("Draft model loaded.")

    def _load_target_head_and_embed(self) -> None:
        """Give the drafter the target's LM head and input embeddings.

        The DFlash2 checkpoint ships neither (config.json has
        tie_word_embeddings=false and the safetensors hold no lm_head or
        embed_tokens): the drafter predicts in the target's hidden space, scores
        its candidates with the target's LM head and embeds real token ids with
        the target's vocabulary table.  The co-located path shares the live
        target modules (load_dflash_model); a separate process has to load the
        same tensors from the target checkpoint, otherwise every candidate logit
        is zero and no draft can ever be accepted.
        """
        from safetensors import safe_open

        if not self.target_model_path:
            logger.warning(
                "No target_model in the draft config: the drafter's lm_head and "
                "embed_tokens stay unloaded and all candidate logits will be 0."
            )
            return
        src = self.target_model_path
        index = os.path.join(src, "model.safetensors.index.json")
        if os.path.exists(index):
            with open(index) as f:
                weight_map = json.load(f)["weight_map"]
        else:
            with safe_open(os.path.join(src, "model.safetensors"), framework="pt") as f:
                weight_map = {k: "model.safetensors" for k in f.keys()}

        def _pick(suffix: str) -> str:
            keys = sorted(k for k in weight_map if k.endswith(suffix))
            if not keys:
                raise RuntimeError(f"{src} has no '*{suffix}' weight")
            return keys[0]

        embed = None
        for name, module in self.model.named_modules():
            if name.endswith("embed_tokens"):
                embed = module
        if embed is None:
            raise RuntimeError("drafter has no embed_tokens module")

        for module, key in (
            (self.model.lm_head, _pick("lm_head.weight")),
            (embed, _pick("embed_tokens.weight")),
        ):
            with safe_open(os.path.join(src, weight_map[key]), framework="pt") as f:
                tensor = f.get_tensor(key)
            param = module.weight
            if tuple(param.shape) != tuple(tensor.shape):
                raise RuntimeError(
                    f"{key}: target shape {tuple(tensor.shape)} does not match "
                    f"the drafter's {tuple(param.shape)}"
                )
            with torch.no_grad():
                param.copy_(tensor.to(device=self.device, dtype=param.dtype))
            logger.info(
                "Loaded target %s: %s %s mean=%.5f std=%.5f",
                key, tuple(param.shape), param.dtype,
                float(param.mean()), float(param.std()),
            )
            del tensor
            torch.cuda.empty_cache()

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
            logger.info("Warmup: running handle_prefill (precompute_and_store_context_kv)…")
            self.handle_prefill(seq_id, dummy_hs)
            torch.cuda.synchronize(self.device)
            logger.info("Warmup: handle_prefill OK")

            dummy_new   = torch.zeros(1, H, dtype=self.dtype, device=self.device)
            dummy_qsl   = torch.tensor([0, 1], dtype=torch.int32, device=self.device)
            dummy_rej   = torch.zeros(1, dtype=torch.int32,   device=self.device)
            dummy_temps = torch.ones(1,  dtype=torch.float32, device=self.device)
            dummy_seeds = torch.zeros(1, dtype=torch.int64,   device=self.device)
            dummy_bonus = torch.zeros(1, dtype=torch.int32,   device=self.device)
            logger.info("Warmup: running handle_decode (model forward pass)…")
            self.handle_decode([seq_id], dummy_new, dummy_qsl, dummy_rej,
                               dummy_temps, dummy_seeds, dummy_bonus)
            torch.cuda.synchronize(self.device)
            self._capture_draft_graph()
            logger.info("FlashInfer warmup complete.")
        except Exception as exc:
            logger.warning("Warmup failed (non-fatal): %s", exc)
        finally:
            self.handle_free(seq_id)

    # ──────────────────────────────────────────────────────────────────────────
    # Request handlers
    # ──────────────────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def _append_context(self, seq_id: str, hidden_states: torch.Tensor) -> None:
        """Append committed tokens to a sequence's context at their true positions.

        `seq_lengths` is the sequence's committed token count, which for a
        coherent draft context is also the target's sequence length — the two
        stay in lockstep because only the accepted prefix of each target batch is
        ever appended here.
        """
        T = hidden_states.shape[0]
        L = self.seq_lengths.get(seq_id, 0)
        blocks = self.seq_block_tables.setdefault(seq_id, [])
        needed = -(-(L + T) // self.block_size)
        if needed > self.num_blocks:
            raise RuntimeError(
                f"APPEND seq={seq_id} L={L} T={T} needs {needed} blocks but only "
                f"{self.num_blocks} available — dropping sequence"
            )
        while len(blocks) < needed:
            blocks.append(self.block_manager.allocate_one())

        positions = torch.arange(L, L + T, dtype=torch.int64, device=self.device)
        slots = torch.tensor(
            [self._slot_for_position(blocks, L + j) for j in range(T)],
            dtype=torch.int64, device=self.device,
        )
        self.model.precompute_and_store_context_kv(hidden_states, positions, slots)
        self.seq_lengths[seq_id] = L + T
        if self._dbg:
            logger.warning("APPEND seq=%s L=%d T=%d blocks=%d",
                           seq_id, L, T, len(blocks))

    @torch.inference_mode()
    def handle_prefill(
        self,
        seq_id: str,
        hidden_states: torch.Tensor,  # [T, H]
        use_aux:       bool = False,
    ) -> None:
        """Out-of-band context load (kernel warmup only).

        Real traffic always arrives as whole-batch DECODE messages.
        """
        hidden_states = hidden_states.to(device=self.device, dtype=self.dtype)
        if use_aux:
            hidden_states = self._combine_aux(hidden_states)
        L = self.seq_lengths.get(seq_id, 0)
        T = hidden_states.shape[0]
        logger.warning("PREFILL seq=%s T=%d pos0=%d posN=%d", seq_id, T, L, L + T - 1)
        self._append_context(seq_id, hidden_states)

    @torch.inference_mode()
    def handle_decode(
        self,
        seq_ids:       list[str],
        hidden_states: torch.Tensor,  # [T_total, H]
        query_start_loc: torch.Tensor,  # [B+1] int32 — offsets into T_total
        num_rejected:  torch.Tensor,  # [B] int32 — trailing tokens the target rejected
        temperatures:  torch.Tensor,  # [B]
        seeds:         torch.Tensor,  # [B]
        bonus_ids:     torch.Tensor,  # [B] int32 — actual token IDs for j=0
        use_aux:       bool = False,
    ) -> torch.Tensor:                # [B, K]
        hidden_states = hidden_states.to(device=self.device, dtype=self.dtype)
        # Concatenated aux rows, pre-projection: the captured step applies the
        # drafter's fc itself when it can, so the projection only has to be
        # computed eagerly for the steps the graph does not cover.
        raw_states = hidden_states
        if use_aux and self._a_raw is None:
            hidden_states = self._combine_aux(hidden_states)
        qsl      = [int(v) for v in query_start_loc.tolist()]
        rejected = [int(v) for v in num_rejected.tolist()]
        B        = len(seq_ids)

        if self._dbg and self._dbg_steps < 40 and not str(seq_ids[0]).startswith("_"):
            self._dbg_steps += 1
            self._dbg_real = True
            logger.warning(
                "DECODE B=%d T=%d qsl=%s rejected=%s ctx=%s bonus=%s aux=%s",
                B, qsl[-1], qsl, rejected,
                [self.seq_lengths.get(s, 0) for s in seq_ids],
                bonus_ids.tolist(), use_aux,
            )

        # A request's run holds [accepted prefix][rejected drafts]; only the
        # prefix became real tokens for the target, so only it may enter the
        # drafter's KV cache (the reference leaves rejected rows on PAD_SLOT_ID).
        graph_pre_L = None
        for i, seq_id in enumerate(seq_ids):
            t0, t1  = qsl[i], qsl[i + 1]
            n_valid = t1 - t0 - rejected[i]
            if n_valid < 0:
                logger.warning("DECODE: %s rejected=%d > T=%d, clamping",
                               seq_id, rejected[i], t1 - t0)
                n_valid = 0
            if n_valid == 0:
                continue
            pre_L  = self.seq_lengths.get(seq_id, 0)
            staged = False
            if (self._cuda_graph is not None and len(seq_ids) == 1
                    and t1 - t0 == self._graph_shape()[1]):
                # The whole target batch is appended and the sequence length is
                # rewound to the committed prefix: the rows past it stay in the
                # cache as the next append's scratch, exactly as the co-located
                # speculator leaves them.
                if self._a_raw is not None:
                    staged = raw_states.shape[-1] == self._a_raw.shape[-1]
                    if staged:
                        self._a_raw[: t1 - t0].copy_(raw_states[t0:t1])
                        if self._graph_verify:
                            self._append_context(
                                seq_id, self._combine_aux(raw_states[t0:t1])[:n_valid]
                                if use_aux else raw_states[t0:t0 + n_valid])
                else:
                    staged = True
                    self._a_hidden[: t1 - t0].copy_(hidden_states[t0:t1])
                    if self._graph_verify:
                        self._append_context(seq_id, hidden_states[t0:t0 + n_valid])
            if staged:
                self.seq_lengths[seq_id] = pre_L + n_valid
                graph_pre_L = pre_L
                continue
            rows = (self._combine_aux(raw_states[t0:t1])
                    if use_aux and self._a_raw is not None
                    else hidden_states[t0:t1])
            self._append_context(seq_id, rows[:n_valid])

        return self._run_draft_forward(seq_ids, bonus_ids, graph_pre_L)

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
        bonus_ids: torch.Tensor,
        pre_L:     int | None = None,
    ) -> torch.Tensor:
        """Run the draft step, replaying the captured graph when it applies.

        The graph is captured for the single-request shape only, so a batching
        scheduler would fall back to the eager path here — as does any step
        whose batch is not exactly one full Q-row append.
        """
        if self._cuda_graph is not None and len(seq_ids) == 1 and pre_L is not None:
            if not self._graph_verify:
                return self._run_draft_forward_graph(seq_ids[0], bonus_ids, pre_L)
            ref = self._run_draft_forward_eager(seq_ids, bonus_ids).clone()
            got = self._run_draft_forward_graph(seq_ids[0], bonus_ids, pre_L).clone()
            if not torch.equal(ref, got):
                logger.warning("GRAPH VERIFY mismatch: eager=%s graph=%s",
                               ref.tolist(), got.tolist())
            return got
        return self._run_draft_forward_eager(seq_ids, bonus_ids)

    def _run_draft_forward_eager(
        self,
        seq_ids:   list[str],
        bonus_ids: torch.Tensor,
    ) -> torch.Tensor:
        from vllm.forward_context import set_forward_context
        from vllm.v1.attention.backend import CommonAttentionMetadata

        B  = len(seq_ids)
        K  = self.num_speculative_tokens
        bs = self.block_size

        # ── sequence metadata ─────────────────────────────────────────────────
        # seq_lengths counts committed tokens, i.e. the context window is
        # positions [0, L) — the same convention as a normal decode step.
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

        # Positions: the bonus token sits at the first free position L and the
        # masks follow it, i.e. query_off = 0..K at absolute positions L+off.
        # Matches the reference's query_pos = last_valid_pos + 1 + query_off.
        query_positions = torch.zeros(num_query_total, dtype=torch.int64, device=self.device)
        for i in range(B):
            base = int(context_lens[i].item())
            for j in range(num_query_per_req):
                query_positions[i * num_query_per_req + j] = base + j

        # Allocate query slots in the same page table as the context (the next
        # step's context append writes over them). Save the pre-query block count
        # per sequence so the temporary query blocks can be released afterwards.
        pre_query_counts: list[int] = []
        query_slots = torch.zeros(num_query_total, dtype=torch.int64, device=self.device)
        for i, seq_id in enumerate(seq_ids):
            blocks = self.seq_block_tables[seq_id]
            pre_query_counts.append(len(blocks))   # save BEFORE extending
            base   = int(context_lens[i].item())
            for j in range(num_query_per_req):
                pos = base + j
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
        # The query block spans num_query_per_req positions starting right after
        # the context, so the window needs one entry per query slot too.
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
        # The DFlash2 drafter returns hidden states; the draft tokens come from
        # its candidate selector, not from an LM head argmax.
        with set_forward_context(
            per_layer_meta,
            self.vllm_config,
            num_tokens=num_query_total,
        ):
            hidden_all = self.model(
                input_ids=input_ids,
                positions=query_positions,
            )

        draft_tokens = self._select_draft_tokens(hidden_all, _bids, B, K)

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
    # Draft token selection (DFlash2 candidate selector)
    # ──────────────────────────────────────────────────────────────────────────

    def _combine_aux(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project concatenated target aux hidden states into drafter space.

        The drafter's KV cache must be built from fc(concat(aux)) — the same
        features the co-located speculator feeds to precompute_and_store_context_kv.
        Feeding the target's plain last hidden state instead silently degrades
        the drafts to noise.
        """
        return self.model.combine_hidden_states(hidden_states)

    def _select_draft_tokens(
        self,
        hidden_all: torch.Tensor,   # [B * (1+K), H]
        bonus_ids:  torch.Tensor,   # [B] bonus (anchor) token ids
        B: int,
        K: int,
        out: torch.Tensor | None = None,   # [B, K] int32, pre-allocated for graphs
    ) -> torch.Tensor:              # [B, K] int32
        """Pick draft tokens with DFlash2's candidate selector.

        Mirrors the co-located DFlash2Speculator: take the top-k unary
        candidates for each mask position, score the edges between consecutive
        candidate slots with the learned codebooks, then walk the best edge.
        A plain argmax over the drafter's LM head (or, worse, over its hidden
        states) yields a different token and kills acceptance.
        """
        num_query_per_req = 1 + K
        if self._sample_idx is None or self._sample_idx.shape[0] != B * K:
            self._sample_idx = torch.tensor(
                [i * num_query_per_req + 1 + k for i in range(B) for k in range(K)],
                dtype=torch.int64, device=self.device,
            )
        sample_idx = self._sample_idx
        hidden = hidden_all.view(-1, hidden_all.shape[-1])[sample_idx]
        hidden = hidden.view(B, K, -1)

        candidate_ids, unary_logits = self._candidates(hidden.reshape(B * K, -1))
        top_k = candidate_ids.shape[-1]
        candidate_ids = candidate_ids.view(B, K, top_k)
        unary_logits  = unary_logits.view(B, K, top_k)

        scores = self.model.model.candidate_selector(
            candidate_ids,
            unary_logits,
            hidden,
            bonus_ids.to(device=self.device, dtype=torch.int64),
        )

        # Greedy walk: step l ranks successors given the candidate picked at
        # step l-1 (the anchor token at l=0).
        if out is None:
            out = torch.zeros(B, K, dtype=torch.int32, device=self.device)
        rows = torch.arange(B, device=self.device)
        prev = torch.zeros(B, dtype=torch.int64, device=self.device)
        if self._live_debug():
            logger.warning(
                "DRAFT hidden mean=%.3f std=%.3f absmax=%.3f top1_ne=%d "
                "cand0=%s unary0=%s anchor=%s",
                float(hidden.mean()), float(hidden.std()),
                float(hidden.abs().max()),
                int((hidden.abs() < 1e-8).all(dim=-1).sum()),
                candidate_ids[0, 0, :5].tolist(),
                [round(float(v), 3) for v in unary_logits[0, 0, :5]],
                bonus_ids.tolist(),
            )
        for _l in range(K):
            idx = scores[:, _l, prev, :].argmax(dim=-1)
            out[:, _l] = candidate_ids[rows, _l, idx].to(torch.int32)
            prev = idx
        if self._live_debug():
            logger.warning("DRAFT out=%s", out.tolist())
        return out

    def _candidates(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Vocab top-k candidates for the DFlash2 selector.

        Same as DFlash2Qwen3ForCausalLM.compute_candidates but the top-k runs in
        torch rather than FlashInfer's radix kernel: on SM75 that kernel returns
        bucket-strided indices with zeroed values, which poisons every draft.
        """
        lp = self.model.candidate_logits_processor
        lm_head = self.model.lm_head
        logits = lp._apply_head(lm_head, hidden_states, None)
        num_pad = lm_head.shard_indices.num_org_vocab_padding
        if num_pad > 0:
            logits[..., -num_pad:] = -float("inf")
        k = self.model.model.candidate_selector.top_k
        values, ids = torch.topk(logits.float(), k, dim=-1)
        ids = ids.to(torch.int64) + lm_head.shard_indices.org_vocab_start_index
        if self._dbg and self._dbg_steps == 1 and self._live_debug():
            fi_ids, fi_values = self.model.compute_candidates(hidden_states)
            logger.warning(
                "CAND tp=%d vstart=%d pad=%d logits=%s/%s mean=%.4f std=%.4f "
                "max=%.4f argmax=%s",
                lm_head.tp_size,
                lm_head.shard_indices.org_vocab_start_index,
                num_pad,
                tuple(logits.shape),
                logits.dtype,
                float(logits.mean()),
                float(logits.std()),
                float(logits.max()),
                logits.argmax(dim=-1)[: 3 * self.num_speculative_tokens].tolist(),
            )
            logger.warning(
                "CAND torch ids=%s vals=%s",
                ids[0, :5].tolist(),
                [round(float(v), 3) for v in values[0, :5]],
            )
            logger.warning(
                "CAND flashinfer ids=%s vals=%s",
                fi_ids[0, :5].tolist(),
                [round(float(v), 3) for v in fi_values[0, :5]],
            )
        if lm_head.tp_size > 1:
            from vllm.distributed import tensor_model_parallel_all_gather

            values = tensor_model_parallel_all_gather(values, dim=-1)
            ids = tensor_model_parallel_all_gather(ids, dim=-1)
            values, selected = torch.topk(values, k, dim=-1)
            ids = ids.gather(-1, selected)
        values = values.float()
        if lp.scale != 1.0:
            values = values * lp.scale
        if lp.soft_cap is not None:
            values = torch.tanh(values / lp.soft_cap) * lp.soft_cap
        return ids, values

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

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

    def _free_query_blocks(self, seq_id: str, pre_blocks: int) -> None:
        """Release the temporary query blocks allocated for one draft step."""
        blocks = self.seq_block_tables.get(seq_id)
        if not blocks:
            return
        extra = blocks[pre_blocks:]
        if extra:
            del blocks[pre_blocks:]
            self.block_manager.free(extra)

    # ──────────────────────────────────────────────────────────────────────────
    # CUDA graph path (batch size 1)
    # ──────────────────────────────────────────────────────────────────────────
    #
    # An eager draft step costs ~12 ms on Turing although only ~4 ms of it is GPU
    # work: the rest is ~5000 Python calls launching ~2000 kernels (5 draft layers
    # plus the full-vocabulary candidate head).  The draft step is strictly serial
    # with the target's — the target cannot verify drafts it has not been handed —
    # so that launch overhead lands straight on the critical path and is the whole
    # reason the disaggregated profile trails the co-located one.
    #
    # Capturing one decode step (model forward + candidate selection) into a CUDA
    # graph removes it.  Everything the recorded kernels read must live at a
    # stable address: per-step inputs are copied into dedicated buffers, and the
    # attention metadata is refreshed in place — FlashInfer's paged_kv_* tensors
    # are persistent buffers, and the SM75 kernel already takes its causal
    # position from a device tensor, so a replay sees current data.

    def _live_debug(self) -> bool:
        """Debug logging reads values back to the host, which a capture region
        cannot do, so it stays off while a graph is being captured."""
        return bool(self._dbg and self._dbg_real
                    and not torch.cuda.is_current_stream_capturing())

    def _graph_shape(self) -> tuple[int, int]:
        return 1, 1 + self.num_speculative_tokens

    def _init_graph_buffers(self) -> None:
        B, Q = self._graph_shape()
        K    = self.num_speculative_tokens
        dev  = self.device
        self._g_input_ids  = torch.zeros(Q, dtype=torch.int32, device=dev)
        self._g_positions  = torch.zeros(Q, dtype=torch.int64, device=dev)
        self._g_slots      = torch.zeros(Q, dtype=torch.int64, device=dev)
        self._g_seq_lens   = torch.zeros(B, dtype=torch.int32, device=dev)
        self._g_bonus      = torch.zeros(B, dtype=torch.int32, device=dev)
        self._g_out        = torch.zeros(B, K, dtype=torch.int32, device=dev)
        self._g_qsl        = torch.arange(B + 1, dtype=torch.int32, device=dev) * Q
        self._g_qsl_cpu    = torch.arange(B + 1, dtype=torch.int32) * Q
        self._sample_idx   = torch.tensor(
            [i * Q + 1 + k for i in range(B) for k in range(K)],
            dtype=torch.int64, device=dev)
        # Padded to the whole pool so the width (and therefore the metadata) is
        # the same at capture and at every replay.
        self._g_block_table = torch.zeros(
            B, max(1, self.num_blocks), dtype=torch.int32, device=dev)
        self._g_cpu_ids    = torch.zeros(Q, dtype=torch.int32)
        self._g_cpu_pos    = torch.zeros(Q, dtype=torch.int64)
        self._g_cpu_slots  = torch.zeros(Q, dtype=torch.int64)
        self._g_cpu_bonus  = torch.zeros(B, dtype=torch.int32)
        self._g_cpu_bt     = torch.zeros(max(1, self.num_blocks), dtype=torch.int32)
        # Context append, captured alongside the forward.  The rows past a
        # request's committed prefix are real target hidden states for positions
        # the query's own K/V write replaces before any attention reads them, and
        # that the next append overwrites — the same ping-pong of committed and
        # scratch rows the co-located speculator runs.  No row's value depends on
        # the batch's valid length, so this shape can stay fixed.
        H = self.draft_model_config.get_hidden_size()
        self._a_hidden    = torch.zeros(Q, H, dtype=self.dtype, device=dev)
        self._a_positions = torch.zeros(Q, dtype=torch.int64, device=dev)
        self._a_slots     = torch.zeros(Q, dtype=torch.int64, device=dev)
        self._a_cpu_pos   = torch.zeros(Q, dtype=torch.int64)
        self._a_cpu_slots = torch.zeros(Q, dtype=torch.int64)
        # When the drafter projects concatenated aux states, that projection can
        # be captured too: it is one dense GEMM whose only input is the batch of
        # raw aux rows, so the target's fc cost leaves the critical path with the
        # rest of the step.
        inner = getattr(self.model, "model", None)
        fc    = getattr(inner, "fc", None)
        if (getattr(inner, "use_aux_hidden_state", False)
                and fc is not None and hasattr(fc, "in_features")):
            self._a_raw = torch.zeros(Q, fc.in_features, dtype=self.dtype, device=dev)
        else:
            self._a_raw = None

    def _stage_context_append(self, seq_id: str, pre_L: int) -> None:
        """Stage one Q-row context append for the captured step.

        `self._a_hidden` must already hold the target's whole batch of hidden
        states; the rows the target rejected still get written, exactly as the
        co-located speculator writes them.
        """
        B, Q   = self._graph_shape()
        bs     = self.block_size
        blocks = self.seq_block_tables.setdefault(seq_id, [])
        for j in range(Q):
            pos = pre_L + j
            while pos >= len(blocks) * bs:
                blocks.append(self.block_manager.allocate_one())
            self._a_cpu_pos[j]   = pos
            self._a_cpu_slots[j] = self._slot_for_position(blocks, pos)
        self._a_positions.copy_(self._a_cpu_pos, non_blocking=True)
        self._a_slots.copy_(self._a_cpu_slots, non_blocking=True)

    def _graph_append(self) -> None:
        states = self._a_hidden
        if self._a_raw is not None:
            states = self.model.combine_hidden_states(self._a_raw)
        self.model.precompute_and_store_context_kv(
            states, self._a_positions, self._a_slots)

    def _plan_graph_step(self, seq_id: str, bonus_id: int,
                         pre_L: int | None = None) -> tuple[dict, int]:
        """Stage one draft step's inputs and rebuild the attention metadata.

        Returns the per-layer metadata and the block count before the temporary
        query blocks were allocated (they must outlive the graph replay).
        """
        from vllm.v1.attention.backend import CommonAttentionMetadata

        B, Q   = self._graph_shape()
        bs     = self.block_size
        blocks = self.seq_block_tables.setdefault(seq_id, [])
        L      = self.seq_lengths.get(seq_id, 0)
        pre_blocks = len(blocks)

        if pre_L is not None:
            self._stage_context_append(seq_id, pre_L)

        self._g_cpu_ids.fill_(self._get_mask_token_id())
        self._g_cpu_ids[0] = bonus_id
        for j in range(Q):
            self._g_cpu_pos[j] = L + j
        for j in range(Q):
            pos = L + j
            while pos >= len(blocks) * bs:
                blocks.append(self.block_manager.allocate_one())
            self._g_cpu_slots[j] = self._slot_for_position(blocks, pos)
        self._g_cpu_bonus[0] = bonus_id
        for i, blk in enumerate(blocks):
            self._g_cpu_bt[i] = blk

        self._g_input_ids.copy_(self._g_cpu_ids, non_blocking=True)
        self._g_positions.copy_(self._g_cpu_pos, non_blocking=True)
        self._g_slots.copy_(self._g_cpu_slots, non_blocking=True)
        self._g_bonus.copy_(self._g_cpu_bonus, non_blocking=True)
        self._g_block_table.zero_()
        self._g_block_table[0, :len(blocks)].copy_(
            self._g_cpu_bt[:len(blocks)], non_blocking=True)
        self._g_seq_lens.fill_(L + Q)

        cad = CommonAttentionMetadata(
            query_start_loc       = self._g_qsl,
            seq_lens              = self._g_seq_lens,
            query_start_loc_cpu   = self._g_qsl_cpu,
            seq_lens_cpu_upper_bound = L + Q,
            num_reqs              = B,
            num_actual_tokens     = Q,
            max_query_len         = Q,
            max_seq_len           = L + Q,
            block_table_tensor    = self._g_block_table,
            slot_mapping          = self._g_slots,
            causal                = not self.requires_non_causal,
        )

        per_layer_meta: dict = {}
        for group_list in self.attn_groups:
            for group in group_list:
                meta = group.get_metadata_builder().build_for_drafting(
                    common_attn_metadata=cad, draft_index=0)
                for ln in group.layer_names:
                    per_layer_meta[ln] = meta
        if not per_layer_meta:
            raise RuntimeError("no attention metadata for the draft step")
        _hook_ok = False
        for meta in per_layer_meta.values():
            wrapper = getattr(getattr(meta, "prefill", None), "wrapper", None)
            if wrapper is not None and hasattr(wrapper, "_sm75_kv_indptr_gpu"):
                _hook_ok = True
                break
        if not _hook_ok:
            raise RuntimeError(
                "draft step did not take the SM75 bypass path; the captured "
                "kernels would replay against stale page metadata"
            )
        if pre_L is not None:
            # The append wrote the same blocks the query slots live in, so none
            # of them are scratch any more: every block allocated here holds
            # committed context or the next append's scratch rows.
            pre_blocks = len(blocks)
        return per_layer_meta, pre_blocks

    @torch.inference_mode()
    def _capture_draft_graph(self) -> bool:
        """Capture one draft step (B=1) into a CUDA graph."""
        from vllm.forward_context import set_forward_context

        B, Q = self._graph_shape()
        K    = self.num_speculative_tokens
        H    = self.draft_model_config.get_hidden_size()
        seq_id = "__cudagraph__"

        self._init_graph_buffers()
        self.handle_prefill(
            seq_id,
            torch.zeros(self.block_size, H, dtype=self.dtype, device=self.device),
        )
        try:
            side = torch.cuda.Stream(device=self.device)
            side.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(side):
                for _ in range(3):
                    plm, pre_blocks = self._plan_graph_step(seq_id, 0, pre_L=Q)
                    with set_forward_context(plm, self.vllm_config, num_tokens=Q):
                        self._graph_append()
                        hidden = self.model(input_ids=self._g_input_ids,
                                            positions=self._g_positions)
                    self._select_draft_tokens(hidden, self._g_bonus, B, K,
                                              out=self._g_out)
                    self._free_query_blocks(seq_id, pre_blocks)
            torch.cuda.current_stream(self.device).wait_stream(side)
            torch.cuda.synchronize(self.device)

            plm, pre_blocks = self._plan_graph_step(seq_id, 0, pre_L=Q)
            self._cuda_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._cuda_graph):
                with set_forward_context(plm, self.vllm_config, num_tokens=Q):
                    self._graph_append()
                    hidden = self.model(input_ids=self._g_input_ids,
                                        positions=self._g_positions)
                self._select_draft_tokens(hidden, self._g_bonus, B, K,
                                          out=self._g_out)
            self._free_query_blocks(seq_id, pre_blocks)
            torch.cuda.synchronize(self.device)
            logger.info("Draft step CUDA graph captured (B=%d, Q=%d).", B, Q)
            return True
        except Exception as exc:
            logger.warning("Draft CUDA graph capture failed, staying eager: %r",
                           exc)
            self._cuda_graph = None
            return False
        finally:
            self.handle_free(seq_id)

    @torch.inference_mode()
    def _run_draft_forward_graph(self, seq_id: str, bonus_ids,
                                 pre_L: int | None = None) -> torch.Tensor:
        """Replay the captured step after refreshing its inputs."""
        bonus_id = int(bonus_ids[0]) if torch.is_tensor(bonus_ids) else int(bonus_ids)
        _plm, pre_blocks = self._plan_graph_step(seq_id, bonus_id, pre_L)
        self._cuda_graph.replay()
        self._free_query_blocks(seq_id, pre_blocks)
        return self._g_out

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

        _seq = 0
        try:
            header = parse_header(header_frame)
            t      = header["t"]
            _seq   = int(header.get("seq", 0))

            if t == MSG_PING:
                resp_h, resp_p = build_ack(_seq)

            elif t == MSG_DECODE:
                (hs, qsl, num_rejected, temps, sds,
                 bonus_ids) = parse_decode_payload(header, payload_frame)
                draft_tokens = runner.handle_decode(
                    header["seq_ids"], hs, qsl, num_rejected, temps, sds,
                    bonus_ids, bool(header.get("aux", False)),
                )
                resp_h, resp_p = build_draft_response(draft_tokens, _seq)

            elif t == MSG_FREE:
                runner.handle_free(header["seq_id"])
                resp_h, resp_p = build_ack(_seq)

            else:
                resp_h, resp_p = build_error(f"unknown message type {t}", _seq)

        except Exception as exc:
            logger.exception("Error handling message type %s", header.get("t"))
            resp_h, resp_p = build_error(str(exc), _seq)

        sock.send_multipart([identity, b"", resp_h, resp_p])

    sock.close()
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    ctx.destroy()
    logger.info("Draft server stopped.")


def _setup_logging() -> None:
    """This process is spawned by the engine worker, which never configures the
    root logger for it: without a handler every INFO message is dropped and
    warnings fall through to logging.lastResort unformatted."""
    import logging
    import sys
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def main() -> None:
    _setup_logging()
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
