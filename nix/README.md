# Nix Package

Language: English | [简体中文](README.zh-CN.md)

This directory contains a standalone Nix flake that builds vLLM and all its
dependencies from source using nixpkgs. It targets SM70/SM75 GPUs (SM70: V100, Titan V; SM75: RTX 2080 Ti)
with CUDA 13.0 and PyTorch 2.13.

## Status

**Experimental.** Built and smoke-tested on a single machine (dual RTX 2080 Ti
22 GiB NVLink, NixOS 26.05, driver 595.84). No CI. No automated tests. The
build takes roughly 30–60 minutes on first run depending on available cores.

Do not use this as a drop-in replacement for the standard venv setup in a
production environment without further validation.

## What it builds

```
nix build .#vllm        # vLLM + all Python deps as a Nix package
nix build .#torch-cu130 # PyTorch 2.13+cu130 standalone
nix build .#flash-qla   # SM70/SM75 GDN kernel (see nix/gdn_forward.cu)
```

All packages are exposed under `packages.x86_64-linux.*`. Run `nix flake show`
for the full list.

## Requirements

- NixOS or nix with `allowUnfree = true` and a CUDA-capable driver installed
- NVIDIA driver >= 565 (CUDA 13.0 requires driver >= 570 for full support;
  550+ works for SM75 inference)
- x86_64-linux only (wheel pins are architecture-specific)

## Running vLLM

The Nix package installs a `vllm` binary. The simplest invocation:

```bash
nix run .#vllm -- serve Qwen/Qwen3.8-27B-FP8 \
  --tensor-parallel-size 2 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.92
```

Or build first and keep the result:

```bash
nix build .#vllm
./result/bin/vllm serve <model> [args...]
```

## Using the reference profiles

The profiles in `profiles/` are `.env` presets for the `launcher.sh` script.
**The launcher currently expects a `.venv` virtualenv and will not work with the
Nix package directly.** To bridge them, create a wrapper that points the
launcher at the Nix-built Python:

```bash
NIX_VLLM=$(nix build .#vllm --no-link --print-out-paths)
mkdir -p .venv/bin
ln -sf $NIX_VLLM/bin/python .venv/bin/python
ln -sf $NIX_VLLM/bin/vllm .venv/bin/vllm
RUNTIME_ROOT=$PWD ./launcher.sh
```

This is a workaround, not a supported path. The environment variables
(`TRITON_CACHE_DIR`, `FLASHINFER_WORKSPACE_BASE`, `LD_LIBRARY_PATH`, etc.) that
the launcher normally sets via `.env` still need to be exported before calling
the binary. See `modules/system/services/vllm.nix` in the NixOS config for the
full set of required environment variables.

## What the patches do

`vllm.nix` applies a large `postPatch` block. Each patch addresses a specific
incompatibility:

| Patch | Reason |
|---|---|
| `CMAKE_CUDA_ARCHITECTURES 75` | Force SM75 target; upstream defaults to all arches |
| `find_package(CUDAToolkit)` | Nix sandbox does not provide system CUDA; must use our `cudaHome` |
| `CMAKE_CUDA_FLAGS` hardcode | Avoids torch introspection of nvcc flags inside the Nix sandbox |
| `GPU_FLAGS` hardcode in `cmake/utils.cmake` | Same — `run_python` calls fail in sandbox |
| `_VLLM_TORCH_GOMP_PATH` clear | gomp lookup fails in sandbox; not needed for SM75 |
| `has_cuda` check bypass | `torch.version.cuda` is `None` at Nix build time |
| `setuptools-rust` removal | We stub `build_rust.py`; Rust extensions are not needed for CUDA |
| `requirements/*.txt` stripping | Remove packages Nix provides directly to avoid pip trying to fetch them |
| `lark` version unpin | nixpkgs ships a newer lark |
| `minimax_m3` stubs | MiniMax-M3 MSA ops import unconditionally; stubs prevent crash on non-M3 GPUs |
| `CUTLASS_FP8_SUPPORTED` guard | SM75 does not support FP8 CUTLASS; the probe raises at import time |
| `triton.next_power_of_2` rewrite | Removed in triton 3.x; replaced with equivalent bit-length expression |
| `triton_utils/importing.py` | Suppresses multi-driver assertion that fires in systemd environment |
| `QuarkConfig` import guard | Quark quantization not packaged; guard prevents hard import failure |
| `torch.version.cuda or "13.0"` | `torch.version.cuda` can be `None` in the Nix-built wheel |

## The SM70/SM75 GDN kernel (`nix/gdn_forward.cu`)

The upstream `FlashQLA-SM70-SM75` repo does not include a kernel that compiles
cleanly under Nix with CUDA 13.0. `nix/gdn_forward.cu` is a hand-written CUDA
kernel implementing the GDN (Gated Delta Networks) forward pass for SM70/SM75.
It is compiled at Nix build time via `torch.utils.cpp_extension.load` and
installed as `flash_qla_legacy_gdn.so` inside the `flash-qla` package.

This kernel is activated when `gdn_prefill_backend=flashqla_legacy` is set in
a profile.

`nix/sm_legacy.py` is the Python shim that loads the `.so` at runtime.

## Known limitations

- Only `x86_64-linux` is supported (all wheels are architecture-pinned)
- Only SM70/SM75 (Volta/Turing); other CUDA architectures are not tested
- `launcher.sh` does not work with the Nix package without the `.venv` bridge
  described above
- `doCheck = false` on all packages — nixpkgs tests are not run
- The `mcp` package has a flaky network test in nixpkgs 26.05; it is
  overridden with `doCheck = false`
- Wheel versions are pinned to what was validated on the reference machine;
  updating any of them may require re-validating the full build
- `torch-stable-api-shim` and `torch-nvidia-shim` from the NixOS config are
  not included here — they were required for the systemd service setup but are
  not needed for the standalone package

## Updating

When upstream releases a new version:

1. Update the `src` hash in `flake.nix` (`vllm` derivation)
2. Update `version` and `SETUPTOOLS_SCM_PRETEND_VERSION`
3. Check `requirements/common.txt` and `requirements/cuda.txt` for new deps
4. Re-run `nix build .#vllm` and fix any new patch failures
5. Update the wheel pins in `vllm-extras.nix` if needed
