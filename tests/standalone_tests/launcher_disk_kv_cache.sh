#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
# shellcheck source=../../launcher.sh
source "$ROOT/launcher.sh"

KV_DISK_CACHE_DIR=/mnt/nvme/kv-cache
KV_DISK_CPU_BYTES=4294967296
ENABLE_PREFIX_CACHING=1
DISABLE_PREFIX_CACHING=0
validate_disk_kv_cache_config

config=$(disk_kv_transfer_config)
python3 - "$config" <<'PY'
import json
import sys

config = json.loads(sys.argv[1])
assert config["kv_connector"] == "OffloadingConnector"
assert config["kv_role"] == "kv_both"
extra = config["kv_connector_extra_config"]
assert extra["cpu_bytes_to_use"] == 4294967296
assert extra["spec_name"] == "TieringOffloadingSpec"
assert extra["secondary_tiers"] == [
    {"type": "fs", "root_dir": "/mnt/nvme/kv-cache"}
]
PY

MODEL_DIR=/mnt/models/test
MODEL_FAMILY=qwen35
SERVED_NAME=test
MODE=normal
PORT=8000
TP_SIZE=4
PP_SIZE=1
GPU_UTIL=0.75
MAX_MODEL_LEN=4096
MAX_NUM_SEQS=1
MAX_BATCHED_TOKENS=1024
build_args 127.0.0.1
args=$(printf '%s\n' "${VLLM_ARGS[@]}")
grep -Fxq -- '--kv-transfer-config' <<< "$args"
grep -Fxq -- "$config" <<< "$args"

KV_DISK_CPU_BYTES=invalid
if validate_disk_kv_cache_config 2>/dev/null; then
  echo "invalid CPU staging size unexpectedly accepted" >&2
  exit 1
fi
KV_DISK_CPU_BYTES=4294967296

DISABLE_PREFIX_CACHING=1
if validate_disk_kv_cache_config 2>/dev/null; then
  echo "disabled prefix cache unexpectedly accepted" >&2
  exit 1
fi
DISABLE_PREFIX_CACHING=0
KV_DISK_CACHE_DIR=relative/path
if validate_disk_kv_cache_config 2>/dev/null; then
  echo "relative cache path unexpectedly accepted" >&2
  exit 1
fi
KV_DISK_CACHE_DIR=/mnt/nvme/kv-cache
PYTHONHASHSEED=random
if validate_disk_kv_cache_config 2>/dev/null; then
  echo "unstable hash seed unexpectedly accepted" >&2
  exit 1
fi
for PYTHONHASHSEED in abc 4294967296; do
  if validate_disk_kv_cache_config 2>/dev/null; then
    echo "invalid hash seed unexpectedly accepted: $PYTHONHASHSEED" >&2
    exit 1
  fi
done
PYTHONHASHSEED=4294967295
validate_disk_kv_cache_config
unset PYTHONHASHSEED

save_manager_state() { :; }
menu_select() { printf 'enabled\n'; }
prompt_default() { printf '%s\n' "$2"; }
KV_DISK_CACHE_DIR=/mnt/nvme/kv-cache
KV_DISK_CPU_BYTES=4294967296
edit_disk_kv_cache_menu
[[ "$KV_DISK_CACHE_DIR" == /mnt/nvme/kv-cache ]]

prompt_default() {
  case "$1" in
    "SSD cache directory") printf 'relative/path\n' ;;
    *) printf '%s\n' "$2" ;;
  esac
}
if edit_disk_kv_cache_menu 2>/dev/null; then
  echo "invalid menu path unexpectedly accepted" >&2
  exit 1
fi
[[ "$KV_DISK_CACHE_DIR" == /mnt/nvme/kv-cache ]]

menu_select() { printf 'disabled\n'; }
edit_prefix_cache_menu
[[ -z "$KV_DISK_CACHE_DIR" && -z "$KV_DISK_CPU_BYTES" ]]

echo "Disk KV cache launcher checks passed"
