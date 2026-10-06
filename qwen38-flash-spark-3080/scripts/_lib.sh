#!/usr/bin/env bash
# Shared environment resolution for every role script.
# Source it: . "$(dirname "$0")/_lib.sh"
#
# Contract: this file only READS the environment and the local filesystem.
# It never contacts the peer, never writes outside $RUN_DIR and $LOG_DIR.
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- paths (all overridable; see env.example) ---------------------------------
# The source tree is the bundle's own src/ unless the operator points at an
# out-of-tree checkout (e.g. the peer already had a tree before this bundle).
STRATA_TREE="${STRATA_TREE:-$BUNDLE_DIR/src}"
STRATA_LLAMACPP_DIR="${STRATA_LLAMACPP_DIR:-}"
STRATA_SGL_KERNEL_INCLUDE="${STRATA_SGL_KERNEL_INCLUDE:-}"
STRATA_TVM_FFI_INCLUDE="${STRATA_TVM_FFI_INCLUDE:-}"
STRATA_TVM_FFI_LIB="${STRATA_TVM_FFI_LIB:-}"

STRATA_CUDA_ARCH="${STRATA_CUDA_ARCH:-}"
STRATA_BUILD_DIR="${STRATA_BUILD_DIR:-$STRATA_TREE/build-nvfp4}"
STRATA_JOBS="${STRATA_JOBS:-16}"

# ---- assets -------------------------------------------------------------------
STRATA_NVFP4_CKPT="${STRATA_NVFP4_CKPT:-}"
STRATA_Q2_GGUF_00001="${STRATA_Q2_GGUF_00001:-}"
STRATA_Q2_GGUF_00002="${STRATA_Q2_GGUF_00002:-}"
STRATA_DENSE_GGUF="${STRATA_DENSE_GGUF:-}"
STRATA_PLE_FP8="${STRATA_PLE_FP8:-}"
STRATA_PLE_FP8_SCALE="${STRATA_PLE_FP8_SCALE:-0.00019931793212890625}"
STRATA_HOT_PACK="${STRATA_HOT_PACK:-}"
STRATA_COLD_PACK="${STRATA_COLD_PACK:-}"
STRATA_PROFILE="${STRATA_PROFILE:-$BUNDLE_DIR/assets/expert-profile-v4.bin}"
STRATA_TOKENIZER="${STRATA_TOKENIZER:-}"

# ---- frozen C1 / FP8-KV operating point --------------------------------------
# These four values are the frozen, verified configuration. Changing them is a
# different operating point and must be re-accepted before it is called stable.
STRATA_EXPERT_CACHE="${STRATA_EXPERT_CACHE:-4608}"
STRATA_MAX_CONTEXT="${STRATA_MAX_CONTEXT:-16384}"
STRATA_KV="${STRATA_KV:-fp8}"
STRATA_PREFILL="${STRATA_PREFILL:-auto}"
STRATA_RESIDENT_PREFILL="${STRATA_RESIDENT_PREFILL:-1}"   # STRATA_PF_NVFP4_RESIDENT

# ---- RDMA / host topology -----------------------------------------------------
# STRATA_PEER_ADDR: the peer's address on the direct data link (Spark side for
# the 3080 engine; the 3080 side for the Spark worker).
STRATA_PEER_ADDR="${STRATA_PEER_ADDR:-}"
STRATA_PEER_PORT="${STRATA_PEER_PORT:-39580}"
STRATA_RDMA_SLOTS="${STRATA_RDMA_SLOTS:-4}"
STRATA_RDMA_SLOT_BYTES="${STRATA_RDMA_SLOT_BYTES:-4194304}"
STRATA_RDMA_GID="${STRATA_RDMA_GID:-}"
# Empty STRATA_HCA means "leave --remote-rdma-device empty" = first verbs device,
# which is the rename-proof form. Only set it if the autodetect picks wrongly.
STRATA_HCA="${STRATA_HCA:-}"

# ---- process bookkeeping ------------------------------------------------------
STRATA_RUN_DIR="${STRATA_RUN_DIR:-/run/strata}"
STRATA_LOG_DIR="${STRATA_LOG_DIR:-$STRATA_RUN_DIR/log}"
STRATA_HTTP_PORT="${STRATA_HTTP_PORT:-8100}"          # 3080 engine HTTP front
STRATA_SPARK_HTTP_PORT="${STRATA_SPARK_HTTP_PORT:-8200}"
STRATA_ROUTER_PORT="${STRATA_ROUTER_PORT:-8400}"
STRATA_PYTHON="${STRATA_PYTHON:-python3}"

die() { echo "strata-deploy: $*" >&2; exit 2; }
note() { echo "[strata-deploy] $*"; }

require_env() {
  local missing=0 name
  for name in "$@"; do
    if [ -z "${!name:-}" ]; then echo "strata-deploy: missing required env $name" >&2; missing=1; fi
  done
  [ "$missing" -eq 0 ] || die "see env.example"
}

require_file() {
  local p="$1" label="${2:-$1}"
  [ -f "$p" ] || die "missing $label: $p"
}

hca_detect() {
  if [ -n "$STRATA_HCA" ]; then printf '%s' "$STRATA_HCA"; return 0; fi
  ls /sys/class/infiniband 2>/dev/null | head -1
}

hca_args() {
  local hca; hca="$(hca_detect)"
  if [ -n "$hca" ]; then printf '%s' "--remote-rdma-device $hca"; fi
}

prepare_run_dir() {
  mkdir -p "$STRATA_RUN_DIR" "$STRATA_LOG_DIR"
}

pidfile() { printf '%s/%s.pid' "$STRATA_RUN_DIR" "$1"; }
logfile() { printf '%s/%s.log' "$STRATA_LOG_DIR" "$1"; }

# start_bg <name> <logname> <command...>
# Detaches the child (setsid), records its PID, never blocks the caller.
start_bg() {
  local name="$1" logname="$2"; shift 2
  local log pf
  log="$(logfile "$logname")"; pf="$(pidfile "$name")"
  if alive "$name"; then die "$name already running (pid $(cat "$pf"))"; fi
  : >"$log"
  setsid "$@" </dev/null >>"$log" 2>&1 &
  echo $! >"$pf"
  note "$name started pid=$(cat "$pf") log=$log"
}

alive() {
  local pf; pf="$(pidfile "$1")"
  [ -f "$pf" ] || return 1
  local pid; pid="$(cat "$pf" 2>/dev/null || true)"
  [ -n "$pid" ] || return 1
  kill -0 "$pid" 2>/dev/null
}

# wait_port <port> <timeout_s>: readiness is LISTEN, never a log line.
wait_port() {
  local port="$1" timeout="${2:-120}" i=0
  while [ "$i" -lt "$timeout" ]; do
    if command -v ss >/dev/null 2>&1; then
      ss -tln 2>/dev/null | grep -q ":$port " && return 0
    elif command -v netstat >/dev/null 2>&1; then
      netstat -tln 2>/dev/null | grep -q ":$port " && return 0
    else
      (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null && return 0
    fi
    i=$((i + 1)); sleep 1
  done
  return 1
}

wait_http() {
  local url="$1" timeout="${2:-120}" i=0
  while [ "$i" -lt "$timeout" ]; do
    if curl -fsS -o /dev/null --max-time 4 "$url" 2>/dev/null; then return 0; fi
    i=$((i + 1)); sleep 2
  done
  return 1
}
