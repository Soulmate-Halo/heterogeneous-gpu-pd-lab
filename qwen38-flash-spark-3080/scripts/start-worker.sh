#!/usr/bin/env bash
# start-worker.sh - start the NVFP4 cold-expert worker on DGX Spark (GB10).
#
# The worker is the Spark half of the cross-machine tier: it owns the 63.28 GiB
# NVFP4 cold-expert arena and serves prefill/decode windows over the flag ring.
#
# Readiness contract (both matter):
#   1. it must LISTEN on STRATA_PEER_PORT before the 3080 engine starts, and
#   2. the arena line must have appeared, otherwise the first window is served
#      from an unfilled arena.
# The worker is single-session: when the engine exits, the worker exits.
#
#   ./scripts/start-worker.sh [--no-ple]
set -euo pipefail
. "$(dirname "$0")/_lib.sh"

WITH_PLE=1
[ "${1:-}" = "--no-ple" ] && WITH_PLE=0

require_env STRATA_COLD_PACK
prepare_run_dir
[ -x "$STRATA_BUILD_DIR/strata-cold-expert-worker" ] || \
  die "worker not built: run scripts/build-spark.sh first"

HCA="$(hca_detect)"
ARGS=(
  --format nvfp4
  --pack "$STRATA_COLD_PACK"
  --port "$STRATA_PEER_PORT"
  --slots "$STRATA_RDMA_SLOTS"
  --slot-bytes "$STRATA_RDMA_SLOT_BYTES"
)
[ -n "$HCA" ] && ARGS+=(--device "$HCA")
[ -n "$STRATA_RDMA_GID" ] && ARGS+=(--gid "$STRATA_RDMA_GID")
# The FP8 PLE table is optional here: with --ple-fp8 the worker pins 51.2 GB and
# registers it for one-sided reads; without it the table-less form serves fine
# (the engine can read the same table from its own local copy).
if [ "$WITH_PLE" -eq 1 ]; then
  require_env STRATA_PLE_FP8
  require_file "$STRATA_PLE_FP8" "PLE FP8 table"
  ARGS+=(--ple-fp8 "$STRATA_PLE_FP8" --ple-fp8-scale "$STRATA_PLE_FP8_SCALE")
fi

# GB10 has unified memory: the page cache counts against CUDA's free memory and
# a stale cache makes the 68 GB arena allocation fail. Drop it first.
if [ -w /proc/sys/vm/drop_caches ]; then
  sync; echo 3 >/proc/sys/vm/drop_caches || true
elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
  sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' || true
else
  note "WARN: cannot drop page cache (need root); the arena fill may fail on a dirty cache"
fi

if [ "${STRATA_FOREGROUND:-0}" = "1" ]; then
  exec "$STRATA_BUILD_DIR/strata-cold-expert-worker" "${ARGS[@]}"
fi
start_bg worker worker-nvfp4 "$STRATA_BUILD_DIR/strata-cold-expert-worker" "${ARGS[@]}"

WLOG="$(logfile worker-nvfp4)"
note "waiting for the arena line and :$STRATA_PEER_PORT ..."
deadline=$((SECONDS + ${STRATA_WORKER_TIMEOUT:-900}))
arena_seen=0
while [ "$SECONDS" -lt "$deadline" ]; do
  if grep -qE 'cold arena|arena .*GiB' "$WLOG" 2>/dev/null; then arena_seen=1; fi
  if [ "$arena_seen" -eq 1 ] && wait_port "$STRATA_PEER_PORT" 2; then
    # LISTEN alone raced the engine's first connect in the field; let the
    # accept path settle before declaring ready.
    sleep "${STRATA_WORKER_SETTLE:-20}"
    alive worker || die "worker died after binding :$STRATA_PEER_PORT (see $WLOG)"
    note "worker READY (arena + :$STRATA_PEER_PORT, pid $(cat "$(pidfile worker)"))"
    exit 0
  fi
  alive worker || { tail -20 "$WLOG" >&2; die "worker exited early (see $WLOG)"; }
  sleep 5
done
tail -20 "$WLOG" >&2
die "worker not ready within ${STRATA_WORKER_TIMEOUT:-900}s"
