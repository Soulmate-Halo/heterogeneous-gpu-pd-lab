#!/usr/bin/env bash
set -euo pipefail
# Run once on each host after replacing placeholders.
IMAGE="${IMAGE:-ghcr.io/soulmate-halo/heterogeneous-gpu-pd-lab/qwen38-flash-spark-3080:latest}"
ROLE="${1:?usage: $0 spark|3080}"
MODELS="${MODELS:-/srv/qwen38-models}"
IMAGE="${IMAGE:-ghcr.io/soulmate-halo/heterogeneous-gpu-pd-lab/qwen38-flash-spark-3080:latest}"
COMMON=(--rm --network host --gpus all --device /dev/infiniband --cap-add IPC_LOCK --init -v "$MODELS:/models:ro" -v "${CACHE_DIR:-/srv/qwen38-cache}:/cache" -v "${STATE_DIR:-/srv/qwen38-state}:/state")
case "$ROLE" in
  spark) exec docker run --name qwen38-spark "${COMMON[@]}" -e STRATA_ROLE=spark "$IMAGE" spark ;;
  3080) exec docker run --name qwen38-rtx3080 "${COMMON[@]}" -e STRATA_ROLE=3080 -e STRATA_REMOTE_HOST="${STRATA_REMOTE_HOST:?set Spark RDMA hostname}" "$IMAGE" 3080 ;;
  *) echo 'role must be spark or 3080' >&2; exit 2 ;;
esac
