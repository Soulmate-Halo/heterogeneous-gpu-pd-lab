#!/usr/bin/env bash
set -euo pipefail
# Run once on each host after replacing placeholders.
IMAGE="${IMAGE:-ghcr.io/soulmate-halo/heterogeneous-gpu-pd-lab/qwen38-flash-spark-3080:latest}"
ROLE="${1:?usage: $0 worker|engine}"
MODELS="${MODELS:-/srv/qwen38-models}"
case "$ROLE" in
  worker) exec docker run --rm --name qwen38-spark-worker --network host --gpus all --device /dev/infiniband --cap-add IPC_LOCK -v "$MODELS:/models:ro" -e ROLE=worker "$IMAGE" ;;
  engine) exec docker run --rm --name qwen38-rtx3080-engine --network host --gpus all --device /dev/infiniband --cap-add IPC_LOCK -v "$MODELS:/models:ro" -e ROLE=engine -e STRATA_RDMA_WORKER_HOST="${STRATA_RDMA_WORKER_HOST:?set Spark RDMA hostname}" "$IMAGE" ;;
  *) echo 'role must be worker or engine' >&2; exit 2 ;;
esac
