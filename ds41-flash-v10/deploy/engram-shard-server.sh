#!/usr/bin/env bash
# DS4.1 Flash V10 — Engram shard servers (8 total: 2 per Spark node)
#
# Args: $1 = layer (1 or 14), $2 = shard index K (0..3), $3 = bind IP.
# Run this twice per Spark node: (14, K, <SPARKx_ROCE_IP>) and (1, K, <SPARKx_ROCE_IP>).
#
# Placeholders to replace (see deploy.conf.example / DEPLOYMENT.zh-CN.md 7.3 step 2):
#   <SPARKx_ROCE_IP>  this node's RoCE address passed as the 3rd argument
set -euo pipefail

LAYER="${1:?usage: engram-shard-server.sh <1|14> <K 0..3> <bind-ip>}"
K="${2:?usage: engram-shard-server.sh <1|14> <K 0..3> <bind-ip>}"
BIND_IP="${3:-<SPARKx_ROCE_IP>}"

D_IMAGE="${D_IMAGE:-dsv41-sglang:mhcbf16parts}"
SLICES_DIR="${SLICES_DIR:-/var/tmp/dsv41-engram-l{1,14}-4of}"
SERVER_DIR="${SERVER_DIR:-/var/tmp/dsv41-engram-remote}"

if [ "${LAYER}" = "14" ]; then
  NAME=dsv41-engram-l14
  PORT=9611
else
  NAME=dsv41-engram-l1
  PORT=9612
fi

exec docker run -d --name "${NAME}" --restart unless-stopped --network host \
  --memory 8g --memory-swap 8g --oom-score-adj -500 \
  --ulimit nofile=65536:65536 \
  -v "${SLICES_DIR}:/slices:ro" \
  -v "${SERVER_DIR}:/srv" \
  --entrypoint python3 "${D_IMAGE}" \
  /srv/engram_gather_server.py \
    --shard "/slices/l${LAYER}-s${K}of4.safetensors" \
    --layer "${LAYER}" \
    --backend disk \
    --lib /srv/libxyrow.so \
    --io-threads 128 \
    --direct 0 \
    --bind "${BIND_IP}" \
    --port "${PORT}" \
    --workers 16 \
    --log-every 200

# Port layout: 9611 = layer 14, 9612 = layer 1 (one container per layer per node).
