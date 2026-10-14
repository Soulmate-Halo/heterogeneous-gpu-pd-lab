#!/usr/bin/env bash
# DS4.1 Flash V10 — P-side Prefill launch (vLLM, 1x RTX 6000D, TP1, 21/40 layers)
#
# Placeholders to replace (see deploy.conf.example / DEPLOYMENT.zh-CN.md 3.4):
#   <P_ROCE_IP>     RoCE address advertised on the NIXL side channel
#   <SPARK0_ROCE_IP>..<SPARK3_ROCE_IP>  Engram remote shard hosts
# Every other value (TP1, 0.92, KV bytes, seq/batch sizes, cudagraph mode,
# NixlConnector kv_producer) is taken verbatim from the deployment record.
set -euo pipefail

P_IMAGE="${P_IMAGE:-dsv41-p5:vllm-exl3-sm120-rdma}"
WEIGHTS_DIR="${WEIGHTS_DIR:-/mnt/dsv41-tp6/fp4h96}"
WEIGHTS_CONTAINER_PATH="${WEIGHTS_CONTAINER_PATH:-/models/DeepSeek-V4.1-Flash}"
PATCH_DIR="${PATCH_DIR:-/mnt/dsv41-tp6/dsv41-pp3/patch-p21-exl3}"
P_ROCE_IP="${P_ROCE_IP:-<P_ROCE_IP>}"
SPARK0_ROCE_IP="${SPARK0_ROCE_IP:-<SPARK0_ROCE_IP>}"
SPARK1_ROCE_IP="${SPARK1_ROCE_IP:-<SPARK1_ROCE_IP>}"
SPARK2_ROCE_IP="${SPARK2_ROCE_IP:-<SPARK2_ROCE_IP>}"
SPARK3_ROCE_IP="${SPARK3_ROCE_IP:-<SPARK3_ROCE_IP>}"

# Official "A+" engram layout: layer 1 -> 9612, layer 14 -> 9611, four shards each.
DSV41_ENGRAM_REMOTE="1=${SPARK0_ROCE_IP}:9612@0:96001542,1=${SPARK1_ROCE_IP}:9612@96001542:192003084,1=${SPARK2_ROCE_IP}:9612@192003084:288004626,1=${SPARK3_ROCE_IP}:9612@288004626:384006168,14=${SPARK0_ROCE_IP}:9611@0:96004171,14=${SPARK1_ROCE_IP}:9611@96004171:192008342,14=${SPARK2_ROCE_IP}:9611@192008342:288012513,14=${SPARK3_ROCE_IP}:9611@288012513:384016682"

exec docker run -d --name dsv41-p21-solo --restart no --network host \
  --device /dev/infiniband:/dev/infiniband \
  --cap-add CAP_IPC_LOCK \
  --ulimit memlock=-1:-1 \
  -v "${WEIGHTS_DIR}:${WEIGHTS_CONTAINER_PATH}" \
  -v "${PATCH_DIR}:/mnt/patch-p21-exl3" \
  -e UCX_NET_DEVICES=mlx4_0:1 \
  -e UCX_IB_GID_INDEX=12 \
  -e UCX_TLS=ib,cuda_copy,cuda_ipc,sm,self \
  -e VLLM_NIXL_SIDE_CHANNEL_HOST="${P_ROCE_IP}" \
  -e VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
  -e XY_NIXL_LAYER_MODE=1 \
  -e XY_SWA_EXTRA_RETAINED=2816 \
  -e XY_KVME_LAYERS=21 \
  -e NCCL_IB_DISABLE=1 \
  -e DSV41_ENGRAM_REMOTE="${DSV41_ENGRAM_REMOTE}" \
  -e DSV41_ENGRAM_DISK_THREADS=32 \
  -e DSV41_ENGRAM_DISK_CHUNK=16 \
  -e DSV41_ENGRAM_REMOTE_FALLBACK=0 \
  "${P_IMAGE}" \
  vllm serve "${WEIGHTS_CONTAINER_PATH}" --served-model-name deepseek-v4.1-flash \
    --tensor-parallel-size 1 --pipeline-parallel-size 1 \
    --gpu-memory-utilization 0.92 --kv-cache-memory-bytes 10737418240 \
    --max-model-len 131072 --max-num-seqs 12 --max-num-batched-tokens 16384 \
    --engram-config '{"cpu_offload": false}' \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
    --kv-cache-dtype auto --block-size 128 --enable-cumem-allocator \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'

# NOTE: the P stage is a 21/40-layer Prefill-only slice; text it emits on its
# own is intentionally garbage. It must run together with D + the proxy.
