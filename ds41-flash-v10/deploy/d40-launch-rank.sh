#!/usr/bin/env bash
# DS4.1 Flash V10 — D-side Decode launch (SGLang TP4), one rank per Spark node
#
# Run once per Spark host with D_NODE_RANK=0..3 (never fake four ranks on one host).
#
# Placeholders to replace (see deploy.conf.example / DEPLOYMENT.zh-CN.md 4.3):
#   <SPARK0_ROCE_IP>  this node's RoCE address (bind/advertise)
#   <ROCE_SUBNET>     RoCE subnet for NCCL_IB_ADDR_RANGE, written as a CIDR block
set -euo pipefail

D_IMAGE="${D_IMAGE:-dsv41-sglang:mhcbf16parts}"
D_NODE_RANK="${D_NODE_RANK:-0}"
D_MASTER_ADDR="${D_MASTER_ADDR:-<SPARK0_ROCE_IP>}"
SPARK0_ROCE_IP="${SPARK0_ROCE_IP:-<SPARK0_ROCE_IP>}"
ROCE_SUBNET="${ROCE_SUBNET:-<ROCE_SUBNET>}"
WEIGHTS_DIR="${WEIGHTS_DIR:-/home/spark-user/models/DeepSeek-V4.1-Flash}"
ENGRAM_DIR="${ENGRAM_DIR:-/home/spark-user/dsv41-engram}"

exec docker run -d --name dsv41-d40-sgl --restart no --network host \
  --device /dev/infiniband:/dev/infiniband \
  --cap-add CAP_IPC_LOCK \
  --ulimit memlock=-1:-1 \
  -v "${WEIGHTS_DIR}:/models/DeepSeek-V4.1-Flash" \
  -v "${ENGRAM_DIR}:/engram" \
  -e UCX_NET_DEVICES=rocep1s0f0:1 \
  -e UCX_IB_GID_INDEX=2 \
  -e UCX_TLS=ib,cuda_copy,cuda_ipc,sm,self \
  -e NCCL_IB_HCA=rocep1s0f0:1,roceP2p1s0f0:1 \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 \
  -e NCCL_IB_ADDR_RANGE="${ROCE_SUBNET}" \
  -e D_NODE_RANK="${D_NODE_RANK}" \
  -e D_MASTER_ADDR="${D_MASTER_ADDR}" \
  "${D_IMAGE}" \
  python3 -m sglang.launch_server \
    --model-path /models/DeepSeek-V4.1-Flash \
    --tp-size 4 --ep-size 4 \
    --host "${SPARK0_ROCE_IP}" --port 8200 \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend nixl

# restart=no: after a host reboot the D40 container must be started manually.
