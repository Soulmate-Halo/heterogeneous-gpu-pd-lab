#!/usr/bin/env bash
# DS4.1 Flash V10 — PD proxy launch (single external port 5701)
#
# Placeholders to replace (see deploy.conf.example):
#   <P_IP>           office-network host of the proxy
#   <P_ROCE_IP>      RoCE address of the Prefill host
#   <SPARK0_ROCE_IP> RoCE address of Spark rank 0 (decode API host)
set -euo pipefail

PROXY_IMAGE="${PROXY_IMAGE:-vllm-dsv41-x86:overlay6}"
P_IP="${P_IP:-<P_IP>}"
P_ROCE_IP="${P_ROCE_IP:-<P_ROCE_IP>}"
SPARK0_ROCE_IP="${SPARK0_ROCE_IP:-<SPARK0_ROCE_IP>}"

# The proxy exposes exactly one port to clients. 8000/8200/5600 stay internal.
exec docker run -d --name xy-pd-proxy-v8-5701 --restart unless-stopped --network host \
  -e PREFILL_URL="http://${P_ROCE_IP}:8000" \
  -e DECODE_URL="http://${SPARK0_ROCE_IP}:8200" \
  -e XY_MAX_PROMPT_CHARS=2000000 \
  -e XY_PROXY_PORT=5701 \
  "${PROXY_IMAGE}" \
  python3 /opt/xy-proxy-v8/xy_pd_proxy_v8cutover_5701.py

# Client entry point: http://<P_IP>:5701  (only 5701 is exposed externally).
