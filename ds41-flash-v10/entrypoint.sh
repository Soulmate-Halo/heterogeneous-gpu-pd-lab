#!/usr/bin/env bash
# DS4.1 Flash V10 container entrypoint.
#
# Commands:
#   show-config   print topology, per-component launch parameter summary and
#                 placeholder legend (also the default when no argument given)
#   verify        run verify_bundle.py --inside-image
#   shell         start an interactive bash
set -euo pipefail

BUNDLE="${DS41_V10_BUNDLE:-/opt/ds41-flash-v10}"

show_config() {
    cat <<'EOF'
DS4.1 Flash V10 (2026-10-14)
Architecture: PD separation with single-card Prefill acceleration
P stage: vLLM, 1 x RTX 6000D, 21/40-layer Prefill-only slice, TP1, agent port 8000
         gpu-memory-utilization 0.92, kv-cache-memory-bytes 10737418240,
         max-model-len 131072, max-num-seqs 12, max-num-batched-tokens 16384,
         cudagraph FULL_AND_PIECEWISE, NixlConnector kv_producer
         UCX_TLS=ib,cuda_copy,cuda_ipc,sm,self; UCX_IB_GID_INDEX=12; UCX_NET_DEVICES=mlx4_0:1
         VLLM_NIXL_SIDE_CHANNEL_PORT=5600; container needs --device /dev/infiniband,
         --cap-add CAP_IPC_LOCK, --ulimit memlock=-1:-1
D stage: SGLang, 4 x DGX Spark/GB10, TP4, API port 8200 (rank0)
         UCX_NET_DEVICES=rocep1s0f0:1; UCX_IB_GID_INDEX=2
         NCCL_IB_HCA=rocep1s0f0:1,roceP2p1s0f0:1; NCCL_SOCKET_IFNAME=enp1s0f0np0
Engram: layers 1 and 14 served remotely by 8 shard servers (2 per Spark)
        9611 = layer 14, 9612 = layer 1; --backend disk --lib libxyrow.so
        --io-threads 128 --workers 16
Proxy:  single external port 5701; XY_MAX_PROMPT_CHARS=2000000
Transport: vLLM -> SGLang NIXL KV handoff over RoCE v1 IPv4

Placeholders used across deploy/ (values redacted for public release):
  <P_IP> <P_ROCE_IP> <SPARK0_ROCE_IP> <SPARK1_ROCE_IP> <SPARK2_ROCE_IP>
  <SPARK3_ROCE_IP> <SPARK0-3_ROCE_IP> <SPARKx_ROCE_IP> <ROCE_SUBNET>
  <SPARK_OFFICE_IP> <OFFICE_NET> <LOCAL_WORK_DIR>

This image does NOT contain the 130 GB vendor role-image package or the
232 GB DS4.1 FP4 weights; mount or load those on the hosts.
EOF
}

verify() {
    if [ -f "${BUNDLE}/verify_bundle.py" ]; then
        exec python3 "${BUNDLE}/verify_bundle.py" --inside-image
    fi
    # Fallback: report SHA256 of every file under deploy/.
    find "${BUNDLE}/deploy" -type f -print0 | sort -z | xargs -0 sha256sum
}

case "${1:-show-config}" in
    show-config) show_config ;;
    verify) verify ;;
    shell) exec bash ;;
    *) echo "usage: ds41-v10 {show-config|verify|shell}" >&2; exit 2 ;;
esac
