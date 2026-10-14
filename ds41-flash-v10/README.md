# DeepSeek-V4.1-Flash V10 Docker bundle

This directory publishes the **DS4.1 Flash V10 deployment layer**: the configs,
launch templates and verification scripts for a layout where **one RTX 6000D
runs all Prefill** and **four DGX Spark nodes run Decode**.

The image is a small configuration image. It contains **only** the deployment
configs, launch templates and verification scripts. It does **not** contain the
130 GB vendor role-image package (the four role images) or the 232 GB DS4.1 FP4
weights. Those stay on the target hosts and are mounted or loaded by the role
launchers, which keeps the image redistributable.

## Architecture

```text
client --> http://<P_IP>:5701 (PD proxy, single external port)
  P stage: 1 x RTX 6000D (vLLM, 21/40-layer Prefill-only slice, TP1)
  D stage: 4 x DGX Spark/GB10 (SGLang, TP4, one four-node group)
  KV: vLLM -> SGLang NIXL handoff over RoCE v1 IPv4 (direct, no TCP fallback)
  Engram: layers 1 and 14 served remotely by 8 shard servers on the Sparks
```

The 6000D holds a 21/40-layer Prefill-only slice (`kv_source_layer_ids=[2,8,14,20]`
covers every KV source in layers 0–20). Its own text output is intentionally
malformed; it must always run together with D and the proxy. Engram tables for
layers 1 and 14 are 384,006,168 / 384,016,682 rows of 264 B and are served over
TCP by eight shard servers (two per Spark node). If the remote shard map is
left empty, P silently degrades to per-row reads from its single consumer NVMe
and Prefill drops from ~10 K tok/s to ~4 K tok/s.

## Prerequisites

- Vendor delivery package with the four role images
  (`dsv41-p5:vllm-exl3-sm120` + RDMA overlay, `dsv41-sglang:mhcbf16parts`,
  `vllm-dsv41-x86:overlay6`, `vllm-dsv41:overlay6`).
- DS4.1 Flash FP4 weights (232 GB) on the Spark nodes and the P host.
- RoCE network between P (ConnectX-3, `mlx4_0`) and the Sparks (ConnectX-7,
  `rocep1s0f0`); RoCE **v1 IPv4** GID indices are required (P=12, D=2).
- `vm.overcommit_memory = 1` on every host (see `deploy/90-dsv41-overcommit.conf`).

## Deploy in six steps

1. **Load and tag the role images.** `docker load -i` each vendor tarball; the
   OCI archives have no `RepoTags`, so compare config digests against the
   package README and `docker tag` each one.
2. **Enable overcommit.** Install `deploy/90-dsv41-overcommit.conf` as
   `/etc/sysctl.d/90-dsv41-overcommit.conf` and run `sysctl --system` on P and
   on each Spark.
3. **Cut engram shards and start 8 shard servers.** Run `deploy/engram-shard-server.sh`
   twice per Spark (`14`→9611, `1`→9612).
4. **Start the four D40 decode ranks.** One `deploy/d40-launch-rank.sh` per
   Spark node with `D_NODE_RANK=0..3`.
5. **Start P.** Run `deploy/p21-launch.sh` (cold start ~1.5–9 min); confirm the
   log shows `Engram ACTIVE read path = REMOTE` for both layers.
6. **Start the proxy.** Run `deploy/pd-proxy-launch.sh`; clients then use
   `http://<P_IP>:5701`.

## Verify

```bash
docker pull ghcr.io/soulmate-halo/heterogeneous-gpu-pd-lab/ds41-flash-v10:latest
docker run --rm ghcr.io/soulmate-halo/heterogeneous-gpu-pd-lab/ds41-flash-v10:latest show-config
docker run --rm ghcr.io/soulmate-halo/heterogeneous-gpu-pd-lab/ds41-flash-v10:latest verify
python ds41-flash-v10/verify_bundle.py   # source-tree self-check
curl -s http://<P_IP>:5701/health        # {"ok": true, "prefill": 200, "decode": 200, ...}
```

Run the smoke suite (client 10 / server 6) after `/health` is green.

## Performance

| Setup | Machines | Prefill tok/s | Aggregate Decode tok/s | Test basis |
| --- | --- | --- | --- | --- |
| **V10 (this)** | 1×6000D + 4×Spark | 100K input, C1/C2/C4/C8: **10082 / 10308 / 10372 / 10340** | 100K context, C8: **214.3**; short context C1/C4/C8: 81.7 / 186.7 / **260.5** | 2026-10-14 full-path proxy run |
| V8 | 2×6000D + 4×Spark | 32K input, C16/C24/C32: **16326.03 / 16297.20 / 16297.81** | short-code C32: **442.02** | 2026-09-21 seq32 batches |
| 4-node baseline | 4×Spark TP4, no 6000D | single stream **4138.75**; C4 **4451.46** | code C8: **323.24** | 4096-chunk baseline |

Rows differ in input length, concurrency and workload; compare orders of
magnitude only. The single 6000D sustains a marginal Prefill rate of
**10,195 tok/s** (98.1 µs/token), above the per-card share of the official
dual-card TP2 deployment.

## Operations

Restart order: **shard servers → D40 → P → proxy**. The shard servers use
`restart=unless-stopped`; `dsv41-p21-solo` and `dsv41-d40-sgl` use `restart=no`
and must be started manually after a reboot. If a request is slow while the GPU
is only half busy, check `docker logs dsv41-p21-solo | grep 'ACTIVE read path'`
— both layers must report `REMOTE`.

See [DEPLOYMENT.zh-CN.md](DEPLOYMENT.zh-CN.md) for the complete Chinese
deployment record, including root-cause analysis of the Prefill slowdown, the
RDMA troubleshooting matrix and the full command reference.
