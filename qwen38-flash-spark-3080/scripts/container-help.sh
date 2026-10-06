#!/usr/bin/env bash
set -euo pipefail
cat <<'HELP'
Qwen3.8 Flash V2 / Spark + RTX 3080
Commands:
  help                  Display setup and role commands (no GPU required).
  verify                Check the frozen bundle and published data.
  doctor [spark|3080]    General diagnostics, or strict role/device/asset checks.
  build spark|3080      Compile the native CUDA role into /cache (no GPU required).
  spark                 Start Spark HTTP and in-process RDMA cold expert service.
  3080                  Start the 3080 engine with Spark cold experts over RDMA.
  router                Start the optional HTTP request router.
  smoke                 Run packaging/startup tests without GPUs or weights.
  shell                 Open a shell for debugging inside the image.

Required before serving: Linux, GPU driver, NVIDIA Container Toolkit, RDMA,
matched model packs + dense.gguf + ple-fp8.bin, and 3080 profile.bin.
Mount models read-only under /models; cache /cache and /state in Docker volumes.
Set STRATA_REMOTE_HOST to the Spark RDMA peer for the 3080 role.
The first role start compiles native CUDA sm_121 (Spark) or sm_86 (3080).
The cached binaries are reused; rebuilding can be requested with build <role>.
Run scripts/deploy.sh spark or scripts/deploy.sh 3080 on separate hosts.

Topology limit: the frozen HTTP router routes whole requests; cross-host KV
export/import is unavailable. Joint Decode uses actual RDMA hot/cold expert
compute, but this bundle cannot perform seamless per-request KV handoff.
V2 reported Decode: aggregate 242.37 / solo 107.60 tok/s;
single stream 62.00 / solo 22.30 tok/s. Experimenter report dated 2026-10-06;
raw workload logs were not included. Packaging smoke is not a GPU benchmark.
HELP
