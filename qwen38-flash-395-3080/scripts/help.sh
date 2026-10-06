#!/usr/bin/env bash
set -euo pipefail
cat <<'HELP'
Qwen3.8-Flash-Next NVFP4: RTX 3080 + AI Max 395 (single host, mx7)
Commands:
  help              Display setup and role commands (no GPU required).
  verify            Check the frozen bundle, sums and evidence values.
  doctor [engine]   Check Linux, RTX 3080 sm_86, AVX-512, memory and model assets.
  engine            Start the Strata NVFP4 engine on port 8095 (hot/cold split).
  freeze            Re-freeze bundle/baseline.json and SHA256SUMS from the tree.
  smoke             Run packaging/startup tests without GPUs or weights.
  shell             Open a shell for debugging inside the image.

Required before serving: Linux on the AI Max 395 host, an RTX 3080 20GB over
OCuLink, >=96 GB RAM, an NVMe data volume, GPU driver and NVIDIA Container
Toolkit, plus the 125B Qwen3.8-Flash-Next NVFP4 checkpoint converted to the
expert pack (experts.bin), dense.gguf, PLE, embedding and MTP assets.
Mount models read-only under /models; keep cache /cache and state /state.

Hot/cold split: the 3080 computes hot experts and the dense main path; cold
experts run on the 8060S iGPU (STRATA_HIPDECODE=1) or the Zen5 AVX-512 CPU pool
(STRATA_HIPDECODE=0). ROCm/HIP is not in the image; the gfx1151 library is
compiled at first start and cached under /cache.

Reported experiment (experimenter values, 2026-10-06; raw logs not attached):
Prefill 800+ tok/s vs 273.04 solo baseline; Decode 40+ tok/s vs 41.33 baseline.
Prefill rises about 3x; Decode stays flat because the 395 decodes in both
deployments. Packaging smoke is not a GPU benchmark.
HELP
