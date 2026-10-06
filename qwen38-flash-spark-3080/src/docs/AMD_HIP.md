# Experimental AMD HIP backend (gfx1100)

This is a manual Linux source build for the RX 7900 XTX. It is opt-in; the
NVIDIA installer and CUDA build remain the default. Other AMD architectures,
wave64, Windows HIP, and mixed AMD/NVIDIA execution are outside this contribution.

The backend maps the CUDA-shaped runtime and BLAS calls to HIP/hipBLAS, uses
RDNA3's signed integer dot instruction for quantized kernels, and supplies
wave32 shuffle/packed-byte operations. CUDA-only QSA matrix instructions have
an ordered FP32 fallback. Prefill supports both dequantization plus hipBLAS GEMM and opt-in HIP ggml MMQ.
An optional, calibrated hipBLASLt path accelerates dense projections on gfx1100.
This does not claim bit-identical model answers across backends. See
[performance settings and evidence](AMD_HIP_PERFORMANCE.md).

## Install with setup (recommended)

On Linux with an RX 7900 XT / XTX and the kernel's amdgpu driver (no ROCm install needed):

```sh
./setup.sh --backend hip
```

- **Detection:** setup finds the card through the kernel's KFD topology. Integrated Radeon GPUs are listed as not
  supported. On a PC without an NVIDIA card Strata can use, `--backend hip` is chosen automatically.
- **ROCm:** installed into `.venv` from AMD's TheRock wheels (~10 GB, no sudo), pinned to the version this backend was
  tested with (`STRATA_ROCM_VERSION` / `STRATA_ROCM_INDEX` override it). A system ROCm in `/opt/rocm` (or
  `$ROCM_PATH`) with hipcc and hipBLAS is used instead when present.
- **Engine:** compiled on your PC (10-20 minutes, once; again after a `git pull` that changes it). This needs a C++
  compiler and git (`sudo apt install build-essential git`).
- **Limits for now:** one GPU, no images, no calibration. The Monitor shows no GPU statistics.

The rest of setup is the same as on NVIDIA: the model download, the start script, the server.

## Build

Requirements: a working ROCm driver/runtime, HIP development headers and
compiler, hipBLAS and (for tuned dense prefill) hipBLASLt development files, CMake 3.24+, a C++20 host compiler, and Git.
The model still needs sufficient system RAM and fast SSD storage for its PLE
table. VRAM occupancy alone is not a throughput measurement.

```sh
cmake -S . -B build-hip \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100
cmake --build build-hip --target strata -j2
```

If CMake cannot find the HIP compiler, add
`-DCMAKE_HIP_COMPILER=/path/to/rocm/llvm/bin/clang++`. On the Fedora-family test
host this was `/usr/lib64/rocm/llvm/bin/clang++`. Use the compiler and libraries
from the same ROCm installation. Do not enable both GPU backends in one build.

Native IQ experts are enabled by default. CMake fetches the llama.cpp revision
pinned by this repository. For an offline build, point `STRATA_GGML_DIR` at a
checkout of that exact revision; using an arbitrary newer checkout changes the
dependency being tested.

## Model and serving configuration

Prepare a supported model pack and its MTP runtime using the existing tools.
For OrcaRouter IQ3_XXS, follow [the explicit compatibility conversion](ORCA.md);
this backend does not change quantization, model licenses, or tokenizers.
Add `--experts-bin` to the `tools/iq_pack.py` command: mmap requires the pack's
`experts.bin`, which the default native packing command does not emit. This
consumes additional disk space. Original GSQ-RCO IQ3_XXS uses shard 2 for PLE;
Orca uses shard 1. Keep each model's own pack and tokenizer together.

Use `build-hip/strata` as the executable in the server JSON. Select the AMD
device with `ROCR_VISIBLE_DEVICES`/`HIP_VISIBLE_DEVICES` if necessary. Start with
a modest context and prefill chunk size before measuring larger workloads.
Keep the PLE file on an SSD. Do not assume a configured context length proves
successful full-window inference.

Large pinned host allocations can fail on ROCm even when ordinary RAM is
available. The existing `--mmap-experts` path avoids allocating the full pinned
expert arena; it still depends on OS file-cache residency and may stall on
storage reads. It does not make SSD access equivalent to RAM.

Starting args for the original GSQ-RCO IQ3_XXS model (replace the paths):

```sh
build-hip/strata --serve \
  --pack packs/iq3xxs \
  --native /path/to/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  --ple-gguf /path/to/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf \
  --mmap-experts --expert-profile data/expert-profile.bin --expert-cache auto \
  --prefill 512 --spec 4 --spec-min-p 0.5 --mtp mtp/rt \
  --max-context 4096 --kv int8 --pool-workers 15 \
  --adapt-every 0 --pcie-frac 0 --vram-reserve-mib 1024
```

`--serve` is the engine's internal token protocol. For a browser or OpenAI API,
put these args in the `args` array of the [server JSON example](ORCA.md#local-server),
set `exe` to `build-hip/strata`, and use the matching pack's `tokenizer` directory.
Launch with `.venv/bin/python -m serve.server --engine strata --config strata-hip.json --port 8080`.
The worker count above was used on a 16-core CPU; measure it for your CPU.
The 4K context is a smoke-test starting point, not a model limit. The expert cache
sizes itself automatically and leaves 1 GiB of VRAM headroom.

The installer supports this backend (see "Install with setup" above). The vision helper and multi-GPU layer
splits are NVIDIA-only for now.

## Original backend validation (PR #94)

The following is historical validation of the original backend, not a fresh
test count for this replacement. Current build/test evidence and performance
limits are recorded in [AMD_HIP_PERFORMANCE.md](AMD_HIP_PERFORMANCE.md).

Tested against upstream `c1e903310f211e6630780c3bd2038778c071c68d` (0.1.20),
with pinned llama.cpp `3cf03257f219afbe7334045ff7c6a06ac68c627d`.
The host has an RX 7900 XTX (24 GiB), Ryzen 9 7950X3D, 64 GiB system RAM,
and Fedora-family Linux with ROCm 7.1.52802 / Clang 20. The original model used
for the smoke check was on mechanical RAID0; cold page faults were slow. Keep
latency-sensitive data on SSD and measure warm residency separately.

- Complete HIP Release build, including the executable and parity targets.
- All 25 tests selected by the command below passed on 2026-09-29.
- Intrinsic parity includes signed packed-byte dot products and overflow.
- CPU/GPU mapped-memory handoff covers delayed publication, changing payloads,
  and repeated graph replay in both directions.
- GR coverage includes the maximum eight-token fused batch and graph replay
  after inputs change. Native QSA checks the scalar HIP fallback.
- Q8_K quantization is byte-exact against the existing reference. HIP's
  `__fmul_rn` can become ordinary multiplication; disabling contraction in
  `quantize_act.cu` preserves the separately rounded product before the magic
  rounding bias is added. CUDA's compile flags remain unchanged.
- The mmap regression covers canonical and differing native layer sizes,
  truncated/oversized files, lookup bounds, and reopening after errors.
- CUDA 13 / GCC 15 / sm_89 executable build passed; GR, sampler, quantization,
  and mmap tests passed on the same tree. This is a regression check, not a
  claim of complete CUDA inference validation.
- CPU-only `strata-plan` build passed without either GPU backend.
- Real GSQ-RCO IQ3_XXS server smoke with mmap, the shipped expert profile,
  automatic GPU cache, and MTP passed arithmetic, executable Python addition,
  system-marker recall, and 1,170-token batched-prefill recall. All four requests
  ended normally with correct answers; one engine process served them all.
  The cache held 11,142 experts (18.08 GiB), with 874 MiB of VRAM free after
  startup. These short checks do not establish a throughput or quality benchmark.

Build all targets before running the registered focused checks:

```sh
cmake --build build-hip -j2
ctest --test-dir build-hip --output-on-failure --timeout 60 \
  -E '^(ple_parity|platform_memory_test)$'
```

`ple_parity` requires an external model fixture. `platform_memory_test` requests
256 MiB of locked memory; the test shell's 8 MiB memlock limit prevented that
test from passing. These are explicit exclusions, not skipped tests counted as
passes. Additional unpublished upstream fixture suites are not covered.

The published performance table explicitly identifies the measured development
snapshot; it must not be read as a benchmark of every subsequent rebase. Vision, long-context stress, broad answer-quality
equivalence, other AMD cards, and mixed-vendor inference are not validated here.
