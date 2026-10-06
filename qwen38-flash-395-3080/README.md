# Qwen3.8-Flash-Next NVFP4: RTX 3080 + AI Max 395 (single host)

One host runs a 125B Qwen3.8-Flash-Next NVFP4 hybrid MoE with a hot/cold expert split: the RTX 3080 20GB (OCuLink) computes hot experts and the dense main path, while the AI Max 395 handles cold experts on the 8060S iGPU (Strata HIP cold-expert library) or on its Zen5 AVX-512 CPU pool. Both the baseline and the split run Decode on the 395, so the gain shows up in Prefill.

| Metric | FLASH-395-01 (3080 + 395) | 395 solo baseline |
| --- | ---: | ---: |
| Prefill | **800+ tok/s** | **273.04 tok/s** |
| Decode | **40+ tok/s** | **41.33 tok/s** |

The experimenter supplied these figures on 2026-10-06. `800+` and `40+` are approximations; the two baselines are exact. Raw workload logs were not attached to this revision. Prefill rises about **3x**; Decode stays flat because the 395 decodes in both deployments. Values are frozen in [evidence/metrics.json](evidence/metrics.json) and enforced by `verify_bundle.py`.

## Prerequisites

- AI Max 395 host (Zen5, AVX-512) with **>= 96 GB RAM**.
- **RTX 3080 20GB over OCuLink**, driver + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html); compute capability 8.6 (`sm_86`).
- Fastest available **NVMe** for the expert pack and the HIP cache (cold-expert working set far exceeds the slot count).
- Linux kernel and Docker with the Compose plugin. Only `linux/amd64` is supported or built.
- Optional: a ROCm/HIP toolchain on the host (`hipcc`, or `/opt/rocm`) if you want the 8060S to compute cold experts. It is **not** in the image, so it is compiled at container start with `hipcc -O2 --offload-arch=gfx1151 -fPIC -shared` and cached under `/cache`.

## Model preparation

The image ships no weights. Download the 125B Qwen3.8-Flash-Next NVFP4 checkpoint and convert it into the pack files below, then mount the directory read-only at `/models`:

| Asset | Default path | Notes |
| --- | --- | --- |
| Expert pack | `/models/experts.bin` | NVFP4 cold/hot expert pack (env `STRATA_EXPERTS_PACK`) |
| Dense weights | `/models/dense.gguf` | env `STRATA_DENSE_GGUF` |
| PLE table | `/models/ple.bin` | env `STRATA_PLE` |
| Embedding | `/models/embedding.bin` | env `STRATA_EMBED` |
| MTP head | `/models/mtp.bin` | env `STRATA_MTP` |
| Engine binary | `/models/strata/strata` | built from the Strata NVFP4 source for `sm_86` (env `STRATA_ENGINE_BIN`) |

Every path is overridable through `bundle/runtime.env` or the environment. A raw checkpoint is not a substitute for a converted pack.

## Start the engine

```bash
git clone https://github.com/Soulmate-Halo/heterogeneous-gpu-pd-lab.git
cd heterogeneous-gpu-pd-lab/qwen38-flash-395-3080
docker pull ghcr.io/soulmate-halo/heterogeneous-gpu-pd-lab/qwen38-flash-395-3080:latest
docker run --rm ghcr.io/soulmate-halo/heterogeneous-gpu-pd-lab/qwen38-flash-395-3080:latest help
docker run --rm ghcr.io/soulmate-halo/heterogeneous-gpu-pd-lab/qwen38-flash-395-3080:latest verify

# docker run, single host
docker run --rm -it --gpus all --network host --ipc host \
  -v /srv/qwen395-models:/models:ro \
  -v /srv/qwen395-cache:/cache -v /srv/qwen395-state:/state \
  ghcr.io/soulmate-halo/heterogeneous-gpu-pd-lab/qwen38-flash-395-3080:latest engine

# docker compose
docker compose --profile engine up -d
docker compose logs -f engine
curl -f http://localhost:8095/health
```

The engine listens on **8095** (`STRATA_HTTP_BIND`, default `0.0.0.0:8095`). Model loading and the first HIP library build can take a long time; the healthcheck allows a 30 minute start period. `apply-bundle.sh --overlay-dir <dir>` copies `runtime.env` and `config/server.json` into an overlay directory for a hand-managed deployment.

## Doctor

```bash
docker compose --profile doctor run --rm doctor     # or: docker run --rm <image> doctor
docker run --rm --gpus all <image> doctor engine    # strict: exits 2 when something is missing
```

`doctor` checks Linux, `nvidia-smi` present with compute capability **8.6**, AVX-512 (`avx512f` in `/proc/cpuinfo`), memory **>= 96 GB**, and the presence of the expert pack / GGUF / PLE / embedding / MTP assets. It prints `PASS`/`MISSING` per item and, when ROCm is found, reports that `STRATA_HIPDECODE=1` can offload cold experts to the 8060S.

## 8060S cold-expert acceleration

`bundle/runtime.env` enables it by default:

```
STRATA_HIPDECODE=1
STRATA_HIPCOLD_LIB=<cache-dir>/hipcold/libstrata_hipdecode.so
FRAC=128
HIPCOLD_SYNC_MODE=block
HIPCOLD_SLOTS=384
```

`FRAC=128` gives the 8060S a 128/256 static hash share of cold experts and leaves the rest to the parallel CPU pool; `HIPCOLD_SYNC_MODE=block` avoids the `sched_yield` starvation that made a full-workload sync take seconds; `HIPCOLD_SLOTS` trades cache slots against the frequent sub-1 GB free memory on the APU.

## Fallback

```
STRATA_HIPDECODE=0
```

No rebuild is needed: the HIP tier turns off at zero cost and cold experts run on the AVX-512 CPU pool. Compose users can also drop the 8060S share by setting `FRAC=0` in `runtime.env`. A full rollback is the baseline engine plus the same environment without the HIP variables.

## Performance

| Deployment | Prefill | Decode |
| --- | ---: | ---: |
| 395 solo baseline | 273.04 tok/s | 41.33 tok/s |
| RTX 3080 + AI Max 395 hot/cold split | 800+ tok/s | 40+ tok/s |

Reported by the experimenter on 2026-10-06; `800+`/`40+` are approximate and no raw logs were attached. Decode is served by the 395 in both deployments, so the split does not change it; the win is Prefill. Packaging smoke tests are not GPU benchmarks.

Evidence and provenance live in [evidence](evidence/README.md); the frozen cold-expert sources are under `bundle/src/` and the r434/r435 reports record the fixes that made the split work.
