# AI 一键构建指南 · Qwen3.8 Flash-Next（RTX 3080 + AI Max 395，单主机）

> 这份文件是写给“替用户自动部署的 AI”看的。先读【最重要的一条】和第 1、2 节。
> （English guide below。）

## ⚠️ 最重要的一条：`doctor engine` 报 `MISSING` 是设计使然

单主机方案：RTX 3080 20GB（OCuLink）算热专家与稠密主路径，AI Max 395（8060S 核显或 Zen5 AVX-512 CPU）算冷专家。基线和加速版都在 395 上解码，所以**收益体现在 Prefill**（约 273 → 843 tok/s，约 3×）。

镜像**按设计不含任何权重**，只含源码、脚本、补丁、配置与脱敏证据。所以 `doctor engine` 列出 6 个资产 `MISSING` 是**正常现象**，不是构建失败。不带角色的 `doctor` **恒返回 0**；带角色的 `doctor engine` 缺任何一项会 **`exit 2`**。

**不要**对着 `MISSING` 反复 `docker build` / 改 Dockerfile，权重不在仓库里。正确主线：**跑起镜像（一键）→ 准备并挂载 6 个资产到 `/models` → `doctor engine` 全 PASS → 起服务。**

## 1. 能一键完成的部分（镜像，无需 GPU）

```bash
git clone https://github.com/Soulmate-Halo/heterogeneous-gpu-pd-lab.git
cd heterogeneous-gpu-pd-lab/qwen38-flash-395-3080
IMG=ghcr.io/soulmate-halo/heterogeneous-gpu-pd-lab/qwen38-flash-395-3080:latest
docker pull $IMG            # 仅 linux/amd64
docker run --rm $IMG help
docker run --rm $IMG verify
docker run --rm $IMG doctor # 通用检查，恒返回 0
```

## 2. 六个资产：仓库没有构建器，必须外部提供 ←【关键】

`doctor engine` 检查这 6 项（环境变量可改路径）。**395 目录本身不附带任何转换脚本**，必须由人工从 125B Qwen3.8-Flash-Next NVFP4 checkpoint 转换后挂载到 `/models`（只读）。

| 资产 | 默认路径 | 环境变量 | 仓库工具能造？ |
| --- | --- | --- | :--: |
| 专家包 | `/models/experts.bin` | `STRATA_EXPERTS_PACK` | ❌ 无（需作者提供 NVFP4 包） |
| 稠密权重 | `/models/dense.gguf` | `STRATA_DENSE_GGUF` | ✅ 可借同仓 `../qwen38-flash-spark-3080/src/tools/nvfp4_dense_gguf.py` 自造 |
| PLE 表 | `/models/ple.bin` | `STRATA_PLE` | ❌ 无转换器 |
| 词嵌入 | `/models/embedding.bin` | `STRATA_EMBED` | ❌ 无 |
| MTP 头 | `/models/mtp.bin` | `STRATA_MTP` | ❌ 无 |
| 引擎二进制 | `/models/strata/strata` | `STRATA_ENGINE_BIN` | ❌ 需从 Strata NVFP4 源为 `sm_86` 编译，且要 llama.cpp / marlin / sgl_kernel / TVM-FFI，本 bundle 都没有 |

**结论**：除 `dense.gguf` 可借用同仓 spark 方案的 `nvfp4_dense_gguf.py` 自造外，其余 5 项仓库**没有公共构建器**，须向作者索取或自行从 checkpoint 转换。公开 checkpoint 不能直接当 pack 用。

## 3. 给自动部署 AI 的硬规则

1. `doctor engine` 报这 6 项 `MISSING` 时，**不要**判成构建失败、**不要**循环重试。只有 `dense.gguf` 能自造；其余请**停下并告诉用户**：“这些是转换后的模型资产，仓库不含转换器，请提供作者的 `experts.bin / ple.bin / embedding.bin / mtp.bin / strata` 或原始 NVFP4 checkpoint。”
2. 挂载齐后 `docker run --rm --gpus all $IMG doctor engine` 必须 **exit 0** 才算齐。
3. 平台前置：Linux、内存 **≥ 96 GB**、`nvidia-smi` 且算力 **8.6**、CPU 支持 `avx512f`。

## 4. 资产齐全后起服务

```bash
# docker run（单主机）
docker run --rm -it --gpus all --network host --ipc host \
  -v /srv/qwen395-models:/models:ro \
  -v /srv/qwen395-cache:/cache -v /srv/qwen395-state:/state \
  $IMG engine
# 或 docker compose
docker compose --profile engine up -d
docker compose logs -f engine
curl -f http://localhost:8095/health
```

- 引擎端口 **8095**（`STRATA_HTTP_BIND`，默认 `0.0.0.0:8095`）；首次 HIP 库编译 + 载权重慢，healthcheck 给 30 分钟启动期。
- **8060S 冷专家加速（可选）**：`bundle/runtime.env` 默认 `STRATA_HIPDECODE=1`，容器启动时用 `hipcc -O2 --offload-arch=gfx1151` 现编（需宿主机 ROCm / `hipcc`，镜像不含）。关掉改 `STRATA_HIPDECODE=0`，冷专家回落到 AVX-512 CPU 池，零重建成本。

## 5. 公开权重入口

- NVFP4 参考：[gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090)
- GGUF / PLE 键：[unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
- Strata 上游：[Niko1221/Strata](https://github.com/Niko1221/Strata)

---

# AI One-Click Build Guide · Qwen3.8 Flash-Next (RTX 3080 + AI Max 395, single host)

> Written for the AI agent that deploys on the user's behalf.

## ⚠️ Read first: `doctor engine` printing `MISSING` is by design

Single host: the RTX 3080 20GB (OCuLink) computes hot experts and the dense path; the AI Max 395 (8060S iGPU or Zen5 AVX-512 CPU) computes cold experts. Both baseline and split decode on the 395, so the gain is in Prefill (~273 → ~843 tok/s). The image ships **no weights** by design, so `doctor engine` listing 6 `MISSING` assets is expected, not a build failure. The role-less `doctor` always returns 0; `doctor engine` exits 2 when anything is missing. Do not loop on `docker build` to "fix" it.

## 1. The one-click part (image, no GPU)

```bash
git clone https://github.com/Soulmate-Halo/heterogeneous-gpu-pd-lab.git
cd heterogeneous-gpu-pd-lab/qwen38-flash-395-3080
IMG=ghcr.io/soulmate-halo/heterogeneous-gpu-pd-lab/qwen38-flash-395-3080:latest
docker pull $IMG       # linux/amd64 only
docker run --rm $IMG help
docker run --rm $IMG verify
docker run --rm $IMG doctor
```

## 2. The 6 assets: no builder in the repo, must be supplied ← KEY

`doctor engine` checks these 6. The 395 directory ships **no converter**; convert them from the 125B Qwen3.8-Flash-Next NVFP4 checkpoint and mount read-only at `/models`.

| Asset | Default path | Env var | Buildable from repo? |
| --- | --- | --- | :--: |
| Expert pack | `/models/experts.bin` | `STRATA_EXPERTS_PACK` | ❌ no (author-provided NVFP4 pack) |
| Dense weights | `/models/dense.gguf` | `STRATA_DENSE_GGUF` | 🔶 yes, via sibling `../qwen38-flash-spark-3080/src/tools/nvfp4_dense_gguf.py` |
| PLE table | `/models/ple.bin` | `STRATA_PLE` | ❌ no converter |
| Embedding | `/models/embedding.bin` | `STRATA_EMBED` | ❌ no |
| MTP head | `/models/mtp.bin` | `STRATA_MTP` | ❌ no |
| Engine binary | `/models/strata/strata` | `STRATA_ENGINE_BIN` | ❌ build from Strata NVFP4 source for `sm_86` (needs llama.cpp / marlin / sgl_kernel / TVM-FFI, none in this bundle) |

Only `dense.gguf` is repo-buildable (borrow the spark tool). The rest must come from the author or your own checkpoint conversion. A raw checkpoint is not a pack.

## 3. Hard rules for the deploying AI

1. On 6× `MISSING`, do not treat it as a build failure and do not loop. Build `dense.gguf`; for the rest, **stop and ask the user** for the converted assets or the raw NVFP4 checkpoint.
2. `docker run --rm --gpus all $IMG doctor engine` must exit 0 before serving.
3. Platform: Linux, RAM **≥ 96 GB**, `nvidia-smi` with compute capability **8.6**, CPU with `avx512f`.

## 4. Serve once assets are complete

```bash
docker run --rm -it --gpus all --network host --ipc host \
  -v /srv/qwen395-models:/models:ro -v /srv/qwen395-cache:/cache -v /srv/qwen395-state:/state \
  $IMG engine
# or: docker compose --profile engine up -d && docker compose logs -f engine
curl -f http://localhost:8095/health
```

Engine port **8095**; 30-minute healthcheck start period for the first HIP build + model load. Optional 8060S cold-expert tier: `STRATA_HIPDECODE=1` (default) compiles a gfx1151 lib at start via host `hipcc`; set `STRATA_HIPDECODE=0` to fall back to the AVX-512 CPU pool at zero rebuild cost.
