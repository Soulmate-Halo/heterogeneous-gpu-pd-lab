# AI 一键构建指南 · Qwen3.8 Flash-Next V2（RTX 3080 + DGX Spark）

> 这份文件是写给“替用户自动部署的 AI”看的。请先读完【最重要的一条】和第 1、2 节，再动手。
> （English guide below / 英文见文末。）

## ⚠️ 最重要的一条：`doctor` 报 `MISSING` 不是构建失败

运行 `doctor`、`doctor spark` 或 `doctor 3080` 时打印的 **`MISSING ...`** 是**设计使然**，不是你哪里做错了，也不是镜像坏了。

本仓库与 GHCR 镜像**按设计只包含**：源码、编译脚本、补丁、配置、脱敏证据。**不包含**任何模型权重和大资产。`MISSING` 的含义是 **“这个资产需要你自己准备并挂载”**，而不是 **“构建失败”**。

因此，请**不要**做这些无用功：
- 不要因为看到 `MISSING` 就反复 `docker build` / 重装依赖 / 改 Dockerfile / 换基础镜像去“修复”它 —— 权重本来就不在仓库里，这样永远修不好。
- 不要把不带角色的 `doctor` 的输出当成失败：它**恒返回 0**（结尾打印 `DOCTOR_GENERAL_OK`），列出的 `MISSING` 只是清单提示。

正确动作只有一条主线：**把镜像跑起来（可一键完成）→ 准备下面 5 个资产并挂到 `MODELS_DIR` → 再跑 `doctor spark` / `doctor 3080` 直到全 PASS。**

## 1. 能一键完成的部分（镜像，无需 GPU 即可验证）

```bash
git clone https://github.com/Soulmate-Halo/heterogeneous-gpu-pd-lab.git
cd heterogeneous-gpu-pd-lab/qwen38-flash-spark-3080
cp env.example .env   # 填 MODELS_DIR、STRATA_REMOTE_HOST、两端 HTTP 地址
IMG=ghcr.io/soulmate-halo/heterogeneous-gpu-pd-lab/qwen38-flash-spark-3080:latest
docker pull $IMG
docker run --rm $IMG help      # 用法
docker run --rm $IMG doctor    # 通用检查，恒返回 0（DOCTOR_GENERAL_OK）
docker run --rm $IMG verify    # 包完整性
docker run --rm $IMG smoke     # 冒烟
```

`help`、`doctor`（不带角色）、`verify`、`smoke` **都不需要 GPU**，用来确认镜像本身没问题。到这一步跑通，就说明“照仓库文件构建”这部分**已经成功**。

## 2. 五个模型资产：哪些能用仓库工具自造、哪些必须外部提供 ←【关键】

这正是截图里 `doctor` 列出的那 5 项。`3080`=热端（X99 平台），`Spark`=冷端（GB10）。

| 资产 | 3080（热） | Spark（冷） | 仓库工具能造？ | 怎么拿到 |
| --- | :--: | :--: | --- | --- |
| `dense.gguf` | 需 | 需 | ✅ **能** | `python src/tools/nvfp4_dense_gguf.py --ref <Q2_0-*.gguf> --src <NVFP4 HF 目录> --out dense.gguf --verify` |
| `profile.bin` | 需 | — | ✅ **能** | `python src/tools/make_profile.py <引擎 --dump-routing 产生的 trace> --base src/data/expert-profile.bin --out profile.bin`；或直接复制仓库自带 `assets/expert-profile-v4.bin`（已验证可覆盖 `--expert-cache 4608`）当 `profile.bin`，前提是与你的 hot pack 路由匹配 |
| `strata-pack-hot/` | 需 | — | ❌ **不能** | 需要 `nvfp4_pack.py --flavor hot`，**该工具不在本仓库**；hot 风味要 GPU + sglang `marlin_utils_fp4`。只能向作者索取，或自行从 NVFP4 checkpoint 打包 |
| `strata-cold-pack/` | — | 需 | ❌ **不能** | 同一个 `nvfp4_pack.py --flavor cold --device cpu`，**不在本仓库**，与 hot 同源同版本 |
| `ple-fp8.bin`（约 51.2 GB） | 需 | 需 | ❌ **不能** | 仓库**只有读端、没有任何写入端**；它是 checkpoint 原生产物（320,001,536 行 × 160 B，e4m3），固定 scale `0.00019931793212890625`。需从 checkpoint 的 PLE 张量自行解包，或向作者索取 |

**结论**：5 项里只有 `dense.gguf`、`profile.bin` 能完全靠「公开仓库 + 公开权重」重建；`strata-pack-hot/`、`strata-cold-pack/`、`ple-fp8.bin` 这 3 项**仓库没有构建器**，必须由人工提供。全仓库也**没有**任何网盘/提取码/Release 附件来分发这些大包，只有 GHCR 配置镜像和下面的公开权重入口。

## 3. 给自动部署 AI 的硬规则

1. 跑 `doctor spark` / `doctor 3080` 后，把 `MISSING` 分两类处理：
   - `dense.gguf` / `profile.bin` → 用上表命令**自己造**出来。
   - `strata-pack-hot/` / `strata-cold-pack/` / `ple-fp8.bin` → 你**造不出来**。请**立即停下**，明确告诉用户：“这 3 个资产仓库无法生成，请提供匹配的 NVFP4 checkpoint 让我打包，或直接把作者给的 pack 目录 / `ple-fp8.bin` 放进 `MODELS_DIR`。”**不要循环重试、不要改镜像。**
2. 公开 checkpoint **不能**直接当 pack 用（格式不同），也不能拿 Q2 权重冒充 NVFP4。
3. 资产挂载齐后再跑 `doctor spark` / `doctor 3080`，**全部 PASS（exit 0）** 才算齐；带角色的 `doctor` 缺任何一项会 `exit 2`。

## 4. 资产齐全后的部署顺序

把 `strata-pack-hot/`、`strata-cold-pack/`、`dense.gguf`、`ple-fp8.bin`、`profile.bin` 放到 `MODELS_DIR`（默认 `/srv/qwen38-models`），两台机分别：

```bash
bash scripts/deploy.sh spark    # 在 Spark（冷）机
bash scripts/deploy.sh 3080     # 在 3080（热）机
bash scripts/deploy.sh router   # 可选：任一台起请求路由
```

- HTTP 端口：Spark **8200**、3080 **8100**、router **8400**；RDMA 控制 **39580**。
- `STRATA_REMOTE_HOST` 必须填 Spark 的 RDMA 端点；按你的链路配 HCA/GID。
- 健康检查：`curl -f http://localhost:8200/health`、`curl -f http://localhost:8100/health`。
- 首次原生 CUDA 编译 + 载入权重较慢，看 `docker compose --env-file .env logs -f spark` / `logs -f gpu3080`。
- 关机：`docker compose --env-file .env --profile spark --profile 3080 --profile router down`。

## 5. 公开权重入口（用来自造 dense.gguf）

- NVFP4 权重参考：[gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090)
- GGUF / PLE 键参考：[unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
- Strata 引擎上游：[Niko1221/Strata](https://github.com/Niko1221/Strata)
- 完整外置资产契约见同目录 [ASSETS.md](ASSETS.md) / [ASSETS.zh-CN.md](ASSETS.zh-CN.md)。

---

# AI One-Click Build Guide · Qwen3.8 Flash-Next V2 (RTX 3080 + DGX Spark)

> Written for the AI agent that deploys on the user's behalf.

## ⚠️ Read this first: `doctor` printing `MISSING` is **not** a build failure

`doctor`, `doctor spark` and `doctor 3080` printing **`MISSING ...`** is **by design**. This repo and its GHCR image intentionally ship **only** source, build scripts, patches, config and redacted evidence — **no model weights**. `MISSING` means *"you must supply and mount this asset"*, not *"the build broke"*.

So do **not** loop on `docker build` / reinstalling / editing the Dockerfile to "fix" `MISSING` — the weights were never in the repo. The role-less `doctor` always returns 0 (`DOCTOR_GENERAL_OK`); its `MISSING` lines are just a checklist.

The only correct path: **bring the image up (one-click) → prepare the 5 assets below and mount them under `MODELS_DIR` → re-run `doctor spark` / `doctor 3080` until all PASS.**

## 1. The one-click part (image; verifiable without a GPU)

```bash
git clone https://github.com/Soulmate-Halo/heterogeneous-gpu-pd-lab.git
cd heterogeneous-gpu-pd-lab/qwen38-flash-spark-3080
cp env.example .env   # set MODELS_DIR, STRATA_REMOTE_HOST and both HTTP addresses
IMG=ghcr.io/soulmate-halo/heterogeneous-gpu-pd-lab/qwen38-flash-spark-3080:latest
docker pull $IMG
docker run --rm $IMG help
docker run --rm $IMG doctor    # general check, always returns 0
docker run --rm $IMG verify
docker run --rm $IMG smoke
```

`help`, `doctor` (no role), `verify`, `smoke` need no GPU. Passing them means the "build from repo files" part already succeeded.

## 2. The 5 model assets: which repo tools can build, which you must supply ← KEY

These are exactly the 5 items `doctor` lists. `3080` = hot side, `Spark` = cold side.

| Asset | 3080 (hot) | Spark (cold) | Buildable from repo? | How to get it |
| --- | :--: | :--: | --- | --- |
| `dense.gguf` | yes | yes | ✅ **yes** | `python src/tools/nvfp4_dense_gguf.py --ref <Q2_0-*.gguf> --src <NVFP4 HF dir> --out dense.gguf --verify` |
| `profile.bin` | yes | — | ✅ **yes** | `python src/tools/make_profile.py <routing trace> --base src/data/expert-profile.bin --out profile.bin`; or copy the bundled `assets/expert-profile-v4.bin` (verified to cover `--expert-cache 4608`) if it matches your hot pack |
| `strata-pack-hot/` | yes | — | ❌ **no** | needs `nvfp4_pack.py --flavor hot`, **not in this repo** (hot flavor needs a GPU + sglang `marlin_utils_fp4`). Ask the author or pack it yourself from an NVFP4 checkpoint |
| `strata-cold-pack/` | — | yes | ❌ **no** | same `nvfp4_pack.py --flavor cold --device cpu`, **not in this repo** |
| `ple-fp8.bin` (~51.2 GB) | yes | yes | ❌ **no** | repo has **only a reader, no writer**; it is a checkpoint-native product (320,001,536 rows × 160 B, e4m3), fixed scale `0.00019931793212890625`. Unpack it from the checkpoint's PLE tensors or ask the author |

**Bottom line:** only `dense.gguf` and `profile.bin` can be rebuilt from public repo + public weights. `strata-pack-hot/`, `strata-cold-pack/`, `ple-fp8.bin` have **no builder in the repo** and must be supplied. There is **no** netdisk / release / download for these large packs anywhere in the repo.

## 3. Hard rules for the deploying AI

1. Split the `MISSING` items in two: build `dense.gguf` / `profile.bin` yourself; for `strata-pack-hot/` / `strata-cold-pack/` / `ple-fp8.bin` **stop and ask the user** for the NVFP4 checkpoint (to pack) or the ready packs / `ple-fp8.bin`. Do not retry in a loop or rebuild the image.
2. A raw checkpoint is not a pack; Q2 weights are not NVFP4.
3. Only `doctor spark` / `doctor 3080` all-PASS (exit 0) means assets are complete; a role `doctor` exits 2 when anything is missing.

## 4. Deploy order once assets are complete

Place the 5 assets under `MODELS_DIR` (default `/srv/qwen38-models`), then:

```bash
bash scripts/deploy.sh spark    # on the Spark (cold) host
bash scripts/deploy.sh 3080     # on the 3080 (hot) host
bash scripts/deploy.sh router   # optional request router on either host
```

HTTP ports: Spark **8200**, 3080 **8100**, router **8400**; RDMA control **39580**. Set `STRATA_REMOTE_HOST` to the Spark RDMA endpoint. Health: `curl -f http://localhost:8200/health` and `:8100/health`. The first native CUDA build and model load take time.

## 5. Public weight entry points (to build `dense.gguf`)

- NVFP4 reference: [gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090)
- GGUF / PLE keys: [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
- Strata engine upstream: [Niko1221/Strata](https://github.com/Niko1221/Strata)
- Full asset contract: [ASSETS.md](ASSETS.md).
