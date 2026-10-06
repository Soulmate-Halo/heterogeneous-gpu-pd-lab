# Qwen3.8-Flash-Next NVFP4：RTX 3080 + AI Max 395（单机）

单机部署 125B Qwen3.8-Flash-Next NVFP4 混合 MoE，采用热/冷专家分离：RTX 3080 20GB（OCuLink）承担热专家与主计算，AI Max 395 承担冷专家——可由 8060S 核显（Strata HIP 冷专家库）或 Zen5 AVX-512 CPU 池执行。基线与分离方案的解码都由 395 承担，因此提升体现在预填充。

| 指标 | FLASH-395-01（3080 + 395） | 395 单机基线 |
| --- | ---: | ---: |
| 预填充 | **800+ tok/s** | **273.04 tok/s** |
| 解码 | **40+ tok/s** | **41.33 tok/s** |

数据由实验者于 2026-10-06 提供；`800+` 与 `40+` 为约值，两个基线为精确值，原始负载日志未随本次修订附入。预填充约提速 **3 倍**；解码基本持平，因为两种部署的解码都由 395 承担。取值冻结在 [evidence/metrics.json](evidence/metrics.json)，由 `verify_bundle.py` 机器校验。

## 硬件要求

- AI Max 395 主机（Zen5，支持 AVX-512），**内存 >= 96 GB**。
- **RTX 3080 20GB，OCuLink 连接**，驱动 + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)；计算能力 8.6（`sm_86`）。
- 最快的 **NVMe**（专家包与 HIP 缓存都放这里；冷专家工作集远大于槽位数）。
- Linux 内核与带 Compose 插件的 Docker。只支持并只构建 `linux/amd64`。
- 可选：宿主机有 ROCm/HIP 工具链（`hipcc` 或 `/opt/rocm`）才能让 8060S 算冷专家。它不进镜像，而是容器启动时用 `hipcc -O2 --offload-arch=gfx1151 -fPIC -shared` 编译并缓存到 `/cache`。

## 模型准备

镜像不含权重。请自行下载 125B Qwen3.8-Flash-Next NVFP4 checkpoint 并转换为下列文件，只读挂载到 `/models`：

| 资产 | 默认路径 | 说明 |
| --- | --- | --- |
| 专家包 | `/models/experts.bin` | NVFP4 冷/热专家包（环境变量 `STRATA_EXPERTS_PACK`） |
| 稠密权重 | `/models/dense.gguf` | 环境变量 `STRATA_DENSE_GGUF` |
| PLE 表 | `/models/ple.bin` | 环境变量 `STRATA_PLE` |
| Embedding | `/models/embedding.bin` | 环境变量 `STRATA_EMBED` |
| MTP 头 | `/models/mtp.bin` | 环境变量 `STRATA_MTP` |
| 引擎二进制 | `/models/strata/strata` | 由 Strata NVFP4 源码按 `sm_86` 构建（环境变量 `STRATA_ENGINE_BIN`） |

所有路径都可通过 `bundle/runtime.env` 或环境变量覆盖。原始 checkpoint 不能当作转换后的 pack 使用。

## 启动

```bash
git clone https://github.com/Soulmate-Halo/heterogeneous-gpu-pd-lab.git
cd heterogeneous-gpu-pd-lab/qwen38-flash-395-3080
docker pull ghcr.io/soulmate-halo/heterogeneous-gpu-pd-lab/qwen38-flash-395-3080:latest
docker run --rm ghcr.io/soulmate-halo/heterogeneous-gpu-pd-lab/qwen38-flash-395-3080:latest help
docker run --rm ghcr.io/soulmate-halo/heterogeneous-gpu-pd-lab/qwen38-flash-395-3080:latest verify

# docker run 单机
docker run --rm -it --gpus all --network host --ipc host \
  -v /srv/qwen395-models:/models:ro \
  -v /srv/qwen395-cache:/cache -v /srv/qwen395-state:/state \
  ghcr.io/soulmate-halo/heterogeneous-gpu-pd-lab/qwen38-flash-395-3080:latest engine

# docker compose
docker compose --profile engine up -d
docker compose logs -f engine
curl -f http://localhost:8095/health
```

引擎监听 **8095**（`STRATA_HTTP_BIND`，默认 `0.0.0.0:8095`）。模型加载与首次 HIP 库编译可能很久，健康检查给了 30 分钟启动宽限。`apply-bundle.sh --overlay-dir <dir>` 会把 `runtime.env` 与 `config/server.json` 拷到指定 overlay 目录，便于手工部署。

## doctor 用法

```bash
docker compose --profile doctor run --rm doctor     # 或 docker run --rm <image> doctor
docker run --rm --gpus all <image> doctor engine    # 严格模式：缺项退出码 2
```

`doctor` 检查 Linux、`nvidia-smi` 存在且计算能力 **8.6**、CPU 支持 AVX-512（`/proc/cpuinfo` 中的 `avx512f`）、内存 **>= 96 GB**，以及专家包 / GGUF / PLE / embedding / MTP 资产是否存在。逐项输出 `PASS`/`MISSING`；检测到 ROCm 时提示可用 `STRATA_HIPDECODE=1` 把冷专家交给 8060S。

## 8060S 冷专家加速开关

`bundle/runtime.env` 默认开启：

```
STRATA_HIPDECODE=1
STRATA_HIPCOLD_LIB=<cache-dir>/hipcold/libstrata_hipdecode.so
FRAC=128
HIPCOLD_SYNC_MODE=block
HIPCOLD_SLOTS=384
```

`FRAC=128` 表示 8060S 按静态哈希承担 128/256 的冷专家份额，其余留给并行的 CPU 池；`HIPCOLD_SYNC_MODE=block` 避免 CPU 全忙时同步被 `sched_yield` 退避饿死（曾导致单次同步耗时数秒）；`HIPCOLD_SLOTS` 用于在缓存槽位与 APU 常见的 1 GB 以下空闲内存之间取舍。

## 回退方法

```
STRATA_HIPDECODE=0
```

无需重建二进制：HIP 层零开销关闭，冷专家改由 AVX-512 CPU 池处理。Compose 用户也可在 `runtime.env` 里把 `FRAC=0` 让 8060S 不承接份额。彻底回滚 = 基线引擎 + 去掉 HIP 相关环境变量。

## 性能表

| 部署 | 预填充 | 解码 |
| --- | ---: | ---: |
| 395 单机基线 | 273.04 tok/s | 41.33 tok/s |
| RTX 3080 + AI Max 395 冷热分离 | 800+ tok/s | 40+ tok/s |

数据由实验者于 2026-10-06 提供；`800+`/`40+` 为约值，原始日志未附。两种部署的解码都由 395 承担，所以分离不改变解码；收益在预填充。打包冒烟测试不是 GPU 性能压测。

证据与出处见 [evidence](evidence/README.md)；冻结的冷专家源码在 `bundle/src/`，r434/r435 报告记录了让该分离方案跑通的关键修复。
