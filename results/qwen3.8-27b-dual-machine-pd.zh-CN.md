# Qwen3.8-27B 双机异构 PD — 独立口径

[English](qwen3.8-27b-dual-machine-pd.md)

本组把实验从 RTX 3060 迁移到 **RTX 3080 20GB** 加速头，记录 Qwen3.8-27B Q4 双机异构 PD 的全量结果。它与此前 `9B Q6_K / pp5064` 主表、以及 `27B IQ3` 验证行严格分开，不参与 v2.4 百分比计算。

## 加速版本与加速头

- **加速头**：**RTX 3080 20GB**（纯 CUDA0 全量 prefill，无 RPC、无 draft 投机）。
- **被加速主机**：**AMD Ryzen AI Max+ 395 / Radeon 8060S**（Vulkan decode，持全量 KV 池，启用 DFlash2 投机解码）。
- **模型**：Qwen3.8-27B，Q4_K_M（约 17.66 GiB）。
- **引擎**：llama.cpp fork（HEAD `18c8dde`，含上游 [PR #18626](https://github.com/ggml-org/llama.cpp/pull/18626) 的 async/RPC 能力），运行于双实例：8082 纯 CUDA0 全量 prefill，8081 纯 395 decode 并持全量 KV 池（`/dev/shm/kvx`）。
- **3080 裸算基准**：pp1024 = 1228.53、pp4096 = 1203.06、tg64 = 33.08 tok/s。

## 服务态 12 档压测（split vs 395 solo）

`bench_3080_cuda_full.log`，ub1024，十二档零错误。

| C | TTFT split/solo (ms) | prefill split/solo (tok/s) | decode split/solo (tok/s) | KV 迁移税 (save+rest) |
| --- | ---: | ---: | ---: | ---: |
| 1 | **1073** / 4825 | **1000.6** / 207.2 | 38.75 / 36.33 | 71 ms |
| 2 | 1680 / 9649 | 1008.4 / 103.6 | 27.94 / 24.36 | 76 ms |
| 3 | 2680 / 15321 | 1014.5 / 65.3 | 21.45 / 21.17 | 68 ms |
| 4 | 3255 / 19763 | 1010.9 / 50.6 | 18.34 / 14.44 | 72 ms |
| 5 | 4457 / 22460 | 1015.5 / 44.5 | 13.10 / 11.48 | 72 ms |
| 6 | 5009 / 22303 | 1014.3 / 44.8 | 9.67 / 8.82 | 70 ms |

## 关键结论

- TTFT 全档约 **4.5 倍**领先于 solo。
- prefill 不随并发衰减（稳定 **1000–1015 tok/s**）。
- decode 无损（C1 38.75 vs solo 36.33）。
- KV 迁移税 71 ms。

此前 RPC 逐 ubatch 同步带来的 45% 税已消除：去掉 RPC 后同口径从 683.2 提升到 **1000.6 tok/s（+46%）**，约达 3080 裸算 1228 的 **82%**。

## 与 DGX Spark 的指标区别

| 指标 | AI Max+ 395 + RTX 3080（本实验） | DGX Spark / GB10 |
| --- | --- | --- |
| 模型/量化 | Qwen3.8-27B，Q4_K_M | Qwen3.8-27B，NVFP4 |
| 引擎 | llama.cpp（CUDA + Vulkan，fork `18c8dde`） | SGLang + DFlash2（1M 上下文） |
| KV 精度 | q4_0 | fp8_e4m3 |
| Prefill（冷） | ~1000–1015 tok/s（短 prompt 服务态） | 1170 / 800 / 615 tok/s（100K / 200K / 300K token 冷清空） |
| Prefill（裸算） | pp1024 = 1228.53 tok/s | 未公布 |
| Decode（C1） | 38.75 tok/s | 未公布 |
| TTFT（C1） | 1073 ms | 未公布 |
| 并发 | C1–C6 全档已测 | 未公布 |
| 拓扑 | 双机异构 PD（3080 prefill + 395 decode） | 单机整机 |

**不可直接排名**：量化（Q4_K_M vs NVFP4）、引擎（llama.cpp vs SGLang）、KV 精度（q4_0 vs fp8）、prompt 深度（短 prompt vs 10 万~30 万 token）、拓扑（双机异构 vs 单机）均不同，仅作同模型量级参照。

机器可读数据：[qwen27b-local-results.csv](../data/qwen27b-local-results.csv)。
