# Heterogeneous GPU PD Lab

[English](README_EN.md)

这是一个异构 GPU Prefill/Decode（PD）实验记录仓库，测试 RTX 3060 12GB 与 AMD Ryzen AI Max+ 395 / Radeon 8060S 在同一台主机上的协作边界。

当前版本：**v2.0 — 融合切层流水**。本仓库只公开架构、测量数据、结论与限制，不包含部署或复现步骤。

## 两个版本

| 版本 | 架构 | 主要目标 | 主要代价 |
| --- | --- | --- | --- |
| [v1.0](results/v1.0-independent-pd.md) | 独立 PD：3060 prefill，395 decode | 请求间双工潜力、KV 集中交接 | 单请求 prefill 只使用 3060 |
| [v2.0](results/v2.0-fused-layer-pipeline.md) | 融合切层：两端按层共同计算 | 提高单请求 prefill 吞吐 | 两端被同一请求占用，失去独立 PD 双工 |

v2.0 使用 llama.cpp 上游 [PR #18626](https://github.com/ggml-org/llama.cpp/pull/18626) 提供的 RPC events/async 能力。模型层分布在两端，prompt 被切成 micro-batch；当 395 处理上一批的后段层时，3060 可以开始下一批的前段层，从而形成真正的异构流水重叠。本实验没有自研引擎补丁。

## 测试环境

| 项目 | 配置 |
| --- | --- |
| 模型 | 9B，Q6_K |
| Prefill 端 | RTX 3060 12GB，CUDA |
| Decode 端 | Ryzen AI Max+ 395 / Radeon 8060S，Vulkan |
| v1.0 输入 / 输出 | 5064 / 128 token |
| v1.0 测量方式 | 关闭缓存，取第二次 server 请求结果 |
| v2.0 测量方式 | pp5064 与 tg128 基准 |

## v1.0 实测：独立 PD

| 配置 | TTFT | Prefill | Decode |
| --- | ---: | ---: | ---: |
| 纯 395 | 5.879 s | 861.55 tok/s | 30.24 tok/s |
| 前段 512 token 独立 PD | 5.720 s | 886.62 tok/s | 30.22 tok/s |
| 前段 2048 token 独立 PD | 5.077 s | 999.16 tok/s | 30.18 tok/s |
| 完整独立 PD | **3.496 s** | **1452.29 tok/s** | **30.28 tok/s** |

完整独立 PD 交接了 5063 token、208.57 MiB 状态数据。它在这组 9B 测试中把 TTFT 从 5.879 s 降至 3.496 s，同时保持约 30 tok/s 的 decode。

## v2.0 实测：融合切层

比例表示 RTX 3060 / Radeon 8060S 的层分配。空白项表示该节点未单独记录该指标。

| 节点 | 层比例 | Prefill | Decode | TTFT |
| --- | ---: | ---: | ---: | ---: |
| 纯 3060 | 1.00 / 0.00 | 1589.00 tok/s | 43.87 tok/s | 未记录 |
| 纯 395 | 0.00 / 1.00 | 970.00 tok/s | 31.27 tok/s | 未记录 |
| 融合切层 | 0.50 / 0.50 | 1865.08 tok/s | 未记录 | 未记录 |
| **融合切层最佳点** | **0.58 / 0.42** | **2129.69 tok/s** | 未记录 | 未记录 |
| 融合切层 | 0.64 / 0.36 | 1999.51 tok/s | 37.16 tok/s | 未记录 |
| 融合切层 | 0.70 / 0.30 | 1893.87 tok/s | 未记录 | 未记录 |

最佳点相较纯 3060 prefill 快约 34.0%，相较 v1.0 完整独立 PD 的工程对照值快约 46.7%；它达到两端单卡 prefill 吞吐之和 2559 tok/s 的约 83.2%。0.64 / 0.36 节点的 decode 相较纯 395 快约 18.8%，相较 v1.0 完整 PD 快约 22.7%，但仍低于纯 3060。

## 结论与边界

- v1.0 证明了 CUDA prefill 端与 Vulkan decode 端之间的异构状态交接可以工作，并保留请求间双工潜力。
- v2.0 证明了异步 micro-batch 能让 CUDA 与 Vulkan 的按层流水产生真实计算重叠；最佳实测层比例为 0.58 / 0.42。
- v1.0 来自 server 请求测试，v2.0 来自 llama-bench pp/tg；跨版本百分比仅作工程对照，不是严格同口径的论文结论。
- 当前证据仅覆盖单机、9B、单并发短基准；27B、100K 上下文、多并发和多机均未验证。
- 这是实验数据，不代表生产可用性或其他模型、硬件上的必然结果。

详细记录见 [v1.0 独立 PD](results/v1.0-independent-pd.md)、[v2.0 融合切层](results/v2.0-fused-layer-pipeline.md) 与 [结构化数据](data/benchmark-results.csv)。
