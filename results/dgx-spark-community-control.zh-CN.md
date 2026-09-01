# DGX Spark 社区外部对照

[English](dgx-spark-community-control.md)

以下节点来自社区公开实测，未做归一化或推算，且与本地 `9B Q6_K / pp5064` 实验**不可直接比较**。

## 对照 A — 公开数据中模型规模最接近

| 字段 | 数值 |
| --- | --- |
| 设备 | NVIDIA DGX Spark / GB10 |
| 工作负载 | Qwen3.5 9B，TQ3_4S |
| 口径 | pp2048，FP4 cache-on |
| Prefill | 2766.28 tok/s |
| 引擎 | `turbo-tan/llama.cpp-tq3`，PR #53 |
| 来源日期 | PR 于 2026-06-30 创建，2026-07-01 合入该 fork |
| 来源 | [PR #53](https://github.com/turbo-tan/llama.cpp-tq3/pull/53) |

它与本地实验的模型规模接近，但量化、prompt 长度、分支与分支专属原生 FP4 cache-on 内核路径均不同。

## 对照 B — 独立 llama-bench 风格社区报告

| 字段 | 数值 |
| --- | --- |
| 设备 | NVIDIA DGX Spark / GB10 |
| 工作负载 | Nemotron-3-Nano-30B-A3B，UD-Q4_K_XL |
| 口径 | pp2048、depth 0、f16 KV |
| Prefill | 809.55 tok/s |
| 引擎 | `TheTom/llama-cpp-turboquant`，commit `1766c9133`，build 8793 |
| 来源日期 | issue 于 2026-04-01 创建 |
| 来源 | [Issue #44](https://github.com/TheTom/llama-cpp-turboquant/issues/44) |

它的模型架构与大小、量化、prompt 长度和分支均不同，因此只能作为背景证据。

机器可读数据：[dgx-spark-community-controls.csv](../data/dgx-spark-community-controls.csv)。
