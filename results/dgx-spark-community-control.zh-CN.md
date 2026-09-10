# EXT-DGX-01 — DGX Spark 社区外部背景

[English](dgx-spark-community-control.md)

## 证据契约

| 项目 | 定义 |
| --- | --- |
| 角色 | 只作外部背景；不是本地实验，也不是匹配对照。 |
| 问题 | 哪些公开实测可以帮助理解周边硬件范围？ |
| 收录规则 | 原样保留来源负载、引擎、量化、日期和 URL，不做归一化。 |
| 禁止用途 | 不得计算本地加速倍率，也不得与口径不同的本地行做排名。 |

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

## 对照 B — Qwen3.8-27B 长 prompt 社区实测

| 字段 | 数值 |
| --- | --- |
| 设备 | NVIDIA DGX Spark / GB10 |
| 工作负载 | Qwen3.8-27B，`RadixArk/Qwen3.8-27B-NVFP4` |
| 口径 | 10万 / 20万 / 30万 token 冷 prefill；各深度之间清空缓存 |
| Prefill | 1170 / 800 / 615 tok/s |
| 引擎 | SGLang + DFlash2，100万上下文配置，FP8 KV，memory fraction 0.70 |
| 来源日期 | 2026-08-29 实测 |
| 来源 | [hasso5703 实测，commit `17e7e228`](https://github.com/hasso5703/dgx-spark-qwen38/blob/17e7e2280e632b0a3ab91839c8c7522b256937ac/BENCHMARKS.md#L232-L243) |

它与本地探索使用同一模型家族和稠密 27B 规模，但量化、引擎、prompt 深度与测量方法不同，因此只能作为背景证据。

机器可读数据：[dgx-spark-community-controls.csv](../data/dgx-spark-community-controls.csv)。
