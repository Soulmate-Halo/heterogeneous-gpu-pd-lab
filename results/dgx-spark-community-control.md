# DGX Spark Community Controls

[简体中文](dgx-spark-community-control.zh-CN.md)

These rows are externally reported community measurements. They are recorded without normalization or extrapolation and are **not directly comparable** with the local `9B Q6_K / pp5064` experiment.

## Control A — closest public model size

| Field | Value |
| --- | --- |
| Device | NVIDIA DGX Spark / GB10 |
| Workload | Qwen3.5 9B, TQ3_4S |
| Metric | pp2048, FP4 cache-on |
| Prefill | 2766.28 tok/s |
| Engine | `turbo-tan/llama.cpp-tq3`, PR #53 |
| Source date | PR opened 2026-06-30; merged into that fork 2026-07-01 |
| Source | [PR #53](https://github.com/turbo-tan/llama.cpp-tq3/pull/53) |

It is close in model size but differs in quantization, prompt length, fork, and a fork-specific native FP4 cache-on kernel path.

## Control B — independent llama-bench-style community report

| Field | Value |
| --- | --- |
| Device | NVIDIA DGX Spark / GB10 |
| Workload | Nemotron-3-Nano-30B-A3B, UD-Q4_K_XL |
| Metric | pp2048 at depth 0 with f16 KV |
| Prefill | 809.55 tok/s |
| Engine | `TheTom/llama-cpp-turboquant`, commit `1766c9133`, build 8793 |
| Source date | Issue opened 2026-04-01 |
| Source | [Issue #44](https://github.com/TheTom/llama-cpp-turboquant/issues/44) |

It differs in model architecture and size, quantization, prompt length, and fork. It is therefore contextual evidence only.

Machine-readable data: [dgx-spark-community-controls.csv](../data/dgx-spark-community-controls.csv).
