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

## Control B — Qwen3.8-27B long-prompt community report

| Field | Value |
| --- | --- |
| Device | NVIDIA DGX Spark / GB10 |
| Workload | Qwen3.8-27B, `RadixArk/Qwen3.8-27B-NVFP4` |
| Metric | Cold prefill at 100K / 200K / 300K prompt tokens; cache flushed between depths |
| Prefill | 1170 / 800 / 615 tok/s |
| Engine | SGLang + DFlash2, 1M-context profile, FP8 KV, memory fraction 0.70 |
| Source date | Measured 2026-08-29 |
| Source | [hasso5703 benchmark at commit `17e7e228`](https://github.com/hasso5703/dgx-spark-qwen38/blob/17e7e2280e632b0a3ab91839c8c7522b256937ac/BENCHMARKS.md#L232-L243) |

It uses the same model family and dense 27B scale as the local exploratory line, but differs in quantization, engine, prompt depths, and measurement method. It is therefore contextual evidence only.

Machine-readable data: [dgx-spark-community-controls.csv](../data/dgx-spark-community-controls.csv).
