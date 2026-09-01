# Heterogeneous GPU PD Lab

[中文](README.md)

This repository records a heterogeneous GPU Prefill/Decode (PD) experiment using an RTX 3060 12GB and an AMD Ryzen AI Max+ 395 / Radeon 8060S in one host.

Current release: **v2.0 — Fused Layer Pipeline**. The public scope is limited to architecture, measured data, conclusions, and limitations. Deployment and reproduction instructions are intentionally excluded.

## Releases

| Release | Architecture | Primary objective | Main tradeoff |
| --- | --- | --- | --- |
| [v1.0](results/v1.0-independent-pd.md) | Independent PD: 3060 prefill, 395 decode | Cross-request duplex potential and centralized KV handoff | Single-request prefill uses only the 3060 |
| [v2.0](results/v2.0-fused-layer-pipeline.md) | Fused layer pipeline: both endpoints compute assigned layers | Higher single-request prefill throughput | Both endpoints are occupied by one request; independent-PD duplex is lost |

v2.0 relies on RPC events/async from upstream llama.cpp [PR #18626](https://github.com/ggml-org/llama.cpp/pull/18626). Model layers are distributed across the endpoints and the prompt is divided into micro-batches. While the 395 processes the later layers of one batch, the 3060 can begin the earlier layers of the next batch, creating genuine heterogeneous pipeline overlap. This experiment claims no custom engine patch.

## Test Environment

| Item | Configuration |
| --- | --- |
| Model | 9B, Q6_K |
| Prefill endpoint | RTX 3060 12GB, CUDA |
| Decode endpoint | Ryzen AI Max+ 395 / Radeon 8060S, Vulkan |
| v1.0 input / output | 5064 / 128 tokens |
| v1.0 measurement | Cache disabled; second server request retained |
| v2.0 measurement | pp5064 and tg128 benchmarks |

## v1.0 Measurements: Independent PD

| Configuration | TTFT | Prefill | Decode |
| --- | ---: | ---: | ---: |
| 395 only | 5.879 s | 861.55 tok/s | 30.24 tok/s |
| Independent PD for first 512 tokens | 5.720 s | 886.62 tok/s | 30.22 tok/s |
| Independent PD for first 2048 tokens | 5.077 s | 999.16 tok/s | 30.18 tok/s |
| Full independent PD | **3.496 s** | **1452.29 tok/s** | **30.28 tok/s** |

Full independent PD handed off 5063 tokens and 208.57 MiB of state. In this 9B test, TTFT fell from 5.879 s to 3.496 s while decode remained near 30 tok/s.

## v2.0 Measurements: Fused Layer Pipeline

Ratios are RTX 3060 / Radeon 8060S layer allocations. A blank measurement means that metric was not recorded separately for the node.

| Node | Layer ratio | Prefill | Decode | TTFT |
| --- | ---: | ---: | ---: | ---: |
| 3060 only | 1.00 / 0.00 | 1589.00 tok/s | 43.87 tok/s | Not recorded |
| 395 only | 0.00 / 1.00 | 970.00 tok/s | 31.27 tok/s | Not recorded |
| Fused layers | 0.50 / 0.50 | 1865.08 tok/s | Not recorded | Not recorded |
| **Best fused node** | **0.58 / 0.42** | **2129.69 tok/s** | Not recorded | Not recorded |
| Fused layers | 0.64 / 0.36 | 1999.51 tok/s | 37.16 tok/s | Not recorded |
| Fused layers | 0.70 / 0.30 | 1893.87 tok/s | Not recorded | Not recorded |

The best node is about 34.0% faster than 3060-only prefill and about 46.7% above the v1.0 full-PD engineering reference. It reaches about 83.2% of the 2559 tok/s sum of both standalone prefill measurements. Decode at 0.64 / 0.36 is about 18.8% faster than the 395-only node and 22.7% above v1.0 full PD, while remaining below 3060-only decode.

## Findings and Limits

- v1.0 demonstrates a working heterogeneous state handoff between CUDA prefill and Vulkan decode endpoints while retaining cross-request duplex potential.
- v2.0 demonstrates that asynchronous micro-batches can create real compute overlap in a CUDA/Vulkan layer pipeline. The best measured allocation was 0.58 / 0.42.
- v1.0 uses server-request measurements, while v2.0 uses llama-bench pp/tg. Cross-release percentages are engineering comparisons, not strict same-method research claims.
- Evidence is limited to a single host, a 9B model, and a short single-concurrency benchmark. 27B, 100K context, concurrent workloads, and multi-host operation remain untested.
- These are experimental measurements, not a production-readiness claim or a guarantee for other models and hardware.

See [v1.0 independent PD](results/v1.0-independent-pd.md), [v2.0 fused layer pipeline](results/v2.0-fused-layer-pipeline.md), and the [structured data](data/benchmark-results.csv).
