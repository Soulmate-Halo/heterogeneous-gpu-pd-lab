# <img src="assets/soulmate-spirit.png" alt="Soulmate spirit" width="44" align="absmiddle"> AI Max+ 395 Acceleration — Heterogeneous GPU PD Lab

[简体中文](README_ZH.md)

An experimental record of heterogeneous GPU Prefill/Decode (PD) and fused-layer inference on one host: an RTX 3060 12GB works with an AMD Ryzen AI Max+ 395 / Radeon 8060S.

Current release: **v2.5 — Calibrated Fused-Layer Pipeline**. This repository publishes architecture, measured data, design evolution, conclusions, and limitations. It intentionally excludes deployment instructions, reproduction commands, patches, endpoints, and internal layer-allocation policy.

## What changed

- **v1.0** separated prefill and decode and transferred state to the AI Max+ 395.
- **v2.1–v2.5** evolved toward an asynchronous micro-batch pipeline in which both devices contribute to one prefill.
- **v2.5 reached 2129.69 tok/s**, 34.0% above the RTX 3060 baseline and 119.6% above the AI Max+ 395 baseline in the local `9B Q6_K / pp5064` test.
- A DGX Spark section now records community measurements as **external controls only**. Different models, quantizations, prompt lengths, forks, and kernels make them unsuitable for direct ranking.

## Architecture

```mermaid
flowchart LR
    P[Prompt] --> Q[Async micro-batch queue]
    Q --> N[RTX 3060 / CUDA<br/>front-stage layers]
    N --> A[AI Max+ 395 / Vulkan<br/>rear-stage layers and state ownership]
    A --> O[Decode and result stream]
    N -. next micro-batch overlaps .-> A
```

The fused design uses the RPC events/async capability from upstream llama.cpp [PR #18626](https://github.com/ggml-org/llama.cpp/pull/18626). Model layers stay on their assigned endpoint; micro-batches move through the stages so the two devices can work concurrently. The experiment does not claim a custom inference-engine patch.

## Design evolution

The v2 labels below are public revision names, not disclosures of the internal layer split.

| Revision | Question | Measured answer | Prefill |
| --- | --- | --- | ---: |
| v1.0 | Can CUDA prefill hand state to Vulkan decode? | Yes; full independent PD reduced TTFT while decode stayed near 30 tok/s. | 1452.29 tok/s |
| v2.1 | Can one request keep both layer stages active? | The first fused pipeline exceeded either local standalone endpoint. | 1865.08 tok/s |
| v2.2 | Can overlap be made more consistent? | An overlap refinement produced a modest additional gain. | 1893.87 tok/s |
| v2.3 | Can stage balance improve without exposing policy? | The balanced revision crossed 1999 tok/s; decode was recorded at 37.16 tok/s. | 1999.51 tok/s |
| **v2.5** | What is the best calibrated public checkpoint? | The final measured checkpoint reached 83.2% of the sum of both standalone prefill rates. | **2129.69 tok/s** |

## Local test envelope

| Item | Configuration |
| --- | --- |
| Model | 9B, Q6_K |
| Devices | RTX 3060 12GB / CUDA + Ryzen AI Max+ 395 / Radeon 8060S / Vulkan |
| v1.0 input / output | 5064 / 128 tokens |
| v1.0 measurement | Cache disabled; second server request retained |
| v2 measurement | `pp5064`; `tg128` only where recorded |

## v1.0 — Independent PD

| Configuration | TTFT | Prefill | Decode |
| --- | ---: | ---: | ---: |
| AI Max+ 395 only | 5.879 s | 861.55 tok/s | 30.24 tok/s |
| First 512 tokens on independent PD | 5.720 s | 886.62 tok/s | 30.22 tok/s |
| First 2048 tokens on independent PD | 5.077 s | 999.16 tok/s | 30.18 tok/s |
| Full independent PD | **3.496 s** | **1452.29 tok/s** | **30.28 tok/s** |

Full independent PD handed off 5063 tokens and 208.57 MiB of state. It preserved the architectural potential for cross-request prefill/decode duplexing, which this release did not benchmark under concurrency.

## v2.1–v2.5 — Fused-layer pipeline

Blank cells mean the metric was not recorded for that checkpoint.

| Checkpoint | Public description | Prefill | Decode | TTFT |
| --- | --- | ---: | ---: | ---: |
| RTX 3060 baseline | CUDA endpoint only | 1589.00 tok/s | 43.87 tok/s | — |
| AI Max+ 395 baseline | Vulkan endpoint only | 970.00 tok/s | 31.27 tok/s | — |
| v2.1 | First fused pipeline | 1865.08 tok/s | — | — |
| v2.2 | Overlap refinement | 1893.87 tok/s | — | — |
| v2.3 | Balance refinement | 1999.51 tok/s | 37.16 tok/s | — |
| **v2.5** | Final calibrated pipeline | **2129.69 tok/s** | — | — |

v2.5 is 34.0% above the RTX 3060 prefill baseline, 119.6% above the AI Max+ 395 baseline, and 83.2% of the 2559 tok/s sum of both standalone measurements. These are calculations from the local table, not claims about other hardware.

## DGX Spark community controls

These externally reported measurements are preserved for context, not ranked against the local experiment.

| Device | Community workload | Prompt metric | Prefill | Directly comparable? | Source |
| --- | --- | ---: | ---: | --- | --- |
| NVIDIA DGX Spark / GB10 | Qwen3.5 9B, TQ3_4S, fork-specific FP4 cache-on path | pp2048 | 2766.28 tok/s | **No** — quantization, prompt length, fork, and kernel differ | [llama.cpp-tq3 PR #53](https://github.com/turbo-tan/llama.cpp-tq3/pull/53) |
| NVIDIA DGX Spark / GB10 | Nemotron-3-Nano-30B-A3B, UD-Q4_K_XL, depth 0, f16 KV | pp2048 | 809.55 tok/s | **No** — model architecture/size, quantization, prompt length, and fork differ | [TurboQuant issue #44](https://github.com/TheTom/llama-cpp-turboquant/issues/44) |

See the [source notes](results/dgx-spark-community-control.md) and [machine-readable external controls](data/dgx-spark-community-controls.csv).

## Findings and limits

- v1.0 demonstrates heterogeneous state handoff between CUDA prefill and Vulkan decode.
- v2.5 demonstrates real asynchronous compute overlap in a heterogeneous layer pipeline, but occupying both endpoints removes v1.0's independent-PD duplex behavior.
- v1.0 uses server-request measurements; v2 uses llama-bench pp/tg. Cross-release percentages are engineering references, not strict same-method research claims.
- Evidence is limited to one host, a 9B model, and short single-concurrency benchmarks. 27B, 100K context, concurrent workloads, and multi-host operation remain untested.
- Community controls retain their original public methodology and are not normalized or extrapolated.

Detailed records: [v1.0 independent PD](results/v1.0-independent-pd.md), [v2.5 fused-layer evolution](results/v2.5-fused-layer-pipeline.md), [local CSV](data/benchmark-results.csv), and [changelog](CHANGELOG.md).
