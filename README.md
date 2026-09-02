# <img src="assets/soulmate-spirit.png" alt="Soulmate spirit" width="44" align="absmiddle"> Small-VRAM Accelerator + Large-VRAM, Low-Compute Host Dense Acceleration — Heterogeneous GPU PD Lab

[简体中文](README_ZH.md)

An experimental record of heterogeneous GPU Prefill/Decode (PD) and Dense Acceleration: RTX 3060 / RTX 3080 accelerator heads work with an AMD Ryzen AI Max+ 395 / Radeon 8060S.

Current release: **v2.7 — two 27B performance profiles: fastest prefill + fastest single-stream decode**. This repository publishes architecture, measured data, design evolution, conclusions, and limitations. It intentionally excludes deployment instructions, reproduction commands, patches, endpoints, and internal layer-allocation policy.

**What Dense Acceleration is.** An accelerator card plus a host being accelerated. Where the two devices' memory overlaps is the memory-dense region; any model falling inside it is strongly accelerated for both decode and prefill. The acceleration is lossless, and every metric exceeds what the accelerator card or the host achieves on its own.

This gives low-compute, large-memory hosts (e.g. DGX Spark, AI Max+ 395) combined with high-compute, small-memory cards (e.g. RTX 3060, RTX 3080) more application opportunity and scenarios.

![Dense Acceleration structure: an overlapping memory cell split into a sparse region that holds the model and a dense region that computes decode & prefill](assets/dense-region-structure.png)

*Structure: the overlapping memory cell splits into a **sparse region** that holds the model and a **dense region** that does the compute; a model falling into the dense region gets significant, lossless decode & prefill acceleration.*

## Reading path and this release

- Read **architecture and measurement envelopes** first, then follow **research evolution → 9B experiment chain → unified 27B experiment area → external references**. Every local experiment starts with a “New experiment + key result” heading.
- The **9B main line** keeps the same-envelope v1.0–v2.4 evolution, ending at 2129.69 tok/s prefill and 50.73 tok/s decode.
- The **27B experiment area** now places the 3060 / IQ3 and 3080 / Q4 records together without merging their measurement envelopes.
- Router v1.1's **1194.4–1210.6 tok/s is prefill**. The 35–38.5 tok/s single-stream decode figure is limited to repetitive text with 100% draft acceptance; v1.2 two-stage preemption and the natural-language audit are reported separately.

## Dense Acceleration architecture

```mermaid
flowchart LR
    P[Prompt] --> Q[Async micro-batch queue]
    subgraph D["Dense Region — concurrent active window"]
        direction LR
        N[RTX 3060 / CUDA<br/>front-stage layers] -->|current micro-batch| A[AI Max+ 395 / Vulkan<br/>rear-stage layers and state ownership]
    end
    Q --> N
    A --> O[Decode and result stream]
    N -. next micro-batch overlaps .-> A
```

The final design uses the RPC events/async capability from upstream llama.cpp [PR #18626](https://github.com/ggml-org/llama.cpp/pull/18626). Model layers stay on their assigned endpoint; micro-batches move through the stages so the two devices can work concurrently. The experiment does not claim a custom inference-engine patch.

### Terminology

- **Dense Acceleration** is this project's name for the final asynchronous layered-PD form: layer ownership is split between the external GPU and AI Max+ 395, while RPC/events keeps adjacent micro-batches active across both stages.
- The **Dense Region** is the overlapping compute window on the pipeline timeline in which the external GPU and AI Max+ 395 are both doing useful work on adjacent micro-batches and their own layer stages. Filling this window reduces pipeline bubbles and strongly accelerates the model-stage work it covers.
- The Dense Region is scheduling overlap, **not** two devices recomputing the same layer, tensor parallelism, or duplicated weight residency.

## Research evolution

Overall experimental goal: verify that when a high-compute, small-VRAM accelerator card cannot fit an entire model by itself, it can jointly host and compute the same model together with a low-compute, large-VRAM host, while accelerating both Prefill and Decode without lowering output quality.

| Stage | Experimental purpose | Validation method | Measured result |
| --- | --- | --- | --- |
| v1.0 | Establish basic feasibility: the RTX 3060 carries Prefill, then hands off the state to the AI Max+ 395 for Decode. | Complete the CUDA prefill → Vulkan decode state handoff reliably. | 1452.29 tok/s; 30.28 tok/s decode |
| v2.1 | Resolve the serial idle time at both endpoints in v1.0; check whether an async micro-batch pipeline lets both endpoints contribute to one prefill. | Verify that the same-request micro-batch pipeline keeps both layer stages active within a single request. | 1865.08 tok/s |
| v2.2 | Resolve the run-to-run overlap variance in v2.1; make the speedup repeatable rather than coincidental. | Tune the async overlap until the pipeline becomes consistent across runs. | 1893.87 tok/s |
| v2.3 | Once the pipeline is repeatable, check whether load balancing between the two backends reduces pipeline wait and improves Prefill and Decode at the same time. | The balanced split lifted decode above the v1.0 figure while raising prefill. | 1999.51 tok/s; 37.16 tok/s decode |
| **v2.4** | Complete the overall goal: compare the combined setup against both endpoints running the same 9B Q6_K, pp5064 local baseline, and confirm state and output correctness, sustained Dense Region work, and lossless Prefill/Decode acceleration. | The calibrated checkpoint keeps the Dense Region full and reaches 83.2% of the sum of both standalone prefill rates. | **2129.69 tok/s; 50.73 tok/s decode** |
| **v2.5** | After the 9B line, test whether full 3080 prefill + 395 decode can move 27B Q4 into serving. | Remove per-ubatch RPC sync and run six C1–C6 tiers across two paths, twelve groups total. | C1 prefill **1000.6 tok/s**; TTFT 1073 ms |
| **v2.7** | Surface the strongest results immediately instead of burying them under implementation detail. | Reframe 27B-C/D as prefill-first and decode-first profiles with only high-level changes and best measurements on the homepage. | Prefill up to **1210.6 tok/s**; single-stream decode up to **63.2 tok/s** |
| **v2.6** | Resolve the split 3060/3080 narrative and add the latest router measurements. | Unify the 27B experiment area into input-acceleration and generation-acceleration paths. | Prefill up to **1210.6 tok/s**; single-stream decode up to **63.2 tok/s** |
| **v3.0 (research direction)** | Adapt the v2.4 one-to-one mechanism to other types of large-memory hosts and small-VRAM accelerator cards. | Build a cross-platform adaptation matrix and validate memory-dense-region mapping, lossless prefill/decode acceleration, and scheduling stability across host architectures and accelerator models. | **Planned: adaptation research for other large-memory hosts + small-VRAM accelerator cards** |
| **v4.0 (research direction)** | Study one accelerator card accelerating X large-memory hosts at the same time. | Study one-to-many scheduling, resource isolation, fairness, fault recovery, and the scaling boundary as the number of concurrent hosts increases. | **Planned: 1 accelerator → X large-memory hosts** |

## 9B local test envelope

| Item | Configuration |
| --- | --- |
| Model | 9B, Q6_K |
| Devices | RTX 3060 12GB / CUDA + Ryzen AI Max+ 395 / Radeon 8060S / Vulkan |
| v1.0 input / output | 5064 / 128 tokens |
| v1.0 measurement | Cache disabled; second server request retained |
| v2 measurement | `pp5064`; `tg128` only where recorded |

## 9B local experiment chain — one main envelope

### New experiment 9B-v1.0 (Independent PD): 1452.29 tok/s prefill; 30.28 tok/s decode

| Configuration | TTFT | Prefill | Decode |
| --- | ---: | ---: | ---: |
| AI Max+ 395 only | 5.879 s | 861.55 tok/s | 30.24 tok/s |
| First 512 tokens on independent PD | 5.720 s | 886.62 tok/s | 30.22 tok/s |
| First 2048 tokens on independent PD | 5.077 s | 999.16 tok/s | 30.18 tok/s |
| Full independent PD | **3.496 s** | **1452.29 tok/s** | **30.28 tok/s** |

Full independent PD handed off 5063 tokens and 208.57 MiB of state. It preserved the architectural potential for cross-request prefill/decode duplexing, which this release did not benchmark under concurrency.

### New experiment 9B-v2.1 (first fused pipeline): 1865.08 tok/s prefill

One request first enters the async micro-batch pipeline, ending whole-request alternating idle time.

### New experiment 9B-v2.2 (overlap refinement): 1893.87 tok/s prefill

Overlap becomes stable across runs, showing that the gain is repeatable.

### New experiment 9B-v2.3 (balance refinement): 1999.51 tok/s prefill; 37.16 tok/s decode

Balancing the two stages lifts both prefill and decode above the prior checkpoint.

### New experiment 9B-v2.4 (Dense Acceleration final): 2129.69 tok/s prefill; 50.73 tok/s decode

#### Same-envelope 9B summary

Blank cells mean the metric was not recorded for that checkpoint.

| Checkpoint | Public description | Prefill | Decode | TTFT |
| --- | --- | ---: | ---: | ---: |
| RTX 3060 baseline | CUDA endpoint only | 1589.00 tok/s | 43.87 tok/s | — |
| AI Max+ 395 baseline | Vulkan endpoint only | 970.00 tok/s | 31.27 tok/s | — |
| v2.1 | First fused pipeline | 1865.08 tok/s | — | — |
| v2.2 | Overlap refinement | 1893.87 tok/s | — | — |
| v2.3 | Balance refinement | 1999.51 tok/s | 37.16 tok/s | — |
| **v2.4** | Dense Acceleration final checkpoint | **2129.69 tok/s** | **50.73 tok/s** | — |

v2.4 is 34.0% above the RTX 3060 prefill baseline, 119.6% above the AI Max+ 395 baseline, and 83.2% of the 2559 tok/s sum of both standalone measurements. These are calculations from the local table, not claims about other hardware.

The 37.16 tok/s fused decode result belongs to the v2.3 checkpoint, while the v2.4 checkpoint recorded a 50.73 tok/s decode.

## Unified 27B experiment area — co-located narrative, separate envelopes

These experiments share the 27B model scale, but hardware, quantization, prompts, and engine revisions differ. **Compare only inside each table.**

### New experiment 27B-A (3060·IQ3): pp4096 658.52 tok/s (+110%)

This exploratory line is deliberately separated from the `9B Q6_K / pp5064` table above. It used Qwen3.8-27B UD-IQ3_XXS (27.32B parameters, 10.17 GiB, 3.06 bpw), so its values are not included in the v2.4 percentages.

| Measurement | AI Max+ 395 only | Dense Acceleration | Observed change |
| --- | ---: | ---: | --- |
| pp4096 | 313.28 tok/s | **658.52 tok/s** | +110% |
| pp65536 | 136.69 tok/s | **319.10 tok/s** | +133% (2.33×) |
| pp98304 | Timed out after 900 s | **225.10 tok/s** | Completed; TTFT ≈ 437 s |
| Decode tg64 | 18.26 tok/s | **19.57 tok/s** | About +7% |

The full 27B model did not provide a valid standalone RTX 3060 baseline in this setup. Dense Acceleration succeeded because the external GPU only had to keep and compute its assigned layer stage; no internal allocation ratio is published. This result is an IQ3 validation, not a Q4 claim.

### New experiment 27B-B (3080·Q4, serving v1.0): C1 prefill 1000.6 tok/s; TTFT 1073 ms

This experiment replaces the accelerator head with an RTX 3080 and records 27B Q4 heterogeneous PD across two endpoints. It remains separate from the 9B main table and the 27B-A IQ3 long-prompt sweep.

#### Configuration

- **Accelerator head**: **RTX 3080 20GB** (pure CUDA0 full prefill, no RPC, no draft speculation).
- **Host being accelerated**: **AMD Ryzen AI Max+ 395 / Radeon 8060S** (Vulkan decode, full KV pool, DFlash2 speculative decoding).
- **Model**: Qwen3.8-27B, Q4_K_M (~17.66 GiB).
- **Engine**: llama.cpp fork (HEAD `18c8dde`, with upstream [PR #18626](https://github.com/ggml-org/llama.cpp/pull/18626) async/RPC capability), running as two instances: 8082 pure CUDA0 full prefill, 8081 pure 395 decode holding the full KV pool (`/dev/shm/kvx`).
- **Raw 3080 baseline**: pp1024 = 1228.53, pp4096 = 1203.06, tg64 = 33.08 tok/s.

#### v1.0 representative rows

This is **six C1–C6 concurrency tiers × split/solo paths = twelve measured groups**. The README keeps only the first and last row; all six tiers live in the detailed record and CSV.

| C | TTFT split/solo | prefill split/solo | decode split/solo | KV migration |
| --- | ---: | ---: | ---: | ---: |
| 1 | **1073 / 4825 ms** | **1000.6 / 207.2 tok/s** | 38.75 / 36.33 tok/s | 71 ms |
| 6 | 5009 / 22303 ms | 1014.3 / 44.8 tok/s | 9.67 / 8.82 tok/s | 70 ms |

Removing RPC lifted same-envelope C1 prefill from 683.2 to **1000.6 tok/s (+46%)**, about 82% of raw 3080 throughput.

### New experiment 27B-C | Prefill-first (fastest prefill): best measured **1210.6 tok/s**

- Rebalanced heterogeneous compute so the 3080 focuses on input processing while the 395 carries generation.
- Best measured prefill reaches **1210.6 tok/s**, aimed at long prompts, RAG, and document workloads where time to first token matters.

### New experiment 27B-D | Decode-first (fastest single-stream decode): best measured **63.2 tok/s**

- Reworked the decode path and task scheduling so the accelerator contributes more directly to continuous generation.
- Best measured single-stream decode reaches **63.2 tok/s**, aimed at low-concurrency chat, coding, and other interactive workloads.

See the [complete 27B record](results/qwen3.8-27b-dual-machine-pd.md) for the v1.0/v1.1/v1.2 tables, natural-language audit, and envelope limits; see the [CSV](data/qwen27b-local-results.csv) for machine-readable data.

#### Differences versus DGX Spark

| Metric | AI Max+ 395 + RTX 3080 (this experiment) | DGX Spark / GB10 |
| --- | --- | --- |
| Model/quantization | Qwen3.8-27B, Q4_K_M | Qwen3.8-27B, NVFP4 |
| Engine | llama.cpp (CUDA + Vulkan, fork `18c8dde`) | SGLang + DFlash2 (1M context) |
| KV precision | q4_0 | fp8_e4m3 |
| Prefill (cold) | ~1000–1015 tok/s (short prompt serving) | 1170 / 800 / 615 tok/s (100K / 200K / 300K cold) |
| Prefill (raw) | pp1024 = 1228.53 tok/s | Not disclosed |
| Decode (C1) | 38.75 tok/s | Not disclosed |
| TTFT (C1) | 1073 ms | Not disclosed |
| Concurrency | C1–C6 all tiers | Not disclosed |
| Topology | dual-machine heterogeneous PD (3080 prefill + 395 decode) | single machine |

**Not directly comparable**: quantization (Q4_K_M vs NVFP4), engine (llama.cpp vs SGLang), KV precision (q4_0 vs fp8), prompt depth (short prompt vs 100K–300K tokens), and topology (dual-machine heterogeneous vs single machine) all differ.

## External reference: DGX Spark community data (not a local new experiment)

These externally reported measurements are preserved for context, not ranked against the local experiment.

| Device | Community workload | Prompt metric | Prefill | Directly comparable? | Source |
| --- | --- | ---: | ---: | --- | --- |
| NVIDIA DGX Spark / GB10 | Qwen3.5 9B, TQ3_4S, fork-specific FP4 cache-on path | pp2048 | 2766.28 tok/s | **No** — quantization, prompt length, fork, and kernel differ | [llama.cpp-tq3 PR #53](https://github.com/turbo-tan/llama.cpp-tq3/pull/53) |
| NVIDIA DGX Spark / GB10 | Qwen3.8-27B, NVFP4, SGLang + DFlash2 | 100K / 200K / 300K-token cold prefill | 1170 / 800 / 615 tok/s | **No** — quantization, engine, prompt depths, and test method differ | [hasso5703 benchmark](https://github.com/hasso5703/dgx-spark-qwen38/blob/17e7e2280e632b0a3ab91839c8c7522b256937ac/BENCHMARKS.md#L232-L243) |

See the [source notes](results/dgx-spark-community-control.md) and [machine-readable external controls](data/dgx-spark-community-controls.csv).

## Findings and limits

- v1.0 demonstrates heterogeneous state handoff between CUDA prefill and Vulkan decode.
- v2.4 Dense Acceleration demonstrates real asynchronous compute overlap in a heterogeneous layer pipeline, but occupying both endpoints removes v1.0's independent-PD duplex behavior.
- v1.0 uses server-request measurements; v2 uses llama-bench pp/tg. Cross-release percentages are engineering references, not strict same-method research claims.
- 27B-A validates the 3060 / IQ3 long-prompt Dense Region, while 27B-B/C/D cover 3080 / Q4 C1–C6 serving, router aggregate throughput, and two-stage preemption. Their envelopes remain distinct and cannot be collapsed into one ranking.
- **1200+ is explicitly 3080 prefill, not decode.** The 35–38.5 tok/s figure is a repetitive-text, 100%-acceptance speculative upper bound; direct natural-language C1 measured 12.1 tok/s. Both are retained to prevent selective reporting.
- Current 27B Q4 evidence covers about 1000-token concurrent stress plus 4k/7k long-prompt points. It cannot be extrapolated to 100K+ prompts, arbitrary text distributions, or a permanently retained fast 3080 decode lane.
- Community controls retain their original public methodology and are not normalized or extrapolated.

Detailed records: [v1.0 independent PD](results/v1.0-independent-pd.md), [v2.4 Dense Acceleration evolution](results/v2.4-fused-layer-pipeline.md), [complete 27B record](results/qwen3.8-27b-dual-machine-pd.md), [9B local CSV](data/benchmark-results.csv), [27B local CSV](data/qwen27b-local-results.csv), and [changelog](CHANGELOG.md).
