# <img src="assets/soulmate-spirit.png" alt="Soulmate spirit" width="44" align="absmiddle"> Dense Acceleration · Heterogeneous GPU PD Lab

[中文](README_ZH.md)

**Put a small-VRAM card and a large-memory host to work together.**

- **9B: pair Prefill is 34.0% faster than the 3060 alone and 119.6% faster than the 395 alone.**
- **27B: 3080 + DGX Spark Prefill is 44.03% faster than this 3080 standalone.** 8K input / 128 output, no speculation.
- **27B long context: pair Prefill at 64K reaches 2.33 times the 395 alone; 98K goes from timeout to completion.**

A small-VRAM NVIDIA card (RTX 3060 12GB / RTX 3080 20GB) plus a large-memory partner can run large-model inference together. Three routes: RTX 3060 12GB + AMD AI Max+ 395, RTX 3080 20GB + 395, and RTX 3080 20GB + DGX Spark GB10. **Dense Acceleration** means both devices compute the same stage of the same model at once. Current release **v1.6**, seven public milestones, plus a 2026-09-11 local Spark addendum that is not a new version. Deployment commands, patches, endpoints, and the layer-allocation policy stay private.

C1–C6 means 1 to 6 concurrent requests; C1 is single-stream. Aggregate is the combined speed of the streams that are running together. Speeds are tok/s unless a row names TTFT.

## Direct comparison: single card, single host, pair

On 9B, the 3060 + 395 pair is the local cell with both a single-card and a single-host control, and the pair is faster than both. On 27B Q4_K_M, DGX Spark is measured only as a 3080 + Spark pair versus this 3080; Spark standalone is **not tested** (not invented).

### 9B Dense Acceleration · Ornith 9B · Q6_K · RTX 3060 12GB + 395

`llama-bench` pp5064 / tg128. Data: [benchmark-results.csv](data/benchmark-results.csv). Record: [v2.4 fused layer pipeline](results/v2.4-fused-layer-pipeline.md). Lift = pair ÷ denominator − 1.

| Metric | 3060 card | 395 host | 3060 + 395 pair | vs 3060 | vs 395 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prefill (tok/s) | 1589.00 | 970.00 | **2129.69** | **+34.0%** | **+119.6%** |
| Decode (tok/s) | 43.87 | 31.27 | **50.73** | **+15.6%** | **+62.2%** |

### DGX Spark · 27B Q4_K_M · RTX 3080 20GB + Spark GB10

Highest Prefill and highest Decode are **different working profiles**. In the Decode-first profile, the pair and the standalone card each use separately tuned speculation settings. The tables compare each profile and input length separately. Data: [qwen27b-spark-3080-profiles.csv](data/qwen27b-spark-3080-profiles.csv). Record: [Spark + 3080 profiles](results/qwen3.8-27b-spark-3080-profiles.md).

**Highest Prefill confirmed — 8K in / 128 out, no speculation** (median of three measured runs after warmup; warmup excluded). Retest of the same profile: 1601.05 / 17.41.

| Metric | 3080 standalone | Spark standalone | 3080 + Spark | vs 3080 |
| --- | ---: | ---: | ---: | ---: |
| Prefill (tok/s) | 1113.13 | not tested | **1603.20** | **+44.03%** |
| Decode (tok/s) | 33.36 | not tested | 17.62 | **-47.18%** |

**Decode-first, DFlash2 — 2K in / 128 out** (3080 standalone uses its own DFlash2 tuning; not the Prefill-confirmed profile).

| Metric | 3080 standalone | Spark standalone | 3080 + Spark | vs 3080 |
| --- | ---: | ---: | ---: | ---: |
| Prefill (tok/s) | 1042.97 | not tested | 1153.95 | **+10.64%** |
| Decode (tok/s) | 60.37 | not tested | **63.97** | **+5.96%** |

**Decode-first, DFlash2 — 8K in / 128 out** (same Decode-confirmed profile as the 2K row; 2K and 8K are not mixed).

| Metric | 3080 standalone | Spark standalone | 3080 + Spark | vs 3080 |
| --- | ---: | ---: | ---: | ---: |
| Prefill (tok/s) | 1015.15 | not tested | 1124.32 | **+10.75%** |
| Decode (tok/s) | 58.36 | not tested | 49.20 | **-15.70%** |

Balanced serving C1–C6 has **no** 3080 standalone and **no** Spark standalone concurrency control, so this page does not claim a pair gain for those rows. Full C1–C6 tables and timing definitions are under [27B-SPARK-01](#27b-spark-01--qwen38-27b--q4_k_m--rtx-3080-20gb--dgx-spark-gb10).

## Results map

Each row is one experiment: short headline, then the record. Full figures sit in [Experiment data](#experiment-data).

| Approach | Experiment | Headline | Record |
| --- | --- | --- | --- |
| Dense Acceleration | 9B-PIPE-01 · Ornith 9B · Q6_K · 3060 + 395 | Pair Prefill **2129.69** / Decode **50.73** tok/s; beats 3060 and 395 | [Record](results/v2.4-fused-layer-pipeline.md) |
| Layer-split loading | 27B-LONG-01 · Qwen3.8-27B · UD-IQ3_XXS · 3060 + 395 | vs 395: pp4096 **658.52** vs 313.28 (**+110.2%**) | [Record](results/qwen3.8-27b-dual-machine-pd.md) |
| Phase-separated PD | 9B-PD-01 · Ornith 9B · Q6_K · 3060 + 395 | vs 395 serving: TTFT **3.496 s** vs 5.879 s (**-40.5%**) | [Record](results/v1.0-independent-pd.md) |
| Phase-separated PD | 27B-PD-01 · Qwen3.8-27B · Q4_K_M · 3080 + 395 | vs 395 serving C1: TTFT **1073 ms** vs 4825 ms; Prefill **1000.6** vs 207.2 (4.83×) | [Record](results/qwen3.8-27b-dual-machine-pd.md) |
| Remote KV pool | 27B-KV-01 · Qwen3.8-27B · Q4_K_M · 3080 + 395 | C1 → C6 aggregate Decode: C (395 Decode) 33.55 → 63.84; D (3080 Decode) 63.2 → 116.3 tok/s | [Record](results/qwen3.8-27b-dual-machine-pd.md) |
| MoE PD | ORNITH-PD-01 · Ornith-1.5-35B-A3B · IQ4_XS · 3080 + 395 | 100K, C1 → C6: 395 aggregate Decode 23.33 → 148.20 (6.35×); 42/42 | [Record](results/ornith-1.5-35b-a3b-dual-machine-pd.md) |
| MoE fused draft | ORNITH-PD-02 · Ornith-1.5-35B-A3B · IQ4_XS · 3080 + 395 | 3080 Prefill **4173.47**; 395 + DFlash single-stream Decode **114.86** tok/s; 107/114 (93.86%) | [Record](results/ornith-1.5-35b-a3b-fused-dflash-pd.md) |
| Single-server split | FLASH-SPLIT-01 · Qwen3.8-Flash · Q4 · 3080 + 395 | Best C4: Prefill 633.685, Decode 71.185, total 338.270 tok/s | [Record](results/qwen3.8-flash-q4-layer-split.md) |
| Dual-device pair | 27B-SPARK-01 · Qwen3.8-27B · Q4_K_M · 3080 + Spark | Prefill **1603.20** vs 3080 1113.13 (**+44.03%**); 2K Decode **63.97** | [Record](results/qwen3.8-27b-spark-3080-profiles.md) |
| External reference | EXT-DGX-01 · Qwen3.5 9B / TQ3_4S; Qwen3.8-27B / NVFP4 | Community Spark figures, background only | [Record](results/dgx-spark-community-control.md) |

## Why Dense Acceleration is this repository's focus

Three advantages, one bound.

1. In this repository, Dense Acceleration is the 9B cell where the pair beats both standalones. Prefill 2129.69 vs 3060 1589.00 (+34.0%, ÷ 1589.00) and vs 395 970.00 (+119.6%, ÷ 970.00); Decode 50.73 vs 43.87 (+15.6%) and vs 31.27 (+62.2%). The extra is real 395 compute. A second local dense pair is RTX 3080 20GB + DGX Spark GB10, Qwen3.8-27B Q4_K_M: Prefill-confirmed 8K/128 no speculation, pair Prefill **1603.20** vs this 3080 1113.13 (**+44.03%**, ÷ 1113.13). Only 9B has both standalone baselines; Spark has no Spark standalone baseline, so +44.03% is not a win over every standalone device.

2. The ceiling of phase-separated PD is the accelerator card's own raw compute. 27B-PD-01 pair Prefill 1000.6 reaches 82% of the 3080 llama-bench raw 1228.53, and 1210.6 with the checkpoint off is 98.5% of that ceiling. Those raw and serving figures are **not the same method**. Decode barely moves under PD (9B: 30.28 vs 30.24; 27B: 38.75 vs 36.33). PD buys TTFT and capacity; it does not add the two devices' compute together.

3. Dense Acceleration does not require the accelerator card to hold the whole model. Layer-split loading in 27B-LONG-01: pp4096 658.52 vs 395 313.28, pp65536 319.10 vs 136.69, pp98304 from a 900 s timeout to 225.10.

4. Bound: a full single-card plus single-host pair of controls exists only for the 9B Q6_K + RTX 3060 cell (D1). The 27B Spark pair has a matched 3080 standalone at the confirmed Prefill profile, but no Spark standalone baseline. Dense overlapping-pipeline controls on the 395 host for 27B and MoE remain on the roadmap. The dense region needs neighbouring micro-batches to run offset (an overlap window); whatever cannot enter the window belongs to the sparse region, which only has to hold the model.

## How the system works

The problem first, then the method.

A consumer card computes fast, but its VRAM is small and a larger model does not fit; a large-memory host holds any model, but computes slowly. The fix is to have both devices compute the same stage of the same model at the same time; this is **Dense Acceleration**. The accelerator card holds the front-stage layers and computes them; the large-memory host holds the rear-stage layers and the intermediate state, and computes that half. The point of the **dense region** is timing: neighbouring micro-batches run offset, so at any moment both devices have work. Whatever cannot fit into the overlap window belongs to the **sparse region**: hold the whole model and carry the context capacity.

![Base architecture: a prompt enters the async micro-batch queue, the small-VRAM card computes the front-stage layers, the large-memory host computes the rear-stage layers and state, and inside the Dense Region both devices work on neighbouring micro-batches at the same time](assets/base-architecture.png)

<details>
<summary>The mermaid source for this diagram (paste it into mermaid.live to view)</summary>

```text
flowchart LR
    P["Prompt"] --> Q["Async micro-batch queue"]
    subgraph D["Dense Region — concurrent active window"]
        direction LR
        N["Small-VRAM accelerator<br/>front-stage layers"] -->|"current micro-batch"| A["Large-memory host<br/>rear-stage layers and state"]
    end
    Q --> N
    A --> O["Decode and result stream"]
    N -.->|"next micro-batch overlaps"| A
```

</details>

What this is not: it is not running layers one after another, not tensor parallelism, not two devices computing the same layer twice, and not the kind of independent PD that uses the 395 as a remote KV store.

![Dense Acceleration structure](assets/dense-region-structure.png)

## How the devices divide work

395-host experiments use the five splits below. RTX 3080 20GB + DGX Spark GB10 is the same pair at three working profiles. Short result, then the record.

| Division of work | Accelerator card | Partner device | Short result | Experiment |
| --- | --- | --- | --- | --- |
| Dense Acceleration | Front-stage layers, same stage as the host | AI Max+ 395 rear-stage layers and state | Pair beats 3060 and 395: Prefill 2129.69 tok/s | [9B-PIPE-01](results/v2.4-fused-layer-pipeline.md) |
| Layer-split loading | Holds and computes only the front-stage layers | AI Max+ 395 holds the rear-stage layers | vs 395: pp4096 658.52 vs 313.28 (**+110.2%**) | [27B-LONG-01](results/qwen3.8-27b-dual-machine-pd.md) |
| Phase-separated PD | Does all Prefill | AI Max+ 395 does Decode | Faster first token; Decode barely moves | [9B-PD-01](results/v1.0-independent-pd.md), [27B-PD-01](results/qwen3.8-27b-dual-machine-pd.md), [ORNITH-PD-01](results/ornith-1.5-35b-a3b-dual-machine-pd.md), [ORNITH-PD-02](results/ornith-1.5-35b-a3b-fused-dflash-pd.md) |
| Remote KV pool | Prefill; configuration D, the 3080 decodes | AI Max+ 395 remote KV; configuration C, the 395 decodes | C Decode 33.55 → 63.84; D 63.2 → 116.3 | [27B-KV-01](results/qwen3.8-27b-dual-machine-pd.md) |
| Single-server split | Tensor split 0.38 inside one llama-server | AI Max+ 395 tensor split 0.62 | C4 Prefill 633.685, Decode 71.185 tok/s | [FLASH-SPLIT-01](results/qwen3.8-flash-q4-layer-split.md) |
| Prefill-first, no speculation | Computes the same model together with Spark | DGX Spark GB10 with the 3080 | Prefill **1603.20** vs 1113.13 (**+44.03%**); Decode 17.62 vs 33.36 | [27B-SPARK-01](results/qwen3.8-27b-spark-3080-profiles.md) |
| Decode-first, DFlash2 | Computes the same model together with Spark | DGX Spark GB10 with the 3080 | 2K Decode **63.97** vs 60.37; 8K 49.20 vs 58.36 | [27B-SPARK-01](results/qwen3.8-27b-spark-3080-profiles.md) |
| Balanced serving, DFlash2 | Computes the same model together with Spark | DGX Spark GB10 with the 3080 | C1–C6 aggregate Prefill 1086.44–1097.40, Decode 24.77–47.69; 126/126; no standalone C1–C6 | [27B-SPARK-01](results/qwen3.8-27b-spark-3080-profiles.md) |

Only Dense Acceleration is both devices computing the same stage at once in the 395 matrix. The Spark pair also has both devices computing the same model together; its Prefill gain is versus this 3080 standalone only, and there is no Spark standalone baseline.

## Three hardware routes and the six experiment cells

This is an experiment coverage map across three hardware routes, not a unified ranking under one load. Route 1 is RTX 3060 12GB + AMD AI Max+ 395. Route 2 is RTX 3080 20GB + 395. Route 3 is RTX 3080 20GB + DGX Spark GB10. The six cells split by how much VRAM the running model needs versus the accelerator's VRAM into fits easily, fills the card, and does not fit; each 395-host cell targets five configurations: RTX 3060 alone, RTX 3080 alone, AI Max+ 395 alone, 3060 + 395, and 3080 + 395. Within a cell the model, quantization, prompt, context, concurrency, and metrics must match; a card that cannot hold the model is itself a valid result, and a smaller model or a harsher quantization may not be substituted to manufacture a baseline. The repository holds no RTX 3090 measurements at all; that card was ruled out during selection.

| Item | Route 1: RTX 3060 12GB + 395 | Route 2: RTX 3080 20GB + 395 | Route 3: RTX 3080 20GB + DGX Spark |
| --- | --- | --- | --- |
| Releases covered | v1.0 → v1.1 | v1.2 → v1.6 | 2026-09-11 addendum (still v1.6) |
| VRAM and what fits | 12GB, only the 9B dense tier fits; 27B fits only as an IQ3 layer split | 20GB, 27B Q4 fits, and that is what makes the MoE layer splits possible | 20GB 3080 plus Spark GB10; 27B Q4 on this pair |
| Models run | Ornith 9B dense Q6_K; Qwen3.8-27B IQ3 | Qwen3.8-27B dense Q4; Ornith-1.5-35B-A3B and Qwen3.8-Flash | Qwen3.8-27B dense Q4_K_M |
| Experiment IDs | 9B-PD-01, 9B-PIPE-01, 27B-LONG-01 | 27B-PD-01, 27B-KV-01, 27B-DRAFT-AUDIT-01, ORNITH-PD-01, ORNITH-PD-02, FLASH-SPLIT-01 | 27B-SPARK-01 |
| Headline | Pair Prefill 2129.69 tok/s vs 3060 1589.00 / 395 970.00; 27B-LONG vs 395 **+110.2%** at pp4096 | Serving PD TTFT 1073 ms vs 395 4825 ms; remote KV and MoE C1–C6 | Prefill **1603.20** vs 3080 1113.13 (**+44.03%**); 2K Decode **63.97**; C1–C6 126/126; Spark standalone not tested |

Spark cells with no local measurement are marked not yet tested; 395 figures are not copied into those Spark cells, and community NVFP4 rows are not used as Spark-route evidence.

| Cell | Question | 395 route measured | Spark route measured |
| --- | --- | --- | --- |
| **D1 Dense · fits easily**<br>Ornith 9B · Q6_K · RTX 3060 | With the model held comfortably on the card, can a dense pipeline actually beat the faster card, rather than just add capacity? | **Verified**: 9B-PIPE-01 has both the 3060 single-card and the 395 single-host controls, and the pair beats both. [Record](results/v2.4-fused-layer-pipeline.md) | Not yet tested locally. |
| **D2 Dense · fills the card**<br>Qwen3.8-27B · Q4_K_M · RTX 3080 | With VRAM nearly full, which wins: the whole model on one card, separated phases, or a layered dense route? | **Verified**: 27B-PD-01, 3080 Prefill + 395 Decode while serving; TTFT vs 395 drops more than three quarters; C1–C6 all pass. [Record](results/qwen3.8-27b-dual-machine-pd.md) | **27B-SPARK-01**: Prefill **1603.20** vs 3080 1113.13 (**+44.03%**); Decode-first 2K **63.97**; balanced C1–C6; Spark standalone not tested. [Record](results/qwen3.8-27b-spark-3080-profiles.md) |
| **D3 Dense · does not fit**<br>Qwen3.8-27B · UD-IQ3_XXS · RTX 3060 | When one card cannot finish the job, can splitting the model finish it and still beat the 395? | **Verified**: 27B-LONG-01 vs 395 **+110.2%** / **+133.4%** at 4K / 64K; 98K timeout → 225.10. [Record](results/qwen3.8-27b-dual-machine-pd.md) | Not yet tested locally. |
| **M1 MoE · fits easily**<br>model and quantization TBD | With few active parameters and VRAM to spare, does MoE routing overhead eat back the time the overlap saves? | Planned (see Roadmap). | Not yet tested locally. |
| **M2 MoE · fills the card**<br>Ornith-1.5-35B-A3B · IQ4_XS · RTX 3080 | With MoE nearly filling the 3080, is phase separation stable, and would a dense overlap on top add anything? | **Verified**: ORNITH-PD-01 42/42, 100K Decode 23.33 → 148.20 ([record](results/ornith-1.5-35b-a3b-dual-machine-pd.md)); ORNITH-PD-02 Prefill 4173.47, Decode 114.86 ([record](results/ornith-1.5-35b-a3b-fused-dflash-pd.md)). | Not yet tested locally. |
| **M3 MoE · does not fit**<br>Qwen3.8-Flash · Q4 · RTX 3080 pilot | When the total footprint exceeds both control cards, can splitting by layer or by expert keep it running, keep throughput, and keep the output correct? | FLASH-SPLIT-01 pilot: C4 Prefill 633.685, Decode 71.185 tok/s. Full experiment planned. [Pilot record](results/qwen3.8-flash-q4-layer-split.md) | Not yet tested locally. |

## Experiment data

One small table per experiment; this is the only place on the front page that holds complete figures. Figures are copied from `results/` and `data/`; if this page and a record disagree, the record and the CSV win. A gain is "pair result ÷ control result − 1", and TTFT is written as how much it dropped. llama-bench raw compute and serving end-to-end are not raced as one method.

**Dense Acceleration.** 9B-PIPE-01 has both a 3060 single-card control and a 395 single-host control. 27B-SPARK-01 has a matched 3080 standalone at the confirmed Prefill profile and no Spark standalone baseline. Highest Prefill is +44.03% versus this 3080 standalone only.

### 9B-PIPE-01 · Ornith 9B · Q6_K · asynchronous layered Dense Acceleration

Accelerator RTX 3060 12GB; model Ornith 9B, weight quantization Q6_K (the records and CSV label this tier "9B · Q6_K"), `llama-bench` pp5064 / tg128; data from [benchmark-results.csv](data/benchmark-results.csv).

| Metric | 3060 card | 395 host | 3060 + 395 pair | vs 3060 | vs 395 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prefill (tok/s) | 1589.00 | 970.00 | **2129.69** | **+34.0%** | **+119.6%** |
| Decode (tok/s) | 43.87 | 31.27 | **50.73** | **+15.6%** | **+62.2%** |

What it verifies: both devices are genuinely working inside the dense region, and the pair beats the fastest single card present. This is the core Dense Acceleration evidence in the repository.

### 27B-SPARK-01 · Qwen3.8-27B · Q4_K_M · RTX 3080 20GB + DGX Spark GB10

Local measurement on the RTX 3080 20GB + DGX Spark GB10 route, listed on the D2 coverage map. Model Qwen3.8-27B, weight quantization Q4_K_M, KV cache q4_0. "Highest" means the working profile confirmed after this search, not a global optimum. Pair scores are combination scores, not 3080-only. Data from [qwen27b-spark-3080-profiles.csv](data/qwen27b-spark-3080-profiles.csv). Record: [qwen3.8-27b-spark-3080-profiles.md](results/qwen3.8-27b-spark-3080-profiles.md).

The three confirmed comparison tables are at the top of this page (Prefill-confirmed 8K/128 no speculation; Decode-first DFlash2 2K/128; Decode-first DFlash2 8K/128). Speculation settings on the Decode-confirmed control were tuned on the 3080 standalone and are not a one-variable change. A later live retest of the Prefill-confirmed profile, three repeats at 8K, is 1601.05 / 17.41. Do not quote 1700.

**Throughput, RTX 3080 20GB + Spark balanced C1–C6**

Speeds below are in tok/s. Prefill comes from separate Prefill waves; Decode and end-to-end output come from complete-request waves generating 256 tokens. The two wave types are summarized by concurrency tier. See the profile record for the timing definitions. No standalone C1–C6 control — no pair-gain column.

| C | Aggregate Prefill (tok/s) | Aggregate Decode (tok/s) | End-to-end output (tok/s) |
| --- | ---: | ---: | ---: |
| C1 | 1097.40 | 47.69 | 19.91 |
| C2 | 1094.21 | 31.18 | 19.49 |
| C3 | 1089.37 | 28.04 | 20.47 |
| C4 | 1086.44 | 24.90 | 19.86 |
| C5 | 1087.57 | 24.77 | 20.53 |
| C6 | 1088.06 | 25.55 | 21.62 |

**Latency, RTX 3080 20GB + Spark balanced C1–C6**

| C | TTFT p95 (s) | Batch time (s) |
| --- | ---: | ---: |
| C1 | 7.489 | 12.857 |
| C2 | 19.145 | 26.273 |
| C3 | 28.990 | 37.524 |
| C4 | 38.932 | 51.564 |
| C5 | 48.471 | 62.336 |
| C6 | 58.317 | 71.048 |

36 official waves, 126/126 official requests. What it verifies: pair Prefill at the confirmed Prefill profile is **+44.03%** versus this 3080 standalone. It does not claim a win over every standalone device, and it does not claim a C1–C6 serving gain.

**It runs even when it does not fit.** 27B-LONG-01 is layer-split loading: a dense model split by layer across two devices, to solve "does not fit". It is not evidence of overlapping micro-batch pipelining.

### 27B-LONG-01 · Qwen3.8-27B · UD-IQ3_XXS · model split across layers

Accelerator RTX 3060 12GB; model Qwen3.8-27B, weight quantization UD-IQ3_XXS; pp4096 / pp65536 / pp98304 / tg64, all in tok/s; data from [qwen27b-local-results.csv](data/qwen27b-local-results.csv). The 3060 alone cannot hold the whole 27B, which is exactly why the model is split across layers. llama-bench here; not a serving end-to-end race against 27B-PD-01.

| Metric | 3060 card | 395 host | 3060 + 395 pair | vs 395 |
| --- | --- | ---: | ---: | ---: |
| pp4096 (tok/s) | cannot hold | 313.28 | **658.52** | **+110.2%** |
| pp65536 (tok/s) | cannot hold | 136.69 | **319.10** | **+133.4%** |
| pp98304 (tok/s) | cannot hold | timed out at 900 s | **225.10** | timeout → finished (no %) |
| tg64 (tok/s) | cannot hold | 18.26 | **19.57** | **+7.2%** |

What it verifies: when the small card cannot hold the model, the layered split finishes the 98K prompt that the 395 alone could not finish, and at 4K and 64K its Prefill runs more than twice as fast as the 395 alone.

**Phase-separated PD.** 9B-PD-01, 27B-PD-01, ORNITH-PD-01, and ORNITH-PD-02 are phase-separated PD: the two stages still run one after the other.

### 9B-PD-01 · Ornith 9B · Q6_K · independent PD

Accelerator RTX 3060 12GB; model Ornith 9B, weight quantization Q6_K, serving, 5064 in / 128 out; data from [benchmark-results.csv](data/benchmark-results.csv). Host baseline is 395 **serving** (861.55 / 30.24), not the llama-bench 970.00 / 31.27 used by 9B-PIPE-01.

| Metric | 395 host (serving) | 3060 Prefill + 395 Decode | vs 395 serving |
| --- | ---: | ---: | ---: |
| TTFT | 5.879 s | **3.496 s** | **-40.5%** |
| Prefill (tok/s) | 861.55 | **1452.29** | **+68.6%** |
| Decode (tok/s) | 30.24 | **30.28** | **+0.1%** |

What it verifies: the state computed on the CUDA side hands over in one piece to the Vulkan side for Decode, the first token comes sooner, and Decode holds. The two phases still run one after the other, so this is phase separation, a different route from Dense Acceleration.

### 27B-PD-01 · Qwen3.8-27B · Q4_K_M · independent PD (serving)

Accelerator RTX 3080 20GB; model Qwen3.8-27B, weight quantization Q4_K_M, KV cache q4_0; all six concurrency tiers C1–C6 passed, tier C1 shown; data from [qwen27b-local-results.csv](data/qwen27b-local-results.csv). Host baseline is 395 serving C1 under the v1.0 method (207.2 tok/s). The 3080 llama-bench raw (pp1024 1228.53, pp4096 1203.06, tg64 33.08) is a **different method** from this serving end-to-end; pair Prefill 1000.6 is 82% of that raw ceiling, not a same-method race.

| Metric | 395 host (serving, v1.0) | 3080 Prefill + 395 Decode, C1 | vs 395 serving |
| --- | ---: | ---: | ---: |
| TTFT | 4825 ms | **1073 ms** | **-77.8%** |
| Prefill (tok/s) | 207.2 | **1000.6** | **+382.9%** (4.83×) |
| Decode (tok/s) | 36.33 | **38.75** | **+6.7%** |

What it verifies: phase separation works while serving, Prefill does not fall off as concurrency rises, and Decode is unharmed. It verifies that the phases can be separated, not that both devices compute the same phase at once.

### ORNITH-PD-01 · Ornith-1.5-35B-A3B · IQ4_XS · MoE PD stress test

The RTX 3080 20GB takes all Prefill and the AI Max+ 395 all Decode; main model Ornith-1.5-35B-A3B with IQ4_XS weight quantization; draft head Qwen3.6-35B-A3B-DFlash with Q4_K_M weight quantization; data from [ornith35a3b-local-results.csv](data/ornith35a3b-local-results.csv). No matched single-host speed control — lifts below are C1 → C6 on the same pair, not pair vs 395.

| Metric | C1 (1 concurrent stream) | C6 (6 concurrent streams) | Change |
| --- | ---: | ---: | ---: |
| 3080 Prefill aggregate, 1000 input tokens (tok/s) | 4017.46 | 3943.88 | **-1.8%** |
| 3080 Prefill aggregate, 100K input (tok/s) | 2895.53 | 2793.24 | **-3.5%** |
| 395 Decode aggregate, 100K input (tok/s) | 23.33 | 148.20 | 6.35× |

The same stress run covered two workloads: 1000 in / 128 out, and 100000 in / 128 out; each workload ran 1 to 6 concurrent streams, 1+2+3+4+5+6 = 21 requests, 42/42 succeeded in total, `route=pd`, `n_reuse=0`. The 1000-input tier did not separately record the 395 Decode rate, so the table only has Decode for the 100K tier. What it verifies: MoE PD runs stably at 100K context under 1 to 6 concurrent streams, with Prefill attributed to the 3080 and Decode to the 395.

### ORNITH-PD-02 · Ornith-1.5-35B-A3B · IQ4_XS · PD with a fused DFlash draft head

The same model on the same pair of devices, with two additions on top of ORNITH-PD-01's independent PD: Decode runs fused DFlash speculative decode (`--spec-type draft-dflash`, draft length 6), and the six slots share one unified KV pool (`--kv-unified`). Main model Ornith-1.5-35B-A3B with IQ4_XS weight quantization; draft head Qwen3.6-35B-A3B-DFlash with Q4_K_M weight quantization; KV cache q4_0; workload 1000 in / 128 out, single stream; data from [ornith35a3b-local-results.csv](data/ornith35a3b-local-results.csv). No matched single-host speed control.

| Configuration | Prefill (tok/s) | Single-stream Decode (tok/s) | Draft acceptance |
| --- | ---: | ---: | ---: |
| Fused recipe, Prefill batch 4096 | **4173.47** | **114.86** | 93.86% |
| Fused recipe, batch raised to 8196, context 128K | **4123.15** | **114.42** | not recorded |
| Same recipe, a prompt the draft head could not predict | 3665.3 | 37.2 | 9.4% |

What it verifies: this 3080 + 395 pair measured **4173.47** Prefill and **114.86** single-stream Decode. A different prompt produced 37.2 Decode; the 3.09-fold difference shows sensitivity to the input and draft acceptance, not a same-prompt speedup from enabling the draft head. Draft acceptance on the main row is **107/114 (93.86%)**. [Record](results/ornith-1.5-35b-a3b-fused-dflash-pd.md)

**Capacity and serving tiers.** 27B-KV-01 is a remote KV pool with two decode routes; FLASH-SPLIT-01 is a single-server split across both devices. Neither has a matched latest standalone speed control for a pair-vs-host claim.

### 27B-KV-01 · Qwen3.8-27B · Q4_K_M · Remote KV pool with two decode routes

Accelerator RTX 3080 20GB; model Qwen3.8-27B, weight quantization Q4_K_M, KV cache q4_0; data from [qwen27b-local-results.csv](data/qwen27b-local-results.csv).

| Configuration | 3080 Prefill (tok/s) | Aggregate Decode · C1 (tok/s) | Aggregate Decode · C6 (tok/s) | C1 → C6 |
| --- | --- | ---: | ---: | ---: |
| Configuration C · 395 decodes | 1194.4–1210.6 | 33.55 | 63.84 | **+90.3%** |
| Configuration D · 3080 decodes | 1077–1090 | 63.2 | 116.3 | **+84.0%** |

1M context per stream. Configuration C: the 395 decodes; configuration D: the 3080 decodes. On both routes the 395 never runs Prefill and the dense weight compute always stays on the 3080. What it verifies: take C and let the 395 decode when Prefill is the constraint; take D and let the 3080 decode when total Decode throughput is the constraint. This is a capacity and serving route, filed apart from Dense Acceleration.

### FLASH-SPLIT-01 · Qwen3.8-Flash · Q4 · one server split across both devices

A single llama-server using the RTX 3080 20GB and the AI Max+ 395 together; model Qwen3.8-Flash, weight quantization Q4, KV cache q4_0; tensor split 0.38 / 0.62, ubatch 1024 / batch 4096, 6 slots at 131072 context; 3080 VRAM peaked at 19129 MiB; data from [qwen38flash-q4-local-results.csv](data/qwen38flash-q4-local-results.csv). No standalone control — lifts are C1 → C4 on this pair.

| Metric | C1 (tok/s) | C4 best tier (tok/s) | C1 → C4 |
| --- | ---: | ---: | ---: |
| Prefill | 569.892 | 633.685 | **+11.2%** |
| Aggregate Decode | 35.204 | 71.185 | **+102.2%** |
| Total throughput | 213.581 | 338.270 | **+58.4%** |

C1–C6 aggregate Prefill ran 569.892–633.685 tok/s and aggregate Decode 35.204–71.185 tok/s. 21/21 scored requests succeeded on a workload of about 2077 in / 256 out. What it verifies: the best tier for this configuration is C4, and C5 and C6 no longer rise. It is an operating point, and a different workload must be measured again.

**Audit and external reference.** 27B-DRAFT-AUDIT-01 is a data audit; EXT-DGX-01 is an external reference, background only. Local Spark Q4_K_M rows above are not merged with those NVFP4 community figures.

### 27B-DRAFT-AUDIT-01 · Qwen3.8-27B · Q4_K_M · speculative-decode audit

Model Qwen3.8-27B, weight quantization Q4_K_M, KV cache q4_0; a data audit run on the AI Max+ 395; data from [qwen27b-local-results.csv](data/qwen27b-local-results.csv).

| Text type | Decode (tok/s) | Acceptance |
| --- | --- | --- |
| Repetitive text | 35.0–38.5 | 100% |
| Natural language, C1 | 12.1 | 17.7% |

What it verifies: repetitive text and natural language are different workloads. Speculative Decode is sensitive to the input, so a repetitive-text high score does not represent natural-language performance.

### EXT-DGX-01 · Qwen3.5 9B / TQ3_4S; Qwen3.8-27B / NVFP4 · DGX Spark external reference

This entry contains two external workloads: Qwen3.5 9B with TQ3_4S weight quantization, and Qwen3.8-27B with NVFP4 weights and an FP8 KV cache. Public figures are about 1000 tok/s Prefill, 25–30 tok/s single-stream Decode, and 107 tok/s aggregate Decode, C1–C6. These are public results from someone else's machine, kept as background and never ranked against local data, including the local 27B-SPARK-01 Q4_K_M profiles. [Record](results/dgx-spark-community-control.md)

<details>
<summary>Number notes: why the same machine has several different figures</summary>

- The 395 has two 9B figures: the `llama-bench` figure (970.00 / 31.27) is the control for 9B-PIPE-01 and the serving figure (861.55 / 30.24) is the control for 9B-PD-01.
- The 395 also has three Prefill figures at the 27B tier. **207.2** is the C1 solo result under the v1.0 serving method, where every prompt also copied out a 524 MiB recurrent-state checkpoint and the Vulkan read-back took about 0.8 s; it is the control for the 1000.6 of 27B-PD-01. **307.1** is the C1 solo result on the same machines after that checkpoint copy was switched off (the v1.1 method); it is kept as the historical control in the configuration C table of the 27B-KV-01 record. **313.28** is `llama-bench` pp4096 on the IQ3 quantization and is the control for 27B-LONG-01. 207.2 is not a typo for 1207.2.
- The 3080 Prefill chain: 683.2 (per-ubatch RPC sync) → 1000.6 (direct CUDA, +46%) → 1210.6 (checkpoint copy off, see 27B-KV-01 configuration C, 98.5% of the raw ceiling 1228.53). The 1000.6 of 27B-PD-01 is 82% of raw compute; serving Prefill holds 1000–1015 tok/s as concurrency rises; KV transfer takes 68–76 ms (the serving record also states 71 ms). After both sides rose, the pair stays around 4× the 395 alone.
- At the 27B tier on the 395-host line, any Prefill above 1200 belongs to the RTX 3080 side (raw 1228.53, serving 1194.4–1210.6); the 395 alone runs a dense 27B model at 200–320 Prefill, and even a 9B model at only 861.55–970.00. The 1603.20 Prefill of 27B-SPARK-01 is a 3080 + Spark **pair** score, not a 3080-only figure; the matched 3080 standalone at that 8K no-speculation load is 1113.13.
- Configuration D reads 1090 / 1081 / 1080 / 1077 / 1082 / 1082 tok/s from C1 to C6, and the aggregate figures from the same run fall from 1079 to 1016 tok/s. The VRAM and compute cost of configuration D: the draft head weights occupy 1080 MiB and the verification batch needs roughly 500 MiB more of compute buffer, so at ctx8192 ubatch has to drop from 1024 to 512 and the slot count from 2 to 1, and the per-step verification matmul is real work (+21 ms on a 4-token batch, +49 ms on an 8-token batch). Prefill therefore falls from 1210.6 to 1077–1090, about ten percent lower.
- In configuration C the 3080 still keeps 2 slots, and the 395 runs a DFlash head at 38.75 tok/s single-stream while holding the full KV pool. Measured single-stream on the 3080 with the head: 42.7 tok/s on natural language and 67.4 tok/s on code, 2.2–2.3 times the 395 running the same head, which lifts aggregate Decode from 63.84 to 116.3.
- The four fused-pipeline checkpoints climbed Prefill **1865.08 → 1893.87 → 1999.51 → 2129.69 tok/s**; the **37.16 tok/s** decode belongs only to the 1999.51 checkpoint, not the final 50.73.
- The v1.0/v1.1 labels in the experiment tables are historical serving-method labels, not the current public milestone versions; those experiments now belong to v1.2/v1.3, respectively.
- ORNITH-PD-02: the 3080 held 18661/20480 MiB; the 395 Vulkan side about 22678/65536 MiB. A separate performance gate in the same round reached only 3785/44 because acceptance on that prompt was low, not because a parameter change was at fault. All three endpoints answered HTTP 200 with no OOM or crash after start-up. The fused recipe's 4173.47 sits 3.9% above ORNITH-PD-01's C1 4017.46. ORNITH-PD-01 did not record a single-stream Decode figure; the 395 pure-Decode numbers published for its 100K stage are six-tier aggregates (C1 23.33 to C6 148.20), which is not the same quantity as the single stream here.

</details>

## Version timeline v1.0 → v1.6

Seven releases map onto seven experiments that have data. 27B-SPARK-01 is a 2026-09-11 data addendum on v1.6, not an eighth public version.

| Release | Experiment | One-line conclusion |
| --- | --- | --- |
| v1.0 | 9B-PD-01 | The 3060 does Prefill and hands state to the 395 for Decode. Setup vs 395 alone: first token 3.496 s vs 5.879 s, Prefill 1452.29 vs 861.55, Decode 30.28 vs 30.24 |
| **v1.1** | 9B-PIPE-01, 27B-LONG-01, EXT-DGX-01 | 3060 + 395 computing the same stage at once, 9B Dense Acceleration works: Setup Prefill four checkpoints 1865.08 → 1893.87 → 1999.51 → 2129.69, final Decode 50.73 (37.16 belongs only to the 1999.51 checkpoint). 27B-LONG-01 layer-split vs 395 alone: pp4096 658.52 vs 313.28, pp65536 319.10 vs 136.69, pp98304 from a 900 s timeout to 225.10 |
| v1.2 | 27B-PD-01 | 3080 Prefill and 395 Decode works while serving. 3080 Prefill rose from 683.2 (per-ubatch RPC sync) to 1000.6 (direct CUDA), holding 1000–1015 from 1 to 6 concurrent streams; C1 TTFT Setup 1073 ms vs 395 alone 4825 ms. 207.2 is the measured v1.0-method 395-alone Prefill |
| v1.3 | 27B-KV-01, 27B-DRAFT-AUDIT-01 | With the checkpoint copy off, 3080 serving Prefill 1000.6 → 1210.6 (98.5% of raw 1228.53), 395-alone Prefill 207.2 → 307.1. Remote KV pool, two routes: configuration C, the 395 decodes, 3080 Prefill 1194.4–1210.6, aggregate Decode 1 stream 33.55 → 6 streams 63.84; configuration D, the 3080 decodes, 3080 Prefill 1077–1090, aggregate Decode 1 stream 63.2 → 6 streams 116.3. Audit: on the 395, natural-language C1 Decode 12.1, draft acceptance 17.7% |
| v1.4 | ORNITH-PD-01 | MoE stress test with 3080 doing all Prefill and 395 all Decode; two workloads × six concurrency tiers, 42/42 requests. 1000 input tokens: 3080 Prefill aggregate 1 stream 4017.46 → 6 streams 3943.88; 100K input: 2895.53 → 2793.24; 395 Decode aggregate (100K tier) 1 stream 23.33 → 6 streams 148.20 |
| v1.5 | FLASH-SPLIT-01 | Single-server split (3080 at 0.38, 395 at 0.62), 21/21 scored requests succeeded. Best tier is 4 streams (C4): Prefill 633.685, aggregate Decode 71.185, total throughput 338.270; 1 to 6 streams Prefill 569.892–633.685 |
| **v1.6** | ORNITH-PD-02 | 3080 Prefill 4173.47, 395 with DFlash draft single-stream Decode 114.86, draft acceptance 107/114 (93.86%). Decode follows acceptance: at 9.4% only 3665.3 / 37.2 |
| 2026-09-11 addendum (still v1.6) | 27B-SPARK-01 | RTX 3080 20GB + DGX Spark GB10, Qwen3.8-27B Q4_K_M. Prefill-first 8K/128 no speculation: pair 1603.20 / 17.62 vs 3080 standalone 1113.13 / 33.36 (**+44.03%**). Decode-first DFlash2: 2K 1153.95 / 63.97, 8K 1124.32 / 49.20. Balanced DFlash2 C1–C6 126/126. Not a new version. |

Version numbers are experiment milestones, not a count of documentation edits; how they map onto older publication numbers is in [VERSION_HISTORY.md](VERSION_HISTORY.md).

27B-C and 27B-D are two configurations of the single experiment 27B-KV-01, not two experiments; the natural-language run on the 395 belongs to 27B-DRAFT-AUDIT-01; EXT-DGX-01 remains someone else's public Spark measurement, kept as background; 27B-SPARK-01 is the local 3080 + Spark Q4_K_M measurement.

## How to read the data

What conclusion a figure is allowed to support.

- **Runs**: the request completes, the state hands over, and every metric can be credited to a device. 27B-KV-01, ORNITH-PD-01, ORNITH-PD-02, FLASH-SPLIT-01, and 27B-SPARK-01 belong here, all verified.
- **Faster**: model, quantization, workload, and metrics all match, and a single-card or single-host control exists. 9B-PIPE-01, 9B-PD-01, 27B-LONG-01, and 27B-PD-01 meet this bar, and their gains are in the tables above. 27B-SPARK-01 meets it only for Prefill versus this 3080 standalone at the confirmed Prefill profile.
- **Fits and serves**: the workload completes with memory and stability data. That is how 27B-KV-01, FLASH-SPLIT-01, and the 27B-SPARK-01 balanced profile are written.
- **Remote KV is a capacity route**: a configuration where the 395 only acts as a remote KV pool and never runs dense Prefill compute verifies capacity, with Decode ownership described by the two routes in 27B-KV-01, and is filed apart from Dense Acceleration.
- **A speculative-decode point test is not interchangeable with a random-seed stress test**: 27B-DRAFT-AUDIT-01 sets a rule and claims no gain.
- **External references are background, and unlike conditions are not ranked**: EXT-DGX-01 is someone else's public measurement, not a local control; data that differ in model, quantization, engine, prompt, or connection are never joined into one ranking table. A release number is not an experiment ID and not a data source. Every figure must say whether it came from the RTX 3060, the RTX 3080, the 395, or the 3080 + Spark pair.

## Roadmap

The controls that are still missing.

| Stage | Question raised by current results | Plan and completion criteria |
| --- | --- | --- |
| **Matched controls and remaining MoE cells** | Cell D1 already has both the single-card and the single-host control at 9B on the 3060; the other cells need their controls filled in under one workload standard, and two MoE cells have not started. | 1. Re-run the RTX 3080 under the same 9B Q6_K conditions. 2. Add matched single-card controls on the 3060 and 3080 for the 27B and MoE cells, and a same-round "3080 without remote KV" control for 27B-KV-01. 3. Run the full experiments for M1 (MoE, fits easily) and M3 (MoE, does not fit). 4. Carry one-to-one Dense Acceleration to more large-memory hosts and more small-VRAM cards. Done when the dense region lines up, the Prefill and Decode gains hold, and scheduling is stable. |
| **Spark pair follow-ups** | Measured versus still missing on RTX 3080 20GB + DGX Spark GB10. | **Measured:** 27B-SPARK-01 three working profiles versus this 3080 standalone microbenchmark (Prefill-first, no speculation; Decode-first DFlash2 at 2K and 8K; balanced DFlash2 C1–C6). **Not yet measured:** Spark standalone at the same model, quantization, and load; 3080 standalone balanced C1–C6; unified speculation settings on the pair versus the control; more models on this pair. |
| **One accelerator to many hosts** | Once one-to-one is stable, can one accelerator serve several large-memory hosts at once? | Study one-to-many scheduling, resource isolation, fair sharing, failure recovery, and the scaling limit. Done when the gain reproduces as hosts are added and the per-host slowdown stays acceptable. |

## Detailed reports and data

This page quotes only the few key figures per experiment; the complete data rows, metric definitions, and field notes are in the records and CSV files below; use the matching CSV for your own calculations, and do not combine data from different experiment IDs unless the record states that a comparable control exists.

| ID | Model · weight quantization · accelerator | Question | Record | CSV |
| --- | --- | --- | --- | --- |
| 9B-PD-01 | Ornith 9B · Q6_K · RTX 3060 12GB | Can CUDA Prefill hand its state to Vulkan Decode? | [v1.0 independent PD](results/v1.0-independent-pd.md) | [CSV](data/benchmark-results.csv) |
| 9B-PIPE-01 | Ornith 9B · Q6_K · RTX 3060 12GB | Can both devices compute one model together through an asynchronous layered pipeline? | [v2.4 fused layer pipeline](results/v2.4-fused-layer-pipeline.md) | [CSV](data/benchmark-results.csv) |
| 27B-LONG-01 · 27B-PD-01 · 27B-KV-01 · 27B-DRAFT-AUDIT-01 | Qwen3.8-27B · UD-IQ3_XXS and Q4_K_M · RTX 3060 / RTX 3080 | The 27B layer split, PD while serving, remote KV, and the speculative-decode audit | [Qwen3.8-27B two-machine PD](results/qwen3.8-27b-dual-machine-pd.md) | [CSV](data/qwen27b-local-results.csv) |
| ORNITH-PD-01 | Ornith-1.5-35B-A3B · IQ4_XS · RTX 3080 20GB | Can an MoE model keep Prefill and Decode attributable while surviving 100K stress from C1 to C6? | [Ornith two-machine PD](results/ornith-1.5-35b-a3b-dual-machine-pd.md) | [CSV](data/ornith35a3b-local-results.csv) |
| ORNITH-PD-02 | Ornith-1.5-35B-A3B · IQ4_XS · RTX 3080 20GB | With a fused draft head and a unified KV pool on top of independent PD, do Prefill and single-stream Decode rise together? | [Ornith fused-draft PD](results/ornith-1.5-35b-a3b-fused-dflash-pd.md) | [CSV](data/ornith35a3b-local-results.csv) |
| FLASH-SPLIT-01 | Qwen3.8-Flash · Q4 · RTX 3080 20GB | With one server split across CUDA and Vulkan, where does throughput level off? | [Qwen3.8-Flash Q4 layer split](results/qwen3.8-flash-q4-layer-split.md) | [CSV](data/qwen38flash-q4-local-results.csv) |
| 27B-SPARK-01 | Qwen3.8-27B · Q4_K_M · RTX 3080 20GB + DGX Spark GB10 | Which Prefill, Decode, and balanced serving profiles does this pair sustain versus the 3080 standalone? | [Qwen3.8-27B Spark + 3080 profiles](results/qwen3.8-27b-spark-3080-profiles.md) | [CSV](data/qwen27b-spark-3080-profiles.csv) |
| EXT-DGX-01 | Qwen3.5 9B · TQ3_4S and Qwen3.8-27B · NVFP4 · external DGX Spark | DGX Spark public figures, background only | [DGX Spark community control](results/dgx-spark-community-control.md) | [CSV](data/dgx-spark-community-controls.csv) |

The mapping from experiment IDs to legacy labels is in [data/experiment-index.csv](data/experiment-index.csv); all records are under [results/](results/). The [changelog](CHANGELOG.md) records what each public experiment version measured. Version mapping: [VERSION_HISTORY.md](VERSION_HISTORY.md).
