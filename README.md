# <img src="assets/soulmate-spirit.png" alt="Soulmate spirit" width="44" align="absmiddle"> Small-VRAM Accelerator + Large-VRAM, Low-Compute Host Dense Acceleration — Heterogeneous GPU PD Lab

[中文](README_ZH.md)

This repository studies how an NVIDIA card with plenty of compute but little VRAM (RTX 3060 12GB / RTX 3080 20GB) can team up with a host that has weak compute but lots of memory (AMD AI Max+ 395) to run large-model inference. Both devices compute the same stage of the same model at the same time; this is **Dense Acceleration**. The hardest figure: on 9B, the 3060 + 395 pair reaches Prefill 2129.69 tok/s, 34.0% above the faster single card at 1589.00 and 119.6% above the 395 alone at 970.00.

Current release **v1.6**, seven public milestones. Deployment commands, patches, endpoints, and the layer-allocation policy stay private.

## One minute: the core results

Each row is one approach, one experiment, one key result, and its control figures.

| Approach | Experiment (ID · model · quant · hardware) | Key result (with controls) | What it shows |
| --- | --- | --- | --- |
| Dense Acceleration | 9B-PIPE-01 · Ornith 9B · Q6_K · RTX 3060 12GB + 395 | Prefill **2129.69** tok/s (3060 alone 1589.00, 395 alone 970.00); Decode **50.73** (43.87 / 31.27) | The pair beats the fastest single device present; the extra is real 395 compute |
| Layer-split loading | 27B-LONG-01 · Qwen3.8-27B · UD-IQ3_XXS · RTX 3060 12GB + 395 | pp4096 **658.52** vs 313.28; pp65536 **319.10** vs 136.69; pp98304 **225.10** vs a 900 s timeout | The 3060 can take part even when 27B does not fit, and the long prompt finishes |
| Phase-separated PD | 9B-PD-01 · Ornith 9B · Q6_K · RTX 3060 12GB + 395 | TTFT **3.496** s vs 5.879 s; Prefill **1452.29** vs 861.55; Decode **30.28** vs 30.24 | TTFT drops while Decode holds; the phases still run one after the other |
| Phase-separated PD | 27B-PD-01 · Qwen3.8-27B · Q4_K_M · RTX 3080 20GB + 395 | TTFT **1073** ms vs 4825 ms; Prefill **1000.6** vs 207.2, only 82% of the 3080 raw 1228.53 | The ceiling is the accelerator card's own raw compute; PD buys TTFT and capacity |
| Remote KV pool | 27B-KV-01 · Qwen3.8-27B · Q4_K_M · RTX 3080 20GB + 395 | C: Prefill 1194.4–1210.6, Decode C1 33.55 / C6 63.84; D: Prefill 1077–1090, Decode C1 63.2 / C6 116.3 | Two decode routes: C, the 395 decodes; D, the 3080 decodes; the 395 never runs Prefill |
| MoE PD | ORNITH-PD-01 · Ornith-1.5-35B-A3B · IQ4_XS · RTX 3080 20GB + 395 | Prefill at 1000 in 4017.46 → 3943.88; 100K 2895.53 → 2793.24; 395 pure Decode 23.33 → 148.20 (6.35×); 42/42 successful requests | MoE PD stays stable at 100K context across six tiers, with both stages attributable |
| MoE fused draft | ORNITH-PD-02 · Ornith-1.5-35B-A3B · IQ4_XS · RTX 3080 20GB + 395 | Prefill **4173.47**, single-stream Decode **114.86**, draft acceptance 107/114 (93.86%) | Decode gain comes from draft acceptance: at 9.4% acceptance it is only 3665.3 / 37.2, and at 93.86% Decode is 3.09 times that |
| Single-server split | FLASH-SPLIT-01 · Qwen3.8-Flash · Q4 · RTX 3080 20GB + 395 | Best tier C4: Prefill 633.685, aggregate Decode 71.185, total throughput 338.270; tensor split 0.38 / 0.62 | One server drives both devices; C4 is the operating point on this workload |

## Why Dense Acceleration is this repository's focus

Three advantages, one bound.

1. Only Dense Acceleration lets the pair beat the fastest single device in the room. Prefill 2129.69 vs the 3060 alone at 1589.00 (+34.0%) and vs the 395 alone at 970.00 (+119.6%); Decode 50.73 vs 43.87 (+15.6%) and vs 31.27 (+62.2%). The extra is real compute from the 395.

2. The ceiling of phase-separated PD is the accelerator card's own raw compute. 27B-PD-01 pair Prefill 1000.6 reaches only 82% of the 3080 raw 1228.53, and 1210.6 with the checkpoint off is still only 98.5%; Decode barely moves under PD (9B: 30.28 vs 30.24; 27B: 38.75 vs 36.33). PD buys TTFT and capacity; it does not add the two devices' compute together.

3. Dense Acceleration does not require the accelerator card to hold the whole model. The card holds the front-stage layers; the host holds the rear-stage layers and the intermediate state. How large a model runs is the two devices' memory together; how fast it runs is the card. Layer-split loading in 27B-LONG-01 shows the 3060 can take part even when 27B does not fit: pp4096 658.52 vs the 395 alone at 313.28, pp65536 319.10 vs 136.69, pp98304 from a 900 s timeout to 225.10.

4. Bound: a full single-card plus single-host pair of controls exists only for the 9B Q6_K + RTX 3060 cell (D1); Dense Acceleration controls for the 27B and MoE cells are still on the roadmap. The dense region needs neighbouring micro-batches to run offset (an overlap window); whatever cannot enter the window belongs to the sparse region, which only has to hold the model.

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

## Five ways of dividing work

Every experiment in this repository falls into one of these five splits.

| Division of work | Accelerator card | AI Max+ 395 | Problem it solves | Experiment |
| --- | --- | --- | --- | --- |
| Dense Acceleration | Computes front-stage layers, same stage as the host | Computes rear-stage layers and intermediate state | The pair beats the fastest single device present | 9B-PIPE-01 |
| Layer-split loading | Holds and computes only the front-stage layers | Holds the rear-stage layers and finishes the model | The small card can run a model it cannot hold, with faster Prefill | 27B-LONG-01 |
| Phase-separated PD | Does all Prefill | Does Decode | Faster first token; Decode barely moves | 9B-PD-01, 27B-PD-01, ORNITH-PD-01, ORNITH-PD-02 |
| Remote KV pool | Prefill; configuration D, the 3080 decodes | Remote KV; configuration C, the 395 decodes | Capacity and concurrent Decode | 27B-KV-01 |
| Single-server split | Tensor split 0.38 inside one llama-server | Tensor split 0.62 | Find the best concurrency tier | FLASH-SPLIT-01 |

Only the first is both devices computing the same stage at once; the other four are division of labor, not stacked compute.

## Two hardware routes and the six experiment cells

Every figure belongs to one of two hardware routes and one of six experiment cells. Only two NVIDIA cards have been used so far, and the large-memory host has always been the same AI Max+ 395. The six cells split by how much VRAM the running model needs versus the accelerator's VRAM into fits easily, fills the card, and does not fit; each cell targets five configurations: RTX 3060 alone, RTX 3080 alone, AI Max+ 395 alone, 3060 + 395, and 3080 + 395. Within a cell the model, quantization, prompt, context, concurrency, and metrics must match; a card that cannot hold the model is itself a valid result, and a smaller model or a harsher quantization may not be substituted to manufacture a baseline. The two lines cannot be ranked against each other in one table. The repository holds no RTX 3090 measurements at all; that card was ruled out during selection.

| Item | Route 1: RTX 3060 12GB | Route 2: RTX 3080 20GB |
| --- | --- | --- |
| Releases covered | v1.0 → v1.1 | v1.2 → v1.6 |
| VRAM and what fits | 12GB, only the 9B dense tier fits; 27B fits only as an IQ3 layer split | 20GB, 27B Q4 fits, and that is what makes the MoE layer splits possible |
| Models run | Ornith 9B dense Q6_K; Qwen3.8-27B IQ3 | Qwen3.8-27B dense Q4; two MoE models, Ornith-1.5-35B-A3B and Qwen3.8-Flash |
| Experiment IDs | 9B-PD-01, 9B-PIPE-01, 27B-LONG-01 | 27B-PD-01, 27B-KV-01, 27B-DRAFT-AUDIT-01, ORNITH-PD-01, ORNITH-PD-02, FLASH-SPLIT-01 |
| What this line verified | Two devices can prefill one model together and beat the fastest single card present; a layered split lets the small card finish a model it cannot hold alone, and finish it faster | A larger card lifts the model size, the context depth, and the concurrency all at once; phase separation and remote KV both hold up while serving |

| Cell | Question | Verified so far |
| --- | --- | --- |
| **D1 Dense · fits easily**<br>Ornith 9B · Q6_K · RTX 3060 | With the model held comfortably on the card, can a dense pipeline actually beat the faster card, rather than just add capacity? | **Verified**: 9B-PIPE-01 has both the 3060 single-card and the 395 single-host controls, and the pair beats both. [Record](results/v2.4-fused-layer-pipeline.md) |
| **D2 Dense · fills the card**<br>Qwen3.8-27B · Q4_K_M · RTX 3080 | With VRAM nearly full, which wins: the whole model on one card, separated phases, or a layered dense route? | **Verified**: 27B-PD-01 runs the 3080 on Prefill and the 395 on Decode while serving, cuts the 395's time to first token by more than three quarters, and passes all six tiers C1–C6. [Record](results/qwen3.8-27b-dual-machine-pd.md) |
| **D3 Dense · does not fit**<br>Qwen3.8-27B · UD-IQ3_XXS · RTX 3060 | When one card cannot finish the job, can splitting the model finish it and still beat the 395? | **Verified**: 27B-LONG-01 runs more than twice as fast as the 395 alone and turns the 98K timeout into a finished run. [Record](results/qwen3.8-27b-dual-machine-pd.md) |
| **M1 MoE · fits easily**<br>model and quantization TBD | With few active parameters and VRAM to spare, does MoE routing overhead eat back the time the overlap saves? | Planned (see Roadmap). |
| **M2 MoE · fills the card**<br>Ornith-1.5-35B-A3B · IQ4_XS · RTX 3080 | With MoE nearly filling the 3080, is phase separation stable, and would a dense overlap on top add anything? | **Verified**: ORNITH-PD-01 routes every request, stays stable at 100K context across six concurrency tiers, and keeps both stages attributable ([record](results/ornith-1.5-35b-a3b-dual-machine-pd.md)); ORNITH-PD-02 adds a fused draft head and a unified KV pool on the same pair of devices, reaching Prefill 4173.47 and single-stream Decode 114.86 ([record](results/ornith-1.5-35b-a3b-fused-dflash-pd.md)). |
| **M3 MoE · does not fit**<br>Qwen3.8-Flash · Q4 · RTX 3080 pilot | When the total footprint exceeds both control cards, can splitting by layer or by expert keep it running, keep throughput, and keep the output correct? | FLASH-SPLIT-01 is the pilot for this cell: it verified that a single-server split runs and found its best concurrency tier. The full experiment is planned (see Roadmap). [Pilot record](results/qwen3.8-flash-q4-layer-split.md) |

## Experiment data

One small table per experiment; this is the only place on the front page that holds complete figures. Figures are copied from `results/` and `data/`; if this page and a record disagree, the record and the CSV win. A gain is "pair result ÷ control result − 1", and TTFT is written as how much it dropped.

**Dense Acceleration.** The direct evidence is only 9B-PIPE-01, with both a 3060 single-card control and a 395 single-host control.

### 9B-PIPE-01 · Ornith 9B · Q6_K · asynchronous layered Dense Acceleration

Accelerator RTX 3060 12GB; model Ornith 9B, weight quantization Q6_K (the records and CSV label this tier "9B · Q6_K"), `llama-bench` pp5064 / tg128; data from [benchmark-results.csv](data/benchmark-results.csv).

| Configuration | Prefill (tok/s) | Decode (tok/s) |
| --- | --- | --- |
| RTX 3060 12GB alone | 1589.00 | 43.87 |
| AI Max+ 395 alone (`llama-bench`) | 970.00 | 31.27 |
| 3060 + 395 pair | **2129.69** | **50.73** |
| Pair vs 3060 alone | +34.0% | +15.6% |
| Pair vs 395 alone | +119.6% | +62.2% |

What it verifies: both devices are genuinely working inside the dense region, and the pair beats the fastest single card present. This is the core Dense Acceleration evidence in the repository.

**It runs even when it does not fit.** 27B-LONG-01 is layer-split loading: a dense model split by layer across two devices, to solve "does not fit". It is not evidence of overlapping micro-batch pipelining.

### 27B-LONG-01 · Qwen3.8-27B · UD-IQ3_XXS · model split across layers

Accelerator RTX 3060 12GB; model Qwen3.8-27B, weight quantization UD-IQ3_XXS; pp4096 / pp65536 / pp98304 / tg64, all in tok/s; data from [qwen27b-local-results.csv](data/qwen27b-local-results.csv). The 3060 alone cannot hold the whole 27B, which is exactly why the model is split across layers.

| Configuration | pp4096 | pp65536 | pp98304 | tg64 |
| --- | --- | --- | --- | --- |
| AI Max+ 395 alone | 313.28 | 136.69 | timed out at 900 s | 18.26 |
| 3060 + 395 layered split | **658.52** | **319.10** | **225.10** | **19.57** |
| Pair vs 395 alone | +110.2% | +133.4% | timeout → finished | +7.2% |

What it verifies: when the small card cannot hold the model, the layered split finishes the 98K prompt that the 395 alone could not finish, and at 4K and 64K its Prefill runs more than twice as fast as the 395 alone.

**Phase-separated PD.** 9B-PD-01, 27B-PD-01, ORNITH-PD-01, and ORNITH-PD-02 are phase-separated PD: the two stages still run one after the other.

### 9B-PD-01 · Ornith 9B · Q6_K · independent PD

Accelerator RTX 3060 12GB; model Ornith 9B, weight quantization Q6_K, serving, 5064 in / 128 out; data from [benchmark-results.csv](data/benchmark-results.csv).

| Configuration | TTFT | Prefill (tok/s) | Decode (tok/s) |
| --- | --- | --- | --- |
| AI Max+ 395 alone (serving) | 5.879 s | 861.55 | 30.24 |
| 3060 does all Prefill, 395 does Decode | **3.496 s** | **1452.29** | **30.28** |
| Pair vs 395 alone | -40.5% | +68.6% | +0.1% |

What it verifies: the state computed on the CUDA side hands over in one piece to the Vulkan side for Decode, the first token comes sooner, and Decode holds. The two phases still run one after the other, so this is phase separation, a different route from Dense Acceleration.

### 27B-PD-01 · Qwen3.8-27B · Q4_K_M · independent PD (serving)

Accelerator RTX 3080 20GB; model Qwen3.8-27B, weight quantization Q4_K_M, KV cache q4_0; all six concurrency tiers C1–C6 passed, tier C1 shown; data from [qwen27b-local-results.csv](data/qwen27b-local-results.csv).

| Configuration | TTFT | Prefill (tok/s) | Decode (tok/s) |
| --- | --- | --- | --- |
| RTX 3080 raw compute (`llama-bench`) | — | pp1024 1228.53, pp4096 1203.06 | tg64 33.08 |
| AI Max+ 395 alone, C1 solo, v1.0 method | 4825 ms | 207.2 | 36.33 |
| 3080 on Prefill, 395 on Decode, C1, v1.0 method | **1073 ms** | **1000.6** | **38.75** |
| Pair vs 395 alone | -77.8% | +382.9% (4.83×) | +6.7% |

What it verifies: phase separation works while serving, Prefill does not fall off as concurrency rises, and Decode is unharmed. It verifies that the phases can be separated, not that both devices compute the same phase at once.

### ORNITH-PD-01 · Ornith-1.5-35B-A3B · IQ4_XS · MoE PD stress test

The RTX 3080 20GB takes all Prefill and the AI Max+ 395 all Decode; main model Ornith-1.5-35B-A3B with IQ4_XS weight quantization; draft head Qwen3.6-35B-A3B-DFlash with Q4_K_M weight quantization; data from [ornith35a3b-local-results.csv](data/ornith35a3b-local-results.csv).

| Metric | C1 | C6 | Change |
| --- | --- | --- | --- |
| Aggregate Prefill at 1000 input (tok/s) | 4017.46 | 3943.88 | -1.8% |
| Aggregate Prefill at 100K (tok/s) | 2895.53 | 2793.24 | -3.5% |
| 395 pure-Decode aggregate (tok/s) | 23.33 | 148.20 | 6.35× |

One stress run covered two workloads, a short one at 1000 in / 128 out and a long one at 100000 in / 128 out, for 42/42 successful requests in total, `route=pd`, `n_reuse=0`. The short stage has no cell for the 395 pure-Decode rate because that rate was not timed separately. What it verifies: MoE PD runs stably at 100K context across six concurrency tiers, and it is clear which device owns Prefill and which owns Decode.

### ORNITH-PD-02 · Ornith-1.5-35B-A3B · IQ4_XS · PD with a fused DFlash draft head

The same model on the same pair of devices, with two additions on top of ORNITH-PD-01's independent PD: Decode runs fused DFlash speculative decode (`--spec-type draft-dflash`, draft length 6), and the six slots share one unified KV pool (`--kv-unified`). Main model Ornith-1.5-35B-A3B with IQ4_XS weight quantization; draft head Qwen3.6-35B-A3B-DFlash with Q4_K_M weight quantization; KV cache q4_0; workload 1000 in / 128 out, single stream; data from [ornith35a3b-local-results.csv](data/ornith35a3b-local-results.csv).

| Configuration | Prefill (tok/s) | Single-stream Decode (tok/s) | Draft acceptance |
| --- | --- | --- | --- |
| Fused recipe, Prefill batch 4096 | **4173.47** | **114.86** | 93.86% |
| Fused recipe, batch raised to 8196, context 128K | **4123.15** | **114.42** | not recorded |
| Same recipe, a prompt the draft head could not predict | 3665.3 | 37.2 | 9.4% |

What it verifies: Prefill holds above 4000 while single-stream Decode reaches 114. When draft acceptance returns from 9.4% to 93.86%, Decode rises from 37.2 to 114.86, a factor of 3.09, so the Decode gain at this tier comes from the fused draft head; re-measuring requires the same prompt, a fixed output length, and the same seed. Draft acceptance on the main row is **107/114 (93.86%)**. [Record](results/ornith-1.5-35b-a3b-fused-dflash-pd.md)

**Capacity and serving tiers.** 27B-KV-01 is a remote KV pool with two decode routes; FLASH-SPLIT-01 is a single-server split across both devices.

### 27B-KV-01 · Qwen3.8-27B · Q4_K_M · Remote KV pool with two decode routes

Accelerator RTX 3080 20GB; model Qwen3.8-27B, weight quantization Q4_K_M, KV cache q4_0; data from [qwen27b-local-results.csv](data/qwen27b-local-results.csv).

| Configuration | Prefill (tok/s) | Aggregate Decode C1 | Aggregate Decode C6 | Growth C1→C6 |
| --- | --- | --- | --- | --- |
| Configuration C · 395 decodes | 1194.4–1210.6 | 33.55 | 63.84 | +90.3% |
| Configuration D · 3080 decodes | 1077–1090 | 63.2 | 116.3 | +84.0% |

1M context per stream; aggregate Decode in tok/s. Configuration C: the 395 decodes; configuration D: the 3080 decodes. On both routes the 395 never runs Prefill and the dense weight compute always stays on the 3080. What it verifies: take C and let the 395 decode when Prefill is the constraint; take D and let the 3080 decode when total Decode throughput is the constraint. This is a capacity and serving route, filed apart from Dense Acceleration.

### FLASH-SPLIT-01 · Qwen3.8-Flash · Q4 · one server split across both devices

A single llama-server using the RTX 3080 20GB and the AI Max+ 395 together; model Qwen3.8-Flash, weight quantization Q4, KV cache q4_0; tensor split 0.38 / 0.62, ubatch 1024 / batch 4096, 6 slots at 131072 context; 3080 VRAM peaked at 19129 MiB; data from [qwen38flash-q4-local-results.csv](data/qwen38flash-q4-local-results.csv).

| Metric | C4 (best tier) | Change C1→C4 |
| --- | --- | --- |
| Prefill (tok/s) | 633.685 | +11.2% |
| Aggregate Decode (tok/s) | 71.185 | +102.2% |
| Total throughput (tok/s) | 338.270 | +58.4% |

C1–C6 aggregate Prefill ran 569.892–633.685 tok/s and aggregate Decode 35.204–71.185 tok/s. 21/21 scored requests succeeded on a workload of about 2077 in / 256 out. What it verifies: the best tier for this configuration is C4, and C5 and C6 no longer rise. It is an operating point, and a different workload must be measured again.

**Audit and external reference.** 27B-DRAFT-AUDIT-01 is a data audit; EXT-DGX-01 is an external reference, background only.

### 27B-DRAFT-AUDIT-01 · Qwen3.8-27B · Q4_K_M · speculative-decode audit

Model Qwen3.8-27B, weight quantization Q4_K_M, KV cache q4_0; a data audit run on the AI Max+ 395; data from [qwen27b-local-results.csv](data/qwen27b-local-results.csv).

| Text type | Decode (tok/s) | Acceptance |
| --- | --- | --- |
| Repetitive text | 35.0–38.5 | 100% |
| Natural language, C1 | 12.1 | 17.7% |

Natural language sits 68.6% below the repetitive-text high score. What it verifies: this is a data audit that sets a rule — a speculative-decode score on repetitive text does not stand in for real text.

### EXT-DGX-01 · Qwen3.5 9B / TQ3_4S; Qwen3.8-27B / NVFP4 · DGX Spark external reference

This entry contains two external workloads: Qwen3.5 9B with TQ3_4S weight quantization, and Qwen3.8-27B with NVFP4 weights and an FP8 KV cache. Public figures are about 1000 tok/s Prefill, 25–30 tok/s single-stream Decode, and 107 tok/s aggregate Decode, C1–C6. These are public results from someone else's machine, kept as background and never ranked against local data. [Record](results/dgx-spark-community-control.md)

<details>
<summary>Number notes: why the same machine has several different figures</summary>

- The 395 has two 9B figures: the `llama-bench` figure (970.00 / 31.27) is the control for 9B-PIPE-01 and the serving figure (861.55 / 30.24) is the control for 9B-PD-01.
- The 395 also has three Prefill figures at the 27B tier. **207.2** is the C1 solo result under the v1.0 serving method, where every prompt also copied out a 524 MiB recurrent-state checkpoint and the Vulkan read-back took about 0.8 s; it is the control for the 1000.6 of 27B-PD-01. **307.1** is the C1 solo result on the same machines after that checkpoint copy was switched off (the v1.1 method); it is kept as the historical control in the configuration C table of the 27B-KV-01 record. **313.28** is `llama-bench` pp4096 on the IQ3 quantization and is the control for 27B-LONG-01. 207.2 is not a typo for 1207.2.
- The 3080 Prefill chain: 683.2 (per-ubatch RPC sync) → 1000.6 (direct CUDA, +46%) → 1210.6 (checkpoint copy off, see 27B-KV-01 configuration C, 98.5% of the raw ceiling 1228.53). The 1000.6 of 27B-PD-01 is 82% of raw compute; serving Prefill holds 1000–1015 tok/s as concurrency rises; KV transfer takes 68–76 ms (the serving record also states 71 ms). After both sides rose, the pair stays around 4× the 395 alone.
- At the 27B tier, any Prefill above 1200 belongs to the RTX 3080 side (raw 1228.53, serving 1194.4–1210.6); the 395 alone runs a dense 27B model at 200–320 Prefill, and even a 9B model at only 861.55–970.00.
- Configuration D reads 1090 / 1081 / 1080 / 1077 / 1082 / 1082 tok/s from C1 to C6, and the aggregate figures from the same run fall from 1079 to 1016 tok/s. The VRAM and compute cost of configuration D: the draft head weights occupy 1080 MiB and the verification batch needs roughly 500 MiB more of compute buffer, so at ctx8192 ubatch has to drop from 1024 to 512 and the slot count from 2 to 1, and the per-step verification matmul is real work (+21 ms on a 4-token batch, +49 ms on an 8-token batch). Prefill therefore falls from 1210.6 to 1077–1090, about ten percent lower.
- In configuration C the 3080 still keeps 2 slots, and the 395 runs a DFlash head at 38.75 tok/s single-stream while holding the full KV pool. Measured single-stream on the 3080 with the head: 42.7 tok/s on natural language and 67.4 tok/s on code, 2.2–2.3 times the 395 running the same head, which lifts aggregate Decode from 63.84 to 116.3.
- The four fused-pipeline checkpoints climbed Prefill **1865.08 → 1893.87 → 1999.51 → 2129.69 tok/s**; the **37.16 tok/s** decode belongs only to the 1999.51 checkpoint, not the final 50.73.
- The v1.0/v1.1 labels in the experiment tables are historical serving-method labels, not the current public milestone versions; those experiments now belong to v1.2/v1.3, respectively.
- ORNITH-PD-02: the 3080 held 18661/20480 MiB; the 395 Vulkan side about 22678/65536 MiB. A separate performance gate in the same round reached only 3785/44 because acceptance on that prompt was low, not because a parameter change was at fault. All three endpoints answered HTTP 200 with no OOM or crash after start-up. The fused recipe's 4173.47 sits 3.9% above ORNITH-PD-01's C1 4017.46. ORNITH-PD-01 did not record a single-stream Decode figure; the 395 pure-Decode numbers published for its 100K stage are six-tier aggregates (C1 23.33 to C6 148.20), which is not the same quantity as the single stream here.

</details>

## Version timeline v1.0 → v1.6

Seven releases map onto seven experiments that have data.

| Release | Experiment | One-line conclusion |
| --- | --- | --- |
| v1.0 | 9B-PD-01 | The handoff works: first token 3.496 s vs 5.879 s, Prefill 1452.29 vs 861.55, Decode 30.28 vs 30.24 |
| **v1.1** | 9B-PIPE-01, 27B-LONG-01, EXT-DGX-01 | Dense Acceleration works at 9B (Prefill 1865.08 → 1893.87 → 1999.51 → 2129.69, Decode 50.73; 37.16 only at 1999.51). 27B-LONG-01 more than doubles Prefill at 4K and 64K (658.52 / 319.10 vs 313.28 / 136.69) and turns the 98K timeout into 225.10 |
| v1.2 | 27B-PD-01 | PD works while serving. Prefill 683.2 → 1000.6, then held 1000–1015 as concurrency rose; C1 TTFT 1073 ms vs 4825 ms. 207.2 is the measured v1.0-method 395 solo Prefill |
| v1.3 | 27B-KV-01, 27B-DRAFT-AUDIT-01 | Checkpoint copy off: 3080 serving Prefill 1000.6 → 1210.6 (98.5% of 1228.53), 395 solo 207.2 → 307.1. C: Prefill 1194.4–1210.6, Decode C1 33.55 / C6 63.84. D: Prefill 1077–1090, Decode C1 63.2 / C6 116.3. Natural-language C1 Decode 12.1 at 17.7% acceptance |
| v1.4 | ORNITH-PD-01 | Every request goes through (42/42). Prefill 4017.46 → 3943.88 at 1000 input and 2895.53 → 2793.24 at 100K; 395 pure-Decode aggregate 23.33 → 148.20 |
| v1.5 | FLASH-SPLIT-01 | 21/21 scored requests succeed. C4 is the best tier: Prefill 633.685, aggregate Decode 71.185, total throughput 338.270. C1–C6 Prefill 569.892–633.685 |
| **v1.6** | ORNITH-PD-02 | Prefill 4173.47, single-stream Decode 114.86, draft acceptance 107/114 (93.86%). Decode follows draft acceptance: 3665.3 / 37.2 at 9.4% |

Version numbers are experiment milestones, not a count of documentation edits; how they map onto older publication numbers is in [VERSION_HISTORY.md](VERSION_HISTORY.md).

27B-C and 27B-D are two configurations of the single experiment 27B-KV-01, not two experiments; the natural-language run on the 395 belongs to 27B-DRAFT-AUDIT-01; DGX Spark is a public result from someone else's machine, kept as background.

## How to read the data

What conclusion a figure is allowed to support.

- **Runs**: the request completes, the state hands over, and every metric can be credited to a device. 27B-KV-01, ORNITH-PD-01, ORNITH-PD-02, and FLASH-SPLIT-01 belong here, all verified.
- **Faster**: model, quantization, workload, and metrics all match, and a single-card or single-host control exists. 9B-PIPE-01, 9B-PD-01, 27B-LONG-01, and 27B-PD-01 meet this bar, and their gains are in the tables above.
- **Fits and serves**: the workload completes with memory and stability data. That is how 27B-KV-01 and FLASH-SPLIT-01 are written.
- **Remote KV is a capacity route**: a configuration where the 395 only acts as a remote KV pool and never runs dense Prefill compute verifies capacity, with Decode ownership described by the two routes in 27B-KV-01, and is filed apart from Dense Acceleration.
- **A speculative-decode point test is not interchangeable with a random-seed stress test**: 27B-DRAFT-AUDIT-01 sets a rule and claims no gain.
- **External references are background, and unlike conditions are not ranked**: DGX Spark is someone else's public measurement, not a local control; data that differ in model, quantization, engine, prompt, or connection are never joined into one ranking table. A release number is not an experiment ID and not a data source. Every figure must say whether it came from the RTX 3060 or the RTX 3080.

## Roadmap

The controls that are still missing.

| Stage | Question raised by current results | Plan and completion criteria |
| --- | --- | --- |
| **Matched controls and remaining MoE cells** | Cell D1 already has both the single-card and the single-host control at 9B on the 3060; the other cells need their controls filled in under one workload standard, and two MoE cells have not started. | 1. Re-run the RTX 3080 under the same 9B Q6_K conditions. 2. Add matched single-card controls on the 3060 and 3080 for the 27B and MoE cells, and a same-round "3080 without remote KV" control for 27B-KV-01. 3. Run the full experiments for M1 (MoE, fits easily) and M3 (MoE, does not fit). 4. Carry one-to-one Dense Acceleration to more large-memory hosts and more small-VRAM cards. Done when the dense region lines up, the Prefill and Decode gains hold, and scheduling is stable. |
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
| EXT-DGX-01 | Qwen3.5 9B · TQ3_4S and Qwen3.8-27B · NVFP4 · external DGX Spark | DGX Spark public figures, background only | [DGX Spark community control](results/dgx-spark-community-control.md) | [CSV](data/dgx-spark-community-controls.csv) |

The mapping from experiment IDs to legacy labels is in [data/experiment-index.csv](data/experiment-index.csv); all records are under [results/](results/). The [changelog](CHANGELOG.md) records what each public experiment version measured. Version mapping: [VERSION_HISTORY.md](VERSION_HISTORY.md).
