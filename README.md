# <img src="assets/soulmate-spirit.png" alt="Soulmate spirit" width="44" align="absmiddle"> Small-VRAM Accelerator + Large-VRAM, Low-Compute Host Dense Acceleration — Heterogeneous GPU PD Lab

[简体中文](README_ZH.md)

This is **v1.6**. Seven measured experiment milestones are published:

- **v1.0** independent 9B PD (9B-PD-01)
- **v1.1** fused 9B Dense Acceleration pipeline, plus the 27B long-context layer split (9B-PIPE-01, 27B-LONG-01)
- **v1.2** 27B serving PD (27B-PD-01)
- **v1.3** 27B remote KV with decode routes C and D, plus the speculative-decode audit (27B-KV-01, 27B-DRAFT-AUDIT-01)
- **v1.4** Ornith independent PD (ORNITH-PD-01)
- **v1.5** Flash Q4 single-server layer split (FLASH-SPLIT-01)
- **v1.6** Ornith fused DFlash (ORNITH-PD-02)

No existing measured value is changed. The exact figures live in the result records under `results/` and the CSV files under `data/`. How public versions map onto older publication numbers is in [VERSION_HISTORY.md](VERSION_HISTORY.md).

This repository studies one thing: how a card with plenty of compute but little VRAM can team up with a host that has weak compute but lots of memory, so that together they run large-model inference. What is published here: the architecture, the measured numbers, how the design grew step by step, and where each conclusion applies. Deployment commands, patches, endpoints, and the exact layer-allocation policy stay private.

## How the System Works

The problem first. A consumer graphics card computes fast, but its VRAM is small and a larger model does not fit. A host with lots of memory holds any model, but computes slowly. This project makes the two devices compute the same stage of the same model together, so that "it fits" and "it is fast" live in one system.

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

We call this approach **Dense Acceleration**: two devices compute the same stage of the same model. The small-VRAM card holds the front layers and computes them; the large-memory host holds the rear layers and the state, and computes that half. The point of the **Dense Region** is timing: two neighbouring micro-batches run offset from each other, so at any moment both devices have work in hand and neither one waits. Whatever cannot fit into the overlap window belongs to the **Sparse Region**: hold the whole model and carry the context capacity.

What this is not: it is not running layers one after another, not tensor parallelism, not two devices computing the same layer twice, and not the kind of independent PD that uses the 395 as a remote KV store.

![Dense Acceleration structure](assets/dense-region-structure.png)

## Two Accelerator Routes and the Six Experiment Cells

Only two NVIDIA cards have been used so far, and the large-memory host has always been the same AI Max+ 395. The two cards hold different models and support different conclusions, so they form two separate lines of experiments; before reading any figure, check which line it came from. The two lines cannot be ranked against each other in one table. The repository holds no RTX 3090 measurements at all; that card was ruled out during selection.

| | Route 1: RTX 3060 12GB | Route 2: RTX 3080 20GB |
| --- | --- | --- |
| Releases covered | v1.0 → v1.1 | v1.2 → v1.6 |
| VRAM and what fits | 12GB, only the 9B dense tier fits; 27B fits only as an IQ3 layer split | 20GB, 27B Q4 fits, and that is what makes the MoE layer splits possible |
| Models run | Ornith 9B dense Q6_K; Qwen3.8-27B IQ3 | Qwen3.8-27B dense Q4; two MoE models, Ornith-1.5-35B-A3B and Qwen3.8-Flash |
| Experiment IDs | 9B-PD-01, 9B-PIPE-01, 27B-LONG-01 | 27B-PD-01, 27B-KV-01, 27B-DRAFT-AUDIT-01, ORNITH-PD-01, ORNITH-PD-02, FLASH-SPLIT-01 |
| What this line verified | Two devices can prefill one model together and beat the fastest single card present; a layered split lets the small card finish a model it cannot hold alone, and finish it faster | A larger card lifts the model size, the context depth, and the concurrency all at once; phase separation and remote KV both hold up while serving |

**The six experiment cells.** The cells are not divided by model name or parameter count but by how much VRAM the model needs in total compared with the accelerator's VRAM (weights, KV, compute buffers, and headroom). **Fits easily** means plenty of headroom is left; **fills the card** means it is close to the VRAM limit; **does not fit** means the card cannot finish the job alone. Each cell targets five configurations: RTX 3060 alone, RTX 3080 alone, AI Max+ 395 alone, 3060 + 395, and 3080 + 395, with the model, quantization, prompt, context length, concurrency, and metrics kept identical within a cell. When a card truly cannot hold the model, "does not fit, cannot finish" is itself a valid result; a smaller model or a harsher quantization may not be substituted to manufacture a baseline. The table below records how far each cell has been verified; the controls still to be added are listed together in the Roadmap.

| Cell | Question | Verified so far |
| --- | --- | --- |
| **D1 Dense · fits easily**<br>Ornith 9B · Q6_K · RTX 3060 | With the model held comfortably on the card, can a dense pipeline actually beat the faster card, rather than just add capacity? | **Verified**: 9B-PIPE-01 has both the 3060 single-card and the 395 single-host controls, and the pair beats both. [Record](results/v2.4-fused-layer-pipeline.md) |
| **D2 Dense · fills the card**<br>Qwen3.8-27B · Q4_K_M · RTX 3080 | With VRAM nearly full, which wins: the whole model on one card, separated phases, or a layered dense route? | **Verified**: 27B-PD-01 runs the 3080 on Prefill and the 395 on Decode while serving, cuts the 395's time to first token by more than three quarters, and passes all six tiers C1–C6. [Record](results/qwen3.8-27b-dual-machine-pd.md) |
| **D3 Dense · does not fit**<br>Qwen3.8-27B · UD-IQ3_XXS · RTX 3060 | When one card cannot finish the job, can splitting the model finish it and still beat the 395? | **Verified**: 27B-LONG-01 runs more than twice as fast as the 395 alone and turns the 98K timeout into a finished run. [Record](results/qwen3.8-27b-dual-machine-pd.md) |
| **M1 MoE · fits easily**<br>model and quantization TBD | With few active parameters and VRAM to spare, does MoE routing overhead eat back the time the overlap saves? | Planned (see Roadmap). |
| **M2 MoE · fills the card**<br>Ornith-1.5-35B-A3B · IQ4_XS · RTX 3080 | With MoE nearly filling the 3080, is phase separation stable, and would a dense overlap on top add anything? | **Verified**: ORNITH-PD-01 routes every request, stays stable at 100K context across six concurrency tiers, and keeps both stages attributable ([record](results/ornith-1.5-35b-a3b-dual-machine-pd.md)); ORNITH-PD-02 adds a fused draft head and a unified KV pool on the same pair of devices, reaching Prefill 4173.47 and single-stream Decode 114.86 ([record](results/ornith-1.5-35b-a3b-fused-dflash-pd.md)). |
| **M3 MoE · does not fit**<br>Qwen3.8-Flash · Q4 · RTX 3080 pilot | When the total footprint exceeds both control cards, can splitting by layer or by expert keep it running, keep throughput, and keep the output correct? | FLASH-SPLIT-01 is the pilot for this cell: it verified that a single-server layer split runs and found its best concurrency tier. The full experiment is planned (see Roadmap). [Pilot record](results/qwen3.8-flash-q4-layer-split.md) |

**Choosing a route by goal.**

- To show two devices computing one model faster together: asynchronous layered Dense Acceleration, backed by 9B-PIPE-01. That result rests on a single-card control and a single-host control; a different model must be verified the same way.
- For a faster first token with Prefill and Decode on different devices: independent PD, the 3060 for 9B (9B-PD-01) and the 3080 for 27B (27B-PD-01). The Prefill side must hold its own copy of the model.
- When the model does not fit the small card: split the model across layers, backed by 27B-LONG-01. For long prompts that need more Prefill, or simply for capacity.
- For the largest possible context, or for concurrent Decode: the 3080 runs Prefill and the 395 acts as a remote KV store, backed by 27B-KV-01. This is a capacity route; the 395 never runs Prefill, and Decode sits on either card depending on the route (C: 395 decodes; D: 3080 decodes).
- To verify MoE PD serving under deep context: see ORNITH-PD-01, which verified stable PD at 100K context across six tiers with both stages clearly attributed.
- To lift MoE Prefill and single-stream Decode together: see ORNITH-PD-02, which adds fused speculative decode and a unified KV pool on top of independent PD for Prefill above 4000 and single-stream Decode above 114. The cost is that the Decode figure follows draft acceptance, so a different prompt must be measured again.
- To find the right concurrency tier for a single server: see FLASH-SPLIT-01, which picked C4 on a workload of about 2077 in / 256 out; a different workload must be measured again.

## Experiment Results: Measured and Verified

The historical serving-method labels v1.0/v1.1 used in the measurement notes below are not public milestone versions: those experiments now belong to v1.2/v1.3, respectively.

**The conclusion first.** Attach a small-VRAM dense accelerator (an RTX 3060 12GB or an RTX 3080 20GB) to an AI Max+ 395, and on the same model the pair's Prefill, Decode, and time to first token all come out clearly ahead of the card running alone and clearly ahead of the 395 running alone. At 9B, the 3060 + 395 asynchronous pipeline beats the fastest single card in the room. At 27B, splitting the layers onto the 3060 finishes long prompts that the 395 alone cannot finish, and phase separation on the 3080 cuts the 395's time to first token by more than three quarters. The card's VRAM decides how large a model fits; the 395 decides how much context it can carry.

One small table per experiment follows; this is the only place on the page with complete data. Headings use the order "experiment ID · model · weight quantization · purpose"; KV-cache quantization is stated below the heading instead of being mixed with weight quantization. Every number comes from the measured records under `results/` and `data/`, and every set has been verified: the 9B pipeline was re-run through four checkpoints (Prefill 1865.08 → 1893.87 → 1999.51 → 2129.69); the 27B serving runs passed all six concurrency tiers C1–C6; the Ornith PD stress run completed 42/42 and Flash 21/21 requests; the Ornith fused-draft recipe kept all three endpoints healthy with no OOM. Each table stays within one model, one quantization, and one workload. The values are copied straight from the CSV files; a gain is "pair result ÷ control result − 1", and TTFT is written as how much it dropped. If this page and a record disagree, the record and the CSV win.

**The first four experiments have a single-card or single-host control; read the pair against it.**

### 9B-PIPE-01 · Ornith 9B · Q6_K · asynchronous layered Dense Acceleration

Accelerator RTX 3060 12GB; model Ornith 9B, weight quantization Q6_K (the records and CSV label this tier "9B · Q6_K"), `llama-bench` pp5064 / tg128; data from [benchmark-results.csv](data/benchmark-results.csv).

| Configuration | Prefill (tok/s) | Decode (tok/s) |
| --- | --- | --- |
| RTX 3060 12GB alone | 1589.00 | 43.87 |
| AI Max+ 395 alone (`llama-bench`) | 970.00 | 31.27 |
| 3060 + 395 pair | **2129.69** | **50.73** |
| Pair vs 3060 alone | +34.0% | +15.6% |
| Pair vs 395 alone | +119.6% | +62.2% |

What it verifies: both devices are genuinely working inside the Dense Region, and the pair beats the fastest single card present. This is the core Dense Acceleration evidence in the repository. The four fused-pipeline checkpoints that now sit under v1.1 climbed Prefill **1865.08 → 1893.87 → 1999.51 → 2129.69 tok/s**; the **37.16 tok/s** decode belongs only to the 1999.51 checkpoint, not the final 50.73.

### 9B-PD-01 · Ornith 9B · Q6_K · independent PD

Accelerator RTX 3060 12GB; model Ornith 9B, weight quantization Q6_K, serving, 5064 in / 128 out; data from [benchmark-results.csv](data/benchmark-results.csv).

| Configuration | TTFT | Prefill (tok/s) | Decode (tok/s) |
| --- | --- | --- | --- |
| AI Max+ 395 alone (serving) | 5.879 s | 861.55 | 30.24 |
| 3060 does all Prefill, 395 does Decode | **3.496 s** | **1452.29** | **30.28** |
| Pair vs 395 alone | -40.5% | +68.6% | +0.1% |

What it verifies: the state computed on the CUDA side hands over in one piece to the Vulkan side for Decode, the first token comes sooner, and Decode holds. The two phases still run one after the other, so this is phase separation, a different route from Dense Acceleration.

The 395 has two 9B figures. They do not disagree; they were measured differently: the `llama-bench` figure (970.00 / 31.27) is the control for 9B-PIPE-01 and the serving figure (861.55 / 30.24) is the control for 9B-PD-01.

### 27B-LONG-01 · Qwen3.8-27B · UD-IQ3_XXS · model split across layers

Accelerator RTX 3060 12GB; model Qwen3.8-27B, weight quantization UD-IQ3_XXS; pp4096 / pp65536 / pp98304 / tg64, all in tok/s; data from [qwen27b-local-results.csv](data/qwen27b-local-results.csv). The 3060 alone cannot hold the whole 27B, which is exactly why the model is split across layers.

| Configuration | pp4096 | pp65536 | pp98304 | tg64 |
| --- | --- | --- | --- | --- |
| AI Max+ 395 alone | 313.28 | 136.69 | timed out at 900 s | 18.26 |
| 3060 + 395 layered split | **658.52** | **319.10** | **225.10** | **19.57** |
| Pair vs 395 alone | +110.2% | +133.4% | timeout → finished | +7.2% |

What it verifies: when the small card cannot hold the model, the layered split finishes the 98K prompt that the 395 alone could not finish, and at 4K and 64K its Prefill runs more than twice as fast as the 395 alone.

### 27B-PD-01 · Qwen3.8-27B · Q4_K_M · independent PD (serving)

Accelerator RTX 3080 20GB; model Qwen3.8-27B, weight quantization Q4_K_M, KV cache q4_0; all six concurrency tiers C1–C6 passed, tier C1 shown; data from [qwen27b-local-results.csv](data/qwen27b-local-results.csv).

| Configuration | TTFT | Prefill (tok/s) | Decode (tok/s) |
| --- | --- | --- | --- |
| RTX 3080 raw compute (`llama-bench`) | — | pp1024 1228.53, pp4096 1203.06 | tg64 33.08 |
| AI Max+ 395 alone, C1 solo, v1.0 method | 4825 ms | 207.2 | 36.33 |
| 3080 on Prefill, 395 on Decode, C1, v1.0 method | **1073 ms** | **1000.6** | **38.75** |
| Pair vs 395 alone | -77.8% | +382.9% (4.83×) | +6.7% |

KV transfer takes 68–76 ms (the serving record also states 71 ms), and the pair's Prefill reaches 82% of the 3080's own raw figure. After removing per-ubatch RPC sync, Prefill went from **683.2** to **1000.6 tok/s** (+46%). Serving Prefill held **1000–1015 tok/s** as concurrency rose. What it verifies: phase separation works while serving, Prefill does not fall off as concurrency rises, and Decode is unharmed. It verifies that the phases can be separated, not that both devices compute the same phase at once.

The 395 also has three Prefill figures at the 27B tier. Again they were measured differently; none of them is a typo. **207.2** is the C1 solo result under the v1.0 serving method, where every prompt also copied out a 524 MiB recurrent-state checkpoint and the Vulkan read-back took about 0.8 s; it is the control for the 1000.6 of 27B-PD-01. **307.1** is the C1 solo result on the same machines after that checkpoint copy was switched off (the v1.1 method); it is kept as the historical control in the configuration C table of the 27B-KV-01 record. **313.28** is `llama-bench` pp4096 on the IQ3 quantization and is the control for 27B-LONG-01. Switching the checkpoint off also lifted the 3080 side's serving Prefill from 1000.6 to 1210.6 (see 27B-KV-01, configuration C); both sides rose together, and the pair stays around four times the 395 alone. At the 27B tier, any Prefill above 1200 belongs to the RTX 3080 side (raw 1228.53, serving 1194.4–1210.6); the 395 alone runs a dense 27B model at 200–320 Prefill, and even a 9B model at only 861.55–970.00. The 3080 Prefill chain is 683.2 (per-ubatch RPC sync) → 1000.6 (direct CUDA) → 1210.6 (checkpoint copy off) → 1228.53 (raw ceiling). 207.2 is not a typo for 1207.2.

**The next four experiments verify service capability: how much fits, how much concurrency it carries, and which tier works best.**

### 27B-KV-01 · Qwen3.8-27B · Q4_K_M · Remote KV pool with two decode routes

Accelerator RTX 3080 20GB; model Qwen3.8-27B, weight quantization Q4_K_M, KV cache q4_0; data from [qwen27b-local-results.csv](data/qwen27b-local-results.csv).

| Configuration | Prefill (tok/s) | Aggregate Decode C1 | Aggregate Decode C6 | Growth C1→C6 |
| --- | --- | --- | --- | --- |
| Configuration C · 395 decodes | 1194.4–1210.6 | 33.55 | 63.84 | +90.3% |
| Configuration D · 3080 decodes | 1077–1090 | 63.2 | 116.3 | +84.0% |

1M context per stream; aggregate Decode in tok/s. These are not two tuning passes over one route but **two decode routes**, and what differs is which card carries Decode. **Configuration C**: the 3080 carries no draft head, spends every bit of its compute on Prefill and writes KV back to the 395, which then does the decoding (that side runs a DFlash head at 38.75 tok/s single-stream while holding the full KV pool, and the headless 3080 still keeps 2 slots), so Prefill reaches 1194.4–1210.6, the highest figure in this entry. **Configuration D**: the DFlash draft head moves onto the 3080 and the 3080 decodes for itself (measured single-stream on that card: 42.7 tok/s on natural language and 67.4 tok/s on code, 2.2–2.3 times the 395 running the same head), which lifts aggregate Decode from 63.84 to 116.3. Prefill pays for it: Decode takes a large share of the compute and VRAM on the 3080 — the draft head weights occupy 1080 MiB and the verification batch needs roughly 500 MiB more of compute buffer, so at ctx8192 ubatch has to drop from 1024 to 512 and the slot count from 2 to 1, and the per-step verification matmul is real work (+21 ms on a 4-token batch, +49 ms on an 8-token batch). Prefill therefore falls from 1210.6 to 1077–1090, about ten percent lower. Both Prefill columns are single-stream figures and compare directly; configuration D reads 1090 / 1081 / 1080 / 1077 / 1082 / 1082 tok/s from C1 to C6, and the aggregate figures from the same run fall from 1079 to 1016 tok/s. What it verifies: take C and let the 395 decode when Prefill is the constraint; take D and let the 3080 decode when total Decode throughput is the constraint. On both routes the 395 never runs Prefill and the dense weight compute always stays on the 3080, so this is a capacity and serving route, filed apart from Dense Acceleration.

### 27B-DRAFT-AUDIT-01 · Qwen3.8-27B · Q4_K_M · speculative-decode audit

Model Qwen3.8-27B, weight quantization Q4_K_M, KV cache q4_0; a data audit run on the AI Max+ 395; data from [qwen27b-local-results.csv](data/qwen27b-local-results.csv).

| Text type | Decode (tok/s) | Acceptance |
| --- | --- | --- |
| Repetitive text | 35.0–38.5 | 100% |
| Natural language, C1 | 12.1 | 17.7% |

Natural language sits 68.6% below the repetitive-text high score. What it verifies: this is a data audit that sets a rule — a speculative-decode score on repetitive text does not stand in for real text.

### ORNITH-PD-01 · Ornith-1.5-35B-A3B · IQ4_XS · MoE PD stress test

The RTX 3080 20GB takes all Prefill and the AI Max+ 395 all Decode; main model Ornith-1.5-35B-A3B with IQ4_XS weight quantization; draft head Qwen3.6-35B-A3B-DFlash with Q4_K_M weight quantization; data from [ornith35a3b-local-results.csv](data/ornith35a3b-local-results.csv).

| Metric | C1 | C6 | Change |
| --- | --- | --- | --- |
| Aggregate Prefill at 1000 input (tok/s) | 4017.46 | 3943.88 | -1.8% |
| Aggregate Prefill at 100K (tok/s) | 2895.53 | 2793.24 | -3.5% |
| 395 pure-Decode aggregate (tok/s) | 23.33 | 148.20 | 6.35× |

One stress run covered two workloads, a short one at 1000 in / 128 out and a long one at 100000 in / 128 out, for 42/42 successful requests in total, `route=pd`, `n_reuse=0`. The short stage has no cell for the 395 pure-Decode rate because that rate was not timed separately. What it verifies: MoE PD runs stably at 100K context across six concurrency tiers, and it is clear which device owns Prefill and which owns Decode.

### ORNITH-PD-02 · Ornith-1.5-35B-A3B · IQ4_XS · PD with a fused DFlash draft head

The same model on the same pair of devices as the first experiment, with two additions on the 395 side: Decode runs fused DFlash speculative decode (`--spec-type draft-dflash`, draft length 6), and the six slots share one unified KV pool (`--kv-unified`). Main model Ornith-1.5-35B-A3B with IQ4_XS weight quantization; draft head Qwen3.6-35B-A3B-DFlash with Q4_K_M weight quantization; KV cache q4_0; workload 1000 in / 128 out, single stream; data from [ornith35a3b-local-results.csv](data/ornith35a3b-local-results.csv).

| Configuration | Prefill (tok/s) | Single-stream Decode (tok/s) | Draft acceptance |
| --- | --- | --- | --- |
| Fused recipe, Prefill batch 4096 | **4173.47** | **114.86** | 93.86% |
| Fused recipe, batch raised to 8196, context 128K | **4123.15** | **114.42** | not recorded |
| Same recipe, a prompt the draft head could not predict | 3665.3 | 37.2 | 9.4% |

On this same 1000 in / 128 out workload the first experiment reached Prefill C1 4017.46 down to C6 3943.88, so the fused recipe's 4173.47 sits 3.9% above its C1. The first experiment did not record a single-stream Decode figure; the 395 pure-Decode numbers published for its 100K stage are six-tier aggregates (C1 23.33 to C6 148.20), which is not the same quantity as the single stream here and cannot be compared directly.

What it verifies: Prefill holds above 4000 while single-stream Decode reaches 114. When draft acceptance returns from 9.4% to 93.86%, Decode rises from 37.2 to 114.86, a factor of 3.09, so the Decode gain at this tier comes from the fused draft head; re-measuring requires the same prompt, a fixed output length, and the same seed, because a different prompt drops it back. Draft acceptance on the main row is **107/114 (93.86%)**. All three endpoints answered HTTP 200 with no OOM or crash after start-up, and the 3080 held 18661/20480 MiB; the 395 Vulkan side about 22678/65536 MiB. A separate performance gate in the same round reached only 3785/44 because acceptance on that prompt was low, not because a parameter change was at fault. [Record](results/ornith-1.5-35b-a3b-fused-dflash-pd.md)

### FLASH-SPLIT-01 · Qwen3.8-Flash · Q4 · one server split across both devices

A single llama-server using the RTX 3080 20GB and the AI Max+ 395 together; model Qwen3.8-Flash, weight quantization Q4, KV cache q4_0; tensor split 0.38 / 0.62, ubatch 1024 / batch 4096, 6 slots at 131072 context; 3080 VRAM peaked at 19129 MiB; data from [qwen38flash-q4-local-results.csv](data/qwen38flash-q4-local-results.csv).

| Metric | C4 (best tier) | Change C1→C4 |
| --- | --- | --- |
| Prefill (tok/s) | 633.685 | +11.2% |
| Aggregate Decode (tok/s) | 71.185 | +102.2% |
| Total throughput (tok/s) | 338.270 | +58.4% |

C1–C6 aggregate Prefill ran 569.892–633.685 tok/s and aggregate Decode 35.204–71.185 tok/s. 21/21 scored requests succeeded on a workload of about 2077 in / 256 out. What it verifies: the best tier for this configuration is C4, and C5 and C6 no longer rise. It is an operating point, and a different workload must be measured again.

### EXT-DGX-01 · Qwen3.5 9B / TQ3_4S; Qwen3.8-27B / NVFP4 · DGX Spark external reference

This entry contains two external workloads: Qwen3.5 9B with TQ3_4S weight quantization, and Qwen3.8-27B with NVFP4 weights and an FP8 KV cache. Public figures are about 1000 tok/s Prefill, 25–30 tok/s single-stream Decode, and 107 tok/s aggregate Decode, C1–C6. These are public results from someone else's machine, kept as background and never ranked against local data. [Record](results/dgx-spark-community-control.md)

## Experiment Timeline: v1.0 → v1.6

Public version numbers are the seven real experiment milestones. They are not a count of documentation edits. Figures are not repeated here except where a deleted maintenance row carried a measured number that belongs with the milestone; the complete tables are in the experiment results above.

| Release | Question | What was done | How the conclusion moved |
| --- | --- | --- | --- |
| v1.0 | Can CUDA Prefill hand its state to Vulkan Decode? | 9B Q6_K; the RTX 3060 prefills a varying share of the tokens, compared with the 395 alone (9B-PD-01). | The handoff works: first token sooner (3.496 s vs 5.879 s), Prefill 1452.29 vs 861.55, Decode unharmed (30.28 vs 30.24). The devices still run one after the other, which leads to pipelining. |
| **v1.1** | With both devices working, can the pair beat the fastest single card, and can a layer split finish a 27B model the 3060 cannot hold? | Final fused-pipeline calibration filed as 9B-PIPE-01 (Prefill climb 1865.08 → 1893.87 → 1999.51 → 2129.69; Decode 50.73, with 37.16 only at the 1999.51 checkpoint). Same period: 27B IQ3 layer-split 27B-LONG-01, and the DGX Spark external reference (EXT-DGX-01, background only). | Dense Acceleration works at 9B. 27B-LONG-01 more than doubles Prefill at 4K and 64K (658.52 / 319.10 vs 313.28 / 136.69) and turns the 98K timeout into 225.10. |
| v1.2 | Can the 3080 take all Prefill and the 395 all Decode, and carry 27B Q4 in a real service? | Moved to the RTX 3080 20GB; six concurrency tiers C1–C6, each on the split and solo paths (27B-PD-01). | PD works while serving. Prefill 683.2 → 1000.6 after dropping per-ubatch RPC sync, then held 1000–1015 as concurrency rose; C1 TTFT 1073 ms vs 4825 ms. 207.2 is the measured v1.0-method 395 solo Prefill, not a typo. |
| v1.3 | After 27B PD, how do remote KV and Decode ownership split, and does a repetitive-text draft score represent real text? | 27B-KV-01 two decode routes and 27B-DRAFT-AUDIT-01. Checkpoint copy off: 3080 serving Prefill 1000.6 → 1210.6 (98.5% of 1228.53), 395 solo 207.2 → 307.1. | C: 395 decodes, Prefill 1194.4–1210.6, Decode C1 33.55 / C6 63.84. D: 3080 decodes, Prefill 1077–1090 (about ten percent lower), Decode C1 63.2 / C6 116.3. Natural-language C1 Decode 12.1 at 17.7% acceptance. The 395 never runs dense Prefill. |
| v1.4 | With MoE, do stage ownership and 100K long-context stability still hold? | Ornith-1.5-35B-A3B two-machine PD stress test: the 3080 takes all Prefill, the 395 all Decode (ORNITH-PD-01). | Every request goes through (42/42). Prefill 4017.46 → 3943.88 at 1000 input and 2895.53 → 2793.24 at 100K; 395 pure-Decode aggregate 23.33 → 148.20. |
| v1.5 | With one server driving both devices, where does throughput level off? | Qwen3.8-Flash Q4: a single llama-server using the 3080 and the 395 together, C1–C6 (FLASH-SPLIT-01). | 21/21 scored requests succeed. C4 is the best tier: Prefill 633.685, aggregate Decode 71.185, total throughput 338.270. C1–C6 Prefill 569.892–633.685. |
| **v1.6** | Can Prefill and single-stream Decode rise together if a fused draft head and a unified KV pool sit on top of independent PD? | ORNITH-PD-02 on the same pair of devices as v1.4. | Prefill 4173.47, single-stream Decode 114.86, draft acceptance 107/114 (93.86%). Decode follows draft acceptance: 3665.3 / 37.2 at 9.4%. |

Things that are easy to double-count: 27B-C and 27B-D are two configurations of the single experiment 27B-KV-01, not two experiments; the natural-language run on the 395 belongs to 27B-DRAFT-AUDIT-01; DGX Spark is a public result from someone else's machine, kept as background.

## How to Read the Data

- **Runs**: the request completes, the state hands over, and every metric can be credited to a device — then we write "this configuration runs". 27B-KV-01, ORNITH-PD-01, ORNITH-PD-02, and FLASH-SPLIT-01 belong here, all verified.
- **Faster**: model, quantization, workload, and metrics all match, and a single-card or single-host control exists — then we write "faster than the control". 9B-PIPE-01, 9B-PD-01, 27B-LONG-01, and 27B-PD-01 meet this bar, and their gains are in the tables above.
- **Fits and serves**: the workload completes with memory and stability data — then the conclusion reads "fits" or "serves to this level". That is how 27B-KV-01 and FLASH-SPLIT-01 are written.
- **Remote KV is a capacity route**: a configuration where the 395 only acts as a remote KV pool and never runs dense Prefill compute verifies capacity, with Decode ownership described by the two routes in 27B-KV-01, and is filed apart from Dense Acceleration.
- **A speculative-decode point test is not interchangeable with a random-seed stress test**: 27B-DRAFT-AUDIT-01 sets a rule and claims no gain.
- **External references are background**: DGX Spark is someone else's public measurement with its method and source stated; it is not used as a local control.
- Every figure must say whether it came from the RTX 3060 or the RTX 3080; data that differ in model, quantization, engine, prompt, or connection are never joined into one ranking table. A release number is not an experiment ID and not a data source.

## Roadmap

| Stage | Question raised by current results | Plan and completion criteria |
| --- | --- | --- |
| **Matched controls and remaining MoE cells** | Cell D1 already has both the single-card and the single-host control at 9B on the 3060; the other cells need their controls filled in under one workload standard, and two MoE cells have not started. | 1. Re-run the RTX 3080 under the same 9B Q6_K conditions. 2. Add matched single-card controls on the 3060 and 3080 for the 27B and MoE cells, and a same-round "3080 without remote KV" control for 27B-KV-01. 3. Run the full experiments for M1 (MoE, fits easily) and M3 (MoE, does not fit). 4. Carry one-to-one Dense Acceleration to more large-memory hosts and more small-VRAM cards. Done when the Dense Region lines up, the Prefill and Decode gains hold, and scheduling is stable. |
| **One accelerator to many hosts** | Once one-to-one is stable, can one accelerator serve several large-memory hosts at once? | Study one-to-many scheduling, resource isolation, fair sharing, failure recovery, and the scaling limit. Done when the gain reproduces as hosts are added and the per-host slowdown stays acceptable. |

## Detailed Reports and Data

This page quotes only the few key figures per experiment; the full rows, the metric definitions, and the field notes live in the records and CSV files below. Use the matching CSV for your own calculations, and do not combine data from different experiment IDs unless the record states that a comparable control exists.

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
