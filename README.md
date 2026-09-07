# <img src="assets/soulmate-spirit.png" alt="Soulmate spirit" width="44" align="absmiddle"> Small-VRAM Accelerator + Large-VRAM, Low-Compute Host Dense Acceleration — Heterogeneous GPU PD Lab

[简体中文](README_ZH.md)

This is **v2.23**. This release only checks and spells out how the Prefill figures were measured: the three Prefill numbers for the AI Max+ 395 alone at the 27B tier (207.2, 307.1, 313.28) each get their method stated, and the RTX 3080 side's climb from 683.2 to 1000.6 to 1210.6 is filled into the timeline. The check found that no existing number on this page needed to change. The exact figures live in the result records under `results/` and the CSV files under `data/`.

This repository studies one thing: how a card with plenty of compute but little VRAM can team up with a host that has weak compute but lots of memory, so that together they run large-model inference. What is published here: the architecture, the measured numbers, how the design grew step by step, and where each conclusion applies. Deployment commands, patches, endpoints, and the exact layer-allocation policy stay private.

## Conclusions and Core Data

**The conclusion first.** Attach a small-VRAM dense accelerator (an RTX 3060 12GB or an RTX 3080 20GB) to an AI Max+ 395, and on the same model the pair's Prefill, Decode, and time to first token all come out clearly ahead of the card running alone and clearly ahead of the 395 running alone. At 9B, the 3060 + 395 asynchronous pipeline beats the fastest single card in the room. At 27B, splitting the layers onto the 3060 finishes long prompts that the 395 alone cannot finish, and phase separation on the 3080 cuts the 395's time to first token by more than three quarters. The card's VRAM decides how large a model fits; the 395 decides how much context it can carry.

The two tables below are the only place on this page with complete data. Every row stays within one model, one quantization, and one workload. The values are copied straight from the CSV files; a gain is "pair result ÷ control result − 1", and TTFT is written as how much it dropped. If this page and a record disagree, the record and the CSV win.

### Experiments with a control: the pair against the card alone and the host alone

| Experiment | Accelerator | Model and workload | Card alone | AI Max+ 395 alone | Card + 395 together | Gain | What it shows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **9B-PIPE-01**<br/>asynchronous layered Dense Acceleration | RTX 3060 12GB | 9B Q6_K, `llama-bench` pp5064 / tg128 | Prefill **1589.00**, Decode **43.87 tok/s** | Same method: Prefill **970.00**, Decode **31.27 tok/s** | Prefill **2129.69**, Decode **50.73 tok/s** | vs the 3060 alone **+34.0% / +15.6%**; vs the 395 alone **+119.6% / +62.2%** | Both devices are genuinely working inside the Dense Region, and the pair beats the fastest single card present. This is the only controlled Dense Acceleration evidence in the repository. |
| **9B-PD-01**<br/>independent PD | RTX 3060 12GB | 9B Q6_K, serving, 5064 in / 128 out | Not measured in this run; the comparison is against the 395 only | Serving: TTFT **5.879 s**, Prefill **861.55**, Decode **30.24 tok/s** | 3060 does all Prefill, 395 does Decode: TTFT **3.496 s**, Prefill **1452.29**, Decode **30.28 tok/s** | TTFT **-40.5%**; Prefill **+68.6%**; Decode **+0.1%** | The state hands over in one piece, the first token comes sooner, and Decode holds. The two phases still run one after the other, so this is not Dense Acceleration. |
| **27B-LONG-01**<br/>model split across layers | RTX 3060 12GB | 27B IQ3, pp4096 / pp65536 / pp98304 / tg64 | The 3060 cannot hold the whole 27B; no single-card figure | Prefill **313.28** / **136.69 tok/s**, pp98304 timed out at 900 s; Decode **18.26 tok/s** | Prefill **658.52** / **319.10** / **225.10 tok/s**; Decode **19.57 tok/s** | 4K **+110.2%**; 64K **+133.4%**; 98K goes from timeout to finished; Decode **+7.2%** | When the small card cannot hold the model and long prompts are the bottleneck, the layered split not only finishes but runs more than twice as fast as the 395 alone. |
| **27B-PD-01**<br/>independent PD, serving | RTX 3080 20GB | 27B Q4, tier C1 (all six tiers C1–C6 pass) | 3080 raw compute: Prefill pp1024 **1228.53** / pp4096 **1203.06 tok/s**; Decode tg64 **33.08 tok/s** | C1, solo path, v1.0 method: TTFT **4825 ms**, Prefill **207.2**, Decode **36.33 tok/s** | 3080 on Prefill, 395 on Decode, v1.0 method: TTFT **1073 ms**, Prefill **1000.6**, Decode **38.75 tok/s**; KV transfer **68–76 ms** | vs the 395 alone: TTFT **-77.8%**, Prefill **+382.9% (4.83×)**, Decode **+6.7%**; Prefill reaches **82%** of the 3080's own raw figure | Phase separation works while serving: Prefill does not fall off as concurrency rises and Decode is unharmed. It proves that the phases can be separated, not that both devices compute the same phase at once. |

The 395 has two 9B figures. They do not disagree; they were measured differently: the `llama-bench` figure is the control for 9B-PIPE-01 and the serving figure is the control for 9B-PD-01. The two 9B rows come from [benchmark-results.csv](data/benchmark-results.csv) and the two 27B rows from [qwen27b-local-results.csv](data/qwen27b-local-results.csv).

The 395 also has three Prefill figures at the 27B tier. Again they were measured differently; none of them is a typo. **207.2** is the C1 solo result under the v1.0 serving method, where every prompt also copied out a 524 MiB recurrent-state checkpoint and the Vulkan read-back took about 0.8 s; it is the control for the 1000.6 of 27B-PD-01. **307.1** is the C1 solo result on the same machines after that checkpoint copy was switched off (the v1.1 method); it is kept as the historical control in the configuration C table of the 27B-KV-01 record. **313.28** is `llama-bench` pp4096 on the IQ3 quantization and is the control for 27B-LONG-01. Switching the checkpoint off also lifted the 3080 side's serving Prefill from 1000.6 to 1210.6 (see 27B-KV-01, configuration C); both sides rose together, and the pair stays around four times the 395 alone. At the 27B tier, any Prefill above 1200 belongs to the RTX 3080 side (raw 1228.53, serving 1194.4–1210.6); the 395 alone runs a dense 27B model at 200–320 Prefill, and even a 9B model at only 861.55–970.00.

### Experiments without a control: how well they serve, not how much faster

These runs have no matching single-card or single-host figure, so they only record how far they got; no gain is claimed. The percentages describe how one configuration changes as concurrency rises. All of them ran on the RTX 3080 20GB.

| Experiment | Key measurements | Change within the same configuration | What it shows |
| --- | --- | --- | --- |
| **27B-KV-01**<br/>3080 does all compute, 395 stores KV only | **1M context per stream**; configuration C holds Prefill at **1194.4–1210.6 tok/s**; aggregate Decode from C1 to C6 is **33.55→63.84** for C and **63.2→116.3 tok/s** for D | From C1 to C6, C rises **+90.3%** and D **+84.0%** | Pick C for Prefill and D for total Decode throughput. The 395 does no model compute at all, so this is a capacity and serving route, not Dense Acceleration. The matching "3080 without remote KV" control is missing. |
| **27B-DRAFT-AUDIT-01**<br/>speculative-decode audit | Repetitive text **35.0–38.5 tok/s, 100% acceptance**; natural language at C1 **12.1 tok/s, 17.7% acceptance** | Natural language sits **68.6%** below the repetitive-text high score | This is a data audit, not an acceleration experiment: a speculative-decode score on repetitive text may not stand in for real text. |
| **ORNITH-PD-01**<br/>MoE PD stress test | **42/42 succeeded**, `route=pd`, `n_reuse=0`; 100K Prefill from C1 to C6 is **2895.53→2793.24**; 395 pure-Decode aggregate **23.33→148.20 tok/s** | Prefill moves only **-3.5%** from C1 to C6; Decode scales **6.35×** with concurrency | MoE PD runs stably, and it is clear which device owns Prefill and which owns Decode. Without a single-device figure it cannot be called faster. |
| **FLASH-SPLIT-01**<br/>one server split across both devices | **21/21 succeeded**; C4 is best: Prefill **633.685**, aggregate Decode **71.185**, total throughput **338.270 tok/s** | From C1 to C4: Prefill **+11.2%**, aggregate Decode **+102.2%**, total throughput **+58.4%** | The best tier for this configuration is C4; C5 and C6 no longer rise. It is an operating point, not a speedup factor. |
| **EXT-DGX-01**<br/>DGX Spark external reference | Public figures of about **1000 tok/s Prefill, 25–30 tok/s single-stream Decode, 107 tok/s aggregate Decode**, C1–C6 | Not applicable | A public result from someone else's machine, kept as background and never ranked against local data. [Record](results/dgx-spark-community-control.md) |

## Base Architecture: How the System Works

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
| Releases covered | v1.0 → v2.4 | v2.5 → v2.14 |
| VRAM and what fits | 12GB, only the 9B dense tier fits; 27B fits only as an IQ3 layer split | 20GB, 27B Q4 fits, and that is what makes the MoE layer splits possible |
| Models run | 9B dense Q6_K; 27B IQ3 | 27B dense Q4; two MoE models, Ornith-1.5-35B-A3B and Qwen3.8-Flash |
| Experiment IDs | 9B-PD-01, 9B-PIPE-01, 27B-LONG-01 | 27B-PD-01, 27B-KV-01, 27B-DRAFT-AUDIT-01, ORNITH-PD-01, FLASH-SPLIT-01 |
| What this line sets out to verify | Whether two devices can prefill one model together and beat the fastest single card present; whether a layered split lets the small card finish a model it cannot hold alone | Whether a larger card lifts the model size, the context depth, and the concurrency all at once; whether phase separation and remote KV hold up while serving |
| What this line lacks | A rerun on the RTX 3080 under the same 9B conditions | Matched single-card controls on the 3060 or 3080 for the 27B and MoE cells |

**The six experiment cells.** The cells are not divided by model name or parameter count but by how much VRAM the model needs in total compared with the accelerator's VRAM (weights, KV, compute buffers, and headroom). **Fits easily** means plenty of headroom is left; **fills the card** means it is close to the VRAM limit; **does not fit** means the card cannot finish the job alone. Each cell aims to measure five configurations: RTX 3060 alone, RTX 3080 alone, AI Max+ 395 alone, 3060 + 395, and 3080 + 395, with the model, quantization, prompt, context length, concurrency, and metrics kept identical within a cell. When a card truly cannot hold the model, "does not fit, cannot finish" is itself a valid result; a smaller model or a harsher quantization may not be substituted to manufacture a baseline.

None of the six cells is complete yet: **0/6 finished, 4 cells with partial data, 2 cells untested**. A cell stays unfinished as long as the matched 3060 or 3080 baseline is missing.

| Cell | Question | Status |
| --- | --- | --- |
| **D1 Dense · fits easily** (card measured: RTX 3060) | With the model held comfortably on the card, can a dense pipeline actually beat the faster card, rather than just add capacity? | **Partly done**: 9B-PIPE-01 has both the 3060 single-card and the 395 single-host controls; the matched 3080 rerun is missing. [Record](results/v2.4-fused-layer-pipeline.md) |
| **D2 Dense · fills the card** (card measured: RTX 3080) | With VRAM nearly full, which wins: the whole model on one card, separated phases, or a layered dense route? | **Partly done**: 27B-PD-01 has serving data with the 3080 on Prefill and the 395 on Decode; a matched two-card dense comparison is missing. [Record](results/qwen3.8-27b-dual-machine-pd.md) |
| **D3 Dense · does not fit** (card measured: RTX 3060) | When one card cannot finish the job, can splitting the model finish it and still beat the 395? | **Partly done**: 27B-LONG-01 already beats the 395 alone and turns the 98K timeout into a finished run; the matching 3080 record is missing. [Record](results/qwen3.8-27b-dual-machine-pd.md) |
| **M1 MoE · fits easily** (neither card measured yet) | With few active parameters and VRAM to spare, does MoE routing overhead eat back the time the overlap saves? | **Untested**: no local experiment yet meets the two-baseline requirement. |
| **M2 MoE · fills the card** (card measured: RTX 3080) | With MoE nearly filling the 3080, is phase separation stable, and would a dense overlap on top add anything? | **Partly done**: ORNITH-PD-01 has every request routed and a stable 100K record; the 3080 and 3060 single-device speed runs are missing. [Record](results/ornith-1.5-35b-a3b-dual-machine-pd.md) |
| **M3 MoE · does not fit** (RTX 3080, pilot only) | When the total footprint exceeds both control cards, can splitting by layer or by expert keep it running, keep throughput, and keep the output correct? | **Untested**: FLASH-SPLIT-01 is only a nearby layer-split serving record and cannot stand in for this cell. [Pilot record](results/qwen3.8-flash-q4-layer-split.md) |

**Choosing a route by goal.**

- To show two devices computing one model faster together: asynchronous layered Dense Acceleration, backed by 9B-PIPE-01. The claim only holds with a single-device control and both stages genuinely working; do not carry the 9B result over to other models.
- For a faster first token with Prefill and Decode on different devices: independent PD, the 3060 for 9B (9B-PD-01) and the 3080 for 27B (27B-PD-01). The Prefill side must hold its own copy of the model.
- When the model does not fit the small card: split the model across layers, backed by 27B-LONG-01. For long prompts that need more Prefill, or simply for capacity.
- For the largest possible context, or for concurrent Decode: the 3080 computes and the 395 acts as a remote KV store, backed by 27B-KV-01. This is a capacity route; the 395 does no compute, so it is not about being faster.
- To verify MoE PD serving under deep context: see ORNITH-PD-01. Speed claims wait for the matched single-device runs.
- To find the right concurrency tier for a single server: see FLASH-SPLIT-01, which picked C4 on a workload of about 2077 in / 256 out; a different workload must be measured again.

## Experiment Timeline: v1.0 → v2.23

Release numbers show publishing order, not how many experiments were run: v2.1 to v2.4 are four checkpoints of one 9B pipeline experiment, and among the later releases some are real new experiments while others only file data or fix wording. Each phase heading names the card used. The figures are not repeated here; see the two tables above.

### Phase 1: v1.0 → v2.4 · RTX 3060 12GB · 9B goes from "can they hand off at all" to "faster than the fastest single card"

First "can it work", then "is it fast". Across this phase Prefill climbs steadily: **1452.29 → 1865.08 → 1893.87 → 1999.51 → 2129.69 tok/s**, and every step can be traced to a specific change.

| Release | Question | What was done | How the conclusion moved |
| --- | --- | --- | --- |
| v1.0 | Can CUDA Prefill hand its state to Vulkan Decode? | 9B Q6_K; the RTX 3060 prefills a varying share of the tokens, compared with the 395 alone (9B-PD-01). | The handoff works: first token sooner, Prefill higher, Decode unharmed. The devices still run one after the other, which leads to pipelining. |
| v2.1 | The devices take turns idling; can micro-batches be staggered? | First fused-pipeline checkpoint, same conditions. | Staggering works; Prefill passes the RTX 3060 alone for the first time. |
| v2.2 | Does the gain repeat? | Further tuning of the overlap. | Run-to-run variation shrinks; the gain holds. |
| v2.3 | Can rebalancing the two stages improve both ends at once? | Fine-tuned the load split; Decode recorded for the first time. | Both metrics improve together. |
| **v2.4** | With both devices working, can the pair beat the fastest single card while output quality stays within bounds? | Final pipeline calibration, filed as 9B-PIPE-01. In the same period: a 27B IQ3 layer-split check (later filed under 27B-LONG-01), the DGX Spark external reference (EXT-DGX-01), and the Chinese mirror. | Dense Acceleration works at this tier, and the Dense Region gets its name. EXT-DGX-01 stays background. |

### Phase 2: v2.5 → v2.10 · switch to the RTX 3080 20GB · model scales up to 27B

The 9B method moves to a larger model and a stronger card and meets five new problems: splitting the model across layers, PD while serving, the 395 as a remote KV store, auditing speculative-decode data, and an external machine as reference. The phase ends with a rule: at this size, pick the route by goal — phase separation for a faster first token, remote KV for capacity, a layer split for very long prompts. None of the three is Dense Acceleration.

| Release | Question | What was done | How the conclusion moved |
| --- | --- | --- | --- |
| v2.5 | Can the 3080 take all Prefill and the 395 all Decode, and carry 27B Q4 in a real service? | Moved to the RTX 3080 20GB; six concurrency tiers C1–C6, each on the split and solo paths (27B-PD-01). After removing the per-ubatch RPC sync, Prefill went from **683.2** to **1000.6 tok/s**. | PD works while serving: Prefill does not fall off as concurrency rises and Decode is unharmed. The sync overhead is identified as the biggest waste at that point. |
| v2.6 | The 3060 IQ3 run and the 3080 Q4 run are two disconnected 27B records; how do they merge? | Merged into one 27B section; released the 3080 router v1.1 and the live v1.2: with both ends no longer copying out a 524 MiB recurrent-state checkpoint on every prompt, the 3080 side's serving Prefill went from **1000.6** to **1210.6 tok/s** (98.5% of the 3080's raw figure) and the 395 solo control went from 207.2 to 307.1 at the same time; added the natural-language audit on the 395 (27B-DRAFT-AUDIT-01). | Prefill-first and Decode-first profiles are told apart, and a rule is set: a high score on repetitive text does not represent real text. |
| v2.7 | The most important result is buried under implementation detail. | Reordered the presentation only; nothing new measured. | Route selection moves ahead of parameter detail. |
| v2.8 | In configurations C and D, "who computes" and "how much fits" are being mixed up. | Corrected the attribution; nothing new measured. | All compute is on the 3080 and the 395 only stores KV; remote KV is filed as a capacity route, not a speed route. |
| v2.9 | The C6 peak of 27B-D and its scope need correcting. | Fixed the figure; no new experiment. | The number and its conditions now match. |
| v2.10 | The C, D, and DGX Spark figures are scattered. | Merged them into one comparison table and listed the DFlash2 draft head as its own category; nothing new measured locally. | External results sit in their own column and are no longer mixed with local data. |

### Phase 3: v2.11 → v2.14 · still the RTX 3080 20GB · extends to MoE models and a single-server layer split

Two questions: with an MoE model, can Prefill and Decode still be attributed to the right device; and with a single server, at which concurrency tier does throughput stop rising. The answers: MoE PD runs stably and the two stages stay attributable, but no single-device figure exists to compare against; the Flash layer split finds C4 as its best tier.

| Release | Question | What was done | How the conclusion moved |
| --- | --- | --- | --- |
| v2.11 | With MoE, do stage ownership and 100K long-context stability still hold? | Ornith-1.5-35B-A3B two-machine PD stress test: the 3080 takes all Prefill, the 395 all Decode (ORNITH-PD-01). | Every request goes through and the two stages stay attributable. No single-device figure in the same run, so no speed claim. |
| v2.12 | Can a figure derived from total wall-clock time be credited to a device? | Metric governance; removed the derived Decode display. Nothing new measured. | Only metrics that can be credited to one device are published. |
| v2.13 | Other experiments may have leaked into the multi-run record. | Pinned the evidence to r337. Nothing new measured. | Ornith data comes from this one source only: 3080 pure Prefill, 395 pure Decode. |
| v2.14 | With one server driving both devices, where does throughput level off? | r374 Qwen3.8-Flash Q4: a single llama-server using the 3080 and the 395 together, C1–C6 (FLASH-SPLIT-01). | All scored requests succeed and C4 is the best tier. No single-device control, so not proof of a speedup. |

### Phase 4: v2.15 → v2.23 · no hardware change and no new tests, only filing the data properly and rewriting the text

This phase ran no new tests and produced no new number. It did two things: put every figure where it belongs, and make the text readable.

| Release | Question | What was done | How the conclusion moved |
| --- | --- | --- | --- |
| v2.15 | The same figures appear in several places and the purposes of the experiments are tangled. | Experiment governance; added `data/experiment-index.csv` and restored the v2.4 Decode figure to `data/benchmark-results.csv`. | Every experiment has a fixed ID, one record, and a stated scope. |
| v2.16 | The homepage was cut too thin; outsiders could not see the gains. | Put the key figures back on the homepage. | Key numbers, gains, and the route table return. |
| v2.17 | The data is visible, but the architecture and the six-cell storyline are not. | Restructured. | Architecture moves to the top; the six-cell matrix is built with its progress marked. |
| v2.18 | The progression between releases cannot be read, and the sentences are awkward. | Rewrote the narrative. | The timeline moves forward in four phases, one row per release. |
| v2.19 | The 3060 and 3080 lines were told together, and two-device data came before single-device figures. | Split the two lines and moved the single-device figures ahead. | Each card has its own section; single-device figures precede all two-device data. |
| v2.20 | Many sentences were forced jargon, in both languages. | Rewrote the wording throughout. | Everyday phrasing and shorter sentences; structure, numbers, and conclusions untouched. |
| v2.21 | The architecture diagram at the top failed to display in some browsers. | Found the cause — whole-page browser translation rewrites keywords inside the code block, so GitHub no longer receives valid source — and replaced the diagram with a static image, source folded underneath. | The diagram no longer depends on the browser's translation setting. |
| v2.22 | The same figures appeared many times from top to bottom, and the point of the experiments was lost. | Trimmed and restructured: complete data appears once, in "Conclusions and Core Data"; the other sections keep only intent, routes, and conclusions. | Nothing new measured, no number changed. The page now shows at a glance how far the pair is ahead of the card alone and the host alone. |
| **v2.23** | The 395's 207.2 Prefill at the 27B tier looks too low; is it a typo? | Went back to the original experiment log and checked line by line: 207.2 is the measured C1 solo result under the v1.0 serving method with the checkpoint copy on, 307.1 is the result with it off, and 313.28 is `llama-bench` on IQ3; every figure above 1200 belongs to the 3080 side. Added the three methods and the 3080 side's 683.2 → 1000.6 → 1210.6 climb to the homepage. | Nothing new measured; no existing number changed. Readers can now tell the 395's 207 from the 3080's 1210. |

Things that are easy to double-count: 27B-C and 27B-D are two configurations of the single experiment 27B-KV-01, not two experiments; the natural-language run on the 395 belongs to 27B-DRAFT-AUDIT-01; DGX Spark is a public result from someone else's machine, kept as background.

## What the Data Can and Cannot Prove

- **Runs**: the request completes, the state hands over, and every metric can be credited to a device — only then may we write "this configuration runs".
- **Faster**: model, quantization, workload, and metrics all match, and a comparable single-card or single-host figure exists — only then may we write "faster than the control". On this page only the four rows under "Experiments with a control" qualify.
- **Fits and serves**: the workload completes with memory and stability data but no comparable speed control — write "fits" or "serves to this level", never "accelerates". The rows under "Experiments without a control" all belong here.
- **Remote KV is not Dense Acceleration**: a configuration where the 395 only stores KV and does no compute is a capacity route.
- **A speculative-decode point test cannot replace a random-seed stress test**: 27B-DRAFT-AUDIT-01 sets a rule and claims no gain.
- **External references are background**: DGX Spark is someone else's public measurement with its method and source stated; it cannot serve as a local control.
- Every figure must say whether it came from the RTX 3060 or the RTX 3080; data that differ in model, quantization, engine, prompt, or connection may not be joined into one ranking table. A release number is not an experiment ID and not a data source.

## Roadmap: v3.0 and v4.0

| Release | Question raised by current results | Plan and completion criteria |
| --- | --- | --- |
| **v3.0** | v2.4 is complete only at 9B on the 3060; most of the six cells are still open. | Fill in the 3060 and 3080 baselines under one workload standard, then carry one-to-one Dense Acceleration to more large-memory hosts and more small-VRAM cards. Done when the Dense Region lines up, the Prefill and Decode gains hold, and scheduling is stable. |
| **v4.0** | Once one-to-one is stable, can one accelerator serve several large-memory hosts at once? | Study one-to-many scheduling, resource isolation, fair sharing, failure recovery, and the scaling limit. Done when the gain reproduces as hosts are added and the per-host slowdown stays acceptable. |

## Detailed Reports and Data

This page quotes only the few key figures per experiment; the full rows, the metric definitions, and the missing fields live in the records and CSV files below. Use the matching CSV for your own calculations, and do not combine data from different experiment IDs unless the record states that a comparable control exists.

| ID | Accelerator | Question | Record | CSV |
| --- | --- | --- | --- | --- |
| 9B-PD-01 | RTX 3060 12GB | Can CUDA Prefill hand its state to Vulkan Decode? | [v1.0 independent PD](results/v1.0-independent-pd.md) | [CSV](data/benchmark-results.csv) |
| 9B-PIPE-01 | RTX 3060 12GB | Can both devices compute one model together through an asynchronous layered pipeline? | [v2.4 fused layer pipeline](results/v2.4-fused-layer-pipeline.md) | [CSV](data/benchmark-results.csv) |
| 27B-LONG-01 · 27B-PD-01 · 27B-KV-01 · 27B-DRAFT-AUDIT-01 | RTX 3060 / RTX 3080 | The 27B layer split, PD while serving, remote KV, and the speculative-decode audit | [Qwen3.8-27B two-machine PD](results/qwen3.8-27b-dual-machine-pd.md) | [CSV](data/qwen27b-local-results.csv) |
| ORNITH-PD-01 | RTX 3080 20GB | Can an MoE model keep Prefill and Decode attributable while surviving 100K stress from C1 to C6? | [Ornith two-machine PD](results/ornith-1.5-35b-a3b-dual-machine-pd.md) | [CSV](data/ornith35a3b-local-results.csv) |
| FLASH-SPLIT-01 | RTX 3080 20GB | With one server split across CUDA and Vulkan, where does throughput level off? | [Qwen3.8-Flash Q4 layer split](results/qwen3.8-flash-q4-layer-split.md) | [CSV](data/qwen38flash-q4-local-results.csv) |
| EXT-DGX-01 | External | DGX Spark public figures, background only | [DGX Spark community control](results/dgx-spark-community-control.md) | [CSV](data/dgx-spark-community-controls.csv) |

The mapping from experiment IDs to legacy labels is in [data/experiment-index.csv](data/experiment-index.csv); all records are under [results/](results/). The [changelog](CHANGELOG.md) records what each release changed or corrected.
