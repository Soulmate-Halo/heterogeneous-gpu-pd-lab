# ORNITH-PD-01 — Ornith-1.5-35B-A3B Role-Attributed PD Stress Envelope

[简体中文](ornith-1.5-35b-a3b-dual-machine-pd.zh-CN.md)

## Experimental contract

| Item | Definition |
| --- | --- |
| Primary question | Can a MoE model keep Prefill and Decode ownership independently attributable while completing short and 100K C1–C6 PD stress? |
| Topology under test | RTX 3080 performs all Prefill; AI Max+ 395 performs all Decode after KV migration. |
| Matched speed control | Not recorded. There is no same-run single-node baseline for the scored workloads. |
| Changed factors | Prompt depth (1000 versus 100000 input tokens) and concurrency C1–C6. |
| Decision metrics | Route success, `n_reuse`, 3080 Prefill in its own window, and 395 Decode in its own window. |
| Pass boundary | All scored requests complete on the PD route with phase metrics attributable and missing fields left unfilled. |
| Does not prove | End-to-end acceleration, a standalone comparison, or superiority over dense 27B. |

This record is limited to the r337 Ornith-1.5-35B-A3B dual-machine Prefill/Decode (PD) experiment added in v2.11: the RTX 3080 is Prefill-only and the AI Max+ 395 is Decode-only. **Ornith-1.5-35B-A3B is a MoE (35B total, A3B active-per-token naming). It must not be ranked directly against the Qwen3.8-27B dense rows in [qwen3.8-27b-dual-machine-pd.md](qwen3.8-27b-dual-machine-pd.md):** different model family, different architecture, different per-token active compute, and different quantization.

## Model and topology

| Item | Value |
| --- | --- |
| Model | Ornith-1.5-35B-A3B (MoE, 35B total, A3B active-per-token) |
| GGUF | Ornith-1.5-35B-A3B-IQ4_XS |
| Architecture | qwen35moe, 40 layers, full_attention_interval=4 |
| Layer mix | 10 full-attention layers + 30 Gated DeltaNet layers |
| Pure Prefill node during the scored matrix | RTX 3080, CUDA: all Prefill, no Decode; batch 4096, ubatch 4096, ctx 114688 |
| KV migration | /dev/shm/kvxo |
| Pure Decode node during the scored matrix | AMD Ryzen AI Max+ 395, Vulkan1: all Decode, no Prefill; ctx 655360 |
| Draft head | Qwen3.6-35B-A3B-DFlash-Q4_K_M, speculative draft n_max 6 |
| Stress result | 42/42 passed, route=pd, n_reuse=0 |
| Post-benchmark online state | Services restored to ctx 8192 / 32768; batch/ubatch 4096 and draft n_max 6 retained |

## Metric boundary — read before the tables

The published 100K table keeps the **395 pure-Decode aggregate** (C1–C6: 23.33–148.20 tok/s), measured only over the 395 Decode window. The 3080 performs no Decode work. The derived whole-stage wall-clock series is excluded from the presentation and its CSV field is set to NA.

The short-task stage (1000 in / 128 out) publishes 3080 pure Prefill only because a separately timed 395 pure-Decode rate was not recorded.

## Short task — 1000 input / 128 output

All values in tok/s, measured over six concurrency tiers in the same PD service.

| C | 3080 aggregate Prefill |
| --- | ---: |
| C1 | **4017.46** |
| C2 | 3947.64 |
| C3 | 3924.83 |
| C4 | 3913.85 |
| C5 | 3906.09 |
| C6 | 3943.88 |

## 100K stage — 100000 input / 128 output

All values in tok/s. The rightmost column measures only the 395 decode window.

| C | 3080 pure Prefill aggregate | 395 pure Decode aggregate |
| --- | ---: | ---: |
| C1 | **2895.53** | **23.33** |
| C2 | 2826.07 | 53.37 |
| C3 | 2796.34 | 75.20 |
| C4 | 2795.28 | 103.33 |
| C5 | 2793.56 | 123.09 |
| C6 | 2793.24 | 148.20 |

The 100K pure Prefill holds within about 3.6% across C1–C6 (2895.53 down to 2793.24 tok/s), while the 395 pure-Decode aggregate grows roughly linearly with concurrency from 23.33 to 148.20 tok/s.

## Stress outcome

42/42 concurrent requests succeeded, all routed through the PD path (`route=pd`), with `n_reuse=0` — every stream ran a fresh prefill, so no cached-KV reuse inflated any number above.

PD restore restores the main-model KV but not the dFlash draft-slot position. A direct jump from the short workload to 100K caused a position-discontinuity failure, so all six draft slots were first advanced to 100K. That warm-up was excluded; the scored C1–C6 rows were then all successful.

## Not recorded (no extrapolation)

The source record does not contain the following for this experiment; the CSV and this document leave them empty or mark them not recorded, and nothing is interpolated:

- 100K TTFT — not recorded
- 100K single-stream Decode — not recorded
- short-task 395 pure-Decode speed — not recorded
- KV migration milliseconds — not recorded
- dFlash draft acceptance rate — not recorded

## Boundaries

- With no matched standalone run, the tables characterize role attribution and stability; they are not a causal speedup claim.
- **MoE vs dense**: Ornith-1.5-35B-A3B activates a far smaller per-token compute than the Qwen3.8-27B dense model; every figure above is an independent MoE envelope and cannot be collapsed into a ranking with the 27B rows.
- The 395 pure-Decode aggregate covers the Decode window only; all Decode runs on the 395, and the metric excludes Prefill and KV restore time.
- The draft head (Qwen3.6-35B-A3B-DFlash-Q4_K_M) runs alongside the IQ4_XS main model; no acceptance-rate figure is published, so the draft's realized contribution cannot be quantified.
- The short-task and 100K calibers differ in input length and must be read within their own tables only.

Data source: the completed `r337_hot_results.json` / `r337_hot_bench.log` run recorded for v2.11 (numbers taken verbatim; nothing extrapolated).

Machine-readable data: [ornith35a3b-local-results.csv](../data/ornith35a3b-local-results.csv).
