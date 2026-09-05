# Qwen3.8-27B Heterogeneous Experiments — Contracted, Separate Envelopes

[简体中文](qwen3.8-27b-dual-machine-pd.zh-CN.md)

This file is the single human-readable source of record for the local Qwen3.8-27B measurements. Legacy labels 27B-A/B/C/D are retained as aliases, but they are not four directly rankable experiments.

## Record map

| Stable ID | Legacy label | Experimental unit | Evidence class |
| --- | --- | --- | --- |
| **27B-LONG-01** | 27B-A | 3060 + 395 IQ3 long-prompt layered placement | Matched-host improvement |
| **27B-PD-01** | 27B-B | 3080 full Prefill → 395 full Decode service handoff | PD feasibility and scheduler envelope |
| **27B-KV-01** | 27B-C / 27B-D | 3080 full compute + 395 KV-only storage; C/D are two profiles | Capacity and serving envelope |
| **27B-DRAFT-AUDIT-01** | 395 natural-language audit | Workload-sensitivity check for speculative Decode | Validation guardrail |

## 27B-LONG-01 — layered long-prompt feasibility

### Experimental contract

| Item | Definition |
| --- | --- |
| Primary question | When the full 27B model does not fit on the RTX 3060, can a 3060 layer stage improve long-prompt execution over the 395 host? |
| Model / envelope | Qwen3.8-27B UD-IQ3_XXS, 27.32B parameters, 10.17 GiB, 3.06 bpw. |
| Matched control | AI Max+ 395-only at the same model, quantization, and prompt depth. |
| Changed factor | Add the RTX 3060 layer stage; prompt depth is swept separately. |
| Decision metrics | Completion at each prompt depth, Prefill, and tg64 Decode. |
| Claim boundary | Supports improvement versus the matched 395 path; no standalone 3060 full-model control exists, and this is not a Q4 result. |

### Results

| Workload | AI Max+ 395 only | Layered 3060 + 395 | Observed change |
| --- | ---: | ---: | ---: |
| pp4096 | 313.28 tok/s | **658.52 tok/s** | +110% |
| pp65536 | 136.69 tok/s | **319.10 tok/s** | +133% (2.33×) |
| pp98304 | Timed out after 900 s | **225.10 tok/s** | Completed; TTFT ≈ 437 s |
| tg64 | 18.26 tok/s | **19.57 tok/s** | About +7% |

**Answer:** the layered path passes the long-prompt feasibility gate and improves the matched 395-only path. It does not establish a full-model 3060 speedup because that control cannot fit.

## 27B-PD-01 — full-Prefill / full-Decode service handoff

### Experimental contract

| Item | Definition |
| --- | --- |
| Primary question | Can an RTX 3080 Prefill service transfer KV state to a 395 Decode service reliably from C1 through C6? |
| Model / envelope | Qwen3.8-27B Q4_K_M, about 17.66 GiB, 255 output tokens, q4_0 KV. |
| Primary path | RTX 3080 CUDA0 full Prefill → shared-memory KV handoff → AI Max+ 395 Vulkan full Decode. |
| Matched control | The same request served on the 395-only path. |
| Changed factor | Router split path versus 395 solo; concurrency is swept C1–C6. |
| Decision metrics | TTFT, Prefill, Decode, KV migration time, and request success. |
| Claim boundary | Historical two-instance PD feasibility; it is not the later KV-only architecture. |

### C1–C6 service matrix

Values left/right are PD / 395-solo.

| C | TTFT ms | Prefill tok/s | Decode tok/s | KV migration ms |
| --- | ---: | ---: | ---: | ---: |
| 1 | **1073 / 4825** | **1000.6 / 207.2** | 38.75 / 36.33 | 71 |
| 2 | 1680 / 9649 | 1008.4 / 103.6 | 27.94 / 24.36 | 76 |
| 3 | 2680 / 15321 | 1014.5 / 65.3 | 21.45 / 21.17 | 68 |
| 4 | 3255 / 19763 | 1010.9 / 50.6 | 18.34 / 14.44 | 72 |
| 5 | 4457 / 22460 | 1015.5 / 44.5 | 13.10 / 11.48 | 72 |
| 6 | 5009 / 22303 | 1014.3 / 44.8 | 9.67 / 8.82 | 70 |

The separate RTX 3080 hardware checks were pp1024 1228.53, pp4096 1203.06, and tg64 33.08 tok/s. They are device references, not substitutes for the service-path control above.

**Answer:** the experiment passes the state-handoff and C1–C6 service-feasibility gate. It measures serial phase ownership; it does not show both devices computing the same phase.

## 27B-KV-01 — remote-KV serving profiles

### Experimental contract

| Item | Definition |
| --- | --- |
| Primary question | What capacity and serving trade-off results when the RTX 3080 performs all Prefill and Decode while the 395 stores remote KV? |
| Compute ownership | RTX 3080: all compute. AI Max+ 395: KV-only storage, no Prefill and no Decode. |
| Profiles | C = Prefill-first; D = Decode-first. They are two profiles of one experiment, not two experiments. |
| Latest matched control | Not recorded: there is no same-run 3080 no-remote-KV series at the same per-stream 1M context. |
| Changed factors | Scheduler profile and C1–C6 concurrency. |
| Decision metrics | Context capacity, Prefill, single-stream Decode, aggregate Decode, and missing-field coverage. |
| Claim boundary | Capacity and serving envelope only. This topology is not Dense Acceleration because the 395 contributes no model compute. |

### Profile C — Prefill-first series, shown once

This series was first published under the router-v1.1 label and later reissued with corrected compute attribution as remote-KV-latest. It is one measured series, so the numbers are not repeated as a second experiment. Values left/right are the primary path / historical 395-solo control captured with the original series.

| C | TTFT ms | Prefill tok/s | Single Decode tok/s | Aggregate Decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| 1 | 829 / 3257 | **1210.6 / 307.1** | 38.50 / 36.56 | 33.55 / 25.00 |
| 2 | 1283 / 3279 | 1204.8 / 304.9 | 28.51 / 20.17 | 45.57 / 28.39 |
| 3 | 1731 / 8060 | 1205.5 / 124.1 | 22.51 / 17.39 | 52.31 / 32.74 |
| 4 | 2195 / 6897 | 1199.5 / 145.0 | 19.66 / 14.24 | 51.30 / 30.15 |
| 5 | 2668 / 7609 | 1194.4 / 131.4 | 16.07 / 8.27 | 61.64 / 25.12 |
| 6 | 3122 / 8354 | 1197.2 / 119.7 | 14.68 / 8.34 | 63.84 / 27.25 |

### Profile D — Decode-first historical scheduler matrix

This router-v1.2 matrix is a historical scheduler envelope. Values left/right are router / 395-solo; aggregate columns apply to the router unless a slash is shown.

| C | TTFT ms | Token 1→2 ms | Prefill single / aggregate | Decode single tok/s | Aggregate Decode | End-to-end aggregate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 921 / 3125 | 96 | 1090 / 1079 | 63.2 / 24.5 | 58.2 | 41.2 / 15.4 |
| 2 | 1420 / 3254 | 396 | 1081 / 1043 | 24.1 / 17.6 | 38.0 | 33.6 / 20.3 |
| 3 | 1917 / 5512 | 575 | 1080 / 1030 | 22.5 / 12.3 | 45.8 | 41.5 / 16.2 |
| 4 | 2424 / 8602 | 739 | 1077 / 1017 | 17.8 / 12.0 | 44.3 | 41.3 / 21.3 |
| 5 | 2911 / 7603 | 836 | 1082 / 1018 | 13.9 / 8.1 | 45.4 | 42.8 / 19.6 |
| 6 | 3404 / 8319 | 1060 | 1082 / 1016 | 13.2 / 7.4 | 48.0 | 45.7 / 23.3 |

### Profile D — later summary-only envelope

The later corrected summary records only C1 and the C6 aggregate-Decode endpoint. It is not spliced into the historical matrix, and missing C2–C5 values are not interpolated.

| C | Prefill | Single Decode | Aggregate Decode |
| --- | ---: | ---: | ---: |
| C1 | 1090 tok/s | 63.2 tok/s | 63.2 tok/s |
| C6 | Not recorded | Not recorded | 116.3 tok/s |

**Answer:** these profiles establish a 1M-per-stream remote-KV serving envelope and expose scheduler trade-offs. Without a matched 3080-only run at the same context, they do not prove that remote KV accelerates the 3080.

## 27B-DRAFT-AUDIT-01 — natural-language guardrail

### Experimental contract

| Item | Definition |
| --- | --- |
| Primary question | Do high speculative-Decode results on repetitive text represent natural-language service behavior? |
| Trigger evidence | Repetitive-text point runs on the 395 reported 35.0–38.5 tok/s at 100% draft acceptance. |
| Audit workload | Direct natural-language requests on the 395 DFlash2 path, C1/C2/C3/C4/C6. |
| Decision metrics | Single-stream and aggregate Decode, acceptance, milliseconds per step, and tokens per step. |
| Claim boundary | A workload-sensitivity audit, not an architecture comparison or matched acceleration test. |

Separate RTX 3080 point checks also showed workload sensitivity: natural-language 42.7 tok/s at 43% acceptance versus code 67.4 tok/s at 87%. They are point observations, not rows in the matrix below.

| C | Single Decode tok/s | Aggregate Decode tok/s | Acceptance % | ms/step | tokens/step |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 12.1 | 10.9 | 17.7 | 178 | 2.17 |
| 2 | 9.9 | 16.4 | 24.3 | 262 | 2.61 |
| 3 | 7.6 | 19.3 | 21.1 | 318 | 2.42 |
| 4 | 5.8 | 19.5 | 17.9 | 378 | 2.19 |
| 6 | 4.0 | 19.8 | 18.3 | 548 | 2.22 |

**Answer:** the repetitive-text headline does not generalize to natural-language service. Production claims must use the natural-language audit or a workload-matched test.

## External context and boundaries

- DGX Spark data has moved to the single [external community-control record](dgx-spark-community-control.md). Different quantization, engine, KV type, prompt depth, and topology prevent a controlled local ranking.
- 27B-LONG-01 uses IQ3 and a 3060; the other primary records use Q4 and a 3080. Do not merge their rates.
- Historical scheduler matrices and later corrected summaries remain separate envelopes. A corrected attribution is not a new experiment, and a missing tier is not an invitation to interpolate.
- Machine-readable source: [qwen27b-local-results.csv](../data/qwen27b-local-results.csv). Its release-snapshot aliases are mapped to stable IDs in [experiment-index.csv](../data/experiment-index.csv).

