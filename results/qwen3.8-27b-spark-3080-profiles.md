# 27B-SPARK-01 — Qwen3.8-27B Q4_K_M on RTX 3080 20GB + DGX Spark GB10

[简体中文](qwen3.8-27b-spark-3080-profiles.zh-CN.md)

This record is the source of the local 3080 + Spark working profiles. Every Prefill and Decode pair below is the same configuration under the same load. 2K and 8K are never spliced into one cell. "Highest" means the working profile confirmed after this search, not a claim of a global optimum. All rates on the pair are **pair** scores; they are not the 3080 running alone.

## Experimental contract

| Item | Definition |
| --- | --- |
| Primary question | After searching working points on Qwen3.8-27B Q4_K_M / KV q4_0, which Prefill, Decode, and balanced serving profiles does the RTX 3080 20GB + DGX Spark GB10 pair sustain? |
| Devices | RTX 3080 20GB and DGX Spark GB10 computing the same model together. |
| Model / envelope | Qwen3.8-27B, weight quantization Q4_K_M, KV cache q4_0. |
| Confirmed Prefill profile | 8192-token cold prompt, 128 output tokens, no speculation; median of three measured runs after system warmup, excluding the warmup run. |
| Confirmed Decode profile | Same 128-output microbenchmark with DFlash2 on; 2048 and 8192 are separate rows. |
| Balanced profile | 8192-token cold prompt, 256 output tokens, DFlash2, six-slot serving, concurrency C1-C6. |
| Matched control | RTX 3080 standalone at the same model, quantization, KV type, prompt length, and output length. Speculation settings on that control were tuned on that system and are not a one-variable change versus the pair. |
| Not recorded | A 3080-only C1-C6 series on the balanced serving workload. |
| Claim boundary | Prefill on the confirmed Prefill profile is faster than this 3080 standalone. The experiment does not claim a win over every standalone device. It is already listed on the three-hardware-route D2 coverage map on the front page. Performance is compared only against same-condition baselines. It does not rank against community NVFP4 Spark rows. |

## What each profile gives up

- **Highest Prefill confirmed profile.** 8K Prefill rises to **1603.20** tok/s versus 1113.13 on the 3080 standalone (**+44.03%**). Decode falls to **17.62** versus 33.36. There is no speculation. A later live retest of the same profile, three repeats at 8K, is 1601.05 / 17.41; do not quote 1700.
- **Highest Decode confirmed profile.** DFlash2 is on. At 2K the pair is 1153.95 / 63.97 versus 1042.97 / 60.37 on the 3080 standalone with its own DFlash2 tuning. At 8K the pair is 1124.32 / 49.20 versus 1015.15 / 58.36. Prefill is far below the Prefill-confirmed profile. At 8K, pair Decode is not above this 3080 standalone; the two systems used different speculation settings.
- **Balanced profile.** Six-slot serving holds Prefill near 1086-1097 tok/s from C1 to C6. Aggregate Decode is timed on the whole-batch generation window and **includes other streams' Prefill interference**. It is not the 128-output single-stream Decode of the microbenchmark, and it is not a peak Decode figure. There is no standalone C1-C6 control.

Search-stage peaks of 1605.21 Prefill and 67.35 Decode exist in the private search log. They are search samples only. They do not replace the confirmation medians above.

## Main table — paired Prefill and Decode

Statistic: median of three measured runs after system warmup unless noted; the warmup run is excluded. Each request uses a cold prompt with no prefix-cache reuse. Output length 128. No speculation on the Prefill-confirmed rows. DFlash2 on for the Decode-confirmed rows.

| Profile | Workload | Pair Prefill (tok/s) | Pair Decode (tok/s) | 3080 standalone Prefill / Decode |
| --- | --- | ---: | ---: | --- |
| Highest Prefill confirmed | 8192 in / 128 out, no speculation | **1603.20** | 17.62 | 1113.13 / 33.36 |
| Same profile, later live retest | 8192 in / 128 out, no speculation, three repeats | 1601.05 | 17.41 | same 8K no-speculation control |
| Highest Decode confirmed | 2048 in / 128 out, DFlash2 | 1153.95 | **63.97** | 1042.97 / 60.37 |
| Highest Decode confirmed | 8192 in / 128 out, DFlash2 | 1124.32 | 49.20 | 1015.15 / 58.36 |

The live retest is the same working profile as the Prefill-confirmed row, measured again after confirmation. It is not a fourth profile.

## Balanced profile — C1-C6 serving

8192-token cold input, DFlash2, six-slot serving. Each concurrency tier has separate Prefill waves and complete-request waves; complete requests generate 256 tokens. Two wave types times three official rounds times (1+2+3+4+5+6) = 126 official requests across 36 official waves. Warmup rounds are excluded, and 126/126 official requests succeeded. Values are per-tier medians of the three official rounds.

- Aggregate Prefill comes from the separate Prefill waves: total input tokens / (latest first-token arrival minus earliest request start), including a small first-token generation and request overhead.
- Aggregate Decode comes from complete-request waves: total output tokens / (latest completion minus earliest first-token arrival), including interference from other requests still in Prefill.
- End-to-end output, TTFT p95 and batch time also come from complete-request waves. End-to-end output throughput is total output tokens / batch elapsed time. The two wave types are summarized by concurrency tier; these are not paired phase speeds from the same batch of requests.

### Throughput

| C | Aggregate Prefill (tok/s) | Aggregate Decode (tok/s) | End-to-end output (tok/s) |
| --- | ---: | ---: | ---: |
| C1 | 1097.40 | 47.69 | 19.91 |
| C2 | 1094.21 | 31.18 | 19.49 |
| C3 | 1089.37 | 28.04 | 20.47 |
| C4 | 1086.44 | 24.90 | 19.86 |
| C5 | 1087.57 | 24.77 | 20.53 |
| C6 | 1088.06 | 25.55 | 21.62 |

### Latency

| C | TTFT p95 (s) | Batch time (s) |
| --- | ---: | ---: |
| C1 | 7.489 | 12.857 |
| C2 | 19.145 | 26.273 |
| C3 | 28.990 | 37.524 |
| C4 | 38.932 | 51.564 |
| C5 | 48.471 | 62.336 |
| C6 | 58.317 | 71.048 |

## Boundaries

- The 395-host 27B records stay in [qwen3.8-27b-dual-machine-pd.md](qwen3.8-27b-dual-machine-pd.md). This Spark line is already listed on the three-hardware-route D2 coverage map; performance is compared only against same-condition baselines.
- Community NVFP4 Spark figures stay in [dgx-spark-community-control.md](dgx-spark-community-control.md). Different quantization, engine, and method; they are not a third profile and are not ranked against these Q4_K_M rows.
- Do not treat 1603.20 as a 3080-only Prefill. The 3080 standalone no-speculation Prefill at this 8K load is 1113.13.
- Do not treat the balanced Decode column as the same quantity as the 128-output microbenchmark Decode.

Machine-readable data: [qwen27b-spark-3080-profiles.csv](../data/qwen27b-spark-3080-profiles.csv). Index: [experiment-index.csv](../data/experiment-index.csv).
