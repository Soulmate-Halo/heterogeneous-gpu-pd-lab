# DeepSeek-V4.1-Flash: up to 7.82× Prefill with six GPUs, V1–V7 and TP4 / H20 comparisons

[中文](deepseek-v4.1-flash-six-gpu-v1-v7.zh-CN.md) · [Home](../README.md) · [Local CSV](../data/deepseek-v4.1-flash-six-gpu-v1-v7.csv) · [Sanitized evidence](../data/deepseek-v4.1-flash-six-gpu-v1-v7-evidence.json)

**Adding two RTX 6000Dpro GPUs to four DGX Sparks primarily accelerates Prefill: matched C8 input throughput rises from 1720.90 to 7812.43 tok/s at 8K (4.54×), and from 1699.75 to 13300.06 at 32K (7.82×).** Dual 6000D handles Prefill; four Sparks replay the tail and Decode. Mean 32K time to first text falls **86.6%**. The September 17 formal matrix completed 100/100 requests; September 18 controls and the longer-output diagnostic completed 76/76. Completion checks are not full model-accuracy or generated-code evaluations.

## Primary results: Prefill throughput and time to first text

Both paths reuse the same four-Spark D service with budget 1536, DSpark K=5 and identical Engram/CUDA Graph settings. Each path and input length has two C8 batches with 512 output tokens per request. Tables use batch means; code templates match but random prefixes differ.

| Input / output / concurrency | Four Spark TP4: input tok/s | Dual 6000D + four Sparks: input tok/s | Speedup |
| --- | ---: | ---: | ---: |
| 8192 / 512 / C8 | 1720.90 | **7812.43** | **4.54×** |
| 32768 / 512 / C8 | 1699.75 | **13300.06** | **7.82×** |

**Timing:** Prefill aggregate input = all input tokens / (latest first text − earliest request start), including Prefill, transfer, replay and waiting. This measures input-serving performance, not isolated kernels.

| Input / output / concurrency | TP4 mean TTFT | Six-GPU mean TTFT | Wait reduction |
| --- | ---: | ---: | ---: |
| 8192 / 512 / C8 | 21.711 s | **5.103 s** | **76.5%** |
| 32768 / 512 / C8 | 86.541 s | **11.596 s** | **86.6%** |

## Six GPUs, standalone TP4 and eight H20 GPUs: long-input comparison

| Hardware / runtime | Input / output / concurrency | Prefill aggregate input tok/s | Time to first text (statistic) |
| --- | --- | ---: | --- |
| Four Spark TP4 / vLLM | 8192 / 512 / C8 | 1720.90 | Mean 21.711 s |
| **Dual 6000D + four Sparks / vLLM PD** | 8192 / 512 / C8 | **7812.43** | **Mean 5.103 s** |
| Eight H20-3e / SGLang | 8192 / 128 / C8 | Not reported | P95 13.650 s |
| Eight H20-3e / vLLM | 8192 / 128 / C8 | Not reported | P95 9.194 s |
| Four Spark TP4 / vLLM | 32768 / 512 / C8 | 1699.75 | Mean 86.541 s |
| **Dual 6000D + four Sparks / vLLM PD** | 32768 / 512 / C8 | **13300.06** | **Mean 11.596 s** |
| Eight H20-3e / SGLang | 24576 / 128 / C32 | Not reported | P95 154.683 s |
| Eight H20-3e / vLLM | 24576 / 128 / C32 | Not reported | P95 102.401 s |

The [H20 source](https://aik8s.run/ai-k8s/practices/deepseek-v41-flash-h20-day0/) reports full-wall output and TTFT, not matching Prefill input throughput; its 63.64 / 97.63 tok/s figures are output rates. Local values are two-batch means; H20 latency is the median of three per-round P95s. Output lengths, concurrency, speculation and statistics differ, so no H20 speedup ratio is inferred.

## Secondary results: Decode and full-wall output

The original Decode results below describe overall delivery gains; Prefill acceleration and shorter first-text waiting are the primary findings. Rates are tok/s.

| Input / output / concurrency | Four Spark TP4 alone | 2×6000D + 4×Spark | Ratio |
| --- | ---: | ---: | ---: |
| 8192 / 512 / C8 | 80.32 | **160.27** | **2.00×** |
| 32768 / 512 / C8 | 27.71 | **123.25** | **4.45×** |

The aggregate generation window includes late requests still prefilling, replaying or waiting. These are serving-efficiency gains, not a 4.45× speedup of each Decode kernel.

| Input / output / C | TP4 full-wall output | Six-GPU full-wall output | TP4 mean TTFT s | Six-GPU mean TTFT s |
| --- | ---: | ---: | ---: | ---: |
| 8192 / 512 / C8 | 73.12 | **151.87** | 21.711 | **5.103** |
| 32768 / 512 / C8 | 24.56 | **111.87** | 86.541 | **11.596** |

**Historical standalone TP4 remains a separate baseline.** The September 13 four-Spark SGLang + DSpark recipe used 2048-token chunks and eight running slots. At C8/C12 it reported aggregate Decode **140.31/138.19** and full-wall output **131.85/133.16**, using **512 input / 256 output**. Its separate **8192 input / 1 output** Prefill tests gave **3232.70/3261.25**. These are not matched controls for the current long-input workload.

## Deployment evolution V1 → V7

These are deployment stages, separate from public research release **v1.6**. V1 is a retrospective label for the initial six-GPU pipeline; no separate V1 frozen-image manifest was found. V2–V5 are recorded experiment stages; V6/V7 have recovery materials. Early remote Engram stages used an additional helper host, so their hardware costs are not identical. Exact layer allocations remain private.

| Deployment | Date | Change | Measured outcome |
| --- | --- | --- | --- |
| **V1** | 2026-09-14–15 | Initial six-GPU TP2×PP3 pipeline; per-row Engram disk reads | Prefill plateaued around 375–380 tok/s; lookup and pipeline waits dominated. |
| **V2** | 2026-09-15 | Resident remote Engram gather, followed by NCCL tuning | 8K Prefill 225.5→2082.0; tuned 32K Prefill 3621.6 tok/s, Decode about 25.6. |
| **V3** | 2026-09-15 | Move the 6000D pair to the final stage; enable DSpark with a special layer allocation | 30K Prefill 4883; ordinary-text single-stream Decode 41.4–42.8 tok/s; limited memory headroom. |
| **V4** | 2026-09-15 | Disable DSpark, reduce chunk size and reserve KV memory for long context | 12×149437-token inputs completed; C1 Prefill 5062 and Decode 35.0 tok/s; observed running peak only 5. |
| **V5** | 2026-09-16 | Cross-stage KV mirroring and revised placement restore DSpark and 8192-token chunks | 12 long-input requests with 512 output tokens each completed in 215.19 s; C1 150K Prefill about 3860 tok/s. |
| **V6** | 2026-09-16–17 | Switch to PD: dual 6000D Prefill plus four-Spark TP4 Decode; repair P2P, GPU gather and NIXL | P single-request stage rate about 13K–16.5K tok/s; end-to-end input still pays handoff and tail replay; golden checkpoint frozen. |
| **V7** | 2026-09-17–18 | Select code-serving settings, D prefetch/GPU staging and tail=1280; package images and add a watchdog | 100/100 requests in the 8K/32K × C1/C4/C8/C12 matrix; matched C8 Prefill input throughput is 4.54×/7.82× direct TP4; mean 32K TTFT falls 86.6%. |

Adding compute first required removing lookup, interconnect and pipeline stalls. In V7, **the dual 6000D handles input Prefill; four Sparks replay the tail and perform full-model generation**. P does not perform the first half of every subsequent Decode step. Special layer allocation belongs to the earlier PP experiments; V7 is a PD deployment.

## V7 code matrix: 8K / 32K × C1 / C4 / C8 / C12

512 output tokens/request; two batches per cell, 16 batches and 100 requests total. Arithmetic means of the two batches, temperature=0, fixed output length, random prefixes and zero additional local prefix-cache hits on P/D. Rates are tok/s.

| Input | C | Aggregate input | Aggregate Decode | Full-wall output | Mean TTFT (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 1 | 5678.91 | 72.96 | 60.62 | 1.443 |
| 8192 | 4 | 7184.36 | 124.31 | 114.21 | 3.075 |
| 8192 | 8 | 7543.46 | 158.78 | 150.45 | 5.218 |
| 8192 | 12 | 7371.07 | 177.25 | 170.02 | 7.520 |
| 32768 | 1 | 9757.62 | 73.77 | 49.75 | 3.367 |
| 32768 | 4 | 12804.59 | 103.46 | 88.96 | 6.854 |
| 32768 | 8 | 13173.51 | 121.70 | 110.97 | 11.713 |
| 32768 | 12 | 13814.56 | 135.73 | 126.59 | 16.278 |

**12K+ aggregate input is verified:** 32K/C4, C8 and C12 yield **12804.59, 13173.51 and 13814.56 tok/s**. Their paired aggregate Decode rates are **103.46, 121.70 and 135.73**. Input throughput is not output throughput.

One additional 32K/2048-output/C12 batch measured input **13485.39**, batch Decode **199.39** and full-wall output **194.24**. The **72.692 s** interval with all twelve requests generating measured **235.93 tok/s**. This diagnostic window is not the full-batch metric; only one longer-output batch was tested.

## Eight H20 GPUs: community reference

[Source: same-model community measurements](https://aik8s.run/ai-k8s/practices/deepseek-v41-flash-h20-day0/), retrieved September 18, 2026. **8×H20-3e, TP8/EP8, no DSpark. Output uses full-round time; median of three rounds.**

| Input / output | C | Engine | Full-wall output tok/s | Success |
| --- | ---: | --- | ---: | ---: |
| 8192 / 128 | 8 | SGLang | 63.64 | 48/48 |
| 8192 / 128 | 8 | vLLM | 97.63 | 48/48 |
| 24576 / 128 | 32 | SGLang | 22.79 | 192/192 |
| 24576 / 128 | 32 | vLLM | 36.30 | 192/192 |
| 1024 / 512 | 32 | SGLang | 824.70 | 192/192 |
| 1024 / 512 | 32 | vLLM | 1219.46 | 191/192 |

[External CSV](../data/deepseek-v4.1-flash-six-gpu-v1-v7-h20.csv). Different lengths and concurrency prevent a matched hardware ranking. Use the local full-wall output metric and align workload, speculation and validity requirements first.

**The project owner additionally reports “12K+ Prefill and 1200 aggregate Decode”.** No corresponding 1200 tok/s batch, workload or timing evidence was found in the referenced session and archived results. It remains unverified. **The published evidence does not establish higher total throughput than eight H20 GPUs.** A verified 1200 result would still need comparable conditions and cannot be ranked only against a selected lower H20 operating point.

## Current parameters and limits

| Item | V7 |
| --- | --- |
| Model / weights | Original DeepSeek-V4.1-Flash checkpoint; MXFP4 experts / FP8 main dense weights |
| Compute hardware | 4×DGX Spark GB10 + 2×RTX 6000Dpro, with proxy and Engram shard services |
| Runtime | vLLM; TP2 P and TP4 D, NIXL handoff |
| Shared per-step token budget | P=4096; D=1536 |
| Request / context limits | 12 / 153600; configured limits are not a full workload validation |
| KV | auto, native fp8_ds_mla; block=128; P 1.5 GiB and D 6 GiB per rank |
| Speculation | D: DSpark K=5, probabilistic/block, adaptive verification disabled |
| Other | CUDA Graph and Engram GPU stage/prefetch enabled; tail=1280 with block alignment |
| Readiness | At least 90 seconds of continuous backend health, followed by two consecutive content smoke checks |
| Recovery | 16 ARM64/AMD64 service images, five export bundles; weights/dictionaries external; watchdog supplement |

D budget 4096 produced **2/12 garbled requests** in 32K/C12 and was rejected; V7 remains at 1536. A later watchdog acceptance run also exposed a CUDA illegal-memory-access error. Recovery passed content checks, but the root cause remains open. The readiness gate does not fix that error. A complete six-GPU cold restore is not yet validated. These are initial deployment results, not claims of lossless model accuracy or long-term reliability.

## Definitions and evidence

- Aggregate input: all input tokens / (latest first text − earliest request start), including P, handoff, D replay and waiting.
- Aggregate Decode: output tokens after each request's first nonempty packet / (latest last text − earliest first text). Deduct the actual first-packet token count.
- Full-wall output: all output tokens / (latest request end − earliest request start).
- Total input-plus-output throughput requires one shared full wall time: `(all input + all output tokens) / full seconds`. Do not add input and Decode rates with different denominators.
- Dividing input by summed per-request P-prefill durations is not aggregate P throughput when those durations overlap.

The evidence JSON contains sixteen timed batches, usage/timestamps for 100 requests, controls, the longer-output diagnostic, version provenance and source SHA256 values. S1 is the formal matrix; S2 the controls; S3–S6 deployment/recovery records; S7 historical TP4; S8 the overall report. Original SSE streams and generated text remain in the source project. Private endpoints, credentials, precise layer allocation and launch patches are excluded.
