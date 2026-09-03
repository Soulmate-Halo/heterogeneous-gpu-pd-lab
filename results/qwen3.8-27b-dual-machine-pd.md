# Qwen3.8-27B Heterogeneous Experiments — multiple envelopes, no direct ranking

[简体中文](qwen3.8-27b-dual-machine-pd.zh-CN.md)

This record places the RTX 3060 / IQ3 and RTX 3080 / Q4 27B results in one chronological experiment area. **The article structure is unified; the measurement envelopes are not.** Results with different quantization, topology, prompts, output lengths, engine revisions, or draft acceptance must not be ranked directly.

**Latest envelope (v2.8):** in the latest 27B-C and 27B-D the RTX 3080 performs all Prefill and Decode compute, and the AI Max+ 395 is a pure KV-only remote storage pool (in neither Prefill nor Decode) providing 1M context capacity per stream (1M context per stream); the 3080 trades remote-KV communication overhead for the per-stream 1M long context.

**Historical envelope:** the full tables for 27B-B (v1.0), 27B-C (v1.1), and 27B-D (v1.2) below are **historical scheduling records** (e.g. the 3080 prefill + 395 decode split across both endpoints) and do not represent the latest “3080 full compute + 395 KV-only” design; the 395 DFlash2 natural-language audit is likewise a **historical control** and does not represent the latest C/D compute path.

## Experiment map

| Experiment | Accelerator and host | Model envelope | Purpose | Headline result |
| --- | --- | --- | --- | --- |
| 27B-A | RTX 3060 12GB + AI Max+ 395 | UD-IQ3_XXS | Long-prompt Dense Region validation | pp4096 658.52 tok/s (+110%) |
| 27B-B | RTX 3080 20GB + AI Max+ 395 | Q4_K_M, serving v1.0 (historical scheduling record) | Remove per-ubatch RPC tax | C1 prefill 1000.6 tok/s; TTFT 1073 ms |
| 27B-C | RTX 3080 full compute + 395 KV-only | Q4_K_M, latest remote-KV envelope | Prefill-first, 1M per stream | 1194.4–1210.6 tok/s prefill; 63.84 tok/s C6 aggregate Decode |
| 27B-D | RTX 3080 full compute + 395 KV-only | Q4_K_M, latest remote-KV envelope | Decode-first, 1M per stream | 63.2 tok/s C1 single-stream Decode; 116.3 tok/s C6 aggregate Decode |

The full v1.1 / v1.2 tables below remain historical scheduling evidence from the development path; they do not describe compute ownership in the latest C/D rows above.

### New experiment 27B-A (3060·IQ3): pp4096 658.52 tok/s (+110%)

The model is Qwen3.8-27B UD-IQ3_XXS (27.32B parameters, 10.17 GiB, 3.06 bpw). The RTX 3060 holds and computes only its assigned layers; internal layer ratios remain undisclosed.

| Metric | 395 only | 3060 + 395 | Measured change |
| --- | ---: | ---: | --- |
| pp4096 | 313.28 tok/s | **658.52 tok/s** | +110% |
| pp65536 | 136.69 tok/s | **319.10 tok/s** | +133% (2.33×) |
| pp98304 | timed out at 900 s | **225.10 tok/s** | completed; TTFT about 437 s |
| Decode tg64 | 18.26 tok/s | **19.57 tok/s** | about +7% |

### New experiment 27B-B (3080·Q4, serving v1.0, historical scheduling record): C1 prefill 1000.6 tok/s; TTFT 1073 ms

- Accelerator head: RTX 3080 20GB, pure CUDA0 full prefill, no RPC and no draft.
- Host: AI Max+ 395 / Radeon 8060S; the v1.0 historical envelope used Vulkan decode with full KV ownership and DFlash2 (kept apart from the latest “3080 full compute + 395 KV-only” design).
- Model: Qwen3.8-27B Q4_K_M (~17.66 GiB); engine fork HEAD `18c8dde`.
- Raw 3080: pp1024 1228.53, pp4096 1203.06, tg64 33.08 tok/s.

This is **six C1–C6 concurrency tiers × split/solo paths = twelve measured groups**, not C1–C12.

| C | TTFT split/solo (ms) | prefill split/solo (tok/s) | decode split/solo (tok/s) | KV migration |
| --- | ---: | ---: | ---: | ---: |
| 1 | **1073** / 4825 | **1000.6** / 207.2 | 38.75 / 36.33 | 71 ms |
| 2 | 1680 / 9649 | 1008.4 / 103.6 | 27.94 / 24.36 | 76 ms |
| 3 | 2680 / 15321 | 1014.5 / 65.3 | 21.45 / 21.17 | 68 ms |
| 4 | 3255 / 19763 | 1010.9 / 50.6 | 18.34 / 14.44 | 72 ms |
| 5 | 4457 / 22460 | 1015.5 / 44.5 | 13.10 / 11.48 | 72 ms |
| 6 | 5009 / 22303 | 1014.3 / 44.8 | 9.67 / 8.82 | 70 ms |

Removing per-ubatch RPC sync lifted same-envelope C1 prefill from 683.2 to **1000.6 tok/s (+46%)**, about 82% of the raw 3080 rate of 1228; TTFT is about 4.5× faster than solo across the tiers.

### New experiment 27B-C (3080·Q4, router v1.1, historical scheduling record): 1194.4–1210.6 tok/s prefill; 35–38.5 tok/s repetitive-text single-stream decode

Envelope: about 1000 prompt tokens / 128 output tokens; router traffic goes through the scheduler and solo traffic connects directly to the 395. Port 8082 serializes prefill, so single-request prefill is also service aggregate prefill. Solo aggregate prefill was not measured and is explicit `NA` in the CSV. This table is a historical scheduling record; in the latest 27B-C (v2.8) the 3080 performs all Prefill and Decode compute while the 395 is a KV-only remote pool, and the homepage compact table follows the latest envelope.

| C | prefill single router/solo | router aggregate prefill | aggregate decode router/solo | single decode router/solo | TTFT router/solo (ms) | end-to-end aggregate router/solo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | **1210.6** / 307.1 | 1210.6 | **33.55** / 25.00 | 38.50 / 36.56 | 829 / 3257 | 16.8 / 12.5 |
| 2 | **1204.8** / 304.9 | 1204.8 | **45.57** / 28.39 | 28.51 / 20.17 | 1283 / 3279 | 22.9 / 14.2 |
| 3 | **1205.5** / 124.1 | 1205.5 | **52.31** / 32.74 | 22.51 / 17.39 | 1731 / 8060 | 26.3 / 16.3 |
| 4 | **1199.5** / 145.0 | 1199.5 | **51.30** / 30.15 | 19.66 / 14.24 | 2195 / 6897 | 25.7 / 15.1 |
| 5 | **1194.4** / 131.4 | 1194.4 | **61.64** / 25.12 | 16.07 / 8.27 | 2668 / 7609 | 30.9 / 12.5 |
| 6 | **1197.2** / 119.7 | 1197.2 | **63.84** / 27.25 | 14.68 / 8.34 | 3122 / 8354 | 32.0 / 15.9 |

**Metric correction: 1200+ is prefill, not decode.** The six prefill points vary by only 1.3% and reach about 97%–98.5% of raw 3080 throughput. The “35–38.5 tok/s single stream” result came from repetitive or degenerate output with 100% draft acceptance. It shows a speculative upper envelope, not natural-language service throughput.

### New experiment 27B-D (3080·Q4, router v1.2, historical scheduling record): C1 single-stream decode 63.2 tok/s; C6 end-to-end aggregate 45.7 tok/s

Live shape: the 3080 runs `np1 / ctx8192 / ub512 / n_max 3 / q4_0 head KV / CUDA graphs off` with a one-slot DFlash2 head. When a new prefill arrives, current 3080 decode saves KV and is restored on the 395. After adding LRU `kv-evict`, repeated 4k×6 and constructed 3×7k residual-KV tests produced no restore failure.

Envelope: 1000 in / 128 out with a random-word seed. Router and solo are comparable within this table only; the seed differs from v1.1. This table is a historical scheduling record; in the latest 27B-D (v2.8) the 3080 performs all Prefill and Decode compute while the 395 is a KV-only remote pool (1M context per stream). The real C1 is Prefill 1090 tok/s and single-stream Decode 63.2 tok/s; aggregate Decode grows linearly with concurrency through C2–C6, peaking at 116.3 tok/s at C6 (exact C2–C5 values were not provided and are not interpolated or fabricated).

| C | TTFT router/solo (ms) | token1→2 (ms) | prefill single/aggregate | single decode router/solo | end-to-end aggregate router/solo | decode-span aggregate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | **921** / 3125 | 96 | 1090 / 1079 | **63.2** / 24.5 | **41.2** / 15.4 | 58.2 |
| 2 | **1420** / 3254 | 396 | 1081 / 1043 | 24.1 / 17.6 | 33.6 / 20.3 | 38.0 |
| 3 | **1917** / 5512 | 575 | 1080 / 1030 | 22.5 / 12.3 | 41.5 / 16.2 | 45.8 |
| 4 | **2424** / 8602 | 739 | 1077 / 1017 | 17.8 / 12.0 | 41.3 / 21.3 | 44.3 |
| 5 | **2911** / 7603 | 836 | 1082 / 1018 | 13.9 / 8.1 | 42.8 / 19.6 | 45.4 |
| 6 | **3404** / 8319 | 1060 | 1082 / 1016 | 13.2 / 7.4 | 45.7 / 23.3 | 48.0 |

All twelve router/solo groups had `miss=0`, `err=0`, and passed three repeated runs. Long-prompt points: 4k / 7k single-request TTFT was 3689 / 6524 ms with 3080 decode at 71 / 69 tok/s; 4k×6 and 7k×2 did not OOM.

Separate point tests of the 3080 DFlash2 head measured **42.7 tok/s natural language (43% acceptance)** and **67.4 tok/s code (87%)**. These are not the random-seed C1 row above and must not be substituted for it.

## 395 DFlash2 natural-language audit (historical control)

This section is a **historical control** and does not represent the latest C/D compute path (in the latest envelope the 395 is a KV-only remote pool, in neither Prefill nor Decode). To test whether “single-stream 35” represents service text, natural-language prose was sent directly to 8081 (395 + DFlash2, `n_max 7`) with chat template, thinking disabled, 128 output tokens, and temperature 0:

| C | Single stream | Aggregate | Per step | Acceptance | Tokens per step |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | **12.1 tok/s** | 10.9 | 178 ms | 17.7% | 2.17 |
| 2 | 9.9 | 16.4 | 262 ms | 24.3% | 2.61 |
| 3 | 7.6 | 19.3 | 318 ms | 21.1% | 2.42 |
| 4 | 5.8 | 19.5 | 378 ms | 17.9% | 2.19 |
| 6 | **4.0** | **19.8** | 548 ms | 18.3% | 2.22 |

C5 was not measured and is not filled in. The headless 395 baseline is 18.26 tok/s at 55 ms/step, so natural-language C1 with the head is 34% slower. On the same 8081, repetitive text at 100% acceptance reaches about 40 tok/s, code at 50%–67% acceptance reaches about 24–29 tok/s, and real prose at 17%–25% acceptance is about 12 tok/s. Thus **35–38.5 is an upper-envelope repetitive-text result, not stable production single-stream throughput**.

## Boundaries and next steps

- v1.1 and v1.2 use different random seeds and aggregate denominators; absolute decode figures must not be compared across them.
- In the historical v1.2 schedule, the 3080 has one slot and new prefill preempts the fast decode lane at C≥2; that limitation does not describe the latest v2.8 KV-only route.
- The latest route exposes 1M context capacity per stream. Published speed points must not be extrapolated to claim identical throughput with a fully populated 1M-token prompt.
- DGX Spark NVFP4 / SGLang / 100K–300K cold-prefill data uses a different envelope and remains background only.

Machine-readable data: [qwen27b-local-results.csv](../data/qwen27b-local-results.csv).
