# FLASH-SPLIT-01 — Qwen3.8-Flash Q4 Dual-Device Layer-Split Concurrency Envelope

[简体中文](qwen3.8-flash-q4-layer-split.zh-CN.md)

## Experimental contract

| Item | Definition |
| --- | --- |
| Primary question | Can one CUDA+Vulkan layer-split server sustain C1–C6, and at which concurrency does its throughput saturate? |
| Configuration under test | One fixed RTX 3080 + AI Max+ 395 layer split, with fixed model, batching, context, and KV precision. |
| Matched standalone control | Not recorded. No 3080-only or 395-only run exists in this workload envelope. |
| Changed factor | Concurrency only: C1 through C6. |
| Decision metrics | HTTP success, aggregate Prefill, single/aggregate Decode, total throughput, and 3080 VRAM headroom. |
| Pass boundary | Complete timings for every scored request and a stable operating envelope without OOM. |
| Does not prove | Speedup over either device, causal Dense Acceleration, or long-context performance. |

This record covers the r374 Qwen3.8-Flash Q4 dual-device layer-split experiment added in v2.14. A single llama-server runs both devices at once: the RTX 3080 (CUDA0) and the AI Max+ 395 (Vulkan1) each own their layer stage, and micro-batches overlap across the stages.

## Model and topology

| Item | Value |
| --- | --- |
| Model | Qwen3.8-Flash Q4 |
| KV cache | q4_0 (both K and V) |
| Engine | llama.cpp fork with upstream [PR #18626](https://github.com/ggml-org/llama.cpp/pull/18626) async/RPC; one llama-server, dual device |
| Devices | RTX 3080 20GB (CUDA0) + AMD Ryzen AI Max+ 395 / Radeon 8060S (Vulkan1) |
| Layer split | tensor split 0.38 / 0.62 |
| Batching | ubatch 1024, batch 4096, flash attention on |
| Context | 131072 per slot, 6 slots |
| Offload | all layers on GPU |
| 3080 VRAM | 19033 MiB at start, 19129 MiB peak (165 one-second samples) |

## Workload

- Six unique prompts calibrated to ~2077 input tokens (2074–2082), 256 output tokens each
- temperature 0, prompt cache off, non-streaming
- Six concurrency tiers C1–C6, each tier reusing the same six prompts
- One warm-up C1 before scoring; excluded from the results
- 21/21 scored requests returned HTTP 200 with complete timings; no anomalies

## C1–C6 results (tok/s)

| C | Prefill aggregate | Single-stream Decode | Aggregate Decode | Total throughput | Tier wall clock s |
| --- | ---: | ---: | ---: | ---: | ---: |
| C1 | 569.892 | 35.204 | 35.204 | 213.581 | 10.923 |
| C2 | 579.215 | 26.055 | 51.877 | 273.089 | 17.079 |
| C3 | 625.250 | 21.714 | 65.143 | 320.267 | 21.863 |
| C4 | **633.685** | 17.829 | **71.185** | **338.270** | 27.596 |
| C5 | 552.070 | 14.578 | 69.595 | 327.366 | 35.639 |
| C6 | 554.873 | 12.317 | 67.286 | 332.790 | 42.060 |

### Per-lane detail (mean / min / max, tok/s)

| C | Per-lane prefill | Per-lane decode |
| --- | ---: | ---: |
| C1 | 569.892 / 569.892 / 569.892 | 35.204 / 35.204 / 35.204 |
| C2 | 330.866 / 289.747 / 371.985 | 26.055 / 25.939 / 26.171 |
| C3 | 208.459 / 208.116 / 208.822 | 21.714 / 21.714 / 21.714 |
| C4 | 159.578 / 158.364 / 160.241 | 17.829 / 17.796 / 17.927 |
| C5 | 130.675 / 110.286 / 174.061 | 14.578 / 13.919 / 15.620 |
| C6 | 134.886 / 92.709 / 252.126 | 12.317 / 11.214 / 13.457 |

Metric definitions:

- **Prefill aggregate** = sum of the tier's prompt tokens / max of the tier's per-lane prompt milliseconds × 1000
- **Single-stream Decode** = mean of the per-lane decode rates (per-lane decode rate = predicted tokens / predicted milliseconds × 1000)
- **Aggregate Decode** = sum of the tier's predicted tokens / max of the tier's per-lane predicted milliseconds × 1000
- **Total throughput** = (sum of prompt and predicted tokens) / tier wall-clock seconds

Prefill peaks at C4 (633.685 tok/s) and stays above 550 tok/s at every tier; aggregate Decode grows from 35.204 at C1 to 71.185 at C4, then eases slightly under the six-lane load.

## Not recorded (no extrapolation)

- Per-tier TTFT — not recorded
- KV migration — not applicable: one service, no cross-device KV handoff
- Whole-run wall clock: 167.199 s for the six-tier run (remote script timing); local driver wall clock 169.585 s

## Boundaries

- This is a concurrency-envelope experiment. Without a matched standalone run, its absolute rates cannot establish an acceleration factor.
- The ~2077-input / 256-output envelope is a short-context workload; it must not be ranked against the 27B or Ornith tables, which use different models, hardware roles, and calibers.
- The 3080 VRAM peak of 19129 MiB fits inside the 20 GB card with headroom for the working set.
- Numbers are taken verbatim from the completed r374 run; nothing is interpolated.

Data source: the completed r374 short-context C1–C6 stress run (numbers taken verbatim; nothing extrapolated).

Machine-readable data: [qwen38flash-q4-local-results.csv](../data/qwen38flash-q4-local-results.csv).
