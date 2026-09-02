# Qwen3.8-27B Dual-Machine Heterogeneous PD — separate envelope

[简体中文](qwen3.8-27b-dual-machine-pd.zh-CN.md)

This line migrates the experiment from the RTX 3060 to an **RTX 3080 20GB** accelerator head and records the full Qwen3.8-27B Q4 dual-machine heterogeneous PD results. It is deliberately separated from the `9B Q6_K / pp5064` main table and the `27B IQ3` validation line, and is not included in the v2.4 percentages.

## Acceleration revision and head

- **Accelerator head**: **RTX 3080 20GB** (pure CUDA0 full prefill, no RPC, no draft speculation).
- **Host being accelerated**: **AMD Ryzen AI Max+ 395 / Radeon 8060S** (Vulkan decode, holds the full KV pool, DFlash2 speculative decoding).
- **Model**: Qwen3.8-27B, Q4_K_M (~17.66 GiB).
- **Engine**: llama.cpp fork (HEAD `18c8dde`, with upstream [PR #18626](https://github.com/ggml-org/llama.cpp/pull/18626) async/RPC capability), running as two instances: 8082 does pure CUDA0 full prefill, 8081 does pure 395 decode and holds the full KV pool (`/dev/shm/kvx`).
- **Raw 3080 baseline**: pp1024 = 1228.53, pp4096 = 1203.06, tg64 = 33.08 tok/s.

## Twelve-tier serving stress (split vs 395 solo)

`bench_3080_cuda_full.log`, ub1024, zero errors across all tiers.

| C | TTFT split/solo (ms) | prefill split/solo (tok/s) | decode split/solo (tok/s) | KV migration cost (save+rest) |
| --- | ---: | ---: | ---: | ---: |
| 1 | **1073** / 4825 | **1000.6** / 207.2 | 38.75 / 36.33 | 71 ms |
| 2 | 1680 / 9649 | 1008.4 / 103.6 | 27.94 / 24.36 | 76 ms |
| 3 | 2680 / 15321 | 1014.5 / 65.3 | 21.45 / 21.17 | 68 ms |
| 4 | 3255 / 19763 | 1010.9 / 50.6 | 18.34 / 14.44 | 72 ms |
| 5 | 4457 / 22460 | 1015.5 / 44.5 | 13.10 / 11.48 | 72 ms |
| 6 | 5009 / 22303 | 1014.3 / 44.8 | 9.67 / 8.82 | 70 ms |

## Key findings

- TTFT leads solo by about **4.5×** across all tiers.
- Prefill does not degrade with concurrency (steady **1000–1015 tok/s**).
- Decode is lossless (C1 38.75 vs solo 36.33).
- KV migration cost 71 ms.

The former 45% tax from per-ubatch RPC synchronization has been eliminated: after removing RPC the same measurement rose from 683.2 to **1000.6 tok/s (+46%)**, reaching about **82%** of the 3080 raw 1228.

## Differences versus DGX Spark

| Metric | AI Max+ 395 + RTX 3080 (this experiment) | DGX Spark / GB10 |
| --- | --- | --- |
| Model/quantization | Qwen3.8-27B, Q4_K_M | Qwen3.8-27B, NVFP4 |
| Engine | llama.cpp (CUDA + Vulkan, fork `18c8dde`) | SGLang + DFlash2 (1M context) |
| KV precision | q4_0 | fp8_e4m3 |
| Prefill (cold) | ~1000–1015 tok/s (short prompt serving) | 1170 / 800 / 615 tok/s (100K / 200K / 300K-token cold) |
| Prefill (raw) | pp1024 = 1228.53 tok/s | Not disclosed |
| Decode (C1) | 38.75 tok/s | Not disclosed |
| TTFT (C1) | 1073 ms | Not disclosed |
| Concurrency | C1–C6 all tiers measured | Not disclosed |
| Topology | dual-machine heterogeneous PD (3080 prefill + 395 decode) | single machine |

**Not directly comparable**: quantization (Q4_K_M vs NVFP4), engine (llama.cpp vs SGLang), KV precision (q4_0 vs fp8), prompt depth (short prompt vs 100K–300K tokens), and topology (dual-machine heterogeneous vs single machine) all differ. Provided only as a same-model-scale reference.

Machine-readable data: [qwen27b-local-results.csv](../data/qwen27b-local-results.csv).
