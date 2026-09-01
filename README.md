# <img src="assets/soulmate-spirit.png" alt="Soulmate spirit" width="44" align="absmiddle"> AI Max+ 395 Acceleration — Heterogeneous GPU PD Lab

[简体中文](README_ZH.md)

This release records the public evolution of a heterogeneous fused-layer pipeline using an RTX 3060 12GB and an AMD Ryzen AI Max+ 395 / Radeon 8060S on one host.

Current release: **v2.1 — First Fused Pipeline**. Public revision labels intentionally omit the internal layer-allocation policy. Deployment steps, commands, patches, endpoints, and credentials are not published.

## Architecture

Prompt micro-batches pass through RTX 3060 / CUDA front-stage layers and AI Max+ 395 / Vulkan rear-stage layers. Async RPC events allow adjacent micro-batches to overlap. The experiment uses upstream llama.cpp RPC events/async capability and claims no custom engine patch.

## Public checkpoints

Test envelope: 9B Q6_K, pp5064; tg128 only where recorded.

| Checkpoint | Public description | Prefill | Decode |
| --- | --- | ---: | ---: |
| RTX 3060 baseline | CUDA endpoint only | 1589.00 tok/s | 43.87 tok/s |
| AI Max+ 395 baseline | Vulkan endpoint only | 970.00 tok/s | 31.27 tok/s |
| v2.1 | First Fused Pipeline | 1865.08 tok/s | Not recorded |

The checkpoint data is experimental and limited to one host and a short single-concurrency benchmark. Internal layer ratios are intentionally undisclosed.

See [the bilingual evolution record](results/v2-evolution.md), [structured data](data/benchmark-results.csv), and [changelog](CHANGELOG.md).