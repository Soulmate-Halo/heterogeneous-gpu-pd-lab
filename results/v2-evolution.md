# Fused-Layer Pipeline Evolution through v2.1

[简体中文](v2-evolution.zh-CN.md)

The public design moved from independent prefill/decode handoff in v1.0 to an asynchronous fused-layer micro-batch pipeline. Revision labels describe measured checkpoints without disclosing internal layer allocation.

| Checkpoint | Public description | Prefill | Decode |
| --- | --- | ---: | ---: |
| RTX 3060 baseline | CUDA endpoint only | 1589.00 tok/s | 43.87 tok/s |
| AI Max+ 395 baseline | Vulkan endpoint only | 970.00 tok/s | 31.27 tok/s |
| v2.1 | First Fused Pipeline | 1865.08 tok/s | Not recorded |

Measurement boundary: one host, 9B Q6_K, pp5064, and tg128 only where recorded. These results do not establish production readiness or performance on untested models, long context, concurrency, or multiple hosts.