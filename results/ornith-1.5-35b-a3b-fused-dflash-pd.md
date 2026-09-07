# ORNITH-PD-02 — Ornith-1.5-35B-A3B PD with Fused DFlash Speculative Decode

[简体中文](ornith-1.5-35b-a3b-fused-dflash-pd.zh-CN.md)

## Experimental contract

| Item | Definition |
| --- | --- |
| Primary question | On top of the independent PD of ORNITH-PD-01, do fused DFlash speculative decode and a unified KV pool lift Prefill and single-stream Decode at the same time? |
| Topology under test | Three-process PD: 8080 router (CPU), 8082 Prefill (RTX 3080 / CUDA0, 1 slot), 8081 Decode + DFlash (AI Max+ 395 / Vulkan1, 6 slots sharing one unified KV pool). |
| Matched speed control | A negative control from the same deployment, taken with a different prompt where draft acceptance fell to 9.4%; the short-task Prefill of ORNITH-PD-01 is the same-workload reference. |
| Changed factors | 8082 batch/ubatch (4096 → 8196) and 8081 ctx (32768 → 131072). |
| Decision metrics | Single-stream Prefill, single-stream Decode, DFlash draft acceptance, three-endpoint health, and whether OOM occurred. |
| Pass boundary | On a 1000 in / 128 out workload, Prefill above 4000 tok/s and Decode above 100 tok/s, with 8080/8081/8082 all HTTP 200 and no OOM or crash after start-up. |
| Does not prove | End-to-end acceleration against a standalone run; concurrent aggregate throughput (this experiment measured a single stream only and ran no C1–C6 matrix). |

This record covers the second PD experiment on Ornith-1.5-35B-A3B. It uses the same model, the same pair of devices, and the same role split as [ORNITH-PD-01](ornith-1.5-35b-a3b-dual-machine-pd.md) — the 3080 does all Prefill and the 395 does all Decode. What changed is the 395 side: Decode now runs fused DFlash speculative decode (`--spec-type draft-dflash`, draft length 6), and the six slots share one unified KV pool (`--kv-unified`). **The two experiments use different workloads**: the first is a C1–C6 concurrency matrix (six tiers each for the short and the 100K stage), while the second is a single stream of 1000 in / 128 out. Each must therefore be read within its own caliber and the two must not be merged into one ranking.

**Ornith-1.5-35B-A3B is a MoE (35B total, A3B active-per-token naming). It must not be ranked directly against the Qwen3.8-27B dense rows in [qwen3.8-27b-dual-machine-pd.md](qwen3.8-27b-dual-machine-pd.md):** different model family, different architecture, different per-token active compute, and different quantization.

## Model and topology

| Item | Value |
| --- | --- |
| Model | Ornith-1.5-35B-A3B (MoE, 35B total, A3B active-per-token) |
| GGUF | Ornith-1.5-35B-A3B-IQ4_XS |
| Architecture | qwen35moe, 40 layers: 10 full-attention layers + 30 Gated DeltaNet layers |
| Draft head | Qwen3.6-35B-A3B-DFlash-Q4_K_M, `--spec-type draft-dflash`, draft length n_max 6 |
| Prefill node | RTX 3080 20GB, CUDA0, port 8082: all Prefill, 1 slot, ctx 8192 |
| Decode node | AMD Ryzen AI Max+ 395, Vulkan1, port 8081: all Decode, 6 slots, unified KV pool |
| Router node | 8080 on the CPU: Prefill to 8082, Decode to 8081 |
| KV channel | /dev/shm/kvxo |
| KV-cache quantization | q4_0 on both ends (k and v) |
| Workload | 1000 input tokens / 128 output tokens, single stream |

## Main result — 1000 input / 128 output, single stream

All values in tok/s.

| Configuration | Prefill | Decode | DFlash draft acceptance |
| --- | ---: | ---: | ---: |
| Final recipe (8082 batch/ubatch 4096, 8081 ctx 32768) | **4173.47** | **114.86** | 107/114 (**93.86%**) |
| Larger batch and context (8082 batch/ubatch 8196, 8081 ctx 131072) | **4123.15** | **114.42** | not recorded |
| Same recipe, different prompt (negative control) | 3665.3 | 37.2 | **9.4%** |

The ctx 131072 in the second row is the maximum context for a single request under the unified KV pool, not 128K reserved per slot. A separate performance gate in the same round reached only `3785/44`, again because that test prompt had low draft acceptance, not because the parameter change was wrong.

**The Decode figure tracks draft acceptance.** With the same three-process recipe and only the prompt changed, acceptance fell from 93.9% to 9.4% and Decode fell from 114.86 to 37.2 tok/s; back at 93.86% acceptance, Decode is 3.09× the 37.2 figure. Re-measuring this tier therefore requires the same prompt, a fixed output length, `ignore_eos`, and the same seed; otherwise the result reflects prompt difficulty rather than the recipe.

## Relationship to ORNITH-PD-01

| Item | ORNITH-PD-01 (independent PD) | ORNITH-PD-02 (fused DFlash draft) |
| --- | --- | --- |
| Prefill on the 1000-input stage | C1 4017.46 → C6 3943.88 (six-tier aggregate) | 4173.47 (single stream) |
| Single-stream Decode | not recorded | 114.86 |
| 100K stage | Prefill C1 2895.53 → C6 2793.24; 395 pure Decode C1 23.33 → C6 148.20 | not measured |
| Draft acceptance | not recorded | 93.86% |
| Workload caliber | C1–C6 concurrency matrix, 42/42 passed | single stream, 1000 in / 128 out |

The Prefill comparison is on the same workload: 4173.47 ÷ 4017.46 − 1 = **+3.88%**, so the fused recipe runs about 3.9% ahead of the C1 figure on the first experiment's short stage.

Decode cannot be compared that way: the short stage of the first experiment has no separately timed 395 pure-Decode rate, and the 23.33–148.20 published for its 100K stage is a six-tier **aggregate**, while the 114.86 here is a **single stream**. They are not the same quantity.

## Three-process parameters, verbatim

8082 Prefill node (RTX 3080 / CUDA0):

```text
--model /home/hfy/ornith-dflash/models/Ornith-1.5-35B-A3B-IQ4_XS.gguf
--device CUDA0
--gpu-layers 99
--ctx-size 8192
--parallel 1
--batch-size 4096
--ubatch-size 4096
--cache-type-k q4_0
--cache-type-v q4_0
--flash-attn auto
--slot-save-path /dev/shm/kvxo
--cache-ram 0
--ctx-checkpoints 0
--cont-batching
--jinja
--metrics
--host 127.0.0.1
--port 8082
```

Extra environment variable `GGML_CUDA_DISABLE_GRAPHS=1`; CPU threads were not set explicitly and resolved to 16. The larger-batch run set both `--batch-size` and `--ubatch-size` to `8196`.

8081 Decode + DFlash node (AI Max+ 395 / Vulkan1):

```text
--model /home/hfy/ornith-dflash/models/Ornith-1.5-35B-A3B-IQ4_XS.gguf
--model-draft /home/hfy/ornith-dflash/models/Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf
--device Vulkan1
--device-draft Vulkan1
--gpu-layers 99
--gpu-layers-draft 99
--spec-type draft-dflash
--spec-draft-n-max 6
--ctx-size 32768
--parallel 6
--kv-unified
--batch-size 2048
--ubatch-size 512
--cache-type-k q4_0
--cache-type-v q4_0
--flash-attn auto
--slot-save-path /dev/shm/kvxo
--cache-ram 0
--ctx-checkpoints 0
--cont-batching
--jinja
--metrics
--host 0.0.0.0
--port 8081
```

Remaining DFlash values in effect: `n_min=0`, `p_min=0.00`, `block_size=16`, `n_extract=8`, `sample_from_anchor=true`. The larger-context run set `--ctx-size` to `131072`.

8080 router node (CPU):

```text
--dec 8081
--pre 8082
--pre-slots 1
--dec-slots 6
--pre-ctx 8192
--pre-decode-slots 0
--pre-busy-policy preempt
--kvx /dev/shm/kvxo
--pre-tps 3387
--dec-pp-tps 1050
```

## Site state and health

- 8080 / 8081 / 8082 all HTTP 200, with no OOM and no crash in the logs after start-up
- RTX 3080 VRAM in use `18661/20480 MiB` (recorded as `18688/20480 MiB` during the parameter re-check on the same recipe)
- AI Max+ 395 Vulkan usage about `22678/65536 MiB`
- The three process IDs were `2775429 / 2775363 / 2774975`
- Port 8090 was untouched throughout; the remote backup for the parameter-upgrade round is at `/home/hfy/r382/`

## Not recorded (no extrapolation)

The source record does not contain the following; this document and the CSV leave them empty or mark them not recorded, and nothing is interpolated:

- C1–C6 concurrency matrix — this experiment measured a single stream only
- 100K long-context figures — not measured here (the first experiment has them)
- TTFT — not recorded
- KV migration milliseconds — not recorded
- Draft acceptance for the larger-batch run — not recorded

## Boundaries

- With no same-round standalone control, this record states how far the recipe serves and how its two parameter settings differ; it is not a speedup claim against the 395 alone or the 3080 alone.
- The 114 Decode tier depends on DFlash draft acceptance and falls back to the 37 tier on a different prompt; the acceptance rate must be quoted alongside the figure.
- The Prefill comparison with ORNITH-PD-01 holds only on the 1000-input stage, and even there the first experiment's column is a six-tier aggregate while this one is a single stream: the direction is comparable, the exact margin should not be over-read.
- **MoE vs dense**: Ornith-1.5-35B-A3B activates far less per-token compute than the Qwen3.8-27B dense model, so these figures cannot be collapsed into a ranking with the 27B rows.

Data source: `r379_summary.md` (final recipe as measured) and `r382_ctx_prefill.md` (re-measured after the batch and context increase); numbers are taken verbatim with nothing extrapolated.

Machine-readable data: [ornith35a3b-local-results.csv](../data/ornith35a3b-local-results.csv).
