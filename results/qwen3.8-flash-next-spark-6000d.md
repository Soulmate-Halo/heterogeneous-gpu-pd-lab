# FLASH-SPARK-01 · Qwen3.8-Flash-Next NVFP4 · RTX 6000D + DGX Spark

[中文](qwen3.8-flash-next-spark-6000d.zh-CN.md) · [Home](../README.md) · [CSV](../data/qwen3.8-flash-next-spark-6000d.csv) · [JSON](../data/qwen3.8-flash-next-spark-6000d-evidence.json)

**Complete two-machine inference works: 8K Prefill 8157.74 tok/s and C6 aggregate output 414.90 tok/s.** These are separate Prefill and generation workloads on the same capacity profile, not paired metrics from an 8K-input/256-output request. Measured **2026-09-14**; public version remains **v1.6**.

## Latest finding: allocation follows measurable constraints

**Compute allocation and memory allocation are related: memory limits how much work can be placed, while compute and transfer time determine which placement is fast.** Models differ in weights, KV demand, lookup work and computation; accelerators differ in compute speed and memory bandwidth. The fastest measured configuration therefore has an explainable structure instead of a fixed layer ratio that can be copied everywhere.

**The reusable result is a method that can be calculated and reasoned about.** Budget weights, KV, activations and runtime memory on each device; use measured stage times and communication costs to shortlist feasible allocations; validate Prefill, Decode, concurrency throughput and output. This is a relationship between capacity and performance constraints, not an established universal linear formula or proof of a global optimum.

On this model and hardware pair, **specially proportioned PP2 layer splitting** raised 8K Prefill from 8157.74 to 8696.94 (**+6.61%**) while C6 aggregate output fell from 414.90 to 284.56 (**-31.42%**). Both stages must influence the choice.

## Latest experimental parameters

| Parameter | Measured setting |
| --- | --- |
| Model | `nvidia/Qwen3.8-Flash-Next-NVFP4` |
| Devices | RTX 6000D + DGX Spark GB10 |
| Device memory | 6000D reports 85651 MiB visible device memory; Spark 128 GB unified memory |
| Weights / KV | NVFP4 MoE; BF16 attention/shared experts; FP8 PLE/MTP; KV `fp8_e4m3` |
| Engine | vLLM nightly · `8a728663c1c3eeace834a95f5654fa653cc1998c` |
| Context / concurrency / Prefill chunk | 65536 / 6 / 2048 tokens |
| Speculative decoding | MTP 3; full draft vocabulary |
| Capacity profile | GMU 0.99; KV pool 198332 tokens; Spark PLE table 47.7 GiB |
| Specially proportioned layer split | PP2; 8 GiB KV on each device; exact layer counts and ratios private |
| C1–C6 serving test | Chinese long-form generation; 256 output tokens/request; 3 measured rounds + 1 warmup; temperature=0; thinking=false; ignore_eos=true |

## Device responsibilities

- **Capacity profile: the 6000D performs main-model computation and generation; Spark holds and gathers the PLE table.** Model loading on the 6000D takes about 75.64 GiB; Spark supplies about 47.7 GiB of table capacity. This complete two-machine profile produces **8157.74 + 414.90**. Some source notes call it a single-machine baseline because main computation is concentrated on the 6000D; it still depends on Spark and is not a standalone control.
- **Specially proportioned layer split: both GPUs perform model computation through PP2, with allocation tuned experimentally.** This profile reaches **8696.94** at 8K Prefill and **284.56** at C6 aggregate output. Exact layer counts, ratios, deployment commands and patches remain private.

## Standalone and paired results

Units: tok/s. Neither standalone column has a local control with matching model, quantization and workload.

| Metric / workload | 6000D standalone | Spark standalone | Pair: capacity profile | Pair: special PP2 split |
| --- | --- | --- | --- | --- |
| 4K Prefill · C1 | not measured | not measured | **8016.63** | 7538.10 |
| 8K Prefill · C1 | not measured | not measured | **8157.74** | **8696.94** |
| C6 aggregate output (includes TTFT) | not measured | not measured | **414.90** | 284.56 |

Do not combine PP2's 8696.94 with the capacity profile's 414.90 as one configuration. The 8K comparison uses similar target input lengths but different repetition counts (capacity: 2, PP2: 3); the percentage is only the difference between these measurements.

## Capacity-profile Prefill

Prefill is actual prompt tokens / streaming TTFT, a serving measurement rather than engine-only compute throughput. Two runs per target length; every output is `OK` (2 tokens). Fresh-seed prompts avoid reusing old PLE rows. The first 4K shape compilation is disclosed separately.

| Target input | Actual input tokens | Prefill tok/s | TTFT s | Selection |
| --- | --- | --- | --- | --- |
| 2K | 2023 | **6754.54** | 0.305395 | median of 2 |
| 4K | 4021 | **8016.63** | 0.501582 | second steady-state run |
| 8K | 8017 / 8018 | **8157.74** | 0.982826 | median of 2 |
| 16K | 16009 | **8309.15** | 1.926756 | median of 2 |
| 32K | 31994 | **8218.55** | 3.892933 | median of 2 |

The first 4K run was **2468.07 tok/s / 1.629206 s**; the median including both runs is **5242.35 tok/s**. The **8016.63** headline deliberately selects the second steady-state run and is not that median. The 8K runs actually contain 8018 and 8017 input tokens; 8157.74 is the median of the two rates.

An earlier NFS row-by-row version of the same pair measured about 548 tok/s at 8K, versus 8157.74 after optimization. This is a change in data access and runtime settings, not a standalone-to-pair gain. Cache and JIT conditions matter. Separate C2 Prefill results are retained in JSON and are not paired with the C6 short-prompt generation test.

## Complete C1–C6 generation test

**Aggregate output = all successful completion tokens / sum of measured round wall times, including Prefill, TTFT and generation.** C1–C6 denotes simultaneous requests. Each level has 3 measured rounds plus 1 excluded warmup. Each request emits 256 tokens. Chinese prompts request long-form writing; their input is not the 8K Prefill workload.

| Concurrency | Capacity: aggregate output | Capacity: mean stream Decode | PP2: aggregate output | Measured requests per profile |
| --- | --- | --- | --- | --- |
| C1 | **94.82** | 108.13 | 81.79 | 3/3 |
| C2 | **172.68** | 97.07 | 138.66 | 6/6 |
| C3 | **232.38** | 87.77 | 166.21 | 9/9 |
| C4 | **296.14** | 82.91 | 205.31 | 12/12 |
| C5 | **315.92** | 79.15 | 251.35 | 15/15 |
| C6 | **414.90** | 76.89 | 284.56 | 18/18 |

Capacity C6: **4608 tokens / 11.106197701 s = 414.903473 tok/s**; TTFT P50 **0.265835 s**, P95 **0.267717 s**. Each profile has **18 measured rounds and 63/63 successful requests**. The 18/18 figure refers to C6 alone. All requests reach the requested length limit, as intended by the fixed-output test.

Mean per-request Decode estimates `(completion tokens - 1) / (last content delta time - first content delta time)`. Speculative deltas may contain multiple tokens; multiplying this estimate by concurrency does not produce aggregate output. CSV and JSON preserve full precision, wall time and TTFT P50/P95.

## Output checks and evidence

All Prefill replies are `OK`. Both profiles complete 63/63 generation requests with a closed stream, complete usage and nonempty text. PP2 summaries include readable samples at every concurrency level; the experiment record also reports Chinese Q&A and arithmetic smoke checks. These are completion and sample checks, not a semantic evaluation of every generated essay.

Only PP2 results after communication and draft synchronization fixes are used. The public JSON contains sanitized metric extracts, original file SHA256 hashes and selection rules, preserving numeric precision. Endpoints, credentials and exact layer allocation are omitted. This publication checks existing artifacts from the user-referenced session; no new benchmark was run.
