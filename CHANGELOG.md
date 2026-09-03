# Changelog

[简体中文](CHANGELOG_ZH.md)

## v2.10

- Consolidated 27B-C, 27B-D, and DGX Spark into one comparison table, with local results consistently shown in **C / D** order.
- Filled the DGX Spark summary with **about 1000 tok/s Prefill, 25–30 tok/s single-stream Decode, 107 tok/s aggregate Decode, and C1–C6 concurrency**.
- Added a dedicated **DFlash2 acceleration-head** category to both the homepage and detailed record, mirrored in English and Chinese.

## v2.9

- Corrected the latest 27B-D C6 aggregate Decode peak to **116.3 tok/s** across all eight target files (homepage, detailed record, changelog, and CSV).
- Removed the sentence about the 3080 trading standalone Prefill/Decode speed for remote-KV communication overhead from the 27B-C homepage section; that statement belongs to 27B-D, where it is kept.
- Fixed the wrong C6 figure in the v2.8 entry below.

## v2.8

- Corrected compute attribution: in the latest 27B-C and 27B-D the RTX 3080 performs all Prefill and Decode compute, while the AI Max+ 395 becomes a pure KV-only remote storage pool (in neither Prefill nor Decode) providing 1M context capacity per stream (1M context per stream).
- Added the real C1–C6 compact table to the homepage (from C1 1210.6 / 38.50 / 33.55 to C6 1197.2 / 14.68 / 63.84, tok/s), and stated the real 27B-D C1 Prefill 1090 tok/s, C1 single-stream Decode 63.2 tok/s, and C6 aggregate Decode up to 116.3 tok/s.
- Clarified the value narrative: the 3080 trades remote-KV communication overhead for the per-stream 1M long context; the old v1.0–v1.2 tables and the 395 natural-language audit are retained as historical scheduling records/controls and do not represent the latest C/D compute path.

## v2.7

- Reframed homepage experiments 27B-C and 27B-D as prefill-first and decode-first profiles, with each headline leading on its best measured result.
- Moved implementation parameters, full concurrency detail, and envelope data out of the homepage narrative; the detailed records and CSV remain the source of record.

## v2.6

- Reorganized both READMEs into architecture and envelopes → research evolution → 9B experiment chain → unified 27B experiment area → external references → findings, and gave every local experiment a result-bearing “New experiment” heading.
- Merged the RTX 3060 / IQ3 and RTX 3080 / Q4 27B records into one experiment area while preserving their distinct quantization, hardware, and measurement envelopes; cross-envelope ranking remains invalid.
- Added router v1.1: serialized prefill holds **1194.4–1210.6 tok/s** for both single-request and service aggregate throughput. The **35–38.5 tok/s** single-stream decode figure is now explicitly limited to repetitive text with 100% draft acceptance; 1200+ is never labeled as decode.
- Added the live v1.2 two-stage-preemption + RTX 3080 DFlash2-head C1–C6 table, long-prompt checks, and the 395 natural-language audit. Direct natural-language C1 measured **12.1 tok/s**, so repetitive-text figures no longer stand in for production prose.
- Replaced the ambiguous “twelve tiers” wording with **six concurrency tiers × router/solo = twelve measured groups**, expanded the machine-readable CSV, and made unavailable baseline fields explicit `NA`.

## v2.5

- Migrated the experiment to an **RTX 3080 20GB** accelerator head and published the full Qwen3.8-27B Q4 dual-machine heterogeneous PD record.
- Published six concurrency tiers across split/solo paths (twelve measured groups): TTFT 4.5× over solo, prefill holding **1000–1015 tok/s** regardless of concurrency, lossless decode (C1 38.75 tok/s), KV migration cost 71 ms.
- Confirmed removing RPC per-ubatch sync lifted the same measurement from 683.2 to **1000.6 tok/s (+46%)**, about 82% of the 3080 raw 1228.
- Added english/chinese detail records and machine-readable local CSVs for this line, plus a differences-vs-DGX-Spark comparison table with explicit non-comparability metadata.

## v2.4

- Named the final asynchronous layered-PD form **Dense Acceleration** and defined its concurrent compute window as the **Dense Region**.
- Published the final Dense Acceleration checkpoint at 2129.69 tok/s, while keeping the v2.3-only 37.16 tok/s decode result explicitly separated.
- Clarified that the Dense Region is micro-batch scheduling overlap, not duplicate same-layer compute, tensor parallelism, or duplicated weights.
- Added the separate-envelope 27B IQ3 validation: 658.52 tok/s at pp4096, 319.10 tok/s at pp65536, 225.10 tok/s at pp98304, and 19.57 tok/s decode.
- Made English the default repository language and added a complete Chinese mirror.
- Added the Soulmate spirit mark and AI Max+ 395 acceleration to the project title.
- Added two source-linked DGX Spark community controls with explicit non-comparability metadata.
- Removed public layer-allocation ratios and replaced them with revision labels.

## v2.3

- Published the balance-refinement checkpoint at 1999.51 tok/s prefill and 37.16 tok/s decode.

## v2.2

- Published the overlap-refinement checkpoint at 1893.87 tok/s prefill.

## v2.1

- Published the first fused-pipeline checkpoint at 1865.08 tok/s prefill.

## v1.0

- Published independent heterogeneous PD with four measured handoff nodes.
- Documented the single-host 9B validation boundary.
