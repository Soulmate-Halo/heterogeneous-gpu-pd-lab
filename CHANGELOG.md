# Changelog

[简体中文](CHANGELOG_ZH.md)

## v2.15

- Rebuilt the homepage as an experiment registry: every stable experiment now names one primary question, its control, changed factor, decision gate, allowed claim, and canonical data entry.
- Removed repeated benchmark tables from both README files. Human-readable numbers now live in one detailed record per experiment, with CSV files kept as machine-readable mirrors.
- Separated experimental work from publication history: v2.1–v2.4 are checkpoints inside one 9B pipeline experiment; 27B-C/D are profiles inside one remote-KV experiment; release and attribution edits are not counted as new experiments.
- Corrected claim boundaries: 27B KV-only is a capacity/serving experiment, Ornith is a role-attribution and stability experiment, and the Flash layer-split run is a concurrency-envelope experiment because the latter two lack matched standalone speed controls.
- Added `data/experiment-index.csv` to map stable IDs to legacy labels, evidence type, canonical record, machine data, and comparability limits.
- Restored the already-published v2.4 Decode value and measurement label in `benchmark-results.csv`, so the machine-readable 9B final checkpoint matches its detailed record.

## v2.14

- Added the Qwen3.8-Flash Q4 dual-device layer-split record (r374): one llama-server on the RTX 3080 (CUDA0) and AI Max+ 395 (Vulkan1) with a 0.38 / 0.62 tensor split, ubatch 1024 / batch 4096, flash attention on, q4_0 KV cache, and 6 slots at 131072 context; 3080 VRAM peaked at 19129 MiB.
- Published the final C1–C6 envelope (~2077 in / 256 out, temperature 0): aggregate Prefill 569.892–633.685 tok/s (peak at C4), aggregate Decode 35.204–71.185 tok/s (peak at C4), and total throughput up to 338.270 tok/s at C4.
- All 21 scored requests returned HTTP 200 with complete timings; the warm-up C1 is excluded.
- Added the bilingual detailed record [qwen3.8-flash-q4-layer-split.md](results/qwen3.8-flash-q4-layer-split.md) and the machine-readable [qwen38flash-q4-local-results.csv](data/qwen38flash-q4-local-results.csv); the Flash-Q4 envelope is independent of the 27B and Ornith tables.

## v2.13

- Locked the Ornith section to the requested r337 dual-machine PD experiment: RTX 3080 is pure Prefill, while AI Max+ 395 performs all Decode work.
- Kept the measured short-task Prefill envelope at 4017.46–3943.88 tok/s and the 100K 395 pure-Decode aggregate at 23.33–148.20 tok/s; no single-node ROCmFP4 result is mixed in.
- Set the non-pure whole-stage wall-clock derivative to NA in all 12 CSV rows and renamed the public metric labels to 3080 pure Prefill / 395 pure Decode. The separately timed short-task 395 pure-Decode rate remains not recorded.

## v2.12

- Removed the derived whole-stage aggregate Decode column and its C1–C6 display values from the Ornith bilingual README tables and detailed reports.
- Kept the raw CSV unchanged for traceability; the published tables retain the 3080 aggregate Prefill and 395 decode-segment measurements.

## v2.11

- Added the Ornith-1.5-35B-A3B dual-machine PD experiment: the scored matrix used RTX 3080 CUDA full prefill (batch 4096 / ubatch 4096 / ctx 114688), KV migration via /dev/shm/kvxo, and 395 Vulkan1 full decode (ctx 655360) on the Ornith-1.5-35B-A3B-IQ4_XS main model with the Qwen3.6-35B-A3B-DFlash-Q4_K_M draft head (spec n_max 6); stress **42/42 passed, route=pd, n_reuse=0**. Online ctx was restored to 8192 / 32768 after the run.
- Short task (1000 in / 128 out) C1–C6 aggregate Prefill: 4017.46 down to 3943.88 tok/s. 100K (100000 in / 128 out) aggregate Prefill: 2895.53 down to 2793.24 tok/s.
- Published the independently attributable 395 decode-segment aggregate from 23.33 to 148.20 tok/s for the 100K tier.
- Advanced all six dFlash draft slots to 100K before the scored long-context rows to maintain positional continuity; that warm-up is excluded from the measurements.
- Marked Ornith-1.5-35B-A3B as a **MoE** (qwen35moe, 40 layers: 10 full attention + 30 Gated DeltaNet) that is not directly rankable against the existing 27B dense data.
- Short-task 395-only decode-segment speed, 100K TTFT, 100K single-stream decode, KV migration milliseconds, and the dFlash acceptance rate are absent from the source record; the CSV and docs leave them empty or not recorded, with no extrapolation.
- Added the bilingual detailed record [ornith-1.5-35b-a3b-dual-machine-pd.md](results/ornith-1.5-35b-a3b-dual-machine-pd.md) and the machine-readable [ornith35a3b-local-results.csv](data/ornith35a3b-local-results.csv); the v2.10 27B content and history remain unchanged.

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
