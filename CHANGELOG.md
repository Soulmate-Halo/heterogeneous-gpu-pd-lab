# Changelog

[简体中文](CHANGELOG_ZH.md)

Public versions are the seven measured experiment milestones below. [VERSION_HISTORY.md](VERSION_HISTORY.md) maps them to older publication numbers. The 2026-09-11 Spark addendum is experiment data, not a new public version.

## 2026-09-11 experiment data addendum

Not a new public version. Public version remains **v1.6**.

- Added 27B-SPARK-01 on Qwen3.8-27B Q4_K_M / KV q4_0, RTX 3080 20GB + DGX Spark GB10: three confirmed working profiles after search (Highest Prefill confirmed, Highest Decode confirmed, Balanced). Highest means this search's confirmed working point, not a global optimum.
- Highest Prefill confirmed, 8K in / 128 out, no speculation, median of three warm repeats: pair Prefill **1603.20** tok/s and Decode **17.62**, versus 3080 standalone 1113.13 / 33.36 (Prefill **+44.03%**). Later live retest of the same profile: 8K 1601.05 / 17.41. Do not quote 1700. Pair scores are combination scores, not 3080-only.
- Highest Decode confirmed, DFlash2 on: 2K 1153.95 / 63.97 and 8K 1124.32 / 49.20. 3080 standalone with its own DFlash2 tuning: 2K 1042.97 / 60.37 and 8K 1015.15 / 58.36. Speculation settings differ; this is a system-to-system control, not a one-variable change.
- Balanced, 8K cold in / 256 out, DFlash2, six-slot serving, C1-C6: aggregate Prefill 1097.40, 1094.21, 1089.37, 1086.44, 1087.57, 1088.06; aggregate Decode 47.69, 31.18, 28.04, 24.90, 24.77, 25.55 (whole-batch window, includes other-stream Prefill). 36 official waves, 126/126 official requests. No standalone C1-C6 control.
- Record: [qwen3.8-27b-spark-3080-profiles.md](results/qwen3.8-27b-spark-3080-profiles.md). CSV: [qwen27b-spark-3080-profiles.csv](data/qwen27b-spark-3080-profiles.csv). Community NVFP4 Spark rows stay external and are not ranked with these three profiles.

## v1.6

- Measured ORNITH-PD-02 on the same Ornith-1.5-35B-A3B IQ4_XS pair as ORNITH-PD-01, adding fused DFlash speculative decode (`--spec-type draft-dflash`, n_max 6) and a unified KV pool (`--kv-unified`).
- Result: single-stream 1000 in / 128 out Prefill **4173.47 tok/s**, Decode **114.86 tok/s**, draft acceptance **107/114 (93.86%)**; after batch/ubatch 8196 and ctx 131072, Prefill **4123.15** and Decode **114.42**. All three endpoints HTTP 200, no OOM; 3080 18661/20480 MiB, 395 Vulkan about 22678/65536 MiB.
- Conclusion: Prefill holds above 4000 while single-stream Decode reaches 114 when draft acceptance is high. A prompt the head could not predict fell to 3665.3 / 37.2 (9.4% acceptance); another gate in the same round reached only 3785/44 for the same reason.
- Limit: Decode follows draft acceptance; re-measure with the same prompt, fixed output length, and the same seed. Record: [ornith-1.5-35b-a3b-fused-dflash-pd.md](results/ornith-1.5-35b-a3b-fused-dflash-pd.md).

## v1.5

- Measured FLASH-SPLIT-01: one llama-server on RTX 3080 (CUDA0) and AI Max+ 395 (Vulkan1), Qwen3.8-Flash Q4, tensor split 0.38 / 0.62, ubatch 1024 / batch 4096, q4_0 KV, 6 slots at 131072 context; 3080 VRAM peaked at 19129 MiB.
- Result: ~2077 in / 256 out, C1–C6 aggregate Prefill 569.892–633.685 tok/s (peak C4 **633.685**), aggregate Decode 35.204–71.185 tok/s (peak C4 **71.185**), total throughput C4 **338.270** tok/s; 21/21 scored requests succeeded. C1→C4 Prefill +11.2%, aggregate Decode +102.2%, total throughput +58.4%.
- Conclusion: C4 is the best concurrency tier for this workload; C5 and C6 no longer rise.
- Limit: an operating point, not a matched speed-up versus a standalone card; a different workload must be measured again. Record: [qwen3.8-flash-q4-layer-split.md](results/qwen3.8-flash-q4-layer-split.md).

## v1.4

- Measured ORNITH-PD-01: RTX 3080 full Prefill and AI Max+ 395 full Decode on Ornith-1.5-35B-A3B IQ4_XS with Qwen3.6-35B-A3B-DFlash Q4_K_M; 42/42 requests, `route=pd`, `n_reuse=0`.
- Result: 1000 in / 128 out: 3080 Prefill aggregate from 1 stream (C1) **4017.46** to 6 streams (C6) **3943.88** (−1.8%); 100K: 3080 Prefill aggregate C1 **2895.53** to C6 **2793.24** (−3.5%); 100K: 395 Decode aggregate C1 **23.33** to C6 **148.20** (6.35×).
- Conclusion: MoE PD stays stable at 100K across 1 to 6 concurrent streams, and Prefill versus Decode stays attributable.
- Limit: the short stage has no separately timed 395 pure-Decode rate; no wall-clock-derived aggregate Decode is published. Record: [ornith-1.5-35b-a3b-dual-machine-pd.md](results/ornith-1.5-35b-a3b-dual-machine-pd.md).

## v1.3

- Measured 27B-KV-01 remote KV with two decode routes, and 27B-DRAFT-AUDIT-01, on Qwen3.8-27B Q4_K_M / KV q4_0. With the 524 MiB recurrent-state checkpoint copy off, 3080 serving Prefill rose from **1000.6** to **1210.6 tok/s** (98.5% of raw 1228.53) and the 395 solo control from **207.2** to **307.1**.
- Result, route C (395 decodes, 3080 headless Prefill): Prefill **1194.4–1210.6**, aggregate Decode C1 **33.55** / C6 **63.84** (+90.3%). Result, route D (3080 decodes): Prefill **1077–1090** (C1–C6: 1090 / 1081 / 1080 / 1077 / 1082 / 1082; same-run aggregate 1079 → 1016), aggregate Decode C1 **63.2** / C6 **116.3** (+84.0%). 3080 with the head: natural language 42.7 tok/s, code 67.4 tok/s (2.2–2.3× the 395 with the same head); 395 with the head 38.75 tok/s single-stream. Draft-head weights 1080 MiB plus ~500 MiB verification buffer; ubatch 1024 → 512, slots 2 → 1; verification matmul +21 ms / +49 ms. Audit: repetitive text 35.0–38.5 tok/s at 100% acceptance; natural-language C1 **12.1 tok/s** at 17.7% (−68.6%).
- Conclusion: take C when Prefill is the constraint, D when aggregate Decode is; the 395 never runs dense Prefill. A repetitive-text speculative-decode score does not stand in for real text.
- Limit: D Prefill is about ten percent below C because Decode shares 3080 compute and VRAM; C and D are two routes of one experiment, not two experiments. Record: [qwen3.8-27b-dual-machine-pd.md](results/qwen3.8-27b-dual-machine-pd.md).

## v1.2

- Measured 27B-PD-01 serving PD: RTX 3080 20GB Prefill and AI Max+ 395 Decode on Qwen3.8-27B Q4_K_M, six concurrency tiers C1–C6 on split and solo paths.
- Result: C1 v1.0 method TTFT **1073 ms** / Prefill **1000.6** / Decode **38.75** versus 395 solo 4825 ms / 207.2 / 36.33 (−77.8% TTFT, Prefill +382.9% / 4.83×, Decode +6.7%). Removing per-ubatch RPC sync lifted Prefill from **683.2** to **1000.6** (+46%), about 82% of 3080 raw pp1024 **1228.53** (pp4096 1203.06, tg64 33.08). Serving Prefill held **1000–1015 tok/s** as concurrency rose. KV transfer 68–76 ms (the serving record also states 71 ms).
- Conclusion: phase separation works while serving; Prefill does not fall off as concurrency rises; Decode is unharmed. 207.2 is the measured v1.0-method 395 solo C1, not a typo for 1207.2.
- Limit: this verifies separated phases, not both devices computing the same phase. Record: [qwen3.8-27b-dual-machine-pd.md](results/qwen3.8-27b-dual-machine-pd.md).

## v1.1

- Measured the fused 9B asynchronous layered pipeline (four checkpoints now under this milestone) and the 27B IQ3 layer-split long-context check 27B-LONG-01. 9B Ornith Q6_K, RTX 3060 12GB + AI Max+ 395; 27B Qwen3.8-27B UD-IQ3_XXS.
- Result, 9B-PIPE-01: pair Prefill **2129.69** / Decode **50.73** versus 3060 alone 1589.00 / 43.87 (+34.0% / +15.6%) and 395 `llama-bench` 970.00 / 31.27 (+119.6% / +62.2%). Checkpoint Prefill climb: **1865.08 → 1893.87 → 1999.51 → 2129.69**; the 37.16 tok/s decode belongs only to the 1999.51 checkpoint, not the final 50.73. Result, 27B-LONG-01: pair pp4096 **658.52**, pp65536 **319.10**, pp98304 **225.10**, tg64 **19.57** versus 395 alone 313.28 / 136.69 / timed out at 900 s / 18.26 (+110.2% / +133.4% / timeout → finished / +7.2%).
- Conclusion: Dense Acceleration at 9B beats the fastest single card present; when the 3060 cannot hold 27B, the layer split finishes 98K and more than doubles Prefill at 4K and 64K.
- Limit: 9B gains need the same controls on another model; 27B-LONG-01 is IQ3, not Q4. Pipeline record: [v2.4-fused-layer-pipeline.md](results/v2.4-fused-layer-pipeline.md). Long-context record: [qwen3.8-27b-dual-machine-pd.md](results/qwen3.8-27b-dual-machine-pd.md).

## v1.0

- Measured 9B-PD-01 independent heterogeneous PD: Ornith 9B Q6_K, RTX 3060 Prefill handing state to Vulkan Decode on the AI Max+ 395, 5064 in / 128 out, four measured handoff nodes, versus the single-host 9B boundary.
- Result: pair TTFT **3.496 s** / Prefill **1452.29** / Decode **30.28** versus 395 serving 5.879 s / 861.55 / 30.24 (−40.5% TTFT, Prefill +68.6%, Decode +0.1%).
- Conclusion: the CUDA-side state hands over in one piece; first token is sooner; Decode holds.
- Limit: the two phases still run one after the other, so this is phase separation, not Dense Acceleration. Record: [v1.0-independent-pd.md](results/v1.0-independent-pd.md).
