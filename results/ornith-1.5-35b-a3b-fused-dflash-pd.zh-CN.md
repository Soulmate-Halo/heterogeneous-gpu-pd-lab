# ORNITH-PD-02 — Ornith-1.5-35B-A3B 融合 DFlash 推测解码的 PD 配方

[English](ornith-1.5-35b-a3b-fused-dflash-pd.md)

## 实验契约

| 项目 | 定义 |
| --- | --- |
| 主要问题 | 在 ORNITH-PD-01 的独立 PD 之上加入 DFlash 融合草稿推测解码与统一 KV 池，Prefill 与单流 Decode 能否同时上一个台阶？ |
| 被测拓扑 | 三进程 PD：8080 路由（CPU）、8082 Prefill（RTX 3080 / CUDA0，1 槽）、8081 Decode + DFlash（AI Max+ 395 / Vulkan1，6 槽共用一个统一 KV 池）。 |
| 匹配速度对照 | 同一次部署内换题的负向对照（草稿接受率掉到 9.4%）；ORNITH-PD-01 短任务档的 3080 Prefill 是同负载参照。 |
| 变化因素 | 8082 的 batch/ubatch（4096 → 8196）与 8081 的 ctx（32768 → 131072）。 |
| 判定指标 | 单流 Prefill、单流 Decode、DFlash 草稿接受率、三端健康与是否 OOM。 |
| 通过边界 | 1000 输入 / 128 输出负载下 Prefill 超过 4000 tok/s、Decode 超过 100 tok/s，8080/8081/8082 全部 HTTP 200 且启动后无 OOM、无崩溃。 |
| 不能证明 | 相对单机的端到端加速；并发聚合吞吐（本实验只测单流，没有跑 C1–C6 矩阵）。 |

本档记录 Ornith-1.5-35B-A3B 上的第二个 PD 实验。它与 [ORNITH-PD-01](ornith-1.5-35b-a3b-dual-machine-pd.zh-CN.md) 用同一个模型、同一对硬件、同样的角色分工（3080 只做 Prefill、395 只做 Decode），区别是这一版在 395 那端把解码换成了 DFlash 融合草稿的推测解码（`--spec-type draft-dflash`，草稿步长 6），并让 6 个槽位共用一个统一 KV 池（`--kv-unified`）。**两个实验的负载口径不同**：1 号是 C1–C6 并发矩阵（短任务档与 100K 档各六路），2 号是单流 1000 输入 / 128 输出，因此只能在各自口径内读，不能合并排名。

**Ornith-1.5-35B-A3B 是 MoE（35B 总参数、A3B 命名即每 token 激活 3B）。它与 [qwen3.8-27b-dual-machine-pd.md](qwen3.8-27b-dual-machine-pd.zh-CN.md) 中的 Qwen3.8-27B dense 行不可直接排名**：不同模型家族、不同架构、不同每 token 激活算力、不同量化。

## 模型与拓扑

| 项目 | 取值 |
| --- | --- |
| 模型 | Ornith-1.5-35B-A3B（MoE，35B 总参数，A3B 每 token 激活） |
| GGUF | Ornith-1.5-35B-A3B-IQ4_XS |
| 架构 | qwen35moe，40 层：10 层全注意力 + 30 层 Gated DeltaNet |
| 草稿头 | Qwen3.6-35B-A3B-DFlash-Q4_K_M，`--spec-type draft-dflash`，草稿步长 n_max 6 |
| Prefill 节点 | RTX 3080 20GB，CUDA0，端口 8082：承担全部 Prefill，1 槽，ctx 8192 |
| Decode 节点 | AMD Ryzen AI Max+ 395，Vulkan1，端口 8081：承担全部 Decode，6 槽，统一 KV 池 |
| 路由节点 | CPU 上的 8080：Prefill 派给 8082，Decode 派给 8081 |
| KV 通道 | /dev/shm/kvxo |
| KV 缓存量化 | 两端都是 q4_0（k 与 v） |
| 负载 | 1000 输入 token / 128 输出 token，单流 |

## 主结果 — 1000 输入 / 128 输出，单流

单位均为 tok/s。

| 配置 | Prefill | Decode | DFlash 草稿接受 |
| --- | ---: | ---: | ---: |
| 最终配方（8082 batch/ubatch 4096，8081 ctx 32768） | **4173.47** | **114.86** | 107/114（**93.86%**） |
| 批次与上下文加大（8082 batch/ubatch 8196，8081 ctx 131072） | **4123.15** | **114.42** | 未记录 |
| 同配方换一道题（负向对照） | 3665.3 | 37.2 | **9.4%** |

第二行的 ctx 131072 是统一 KV 池下单请求的最大上下文，不是每个槽各留 128K。同一轮里另有一次附加性能门禁只测到 `3785/44`，原因同样是那道测试题的草稿接受率低，不是参数改坏了。

**解码成绩跟着草稿接受率走。** 同一套三进程配方，只换一道提示词，接受率从 93.9% 掉到 9.4%，Decode 就从 114.86 掉到 37.2 tok/s；接受率回到 93.86% 时 Decode 是 37.2 的 3.09 倍。所以这一档要复测必须锁同一道题、固定输出长度、`ignore_eos` 与同一个随机种子，否则拿到的是题目难度的差异，不是配方的差异。

## 与 ORNITH-PD-01 的关系

| 对比项 | ORNITH-PD-01（独立 PD） | ORNITH-PD-02（融合 DFlash 草稿） |
| --- | --- | --- |
| 1000 输入档 Prefill | C1 4017.46 → C6 3943.88（六路聚合） | 4173.47（单流） |
| 单流 Decode | 未记录 | 114.86 |
| 100K 档 | Prefill C1 2895.53 → C6 2793.24；395 纯 Decode C1 23.33 → C6 148.20 | 未测 |
| 草稿接受率 | 未记录 | 93.86% |
| 负载口径 | C1–C6 并发矩阵，42/42 成功 | 单流 1000 in / 128 out |

Prefill 的对比是同负载的：4173.47 ÷ 4017.46 − 1 = **+3.88%**，即融合配方比 1 号短任务档的 C1 高约 3.9%。

Decode 不能这样对比：1 号短任务档没有单独计时的 395 纯 Decode 速率，而 1 号 100K 档公布的 23.33–148.20 是六路并发的**聚合**值，2 号的 114.86 是**单流**一路的速度，两者不是同一个量。

## 三进程参数原文

8082 Prefill 节点（RTX 3080 / CUDA0）：

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

额外环境变量 `GGML_CUDA_DISABLE_GRAPHS=1`；CPU 线程未显式指定，实际为 16。批次加大那一组把 `--batch-size` 与 `--ubatch-size` 都改成 `8196`。

8081 Decode + DFlash 节点（AI Max+ 395 / Vulkan1）：

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

DFlash 其余实际值：`n_min=0`、`p_min=0.00`、`block_size=16`、`n_extract=8`、`sample_from_anchor=true`。上下文加大那一组把 `--ctx-size` 改成 `131072`。

8080 路由节点（CPU）：

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

## 现场与健康

- 8080 / 8081 / 8082 三端全部 HTTP 200，启动后日志无 OOM、无崩溃
- RTX 3080 显存占用 `18661/20480 MiB`（同一配方在参数复核那一轮记为 `18688/20480 MiB`）
- AI Max+ 395 Vulkan 占用约 `22678/65536 MiB`
- 三进程 PID 分别为 `2775429 / 2775363 / 2774975`
- 8090 端口全程未动；参数升级那一轮的远端备份在 `/home/hfy/r382/`

## 未记录（不做推测）

源记录不包含以下指标，本档与 CSV 一律留空或标注未记录，不做任何插值：

- C1–C6 并发矩阵 — 本实验只测单流
- 100K 长上下文成绩 — 未测（1 号实验有）
- TTFT — 未记录
- KV 迁移毫秒 — 未记录
- 批次加大那一组的草稿接受率 — 未记录

## 边界

- 没有同轮的单机对照，所以本档写的是这套配方能服务到什么程度，以及配方内部两次改参的差别，不能当作相对 395 单机或 3080 单卡的加速结论。
- Decode 的 114 这一档依赖 DFlash 草稿接受率，换题会掉回 37 一档；引用这个数时必须同时写明接受率。
- Prefill 与 ORNITH-PD-01 的对比只在 1000 输入档成立，且 1 号那一列是六路聚合、2 号是单流，方向可比、绝对值不宜精算。
- **MoE vs dense**：Ornith-1.5-35B-A3B 每 token 激活的算力远小于 Qwen3.8-27B dense 模型，上述数字不能并入 27B 行的任何排名。

数据来源：`r379_summary.md`（最终配方实测）与 `r382_ctx_prefill.md`（批次与上下文加大后的复测），数字逐字取自源记录，未做外推。

机器可读数据：[ornith35a3b-local-results.csv](../data/ornith35a3b-local-results.csv)。
