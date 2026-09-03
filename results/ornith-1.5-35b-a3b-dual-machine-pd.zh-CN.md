# Ornith-1.5-35B-A3B 双机 PD — 单一 MoE 口径，不与 27B dense 行直接排名

[English](ornith-1.5-35b-a3b-dual-machine-pd.md)

本档记录 v2.11 新增的 Ornith-1.5-35B-A3B 双机 Prefill/Decode（PD）实验。**Ornith-1.5-35B-A3B 是 MoE（35B 总参数、A3B 命名即每 token 激活 3B）。它与 [qwen3.8-27b-dual-machine-pd.md](qwen3.8-27b-dual-machine-pd.zh-CN.md) 中的 Qwen3.8-27B dense 行不可直接排名**：不同模型家族、不同架构、不同每 token 激活算力、不同量化。

## 模型与拓扑

| 项目 | 取值 |
| --- | --- |
| 模型 | Ornith-1.5-35B-A3B（MoE，35B 总参数，A3B 每 token 激活） |
| GGUF | Ornith-1.5-35B-A3B-IQ4_XS |
| 架构 | qwen35moe，40 层，full_attention_interval=4 |
| 层构成 | 10 层全注意力 + 30 层 Gated DeltaNet |
| 正式矩阵 Prefill 端 | RTX 3080，CUDA：全量 prefill，batch 4096、ubatch 4096、ctx 114688 |
| KV 迁移 | /dev/shm/kvxo |
| 正式矩阵 Decode 端 | AMD Ryzen AI Max+ 395，Vulkan1：全量 decode，ctx 655360 |
| 草稿头 | Qwen3.6-35B-A3B-DFlash-Q4_K_M，投机草稿 n_max 6 |
| 压测结果 | 42/42 全部成功，route=pd，n_reuse=0 |
| 压测后线上状态 | 服务恢复为 ctx 8192 / 32768；保留 batch/ubatch 4096 与草稿 n_max 6 |

## 两个 Decode 口径 — 读表前先明确

100K 档公布**两个不同的 Decode 指标**，不得混用：

1. **整档墙钟 Decode**（C1–C6：3.15–3.48 tok/s）。这是端到端墙钟速率，**包含 prefill、KV restore 与 decode**。它**不是 395 纯 decode 速率**——100K 输入下仅 prefill 就占据绝大部分墙钟时间。
2. **395 解码段聚合**（C1–C6：23.33–148.20 tok/s）。这是**只在 395 解码窗口内**测得的聚合速率。

短任务档（1000 in / 128 out）公布 Prefill 聚合与解码聚合；其解码聚合同样定义为总输出 token 除以整档墙钟，并非单独测得的 395 纯后端解码速率。

## 短任务 — 1000 输入 / 128 输出

单位均为 tok/s，同一 PD 服务内六个并发档实测。

| C | 3080 Prefill 聚合 | 解码聚合（总输出 token / 整档墙钟） |
| --- | ---: | ---: |
| C1 | **4017.46** | **112.71** |
| C2 | 3947.64 | 41.07 |
| C3 | 3924.83 | 72.56 |
| C4 | 3913.85 | 87.19 |
| C5 | 3906.09 | 75.92 |
| C6 | 3943.88 | 84.13 |

## 100K 档 — 100000 输入 / 128 输出

单位均为 tok/s。中间一列是整档墙钟口径，绝不能写成 395 纯 decode。

| C | 3080 Prefill 聚合 | 解码聚合（总输出 token / 整档墙钟） | 395 解码段聚合 |
| --- | ---: | ---: | ---: |
| C1 | **2895.53** | 3.15 | **23.33** |
| C2 | 2826.07 | 3.33 | 53.37 |
| C3 | 2796.34 | 3.38 | 75.20 |
| C4 | 2795.28 | 3.43 | 103.33 |
| C5 | 2793.56 | 3.48 | 123.09 |
| C6 | 2793.24 | 3.46 | 148.20 |

100K prefill 在 C1–C6 间保持约 3.6% 的波动（2895.53 降至 2793.24 tok/s）。整档墙钟速率维持在个位数低位，是因为 100K token 的 prefill 与 KV restore 主导了墙钟时间；395 解码段聚合则随并发近似线性增长，从 23.33 增至 148.20 tok/s。

## 压测结果

42/42 并发请求全部成功，全部走 PD 路径（`route=pd`），`n_reuse=0`——每一条流都做了全新 prefill，没有任何缓存 KV 复用抬高上述数字。

PD restore 只恢复主模型 KV，不恢复 dFlash 草稿槽位置；从短任务直接跳到 100K 会触发位置不连续错误。因此正式计分前先把 6 个草稿槽分别推进到 100K，该预热不计入成绩，随后 C1–C6 全部成功。

## 未记录（不做推测）

本实验的源记录不包含以下指标；CSV 与本文档一律留空或标注未记录，不做任何插值：

- 100K TTFT — 未记录
- 100K 单流 Decode — 未记录
- 短任务 395 纯解码段速率 — 未记录
- KV 迁移毫秒 — 未记录
- dFlash 草稿接受率 — 未记录

## 边界

- **MoE vs dense**：Ornith-1.5-35B-A3B 每 token 激活的算力远小于 Qwen3.8-27B dense 模型；上述每个数字都是独立的 MoE 口径，不能并入 27B 行的任何排名。
- 短任务与 100K 的解码聚合都采用总输出 token / 整档墙钟，绝不代表 395 后端纯 decode 指标。
- 395 解码段聚合只覆盖解码窗口；不含 prefill 与 KV restore 时间。
- 草稿头（Qwen3.6-35B-A3B-DFlash-Q4_K_M）与 IQ4_XS 主模型并行运行；未公布接受率，草稿的实际贡献无法量化。
- 短任务与 100K 口径输入长度不同，只能在各自表内比较。

数据来源：v2.11 对应的完整 `r337_hot_results.json` / `r337_hot_bench.log` 运行记录（数字逐字取自源记录，未做外推）。

机器可读数据：[ornith35a3b-local-results.csv](../data/ornith35a3b-local-results.csv)。
