# <img src="assets/soulmate-spirit.png" alt="器灵 Soulmate 小灵魂" width="44" align="absmiddle"> AI Max+ 395 加速 — 异构 GPU PD 实验室

[English](README.md)

本版本记录 RTX 3060 12GB 与 AMD Ryzen AI Max+ 395 / Radeon 8060S 在单机上的异构融合切层流水演进。

当前版本：**v2.3 — 平衡细化**。公开版本名有意隐藏内部切层策略；不发布部署步骤、命令、补丁、端点或凭据。

## 架构

prompt 微批依次经过 RTX 3060 / CUDA 前段模型层与 AI Max+ 395 / Vulkan 后段模型层。异步 RPC events 让相邻微批重叠执行。本实验使用 llama.cpp 上游 RPC events/async 能力，不宣称拥有自研引擎补丁。

## 公开检查点

测试口径：9B Q6_K、pp5064；仅在有记录的节点给出 tg128。

| 检查点 | 公开说明 | Prefill | Decode |
| --- | --- | ---: | ---: |
| RTX 3060 基线 | 仅 CUDA 端 | 1589.00 tok/s | 43.87 tok/s |
| AI Max+ 395 基线 | 仅 Vulkan 端 | 970.00 tok/s | 31.27 tok/s |
| v2.1 | 首个融合流水 | 1865.08 tok/s | 未记录 |
| v2.2 | 重叠细化 | 1893.87 tok/s | 未记录 |
| v2.3 | 平衡细化 | 1999.51 tok/s | 37.16 tok/s |

这些数据仅覆盖单机、单并发短基准；内部切层比例有意不公开。

详见 [双语演进记录](results/v2-evolution.zh-CN.md)、[结构化数据](data/benchmark-results.csv) 与 [更新记录](CHANGELOG_ZH.md)。