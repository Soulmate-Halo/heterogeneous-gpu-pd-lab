# <img src="assets/soulmate-spirit.png" alt="器灵 Soulmate 小灵魂" width="44" align="absmiddle"> 稠密加速 · 异构 GPU PD 实验室

[English](README.md)

**让小显卡与大内存主机一起算，把已有设备的算力用起来。**

- **9B：组合 Prefill 比 3060 单卡快 34.0%，比 395 单主机快 119.6%。**
- **27B：3080 + DGX Spark 的 Prefill 比本次 3080 单卡快 44.03%。** 8K 输入 / 128 输出，无投机。
- **27B 长上下文：组合的 64K Prefill 达到 395 单机的 2.33 倍，98K 从超时到跑完。**

一张算力强但显存小的 NVIDIA 显卡（RTX 3060 12GB / RTX 3080 20GB）配一台大内存伙伴，可以一起跑大模型推理。三条路线：RTX 3060 12GB + AMD AI Max+ 395，RTX 3080 20GB + 395，以及 RTX 3080 20GB + DGX Spark GB10。**稠密加速**指两台设备同时算同一个模型的同一个阶段。当前版本 **v1.6**，七个公开里程碑，另有 2026-09-11 的本地 Spark 实测补录（不是新版本）。不公开部署命令、代码补丁、服务地址和切层策略。

C1–C6 指同时发 1 到 6 路请求，C1 是单流；聚合是同时在跑的几路加起来的速度。速度单位为 tok/s，除非该行写的是 TTFT。

## 一眼对照：单卡、单主机、组合

9B 上，3060 + 395 是本地同时具备单卡和单主机对照的格子，组合比两者都快。27B Q4_K_M 上，DGX Spark 只测了 3080 + Spark 组合对本次 3080；Spark 单机 **未测**。

### 9B 稠密加速 · Ornith 9B · Q6_K · RTX 3060 12GB + 395

`llama-bench` pp5064 / tg128。数据：[benchmark-results.csv](data/benchmark-results.csv)。详档：[v2.4 融合分层流水](results/v2.4-fused-layer-pipeline.zh-CN.md)。提升 = 组合 ÷ 分母 − 1。

| 指标 | 3060 单卡 | 395 单主机 | 3060 + 395 组合 | 比 3060 | 比 395 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prefill（tok/s） | 1589.00 | 970.00 | **2129.69** | **+34.0%** | **+119.6%** |
| Decode（tok/s） | 43.87 | 31.27 | **50.73** | **+15.6%** | **+62.2%** |

### DGX Spark · Qwen3.8-27B · Q4_K_M · RTX 3080 20GB + Spark GB10

最高 Prefill 与最高 Decode 是**不同工作档**；Decode 优先档中，组合与单卡分别调优了投机参数。下表按工作档和输入长度分开对照。数据：[qwen27b-spark-3080-profiles.csv](data/qwen27b-spark-3080-profiles.csv)。详档：[Spark + 3080 三档](results/qwen3.8-27b-spark-3080-profiles.zh-CN.md)。

**最高 Prefill 确认档 — 8K 输入 / 128 输出，无投机**（系统预热后三次正式测量中位数；预热轮不计）。同档复测：1601.05 / 17.41。

| 指标 | 3080 单卡 | Spark 单机 | 3080 + Spark 组合 | 相对 3080 增减 |
| --- | ---: | ---: | ---: | ---: |
| Prefill（tok/s） | 1113.13 | 未测 | **1603.20** | **+44.03%** |
| Decode（tok/s） | 33.36 | 未测 | 17.62 | **-47.18%** |

**Decode 优先，DFlash2 — 2K 输入 / 128 输出**（3080 单机用自己那套 DFlash2 调参；不是 Prefill 确认档）。

| 指标 | 3080 单卡 | Spark 单机 | 3080 + Spark 组合 | 相对 3080 增减 |
| --- | ---: | ---: | ---: | ---: |
| Prefill（tok/s） | 1042.97 | 未测 | 1153.95 | **+10.64%** |
| Decode（tok/s） | 60.37 | 未测 | **63.97** | **+5.96%** |

**Decode 优先，DFlash2 — 8K 输入 / 128 输出**（与 2K 行同一 Decode 确认档；2K 与 8K 不拼接）。

| 指标 | 3080 单卡 | Spark 单机 | 3080 + Spark 组合 | 相对 3080 增减 |
| --- | ---: | ---: | ---: | ---: |
| Prefill（tok/s） | 1015.15 | 未测 | 1124.32 | **+10.75%** |
| Decode（tok/s） | 58.36 | 未测 | 49.20 | **-15.70%** |

均衡服务 C1–C6 **没有** 3080 单机、也 **没有** Spark 单机的并发对照，所以这些行不宣称组合收益。完整 C1–C6 表和计时定义见 [27B-SPARK-01](#27b-spark-01--qwen38-27b--q4_k_m--rtx-3080-20gb--dgx-spark-gb10)。

## 结果导航

每行一个实验：短标题，再给详档。完整数字在 [实验数据](#实验数据)。

| 做法 | 实验 | 短结果 | 详档 |
| --- | --- | --- | --- |
| 稠密加速 | 9B-PIPE-01 · Ornith 9B · Q6_K · 3060 + 395 | 组合 Prefill **2129.69** / Decode **50.73** tok/s；比 3060 和 395 都快 | [详档](results/v2.4-fused-layer-pipeline.zh-CN.md) |
| 分层装 | 27B-LONG-01 · Qwen3.8-27B · UD-IQ3_XXS · 3060 + 395 | 对 395：pp4096 **658.52** 对 313.28（**+110.2%**） | [详档](results/qwen3.8-27b-dual-machine-pd.zh-CN.md) |
| 阶段分离 PD | 9B-PD-01 · Ornith 9B · Q6_K · 3060 + 395 | 对 395 服务态：TTFT **3.496 秒** 对 5.879 秒（**-40.5%**） | [详档](results/v1.0-independent-pd.zh-CN.md) |
| 阶段分离 PD | 27B-PD-01 · Qwen3.8-27B · Q4_K_M · 3080 + 395 | 对 395 服务态 C1：TTFT **1073 毫秒** 对 4825 毫秒；Prefill **1000.6** 对 207.2（4.83×） | [详档](results/qwen3.8-27b-dual-machine-pd.zh-CN.md) |
| 远端 KV 池 | 27B-KV-01 · Qwen3.8-27B · Q4_K_M · 3080 + 395 | C1 → C6 聚合 Decode：C（395 解码）33.55 → 63.84；D（3080 解码）63.2 → 116.3 tok/s | [详档](results/qwen3.8-27b-dual-machine-pd.zh-CN.md) |
| MoE 的 PD | ORNITH-PD-01 · Ornith-1.5-35B-A3B · IQ4_XS · 3080 + 395 | 100K、C1 → C6：395 Decode 聚合 23.33 → 148.20（6.35×）；42/42 | [详档](results/ornith-1.5-35b-a3b-dual-machine-pd.zh-CN.md) |
| MoE 融合草稿 | ORNITH-PD-02 · Ornith-1.5-35B-A3B · IQ4_XS · 3080 + 395 | 3080 Prefill **4173.47**；395 + DFlash 单流 Decode **114.86** tok/s；107/114（93.86%） | [详档](results/ornith-1.5-35b-a3b-fused-dflash-pd.zh-CN.md) |
| 单服务切层 | FLASH-SPLIT-01 · Qwen3.8-Flash · Q4 · 3080 + 395 | 最好 C4：Prefill 633.685、Decode 71.185、总吞吐 338.270 tok/s | [详档](results/qwen3.8-flash-q4-layer-split.zh-CN.md) |
| 双机组合 | 27B-SPARK-01 · Qwen3.8-27B · Q4_K_M · 3080 + Spark | Prefill **1603.20** 对 3080 1113.13（**+44.03%**）；2K Decode **63.97** | [详档](results/qwen3.8-27b-spark-3080-profiles.zh-CN.md) |
| 外部参考 | EXT-DGX-01 · Qwen3.5 9B / TQ3_4S；Qwen3.8-27B / NVFP4 | 社区 Spark 数字，只当背景 | [详档](results/dgx-spark-community-control.zh-CN.md) |

## 为什么稠密加速是这个仓库的重点

三点优势，一点边界。

① 在本仓库的本地数据里，稠密加速是 9B 上组合同时超过单卡和单主机的那一格。Prefill 2129.69 对 3060 1589.00（+34.0%，÷ 1589.00）、对 395 970.00（+119.6%，÷ 970.00）；Decode 50.73 对 43.87（+15.6%）、对 31.27（+62.2%）。多出来的是 395 加进来的算力。第二条本地稠密组合是 RTX 3080 20GB + DGX Spark GB10，Qwen3.8-27B Q4_K_M：最高 Prefill 确认档 8K/128、无投机，组合 Prefill **1603.20** 对本次 3080 1113.13（**+44.03%**，÷ 1113.13）。只有 9B 有双单机基线；Spark 没有 Spark 单机基线，所以 +44.03% 不是超过所有单机。

② 阶段分离 PD 的天花板是加速卡自己的裸算。27B-PD-01 组合 Prefill 1000.6 到 3080 llama-bench 裸算 1228.53 的 82%，关掉检查点后的 1210.6 是该上限的 98.5%。裸算和服务端到端**不是同一口径**。PD 下 Decode 基本不动（9B：30.28 对 30.24；27B：38.75 对 36.33）。PD 解决的是首字时间与容量，不是把两台设备的算力加起来。

③ 稠密加速不要求加速卡装下整个模型。分层装的 27B-LONG-01：pp4096 658.52 对 395 313.28、pp65536 319.10 对 136.69、pp98304 从 900 秒超时变成 225.10。

④ 边界：完整的单卡 + 单机两组对照目前只有 9B Q6_K + RTX 3060 这一格（D1）。27B Spark 组合在最高 Prefill 确认档上有匹配的 3080 单机对照，但没有 Spark 单机基线。395 主机上 27B 与 MoE 的重叠流水对照仍在后续规划。稠密区需要相邻微批能错开（重叠窗口），排不进窗口的部分归稀疏区，只负责把模型装下。

## 这套方案怎么工作

先说问题，再说做法。

一张消费级显卡算得快，但显存小，大一点的模型装不进去；一台大内存主机什么模型都装得下，但算得慢。解法是让两台设备同时算同一个模型的同一个阶段，这叫**稠密加速**。加速卡装前段层并计算这些层；大内存主机装后段层与中间状态，计算后半段。关键在**稠密区**：相邻微批在这里错开跑，同一时刻两台设备都有活。排不进重叠窗口的部分归**稀疏区**：把整个模型装下来，撑住上下文容量。

![基础架构：Prompt 先进异步微批队列，小显存加速卡算前段层，大内存主机算后段层与状态，两台设备在稠密区里对相邻的微批同时开工](assets/base-architecture.zh-CN.png)

<details>
<summary>这张图的 mermaid 源码（可以复制到 mermaid.live 打开）</summary>

```text
flowchart LR
    P["Prompt"] --> Q["异步微批队列"]
    subgraph D["稠密区——并发有效窗口"]
        direction LR
        N["小显存加速卡<br/>前段层"] -->|"当前微批"| A["大内存主机<br/>后段层与状态"]
    end
    Q --> N
    A --> O["Decode 与结果流"]
    N -.->|"下一微批重叠"| A
```

</details>

它不是什么：不是把层一个个排队跑完，不是张量并行，不是两台设备重复算同一层，也不是把 395 当远端 KV 仓库的那种独立 PD。

![稠密加速结构](assets/dense-region-structure.png)

## 设备怎么分工

395 主机上的实验归到下面五种分工。RTX 3080 20GB + DGX Spark GB10 是同一组合的三个工作档。短结果后接详档。

| 分工方式 | 加速卡做什么 | 伙伴设备 | 短结果 | 实验 |
| --- | --- | --- | --- | --- |
| 稠密加速 | 算前段层，与主机同时算同一阶段 | AI Max+ 395 算后段层与中间状态 | 组合超过 3060 和 395：Prefill 2129.69 tok/s | [9B-PIPE-01](results/v2.4-fused-layer-pipeline.zh-CN.md) |
| 分层装 | 只装前段层并计算 | AI Max+ 395 装后段层 | 对 395：pp4096 658.52 对 313.28（**+110.2%**） | [27B-LONG-01](results/qwen3.8-27b-dual-machine-pd.zh-CN.md) |
| 阶段分离 PD | 做全部 Prefill | AI Max+ 395 做 Decode | 首字更快；Decode 基本不动 | [9B-PD-01](results/v1.0-independent-pd.zh-CN.md)、[27B-PD-01](results/qwen3.8-27b-dual-machine-pd.zh-CN.md)、[ORNITH-PD-01](results/ornith-1.5-35b-a3b-dual-machine-pd.zh-CN.md)、[ORNITH-PD-02](results/ornith-1.5-35b-a3b-fused-dflash-pd.zh-CN.md) |
| 远端 KV 池 | Prefill；配置 D 由 3080 解码 | AI Max+ 395 远端 KV；配置 C 由 395 解码 | C Decode 33.55 → 63.84；D 63.2 → 116.3 | [27B-KV-01](results/qwen3.8-27b-dual-machine-pd.zh-CN.md) |
| 单服务切层 | 同一 llama-server 内张量切分 0.38 | AI Max+ 395 张量切分 0.62 | C4 Prefill 633.685、Decode 71.185 tok/s | [FLASH-SPLIT-01](results/qwen3.8-flash-q4-layer-split.zh-CN.md) |
| Prefill 优先，无投机 | 与 Spark 一起算同一个模型 | DGX Spark GB10 与 3080 一起算 | Prefill **1603.20** 对 1113.13（**+44.03%**）；Decode 17.62 对 33.36 | [27B-SPARK-01](results/qwen3.8-27b-spark-3080-profiles.zh-CN.md) |
| Decode 优先，DFlash2 | 与 Spark 一起算同一个模型 | DGX Spark GB10 与 3080 一起算 | 2K Decode **63.97** 对 60.37；8K 49.20 对 58.36 | [27B-SPARK-01](results/qwen3.8-27b-spark-3080-profiles.zh-CN.md) |
| 均衡服务，DFlash2 | 与 Spark 一起算同一个模型 | DGX Spark GB10 与 3080 一起算 | C1–C6 聚合 Prefill 1086.44–1097.40，Decode 24.77–47.69；126/126；无单机 C1–C6 | [27B-SPARK-01](results/qwen3.8-27b-spark-3080-profiles.zh-CN.md) |

395 矩阵里只有第一种是两台设备同时算同一阶段。Spark 组合也是两机一起算同一个模型；Prefill 提升只对照本次 3080 单机，没有 Spark 单机基线。

## 三条硬件路线与六类实验矩阵

这是三条硬件路线上的实验覆盖地图，不是同一套负载下的统一排名。路线一是 RTX 3060 12GB + AMD AI Max+ 395。路线二是 RTX 3080 20GB + 395。路线三是 RTX 3080 20GB + DGX Spark GB10。六格按「模型跑起来占多少显存 vs 加速卡显存」分为轻松装、装满、装不下；395 主机每格五种目标配置：RTX 3060 单卡、RTX 3080 单卡、AI Max+ 395 单机、3060 + 395、3080 + 395。同一格内模型、量化、prompt、上下文、并发、指标必须一致；单卡装不下本身是有效结果，不能换小模型或更狠量化去凑基线。仓库里没有任何 RTX 3090 的实测数据，这张卡在选型阶段就被排除了。

| 对比项 | 路线一：RTX 3060 12GB + 395 | 路线二：RTX 3080 20GB + 395 | 路线三：RTX 3080 20GB + DGX Spark |
| --- | --- | --- | --- |
| 覆盖版本 | v1.0 → v1.1 | v1.2 → v1.6 | 2026-09-11 补录（仍是 v1.6） |
| 显存与能装什么 | 12GB，只装得下 9B 这一档稠密模型；27B 只能靠 IQ3 分层装 | 20GB，27B Q4 装得下，MoE 切层也因此成立 | 3080 20GB 加 Spark GB10；这条线上跑 27B Q4 |
| 跑过的模型 | Ornith 9B 稠密 Q6_K；Qwen3.8-27B IQ3 | Qwen3.8-27B 稠密 Q4；Ornith-1.5-35B-A3B 和 Qwen3.8-Flash | Qwen3.8-27B 稠密 Q4_K_M |
| 实验编号 | 9B-PD-01、9B-PIPE-01、27B-LONG-01 | 27B-PD-01、27B-KV-01、27B-DRAFT-AUDIT-01、ORNITH-PD-01、ORNITH-PD-02、FLASH-SPLIT-01 | 27B-SPARK-01 |
| 短结果 | 组合 Prefill 2129.69 tok/s 对 3060 1589.00 / 395 970.00；27B-LONG 对 395 pp4096 **+110.2%** | 服务态 PD TTFT 1073 毫秒对 395 4825 毫秒；远端 KV 与 MoE C1–C6 | Prefill **1603.20** 对 3080 1113.13（**+44.03%**）；2K Decode **63.97**；C1–C6 126/126；Spark 单机未测 |

Spark 格没有本地实测就写尚未测试；不把 395 数字填进那些 Spark 格，也不用社区 NVFP4 行当 Spark 路线证据。

| 实验格 | 要回答什么 | 395 路线实测 | Spark 路线实测 |
| --- | --- | --- | --- |
| **D1 稠密 · 轻松装**<br>Ornith 9B · Q6_K · RTX 3060 | 模型装得下还有余量时，稠密流水能不能真比更快的那张卡还快，而不只是多装点东西？ | **已验证**：9B-PIPE-01 有 3060 单卡和 395 单机两组对照，组合比两者都快。[详档](results/v2.4-fused-layer-pipeline.zh-CN.md) | 本地尚未测试。 |
| **D2 稠密 · 装满**<br>Qwen3.8-27B · Q4_K_M · RTX 3080 | 显存快用满时，整模塞一张卡、阶段分离、分层稠密这三条路哪条更好？ | **已验证**：27B-PD-01，3080 Prefill + 395 Decode 服务态；TTFT 对 395 降四分之三以上；C1–C6 全部跑通。[详档](results/qwen3.8-27b-dual-machine-pd.zh-CN.md) | **27B-SPARK-01**：Prefill **1603.20** 对 3080 1113.13（**+44.03%**）；Decode 优先 2K **63.97**；均衡 C1–C6；Spark 单机未测。[详档](results/qwen3.8-27b-spark-3080-profiles.zh-CN.md) |
| **D3 稠密 · 装不下**<br>Qwen3.8-27B · UD-IQ3_XXS · RTX 3060 | 一张卡干不完这个活时，分层装能不能跑完，而且比 395 快？ | **已验证**：27B-LONG-01 对 395 在 4K / 64K 为 **+110.2%** / **+133.4%**；98K 超时 → 225.10。[详档](results/qwen3.8-27b-dual-machine-pd.zh-CN.md) | 本地尚未测试。 |
| **M1 MoE · 轻松装**<br>模型与量化待定 | 激活参数不大、显存也有余量时，MoE 的路由开销会不会把重叠省下来的时间吃回去？ | 列入后续规划。 | 本地尚未测试。 |
| **M2 MoE · 装满**<br>Ornith-1.5-35B-A3B · IQ4_XS · RTX 3080 | MoE 快把 3080 装满时，阶段分离稳不稳；再往上做稠密重叠还有没有收益？ | **已验证**：ORNITH-PD-01 42/42，100K Decode 23.33 → 148.20（[详档](results/ornith-1.5-35b-a3b-dual-machine-pd.zh-CN.md)）；ORNITH-PD-02 Prefill 4173.47、Decode 114.86（[详档](results/ornith-1.5-35b-a3b-fused-dflash-pd.zh-CN.md)）。 | 本地尚未测试。 |
| **M3 MoE · 装不下**<br>Qwen3.8-Flash · Q4 · RTX 3080 先导 | 模型总占用超过两张基准卡时，按层或按专家分开装，能不能同时保住能跑、吞吐和输出正确？ | FLASH-SPLIT-01 先导：C4 Prefill 633.685、Decode 71.185 tok/s。完整实验列入后续规划。[先导记录](results/qwen3.8-flash-q4-layer-split.zh-CN.md) | 本地尚未测试。 |

## 实验数据

每个实验一张小表，这是首页唯一放完整数据的地方。数字抄自 `results/` 与 `data/`；首页和详档如果有出入，以详档和 CSV 为准。提升幅度按「组合成绩 ÷ 对照成绩 − 1」算，TTFT 写的是降了多少。llama-bench 裸算和服务端到端不当同一口径比赛。

**稠密加速。** 9B-PIPE-01 有 3060 单卡与 395 单机两组对照。27B-SPARK-01 在最高 Prefill 确认档上有匹配的 3080 单机对照，没有 Spark 单机基线。最高 Prefill 只比本次 3080 单机 +44.03%。

### 9B-PIPE-01 · Ornith 9B · Q6_K · 异步分层稠密加速

加速卡 RTX 3060 12GB；模型 Ornith 9B，权重量化 Q6_K（详档与 CSV 记作「9B · Q6_K」），`llama-bench` pp5064 / tg128；数据出自 [benchmark-results.csv](data/benchmark-results.csv)。

| 指标 | 3060 单卡 | 395 单主机 | 3060 + 395 组合 | 比 3060 | 比 395 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prefill（tok/s） | 1589.00 | 970.00 | **2129.69** | **+34.0%** | **+119.6%** |
| Decode（tok/s） | 43.87 | 31.27 | **50.73** | **+15.6%** | **+62.2%** |

验证了什么：两台设备在稠密区里都真在出力，组合成绩比在场最快的单卡还快。这是仓库里稠密加速的核心证据。

### 27B-SPARK-01 · Qwen3.8-27B · Q4_K_M · RTX 3080 20GB + DGX Spark GB10

本地实测，RTX 3080 20GB + DGX Spark GB10 路线，已列入 D2 覆盖图。模型 Qwen3.8-27B，权重量化 Q4_K_M，KV 缓存 q4_0。「最高」只表示本次搜索后确认的工作档，不宣称全局最优。组合成绩不是 3080 单机成绩。数据出自 [qwen27b-spark-3080-profiles.csv](data/qwen27b-spark-3080-profiles.csv)。详档：[qwen3.8-27b-spark-3080-profiles.zh-CN.md](results/qwen3.8-27b-spark-3080-profiles.zh-CN.md)。

三张确认对照表在本页顶部（最高 Prefill 确认档 8K/128 无投机；Decode 优先 DFlash2 2K/128；Decode 优先 DFlash2 8K/128）。Decode 确认档对照上的投机参数按 3080 单机自己调，不是只改一个变量。后来同档 Prefill 确认档 live 复测，8K 三次为 1601.05 / 17.41。不要写成 1700。

**吞吐，RTX 3080 20GB + Spark 均衡 C1–C6**

以下速度单位为 tok/s。Prefill 来自独立预填充波次；Decode 和全程输出来自输出 256 token 的完整请求波次，两类波次按并发档汇总。具体计时定义见三档详档。无单机 C1–C6 对照——不写组合提升列。

| C | 聚合 Prefill（tok/s） | 聚合 Decode（tok/s） | 全程输出（tok/s） |
| --- | ---: | ---: | ---: |
| C1 | 1097.40 | 47.69 | 19.91 |
| C2 | 1094.21 | 31.18 | 19.49 |
| C3 | 1089.37 | 28.04 | 20.47 |
| C4 | 1086.44 | 24.90 | 19.86 |
| C5 | 1087.57 | 24.77 | 20.53 |
| C6 | 1088.06 | 25.55 | 21.62 |

**时延，RTX 3080 20GB + Spark 均衡 C1–C6**

| C | TTFT p95（秒） | 整批耗时（秒） |
| --- | ---: | ---: |
| C1 | 7.489 | 12.857 |
| C2 | 19.145 | 26.273 |
| C3 | 28.990 | 37.524 |
| C4 | 38.932 | 51.564 |
| C5 | 48.471 | 62.336 |
| C6 | 58.317 | 71.048 |

36 个正式波次，126/126 正式请求。验证了什么：最高 Prefill 确认档的组合 Prefill 比本次 3080 单机 **+44.03%**。不宣称超过所有单机，也不宣称 C1–C6 服务收益。

**装不下也能跑。** 27B-LONG-01 是分层装：稠密模型按层分到两台设备，解决装不下。它不是微批重叠流水的证据。

### 27B-LONG-01 · Qwen3.8-27B · UD-IQ3_XXS · 模型分层装

加速卡 RTX 3060 12GB；模型 Qwen3.8-27B，权重量化 UD-IQ3_XXS；pp4096 / pp65536 / pp98304 / tg64，单位 tok/s；数据出自 [qwen27b-local-results.csv](data/qwen27b-local-results.csv)。3060 单卡装不下整个 27B，这正是做分层装的原因。这里是 llama-bench，不和 27B-PD-01 的服务端到端当同一口径。

| 指标 | 3060 单卡 | 395 单主机 | 3060 + 395 组合 | 比 395 |
| --- | --- | ---: | ---: | ---: |
| pp4096（tok/s） | 装不下 | 313.28 | **658.52** | **+110.2%** |
| pp65536（tok/s） | 装不下 | 136.69 | **319.10** | **+133.4%** |
| pp98304（tok/s） | 装不下 | 900 秒超时 | **225.10** | 超时 → 跑完（无 %） |
| tg64（tok/s） | 装不下 | 18.26 | **19.57** | **+7.2%** |

验证了什么：小卡装不下整模时，分层装把 395 单机跑不完的 98K 长 prompt 跑完了，4K 和 64K 的 Prefill 也比 395 单机快一倍以上。

**阶段分离 PD。** 9B-PD-01、27B-PD-01、ORNITH-PD-01、ORNITH-PD-02 都是阶段分离：两段先后跑。

### 9B-PD-01 · Ornith 9B · Q6_K · 独立 PD

加速卡 RTX 3060 12GB；模型 Ornith 9B，权重量化 Q6_K，服务状态，5064 输入 / 128 输出；数据出自 [benchmark-results.csv](data/benchmark-results.csv)。主机基线是 395 **服务态**（861.55 / 30.24），不是 9B-PIPE-01 用的 llama-bench 970.00 / 31.27。

| 指标 | 395 单主机（服务态） | 3060 Prefill + 395 Decode | 比 395 服务态 |
| --- | ---: | ---: | ---: |
| TTFT | 5.879 秒 | **3.496 秒** | **-40.5%** |
| Prefill（tok/s） | 861.55 | **1452.29** | **+68.6%** |
| Decode（tok/s） | 30.24 | **30.28** | **+0.1%** |

验证了什么：CUDA 侧算出的中间状态能整体交给 Vulkan 侧继续 Decode，首字更快，Decode 不掉。两段先后跑，属于阶段分离，和稠密加速是两条路。

### 27B-PD-01 · Qwen3.8-27B · Q4_K_M · 独立 PD（服务状态）

加速卡 RTX 3080 20GB；模型 Qwen3.8-27B，权重量化 Q4_K_M，KV 缓存 q4_0；C1–C6 六个并发档全部跑通，表里取 C1；数据出自 [qwen27b-local-results.csv](data/qwen27b-local-results.csv)。主机基线是 v1.0 口径下 395 服务态 C1（207.2 tok/s）。3080 llama-bench 裸算（pp1024 1228.53、pp4096 1203.06、tg64 33.08）与这份服务端到端是**不同口径**；组合 Prefill 1000.6 是该裸算上限的 82%，不是同一测法比赛。

| 指标 | 395 单主机（服务态，v1.0） | 3080 Prefill + 395 Decode，C1 | 比 395 服务态 |
| --- | ---: | ---: | ---: |
| TTFT | 4825 毫秒 | **1073 毫秒** | **-77.8%** |
| Prefill（tok/s） | 207.2 | **1000.6** | **+382.9%**（4.83×） |
| Decode（tok/s） | 36.33 | **38.75** | **+6.7%** |

验证了什么：服务状态下的阶段分离能跑，Prefill 不随并发下滑，Decode 也不掉。它验证的是阶段分离，不是同一阶段一起算。

### ORNITH-PD-01 · Ornith-1.5-35B-A3B · IQ4_XS · MoE 的 PD 压测

RTX 3080 20GB 包下全部 Prefill、AI Max+ 395 包下全部 Decode；主模型 Ornith-1.5-35B-A3B，权重量化 IQ4_XS；草稿头 Qwen3.6-35B-A3B-DFlash，权重量化 Q4_K_M；数据出自 [ornith35a3b-local-results.csv](data/ornith35a3b-local-results.csv)。没有匹配的单机速度对照——下面的提升是同一组合上 C1 → C6，不是组合对 395。

| 指标 | C1（1 路并发） | C6（6 路并发） | 变化 |
| --- | ---: | ---: | ---: |
| 3080 Prefill 聚合，输入 1000 token（tok/s） | 4017.46 | 3943.88 | **-1.8%** |
| 3080 Prefill 聚合，输入 100K（tok/s） | 2895.53 | 2793.24 | **-3.5%** |
| 395 Decode 聚合，输入 100K（tok/s） | 23.33 | 148.20 | 6.35× |

同一次压测跑了两份负载：1000 输入 / 128 输出，以及 100000 输入 / 128 输出；每份负载跑 1 到 6 路并发，1+2+3+4+5+6 = 21 个请求，两份合计 42/42 成功，`route=pd`、`n_reuse=0`。1000 输入那档当时没有单独记录 395 的 Decode 速率，所以表里只有 100K 档的 Decode。验证了什么：MoE 的 PD 在 100K 上下文、1 到 6 路并发下跑得稳，Prefill 算在 3080 头上、Decode 算在 395 头上，分得清。

### ORNITH-PD-02 · Ornith-1.5-35B-A3B · IQ4_XS · 融合 DFlash 草稿的 PD 配方

同一个模型、同一对硬件，在 ORNITH-PD-01 的独立 PD 之上加了两件事：395 那端的解码改走 DFlash 融合草稿的推测解码（`--spec-type draft-dflash`，草稿步长 6），六个槽位共用一个统一 KV 池（`--kv-unified`）。主模型 Ornith-1.5-35B-A3B，权重量化 IQ4_XS；草稿头 Qwen3.6-35B-A3B-DFlash，权重量化 Q4_K_M；KV 缓存 q4_0；负载 1000 输入 / 128 输出，单流；数据出自 [ornith35a3b-local-results.csv](data/ornith35a3b-local-results.csv)。没有匹配的单机速度对照。

| 配置 | Prefill（tok/s） | 单流 Decode（tok/s） | 草稿接受率 |
| --- | ---: | ---: | ---: |
| 融合配方，Prefill 批次 4096 | **4173.47** | **114.86** | 93.86% |
| 融合配方，批次加大到 8196、上下文 128K | **4123.15** | **114.42** | 未记录 |
| 同配方换一道题，草稿基本没被接受 | 3665.3 | 37.2 | 9.4% |

验证了什么：这套 3080 + 395 组合测得 Prefill **4173.47**、单流 Decode **114.86**。换一道题后 Decode 为 37.2，两种输入的速度相差 3.09 倍，说明成绩对输入和草稿接受率敏感；这不是同一道题开启与关闭草稿的加速倍率。主行草稿接受为 **107/114（93.86%）**。[详档](results/ornith-1.5-35b-a3b-fused-dflash-pd.zh-CN.md)

**容量与服务档位。** 27B-KV-01 是远端 KV 池 + 两条解码路线；FLASH-SPLIT-01 是单服务双设备切层。两者都没有匹配的最新单机速度对照，不能写组合对主机的收益。

### 27B-KV-01 · Qwen3.8-27B · Q4_K_M · 远端 KV 池 + 两条解码路线

加速卡 RTX 3080 20GB；模型 Qwen3.8-27B，权重量化 Q4_K_M，KV 缓存 q4_0；数据出自 [qwen27b-local-results.csv](data/qwen27b-local-results.csv)。

| 配置 | 3080 Prefill（tok/s） | 聚合 Decode · C1（tok/s） | 聚合 Decode · C6（tok/s） | C1 → C6 |
| --- | --- | ---: | ---: | ---: |
| 配置 C · 395 解码 | 1194.4–1210.6 | 33.55 | 63.84 | **+90.3%** |
| 配置 D · 3080 解码 | 1077–1090 | 63.2 | 116.3 | **+84.0%** |

每条流 1M 上下文。配置 C 由 395 解码，配置 D 由 3080 解码；两条路线里 395 都不参与 Prefill，模型权重的稠密计算始终在 3080。验证了什么：Prefill 吃紧就走 C、让 395 解码；解码总吞吐吃紧就走 D、让 3080 解码。这是容量和服务路线，和稠密加速分开归类。

### FLASH-SPLIT-01 · Qwen3.8-Flash · Q4 · 单服务双设备切层

一个 llama-server 同时使用 RTX 3080 20GB 和 AI Max+ 395；模型 Qwen3.8-Flash，权重量化 Q4，KV 缓存 q4_0；张量切分 0.38 / 0.62，ubatch 1024 / batch 4096，6 槽、每槽 131072 上下文；3080 显存峰值 19129 MiB；数据出自 [qwen38flash-q4-local-results.csv](data/qwen38flash-q4-local-results.csv)。无单机对照——提升是这一组合上 C1 → C4。

| 指标 | C1（tok/s） | C4 最好用的档（tok/s） | C1 → C4 |
| --- | ---: | ---: | ---: |
| Prefill | 569.892 | 633.685 | **+11.2%** |
| 聚合 Decode | 35.204 | 71.185 | **+102.2%** |
| 总吞吐 | 213.581 | 338.270 | **+58.4%** |

C1–C6 Prefill 聚合为 569.892–633.685 tok/s，聚合 Decode 为 35.204–71.185 tok/s。21/21 计分请求成功，负载约 2077 输入 / 256 输出。验证了什么：这套配置最好用的并发档是 C4，C5、C6 已经不再上涨；这是一个工作档位，换负载要重测。

**核查与外部参考。** 27B-DRAFT-AUDIT-01 是数据核查；EXT-DGX-01 是外部参考，只当背景。上面的本地 Spark Q4_K_M 行不和那些 NVFP4 社区数字合并。

### 27B-DRAFT-AUDIT-01 · Qwen3.8-27B · Q4_K_M · 投机解码核查

模型 Qwen3.8-27B，权重量化 Q4_K_M，KV 缓存 q4_0；在 AI Max+ 395 上做的数据核查；数据出自 [qwen27b-local-results.csv](data/qwen27b-local-results.csv)。

| 文本类型 | Decode（tok/s） | 接受率 |
| --- | --- | --- |
| 重复文本 | 35.0–38.5 | 100% |
| 自然语言，C1 | 12.1 | 17.7% |

验证了什么：重复文本与自然语言是两种负载，投机解码对输入内容敏感；重复文本上的高分不能代表真实文本。

### EXT-DGX-01 · Qwen3.5 9B / TQ3_4S；Qwen3.8-27B / NVFP4 · DGX Spark 外部参考

本条包含两个外部工作负载：Qwen3.5 9B，权重量化 TQ3_4S；Qwen3.8-27B，权重量化 NVFP4、KV 缓存 FP8。公开成绩约为 1000 tok/s Prefill、25–30 tok/s 单流 Decode、107 tok/s 聚合 Decode，C1–C6。这是别人机器上的公开数据，只当背景，不和本地数据排名，包括本地 27B-SPARK-01 的 Q4_K_M 三档。[详档](results/dgx-spark-community-control.zh-CN.md)

<details>
<summary>数字口径备忘：为什么同一台机器会有几个不同的数</summary>

- 395 在 9B 档有两个成绩：`llama-bench` 那组（970.00 / 31.27）对着 9B-PIPE-01 看，服务状态那组（861.55 / 30.24）对着 9B-PD-01 看。
- 27B 档 395 单机 Prefill 有三个数。**207.2** 是 v1.0 服务态的 C1 solo 成绩，那一版每处理一个 prompt 都要整拷一份 524 MiB 的递归态检查点，Vulkan 侧回读约 0.8 秒，这个数对着 27B-PD-01 的组合成绩 1000.6 看。**307.1** 是同一批机器关掉检查点后（v1.1 口径）的 C1 solo 成绩，记在 27B-KV-01 配置 C 的详档表里作历史对照。**313.28** 是 IQ3 量化下 `llama-bench` pp4096 的成绩，对着 27B-LONG-01 看。207.2 不是 1207.2 的笔误。
- 3080 侧 Prefill 链条：683.2（RPC 逐 ubatch 同步）→ 1000.6（纯 CUDA 直挂，+46%）→ 1210.6（关掉检查点，见 27B-KV-01 配置 C，是裸算上限 1228.53 的 98.5%）。27B-PD-01 的 1000.6 是裸算的 82%，服务态不随并发下滑、稳定在 1000–1015 tok/s；KV 搬运 68–76 毫秒（服务记录也写过 71 ms）。两侧一起涨之后，组合对 395 单机的优势仍在 4 倍上下。
- 27B 档在 395 主机线上，1200 以上的 Prefill 只出现在 RTX 3080 这一侧（裸算 1228.53、服务态 1194.4–1210.6）；395 单机跑 27B 稠密模型的 Prefill 在 200–320 之间，它跑 9B 也只有 861.55–970.00。27B-SPARK-01 的 1603.20 Prefill 是 3080 + Spark 的**组合成绩**，不是 3080 单机；这次 8K、无投机的 3080 单机 Prefill 是 1113.13。
- 配置 D 从 C1 到 C6 依次为 1090 / 1081 / 1080 / 1077 / 1082 / 1082 tok/s，同一轮的聚合口径从 1079 降到 1016 tok/s。配置 D 的显存与算力代价：草稿头权重占 1080 MiB、验证批还要约 500 MiB 计算缓冲，ctx8192 下 ubatch 只能从 1024 压到 512、slot 从 2 个减到 1 个，每步验证批的矩阵乘也实打实吃算力（4 token 批 +21 ms、8 token 批 +49 ms），Prefill 于是从 1210.6 掉到 1077–1090，低约一成。
- 配置 C 里 3080 还留 2 个 slot，395 带 DFlash 头单流 38.75 tok/s、持全量 KV 池。3080 带头单流实测自然语言 42.7 tok/s、代码 67.4 tok/s，是 395 带同一个头的 2.2–2.3 倍，聚合 Decode 因此从 63.84 抬到 116.3。
- 9B 流水四个检查点 Prefill 爬升为 **1865.08 → 1893.87 → 1999.51 → 2129.69 tok/s**；**37.16 tok/s** 的 Decode 只属于 1999.51 那个检查点，不是最终的 50.73。
- 实验表里的 v1.0/v1.1 口径是历史服务测法标签，不是当前公开里程碑编号；对应实验现在分别归入 v1.2/v1.3。
- ORNITH-PD-02 的 3080 显存占用 18661/20480 MiB，395 Vulkan 约 22678/65536 MiB。同轮另一次附加性能门禁只测到 3785/44，原因是那道题的草稿接受率低，不是参数改坏了。三端 HTTP 200、启动后无 OOM 无崩溃。融合配方的 4173.47 比 ORNITH-PD-01 的 C1 4017.46 高 3.9%。ORNITH-PD-01 那一档当时没有单独记录单流 Decode；它 100K 档写的 395 纯 Decode 是六路并发的合计（C1 23.33 到 C6 148.20），和单流速度不是一个口径。

</details>

## 版本时间线 v1.0 → v1.6

七个版本对应七次有数据的实验。27B-SPARK-01 是 v1.6 上 2026-09-11 的数据补录，不是第八个公开版本。

| 版本 | 实验 | 一句结论 |
| --- | --- | --- |
| v1.0 | 9B-PD-01 | 3060 做 Prefill、状态交给 395 做 Decode 能成。组合对 395 单机：首字 3.496 秒对 5.879 秒、Prefill 1452.29 对 861.55、Decode 30.28 对 30.24 |
| **v1.1** | 9B-PIPE-01、27B-LONG-01、EXT-DGX-01 | 3060 + 395 同时算同一阶段的 9B 稠密加速走通：组合 Prefill 四个检查点 1865.08 → 1893.87 → 1999.51 → 2129.69，最终 Decode 50.73（37.16 只属于 1999.51 那个检查点）。27B-LONG-01 分层装对 395 单机：pp4096 658.52 对 313.28、pp65536 319.10 对 136.69，pp98304 从 900 秒超时变成 225.10 |
| v1.2 | 27B-PD-01 | 3080 做 Prefill、395 做 Decode 在服务状态下能跑。3080 Prefill 从 683.2（RPC 逐 ubatch 同步）提到 1000.6（纯 CUDA 直挂），1 到 6 路并发稳定在 1000–1015；C1 TTFT 组合 1073 毫秒对 395 单机 4825 毫秒。207.2 是 v1.0 口径下 395 单机 Prefill 的实测 |
| v1.3 | 27B-KV-01、27B-DRAFT-AUDIT-01 | 关掉检查点拷贝后，3080 服务态 Prefill 1000.6 → 1210.6（裸算 1228.53 的 98.5%），395 单机 Prefill 207.2 → 307.1。远端 KV 池两条路线：配置 C 由 395 解码，3080 Prefill 1194.4–1210.6、聚合 Decode 1 路 33.55 → 6 路 63.84；配置 D 由 3080 解码，3080 Prefill 1077–1090、聚合 Decode 1 路 63.2 → 6 路 116.3。核查：395 上自然语言 C1 Decode 12.1、草稿接受率 17.7% |
| v1.4 | ORNITH-PD-01 | 3080 做全部 Prefill、395 做全部 Decode 的 MoE 压测，两份负载 × 六档并发 42/42 请求走通。输入 1000 token：3080 Prefill 聚合 1 路 4017.46 → 6 路 3943.88；输入 100K：2895.53 → 2793.24；395 Decode 聚合（100K 档）1 路 23.33 → 6 路 148.20 |
| v1.5 | FLASH-SPLIT-01 | 单服务切层（3080 占 0.38、395 占 0.62）21/21 计分请求成功。最好用的档是 4 路（C4）：Prefill 633.685、聚合 Decode 71.185、总吞吐 338.270；1 到 6 路 Prefill 569.892–633.685 |
| **v1.6** | ORNITH-PD-02 | 3080 Prefill 4173.47，395 带 DFlash 草稿单流 Decode 114.86，草稿接受 107/114（93.86%）。解码跟着接受率走：9.4% 时只有 3665.3 / 37.2 |
| 2026-09-11 补录（仍是 v1.6） | 27B-SPARK-01 | RTX 3080 20GB + DGX Spark GB10，Qwen3.8-27B Q4_K_M。Prefill 优先 8K/128 无投机：组合 1603.20 / 17.62 对 3080 单机 1113.13 / 33.36（**+44.03%**）。Decode 优先 DFlash2：2K 1153.95 / 63.97，8K 1124.32 / 49.20。均衡 DFlash2 C1–C6 126/126。不创造新版本。 |

版本号是实验里程碑，不是文档维护次数；与旧发布号的对应见 [VERSION_HISTORY.md](VERSION_HISTORY.md)。

27B-C 和 27B-D 是 27B-KV-01 一个实验的两种配置；395 上那组自然语言测试属于 27B-DRAFT-AUDIT-01；EXT-DGX-01 仍是别人机器上的公开 Spark 成绩，只当背景；27B-SPARK-01 是本地 3080 + Spark 的 Q4_K_M 实测。

## 怎么读这些数据

一个数字能写成什么结论，按下面几条定。

- **能跑**：请求完整跑完、状态交接成功、每个指标都能说清算在谁头上。27B-KV-01、ORNITH-PD-01、ORNITH-PD-02、FLASH-SPLIT-01、27B-SPARK-01 属于这一档，都已验证。
- **更快**：模型、量化、负载、指标全都一致，而且有单卡或单机对照。9B-PIPE-01、9B-PD-01、27B-LONG-01、27B-PD-01 满足这条，提升幅度都在上面的表里。27B-SPARK-01 只在最高 Prefill 确认档上，对本次 3080 单机的 Prefill 满足这条。
- **装得下、服务得住**：负载跑完、有显存和稳定性数据。27B-KV-01、FLASH-SPLIT-01 以及 27B-SPARK-01 均衡档的结论就是这样写的。
- **远端 KV 归容量路线**：395 只作远端 KV 池、不参与 Prefill 稠密计算，验证的是容量（解码归属见 27B-KV-01 的两条路线），和稠密加速分开归类。
- **投机解码的点测不和随机种子压测互换**：27B-DRAFT-AUDIT-01 只立规矩，不给提升。
- **外部参考只当背景，不同条件不拼横向排名**：EXT-DGX-01 是别人公开的实测，不当本地对照；模型、量化、引擎、prompt 或连接方式不同的数据不拼排名。版本号不是实验编号，也不当数据来源。每个数据都要能说清用的是 RTX 3060、RTX 3080、395，还是 3080 + Spark 组合。

## 后续规划

还没做完的对照。

| 阶段 | 当前结果引出的问题 | 计划做什么、做到什么算完成 |
| --- | --- | --- |
| **补齐匹配对照与其余 MoE 格** | D1 这一格已经在 9B + 3060 上做齐了单卡与单机两组对照；其余几格要按同一套负载标准把对照补齐，两格 MoE 实验还没有开始。 | 一、9B Q6_K 同样条件下补 RTX 3080 的复测；二、27B 与 MoE 各格补 3060、3080 的同条件单卡对照，27B-KV-01 补同一轮「3080 不接远端 KV」的对照；三、做 M1（MoE 轻松装）和 M3（MoE 装不下）两格的完整实验；四、把一对一的稠密加速搬到更多大内存主机和更多小显存显卡上。做完的标准：稠密区能对得上、Prefill 和 Decode 的收益不打折、调度稳定。 |
| **Spark 组合后续** | 3080 + Spark 上已测与待测要分开写。 | **已测：** 27B-SPARK-01 三档，对照本次 3080 单机微基准（Prefill 优先无投机；Decode 优先 DFlash2 的 2K 与 8K；均衡 DFlash2 C1–C6）。**尚未测试：** 同模型、同量化、同负载的 Spark 单机基线；3080 单机均衡 C1–C6；组合与对照统一投机参数；这条线上的更多模型。 |
| **一卡对多主机** | 一对一跑稳之后，一张加速卡能不能同时带多台大内存主机？ | 研究一对多的调度、资源隔离、公平分配、故障恢复和扩展上限。做完的标准：主机数量增加后收益还能复现，单台主机的性能下降在可接受范围内。 |

## 详档与数据

首页每个实验只摘最关键的几个数，完整数据行、指标定义和字段说明都在下面的详档和 CSV 里；自己算的时候用对应的 CSV；除非详档写明有可比的对照，不要把不同编号的实验数据合起来算。

| 编号 | 模型 · 权重量化 · 加速卡 | 要回答的问题 | 详档 | CSV |
| --- | --- | --- | --- | --- |
| 9B-PD-01 | Ornith 9B · Q6_K · RTX 3060 12GB | CUDA 做 Prefill，能不能把状态交给 Vulkan 去 Decode？ | [v1.0 独立 PD](results/v1.0-independent-pd.zh-CN.md) | [CSV](data/benchmark-results.csv) |
| 9B-PIPE-01 | Ornith 9B · Q6_K · RTX 3060 12GB | 两台设备能不能通过异步分层流水一起算同一个模型？ | [v2.4 融合分层流水](results/v2.4-fused-layer-pipeline.zh-CN.md) | [CSV](data/benchmark-results.csv) |
| 27B-LONG-01 · 27B-PD-01 · 27B-KV-01 · 27B-DRAFT-AUDIT-01 | Qwen3.8-27B · UD-IQ3_XXS 与 Q4_K_M · RTX 3060 / RTX 3080 | 27B 分层装、服务状态 PD、远端 KV 和投机解码核查 | [Qwen3.8-27B 双机 PD](results/qwen3.8-27b-dual-machine-pd.zh-CN.md) | [CSV](data/qwen27b-local-results.csv) |
| ORNITH-PD-01 | Ornith-1.5-35B-A3B · IQ4_XS · RTX 3080 20GB | MoE 能不能分清 Prefill / Decode 归属，并扛住 100K 的 C1–C6 压测？ | [Ornith 双机 PD](results/ornith-1.5-35b-a3b-dual-machine-pd.zh-CN.md) | [CSV](data/ornith35a3b-local-results.csv) |
| ORNITH-PD-02 | Ornith-1.5-35B-A3B · IQ4_XS · RTX 3080 20GB | 在独立 PD 之上加融合草稿和统一 KV 池，Prefill 和单流 Decode 能不能一起上台阶？ | [Ornith 融合草稿 PD](results/ornith-1.5-35b-a3b-fused-dflash-pd.zh-CN.md) | [CSV](data/ornith35a3b-local-results.csv) |
| FLASH-SPLIT-01 | Qwen3.8-Flash · Q4 · RTX 3080 20GB | 一个服务同时用 CUDA 和 Vulkan 切层，吞吐在哪一档到顶？ | [Qwen3.8-Flash Q4 切层](results/qwen3.8-flash-q4-layer-split.zh-CN.md) | [CSV](data/qwen38flash-q4-local-results.csv) |
| 27B-SPARK-01 | Qwen3.8-27B · Q4_K_M · RTX 3080 20GB + DGX Spark GB10 | 这套组合相对 3080 单机，Prefill、Decode、均衡服务档分别能站住什么？ | [Qwen3.8-27B Spark + 3080 三档](results/qwen3.8-27b-spark-3080-profiles.zh-CN.md) | [CSV](data/qwen27b-spark-3080-profiles.csv) |
| EXT-DGX-01 | Qwen3.5 9B · TQ3_4S 与 Qwen3.8-27B · NVFP4 · 外部 DGX Spark | DGX Spark 的公开成绩，只当背景 | [DGX Spark 社区对照](results/dgx-spark-community-control.zh-CN.md) | [CSV](data/dgx-spark-community-controls.csv) |

实验编号和旧标签的对应关系在 [data/experiment-index.csv](data/experiment-index.csv)；全部详档在 [results/](results/) 目录。[更新记录](CHANGELOG_ZH.md)记录每一版公开实验版本测了什么。版本对照：[VERSION_HISTORY.md](VERSION_HISTORY.md)。
