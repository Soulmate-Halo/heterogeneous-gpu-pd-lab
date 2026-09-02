# <img src="assets/soulmate-spirit.png" alt="器灵 Soulmate 小灵魂" width="44" align="absmiddle"> 小显存加速卡 + 大显存低算力主机稠密加速 — 异构 GPU PD 实验室

[English](README.md)

本仓库记录异构 GPU Prefill/Decode（PD）与稠密加速实验：RTX 3060 / RTX 3080 加速头与 AMD Ryzen AI Max+ 395 / Radeon 8060S 协同工作。

当前版本：**v2.6 — 统一 27B 实验区 + router v1.1/v1.2 最新实测**。公开范围仅包含架构、实测数据、思维演进、结论与限制；不提供部署步骤、复现命令、补丁、端点或内部切层策略。

**什么是稠密加速。** 由一张加速卡 + 一台被加速机器组成。两者显存重叠的区域即为显存稠密区；落入该区域内的模型，其解码（decode）与预填充（prefill）都会得到显著加速。该加速是无损的，且各项指标均高于加速卡或被加速机器单独运行时的数值。

这为低算力大显存主机（如 DGX Spark、AI Max+ 395）与高算力小显存卡（如 RTX 3060、RTX 3080）提供了更多应用机会与场景。

![稠密加速结构：可重叠的显存单元分为存放模型的稀疏区与进行解码/预填充计算的稠密区](assets/dense-region-structure.png)

*结构：可重叠的显存单元分为**稀疏区**（存放模型）与**稠密区**（进行解码与预填充计算）；落入稠密区的模型，其 decode 与 prefill 均获显著、无损加速。*

## 阅读路径与本次变化

- 先读**架构与测试口径**，再按**研究演进 → 9B 实验链 → 27B 实验区 → 外部对照**阅读；每个本地实验都用“新实验 + 关键结果”标题开场。
- **9B 主线**保留 v1.0–v2.4 的同口径演进，最终达到 prefill 2129.69 tok/s、decode 50.73 tok/s。
- **27B 实验区**现已融合 3060 / IQ3 与 3080 / Q4 两条记录；融合只为编排清晰，不改变彼此独立的测量口径。
- 最新 3080 router v1.1 的 **1194.4–1210.6 tok/s 是 prefill**；单流 decode 35–38.5 tok/s 只属于重复文本、100% 草稿接受率。v1.2 两阶段抢占与自然文本审计另列，避免口径混写。

## 稠密加速架构

```mermaid
flowchart LR
    P[Prompt] --> Q[异步微批队列]
    subgraph D["稠密区域 — 并发有效计算窗口"]
        direction LR
        N[RTX 3060 / CUDA<br/>前段模型层] -->|当前微批| A[AI Max+ 395 / Vulkan<br/>后段模型层与状态持有]
    end
    Q --> N
    A --> O[Decode 与结果流]
    N -. 下一微批重叠执行 .-> A
```

最终设计依赖 llama.cpp 上游 [PR #18626](https://github.com/ggml-org/llama.cpp/pull/18626) 的 RPC events/async 能力。模型层常驻各自端点，微批依次穿过两个阶段，使两端能够并发工作；本实验不宣称拥有自研推理引擎补丁。

### 术语定义

- **稠密加速**是本项目对最终异步分层 PD 形态的正式命名：外接显卡与 AI Max+ 395 分别持有自己的模型层，RPC/events 让相邻微批持续占用两个阶段。
- **稠密区域**是异步微批流水时间轴上的重叠计算窗口：外接显卡与 AI Max+ 395 同时处于有效计算状态，分别处理相邻微批和各自层段。填满这段窗口可减少流水气泡，对其覆盖的模型层段进行强加速。
- 稠密区域描述的是调度重叠，**不代表**两卡重复计算同一层、进行张量并行或重复驻留权重。

## 研究演进

实验总目标：验证一张小显存、高算力的加速卡在无法独立装下完整模型时，能否与一台大显存、低算力的主机共同承载并计算同一模型，在不降低输出质量的前提下，同时加速 Prefill 与 Decode。

| 阶段 | 实验目的（为什么做） | 验证方法（怎么测） | 实测结果 |
| --- | --- | --- | --- |
| v1.0 | 建立基本可行性：RTX 3060 承担 Prefill，随后把状态交给 AI Max+ 395 完成 Decode。 | 稳定完成 CUDA prefill → Vulkan decode 的状态交接。 | 1452.29 tok/s；decode 30.28 tok/s |
| v2.1 | 解决 v1.0 中两端按请求串行、交替空闲的问题；验证同一请求的微批流水能否让两端同时参与一次 Prefill。 | 验证同一请求的微批流水让两个切层阶段在同一请求内同时保持激活。 | 1865.08 tok/s |
| v2.2 | 解决 v2.1 的重叠在轮次间波动的问题；验证提速可复现，而不是偶然。 | 调整异步重叠，直到流水在各轮次间保持一致。 | 1893.87 tok/s |
| v2.3 | 流水可复现后，验证两端负载平衡能否减少流水等待，并同时改善 Prefill 与 Decode。 | 平衡后的切层在提高 prefill 的同时把 decode 抬到 v1.0 之上。 | 1999.51 tok/s；decode 37.16 tok/s |
| **v2.4** | 完成实验总目标：用同一 9B Q6_K、pp5064 本地口径对照两端单机与组合结果，验证状态与输出正确、稠密区持续工作、Prefill 与 Decode 无损加速。 | 校准检查点把稠密区域填满，达到两端单卡 prefill 之和的 83.2%。 | **2129.69 tok/s；decode 50.73 tok/s** |
| **v2.5** | 9B 主线完成后，验证 3080 全量 prefill + 395 decode 能否把 27B Q4 带进服务态。 | 去掉逐 ubatch RPC 同步，完成 C1–C6 六档、两种路径共 12 组。 | C1 prefill **1000.6 tok/s**；TTFT 1073 ms |
| **v2.6** | 解决 3060/3080 两组 27B 数据割裂，并补入 1200+ 与单流 35 的真实口径。 | 发布 router v1.1/v1.2 全表，并用自然文本审计区分投机上限与业务速度。 | v1.1 prefill **1194.4–1210.6**；v1.2 C6 E2E 聚合 **45.7 tok/s** |
| **v3.0（研究方向）** | 把 v2.4 的一对一机制适配到其他各类大显存主机与小显存加速卡。 | 建立跨平台适配矩阵，验证不同主机架构与加速卡型号下的显存稠密区映射、prefill/decode 无损加速和调度稳定性。 | **计划：其他各类大显存主机 + 小显存加速卡的适配加速研究** |
| **v4.0（研究方向）** | 研究一张加速卡同时加速 X 台大显存主机。 | 研究一对多任务调度、资源隔离、公平性、故障恢复，以及并发主机数量扩大时的性能边界。 | **计划：1 张加速卡 → X 台大显存主机** |

## 9B 本地测试口径

| 项目 | 配置 |
| --- | --- |
| 模型 | 9B，Q6_K |
| 设备 | RTX 3060 12GB / CUDA + Ryzen AI Max+ 395 / Radeon 8060S / Vulkan |
| v1.0 输入 / 输出 | 5064 / 128 token |
| v1.0 测量 | 关闭缓存，保留第二次 server 请求结果 |
| v2 测量 | `pp5064`；仅在有记录的节点给出 `tg128` |

## 9B 本地实验链 — 同一主口径

### 新实验 9B-v1.0（独立 PD）：prefill 1452.29 tok/s，decode 30.28 tok/s

| 配置 | TTFT | Prefill | Decode |
| --- | ---: | ---: | ---: |
| 纯 AI Max+ 395 | 5.879 s | 861.55 tok/s | 30.24 tok/s |
| 前段 512 token 独立 PD | 5.720 s | 886.62 tok/s | 30.22 tok/s |
| 前段 2048 token 独立 PD | 5.077 s | 999.16 tok/s | 30.18 tok/s |
| 完整独立 PD | **3.496 s** | **1452.29 tok/s** | **30.28 tok/s** |

完整独立 PD 交接 5063 token、208.57 MiB 状态数据。它保留了多请求下 prefill/decode 双工的架构潜力，但本版本没有进行并发量化。

### 新实验 9B-v2.1（首个融合流水）：prefill 1865.08 tok/s

同一请求首次进入异步微批流水，两端不再按整请求交替空闲。

### 新实验 9B-v2.2（重叠细化）：prefill 1893.87 tok/s

跨轮次重叠稳定，确认提升不是偶然波动。

### 新实验 9B-v2.3（平衡细化）：prefill 1999.51 tok/s，decode 37.16 tok/s

两端负载平衡后，prefill 与 decode 同时越过上一检查点。

### 新实验 9B-v2.4（稠密加速定型）：prefill 2129.69 tok/s，decode 50.73 tok/s

#### 9B 同口径汇总

空白表示该检查点未记录该指标。

| 检查点 | 公开说明 | Prefill | Decode | TTFT |
| --- | --- | ---: | ---: | ---: |
| RTX 3060 基线 | 仅 CUDA 端 | 1589.00 tok/s | 43.87 tok/s | — |
| AI Max+ 395 基线 | 仅 Vulkan 端 | 970.00 tok/s | 31.27 tok/s | — |
| v2.1 | 首个融合流水 | 1865.08 tok/s | — | — |
| v2.2 | 重叠细化 | 1893.87 tok/s | — | — |
| v2.3 | 平衡细化 | 1999.51 tok/s | 37.16 tok/s | — |
| **v2.4** | 稠密加速最终检查点 | **2129.69 tok/s** | **50.73 tok/s** | — |

v2.4 比 RTX 3060 prefill 基线高 34.0%，比 AI Max+ 395 基线高 119.6%，达到两端单卡实测之和 2559 tok/s 的 83.2%。这些结果仅由本地表格计算，不外推到其他硬件。

融合 decode 的 37.16 tok/s 来自 v2.3 检查点；v2.4 检查点记录到 50.73 tok/s 的 decode。

## 27B 实验区 — 同区编排、口径独立

以下实验共享“27B”这一模型量级，但硬件、量化、prompt 和引擎版本不同，**只在各自表内比较**。

### 新实验 27B-A（3060·IQ3）：pp4096 658.52 tok/s（+110%）

本组探索数据与上方 `9B Q6_K / pp5064` 主表严格分开。模型为 Qwen3.8-27B UD-IQ3_XXS（27.32B 参数、10.17 GiB、3.06 bpw），因此不参与 v2.4 百分比计算。

| 测量项 | 纯 AI Max+ 395 | 稠密加速 | 实测变化 |
| --- | ---: | ---: | --- |
| pp4096 | 313.28 tok/s | **658.52 tok/s** | +110% |
| pp65536 | 136.69 tok/s | **319.10 tok/s** | +133%（2.33 倍） |
| pp98304 | 900 秒超时 | **225.10 tok/s** | 完成，TTFT ≈ 437 秒 |
| Decode tg64 | 18.26 tok/s | **19.57 tok/s** | 约 +7% |

本环境没有可用的 RTX 3060 单卡 27B 同口径基线。稠密加速能够完成，是因为外接显卡只需驻留并计算分配给它的层段；文档不公开内部切层比例。本组是 IQ3 验证，不能写成 Q4 结论。

### 新实验 27B-B（3080·Q4，服务态 v1.0）：C1 prefill 1000.6 tok/s，TTFT 1073 ms

本实验把加速头换为 RTX 3080，记录 27B Q4 双端异构 PD。它与 9B 主表、27B-A 的 IQ3 长 prompt 扫描严格分开。

#### 配置

- **加速头**：**RTX 3080 20GB**（纯 CUDA0 全量 prefill，无 RPC、无 draft 投机）。
- **被加速主机**：**AMD Ryzen AI Max+ 395 / Radeon 8060S**（Vulkan decode，持全量 KV 池，启用 DFlash2 投机解码）。
- **模型**：Qwen3.8-27B，Q4_K_M（约 17.66 GiB）。
- **引擎**：llama.cpp fork（HEAD `18c8dde`，含上游 [PR #18626](https://github.com/ggml-org/llama.cpp/pull/18626) 的 async/RPC 能力），运行于双实例：8082 纯 CUDA0 全量 prefill，8081 纯 395 decode 并持全量 KV 池（`/dev/shm/kvx`）。
- **3080 裸算基准**：pp1024 = 1228.53、pp4096 = 1203.06、tg64 = 33.08 tok/s。

#### v1.0 代表行

这里是 **C1–C6 六个并发档 × split/solo 两种路径 = 12 组**；README 只留首尾代表行，完整六档见详档与 CSV。

| C | TTFT split/solo | prefill split/solo | decode split/solo | KV 迁移税 |
| --- | ---: | ---: | ---: | ---: |
| 1 | **1073 / 4825 ms** | **1000.6 / 207.2 tok/s** | 38.75 / 36.33 tok/s | 71 ms |
| 6 | 5009 / 22303 ms | 1014.3 / 44.8 tok/s | 9.67 / 8.82 tok/s | 70 ms |

同口径去掉 RPC 后，C1 prefill 从 683.2 升至 **1000.6 tok/s（+46%）**，约为 3080 裸算的 82%。

### 新实验 27B-C（3080·Q4，router v1.1）：prefill 1194.4–1210.6 tok/s；重复文本单流解码 35–38.5 tok/s

- 约 1000 in / 128 out；8082 串行 prefill，因此单流值也是服务聚合值，C1–C6 仅波动 1.3%。
- C1→C6 decode 段聚合为 **33.55 / 45.57 / 52.31 / 51.30 / 61.64 / 63.84 tok/s**；端到端聚合为 16.8→32.0 tok/s。
- **1200+ 是 prefill，不是 decode。** 单流 35–38.5 来自重复/退化输出且草稿接受率为 100%，只能视为投机解码上限。

### 新实验 27B-D（3080·Q4，router v1.2）：C1 单流解码 63.2 tok/s；C6 端到端聚合 45.7 tok/s

- 3080 DFlash2 头采用 `np1 / ctx8192 / ub512 / n_max 3 / q4_0 KV / graphs off`；新 prefill 到达时，两阶段抢占先保存 KV，再交给 395 续算。
- 同一随机词 seed 内，C1 router/solo 单流为 **63.2 / 24.5**，C6 端到端聚合为 **45.7 / 23.3 tok/s**；六档两种路径共 12 组，连续 3 轮 `miss=0`、`err=0`。
- 单独自然文本审计显示 395 + DFlash2 C1 仅 **12.1 tok/s（17.7% 接受率）**，低于无头基线 18.26；因此重复文本 35–38.5 不代表真实业务单流。C5 自然文本档未测，未补值。
- 4k / 7k 单流首字为 3689 / 6524 ms，3080 decode 71 / 69 tok/s；加入 LRU KV 驱逐后，4k×6 与 7k×2 均无 OOM 或 restore failure。

完整 v1.0/v1.1/v1.2 表、自然文本审计与口径限制见 [27B 异构实验全集](results/qwen3.8-27b-dual-machine-pd.zh-CN.md)；机器可读数据见 [CSV](data/qwen27b-local-results.csv)。

#### 与 DGX Spark 的指标区别

| 指标 | AI Max+ 395 + RTX 3080（本实验） | DGX Spark / GB10 |
| --- | --- | --- |
| 模型/量化 | Qwen3.8-27B，Q4_K_M | Qwen3.8-27B，NVFP4 |
| 引擎 | llama.cpp（CUDA + Vulkan，fork `18c8dde`） | SGLang + DFlash2（1M 上下文） |
| KV 精度 | q4_0 | fp8_e4m3 |
| Prefill（冷） | ~1000–1015 tok/s（短 prompt 服务态） | 1170 / 800 / 615 tok/s（100K / 200K / 300K token 冷清空） |
| Prefill（裸算） | pp1024 = 1228.53 tok/s | 未公布 |
| Decode（C1） | 38.75 tok/s | 未公布 |
| TTFT（C1） | 1073 ms | 未公布 |
| 并发 | C1–C6 全档已测 | 未公布 |
| 拓扑 | 双机异构 PD（3080 prefill + 395 decode） | 单机整机 |

**不可直接排名**：量化（Q4_K_M vs NVFP4）、引擎（llama.cpp vs SGLang）、KV 精度（q4_0 vs fp8）、prompt 深度（短 prompt vs 10 万~30 万 token）、拓扑（双机异构 vs 单机）均不同，仅作同模型量级参照。

## 外部对照：DGX Spark 社区数据（非本地新实验）

以下数据保留社区原始公开口径，仅提供背景，不与本地实验排名。

| 设备 | 社区工作负载 | Prompt 口径 | Prefill | 可直接比较？ | 来源 |
| --- | --- | ---: | ---: | --- | --- |
| NVIDIA DGX Spark / GB10 | Qwen3.5 9B，TQ3_4S，分支专属 FP4 cache-on 路径 | pp2048 | 2766.28 tok/s | **否** — 量化、prompt、分支与内核不同 | [llama.cpp-tq3 PR #53](https://github.com/turbo-tan/llama.cpp-tq3/pull/53) |
| NVIDIA DGX Spark / GB10 | Qwen3.8-27B，NVFP4，SGLang + DFlash2 | 10万 / 20万 / 30万 token 冷 prefill | 1170 / 800 / 615 tok/s | **否** — 量化、引擎、prompt 深度与测试方法不同 | [hasso5703 实测](https://github.com/hasso5703/dgx-spark-qwen38/blob/17e7e2280e632b0a3ab91839c8c7522b256937ac/BENCHMARKS.md#L232-L243) |

详见 [来源说明](results/dgx-spark-community-control.zh-CN.md) 与 [机器可读外部对照](data/dgx-spark-community-controls.csv)。

## 结论与限制

- v1.0 证明了 CUDA prefill 与 Vulkan decode 间的异构状态交接。
- v2.4 稠密加速证明了异构切层流水能产生真实异步计算重叠，但同时占用两端会失去 v1.0 的独立 PD 双工能力。
- v1.0 来自 server 请求测量，v2 来自 llama-bench pp/tg；跨版本百分比仅为工程参考，不是严格同口径研究结论。
- 27B-A 证明 3060 / IQ3 长 prompt 可进入稠密区域；27B-B/C/D 进一步覆盖 3080 / Q4 的 C1–C6 服务态、router 聚合与两阶段抢占。它们属于不同口径，不能合并成同一排名。
- **1200+ 明确是 3080 prefill；不是 decode。** 35–38.5 tok/s 是重复文本、100% 接受率的投机解码上限；自然文本直测 C1 为 12.1 tok/s，文章同时保留两者以避免选择性展示。
- 现有 27B Q4 证据覆盖约 1000-token 并发压测与 4k/7k 长 prompt 点测；尚不能外推到 100K+ prompt、任意文本分布或永久保留一条 3080 快速 decode 流。
- 社区外部对照不做归一化或推算，完整保留原始公开口径。

详细记录：[v1.0 独立 PD](results/v1.0-independent-pd.zh-CN.md)、[v2.4 稠密加速演进](results/v2.4-fused-layer-pipeline.zh-CN.md)、[27B 异构实验全集](results/qwen3.8-27b-dual-machine-pd.zh-CN.md)、[9B 本地 CSV](data/benchmark-results.csv)、[27B 本地 CSV](data/qwen27b-local-results.csv) 与 [更新记录](CHANGELOG_ZH.md)。
