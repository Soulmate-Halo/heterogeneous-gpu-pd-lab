# 更新记录

[English](CHANGELOG.md)

公开版本就是下面七个实测实验里程碑。与旧发布号的对应见 [VERSION_HISTORY.md](VERSION_HISTORY.md)。

## v1.6

- 测了 ORNITH-PD-02：与 ORNITH-PD-01 同一对 Ornith-1.5-35B-A3B IQ4_XS 硬件，加上 DFlash 融合草稿推测解码（`--spec-type draft-dflash`，n_max 6）和统一 KV 池（`--kv-unified`）。
- 结果：1000 输入 / 128 输出单流 Prefill **4173.47 tok/s**、Decode **114.86 tok/s**、草稿接受 **107/114（93.86%）**；batch/ubatch 提到 8196、ctx 提到 131072 后为 Prefill **4123.15**、Decode **114.42**。三端 HTTP 200，无 OOM；3080 18661/20480 MiB，395 Vulkan 约 22678/65536 MiB。
- 结论：草稿接受率高时，Prefill 站上 4000 以上、单流 Decode 站上 114。草稿几乎不被接受的题掉到 3665.3 / 37.2（接受率 9.4%）；同轮另一次门禁只有 3785/44，原因相同。
- 限制：解码成绩跟着草稿接受率走；复测必须锁同一道题、固定输出长度和同一个随机种子。详档：[ornith-1.5-35b-a3b-fused-dflash-pd.zh-CN.md](results/ornith-1.5-35b-a3b-fused-dflash-pd.zh-CN.md)。

## v1.5

- 测了 FLASH-SPLIT-01：一个 llama-server 同时用 RTX 3080（CUDA0）和 AI Max+ 395（Vulkan1），Qwen3.8-Flash Q4，张量切分 0.38 / 0.62，ubatch 1024 / batch 4096，KV q4_0，6 槽、每槽 131072 上下文；3080 显存峰值 19129 MiB。
- 结果：约 2077 输入 / 256 输出，C1–C6 Prefill 聚合 569.892–633.685 tok/s（C4 峰值 **633.685**），聚合 Decode 35.204–71.185 tok/s（C4 峰值 **71.185**），总吞吐 C4 **338.270** tok/s；21/21 计分请求成功。C1→C4 Prefill +11.2%、聚合 Decode +102.2%、总吞吐 +58.4%。
- 结论：这份负载最好用的并发档是 C4，C5、C6 已经不再上涨。
- 限制：这是工作档位，不是相对单卡的匹配加速；换负载必须重测。详档：[qwen3.8-flash-q4-layer-split.zh-CN.md](results/qwen3.8-flash-q4-layer-split.zh-CN.md)。

## v1.4

- 测了 ORNITH-PD-01：RTX 3080 包全部 Prefill、AI Max+ 395 包全部 Decode，主模型 Ornith-1.5-35B-A3B IQ4_XS，草稿头 Qwen3.6-35B-A3B-DFlash Q4_K_M；42/42 请求成功，`route=pd`、`n_reuse=0`。
- 结果：输入 1000 / 输出 128：3080 Prefill 聚合从 1 路（C1）**4017.46** 到 6 路（C6）**3943.88**（−1.8%）；输入 100K：3080 Prefill 聚合 C1 **2895.53** 到 C6 **2793.24**（−3.5%）；输入 100K：395 Decode 聚合 C1 **23.33** 到 C6 **148.20**（6.35×）。
- 结论：MoE 的 PD 在 100K、1 到 6 路并发下跑得稳，Prefill 与 Decode 归属分得清。
- 限制：短任务档没有单独计时的 395 纯 Decode 速率；不公布由整档墙钟派生的聚合 Decode。详档：[ornith-1.5-35b-a3b-dual-machine-pd.zh-CN.md](results/ornith-1.5-35b-a3b-dual-machine-pd.zh-CN.md)。

## v1.3

- 测了 27B-KV-01 远端 KV 的两条解码路线，以及 27B-DRAFT-AUDIT-01；模型 Qwen3.8-27B Q4_K_M，KV q4_0。关掉每 prompt 整拷 524 MiB 递归态检查点后，3080 服务态 Prefill 从 **1000.6** 涨到 **1210.6 tok/s**（裸算 1228.53 的 98.5%），395 solo 对照从 **207.2** 涨到 **307.1**。
- 结果，配置 C（395 解码，3080 无头 Prefill）：Prefill **1194.4–1210.6**，聚合 Decode C1 **33.55** / C6 **63.84**（+90.3%）。结果，配置 D（3080 解码）：Prefill **1077–1090**（C1–C6：1090 / 1081 / 1080 / 1077 / 1082 / 1082；同轮聚合 1079 → 1016），聚合 Decode C1 **63.2** / C6 **116.3**（+84.0%）。3080 带头：自然语言 42.7 tok/s、代码 67.4 tok/s（是 395 带同一头的 2.2–2.3 倍）；395 带头单流 38.75 tok/s。草稿头权重 1080 MiB，验证批约 500 MiB；ubatch 1024 → 512，slot 2 → 1；验证矩阵乘 +21 ms / +49 ms。核查：重复文本 35.0–38.5 tok/s、接受率 100%；自然语言 C1 **12.1 tok/s**、接受率 17.7%（低 68.6%）。
- 结论：Prefill 吃紧走 C、让 395 解码；解码总吞吐吃紧走 D、让 3080 解码。395 不参与 Prefill 稠密计算。重复文本上的投机解码高分不能代表真实文本。
- 限制：D 的 Prefill 比 C 低约一成，因为解码占用 3080 的算力和显存；C 与 D 是同一实验的两条路线，不是两个实验。详档：[qwen3.8-27b-dual-machine-pd.zh-CN.md](results/qwen3.8-27b-dual-machine-pd.zh-CN.md)。

## v1.2

- 测了 27B-PD-01 服务态 PD：RTX 3080 20GB 做 Prefill、AI Max+ 395 做 Decode，Qwen3.8-27B Q4_K_M，C1–C6 六个并发档各跑 split 与 solo。
- 结果：C1、v1.0 口径 TTFT **1073 毫秒** / Prefill **1000.6** / Decode **38.75**，对照 395 单机 4825 毫秒 / 207.2 / 36.33（TTFT −77.8%，Prefill +382.9% / 4.83×，Decode +6.7%）。去掉 RPC 逐 ubatch 同步后 Prefill 从 **683.2** 涨到 **1000.6**（+46%），约为 3080 裸算 pp1024 **1228.53** 的 82%（pp4096 1203.06，tg64 33.08）。服务态 Prefill 不随并发下滑，稳定在 **1000–1015 tok/s**。KV 搬运 68–76 毫秒（服务记录也写过 71 ms）。
- 结论：服务状态下的阶段分离能跑，Prefill 不随并发下滑，Decode 不掉。207.2 是 v1.0 口径 395 solo C1 的实测，不是 1207.2 的笔误。
- 限制：验证的是阶段分离，不是同一阶段一起算。详档：[qwen3.8-27b-dual-machine-pd.zh-CN.md](results/qwen3.8-27b-dual-machine-pd.zh-CN.md)。

## v1.1

- 测了 9B 融合异步分层流水（四个检查点现归入本里程碑）以及 27B IQ3 分层长上下文 27B-LONG-01。9B 为 Ornith Q6_K，RTX 3060 12GB + AI Max+ 395；27B 为 Qwen3.8-27B UD-IQ3_XXS。
- 结果，9B-PIPE-01：组合 Prefill **2129.69** / Decode **50.73**，对照 3060 单卡 1589.00 / 43.87（+34.0% / +15.6%）和 395 `llama-bench` 970.00 / 31.27（+119.6% / +62.2%）。检查点 Prefill 爬升：**1865.08 → 1893.87 → 1999.51 → 2129.69**；37.16 tok/s 的 Decode 只属于 1999.51 那个检查点，不是最终的 50.73。结果，27B-LONG-01：组合 pp4096 **658.52**、pp65536 **319.10**、pp98304 **225.10**、tg64 **19.57**，对照 395 单机 313.28 / 136.69 / 900 秒超时 / 18.26（+110.2% / +133.4% / 超时 → 跑完 / +7.2%）。
- 结论：9B 稠密加速比在场最快的单卡还快；3060 装不下 27B 时，分层装能跑完 98K，4K 和 64K 的 Prefill 快一倍以上。
- 限制：9B 的提升换模型要按同样对照重测；27B-LONG-01 是 IQ3，不是 Q4。流水详档：[v2.4-fused-layer-pipeline.zh-CN.md](results/v2.4-fused-layer-pipeline.zh-CN.md)。长上下文详档：[qwen3.8-27b-dual-machine-pd.zh-CN.md](results/qwen3.8-27b-dual-machine-pd.zh-CN.md)。

## v1.0

- 测了 9B-PD-01 异构独立 PD：Ornith 9B Q6_K，RTX 3060 做 Prefill，把状态交给 AI Max+ 395 的 Vulkan Decode，5064 输入 / 128 输出，四个实测交接节点，对照单机 9B 边界。
- 结果：组合 TTFT **3.496 秒** / Prefill **1452.29** / Decode **30.28**，对照 395 服务态 5.879 秒 / 861.55 / 30.24（TTFT −40.5%，Prefill +68.6%，Decode +0.1%）。
- 结论：CUDA 侧状态能整体交接；首字更快；Decode 不掉。
- 限制：两段仍是先后跑，属于阶段分离，不是稠密加速。详档：[v1.0-independent-pd.zh-CN.md](results/v1.0-independent-pd.zh-CN.md)。
