# RDMA 冷专家层（跨机 Q2_0 专家计算）

给「CPU 跑不了 AVX-512 专家内核」的主机（如 RTX 3080 主机）用的跨机专家层：本机 GPU 跑稠密段 + 热专家缓存 + CUDA Graph，**每一个未被本机缓存命中的专家行**经单边 RDMA flag 环交给远端 worker（Spark / GB10）计算并取回。

## 架构

```
3080 主机 (strata generate --remote-rdma ...)        Spark (strata-cold-expert-worker)
──────────────────────────────                    ──────────────────────────────────
每层 pool 回调（在 CUDA Graph 捕获之外）:
  begin(): 收齐本层 miss 行 → 一个窗口            轮询本地 doorbell（acquire load）
    RDMA Write 请求 → worker inbox                  ↓ 收到新 seq
    RDMA Write seq  → doorbell（链式 WR，有序）     mmap/显存里的 experts.bin 按
  … 本机同时跑 CPU pool 余下行（通常 0 行）…         (layer*512+expert)*1,382,400 寻址
  finish():                                          ↓ quantize_q8_0_scaled + moe_grouped_s2
    RDMA Read 轮询结果槽 8B seq 头                   ↓
    seq 命中 → 一次 RDMA Read 取回全部行             ring_publish: memcpy 结果 + release 置 seq
    按路由序散回 out 行                              （worker CPU 全程不参与传输）
```

与单机 pinned 内存 flag 机制同构：worker 只轮询本地内存、置位即走；取数方单边拉取。每层每方向一次传输，无 RPC 往返。当前 begin→finish 串行（每层一个在飞窗口），等待上限为 `max(0, T_cold − T_local)`。

## Push 模式（默认开启，`--no-remote-rdma-push` 回退拉取）

拉取模式下 3080 host 每层要两次 verbs 调用（begin 的链式写 + finish 的轮询读），实测每层 0.394 ms。Push 模式把方向反转，**host 的 verbs 完全退出逐层回路**——这正是 Nico 单机方案里「GPU 直轮 pinned flag、CPU 不沾手」的跨机等价物，Spark 的 GPU 扮演 Nico 里 CPU 的角色：

```
3080（引擎）                                        Spark（worker）
每层 pool 回调只做一件事：                           混合循环：
  把 miss 条目写进 host 内存请求块                     ① 本地门铃有 legacy 窗（verify）→ 旧路径
  （PushReq，纯内存写，无系统调用）                    ② 单边 RDMA Read 轮询 3080 请求块
图内 doorbell_wait 自旋等 flag（原有机制不动）           ↓ 发现新 seq
                                                    RDMA Read 拉走激活（h_x_f）
                                                    ↓ GPU 算冷专家（同一 grouped 内核）
                                                    链式 RDMA Write：结果行 → y_miss
                                                    最后一节 WR 写 doorbell flag
                                                    （RC 有序：flag 到 ⇒ 全部行已到）
```

- 前提：token graph 必须捕获（`graph_hits` 需要 `--expert-profile`，静态驻留表）；未捕获时 push 模式拒绝启动（fail-closed）。`--expert-profile` 同时是 prefill 6× 提速的前提——没有它 decode/prefill 都不走 token graph。
- 绑定：引擎在 session 就绪后把请求块、h_x_f、y_miss、h_flag 四处内存注册到本端 PD，经 flag 环发一条 `RXE2` bind 记录；worker 识别后进入 push 循环。verify 窗口（`--spec`，多 token）仍走拉取路径，两种流量在同一条 QP 上互不干扰。
- **W1（bind v3，2026-10-03）**：decode（channel 0）的写回改为「连续块 + 设备端回散」。bind 记录新增 `yc_addr/yc_rkey/yc_rows`：worker 的链式写变成 `[u32 count][u32 pad][i32 rows[count]]` 头 + `count` 条 entry 序连续行（偏移 256）+ flag，任意 ne 只有两条 SGE（原来每行一条）；引擎图内 `doorbell_wait` 之后用 `scatter_contig_mapped` 内核把行放回路由位、未覆盖行写零（取代 host pool 往 y_miss memset 的零填充语义）。v2 对端无 yc 字段，worker 拒收——两端必须同批新二进制（同 v1→v2 的规矩）。y_miss 在 push 模式下不再上链路。
- 同步不变式：引擎发布 layer l 的请求只在 GPU 已 ring l 之后；worker 写完结果最后才写 flag；flag 推进后 GPU 才发布 layer l+1——请求块与激活永不存在读写竞态。worker 死亡由 host 侧每层 20 s ring 超时 fail-closed。
- 实测（2026-10-01，cache 8192 + profile，prompt 512 + decode 64）：decode 52.85 → 60.71 tok/s（基线无 profile 无 graph），prefill 9.15 → 56.07 tok/s；host pool 1.214 → 0.084 ms/token；worker push 窗 p50 约 75 µs。push/pull 的 final-r 对拍 rel_l2 = 0（5,890,300/5,890,300 逐比特一致）。
- 已知边界：`--spec`（MTP verify 窗）在跨机下拉取路径每层开销被窗内 token 摊薄不足，实测 decode 降到 14.7 tok/s——verify 窗的 push 化（多 token 一窗）是下一阶段工作，当前配置不要带 `--spec`。

## Push channel 1：prefill 多 token 合窗（`--prefill CHUNK`，CHUNK ≤ 2048）

decode 的 channel 0 是逐 token 窗口；prefill 若照此会把 512 token 的 prompt 打成 24576 个单 token 窗（实测仅 56 tok/s）。Channel 1 把粒度换成「每层每 chunk 一窗」：

```
引擎（3080，每层每 chunk 一次）：                worker（Spark）：
  路由后 host 分组：miss → 条目表（路由序，          轮询请求块 28B 头（seq 变化才读条目）
  row = token*k+j，Dm 行号 = slot_h[row]）          ↓
  命中行 → 一次 moe_grouped_s2 分组内核             单边 Read 整 chunk 激活（n_tok 行）
  （直写 Dm，GMAX=8 子分组，与 decode 同内核）       ↓ GPU 分组内核（与 decode 同内核）
  x 块 D2H → PushReq{channel:1} 纯内存写            结果行连续 RDMA Write → y_pf，最后写
  （零 verbs，seq release 序收尾）                  flag = 窗口 seq
  ↓ push_wait 自旋 flag（与本地命中计算重叠）
  scatter_rows_f32：y_pf 行 → Dm[分组行号] → combine
```

- 第二条 bind（magic `RXE3`）在 RXE2 之后送达，携带 prefill 的激活块（x2，T_max 行）与结果块（y2，T_max×k 行）的地址/rkey/容量；channel 字段选择使用哪一对区域，decode 行为逐比特不变。
- 容量上限 `kMaxPushTok=2048`、`kMaxPushEntries=20480`（结果块 210 MB mapped pinned——**3080 需要 memlock unlimited**，见 `/etc/security/limits.d/`）；`--prefill` 超过上限会被钳到 2048 并打印提示。
- 同步不变式与 channel 0 相同：下一层的发布会等本层 routing 的 host 同步排干 stream，worker 写完结果才写 flag；prefill 的 flag 值是窗口 seq（单调），与 decode 的 layer+1 互不冲突（阶段不交错）。
- 实测（2026-10-01，cache 8192+profile，40G 链路）：512 随机 prompt prefill 58.0 → **267.7 tok/s**；4096 真实文本 prompt **545.8 tok/s**（decode 62.3 不回退）；4K 随机 435。对照 Nico 官方 5070 基线（4K=1226）仍有 2.2 倍差距，残差分解与后续杠杆见 `evidence/prefill-channel1-20261001.json`。PD 分离（prefill 全给 Spark）经实测否决：Spark 单机同口径仅 130.8 tok/s。
- 注意：命中行的分组内核要求 cache 槽 blob 与 worker arena 同布局（Q2_0 canonical pack）；`--no-remote-rdma-push` 时 prefill 回退本地逐专家 f16 循环 + SSD 流式（慢但对）。

## Channel 1 瘦身线格式（bind v2，2026-10-01）

4K prefill 的墙钟里约 4.4 s 是在等 worker 的逐窗服务，而每窗传输量是大头（f32 结果行 89 MB + f32 激活
21 MB）。bind v2 把线格式改成引擎声明、worker 适配（`PushBind2.version=2` 带 `x2_dtype/y2_dtype/
x2_row_bytes/y2_row_bytes`；两端版本或 dtype 不一致一律拒绝握手，fail-closed）：

- **激活 q8_0**（`kPushXq8`，默认）：引擎本来就要为命中行的分组内核跑 `quantize_q8_0_scaled`，现在直接把
  这份 `[n_tok×80 个 36B 块][n_tok×80 个 fp32 scale]` 发布出去（21 MB → 6.6 MB/窗），worker 免去自己的
  重复量化、D2D 拷进 `d_q8/d_scales` 即算。
- **结果行 f16**（`kPushYf16`，默认）：worker 在 GPU 上把分组内核的 f32 输出转 f16 再 D2H 写回
  （89 → 44.5 MB/窗），引擎侧新增 `scatter_rows_f16` 在加载时转回 f32 写进 Dm。
- 回退开关：`STRATA_PF_X_F32=1` / `STRATA_PF_Y_F32=1`（A/B 与数值排查用，走同一 v2 管道）；
  全窗线量 220 MB → ~106 MB，同时 y2 映射 pinned 从 210 MB 降到 105 MB（14 G 内存主机的 memlock 压力减半）。
- 探针：`STRATA_PF_PROBE=1` 两端各自汇总——引擎打印窗数/线量/逐窗等待 mean/p50/p99；worker 打印
  每窗四段分解（窗间空闲 / 条目+激活读 / 计算 / 写回）的 mean/p50/p99。
- 路由 trace：`STRATA_PREFILL_TRACE=路径` 让 prefill 按 `--dump-routing` 的记录格式落盘每层每 token 的
  路由（prefill 批量路径原本不产生 trace），配 `tools/make_profile.py --no-base` 重生成 profile.bin。
  trace 可离线算命中率（top-8192 的 freq 占比），不用烧引擎跑；prefill+decode 双 trace 配比
  dec×16 是本工作负载的折中（prefill 命中 .814 / decode .864）。
- 实测（2026-10-01，4K 真实文本，cache 8192）：520.3 → **759.6**（仅瘦身）→ **965.7 tok/s**（瘦身 +
  dec×16 配比的 profile_v3，Nico 1226 的 79%）；decode 63.5（命中 0.863）。数值门：X q8 跨 GPU 逐比特
  一致（512 对拍 64/64）；Y f16 是 cache-parity 级舍入差（真实文本 64/64 一致；随机 prompt logits 平坦，
  会在低边际处翻牌，属同级既有现象）。worker 四段（Run E）：读 1.45 / 算 11.19 / 写 4.99 / 空闲 p50
  20.1 ms——引擎逐窗等待 17.6 ms 与 worker 忙时 17.6 ms 精确相等，等待 100% 是逐层串行暴露的服务时间。
- 顺带修复：decode 首 token 偶发 SIGILL（act_quant_q8_1 的 AVX-512 指令）——根因是借槽回填线程与
  派发的竞争（host_res 一次派发被读三遍，可被中途翻转造出无主行落 CPU 池）；现 kind[] 为每次派发的
  唯一归属快照，begin() 不再重读驻留表。

## Push channel 2：verify 窗（`--spec`，2026-10-01）

verify 窗（MTP 投机的多 token 验证窗）原本走拉取路径：每层一次 verbs 往返由 3080 的老 CPU 驱动，
实测开 `--spec` decode 崩到 ~15 tok/s。channel 2 把它改成与 channel 0/1 同构的 push：引擎
`arm_push_verify()` 在 `Verifier::init` 后把 verify 窗的 mapped 激活块（h_x_，T≤8 token）、结果块
（h_ymiss_，T×k 行）和一个专用门铃 flag（h_vflag_）注册给 worker（RXE4 bind，`PushBind3`）；
`begin()`（n_tok>1）只做纯内存发布（`PushReq.channel=2`，带 `x_off/y_off` 段偏移——分窗时两段各发
一次），`finish()` 自旋等 h_vflag_==seq，worker 单边读激活、GPU 算完按路由行 scatter 写回段内
（与 channel 0 同语义，**不是** channel 1 的连续 entry 序——dispatch 期望每行在它的路由位置上），
最后写 flag。引擎侧零 verbs；`--no-remote-rdma-push` 时 channel 2 不 arm，verify 回落拉取路径。
协议头 `PushReqHead` 因此扩到 36 B（+x_off/y_off，仍在同一 64 B cache line，两端同批编译，bind 版本
不匹配 fail-closed）。worker 服务实测 93~194 µs/窗（T=2、2~16 entries）。

**现状（2026-10-01 定论）：channel 2 落地、协议正确、无回归（无 --spec decode 65.8 ≥ 基线 63.5 的
-3%），`--spec` 数值正确（2/4/8 输出与无 spec 基线逐字一致）——但任何 T 都比无 spec 慢**：实测
spec 2 → 11.0、spec 4（T=6 峰值）→ 20.8、spec 8 → 5.3 tok/s。原因：remote 环节经 channel 2 已压到
~0.25 ms/层（93~194 µs/窗服务），瓶颈是 verify 窗本体——GPU 图执行 ≈ T×单 token decode 成本，再加
host 逐层同步 ~24.5 ms/窗；此结构下 MTP 数学上限约 54 tok/s（T=6、100% 接受），追不平基线 65.79，
**MTP 加速路线证伪，生产继续不开 --spec**。decode 优化回到：① worker GB10 内核吞吐（11.2 ms/步）
② 引擎命中 gemm 路径（MMQ 仅 HIP）③ 跨层流水。诊断开关：`STRATA_VERIFY_TRACE=1`（逐窗
window/outv）、`STRATA_VERIFY_NO_GPU_PLAN=1`（verify 窗全行走 remote）。
**教训**：cmake target 逻辑名是 `strata_cold_expert_worker`（下划线），产物文件名带连字符——
`--target` 传连字符会静默空转（exit 0、无输出、不重链），本轮两个修复因此一度"以为编了实际没编"。

运维固化（同批）：worker 的 arena/PLE 加载后 `posix_fadvise(DONTNEED)` 清页缓存——GB10 统一内存上
34+28.8 GiB 的读缓存会把随后的 pinned 大块分配压进 direct reclaim 并卡死；`start_worker.sh` 已内置
drop_caches（worker 每次引擎运行前重启 + 清缓存是固定流程）；worker listen 只等引擎 10 分钟，
不带 `--ple-io rdma` 的引擎启动必须等 worker READY（PLE 28.8G fread 在引擎连接后才开始）。



## 协议（小端，packed，见 `rdma_expert_tier.hpp`）

- 请求：`ReqHeader{u32 magic='RXE1', u32 version=1, i32 layer, i32 n_tok, i32 k, i32 n_entries}` + `f32 x[n_tok*2560]` + `i32 {row,expert}[n_entries]`（仅 miss 行，路由序）。
- 应答：`f32 rows[n_entries*2560]`，按请求条目序；router 权重由本机 `moe_combine` 应用，worker 不乘。
- seq 从 1 单调递增；结果槽 = `(seq-1) % slots`；槽头 8B 为 seq（release 序）。

## 启动顺序（fail-closed，先 worker 后引擎）

```bash
# Spark（先启动，加载 34G 专家 arena 后监听）：
strata-cold-expert-worker --pack /path/to/pack-q2_0 --device rocep1s0f1 --port 39580 \
    --slots 4 --slot-bytes 4194304
# 看到 "worker: READY rdma=true" 后，3080 端：
strata generate --pack /path/to/pack-q2_0 --mmap-experts \
    --remote-rdma <spark-ip> --remote-rdma-port 39580 --remote-rdma-device mlx5_0 \
    --remote-rdma-slots 4 --remote-rdma-slot-bytes 4194304 \
    --expert-cache <N> --expert-profile profile.bin ...（其余参数照旧）
```

- `--remote-rdma` 与 `--expert-cache-device1..3` 互斥（两者认领同一批 miss 行）；只支持 canonical pack（native IQ pack 拒绝）。
- 设置 `--remote-rdma` 后跳过 AVX-512 强制检查（CPU 池不再有活）；本机必须用 `--mmap-experts`（FileExpertSource 纯映射，不在 14G 内存机器上建 34G 常驻 arena）。
- 两端 slots/slot-bytes 必须一致（握手时校验，不一致直接失败）。slot_bytes 默认 4 MiB、协议下限约 1.4 MB（一个 8×10 全 miss 验证窗口的应答为 800 KiB，协议上限 128 条目）。
- RDMA 初始化失败 / 对端超时（默认每层 30s，`--remote-rdma-wait-ms`）一律报错退出，数据面绝不回退 TCP。

## 验收

```bash
# worker 单机数值自检（CUDA 分组内核 vs 量化对齐标量参考 q2_0_expert_gpu_ref，rel_l2 < 5e-3）：
strata-cold-expert-worker --selftest --pack /path/to/pack-q2_0          # → worker_selftest PASS
# 端到端探针（3080 侧发送端 + 真实 RDMA + worker + 数值校验）：
strata-cold-expert-worker --probe --pack /path/to/pack-q2_0 --peer <spark-ip> --windows 8   # → expert_probe PASS
```

标量参考 `include/strata/kernels/q2_0_scalar_ref.hpp`：`q2_0_expert_scalar_ref` 是 `s2_expert_scalar(quant_acts=false)` 的可移植转写；`q2_0_expert_gpu_ref` 再叠加与 `quantize_q8_0_scaled`/`quantize_q8_0` 逐比特同规则的两级激活量化（fp32 amax/127 半远离零 + fp16 尺度最近偶），正确内核应落在 fp32 求和顺序量级（~1e-4）。两端共用同一定义。

## PLE 表走 RDMA（--ple-io rdma，2026-10-01）

3080 只有 14G 内存，320,001,536 行 × 90 B（28.8 GiB）的 PLE 表只能 O_DIRECT 逐行读 SSD（decode 每 token
16 行、实测 p50 约 4 ms，4K prefill 合计约 1.5 s）。`--ple-io rdma` 把表换成 Spark 的常驻内存：

- worker 加 `--ple-gguf <Q2_0-00002.gguf>`：解析 GGUF 头取 `per_layer_token_embd.weight` 的行数与偏移，
  整表 fread 进 cudaHostAlloc 的 pinned 内存，`expose_region(REMOTE_READ)` 注册，随后把
  `RDMAExpertTier::PleInfo{magic 'PLER', addr, rkey, row_bytes=90, rows}` 发布到结果环 slot 0
  （在进服务循环之前）。Spark 侧注册 28.8 GiB 需要 memlock unlimited
  （/etc/security/limits.d/<spark-user>-memlock.conf，已配）。
- 引擎在 tier open 之后、`--ple-io rdma` 时 `fetch_ple_info()` 轮询 slot 0 拿到区域信息（worker 装表
  要几十秒，预算 300 s），然后 PleTable 以 `PleIo::Rdma` 打开：本地 GGUF 只解析头部做形状校验，
  行数据全部经 `RemoteStage::rdma_read_rows` 单边读（48 行一条 WR 链、尾 WR 才置 signaled，QP
  max_send_wr=64 内）。行缓存（RowCache）照常在最前面；字节与磁盘完全一致（数值对拍逐比特通过）。
- 实测（4K 真实文本）：PLE 段 1591 ms → 186 ms；decode 每 token PLE 相位 1.579 ms → 0.026 ms；
  行读 p50 11 µs。引擎不加 --ple-io rdma 时行为与之前完全一致（fail-closed：worker 没发表面区域
  信息则报错退出，不回退 SSD）。
