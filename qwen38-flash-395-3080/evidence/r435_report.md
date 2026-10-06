# r435：8060S+CPU 并行 decode 冷专家——上线报告

日期：2026-10-06 · 执行：主力模型（调研单交付设计契约后死亡、写码单零产出死亡，主线按单次机会制接手完成全部实现/调优/门禁）

## 结论

1. **已上线**：8095 跑新二进制 + `STRATA_HIPDECODE=1 FRAC=128`，decode 冷专家由 8060S 与 CPU 池**并行**分摊（静态哈希份额 128/256）。算式验收 **17×23=391 通过**，三端口正常。
2. **速度**：热态门内同协议五连对位 **B 全部跑赢基线 A**（21.9/23.5/25.9/20.6/23.3 vs 19.5/17.3/17.3/19.4/22.4 t/s），中位 **23.3 vs 19.4 ≈ +20%**。FRAC=160（22.0）次之，FRAC=256 更差（CPU 池本身是并行算力，全给 hip 反而浪费）。
3. 引擎自身方差大（同协议基线 17~23 t/s 浮动），+20% 以门内对位为准。

## 架构

- 新组件 `HipExperts`（include/strata/core/hip_experts.hpp + src/core/hip_experts.cpp），形状镜像 PeerExperts：kind=3 标记 hip 条目，launch 在 CPU 池开工前发射、finish 在池收工后取回。
- 行语义与 CPU 池一致：每 entry 一行**未加权**输出（moe_combine 后加权）——库 sparse 传 weights=nullptr，激活按 entry 复制（先按 token 量化 5 行再 memcpy 扇出到 50 行，省 10 倍量化）。
- 复用 r434 库派生的 r435 版（/home/hfy/strata/r435/libstrata_hipdecode.so）：暂存槽零拷贝，页锁补集只登记元数据。
- `has()` = offsets 表存在性 + 专家号哈希 < FRAC；不在页锁补集的专家永远留 CPU（防 sparse 失败致命）。
- 惰性开库：启动时 arena 未就绪不判死，has() 每窗口补试直到 complement pinned。

## 三个决定性修复（缺一个都到不了 +20%）

| # | 问题 | 实测 | 修复 |
|---|------|------|------|
| 1 | hipDeviceSynchronize/EventSynchronize 在 CPU 全忙时被 sched_yield 退避饿死 | 同步 14.7ms/次、28s/请求 | 库 init 加 `hipSetDeviceFlags(hipDeviceScheduleBlockingSync)`（HIPCOLD_SYNC_MODE=block，默认开）→ sync_ms 28s→0.9s |
| 2 | 淘汰 memcpy（0.19ms×专家）串行在池线程关键路径 | stage_ms=4.8s/7 请求，decode 反慢 18% | 后台暂存线程池（HIPCOLD_STAGE_THREADS 默认 3）：map 只入队、worker 等槽事件+覆写、sparse 前等 state==0 |
| 3 | 激活按 entry 量化（10×冗余）+ prep 行数逐窗口变导致反复注销重注册 | 两处合计 ~0.6ms/层 | 按 token 量化+memcpy 扇出；prep 固定 128 行保 (指针,长度) 幂等 |

## 关键实测（本机 gfx1151）

- malloc+hipHostRegister userptr 默认标志 GPU 直读 **207.5 GiB/s**（r434 的「userptr uncached 0.8GiB/s」猜想**证伪**，kernel 3.24ms 是自身开销非带宽）
- decode 层窗口 hip 侧（5 专家×5tok）：合计 **0.245ms**（prep 0.002 / 发射 0.025 / fetch 0.213）vs CPU 池同层 ~0.35ms
- 冷专家工作集 ≫ 槽数：384~512 槽下淘汰率 ~60%，后台暂存是刚需

## 产物与开关

- 引擎二进制：/home/hfy/strata/strata-nvfp4/build/strata（新）；备份 r435/backup/（原二进制+原 4 源文件+原 wrapper）
- 库源码：tools_pd/strata_hipcold_lib_r435.cpp → 远端 r435/src/；编译 `hipcc -O2 --offload-arch=gfx1151 -fPIC -shared`
- wrapper 现值：STRATA_HIPDECODE=1 / LIB=r435/libstrata_hipdecode.so / FRAC=128 / HIPCOLD_SYNC_MODE=block / HIPCOLD_SLOTS=384
- 回退：`STRATA_HIPDECODE=0`（tier off 零开销，不必换二进制）；彻底回滚 = r435/backup 还原二进制+wrapper
- 脚本：tools_pd/r435_build.sh（备份+落码+编译）、r435_gate.sh（三臂门禁）、r435_debug.sh、r435_final.sh、_r435_ssh.py（paramiko 通道；sftp 不可用走 base64 分块，MSYS2_ARG_CONV_EXCL='*' 必加）
- 探针：tools_pd/r435_reg_bench.cpp（注册方式带宽）、r435_decode_bench.cpp（decode 形态延迟）

## 下一步（如继续）

1. FRAC 精细扫（140~150）与按层自适应份额；槽位 384 vs 512 与内存压力权衡（free 常 <1GiB）。
2. 库的 prep/fetch 合并进层内流水（一层一次事件而非全设备 sync）还有零头可榨。
3. 教训复用：hipcold prefill 链路（STRATA_HIPCOLD）如复开，同样吃 block 同步+后台暂存红利——r435 库已兼容 prefill 桥（ABI 未变）。
