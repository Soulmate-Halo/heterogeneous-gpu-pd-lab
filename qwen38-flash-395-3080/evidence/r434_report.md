# r434：8060S 冷专家接入 prefill——问题清单、修复与最终结论

日期：2026-10-06 · 执行：主力模型（器灵两单均 540s 死亡后主线接手）

## 结论（先说结果）

1. 故障根因全部查清并修复，8060S 冷专家链路已能**稳定**跑通 1K/4K/8K（引擎全程存活、数值正确 17×23=391、库自检 rel_l2 1.979e-4/1.359e-7）。
2. 但实测速度**不达标**：hipcold 路径 4K≈107 t/s、8K≈197-208 t/s，对比基线（同冷态 4K≈117、8K≈213；热态 4K≈260、8K≈486），冷态慢约 10%、热态慢约 2.4 倍。
3. **已回滚基线**（原二进制 d02888e9 + STRATA_HIPCOLD=0），8095/8080/8081 正常，算式验收 391 通过。
4. 复开方式：wrapper 设 `STRATA_HIPCOLD=1` + `STRATA_HIPCOLD_LIB=/home/hfy/strata/r434/libstrata_hipcold.so`（可选 HIPCOLD_SLOTS=192、HIPCOLD_TRACE）。

## 修复的 5 个缺陷（按发现顺序）

| # | 缺陷 | 现象 | 修复 |
|---|------|------|------|
| 1 | 逐 blob hipHostRegister/Unregister：blob 2764816B 非 4KiB 页对齐，注册按页取整后与相邻 blob 及引擎自身 CUDA 注册的页范围互相踩踏 | amdgpu gfxhub page fault，1K 后引擎 exit 1（r432/r433） | 废弃引擎内存注册，改库自持暂存槽（r434b） |
| 2 | 桥接 out 扩容先 resize 后 cudaHostUnregister（对未注册的新指针），留下 sticky CUDA 错误 | 引擎 mmq quantize 捡到残留错误 fatal exit(1)——「1K 能过、4K 必死」真凶之一 | r434c：先 unregister 旧指针再 resize，并清残留错误 |
| 3 | hipcold_host_register 幂等只认指针不认长度：xq80 在 1K 注册 3.3MB，4K 直接复用，GPU 读超出旧长度 | sparse pipeline illegal memory access（4K 起必现） | r434g：同指针更大长度时先撤销再按新长度注册 |
| 4 | hipHostAlloc 大额 pinned 分配在内存压力下随机卡死（回收/紧致化，440MB~1.35GB 均复现，分块 88MB 也卡） | 启动/首请求挂死，引擎 60s 看门狗（issue#29）杀引擎 | r434o：改 malloc 驻留 + hipHostRegister（userptr 钉自有页，实测 620MiB=260ms 稳定） |
| 5 | sparse 每专家一次全设备同步 + 淘汰覆写前又一次全同步（满工作集每 map 都淘汰） | sync_ms=27s/请求，是慢 2.4 倍主因 | r434h/m：swork 64 槽轮转+每槽事件、stage 槽事件，统一排干只在 fetch_out |

## 最终架构

- 暂存槽：自有 malloc + hipHostRegister 零拷贝（默认 256 槽×2.76MB≈707MB，HIPCOLD_SLOTS 可调），LRU 复用，跨请求缓存。
- blob 用前 CPU memcpy 进槽（~0.2ms），GPU 零拷贝直读；淘汰只等被淘汰槽的 hipEvent。
- tidx/weights 工作区 64 槽轮转 + 每槽事件；fetch_out 统一排干后 D2H。
- 引擎 stderr 已修复落盘（server.json 加 "log" 键，原为 /dev/null——本次排查的关键前提）。

## APU（gfx1151）实测带宽/耗时表（供后续引用）

- 零拷贝直读注册过的 host：r428 测 224.7 GiB/s（hipHostAlloc 来源）
- hipHostRegister 620MiB userptr：260ms，稳定不卡
- hipHostAlloc ≥~400-900MB：内存压力下随机卡死（不可用于生产）
- hipMemcpyAsync H2D（源=引擎 arena userptr 未注册）：0.17 GiB/s（极慢，不可用）
- hipMemcpy H2D（源=pageable 或 HIP 注册内存）：0.5-2 GiB/s（慢）
- hipMemcpy D2H（设备→host）：78 GiB/s
- 生产中 GPU 每专家有效墙钟：~14ms（自测同内核 3.24ms/1000tok）

## 未达标原因与下一步

基线引擎把一层的多个专家攒成大批次一次 MMQ 发射（摊薄启动与读权重），而 hipcold 目前逐专家三次 kernel 发射 + 小批次 + APU 实际访问带宽远低于纸面，单位成本反超 PCIe 喂给 3080 的路径。要反超需要：
1. 按层批量化内核：一层一次 gu/quant/dot 发射（专家维进 grid.z + 每 block 一张 blob 指针表），摊薄到与基线同级；
2. 或先查清 userptr 页在 gfx1151 上的 GPU 缓存属性（自测 0.8GiB/s vs r428 hipHostAlloc 页 224GiB/s，怀疑 userptr 页是 uncached 访问）。

## 产物位置

- 库源码（最终版）：`tools_pd/strata_hipcold_lib_r434.cpp`（远端 /home/hfy/strata/r434/src/）
- 桥接（r433 + r434c/d/e 修复）：`tools_pd/hipcold_bridge_r433.cpp`
- 库二进制：/home/hfy/strata/r434/libstrata_hipcold.so（自检 PASS）
- 部署/门禁脚本：`tools_pd/r434_build.sh`、`tools_pd/r434_final.sh`（含自动回滚）
- 回滚备份：/home/hfy/strata/r434/backup/（二进制/桥接/wrapper/server.json）
- 日志：/home/hfy/strata/r434/logs/（engine_stderr.log 现为生产引擎 stderr 落盘处）
