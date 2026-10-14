# DeepSeek-V4.1-Flash PD 分离部署全记录

> 起点：镜像包下载完成，Ubuntu 服务器尚未安装 Docker
> 终点：交付内网可用 URL 三要素，冒烟测试全绿
> 记录时间：2026-10-14

---

## 0. 交付结果（TL;DR）

### 0.1 URL 三要素

| 要素 | 值 |
|---|---|
| 协议 scheme | `http` |
| 主机 host | `<P_IP>`（<SPARKx_ROCE_IP> 网段内用 `<P_ROCE_IP>`） |
| 端口 port | `5701` |

```
http://<P_IP>:5701
```

OpenAI 兼容配置：

```
base_url : http://<P_IP>:5701/v1
api_key  : none / sk-任意      （代理不校验）
model    : deepseek-v4.1-flash
```

### 0.2 冒烟结果

* 客户端（从 <OFFICE_NET> 发起）：**PASS 10 / FAIL 0**
* 服务端（P 本机自检）：**PASS 6 / FAIL 0**
* 8/8 engram 分片服务器 HELLO 正常；4/4 台 Spark 的 D40 解码容器 healthy
* 一旦发起真实请求，P 的 CX3 RDMA 发送计数增长 **701 MB** → KV 确实走 RDMA，未退化为 TCP

### 0.3 性能

| 指标 | 修复前 | 修复后 |
|---|---|---|
| P 侧 prefill 边际速率 | 242 µs/token（4,105 tok/s） | **98.1 µs/token（10,195 tok/s）** |
| 88K prompt 单请求 | 21.39 s | **8.62 s** |
| 代理全链路 prefill 聚合（C1/C2/C4/C8） | 4054 / 4019 / 4038 / 3991 tok/s | **10082 / 10308 / 10372 / 10340 tok/s** |
| 端到端 e2e（C8，100K 入 / 500 出） | 19.08 tok/s | **44.72 tok/s** |
| 短上下文 decode 聚合（C1/C4/C8） | 43.6 / 116.6 / 254.0 tok/s | 81.7 / 186.7 / **260.5 tok/s** |

---

## 1. 最终架构

```
                 ┌──────────────────────────────────────────────┐
   客户端 ──────► │  http://<P_IP>:5701                     │
                 │  xy-pd-proxy-v8-5701 (vllm-dsv41-x86:overlay6)│
                 │  PD 代理：拆请求、管 room、转发 KV 元数据      │
                 └───────┬──────────────────────────┬───────────┘
                         │ POST max_tokens=1        │ 转发原始请求
                         ▼                          ▼
        ┌────────────────────────────┐   ┌───────────────────────────────┐
        │ P 侧 prefill（1×RTX 6000D）│   │ D 侧 decode（4×DGX Spark）    │
        │ <P_IP> / <P_ROCE_IP>  │   │ <SPARK0-3_ROCE_IP>  SGLang D40      │
        │ vLLM 21/40 层，agent 8000  │   │ TP4，API 8200（rank0）         │
        │ NixlConnector kv_producer  │   │ NIXL 侧信道 5600 ← RDMA 拉 KV  │
        └──────────┬─────────────────┘   └───────────────▲───────────────┘
                   │ engram 行（TCP, 层=host:port@lo:hi） │
                   ▼                                     │
        ┌──────────────────────────────────────────────────────────────┐
        │ 8 个 engram 分片服务器（每台 Spark 2 个）                      │
        │   :9611 = layer 14 的 1/4 行区间   :9612 = layer 1 的 1/4      │
        │   --backend disk --lib libxyrow.so --io-threads 128           │
        │   数据 = xy-engram-slice-v1 切片（25.34 GB/片，原生 ext4）      │
        └──────────────────────────────────────────────────────────────┘
```

### 1.1 角色与端口

| 角色 | 位置 | 容器/镜像 | 端口 | 重启策略 |
|---|---|---|---|---|
| PD 代理 | P | `xy-pd-proxy-v8-5701` / `vllm-dsv41-x86:overlay6` | 0.0.0.0:**5701** | unless-stopped |
| P prefill | P | `dsv41-p21-solo` / `dsv41-p5:vllm-exl3-sm120-rdma` | 0.0.0.0:**8000** | ⚠️ `no` |
| P NIXL 侧信道 | P | 同上 | <P_ROCE_IP>:**5600** | — |
| D decode | 4×Spark | `dsv41-d40-sgl` / `dsv41-sglang:mhcbf16parts` | <SPARK0_ROCE_IP>:**8200** | ⚠️ `no` |
| engram l14 | 4×Spark | `dsv41-engram-l14` / `dsv41-sglang:mhcbf16parts` | <SPARKx_ROCE_IP>:**9611** | unless-stopped |
| engram l1 | 4×Spark | `dsv41-engram-l1` / 同上 | <SPARKx_ROCE_IP>:**9612** | unless-stopped |

> 只有 **5701** 需要对外暴露；8000 / 8200 / 5600 是内部端口。

### 1.2 硬件与网络

| 节点 | 硬件 | 系统 | 地址 |
|---|---|---|---|
| P（prefill） | ASUS ProArt X870E-CREATOR WIFI，1×RTX 6000D 85651 MiB，32 核，59 GiB RAM，1×ZHITAI TiPlus9100 1 TB | Ubuntu，kernel 7.0.0-38-generic，驱动 610.57.04 | `enp12s0 <P_IP>/24`（办公网）、`enp3s0 <P_ROCE_IP>/24`（RoCE 网，CX3 mlx4_0 40 Gb/s） |
| spark-0 | DGX Spark，121 GiB 统一内存，ConnectX-7 200G | aarch64，kernel 7.0.0-1019-nvidia，驱动 580.178.04 | `<SPARK0_ROCE_IP>`（<SPARK_OFFICE_IP>） |
| spark-3 | 同上 | 同上 | `<SPARK1_ROCE_IP>`（.112） |
| spark-2 | 同上 | 同上 | `<SPARK2_ROCE_IP>`（.223） |
| spark-1 | 同上 | 同上 | `<SPARK3_ROCE_IP>`（.69） |

### 1.3 模型

DeepSeek-V4.1-Flash：40 层，hidden 5120，MoE 384 路由专家 + 1 共享专家 top-6（专家 FP4/MXFP4，dense FP8 block 32×32 ue8m0），MLA 64 head / 1 KV head / head_dim 512。
* 跨层 KV 共享：`kv_source_layer_ids=[2,8,14,20]` → 层 0-20 已包含全部 KV 源 → 可切 21/40 做 P
* Engram：`engram_layer_ids=[1,14]`，行数 384,006,168 / 384,016,682，**24 个 hash head**，行 = 256 B fp8 + 8 B ue8m0 scale = 264 B，单层 94.42 GiB
* 投机解码 DSpark：`num_nextn_predict_layers=3`，`dspark_target_layer_ids=[37,38,39]`

---

## 2. 起点：镜像包加载

### 2.1 包内 4 个镜像 tar

| tar | 大小 | 加载后 tag |
|---|---|---|
| `p-dsv41-p5_vllm-exl3-sm120.tar` | 14.67 GB | `dsv41-p5:vllm-exl3-sm120` |
| `d40-dsv41-sglang_mhcbf16parts-pkg-host.tar` | 33.58 GB | `dsv41-sglang:mhcbf16parts` |
| `proxy-vllm-dsv41-x86_overlay6.tar` | 13.47 GB | `vllm-dsv41-x86:overlay6` |
| `engram-vllm-dsv41_overlay6-pkg-host.tar` | 23.60 GB | `vllm-dsv41:overlay6` |

### 2.2 问题 1：download.docker.com / registry-1.docker.io 被墙

* **现象**：`curl https://download.docker.com/...` 连接被 reset；`docker pull` 拉 `registry-1.docker.io` 超时。
* **解决**：apt 源换阿里云 `mirrors.aliyun.com/docker-ce`（该源有 Ubuntu `resolute` 版本，需确认 codename），registry mirror 配 `docker.m.daocloud.io`。
* `dsv41-p5:vllm-exl3-sm120-rdma` 这个 tag 是本机在 `dsv41-p5:vllm-exl3-sm120` 基础上加了 RDMA 补丁后自己打的（见问题 4）。

### 2.3 问题 2：4 个 tar 都是 OCI archive，`RepoTags: null`

* **现象**：`docker load -i xxx.tar` 成功，但 `docker images` 里出现 `<none>:<none>`，没有 tag。
* **根因**：包里的 tar 是 OCI image layout 打的，manifest 里没有 `RepoTags`。
* **解决**：用 `docker inspect <none 镜像>` 读 config digest，和包 README 里记录的 digest 逐个比对确认哪个是哪个，再 `docker tag`。
  记录：`d699e0825ad7…`、`e1a765d6e036…` 两组 digest 与 README 一致。
* **结果**：最终镜像表（P 上）

```
dsv41-p5:vllm-exl3-sm120-rdma   21.6GB  f49ad7734f27    ← P 用这个
vllm-dsv41-x86:overlay6         19.8GB  8d1f7be0cd33    ← 代理用这个
dsv41-p5:vllm-exl3-sm120        21.6GB  2fc52dc6227a    ← 补丁前
dsv41-sglang:mhcbf16parts       67.7GB  36793ef25b3b    ← D40 + engram 服务用
vllm-dsv41:overlay6             23.6GB  dfa280f918a1
```

### 2.4 问题 3：`unable to mmap 101535150936 bytes ... Cannot allocate memory (12)`

* **现象**：P 启动 vLLM 时崩在加载 engram 分片（映射 101.5 GB 文件）——而 P 只有 59 GiB 内存。
* **根因**：`vm.overcommit_memory=0`（默认启发式）拒绝这种远超物理内存的 mmap。
* **解决**：`sysctl -w vm.overcommit_memory=1`；原值 `0` 备份在 `/root/.dsv41-overcommit-orig`。
* **注意**：这是**运行时设置，重启会丢**。本次会话末尾已持久化：`/etc/sysctl.d/90-dsv41-overcommit.conf`（内容 `vm.overcommit_memory = 1`）。
* 官方启动脚本同样在 preflight 里硬校验这一项。

---

## 3. P 侧（vLLM prefill，RTX 6000D + ConnectX-3）

### 3.1 问题 4：`NIXL_ERR_BACKEND` / `no active messages transport`

* **现象**：P 侧 NIXL 初始化失败，日志 `no active messages transport`、`NIXL_ERR_BACKEND`；UCX 报找不到可用设备。
* **定位**：镜像里 vendored 的 UCX 自带一份 `libibverbs-1de1263b.so.1.14.48.0`（manylinux/RHEL8 构建），它枚举到的设备数是 **0**；而系统 `/usr/lib/x86_64-linux-gnu/libibverbs.so.1.14.50.0` 能正常枚举出 `mlx4_0`。
* **根因**：镜像自带 libibverbs 与宿主的 mlx4 驱动/mlnx 用户态不兼容 → 看不见 ConnectX-3。
* **解决**：把系统 libibverbs 覆盖到镜像里 **两份** vendored 副本上（`nixl_cu12.libs` 和 `nixl_cu13.libs` 各一份），重建镜像 → `dsv41-p5:vllm-exl3-sm120-rdma` (`f49ad7734f27`)。
* **配套**：容器需 `--device /dev/infiniband`、`--cap-add CAP_IPC_LOCK`、`--ulimit memlock=-1:-1`。

### 3.2 问题 5：`registerMem: registration failed`

* **现象**：libibverbs 修好后，NIXL 注册显存失败。
* **根因**：`UCX_TLS` 里只有 `ib,sm,self`，没有 CUDA 相关传输，无法注册 device memory。
* **解决**：`UCX_TLS=ib,cuda_copy,cuda_ipc,sm,self`；同时 `UCX_NET_DEVICES=mlx4_0:1`。

### 3.3 问题 6：代理返回 HTTP 400 `prompt_too_long`

* **现象**：100K token 的大 prompt 被代理直接拒绝。
* **根因**：代理的 `XY_MAX_PROMPT_CHARS` 默认只有 `12288`；8080 那个旧实例有设成 2000000，5701 这个没设。
* **解决**：`/home/p-user/dsv41-launch-proxy5701.sh` 里加 `-e XY_MAX_PROMPT_CHARS=2000000`。

### 3.4 P 的启动参数（最终）

```
镜像   : dsv41-p5:vllm-exl3-sm120-rdma
patch  : /mnt/dsv41-tp6/dsv41-pp3/patch-p21-exl3/*.py  （35 个 bind mount，覆盖镜像内 vllm 源码）
权重   : /mnt/dsv41-tp6/fp4h96 → /models/DeepSeek-V4.1-Flash (232 GB)
启动   : /home/p-user/dsv41-p21-handshake.sh
vllm   : serve /models/DeepSeek-V4.1-Flash --served-model-name deepseek-v4.1-flash
         --tensor-parallel-size 1 --pipeline-parallel-size 1
         --gpu-memory-utilization 0.92 --kv-cache-memory-bytes 10737418240
         --max-model-len 131072 --max-num-seqs 12 --max-num-batched-tokens 16384
         --engram-config '{"cpu_offload": false}'
         --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
         --kv-cache-dtype auto --block-size 128 --enable-cumem-allocator
         --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'
关键env: UCX_NET_DEVICES=mlx4_0:1  UCX_IB_GID_INDEX=12  UCX_TLS=ib,cuda_copy,cuda_ipc,sm,self
         VLLM_NIXL_SIDE_CHANNEL_HOST=<P_ROCE_IP>  VLLM_NIXL_SIDE_CHANNEL_PORT=5600
         XY_NIXL_LAYER_MODE=1  XY_SWA_EXTRA_RETAINED=2816  XY_KVME_LAYERS=21
```

> P 是 **21/40 层的 prefill-only 切片**，单独跑输出的文本是乱的，这是**设计如此**，必须配合 D + 代理。

---

## 4. D 侧（SGLang decode，4×DGX Spark）

### 4.1 问题 7：`AttributeError: 'ServerArgs' object has no attribute 'enable_prefill_context_parallel'`

* **现象**：D 侧起不来，报这个属性不存在。
* **根因**：Spark 上原有的 `dsv41-4x-spark:canary-roce` 是本地自建的 `local/sglang:dev`（COMMIT 未知），它的 arg-group 框架和 overlay 补丁要用的版本不一致。
* **解决**：改用官方 tar `d40-dsv41-sglang_mhcbf16parts-pkg-host.tar`（33.6 GB，`lmsysorg/sglang:dev-dsv41`，COMMIT `da64c5cb…`）推到 4 台 Spark 加载。

### 4.2 问题 8：`missing /models/DeepSeek-V4.1-Flash/config.json`

* **现象**：worker 起来后找不到权重。
* **根因**：D40 启动脚本挂的是一个**空的 docker volume** `dsv41-weights`，不是宿主目录。
* **解决**：改成 bind mount `-v /home/spark-user/models/DeepSeek-V4.1-Flash:/models/DeepSeek-V4.1-Flash`，并把 `/engram` 映射到 `/home/spark-user/dsv41-engram`。

### 4.3 问题 9：D 侧 `NIXL_ERR_BACKEND`

* **现象**：D 侧 NIXL 初始化失败。
* **根因**：D40 的启动 env 里**根本没设 `UCX_NET_DEVICES`**（P 侧设了，D 侧漏了）。
* **解决**：加 `UCX_NET_DEVICES=rocep1s0f0:1` 与 `UCX_TLS=ib,cuda_copy,cuda_ipc,sm,self`，并补 `NCCL_IB_HCA=rocep1s0f0:1,roceP2p1s0f0:1`、`NCCL_SOCKET_IFNAME=enp1s0f0np0`、`NCCL_IB_ADDR_RANGE=<ROCE_SUBNET>`。

---

## 5. RDMA 打通（用户明确要求：官方不支持 CX3↔Spark，要用镜像内补丁打通）

### 5.1 问题 10：`NIXL_ERR_REMOTE_DISCONNECT`

* **现象**：rank3 真实报错，rank0-2 是 gloo 级联失败；P 侧日志里 UD 传输 `resending ... acked_psn 0` 无限循环；P 的 RoCE 计数器几乎不动 → 实际没通。
* **定位过程**：写了一个**最小 NIXL 探针**（只做 `nixl_agent` + 注册一块 4 MB 显存 + 一次 `initialize_xfer/transfer/check_xfer_state`），在 P 与各 Spark 之间跑 GID 组合矩阵：

| 组合 | GID | 结果 |
|---|---|---|
| RoCE v2 + IPv4-mapped GID | P gid[13] ↔ D gid[3] | ❌ FAIL |
| **RoCE v1 + IPv4 GID** | **P gid[12] ↔ D gid[2]** | ✅ **OK，4 MB / 10 ms** |
| RoCE v2 + ULA IPv6 GID | P gid[5] ↔ D gid[9] | ✅ OK |

* **根因**：P 的 mlx4（CX3，老驱动）与 Spark 的 mlx5（CX7）在 **UD 队列对 wireup** 阶段对 IPv4-mapped 的 RoCE v2 GID（`::ffff:<SPARKx_ROCE_IP>`）处理不一致 → CM/UD 建链失败。
* **解决**：全链路切 **RoCE v1 IPv4**：P `UCX_IB_GID_INDEX=12`、D `UCX_IB_GID_INDEX=2`。
* **GID 对照（供后续排查）**：

```
P  mlx4_0    gid[12]  ::ffff:<P_ROCE_IP>  IB/RoCE v1   ← 在用
             gid[13]  ::ffff:<P_ROCE_IP>  RoCE v2
Spark rocep1s0f0 gid[2] ::ffff:<SPARKx_ROCE_IP> IB/RoCE v1  ← 在用
                gid[3] ::ffff:<SPARKx_ROCE_IP> RoCE v2
```

* **验证**：修好后 P 的 `mlx4_0 port_xmit_data` 在真实请求期间增长 **701 MB**，且 `docker logs` 出现 `transfer complete`。

### 5.2 附带的坑

* 走 RDMA 需要 `--device /dev/infiniband:/dev/infiniband`、`--cap-add CAP_IPC_LOCK`、`--ulimit memlock=-1:-1`，缺一不可。
* P 侧 `NCCL_IB_DISABLE=1`（本机 NCCL 走 RoCE 不稳），KV 传输完全交给 NIXL/UCX，两条路互不干扰。

---

## 6. PD 链路与代理行为

### 6.1 代理的真实行为（读源码 + 实测）

* 路由**只**有：`GET /health|/live|/stats|/xyblocks`、`POST /v1/chat/completions|/v1/completions`
* **没有** `/v1/models`（404）、**没有** `/tokenize`（404）→ 客户端注意
* 请求处理：`p_body = dict(body)` → 给 P 发 `max_tokens=1, stream=False, return_token_ids=True, cache_salt="xyvllm-<ns>-<rid>"`（**cache_salt 让每个请求都前缀缓存 miss**，这是设计，不是 bug）
* 超时/prompt 上限：`XY_MAX_PROMPT_CHARS`（见问题 6）
* `POST /v1/completions` 有额外限制：**提示词至少 8 token**，否则 400

### 6.2 问题 11：压测只出 33 个 token，不是 500

* **现象**：`max_tokens=500` 但 `completion_tokens=33`。
* **根因**：提示语结尾写了 "Summarize in one sentence"，模型自然停住了。
* **解决**：请求里加 `"ignore_eos": true`（`min_tokens` 不行——代理给 P 只发 `max_tokens=1`，min_tokens 会与之冲突）。

### 6.3 问题 12：C4/C8 的 prefill/decode 明细缺失

* **现象**：并发压测时 `/stats` 只能看到最后一两个请求的耗时。
* **根因**：代理的 `/stats` 返回的是 `self.p_lat[-20:]`（**只保留最近 20 条**），我的增量切片写法在 C4/C8 时取错了窗口。
* **解决**：改成固定取**尾部 C 条** `[-C:]`（C ≤ 20 时正确）。

---

## 7. ★ 核心问题：prefill 只有 4.1K tok/s（用户："速度不对，prefill 应该能到 10000 左右才对"）

这是本次会话最后、也是最重要的一战。用户要求"全流程插入探针告诉我哪里慢，对比镜像告诉我问题在哪"。

### 7.1 诊断链（7 个探针，逐层缩小范围）

**探针 1 — 三组对照计时**

| 组 | 结果 | 推论 |
|---|---|---|
| A. 代理 16-token 输入 | 0.18 s | 固定开销极小 |
| B. P 直连 100K，`max_tokens=1`，无 kv_transfer | 21.48 s | 慢在 P |
| C. P 直连 100K + kv_transfer | 0.22 s | ⚠️ 假象：与 B 同 prompt，命中了前缀缓存，不能用来测导出开销 |
| D. 代理全链 100K | 22.32 s | 代理只多 0.84 s |

**探针 2 — `/metrics` 阶段拆分 + 长度扫描**

```
request_queue_time_seconds      +0.000     ← 没有排队
request_prefill_time_seconds    +20.547    ← 20.5 s 全在 prefill 内部
time_to_first_token_seconds     +20.662
/tokenize 88,404 token          0.12 s     ← 分词不是瓶颈（755K tok/s）
```

长度扫描（每个 prompt 唯一前缀，避免前缀缓存）：

| prompt token | 客户端耗时 | µs/token |
|---|---|---|
| 1,013 | 0.31 s | 306 |
| 7,992 | 2.01 s | 252 |
| 23,940 | 5.84 s | 244 |
| 47,958 | 11.47 s | 239 |
| 87,859 | 21.30 s | 242 |

拟合斜率 **242 µs/token**、截距 **0.07 s** → **与长度严格成正比、几乎没有固定开销**，说明是"每个 token 的额外活"，不是启动/握手/排队问题。

**探针 3 — GPU / CPU 采样（决定性证据）**

```
nvidia-smi dmon:  SM 100% 两秒 → 0% 两秒 → 100% 两秒 → …      ← GPU 约一半时间空转
vmstat 1:         bi ≈ 280,000 块/s（≈143 MB/s）  wa 12-15%  cs 1.0-1.34M/s
                  → 128 个读线程 100% 堵在块设备 I/O（12.5% wa ≈ 4 核跑满 iowait）
```

同时 vLLM 自己的吞吐日志里出现过 `Avg prompt throughput: 8850 tokens/s` —— **GPU 真正算起来时是 8.8K tok/s，正好接近用户期望的 1 万**。

**探针 4 — P 的 engram 配置 vs 现网配置**

```
我（改前）: DSV41_ENGRAM_REMOTE=            ← 空！
            DSV41_ENGRAM_DISK=1  DISK_THREADS=128  DISK_CHUNK=64
现网official-ref-host: DSV41_ENGRAM_REMOTE=1=…:9612@0:76801234,1=…@76801234:153602468,…,
                               14=…:9611@0:76803337,…（每层 5 片，落在 Spark 原生 NVMe）
            DSV41_ENGRAM_DISK_THREADS=32  DSV41_ENGRAM_DISK_CHUNK=16
```

**探针 5 — 算清"每个 chunk 到底要读多少行"**

P 启动日志（最关键的一行）：

```
Engram DISK rows staged before the forward: 2 layers, heads [0, 24) of 24,
up to 16384 tokens/step, 32/128 read threads
Engram DISK stager: … max 393216 rows/layer
```

* 每个 16K token 的 chunk 要查：**16384 token × 24 个 hash head × 2 层 = 786,432 行**
* 再加同样数量的 scale 行 → **每 chunk 约 157 万次 pread**
* 88K prompt ⇒ 约 **840 万次 pread**
* 本地实测随机行读取速率 ~26-55 万行/s ⇒ **每 chunk 1.4-3.0 s**，与观测到的 1.6-2 s GPU 空转**完全吻合**

**探针 6 — 本地盘微基准（排除"是不是盘坏了"）**

容器内直接测 `pread` 同一批 safetensors 文件：

```
20k 随机 256B 行：62 ms（冷/带读放大）  30 ms（命中页缓存）
O_DIRECT 4KB x16384：57 ms  (1.17 GB/s)
顺序读：1.87 GB/s
```

结论：盘本身没坏、也不慢，**但它是"一块消费级 NVMe"，而生产设计里这些行根本不从本地盘读**。

**探针 7 — 与官方启动脚本逐条对照**

读了包里的 `official-ref-host-launch_p21.sh`（现网 P 的启动脚本）和 `start_engram_disk.sh`，找到官方原话：

* 官方 model 目录 `model-p21` 是 `make_p21_model.py` 造的：**engram embed 行是稀疏空洞**（只留 header 和 q/k/wkv 张量）→ 也就是说生产环境**物理上不可能**走本地读，`DSV41_ENGRAM_REMOTE_FALLBACK=0` 配合 REMOTE 使用
* 官方注释还写着：磁盘后端 **"NEVER point it at the loop-backed /var/tmp/models copy (4.7k rows/s there)"**、*"28 ms / 20k rows"*
* 官方 A/B 数据：`ENGRAM_GPU_STAGE=1` 后 149k prompt 从 6408 → **12996 tok/s**（注意：那是 **official-ref-host 双卡 TP2** 的数字）

我这边用的是带**真实 engram 数据**的 `fp4h96` 目录，所以"本地读"这条路是**能跑但极慢**的静默退化——不报错，只是慢。

### 7.2 根因（一句话）

> **P 的 `DSV41_ENGRAM_REMOTE` 被我留空了**，P 于是回退到"进程内从本地 NVMe 逐行 pread"的回退路径，每个 16K chunk 要发 ~157 万次随机小读、整个 88K prompt ~840 万次，全部压在 P 唯一一块消费级 NVMe 上，GPU 一半时间在等盘 —— 而 GPU 本身能跑 8.8K tok/s。

不是镜像问题、不是编译问题、不是 RDMA 问题、不是 GPU 问题。

### 7.3 修复（完全按官方 "A+" 布局）

**步骤 1：切 engram 分片（4 路）**

4 台 Spark 各自用官方 `make_l14_slice.py` 把自己那 1/4 全局行区间切成 `xy-engram-slice-v1`：

* 布局：`u64 header_len | JSON header（补齐到 4096 对齐）| weight 行区 | scale 行区`
* **4096 对齐的意义**：256 B 的权重行永不跨页 → 一次 NVMe 读 = 一行
* 自带校验：写完随机抽 64 行与源分片逐字节比对，打 `PASS/FAIL`
* 行区间：

| 层 | 分片 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 1（384,006,168 行） | 0:96001542 | 96001542:192003084 | 192003084:288004626 | 288004626:384006168 |
| 14（384,016,682 行） | 0:96004171 | 96004171:192008342 | 192008342:288012513 | 288012513:384016682 |

* 产物：`/var/tmp/dsv41-engram-l{1,14}-4of/l{L}-s{K}of4.safetensors`，每片 **25.34 GB**，8 片共 203 GB，全部在**原生 ext4**（`/dev/nvme0n1p2`）上

**步骤 2：起 8 个分片服务器**（每台 Spark 2 个）

```bash
docker run -d --name dsv41-engram-l14 --restart unless-stopped --network host \
  --memory 8g --memory-swap 8g --oom-score-adj -500 --ulimit nofile=65536:65536 \
  -v /var/tmp/dsv41-engram-l14-4of:/slices:ro -v /var/tmp/dsv41-engram-remote:/srv \
  --entrypoint python3 dsv41-sglang:mhcbf16parts \
  /srv/engram_gather_server.py --shard /slices/l14-s{K}of4.safetensors --layer 14 \
  --backend disk --lib /srv/libxyrow.so --io-threads 128 --direct 0 \
  --bind <SPARKx_ROCE_IP> --port 9611 --workers 16 --log-every 200
```

* `--backend disk`：scale 行（8 B×96M ≈ 0.72 GiB）常驻内存，weight 行（256 B）每次请求经 `libxyrow.so` 用 128 个 pthread `pread`
* 9611 = layer 14，9612 = layer 1（与官方端口布局一致）
* 协议：`DSEG` 二进制协议，请求带**全局行 id**，服务端按 `row_base` 自己减；HELLO 返回 dim/sb/整表 num_rows

**步骤 3：验收（起服务后立刻做）**

* HELLO 8/8 全对：`dim=256 sb=8 num_rows=384006168/384016682`
* 每层抽 64 行，与 **P 自己那份完整分片**逐字节比对：**0 处不一致（PASS）**
* P 侧实测抓取速度：单层 393,216 行 **0.47-1.48 s ≈ 82 万行/s**（本地回退路径需要 ~1.6-3 s 且占用 P 自己的 CPU）

**步骤 4：把 P 接上**

在 `/home/p-user/dsv41-p21-handshake.sh` 里把

```
-e DSV41_ENGRAM_REMOTE= \
```

改成

```
-e DSV41_ENGRAM_REMOTE=1=<SPARK0_ROCE_IP>:9612@0:96001542,1=<SPARK1_ROCE_IP>:9612@96001542:192003084,
  1=<SPARK2_ROCE_IP>:9612@192003084:288004626,1=<SPARK3_ROCE_IP>:9612@288004626:384006168,
  14=<SPARK0_ROCE_IP>:9611@0:96004171,14=<SPARK1_ROCE_IP>:9611@96004171:192008342,
  14=<SPARK2_ROCE_IP>:9611@192008342:288012513,14=<SPARK3_ROCE_IP>:9611@288012513:384016682
```

同时把 `DSV41_ENGRAM_DISK_THREADS/CHUNK` 对齐官方 `32/16`（仅回退路径用），重启容器。
启动日志确认：

```
Engram REMOTE mode: layer 1 rows [0, 384006168) served by 4 shards: <SPARK0_ROCE_IP>:9612[0:96001542], …
Engram ACTIVE read path: layer 1 rows [0, 384006168) = REMOTE
Engram ACTIVE read path: layer 14 rows [0, 384016682) = REMOTE
```

### 7.4 验证：修复前后对照

| prompt | 修复前 | 修复后 | 前 µs/token | 后 µs/token |
|---|---|---|---|---|
| 1,013 | 0.31 s | 0.14 s | 306 | 139 |
| 7,992 | 2.01 s | 0.86 s | 252 | 107 |
| 23,940 | 5.84 s | 2.45 s | 244 | 102 |
| 47,958 | 11.47 s | 4.76 s | 239 | 99 |
| 87,859 | 21.30 s | **8.62 s** | 242 | **98.1 → 10,195 tok/s** |

旁证：

* GPU dmon 变成 **prefill 期间持续 100% SM**（不再 2 秒一停）
* `vmstat`：`bi` 28 万块/s → **~0**；`wa` 12-15% → **0%**；`cs` 1.3M/s → 10-35K/s
* `Engram ACTIVE read path = REMOTE`（两层都是）

代理全链路压测（100K 输入 / 500 输出，C1-C8）：

| C | prefill tok/s 前 | prefill tok/s 后 | decode 聚合 后 | e2e tok/s 后 |
|---|---|---|---|---|
| 1 | 4,054 | **10,082** | 23.3 | 15.94 |
| 2 | 4,019 | **10,308** | 91.4 | 33.24 |
| 4 | 4,038 | **10,372** | 199.6 | 41.16 |
| 8 | 3,991 | **10,340** | 214.3 | 44.72 |

### 7.5 一个必须说清的对照

官方 `official-ref-host` 的 **12,996 tok/s 是双卡 TP2** 的数字（折合单卡约 7K）；我们这台 P 只有**一张** 6000D、TP1，
现在 10.2K tok/s **已经超过官方单卡水平**，再往上就是这张卡的算力上限（GPU 已 100% 打满）。

---

## 8. 磁盘与大文件清理（用户逐条确认）

* P：只有 1 块 ZHITAI TiPlus9100 1 TB，**单一分区、无剩余空间**；空余扩展位：3×M.2 + 1×PCIEX16(G4) + 12×SATA
* 经用户逐项确认后**删除**：
  * `._____temp/deepseek-ai/DeepSeek-V4-Pro-DSpark`（76 GB，未下载完）
  * `Qwen3.8-27B`（52 GB）
  * `Ornith-1.0-9B-GGUF`（44 GB）
* **保留**：pip/vllm 缓存、CUDA deb、镜像 tar、arm64 镜像、Qwen3.6-35B-A3B、Huihui、NVFP4、`/data` 下的 Qwen3-Coder、conda 环境
* Spark 侧本轮新增占用：每台 50.7 GB 切片（`/var/tmp/dsv41-engram-l{1,14}-4of/`），剩余 518-612 GB
* 遗留可清理：`/home/spark-user/dsv41-engram/engram-l{1,14}-r{0..3}of4.bin`（我早期打的 264 B 行格式，已被官方 safetensors 切片取代），**每台 50.7 GB，未经用户确认没有删除**

---

## 9. 工具与流程坑（都是我自己踩的，记录下来避免复现）

| # | 坑 | 现象 | 正确做法 |
|---|---|---|---|
| 1 | `ssh -n` 吞掉 stdin | 用 `ssh host 'base64 -d > f' <<< "$b64"` 上传，结果文件 **0 字节** | 上传时**不要**加 `-n`，或把 base64 作为参数：`ssh host "printf %s '$b64' \| base64 -d > f"` |
| 2 | `pkill -f <脚本名>` 自杀 | 远程命令行里就含 `./spark_cut4.sh`，`pkill -f` 匹配到自己的 shell，后面的命令全没执行；括号技巧 `[s]park` 也救不了（命令行里还有别的 `spark_cut4.sh`） | 启动脚本里**不要 pkill 自己**；要清理就单开一次 ssh |
| 3 | PowerShell 毁掉引号 | `--cmd "... \|\| ... $(...) ..."` 被 PowerShell 解析，`grep`/`\|\|` 被本地执行 | 复杂命令一律写成 **`.sh` 脚本文件**再 `--file` 送过去 |
| 4 | PowerShell 里 `curl` 是别名 | 调 `curl` 实际走 `Invoke-WebRequest`，参数语义不同 | 用 `curl.exe` |
| 5 | 陈旧脚本 | 本地改了 `/tmp/gen_d40c.py` 但 P 上还是旧版，导致生成的启动脚本里 GID 一直是 13 | 每次生成前重新上传/校验哈希 |
| 6 | `sed` 模式写错 | 用 Python dict 形式去匹配 `-e VAR=val`，替换没生效 | 改这类文件用 Python 精确改行 + `bash -n` 语法校验 |
| 7 | `tar` 解错目录 | 解到 `/var/tmp` 而预期是 `/` | 解包后逐层核对路径 |
| 8 | `docker load` OCI tar 无 tag | 见问题 2 | 用 config digest 比对后 `docker tag` |
| 9 | my test bug（冒烟） | 工具调用误报 FAIL：我读的是 `choices[0].tool_calls` | 正确路径是 **`choices[0].message.tool_calls`**；改完 10 组中英文 A/B 全通过 |
| 10 | 前缀缓存污染对照实验 | "P 直连 + kv_transfer" 只花 0.22 s，看着像"导出零开销"，其实命中了前缀缓存 | A/B 时每个 prompt 加**唯一前缀**（或用不同 cache_salt） |

---

## 10. 性能相关参数对照（我 vs 官方 official-ref-host）

| 项目 | 官方 official-ref-host | 本次部署 | 说明 |
|---|---|---|---|
| `DSV41_ENGRAM_REMOTE` | 每层 5 片（9611/9612 + 备用 9613/9614） | 每层 4 片（9611/9612） | ★ 修复的核心 |
| `DSV41_ENGRAM_DISK_THREADS/CHUNK` | 32 / 16 | 32 / 16（已对齐） | 仅回退路径 |
| `GPU_STAGE` / `PREFETCH` | 1 / 1 | 1 / 1 | 预取机制一直是对的 |
| 模型目录 | `model-p21`（engram 行是稀疏空洞） | `fp4h96`（engram 真实数据） | 我这边因此静默退化而不是报错 |
| `--gpu-memory-utilization` | 0.98 | 0.92 | 只影响 KV 容量 |
| `--max-num-batched-tokens` | 8192 | 16384 | 都可 |
| `--max-model-len` | 153600 | 131072 | 都可 |
| TP | **2（两张 6000D）** | 1 | 官方 12,996 tok/s 是双卡数字 |

---

## 11. 运维手册

### 11.1 健康检查（一条命令）

```bash
# 从客户端
curl.exe -s http://<P_IP>:5701/health
# 期望 {"ok": true, "prefill": 200, "decode": 200, "rooms": N}
```

### 11.2 组件状态

```bash
# P（在 <P_IP> 上）
docker ps --format '{{.Names}} {{.Status}}' | grep -E 'p21|proxy'
docker logs --tail 50 dsv41-p21-solo | grep -E 'ACTIVE read path|ERROR'
ss -ltn | grep -E '5701|8000|5600'

# 每台 Spark（<SPARK0-3_ROCE_IP>）
docker ps --format '{{.Names}} {{.Status}}'
docker logs --tail 20 dsv41-engram-l14
ss -ltn | grep -E '9611|9612'
```

### 11.3 重启顺序（重要）

1. 先起 4 台 Spark 的 **engram 分片服务器**（`restart=unless-stopped`，一般自动起）
2. 再起 4 台 Spark 的 **D40 decode**
3. 最后起 **P**（`bash /home/p-user/dsv41-p21-handshake.sh`，冷启动约 1.5-9 分钟）
4. 代理 `restart=unless-stopped`，会自动重连

> ⚠️ `dsv41-p21-solo` 和 `dsv41-d40-sgl` 的 restart 策略目前是 **`no`**，机器重启后需要手动拉起。
> 建议改成 `docker update --restart unless-stopped dsv41-p21-solo`（Spark 上同理）。
> P 的启动脚本会做 REMOTE 可达性 preflight，分片服务没起时会明确报错，不会静默变慢。

### 11.4 常见故障处置

| 症状 | 检查 | 处置 |
|---|---|---|
| `/health` 里 prefill≠200 | P 容器是否在跑、模型是否加载完 | `docker logs dsv41-p21-solo`；等加载完（约 1.5-9 min） |
| `/health` 里 decode≠200 | 4 台 Spark 的 D40 是否 healthy | `docker ps`；起 D40 |
| 请求慢、GPU 只有一半占用 | **是否退化成本地 engram 读** | `docker logs dsv41-p21-solo \| grep 'ACTIVE read path'` 必须都是 `REMOTE`；`vmstat 1` 看 `bi`/`wa` |
| engram 分片服务挂了 | `docker ps` + HELLO | `docker restart dsv41-engram-l1 dsv41-engram-l14` |
| KV 传不过去（D 拿不到） | NIXL/UCX GID | 确认 P `UCX_IB_GID_INDEX=12`、D `=2`（RoCE **v1** IPv4） |
| 大 prompt 被 400 拒 | 代理 env | 确认 `XY_MAX_PROMPT_CHARS=2000000` |
| 进程起不来、mmap 失败 | `vm.overcommit_memory` | 应为 1（已持久化到 `/etc/sysctl.d/90-dsv41-overcommit.conf`） |

### 11.5 回滚

```bash
# 回滚 P 的 engram 配置（回到本地读的旧行为，性能会退回去）
cp /home/p-user/dsv41-p21-handshake.sh.bak-before-engram-remote \
   /home/p-user/dsv41-p21-handshake.sh
bash /home/p-user/dsv41-p21-handshake.sh

# 停 engram 分片服务（每台 Spark）
docker rm -f dsv41-engram-l1 dsv41-engram-l14

# 生产（旧 vLLM）容器仍保留为 Exited 状态，可 docker start 恢复：
#   spark-0/1: dsv41-head   spark-2/3: dsv41-worker
# 原始 inspect 存档：/home/p-user/dsv41-rollback/
```

---

## 12. 关键文件清单

### P（<P_IP>）

| 路径 | 说明 |
|---|---|
| `/home/p-user/dsv41-p21-handshake.sh` | **P 启动脚本**（含最终 REMOTE 配置） |
| `/home/p-user/dsv41-p21-handshake.sh.bak-before-engram-remote` | 改 REMOTE 前的备份（回滚用） |
| `/home/p-user/dsv41-launch-proxy5701.sh` | 代理启动脚本（含 `XY_MAX_PROMPT_CHARS=2000000`） |
| `/opt/xy-proxy-v8/xy_pd_proxy_v8cutover_5701.py` | 代理源码（506 行） |
| `/mnt/dsv41-tp6/fp4h96/` | 权重目录 232 GB（48 个 shard，含 2×101.5 GB engram 分片） |
| `/mnt/dsv41-tp6/dsv41-pp3/patch-p21-exl3/` | 35 个 vllm 源码补丁（含 `engram.py`/`model_state.py`） |
| `/mnt/dsv41-tp6/dsv41-pp3/cache-p21-exl3/` | vLLM/FlashInfer 编译缓存 |
| `/etc/sysctl.d/90-dsv41-overcommit.conf` | `vm.overcommit_memory = 1` |
| `/home/p-user/dsv41-pd-package-20261013/` | 官方交付包（images/launch/scripts） |

### 4×Spark（<SPARK0-3_ROCE_IP>）

| 路径 | 说明 |
|---|---|
| `/home/spark-user/dsv41-d40c-launch-rank{0..3}.sh` | D40 decode 启动脚本 |
| `/home/spark-user/models/DeepSeek-V4.1-Flash/` | 权重（本地 ext4，非 NFS） |
| `/var/tmp/dsv41-engram-remote/engram_gather_server.py` | 分片服务端（支持 resident/mmap/disk 三种 backend） |
| `/var/tmp/dsv41-engram-remote/libxyrow.so` + `xy_rowread.c` | 行读取 C 库（128 pthread） |
| `/var/tmp/dsv41-engram-remote/make_l14_slice.py` | 切片工具（自校验） |
| `/var/tmp/dsv41-engram-remote/spark_cut4.sh` / `spark_start_engram.sh` | 本次编写的切片/起服务脚本 |
| `/var/tmp/dsv41-engram-l{1,14}-4of/l{L}-s{K}of4.safetensors` | **8 个切片，25.34 GB/片** |
| `/home/spark-user/dsv41-engram/*.bin` | 早期打包（已被取代，待确认后清理） |

### 本地工作目录 `<LOCAL_WORK_DIR>\`

| 文件 | 用途 |
|---|---|
| `rssh.py` / `rjump.py` / `rfetch.py` | SSH 到 P / 经 P 跳转到 Spark 的取放文件工具（**rssh.py 内含明文密码，注意保管**） |
| `probe1.sh` ~ `probe8.sh` | 逐层诊断探针（对照计时、metrics 拆分、GPU/CPU 采样、切片进度等） |
| `bench_disk.py` | 容器内本地 engram 读路径微基准 |
| `verify_engram_remote.py` | 分片服务验收（HELLO + 逐字节比对 + 抓取计时） |
| `pd_bench.py` | PD 压测（C1-C8，100K 入 / 500 出，用代理 `/stats` 拆 prefill/decode） |
| `decode_only.py` | 纯 decode 速率测试（排除 prefill 干扰） |
| `smoke_server.sh` / `smoke_client.py` | 冒烟测试（服务端 6 项 / 客户端 10 项） |
| `client_check.py` / `client_check2.py` / `tools_ab.py` | 客户端可用性探测、工具调用 A/B |
| `cmp_ref.py` | 与官方 `official-ref-host-launch_p21.sh` 逐项对照 |
| `PREFILL-10K-报告.md` / `访问方式.md` | 性能报告 / 使用说明 |

---

## 13. 遗留项与建议

| 优先级 | 事项 | 说明 |
|---|---|---|
| 高 | 把 `dsv41-p21-solo` 和 `dsv41-d40-sgl` 的 restart 策略改成 `unless-stopped` | 目前是 `no`，重启机器后不会自动恢复 |
| 中 | 清理 `/home/spark-user/dsv41-engram/*.bin` | 8 个文件共 203 GB，已被官方切片取代，**等确认** |
| 中 | 100K 长上下文下单流 decode 约 23-33 tok/s（16-token 上下文是 81.7） | 属 MLA 长上下文本身开销，与本次修复无关；如在意可另开一轮优化 |
| 低 | `--gpu-memory-utilization 0.92 → 0.98`（官方值） | 换取更大 KV 容量 |
| 低 | `--max-num-batched-tokens 16384 → 8192`（官方值） | 更细的 chunk，预取窗口更密；影响很小 |
| 低 | 给 engram 容器加 `--cpuset-cpus` | 避免 128 线程抓取与同机 D40 抢 20 核 CPU |
| 低 | `rssh.py` 里的明文密码 | 建议改用密钥或环境变量注入 |

---

## 14. 复现命令速查

```bash
# 1) 从客户端验证服务
curl.exe -s http://<P_IP>:5701/health
curl.exe -s http://<P_IP>:5701/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"deepseek-v4.1-flash\",\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}],\"max_tokens\":128}"

# 2) 重跑冒烟（本地 Windows）
python <LOCAL_WORK_DIR>\smoke_client.py      # 客户端 10 项
python <LOCAL_WORK_DIR>\rssh.py --file <LOCAL_WORK_DIR>\smoke_server.sh   # 服务端 6 项

# 3) 重跑压测
python <LOCAL_WORK_DIR>\rjump.py put pd_bench.py /tmp/pd_bench.py
python <LOCAL_WORK_DIR>\rssh.py --cmd "cd /tmp && PROMPT_TOKENS=100000 MAX_TOKENS=500 LEVELS=1,2,4,8 python3 /tmp/pd_bench.py"

# 4) 重跑分片验收
python <LOCAL_WORK_DIR>\rjump.py put verify_engram_remote.py /tmp/verify_engram_remote.py
python <LOCAL_WORK_DIR>\rssh.py --cmd "docker cp /tmp/verify_engram_remote.py dsv41-p21-solo:/tmp/ && docker exec dsv41-p21-solo python3 /tmp/verify_engram_remote.py"
```
