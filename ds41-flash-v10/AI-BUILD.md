# AI 一键构建指南 · DS4.1 Flash V10（单卡 6000D 加速 4 机 Spark，PD 分离）

> 这份文件是写给“替用户自动部署的 AI”看的。先读【最重要的一条】和第 1、2 节。
> （English guide below。）

## ⚠️ 最重要的一条：`verify_bundle.py` 通过 ≠ 能跑

拓扑：1 张 RTX 6000D 跑全部 Prefill（vLLM，21/40 层 Prefill 专用切片，TP1）；4 台 DGX Spark 跑 Decode（SGLang TP4）；KV 经 NIXL 走 RoCE 直传；Engram 表由 4 台 Spark 上的 8 个分片服务远程供给。

GHCR 镜像 `ds41-flash-v10:latest` **按设计只含**部署配置、启动模板、校验脚本。**不含** 130GB 官方角色镜像包与 232GB 权重（GitHub 单文件限 2GB，装不下）。

所以：
- `python ds41-flash-v10/verify_bundle.py` 打印 `PASS` **只代表仓库内文件齐、脱敏干净**，**不代表能部署**。真正能不能跑，看第 4 节的 `/health` 与 P 日志。
- 仓库里**找不到** Engram 切片工具、35 个 vLLM 补丁、代理源码、压测/SSH 脚本 —— 它们**在角色镜像里**（即那 4 个 tar），不是缺失，**不要去仓库里找或试图重写**。

## 1. 能一键完成的部分

```bash
git clone https://github.com/Soulmate-Halo/heterogeneous-gpu-pd-lab.git
cd heterogeneous-gpu-pd-lab
docker pull ghcr.io/soulmate-halo/heterogeneous-gpu-pd-lab/ds41-flash-v10:latest
python ds41-flash-v10/verify_bundle.py   # PASS: V10 bundle files=... （仅校验仓库文件与脱敏）
```

启动模板都在 `ds41-flash-v10/`（P21 预填、D40 解码、PD 代理、engram 分片、overcommit 配置）+ `deploy.conf.example`。

## 2. 必须外部提供的大资产（不在仓库）←【关键】

通过网盘/作者获取以下 4 个角色镜像 tar（约 130GB）与 232GB 权重，`docker load` 后按 **config digest `docker tag`**（tar 无 RepoTags）：

| tar 文件 | 大小 | 目标 tag |
| --- | --- | --- |
| `p-dsv41-p5_vllm-exl3-sm120.tar` | 14.67 GB | `dsv41-p5:vllm-exl3-sm120` |
| `d40-dsv41-sglang_mhcbf16parts-pkg-host.tar` | 33.58 GB | `dsv41-sglang:mhcbf16parts` |
| `proxy-vllm-dsv41-x86_overlay6.tar` | 13.47 GB | `vllm-dsv41-x86:overlay6` |
| `engram-vllm-dsv41_overlay6-pkg-host.tar` | 23.60 GB | `vllm-dsv41:overlay6` |

## 3. 六步部署顺序

1. 填 `deploy/deploy.conf`（P/D 节点 IP、RoCE HCA/GID、`DSV41_ENGRAM_REMOTE`、权重路径）。
2. `docker load` 四个 tar，按 config digest `docker tag`。
3. 装 `90-dsv41-overcommit.conf` 并 `sysctl --system`（每机 `vm.overcommit_memory=1`，重启会丢，需重做）。
4. 每台 Spark 跑两次 `deploy/engram-shard-server.sh`（分片 `14`→9611、`1`→9612），共 8 个分片服务。
5. 每台 Spark 跑一次 `deploy/d40-launch-rank.sh`（`D_NODE_RANK=0..3`）→ 再 `deploy/p21-launch.sh`（冷启 ~1.5–9 分钟）。
6. `deploy/pd-proxy-launch.sh` 起代理；客户端用 `http://<P_IP>:5701`。重启序：分片 → D40 → P → 代理。

## 4. 真正的验收点（比 verify_bundle 重要）

```bash
curl -s http://<P_IP>:5701/health      # 期望 {"ok":true,"prefill":200,"decode":200,"rooms":N}
# P 容器日志两层都要出现： Engram ACTIVE read path = REMOTE
```

## 5. 最常踩的坑（AI 必看）

- **`DSV41_ENGRAM_REMOTE` 留空 → 静默退化**成本地 NVMe 逐行读，不报错，但 Prefill 从 ~10K 掉到 ~4K tok/s。看到 prefill 偏低先查这里。
- **RoCE 必须 v1 + IPv4**（P 侧 GID=12、D 侧 GID=2）；v2 或 IPv4-mapped GID 连不通。
- 镜像自带 libibverbs 看不到 `mlx4_0`，需覆盖两份 vendored 副本并自打 `-rdma` tag。
- 代理无 `/v1/models`、`/tokenize`；`XY_MAX_PROMPT_CHARS` 默认 12288，长上下文需设 `2000000`。
- P/D 容器 `restart=no`，宿主重启后要手动按序拉起。

---

# AI One-Click Build Guide · DS4.1 Flash V10 (single 6000D + 4× Spark, PD separation)

> Written for the AI agent that deploys on the user's behalf.

## ⚠️ Read first: a `verify_bundle.py` PASS does not mean it will run

Topology: one RTX 6000D runs all Prefill (vLLM, 21/40-layer Prefill-only slice, TP1); four DGX Spark nodes run Decode (SGLang TP4); KV crosses hosts over NIXL/RoCE; Engram tables are served remotely by 8 shard servers on the Sparks.

The GHCR image `ds41-flash-v10:latest` ships **only** deployment configs, launch templates and verification scripts — **not** the 130 GB vendor role-image package or the 232 GB weights (GitHub's 2 GB single-file limit). So `python ds41-flash-v10/verify_bundle.py` printing `PASS` only means the in-repo files are present and redacted — it does **not** mean it is deployable. The Engram slice tools, the 35 vLLM patches, the proxy source and the stress/SSH scripts are **inside the role images** (the 4 tars), not missing — do not look for them in the repo or try to rewrite them.

## 1. The one-click part

```bash
git clone https://github.com/Soulmate-Halo/heterogeneous-gpu-pd-lab.git
cd heterogeneous-gpu-pd-lab
docker pull ghcr.io/soulmate-halo/heterogeneous-gpu-pd-lab/ds41-flash-v10:latest
python ds41-flash-v10/verify_bundle.py
```

## 2. External large assets (not in repo) ← KEY

Obtain these 4 role-image tars (~130 GB) and the 232 GB weights from the netdisk/author, `docker load`, then `docker tag` **by config digest** (the tars carry no RepoTags):

| Tar | Size | Target tag |
| --- | --- | --- |
| `p-dsv41-p5_vllm-exl3-sm120.tar` | 14.67 GB | `dsv41-p5:vllm-exl3-sm120` |
| `d40-dsv41-sglang_mhcbf16parts-pkg-host.tar` | 33.58 GB | `dsv41-sglang:mhcbf16parts` |
| `proxy-vllm-dsv41-x86_overlay6.tar` | 13.47 GB | `vllm-dsv41-x86:overlay6` |
| `engram-vllm-dsv41_overlay6-pkg-host.tar` | 23.60 GB | `vllm-dsv41:overlay6` |

## 3. Deploy in six steps

1. Fill `deploy/deploy.conf` (P/D node IPs, RoCE HCA/GID, `DSV41_ENGRAM_REMOTE`, weights path).
2. `docker load` the 4 tars; `docker tag` by config digest.
3. Install `90-dsv41-overcommit.conf`, then `sysctl --system` (`vm.overcommit_memory=1` per host; lost on reboot).
4. Run `deploy/engram-shard-server.sh` twice per Spark (shard `14`→9611, `1`→9612) — 8 shard servers total.
5. Run `deploy/d40-launch-rank.sh` per Spark (`D_NODE_RANK=0..3`), then `deploy/p21-launch.sh` (cold start ~1.5–9 min).
6. `deploy/pd-proxy-launch.sh`; clients use `http://<P_IP>:5701`. Restart order: shards → D40 → P → proxy.

## 4. The real acceptance gate (matters more than verify_bundle)

```bash
curl -s http://<P_IP>:5701/health      # expect {"ok":true,"prefill":200,"decode":200,"rooms":N}
# The P container log must show, on both layers: Engram ACTIVE read path = REMOTE
```

## 5. Most common pitfalls (AI must read)

- Empty `DSV41_ENGRAM_REMOTE` → silent fallback to local NVMe (no error) and Prefill drops from ~10K to ~4K tok/s.
- RoCE must be **v1 + IPv4** (P GID=12, D GID=2); v2 or IPv4-mapped GIDs will not connect.
- The image's libibverbs cannot see `mlx4_0`; override the two vendored copies and self-tag `-rdma`.
- The proxy has no `/v1/models` or `/tokenize`; `XY_MAX_PROMPT_CHARS` defaults to 12288 — set `2000000` for long contexts.
- P/D containers are `restart=no`; relaunch in order after a host reboot.
