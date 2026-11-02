# AI 一键构建指南 · DS4.1 Flash V8 Docker 镜像（PP2 / PD 分离）

> 这份文件是写给“替用户自动部署的 AI”看的。先读【最重要的一条】和第 1、2 节。
> （English guide below。）

## ⚠️ 最重要的一条：这是“补丁/配置镜像”，不是“从零部署镜像”

拓扑：P 段 2×RTX 6000D（vLLM TP2，验证后可换 RTX 5500 Pro / 6000 Pro）；D 段 4×DGX Spark/GB10（SGLang TP4/EP4），经 NIXL 交接；整体是 PP2 的 PD 分离。

GHCR 镜像 `ds41-flash-v8:latest` **按设计只含**：V8 跨引擎补丁、DSpark 最终增量、运行环境、SHA256 证据。**不含** 510GB 官方权重，也**不含**私有角色镜像 —— 它们必须**已经在目标 GB10 主机上**，由主机侧的角色启动器挂载/选用。

所以：
- V8 的作用是把补丁**叠加**到你已有的 DS4.1 集群上。如果目标主机**还没有** DS4.1 的角色镜像与权重，单靠 V8 镜像**无法从零拉起服务** —— 这不是 bug，是设计边界。
- `verify` 打印 `PASS` 只代表**包内容与校验和完整**，不代表服务能起。
- ⚠️ 仓库里的 `ds41-flash-v8/bundle/README.md` 引用了 `../ds4.1flash6gpu-v7/DELIVERY.md`（本仓库**不存在**该路径）。AI 若顺着这个链接找文件会失败，**请忽略它**，以 `ds41-flash-v8/README.md` 为准。

## 1. 能一键完成的部分

```bash
git clone https://github.com/Soulmate-Halo/heterogeneous-gpu-pd-lab.git
cd heterogeneous-gpu-pd-lab
IMG=ghcr.io/soulmate-halo/heterogeneous-gpu-pd-lab/ds41-flash-v8:latest
docker pull $IMG
docker run --rm $IMG verify   # PASS: DS4.1 Flash V8 bundle, package order, and embedded checksums
```

## 2. 把补丁叠加到目标主机

```bash
# 进镜像看内容：
docker run --rm -v /var/tmp/dsv41-d40:/var/tmp/dsv41-d40 $IMG shell
# 或把 bundle 叠加到宿主机 overlay 目录：
./apply-bundle.sh --overlay-dir /var/tmp/dsv41-d40/overlay
```

`apply-bundle.sh` 把 V8 补丁与运行环境落到宿主机 overlay；随后由**主机已有**的 P21 / D40 / 代理角色启动器挂载 510GB 权重并拉起 PD 两段。权重与角色镜像由你在主机侧准备，镜像不提供、也不下载。

## 3. 给自动部署 AI 的硬规则

1. 先确认目标主机**已具备** DS4.1 的角色镜像与 510GB 权重。没有就**停下告诉用户**：“V8 是补丁层，需要先有可运行的 DS4.1 P/D 集群；请提供角色镜像与权重，或改用从零方案。”**不要**试图从仓库构建权重或角色镜像。
2. `verify` 的 `PASS` 不是服务可用的证据；服务可用以 P/D 端点健康为准。
3. 需要完整的单卡→多机加速（含权重获取说明）时，优先看 [V10 方案](../ds41-flash-v10/AI-BUILD.md)。

## 4. 参考

[完整配置与应用步骤](README.md) · [Dockerfile](Dockerfile) · [Compose 配置](compose.yml)

---

# AI One-Click Build Guide · DS4.1 Flash V8 Docker image (PP2 / PD separation)

> Written for the AI agent that deploys on the user's behalf.

## ⚠️ Read first: this is a patch/config image, not a from-scratch deploy image

Topology: P stage = 2×RTX 6000D (vLLM TP2; RTX 5500 Pro / 6000 Pro are validated substitutes); D stage = 4×DGX Spark/GB10 (SGLang TP4/EP4) handed off via NIXL; overall PP2 with PD separation.

The GHCR image `ds41-flash-v8:latest` ships **only** the V8 cross-engine patches, the final DSpark delta, the runtime environment and SHA256 evidence. It deliberately does **not** redistribute the 510 GB official weights or the private role images — those must **already be on the target GB10 hosts**, mounted/selected by the host-side role launchers.

So V8 **overlays patches** onto an existing DS4.1 cluster. If the target host has no DS4.1 role images and weights yet, the V8 image alone **cannot bring the service up from scratch** — that is a design boundary, not a bug. A `verify` `PASS` only means the bundle and checksums are intact.

⚠️ `ds41-flash-v8/bundle/README.md` references `../ds4.1flash6gpu-v7/DELIVERY.md`, a path that does **not** exist in this repo. An AI that follows that link will fail — ignore it and use `ds41-flash-v8/README.md`.

## 1. The one-click part

```bash
git clone https://github.com/Soulmate-Halo/heterogeneous-gpu-pd-lab.git
cd heterogeneous-gpu-pd-lab
IMG=ghcr.io/soulmate-halo/heterogeneous-gpu-pd-lab/ds41-flash-v8:latest
docker pull $IMG
docker run --rm $IMG verify   # PASS: DS4.1 Flash V8 bundle, package order, and embedded checksums
```

## 2. Overlay the patches onto the target host

```bash
docker run --rm -v /var/tmp/dsv41-d40:/var/tmp/dsv41-d40 $IMG shell
# or overlay the bundle onto a host directory:
./apply-bundle.sh --overlay-dir /var/tmp/dsv41-d40/overlay
```

`apply-bundle.sh` lands the V8 patches and runtime onto a host overlay; the host's existing P21 / D40 / proxy role launchers then mount the 510 GB weights and bring up the two PD stages. You provide the weights and role images on the host; the image neither ships nor downloads them.

## 3. Hard rules for the deploying AI

1. Confirm the target host already has the DS4.1 role images and 510 GB weights. If not, **stop and tell the user**: "V8 is a patch layer; it needs a runnable DS4.1 P/D cluster first." Do not try to build weights or role images from the repo.
2. A `verify` PASS is not proof the service runs; use P/D endpoint health for that.
3. For a complete single-card → multi-host path (including how to obtain weights), prefer the [V10 guide](../ds41-flash-v10/AI-BUILD.md).

## 4. References

[Docker configuration and apply procedure](README.md) · [Dockerfile](Dockerfile) · [Compose template](compose.yml)
