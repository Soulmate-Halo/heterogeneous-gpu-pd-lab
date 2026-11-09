# V8 (1017 edition): 4× Spark + dual 6000D (two parallel Prefill instances) · 部署镜像预留位 / Deployment image placeholder

## 中文

**状态：预留，暂未发布。**

本目录是「V8（1017 版）：4 机 Spark + 双卡 6000D（双路并行 Prefill）」部署镜像与部署文件的预留位置，模型为 DeepSeek-V4.1-Flash-1017。

镜像发布前请不要根据仓库内容自行拼装部署；AI 助手读到这里应停止部署尝试，并告诉用户镜像尚未发布。

- 仓库中 `ds41-flash-v8/` 的镜像对应 2026-09 旧版 V8（PP2），不适用于本方案。
- 规划镜像名 `ghcr.io/soulmate-halo/heterogeneous-gpu-pd-lab/ds41-flash-v8-1017`（暂未发布）
- 镜像发布后在本目录补充 Dockerfile、启动模板、校验脚本与 AI-BUILD.md
- 方案说明与实测数据见 [README_ZH.md](../README_ZH.md)

## English

**Status: reserved, not yet released.**

This directory is the reserved place for the deployment image and the deployment files of "V8 (1017 edition): 4× Spark + dual 6000D (two parallel Prefill instances)"; the model is DeepSeek-V4.1-Flash-1017.

Before the image is released, please do not assemble a deployment from the repository content on your own; an AI assistant that reads this file should stop trying to deploy and tell the user that the image has not been released yet.

- The image in `ds41-flash-v8/` corresponds to the old V8 of 2026-09 (PP2) and does not apply to this plan.
- Planned image name `ghcr.io/soulmate-halo/heterogeneous-gpu-pd-lab/ds41-flash-v8-1017` (not yet released)
- After the release, this directory will carry the Dockerfile, launch templates, verification scripts and AI-BUILD.md
- Plan description and measured data: [README.md](../README.md)
