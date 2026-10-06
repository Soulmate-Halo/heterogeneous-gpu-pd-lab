# 外置资产契约

镜像和仓库只包含源码、脚本和脱敏证据。模型与运行资产必须由使用者在两台目标机上准备，并通过环境变量挂载。

公开模型入口：

- NVFP4 权重参考：[gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090)
- GGUF/P﻿LE 键参考：[unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
- Strata 上游：[Niko1221/Strata](https://github.com/Niko1221/Strata)

公开 checkpoint 不能直接替代本实验的 `pack-hot`。运行 NVFP4 双机路径还需要：

1. 3080 的 `pack-hot`：包含 48 个 `layer-*-hot-nvfp4.bin` 及 `dense.bin`、`embd.bin`、`index.txt`、tokenizer 和 `manifest-hot.json`。
2. Spark 的 `strata-cold-pack`：与 hot pack 同一模型版本和 blob v2 布局。
3. `dense.gguf`：用 `src/tools/nvfp4_dense_gguf.py` 从匹配的 dense/Q2 参考和原始模型目录转换，不能混用别的模型。
4. `Q2_0-00002.gguf`：提供 PLE native 键；裸 `ple-fp8.bin` 是 **320,001,536 行 × 160 B（约 51.2 GB）**，必须同时提供 `--ple-fp8-scale 0.00019931793212890625`。
5. `profile.bin`：与 hot pack 的路由统计生成，槽数必须覆盖 `--expert-cache 4608`；tokenizer 要包含完整模板和 merges 文件。

脚本不会下载、上传或猜测这些大文件，也不会把 Q2 权重偷换成 NVFP4。路径由 `env.example` 和命令行参数注入。

## Docker 挂载名

Compose 将宿主机的 `MODELS_DIR` 只读挂载到容器 `/models`：`strata-pack-hot/`、`strata-cold-pack/`、`dense.gguf`、`ple-fp8.bin` 和 `profile.bin`。`assets/expert-profile-v4.bin` 是公开的示例资产，只有在与当前 hot pack 的路由统计匹配时才能复制为 `profile.bin`；镜像不会自动下载或生成这些文件。缓存和状态分别挂载到 `/cache`、`/state`，用来保存首次 CUDA 编译产物和服务日志。

启动前可运行 `docker run --rm <image> doctor` 做通用检查；`doctor spark` 与 `doctor 3080` 会进一步检查对应架构、CUDA、RDMA 和模型资产。缺文件时退出码为 2，`help`、`verify`、`smoke` 不需要 GPU。
