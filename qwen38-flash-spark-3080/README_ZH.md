# Qwen3.8 Flash-Next V2：RTX 3080 + DGX Spark

这是一个可复现的双机 Strata NVFP4 运行包。RTX 3080 20GB 承担 NVFP4 dense/prefill、attention/QSA 和 FP8 KV；DGX Spark / GB10 承担 MoE 冷专家、PLE/KV worker。启动顺序是先在 Spark 上启动 worker，再在 3080 上启动 engine，必要时再启动 router。

## V2 数据口径

| 指标 | 数值 | 口径 |
| --- | ---: | --- |
| Prefill 三次 | 1030.80 / 1163.40 / 1161.10 tok/s | prompt=3600，实测 |
| Prefill 峰值 | 1163.40 tok/s | 实测 |
| 聚合解码峰值 | 60.25 tok/s | C12，12 路，2048 in/256 out，156/156，端到端实测 |
| V2 聚合解码目标 | 240.00 tok/s | N=8 内核批量缩放推算，**不是端到端实测** |
| 单请求端到端解码 | 45.54 tok/s | N=1 实测 |

证据在 `evidence/v2-metrics.json`、`evidence/performance-v2.json` 和 `evidence/README.md`。镜像只带脚本、配置、源快照和脱敏数据，不带 510 GB 官方权重、hot/cold pack、PLE 表或 role 镜像；这些路径必须通过 volume 挂载。

## 使用

```bash
python qwen38-flash-spark-3080/verify_bundle.py
bash qwen38-flash-spark-3080/apply-bundle.sh --overlay-dir /opt/strata-overlay
# 在两个目标主机分别运行：
bash qwen38-flash-spark-3080/run-dual-host.example.sh worker
bash qwen38-flash-spark-3080/run-dual-host.example.sh engine
```

公共镜像：`ghcr.io/soulmate-halo/heterogeneous-gpu-pd-lab/qwen38-flash-spark-3080:latest`。示例启动脚本使用 host network、NVIDIA runtime 和 `/dev/infiniband`；请把 `STRATA_RDMA_WORKER_HOST` 换成你自己的 RDMA 地址。`bundle/runtime.env` 是模板，不含凭据。

`Dockerfile` 可由 GitHub Actions 以 `linux/amd64,linux/arm64` 构建。上游和第三方许可见 `src/`、`src/third_party/` 与 `NOTICE`。
