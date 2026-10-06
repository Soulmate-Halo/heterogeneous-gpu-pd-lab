# FLASH-395-01 性能数据

汇总见 `metrics.json`，机器校验的冻结值就是其中四个数字：`prefill_tok_s=800`（约值，`prefill_tok_s_approximate`）、`decode_tok_s=40`（约值）、`baseline_prefill_tok_s=273.04`、`baseline_decode_tok_s=41.33`，模型 `qwen3.8-flash-next`，日期 `2026-10-06`。

实验者 2026-10-06 提供：RTX 3080 20GB（OCuLink）＋ AI Max 395 单机冷热专家分离，预填充 **800+ tok/s**（395 单机基线 273.04），解码 **40+ tok/s**（395 单机基线 41.33）。`800+` 与 `40+` 为约值，原始日志与精确负载、并发档位未随本次修订附入；机器校验只验证汇总一致性，未重新执行 GPU 性能压测。

两种部署的解码都由 AI Max 395 承担，因此解码基本持平；提升在预填充（约 3 倍）。

`r434_report.md` 与 `r435_report.md` 是冷专家链路的原始上线/排障报告（prefill 链路 r434、decode 链路 r435），连同 `bundle/src/` 下的三个冻结源码文件一起，用于复核 HIP 冷专家路径的取舍与回退方式。
