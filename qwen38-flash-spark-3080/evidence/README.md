# V2 性能数据

当前汇总为 `v2-metrics.json`、`performance-v2.json` 和 `metrics.csv`。

2026-10-06 实验者提供：联合聚合解码 **242.37 tok/s**（Spark 单机 **107.60 tok/s**），联合单流 **62.00 tok/s**（Spark 单机 **22.30 tok/s**）。原始日志及精确负载、并发档位未随本次修订附入，机器校验验证汇总一致性，未重新执行 GPU 性能压测。

Prefill 三次记录 1030.80 / 1163.40 / 1161.10 tok/s 沿用 `nvfp4-dflash-bench-20261005.json`。单机 Prefill 1117.93 tok/s 来自 `nvfp4-spark-solo-prefill-20261004.json`，两者输入长度不同。

带日期的原始实验 JSON 保留历史原值，不按本次实验者报告改写。旧推算目标不再作为当前 V2 结果展示。
