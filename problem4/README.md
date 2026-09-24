# 问题四复现

在项目根目录运行：

```bash
python problem4/run_problem4.py
```

脚本只读取 `real_attachments/C_efficiency_evolution/` 和问题一至三的既有结果；全部新文件写入 `problem4/`。主要交付为 `problem4_report.md`、`问题四完整详解.md`、`problem4_summary.md`、`outputs/tables/final_project_summary.json` 和 `outputs/tables/verification_results.json`。

模块：`data_pipeline.py` 负责数据审计、严格实体匹配和 C8 逐任务解析；`bridge_model.py` 负责 C5/C6 桥接及分组验证；`history_forecast.py` 负责历史前沿、规模/时间项分解、算力情景与探索性预测；`mechanism.py` 只读取问题二/三 V2 接口进行机制路径；`build_report.py` 自动出图与文稿。

**证据限制：** 直接榜单只有 2024-06 到 2025-03；严格开放样本截止 2025-01-31，只有四个季度。12/24 个月没有直接观测回测，预测数字属于情景演算。`verification_results.json` 因此保留未通过项，不把它们伪装成已验证。
