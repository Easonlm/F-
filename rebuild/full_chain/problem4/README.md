# 问题四复现

在项目根目录运行：

```bash
python problem4/run_problem4.py
```

脚本只读取 `real_attachments/C_efficiency_evolution/` 和问题一至三的既有结果；全部新文件写入 `problem4/`。主要交付为 `problem4_report.md`、`问题四完整详解.md`、`problem4_summary.md`、`outputs/tables/final_project_summary.json` 和 `outputs/tables/verification_results.json`。

模块：`data_pipeline.py` 负责数据审计、严格实体匹配和 C8 逐任务解析；`bridge_model.py` 负责 C5/C6 桥接及分组验证；`history_forecast.py` 负责历史前沿、规模/时间项分解、算力情景与探索性预测；`mechanism.py` 只读取问题二/三 V2 接口进行机制路径；`build_report.py` 自动出图与文稿。

`outputs/tables/short_term_baseline_comparison.csv` 在现有短期回测的相同预测原点和目标季度，额外给出“上一季度前沿保持不变”的朴素基线；`short_term_baseline_summary.csv` 汇总误差与目标季最低样本规则。它只用于短期风险参照，不改变 12/24 月情景或机制路径。

`frontier_definition_sensitivity.csv` 并列最低样本数、滚动窗口及同截止日的 C2 口径；严格末季 n=2 不用于判断前沿方向。预测表逐行标记能力锚的样本量，`mechanism.py` 会校验 P2 artifact 与 P3/P4 接口参数是否一致。

**证据限制：** 直接榜单只有 2024-06 到 2025-03；严格开放样本截止 2025-01-31，只有四个季度。12/24 个月没有直接观测回测，预测数字属于情景演算。`verification_results.json` 因此保留未通过项，不把它们伪装成已验证。
