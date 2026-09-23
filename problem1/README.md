# F 题问题一复现说明

## Q 评价体系补强验证

运行 python problem1/validate_quality_proxy.py 可新增 A1 的逐指标内部一致性核查、分域相关、A2/A3 去重后均值与冲突率及两张论文图；原有 Q 和已有结果文件不被覆盖。详细解释见 [Q 评价体系验证](quality_proxy_validation.md)。

140 条盲评样本保留在原有 JSON 和模板中；新增空白模板为 [人工质量盲评_140条.xlsx](outputs/v2_review/人工质量盲评_140条.xlsx)。每位评审者各复制一份，给所有已评分行填写相同的 rater_id，再独立录入 1–5 分和广告/垃圾 0/1。模板不含 Q 或冲突信息。评审完成后运行 python problem1/analyze_manual_quality.py --ratings 评审者1.xlsx 评审者2.xlsx。脚本会生成 manual_validation_results.csv、样本汇总和图表；空白模板只产生带表头的结果文件，不会生成虚构统计值。

## 第二版（当前推荐查看）

第二版采用训练集内重复交叉验证选出的二阶 ALR Ridge；第一版文件保持不变，便于逐项对照。先看 [第二版方案](problem1_v2_report.md) 和 [第一版与第二版对比](problem1_v1_v2_comparison.md)。第二版在既定 1M 测试集的 RMSE 从 0.3415 降至 0.2702，13 个验证领域均下降；但该测试集已经在第一版中使用，不是全新盲测。跨规模绝对 Loss 仍有明显偏差，10B/70B 估算标签上的指标退步，均已在对比文件中列出。

在已完成第一版原始数据处理的基础上，运行：

```powershell
python problem1/problem1_v2.py
python problem1/nested_v2_validation.py
python problem1/prepare_v2_review.py
python problem1/build_v2_report.py
python problem1/verify_problem1_v2.py
```

第二版质量分 Q 沿用第一版，因为没有独立人工真值。`outputs/v2_review/人工盲评模板.xlsx` 可分别复制给两名评审者；`outputs/tables/v2_blind_review_key.json` 单独保管，在评分完成前不要提供给评审者。第二版核验见 [verification_report_v2.md](verification_report_v2.md)。

## 第一版（保留基线）

该目录的脚本直接读取上一级 `real_attachments/`，不会修改原始数据。Python 3.11 及 `numpy`、`pandas`、`scipy`、`scikit-learn`、`matplotlib`、`pyarrow` 可运行。图表保存在 `outputs/figures/`，表格与中间数据保存在 `outputs/tables/`。随机种子固定为 `20260923`。

已删除的旧版《数据说明.pdf》中出现的方法推荐和预设结论未用于拟合或报告结论；现行《数据说明2.pdf》只用于结构、字段和数据性质核对。核查记录见 [数据说明更换核查.md](数据说明更换核查.md)。

## 运行顺序

```powershell
python problem1/quality_pipeline.py
python problem1/data_audit.py
python problem1/evaluate_quality.py
python problem1/sensitivity.py
python problem1/mixture_pipeline.py
python problem1/effect_bootstrap.py
python problem1/mixture_support.py
python problem1/quality_mixture_link.py
python problem1/build_report.py
```

也可运行 `python problem1/run_problem1.py` 完成以上全部步骤。首次处理全量压缩质量数据可能需要数分钟。

运行 `python problem1/verify_problem1.py` 可直接重读原始压缩质量文件和 A6–A11 CSV，复算样本数、质量分与 1M/60M/1B 检验误差；结果写入 [verification_report.md](verification_report.md)。

关键输出：`data_audit.csv`、`quality_indicator_rules.csv`、`quality_weights.csv`、`sample_quality_scores.parquet`、`domain_quality_scores.csv`、`conflict_summary.csv`、`mixture_model_comparison.csv`、`test_predictions.csv`、`extrapolation_results.csv`、`domain_effects.csv`、`interaction_effects.csv`、`problem1_report.md` 和 `problem1_summary.md`。

`sample_quality_scores.parquet` 含全部 272,505 条记录的得分；原始文本未全文复制到输出，只有 `manual_review_candidates.csv` 保存 40 个不超过 800 字符的摘要，供人工盲评。`quality_standardized.parquet` 保留 22 个统一方向的指标以便复核。A1 与 A2/A3 有重叠，任何跨文件总量计算应先按来源域和 ID 去重。

## 参考与限制

模型选择用 A4/A5 的五折交叉验证；A6–A11 仅作检验，A12–A15 仅作外推稳健性分析。1M 模型直接迁移到 60M/1B 时绝对 Loss 有尺度偏移。质量分没有独立人工标注真值；`manual_review_candidates.csv` 是待评样本，不能作为已完成的人工验证。具体假设见 `assumptions.md`。
