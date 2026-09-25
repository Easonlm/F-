# 问题二第二轮研究

从项目根目录运行 `python problem2_v2/run_v2.py`。程序只读取附件 B 与 V1 参数，不写入 `problem2/`。输出包含诊断、模型比较、来源校准、质量情景、图表、报告和 `outputs/tables/verification_results.json`。

依赖与 V1 相同：numpy、pandas、scipy、scikit-learn、matplotlib、joblib。报告表格由脚本直接生成，不需要额外排版包。

`models.py:predict_recommended` 读取 `outputs/models/recommended_model.joblib`，提供完整 N-D-Q-p 论文主模型 C 的预测；Q 必须使用 B6 机制尺度。`models.py:predict_calibrated_ND` 使用 `outputs/models/model_B_source_calibrated.joblib`，只对获得目标来源校准标签后的 N-D 预测有效；Q 和 17 域 p 的跨源幅度没有被这些数据识别。V1 原文件保持不变。

`p_scale_transfer_from_problem1_v2.csv` 记录最终 P1 V2 的回顾性跨规模幅度；原 `p_scale_transfer_from_problem1.csv` 保留 V1 证据。两表均不能将默认 `lambda_p=1` 识别为跨规模规律。

## 可选：B8 有标签来源校准

主 Model C 及供问题三、四调用的 `recommended_model.joblib` 不变。只有掌握目标 B8 来源的已标注 Loss 时，才运行 [`source_quality_calibration.py`](source_quality_calibration.py)。它要求显式给出互不重合的校准、评估 `experiment_id` 列表，且两侧不能共享 N–D 单元；全交互校准的每个 N 至少需要两个不同 D 单元。完整实验与失败边界见 [`source_quality_calibration_addendum.md`](source_quality_calibration_addendum.md)。

模式须显式区分：默认 `--mode full_interaction` 拟合 `Q、logN、logD、Q×logN、Q×logD` 残差项，需每个校准 N 至少两个 D 单元；`--mode q_only` 只拟合 `Q` 残差项，适用于仅一个 D 锚点的低标签设计。内层交叉验证只选择各模式的 Ridge 惩罚系数，**不会自动切换模式**。两种模式的留出行与标签预算不同，应在各自相同切分内同零样本及截距校准比较。

从项目根目录复现标准 200 标签实验：

```powershell
python problem2_v2/source_quality_calibration.py --base-model problem2_v2/outputs/models/recommended_model.joblib --b8-file real_attachments/B_scaling_laws/supplementary_NQ_experiment_large.csv --calibration-index optimization_20260924/round2/q2/b8_calibration_ids.csv --evaluation-index optimization_20260924/round2/q2/b8_evaluation_ids.csv --output-dir optimization_20260924/round2/q2/integrated
```

复现一单元/N 的 100 标签 `q_only` 实验：

```powershell
python problem2_v2/source_quality_calibration.py --mode q_only --base-model problem2_v2/outputs/models/recommended_model.joblib --b8-file real_attachments/B_scaling_laws/supplementary_NQ_experiment_large.csv --calibration-index optimization_20260924/round2/q2/b8_one_ND_cell_per_N_calibration_ids.csv --evaluation-index optimization_20260924/round2/q2/b8_one_ND_cell_per_N_evaluation_ids.csv --output-dir optimization_20260924/round2/q2/integrated_q_only_100
```

可选适配预测用 `source_quality_calibration.predict_b8_calibrated(calibrator, base_artifact_path, frame)`；输出只包含 N–D–Q，输入质量分必须在 B6/B8 的 `Q_score` 量尺上。这是使用 B8 目标标签的有监督预测校正，可能改变 Q→Loss 方向，不能解释为通用质量机制。B8 的 `extrapolated` 标签是附件估算值，不等同现实大模型实测。运行适配器不写入主 artifact，也不会自动接入问题三、四。
