# 问题二第二批模型实验：保留零样本 Model C，增加 B8 有标签来源适配

**运行日期：2026-09-24。** 本报告只比较新候选与当前问题二 V2，不改问题三、四的输入。两份项目优化建议仅作候选来源，取舍依赖本次相同切分的数值结果。详细方法、接口和论文表述已写入 [`problem2_v2_report.md`](../../../problem2_v2/problem2_v2_report.md)。

## 一、基线和划分

基线为当前 Model C：经典 $E+AN^{-\alpha}+BD^{-\beta}$（B1 拟合）加 $G(1-Q)^\kappa N^{-\eta_N}$（B6 拟合）。B1 在 8 个 N 组上逐组留出；B7 只用相对 B6 的 90 个精确新 $(N,D,Q)$ 点；B8 只用相对 B7 的 1480 个新点。所有 Q 候选在 B6 拟合，B7/B8 标签不参与零样本候选拟合。B2/B4/B5 是未经目标来源标签校准的原始跨源对照。

## 二、新零样本候选

| 新形式 | 关键结果 | 取舍 |
|---|---|---|
| $E+[(AN^{-\alpha})^r+(BD^{-\beta})^r]^{1/r}$ | $r=1.000007$；B1 留 N RMSE 0.00015118→0.00015248，B2 1.24308487→1.24308483；BIC -20729.31→-20722.25。 | 拒绝：退化回经典式，额外参数无用。 |
| $L_{ND}+G(1-Q)^\kappa N^{-\eta_N}(D/100)^{-\eta_D}$ | B6 留 N 0.052346→0.051061；B7 新点 0.042127→0.041167；B8 新点 1.341755→1.342263。B7 成对 N–D 簇 bootstrap 的 RMSE 差 95% 区间 [-0.00342,0.00146]。 | 保留为敏感性候选；B7 改善尚不稳定，不替换主模型。 |
| $L_{ND}+G(1-Q)^\kappa(ND/100)^{-\eta}$ | B6 留 N 0.052346→0.059144；B7 新点 0.042127→0.046356。 | 拒绝。 |

B6/B8 有 **160** 个完全相同的输入却全部对应不同 Loss，平均差（B8−B6）为 **−1.15361**。一条不读取来源的确定性函数无法同时拟合两套标签；两来源等权时这些重合点的最小可能 RMSE 为 **0.69768**。因此 B8 零样本失败是数据口径/机制冲突，不能靠增加统一式参数解决。

## 三、B8 来源校准：明确消耗目标来源标签

候选只对 B8 启用：固定主 Model C，再让 Ridge 拟合 `残差 ~ Q + logN + logD + Q×logN + Q×logD`。惩罚只在校准集内按 N–D 单元分组交叉验证选择。比较零样本主模型、同样校准标签预算的截距修正、Q-only、主效应和全交互修正。标签索引由 N/D/Q 与 B7 精确重合情况确定，未按 B8 Loss 挑选。保存的 [`b8_calibration_ids.csv`](b8_calibration_ids.csv) 和 [`b8_evaluation_ids.csv`](b8_evaluation_ids.csv) 分别有 200、1280 条且 N–D 单元完全不交叉。

| 划分 | 标签预算与测试 | 零样本 / 截距 RMSE | Q-only / 主效应 / 全交互 RMSE | 结果 |
|---|---|---:|---:|---|
| 每 N 校准两个 D 单元 | 200/1480 新点用于校准，1280 测试 | 1.3351 / 0.7517 | 0.1901 / 0.1636 / **0.1336** | 全交互胜出；相对截距的成对 120 单元 bootstrap ΔRMSE 95% 区间 [-0.6414,-0.5948]。 |
| N–D 单元和 Q 水平双重留出 | 校准 102 条、5 个 Q 水平；测试 728 条、7 个未见 Q 水平 | 1.2826 / 0.7804 | 0.1762 / 0.1525 / **0.1279** | 共享同一来源，仍能预测未见 Q 水平。 |
| `calibrated` → `extrapolated` N | 校准 56 条、9 个 N；测试 720 条、6 个新 N | 1.2767 / 0.6987 | 0.2409 / 0.2206 / **0.1858** | 标签是附件的估算值，不能当现实 20B–700B 实测。 |
| 每 N 仅校准一个 D 单元 | 校准 100 条、15 单元；测试 1380 条、135 新单元 | 1.3291 / 0.7608 | **0.2387** / 0.2391 / 0.7550 | 全交互失败；默认 `full_interaction` 模式拒绝，显式 `q_only` 可用。 |

在**同一个 100/1380 划分**中，`q_only` 相对截距校准的 RMSE 差为 -0.5221；按 135 个测试 N–D 单元成对重采样的 95% 区间 [-0.5512,-0.4935]。另将校准 Q 限为 5 个水平、测试限定其余 7 个未见水平，同时保持 N–D 单元不交叉：51 标签/777 测试的零样本、截距、`q_only` RMSE 分别为 **1.2745 / 0.8080 / 0.2207**，按 111 个测试单元成对重采样的 `q_only`−截距区间为 [-0.6115,-0.5621]。这支持 B8 内新 Q 水平预测；两个低标签协议与 200/1280 的全交互协议有不同标签成本及测试行，RMSE 不宜直接排名。

在 15 次逐 N 留出实验中，每次只从其余 14 个 N 各取两个 D 单元的 B8 标签训练，合并 1480 条测试的全交互 RMSE **0.1372**，截距 **0.7582**。标准 200 标签训练后，对未进入 B8 校准的 160 条 B6/B8 精确重合输入，B8 预测 RMSE **0.0652**；零样本主模型为 **1.3925**。这些数字支持“有标签的来源适配”，不支持“B1/B6 零样本已跨源泛化”。

将新 $Q\times N\times D$ 底座再叠加来源校准，标准留出仅由 **0.13357→0.13184**，在 `calibrated→extrapolated` 划分则 **0.18579→0.18751**。收益不相加，继续保留当前底座。

## 四、交付与限制

- 可选接口：[`source_quality_calibration.py`](../../../problem2_v2/source_quality_calibration.py)。它**必须**接收 B8 文件、校准 ID、测试 ID、主模型路径，并检查行和 N–D 单元不交叉。默认 `full_interaction` 要求每个校准 N 至少两个 D 单元；显式 `q_only` 允许一个 D 锚点。两者是 B8 附件标签的**有监督预测校正**，可能改变 Q→Loss 方向，不是通用质量机制，也不传入 P3/P4。独立 artifact 分别写入 [`integrated`](integrated/) 与 [`integrated_q_only_100`](integrated_q_only_100/)。
- 实验脚本：[`run_experiments.py`](run_experiments.py)、[`analyze_experiments.py`](analyze_experiments.py)、[`ablate_source.py`](ablate_source.py)、[`analyze_q_only.py`](analyze_q_only.py)。分数、成对重采样与逐行预测见 [`scorecard.csv`](scorecard.csv)、[`paired_effects.csv`](paired_effects.csv)、[`q_only_paired_effects.csv`](q_only_paired_effects.csv)、[`b8_source_ablation.csv`](b8_source_ablation.csv) 及同目录 CSV。
- 主 `recommended_model.joblib` SHA-256 前后均为 `2dbde20bb7da6bceace26bea2fecae9f175564a3c64d5084fe967ce9020b3de8`。原问题二验证 **13/13** 项通过；本轮接口安全测试 **6/6** 项通过。
- 本批候选已经查看 B2/B7/B8，留出只是在预先固定的行/组划分上的比较，不能称全新盲测。B6 是半合成数据，B8 包含 `calibrated` 与 `extrapolated` 标签；未来若有真实同尺度跨来源训练观测，应先复核校准效果。

运行顺序：

```powershell
python optimization_20260924/round2/q2/run_experiments.py
python optimization_20260924/round2/q2/analyze_experiments.py
python optimization_20260924/round2/q2/ablate_source.py
python problem2_v2/source_quality_calibration.py --base-model problem2_v2/outputs/models/recommended_model.joblib --b8-file real_attachments/B_scaling_laws/supplementary_NQ_experiment_large.csv --calibration-index optimization_20260924/round2/q2/b8_calibration_ids.csv --evaluation-index optimization_20260924/round2/q2/b8_evaluation_ids.csv --output-dir optimization_20260924/round2/q2/integrated
python problem2_v2/source_quality_calibration.py --mode q_only --base-model problem2_v2/outputs/models/recommended_model.joblib --b8-file real_attachments/B_scaling_laws/supplementary_NQ_experiment_large.csv --calibration-index optimization_20260924/round2/q2/b8_one_ND_cell_per_N_calibration_ids.csv --evaluation-index optimization_20260924/round2/q2/b8_one_ND_cell_per_N_evaluation_ids.csv --output-dir optimization_20260924/round2/q2/integrated_q_only_100
python problem2_v2/source_quality_calibration.py --mode q_only --base-model problem2_v2/outputs/models/recommended_model.joblib --b8-file real_attachments/B_scaling_laws/supplementary_NQ_experiment_large.csv --calibration-index optimization_20260924/round2/q2/b8_one_ND_cell_per_N_Q_disjoint_calibration_ids.csv --evaluation-index optimization_20260924/round2/q2/b8_one_ND_cell_per_N_Q_disjoint_evaluation_ids.csv --output-dir optimization_20260924/round2/q2/integrated_q_only_q_disjoint
python optimization_20260924/round2/q2/analyze_q_only.py
python optimization_20260924/round2/q2/test_source_quality_calibration.py
```

方法依据：[Hoffmann 等，NeurIPS 2022](https://proceedings.nips.cc/paper_files/paper/2022/hash/c1e2faff6f588870935f114ebe04a3e5-Abstract-Conference.html)支持保留可解释的 N–D 预算基线；[Geras 与 Sutton，ICML 2013](https://proceedings.mlr.press/v28/geras13.html)指出多来源样本的验证划分须考虑来源结构。两篇论文都不证明本项目某个候选有效；有效性判断来自上述成对留出实验。
