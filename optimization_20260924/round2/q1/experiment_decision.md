# 问题一第二轮：两种新配比模型的训练内选型与跨规模检验

本目录保存独立实验，不覆盖问题一旧产物。脚本 `run_new_models.py` 从原始 RegMix CSV 读取 A4/A5、A6–A11 和估算 A12–A15，读取固定 V2 artifact 与 V2 五折选型表；输入哈希见 `input_sha256.json`。复现命令：

```powershell
python optimization_20260924/round2/q1/run_new_models.py
```

## 候选及实验门槛

1. **CLR 加性三次样条 + Ridge。** 让各领域对数比有平滑非线性主效应，减少 V2 全二阶 ALR 交互的自由度。A4/A5 内搜索零替换 0.25/0.5×最小正配比、4/6 结点、Ridge α=1/10/100，共 12 项。
2. **ILR→PCA 低秩二阶 Ridge。** 假设交互主要落在少数正交配比轴。A4/A5 内搜索零替换两档、主轴 4/8/12、α 三档，共 18 项。

使用与原 V2 完全相同的外层 5 折和每折内层 3 折，V2 用该折已选规格重新拟合，数值复算与旧报告一致；全训练最终规格用 2×5 折重复 CV 选择。A6–A15 不进入拟合或选参。既定测试标签曾被旧研究查看，故其对比是回顾性验证。估算标签单列。

## 核心结果

| 数据 | V2 RMSE | CLR 样条 RMSE | ILR 低秩 RMSE | CLR 样条的 13 域平均 Spearman（V2） |
| --- | ---: | ---: | ---: | ---: |
| A4/A5 嵌套 OOF | 0.3340 | **0.2522** | 0.3676 | **0.9590**（0.9244） |
| 1M 实测 | 0.2702 | **0.2082** | 0.2963 | **0.9646**（0.9452） |
| 60M 实测 | 1.5959 | **1.5614** | 1.5996 | **0.9605**（0.9408） |
| 1B 实测 | **2.7021** | 2.7357 | 2.7883 | 0.8769（0.8767） |
| 10B 估算 | **3.6212** | 3.6395 | 3.6328 | 0.5331（0.6328） |
| 70B 估算 | **4.0090** | 4.0268 | 4.0204 | 0.4445（0.5536） |

CLR 样条五个外层折均改善 V2。1M 为 11/13 域 RMSE 改善，60M 为 13/13，1B 仅 4/13。V2 RMSE 减 CLR 样条 RMSE 的按配方配对 bootstrap 95% 区间依次为：嵌套 OOF **+0.0818 [0.0653, 0.0973]**、1M **+0.0620 [0.0447, 0.0789]**、60M **+0.0345 [0.0076, 0.0612]**、1B **−0.0336 [−0.0613, −0.0073]**。这是给定模型与数据分布的条件区间，不能消除测试集被重复查看造成的偏乐观风险。

配方平均 Loss 的排序 Spearman：V2→CLR 样条在 1M 为 0.7153→0.9289，在 60M 为 0.6823→0.8932，在 1B 为 0.5667→0.6630；1B 排序增益 bootstrap 区间 [−0.0514, 0.2657] 跨零，且绝对 RMSE 退步。1M 与 60M 使用同一批测试配方，不能算两个独立配比分布检验；1B 最近训练配方的原比例空间中位 L2 距离为 0.217，1M/60M 为 0.177。CLR 坐标出训练范围的配方分别为 6.25% 和 2.73%，但这不足以确认 1B 失效的原因。10B/70B 是估算标签，配方与训练集重合，不能用作真实大规模验证。

## 决策与产物

- **保留 CLR 样条为 1M 条件候选。** 已接入 `problem1/problem1_v3.py` 并保存独立 artifact `problem1/outputs/models/mixture_v3_1m.joblib`；问题一的报告与核验详见 `problem1/problem1_v3_report.md`、`problem1/verification_report_v3.md`。
- **保留 V2 为跨问接口。** CLR 样条在 1B 绝对 Loss 上显著退步，跨规模共同偏差仍存在，不替换问题二、三引用的 V2 artifact。
- **拒绝 ILR 低秩二阶候选。** 嵌套 OOF、1M、1B 均退步。

本目录 `metrics_by_scale.csv`、`nested_outer_folds.csv`、`nested_inner_search.csv`、`full_train_search.csv`、`paired_bootstrap_gain.csv`、`rank_bootstrap_gain.csv`、`per_domain_metrics.csv`、`composition_support.csv` 和 `predictions_by_mixture.csv` 保留完整数值；`additive_clr_spline_candidate.joblib` 是独立实验版模型，正式问题一产物使用 `problem1/outputs/models/mixture_v3_1m.joblib`。
