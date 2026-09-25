# 问题四复赛结果

全部数值来自 `rebuild/problem4/` 内重新运行的脚本及逐折/逐目标输出；见 `problem4_model_tournament.csv`。相同切分与原基线匹配的断言通过。已有历史验证集曾在前轮工作中被查看，因此这是回顾性重新挑战，而非新盲测。

| 模块 | 原 Champion | 新候选与主要结果 | 决定 |
|---|---|---|---|
| 能力指数 | 六基准原始均值 | 与 C8 详细任务 Spearman：原均值 0.9527、标准化稳健均值 0.9738、PCA 0.9750；C8 详细任务自身为 1.000 | **原均值保留**。C8 与榜单指标相关，相关性不构成独立能力真值；PCA 与标准化分数改变尺度解释。后两项只做敏感性。 |
| 严格历史前沿 | W1 季度 p95 | 末季 n=2；p90/p95/Top-3 与滚动 2/3 季方向随口径/端点变动。线性分位数回归 p95 的两次时间外 MAE 5.486，季度持平 3.947 | **保留 W1 p95 作描述，但 n<5 时不解释趋势**。时间外目标太少，拒绝宣称任何前沿估计器获胜。 |
| 历史规模/时间 | compute+time 结构式 | 同 32 行、13 组织，compute-only 组外 RMSE 7.052、结构式 7.103；新增时间系数 bootstrap 95% 区间 [-0.369, 2.534]，包含零 | 预测对照采用更简单 **compute-only**；结构式仅保留为情景分解，不再把时间项份额当实证贡献。|
| Loss→能力桥接 | 加权有界 logistic | 原 11.864；isotonic 11.451（增益 CI 跨零且端点平台）；自由 Loss+规模 10.560，但 Loss 斜率五折四次退化；六任务共享斜率部分汇聚 11.786，家族重抽样增益 95% 区间 [-0.043, 0.201] | **原 logistic 保留**。所有复杂候选均未同时满足家族外稳定性与可识别性。|
| 短期预测 | 直接时间趋势 | 同两次 3 个月目标 MAE 6.477；持平 3.947，半速 5.198 | 持平增为短期朴素参照。只有一个 n≥5 目标，故**不升级 12/24 月情景预测**。|

## 可复现命令

从项目根目录依次运行：

```powershell
python rebuild/problem4/run_frontier_bridge.py
python rebuild/problem4/run_bridge_structures.py
python rebuild/problem4/run_forecast_backtest.py
python rebuild/problem4/run_ability_multitask.py
python rebuild/problem4/run_quantile_frontier.py
python rebuild/problem4/select_models.py
```

## 证据边界

- C6 的高可比样本只有一个家族，不能给该层单独做家族外判断。11.864 分桥接误差远大于精细点数差异。
- W1 四季度 n 为 22、16、5、2；12/24 月滚动原点误差和区间覆盖率无目标可算。P4 的未来能力值仅是参数/算力/技术情景计算结果。
- 结构式模型内的时间剩余项混入模型选择、后训练、评测变化和数据口径。计算与时间 Shapley 项相加正确，不说明因果贡献已识别。
- 本轮可确认的改进主要是**删除不稳的实证解释**和增加持平风险参照；没有证据把新的复杂桥接或长期预测宣布为 Champion。
