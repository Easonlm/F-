# 问题三 V1 与 V2 比较

| 项目 | V1 基线 | V2 |
|---|---|---|
| 三档预算 | 已求解 | 保留并做证据对照 |
| 结构点 | 71 点网格首次触发 | log 预算二分的活动集转移，另做 BIC 分段 |
| 参数不确定性 | G、κ、η 边际独立抽样 | B1/B6 簇 bootstrap，八参数条件联合传播 |
| p 支持度 | 最近邻基础指标 | ALR 5NN、PCA Mahalanobis、重构残差 |
| λp | 0.5–1.5 常数 | 0–1.5 常数及三类规模衰减情景 |
| 外推 | empirical/free | empirical/moderate 3×/free 及软罚 |
| 优化 | nominal | nominal 与 36 场景 minimax regret |
| KKT | 三档基础比值 | 全部主场景和 joint bootstrap 定量残差 |
| 高预算解析 | 数值轨迹 | 理论预算弹性及数值检验 |

V2 不替换 V1 的成本单位、Model C 或原始数据。V2 更适合论文主结论，因为每个强假设都有明确标签和对照。
