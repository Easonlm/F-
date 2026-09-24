# 假设与口径

1. 主能力为六项完整 Benchmark 等权均值，主时间为 Submission Date。
2. Open-W1 要求 C4 明确 `Open model weights?=Yes`，模型名规范化一致、Hugging Face 发布方一致、参数差不超过 35%、发布日期不能晚于提交超过 30 天、Epoch Confidence 不得为 Speculative。同一模型只保留最后一次提交用于前沿。
3. Open-W2 还要求榜单 License 标签可归类为 permissive 或 research-only；这只是分析筛选，不构成法律判断。
4. C2 的宽口径开放标记可能继承基础模型，因此仅作敏感性。
5. C8 同模型目录取文件名时间最新的可解析 JSON；损坏文件逐一记录。叶任务固定 metric，不混用不同 metric；逐任务 percentile 后等权，最低覆盖 50% 且至少三任务族。
6. C4 训练算力增长趋势使用 Language、公开权重、Confident/Likely、reported FLOP、发布于预测原点之前的模型；最后不完整季度排除。未来算力增长率为 2023–2024 季度 p90 的 log-linear 斜率。
7. 贡献是固定历史模型与季度端点的对称 Shapley 反事实分解；剩余时间项不等于具体技术的因果效应。由于模型方向与观测前沿冲突，它只作探索性。
8. 未来算力倍率 1/0.5/0.25、技术趋势倍率 1/0.5/0 是情景假设，不是估计概率。条件 bootstrap 区间不能当作历史覆盖率已验证区间。
9. 问题三主机制使用 E1 3× N/D 边界、Q0=0.6、context=4096、exponential quality cost，P0 基准配比；P1 观测支持配比以 lambda=0.25 作敏感性。E1 3× 和 lambda 均不是联合识别结果。
