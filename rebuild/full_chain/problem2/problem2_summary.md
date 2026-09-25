# 问题二摘要

- 经典律：E=1.6898, A=0.35398, α=0.339977, B=1.24031, β=0.279878（N、D 为 billion）。B1 模型留一 CV RMSE=0.0002。
- 广义律：L=E+A N^{-\alpha}+B D^{-\beta}+G(1-Q)^\kappa+\lambda_p\Delta_p(p)。质量模型 additive，E=1.6898, A=0.35398, alpha=0.339977, B=1.24031, beta=0.279878, G=0.362175, kappa=0.990735；λ_p=1 是敏感性基准。
- 外部 raw RMSE：B2 1.2431，B3 插值 0.0038，B4 0.2927，B5 0.1976。
- N=7B/D=300B/Q_B=0.6 的损失弹性：N: -0.0274 (可约损失 -0.1071), D: -0.0310 (可约损失 -0.1213), Q: -0.0957 (可约损失 -0.3743)。
- 同情景 Q_B+0.1 等效参数 13.4B，倍数 1.92；Q_A 不可直接等同 Q_B。
- p：复用问题一第二版 17 域模型和中心化效应；λ_p 不可由 B 识别。较稳健替代：pile_cc→ubuntu_irc: -0.0155; pile_cc→dm_mathematics: -0.0108; pile_cc→stackexchange: -0.0019; pile_cc→europarl: +0.0089; pile_cc→nih_exporter: +0.0255。尚无充分的互补证据。
- 100B+：对 B10 **估算标签** 的 RMSE 0.0011，非真实外部验证。
- 稳健性：200 次模型簇 bootstrap；质量形式和拟合方式对照见表。最重要限制是 Q 的半合成机制、跨源标尺无锚点、B4/B5 损失口径和 λ_p 未识别。
- 问题三读取 `outputs/tables/problem2_to_problem3.json`、`outputs/models/final_generalized_scaling.joblib` 和 `generalized_scaling.py` 中的 `predict_generalized`、`equivalent_parameters`。
