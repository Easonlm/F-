# 文献驱动候选模型库

| 候选 | 形式 / 自由参数 | 依据 | 本题识别与目标 | 主要风险 |
|---|---|---|---|---|
| M1 经典 | (E+AN^{-a}+BD^{-b})，5 | Hoffmann et al. 2022 | B1 充分识别；V1 基线 | 跨测评集 E 未必相同 |
| M2 N×D | M1 + (H N^{-a}D^{-b})，6 | 受控非加性检验 | B1 可检验小交互 | 在外推区破坏单调、弱识别 |
| M3 compute-ratio | (E+K(6ND)^{-c}(N/D)^{ct})，4 | Kaplan/Hoffmann 的算力讨论 | 比较更紧的共同指数约束 | 压缩过度，牺牲 N/D 独立效应 |
| M4 broken-N | (E+AN^{-a}[1+(N/N_c)^2]^{-\delta/2}+BD^{-b})，7 | Caballero et al. 2023 | B1 剖面检验阈值，测试小→大 N | 转折可能被同源噪声“拟合” |
| M5 source intercept | (L_s=f_{ND}+b_s) | 损失口径差异假设 | 少量该源锚点可校准 | 用全源标签会泄漏；严格 20/80 切分 |
| M6 source affine | (L_s=a_s+c_sf_{ND}) | 斜率/口径差异诊断 | 同上 | 校准点不足时不稳定 |
| Q1 加性 | (G(1-Q)^\kappa)，2 | V1 | B6/B7 支持 | 假设与 D 独立 |
| Q2 幂有效 token | (D_{eff}=DQ^\gamma)，1 | Chang et al. 2024 | B6 可识别 | 无重复率/多样性原指标 |
| Q3 指数有效 token | (D_{eff}=D e^{\gamma(Q-1)})，1 | 有效数据量视角 | B6 可识别 | B8 方向反转无法解释 |
| Q4 饱和质量项 | (G(e^{-tQ}-e^{-t})/(1-e^{-t}))，2 | 边际收益递减候选 | B6/B7 检验 | 低 Q 外推形状敏感 |
| Q5 质量×D | (G(1-Q)^\kappa(D/100)^{-\eta})，3 | Goyal et al. 2024 启示 | 检验 Q 收益是否随 token 量变 | 额外参数、B8 机制不一致 |
| Q6 质量×N | (G(1-Q)^\kappa N^{-\eta_N})，3 | 跨规模质量价值探索 | B6 可拟合，B7 非重合检验 | 半合成数据，N 外推脆弱 |
| P1 additive | (L=L_{NDQ}+\lambda\Delta_p) | V1；Ye et al. 2025 | 问题一可给 Δp | λ 跨规模未识别 |
| P2 multiplicative | (L-E=(L_{NDQ}-E)h(p)) | 嵌套配比候选 | 仅情景比较 | 缺联合 N,D,p 实验 |
| P3 scale-varying | (\lambda_p(N)\Delta_p) | Ye et al. 2025 启示 | 只做敏感性 | 不允许伪造精确指数 |

预设评估顺序：B1 小→大 N 与早→晚 D、B2 raw、B4/B5 raw、B7 新点；B8 作为最终机制诊断。B2 被用于候选筛选后，其数值不能再写作全新独立确认。复杂模型需有 BIC、外推和跨源多数指标共同支持。
