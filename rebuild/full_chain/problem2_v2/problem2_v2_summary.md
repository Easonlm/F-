# 问题二第二轮研究摘要

- V1 保持原样。B1 近乎确定性，LOMO 检验的是同一 Pythia 生成规律；B2 raw RMSE=1.2431。
- B8 质量方向与 B6 相反，且同一 N,D,Q 有 160 个 Loss 冲突点；任何保持 Q 越高 Loss 越低的单一规律都无法同时准确拟合两者。
- 候选预测型：V2 quality_model + prespecified source affine calibration (20% source labels)；论文主模型：classic + quality_model。具体 raw / 外推对照见 `outputs/tables/v1_v2_comparison.csv`。
- B2 的来源校准须先提供少量该来源 Loss，不能算零样本泛化。B4/B5 raw 指标单列。
- Q×N 质量项在 B7 新点和 B8 诊断上均优于加性 Q；新模型 Q_B:0.6→0.7、D=300B 的等价参数倍数：7B 1.414×，70B 1.614×，120B 1.672×。B8 仍存在反向机制，不能视为统一实测结论。
- p 的规模系数与 Q_A→Q_B 映射没有共同锚点，仅给情景，不给精确估计。
