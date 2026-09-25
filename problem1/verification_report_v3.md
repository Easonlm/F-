# 问题一 V3 条件候选核验

- A4/A5 的 12 候选中仅选最小 CV RMSE；5 个外层折每折只在训练部分做内层选择，V3 在 5/5 折优于 V2。
- V2 artifact SHA-256 与选型时记录一致；V3 artifact 独立保存。
- test_1m: raw CSV and saved artifact RMSE agree (0.208173); all 13 domain RMSEs agree
- test_60m: raw CSV and saved artifact RMSE agree (1.561406); all 13 domain RMSEs agree
- test_1B: raw CSV and saved artifact RMSE agree (2.735706); all 13 domain RMSEs agree
- est_10b: raw CSV and saved artifact RMSE agree (3.639506); all 13 domain RMSEs agree
- est_70b: raw CSV and saved artifact RMSE agree (4.026819); all 13 domain RMSEs agree

既定测试集已经被项目此前查看，核验确认产物一致，不等于全新盲测；10B/70B 标签为估算。
