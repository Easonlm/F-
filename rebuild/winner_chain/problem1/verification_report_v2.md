# 问题一第二版核验

- test_1m/v1: raw-CSV RMSE 0.341455 matches report
- test_1m/v2: raw-CSV RMSE 0.270179 matches report
- test_60m/v1: raw-CSV RMSE 1.609271 matches report
- test_60m/v2: raw-CSV RMSE 1.595903 matches report
- test_1B/v1: raw-CSV RMSE 2.982004 matches report
- test_1B/v2: raw-CSV RMSE 2.702111 matches report
- est_10b/v1: raw-CSV RMSE 3.605364 matches report
- est_10b/v2: raw-CSV RMSE 3.621215 matches report
- est_70b/v1: raw-CSV RMSE 3.992691 matches report
- est_70b/v2: raw-CSV RMSE 4.008988 matches report
- 140 blinded review rows align with the separate key
- nested A4/A5 CV records five outer folds and v2 improvement

核验直接重新读取原始 RegMix CSV，使用保存的模型重算 RMSE；不把估算标签当作真实实验。
