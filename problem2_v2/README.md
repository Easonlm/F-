# 问题二第二轮研究

从项目根目录运行 `python problem2_v2/run_v2.py`。程序只读取附件 B 与 V1 参数，不写入 `problem2/`。输出包含诊断、模型比较、来源校准、质量情景、图表、报告和 `outputs/tables/verification_results.json`。

依赖与 V1 相同：numpy、pandas、scipy、scikit-learn、matplotlib、joblib。报告表格由脚本直接生成，不需要额外排版包。

`models.py:predict_recommended` 读取 `outputs/models/recommended_model.joblib`，提供完整 N-D-Q-p 论文主模型 C 的预测；Q 必须使用 B6 机制尺度。`models.py:predict_calibrated_ND` 使用 `outputs/models/model_B_source_calibrated.joblib`，只对获得目标来源校准标签后的 N-D 预测有效；Q 和 17 域 p 的跨源幅度没有被这些数据识别。V1 原文件保持不变。
