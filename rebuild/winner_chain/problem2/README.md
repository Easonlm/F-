# 问题二复现

在项目根目录运行 `python problem2/run_problem2.py`。程序只读取 `problem1/outputs/` 和附件 B，全部新结果写入 `problem2/outputs/`；最后自动调用 `verify_problem2.py`。Python 依赖：numpy、pandas、scipy、scikit-learn、matplotlib、joblib。

`generalized_scaling.py` 提供可复用的 `predict_ndq`、`predict_generalized`、`derivatives` 和 `equivalent_parameters`。N 和 D 的单位是 billion。最终模型与问题三接口分别保存在 `outputs/models/final_generalized_scaling.joblib` 和 `outputs/tables/problem2_to_problem3.json`。

报告：`problem2_report.md`；摘要：`problem2_summary.md`；核验：`outputs/tables/verification_results.json`。
