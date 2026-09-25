# 问题三独立复赛

在项目根目录执行：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python rebuild/problem3/run_decision_tournament.py
python rebuild/problem3/verify_tournament.py
python rebuild/problem3/build_paper_assets.py
```

中心 P2 Model C 和联合 bootstrap 可用 `--model-artifact`、`--bootstrap-parameters` 指定。若新 P2 模型不再采用八参数 Model C 公式，脚本会拒绝运行，须先更新 P3 模型接口和重新拟合联合 bootstrap。

先读 [`why_this_candidate.md`](why_this_candidate.md) 的预先评价协议，再读 [`problem3_decision_report.md`](problem3_decision_report.md) 的完整解释。主比较在 [`problem3_decision_tournament.csv`](problem3_decision_tournament.csv)，逐候选门槛审计在 [`decision_audit.csv`](decision_audit.csv)，输入哈希及情景样本 ID 在 [`run_metadata.json`](run_metadata.json)，复验在 [`verification_results.json`](verification_results.json)。

本轮**未改动** `problem3/`、`problem3_v2/` 和 P2 artifact。当前条件性 Champion 为 3× minimax regret；3× 界限本身未经外部验证。

论文材料由 [`build_paper_assets.py`](build_paper_assets.py) 自动读取最终复赛 CSV 与冻结 V2 数值生成：[问题三最终详解.md](问题三最终详解.md)、[`paper_tables/`](paper_tables/) 的 7 张表和 [`figures/`](figures/) 的 5 张 PNG/PDF 图。输入与产物 SHA256 在 [`paper_assets_manifest.json`](paper_assets_manifest.json)。
