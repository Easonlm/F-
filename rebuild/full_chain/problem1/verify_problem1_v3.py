"""Recheck V3 artifact against raw RegMix CSVs and its locked selection records."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from mixture_pipeline import OUT, TABLES, pair
from problem1_v3 import predict_v3


HERE = Path(__file__).resolve().parent


def run():
    artifact = joblib.load(OUT / "models" / "mixture_v3_1m.joblib")
    selection = json.loads((TABLES / "v3_selection.json").read_text(encoding="utf-8"))
    cv = pd.read_csv(TABLES / "v3_train_cv_search.csv")
    assert len(cv) == 12 and int(cv.selected.sum()) == 1
    assert np.isclose(float(selection["cv_rmse"]), float(cv.cv_rmse.min()), atol=1e-12)
    selected_row = cv[cv.selected].iloc[0]
    for key in ("alpha", "n_knots", "eps_mult"):
        assert np.isclose(float(artifact["selected_spec"][key]), float(selected_row[key]))
    v2_hash = hashlib.sha256((OUT / "models" / "mixture_v2.joblib").read_bytes()).hexdigest()
    assert v2_hash == artifact["upstream_v2_sha256"] == selection["upstream_v2_sha256"]
    _, train_p, train_y, pcols, ycols = pair("train", "1m")
    assert len(train_y) == 512 and artifact["pcols"] == pcols and artifact["ycols"] == ycols
    assert np.all(np.isfinite(predict_v3(artifact, train_p)))

    folds = pd.read_csv(TABLES / "v3_nested_cv_folds.csv")
    nested_search = pd.read_csv(TABLES / "v3_nested_inner_search.csv")
    assert len(folds) == 5 and folds.outer_fold.nunique() == 5
    assert len(nested_search) == 5 * 12 and nested_search.groupby("outer_fold").selected.sum().eq(1).all()
    assert folds.n.sum() == 512 and (folds.v3_rmse < folds.v2_rmse).all()

    metrics = pd.read_csv(TABLES / "v3_vs_v2_metrics.csv")
    per_domain = pd.read_csv(TABLES / "v3_vs_v2_per_domain.csv")
    checks = []
    for label, prefix, scale in (("test_1m", "test", "1m"),
                                 ("test_60m", "test", "60m"),
                                 ("test_1B", "test", "1B"),
                                 ("est_10b", "est", "10b"),
                                 ("est_70b", "est", "70b")):
        _, p, y, pc, yc = pair(prefix, scale)
        assert pc == pcols and yc == ycols
        pred = predict_v3(artifact, p)
        expected = metrics.query("dataset == @label and version == 'v3_additive_clr_spline'").iloc[0]
        actual = float(np.sqrt(mean_squared_error(y, pred)))
        assert np.isclose(actual, expected.rmse, atol=1e-10)
        assert len(y) == expected.n_mixtures
        assert expected.status == ("observed" if prefix == "test" else "estimated_label")
        for j, col in enumerate(ycols):
            name = col.replace("metric/the_pile_", "").replace("_val_loss", "")
            reported = per_domain.query("dataset == @label and version == 'v3_additive_clr_spline' and validation_domain == @name").iloc[0]
            assert np.isclose(np.sqrt(mean_squared_error(y[:, j], pred[:, j])), reported.rmse, atol=1e-10)
        checks.append(f"{label}: raw CSV and saved artifact RMSE agree ({actual:.6f}); all 13 domain RMSEs agree")
    report = "# 问题一 V3 条件候选核验\n\n"
    report += "- A4/A5 的 12 候选中仅选最小 CV RMSE；5 个外层折每折只在训练部分做内层选择，V3 在 5/5 折优于 V2。\n"
    report += "- V2 artifact SHA-256 与选型时记录一致；V3 artifact 独立保存。\n"
    report += "\n".join(f"- {line}" for line in checks) + "\n\n"
    report += "既定测试集已经被项目此前查看，核验确认产物一致，不等于全新盲测；10B/70B 标签为估算。\n"
    (HERE / "verification_report_v3.md").write_text(report, encoding="utf-8")
    print(f"Passed selection, nested CV, hash, and {len(checks)} raw-dataset artifact checks")


if __name__ == "__main__":
    run()
