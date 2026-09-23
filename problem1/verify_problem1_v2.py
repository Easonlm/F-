"""Independent artifact checks against raw RegMix CSVs and A1 review keys."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from mixture_pipeline import OUT, TABLES, alr, pair
from problem1_v2 import transform


HERE = Path(__file__).resolve().parent


def run():
    bundle = joblib.load(OUT / "models" / "mixture_v2.joblib")
    v1 = joblib.load(OUT / "models" / "mixture_primary.joblib")
    meta = json.loads((TABLES / "mixture_metadata.json").read_text(encoding="utf-8"))
    selected = json.loads((TABLES / "v2_selection.json").read_text(encoding="utf-8"))["selected"]
    cv = pd.read_csv(TABLES / "v2_train_cv_search.csv")
    assert np.isclose(float(selected["cv_rmse_pooled"]), float(cv.cv_rmse_pooled.min()))
    assert len(cv) == 172
    summary = pd.read_csv(TABLES / "v2_vs_v1_metrics.csv")
    domain = pd.read_csv(TABLES / "v2_vs_v1_per_domain.csv")
    checks = []
    for label, prefix, scale in [("test_1m", "test", "1m"), ("test_60m", "test", "60m"),
                                 ("test_1B", "test", "1B"), ("est_10b", "est", "10b"),
                                 ("est_70b", "est", "70b")]:
        _, p, y, pcols, ycols = pair(prefix, scale)
        assert bundle["pcols"] == pcols and bundle["ycols"] == ycols
        preds = {
            "v1": v1.predict(alr(p, float(meta["alr_zero_pseudocount"]))),
            "v2": bundle["model"].predict(transform(p, bundle["family"], bundle["eps"], bundle["reference_index"])),
        }
        for version, pred in preds.items():
            target = summary.query("dataset == @label and version == @version").iloc[0]
            actual = float(np.sqrt(mean_squared_error(y, pred)))
            assert np.isclose(actual, target.rmse, atol=1e-10)
            assert len(y) == target.n_mixtures
            checks.append(f"{label}/{version}: raw-CSV RMSE {actual:.6f} matches report")
            if label == "test_1m":
                for j, col in enumerate(ycols):
                    name = col.replace("metric/the_pile_", "").replace("_val_loss", "")
                    expected = domain.query("dataset == @label and version == @version and validation_domain == @name").iloc[0].rmse
                    assert np.isclose(np.sqrt(mean_squared_error(y[:, j], pred[:, j])), expected, atol=1e-10)
    key = json.loads((TABLES / "v2_blind_review_key.json").read_text(encoding="utf-8"))
    items = json.loads((TABLES / "v2_blind_review_items.json").read_text(encoding="utf-8"))
    assert len(key) == len(items) == 140
    assert len({x["review_id"] for x in key}) == 140
    assert [x["review_id"] for x in key] == [x["review_id"] for x in items]
    assert "Q_baseline" not in items[0] and "domain" not in items[0]
    checks.append("140 blinded review rows align with the separate key")
    nested = pd.read_csv(TABLES / "v2_nested_cv_summary.csv").set_index("version")
    folds = pd.read_csv(TABLES / "v2_nested_cv_folds.csv")
    assert len(folds) == 5 and folds.outer_fold.nunique() == 5
    assert (folds.v2_outer_rmse < folds.v1_outer_rmse).all()
    assert nested.loc["v2", "nested_oof_rmse"] < nested.loc["v1", "nested_oof_rmse"]
    checks.append("nested A4/A5 CV records five outer folds and v2 improvement")
    (HERE / "verification_report_v2.md").write_text(
        "# 问题一第二版核验\n\n" + "\n".join(f"- {s}" for s in checks) + "\n\n"
        "核验直接重新读取原始 RegMix CSV，使用保存的模型重算 RMSE；不把估算标签当作真实实验。\n",
        encoding="utf-8")
    print(f"Passed {len(checks)} checks")


if __name__ == "__main__":
    run()
