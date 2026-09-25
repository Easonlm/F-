"""Independent raw-CSV and frozen-file verification for the selected P1 model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = ROOT / "problem1" / "outputs"
DATA = ROOT / "real_attachments" / "A_data_value" / "regmix_tables"


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_raw(prefix, scale):
    mix = pd.read_csv(DATA / f"{prefix}_mixture_{scale}.csv")
    loss = pd.read_csv(DATA / f"{prefix}_pile_loss_{scale}.csv")
    pcols = [c for c in mix if c.startswith("train_the_pile_")]
    ycols = [c for c in loss if c.startswith("metric/the_pile_")]
    both = mix.merge(loss, on="index", validate="one_to_one")
    p, y = both[pcols].to_numpy(float), both[ycols].to_numpy(float)
    p = p / p.sum(axis=1, keepdims=True)
    return p, y, pcols, ycols


def alr(p, eps, ref):
    q = p + eps
    q /= q.sum(axis=1, keepdims=True)
    z = np.log(q)
    return z[:, np.arange(p.shape[1]) != ref] - z[:, ref, None]


def main():
    manifest = json.loads((ROOT / "model_baseline_manifest.json").read_text(encoding="utf-8"))
    protected = ["problem1/problem1_v2.py", "problem1/problem1_v3.py",
                 "problem1/outputs/models/mixture_v2.joblib",
                 "problem1/outputs/models/mixture_v3_1m.joblib",
                 "problem1/outputs/tables/v2_selection.json",
                 "problem1/outputs/tables/v2_nested_cv_folds.csv",
                 "problem1/outputs/tables/v2_vs_v1_metrics.csv",
                 "problem1/outputs/tables/manual_validation_results.csv"]
    sha_audit = {name: dict(expected=manifest["files"][name]["sha256"], actual=digest(ROOT / name))
                 for name in protected}
    if any(v["expected"] != v["actual"] for v in sha_audit.values()):
        raise AssertionError("frozen Problem 1 file was changed")
    base = joblib.load(BASE / "models" / "mixture_v2.joblib")
    new = joblib.load(OUT / "selected_model.joblib")
    assert base["family"] == new["family"] == "poly_alr"
    assert base["pcols"] == new["pcols"] and base["ycols"] == new["ycols"]
    assert new["reference_index"] == 11 and new["selected_spec"] == {"alpha": .01, "l1_ratio": .7}
    stored = pd.read_csv(OUT / "metrics_by_scale.csv").set_index(["dataset", "family"])
    domain_rows = []
    checks = {}
    for i, (label, prefix, scale) in enumerate((("test_1m", "test", "1m"),
                                                ("test_60m", "test", "60m"),
                                                ("test_1B", "test", "1B"))):
        p, y, pc, yc = read_raw(prefix, scale)
        assert pc == new["pcols"] and yc == new["ycols"]
        old_pred = base["model"].predict(alr(p, base["eps"], base["reference_index"]))
        new_pred = new["model"].predict(alr(p, new["eps"], new["reference_index"]))
        for family, pred in (("v2_champion", old_pred), ("alr_elasticnet", new_pred)):
            value = float(np.sqrt(np.mean((pred-y)**2)))
            expected = float(stored.loc[(label, family), "rmse"])
            if not np.isclose(value, expected, atol=1e-10):
                raise AssertionError((label, family, value, expected))
            checks[f"{label}/{family}"] = value
        rng = np.random.default_rng(20260925+i)
        draws = rng.integers(0, len(y), size=(2000, len(y)))
        for j, col in enumerate(yc):
            err_old = (old_pred[:, j]-y[:, j])**2
            err_new = (new_pred[:, j]-y[:, j])**2
            gain = np.sqrt(err_old[draws].mean(axis=1)) - np.sqrt(err_new[draws].mean(axis=1))
            domain_rows.append(dict(dataset=label,
                                    validation_domain=col.replace("metric/the_pile_", "").replace("_val_loss", ""),
                                    v2_minus_selected_rmse=float(np.sqrt(err_old.mean())-np.sqrt(err_new.mean())),
                                    ci_low=float(np.quantile(gain, .025)),
                                    ci_high=float(np.quantile(gain, .975)),
                                    evidence_status="retrospective_descriptive_diagnostic"))
    pd.DataFrame(domain_rows).to_csv(OUT / "per_domain_paired_bootstrap.csv", index=False)
    output = dict(status="PASS", frozen_file_sha256=sha_audit,
                  selected_artifact_sha256=digest(OUT / "selected_model.joblib"),
                  directly_recomputed_raw_csv_rmse=checks,
                  validation_status="retrospective; no new blind test")
    (OUT / "verification_results.json").write_text(json.dumps(output, ensure_ascii=False, indent=2),
                                                      encoding="utf-8")
    print(json.dumps(dict(status=output["status"], checks=checks), indent=2))


if __name__ == "__main__":
    main()
