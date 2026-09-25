"""New-structure C6 bridge experiments on the unchanged original family folds.

Run at project root: python optimization_20260924/round2/q4/run_new_models.py
Writes only to this directory. No candidate replaces the deployed P4 bridge.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import expit
from sklearn.model_selection import GroupKFold


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "problem4"))
from bridge_model import family, fit as original_fit, predict as original_predict  # noqa: E402

KINDS = ("original_logistic", "monotone_kink", "loss_plus_scale",
         "fixed_loss_plus_scale", "scale_only_diagnostic", "gompertz")


def fit_model(kind: str, x: np.ndarray, y: np.ndarray, w: np.ndarray, n: np.ndarray) -> dict:
    if kind == "original_logistic":
        return original_fit(x, y, w, "logistic")
    if kind == "monotone_kink":
        # One prechosen hinge; its location is learned from TRAINING losses only.
        knot = float(np.median(x))

        def raw(p, v):
            delta = v - knot
            return p[0] - np.exp(p[1]) * np.minimum(delta, 0) - np.exp(p[2]) * np.maximum(delta, 0)

        res = least_squares(lambda p: (100 * expit(raw(p, x)) - y) * np.sqrt(w),
                            x0=[-1.5, 0.0, 0.0], bounds=([-20, -8, -8], [20, 5, 5]),
                            max_nfev=5000)
        return {"kind": kind, "intercept": float(res.x[0]),
                "left_slope": float(np.exp(res.x[1])),
                "right_slope": float(np.exp(res.x[2])), "knot": knot}
    if kind == "loss_plus_scale":
        # N is in billions. Fixed 10B reference; the negative loss derivative
        # holds at every fixed N. The scale coefficient can have either sign.
        logn = np.log10(n / 10.0)

        def raw(p, xx, nn):
            return p[0] - np.exp(p[1]) * xx + p[2] * nn

        res = least_squares(lambda p: (100 * expit(raw(p, x, logn)) - y) * np.sqrt(w),
                            x0=[4.0, 1.0, 0.0],
                            bounds=([-20, -8, -10], [20, 5, 10]), max_nfev=5000)
        return {"kind": kind, "intercept": float(res.x[0]),
                "loss_slope": float(np.exp(res.x[1])), "scale_coefficient": float(res.x[2])}
    if kind == "fixed_loss_plus_scale":
        base = original_fit(x, y, w, "logistic")
        fixed_slope = -base["b"]
        logn = np.log10(n / 10.0)
        fun = lambda p: 100 * expit(p[0] - fixed_slope * x + p[1] * logn)
        res = least_squares(lambda p: (fun(p) - y) * np.sqrt(w),
                            x0=[base["a"], 0.0], bounds=([-20, -10], [20, 10]),
                            max_nfev=5000)
        return {"kind": kind, "intercept": float(res.x[0]),
                "loss_slope": float(fixed_slope), "scale_coefficient": float(res.x[1])}
    if kind == "scale_only_diagnostic":
        logn = np.log10(n / 10.0)
        fun = lambda p: 100 * expit(p[0] + p[1] * logn)
        res = least_squares(lambda p: (fun(p) - y) * np.sqrt(w),
                            x0=[-1.5, 1.0], bounds=([-20, -10], [20, 10]),
                            max_nfev=5000)
        return {"kind": kind, "intercept": float(res.x[0]),
                "scale_coefficient": float(res.x[1])}
    if kind == "gompertz":
        # Asymmetric bounded curve, unlike logistic/probit or a log-loss link.
        def pred(p, xx):
            u = np.clip(p[0] + np.exp(p[1]) * xx, -30, 15)
            return 100 * np.exp(-np.exp(u))

        res = least_squares(lambda p: (pred(p, x) - y) * np.sqrt(w),
                            x0=[-2.0, 0.0], bounds=([-20, -8], [20, 5]), max_nfev=5000)
        return {"kind": kind, "intercept": float(res.x[0]),
                "loss_slope": float(np.exp(res.x[1]))}
    raise ValueError(kind)


def predict(m: dict, x: np.ndarray, n: np.ndarray) -> np.ndarray:
    kind = m["kind"]
    if kind == "logistic":
        return original_predict(m, x)
    if kind == "monotone_kink":
        delta = x - m["knot"]
        raw = m["intercept"] - m["left_slope"] * np.minimum(delta, 0) - m["right_slope"] * np.maximum(delta, 0)
        return 100 * expit(raw)
    if kind in {"loss_plus_scale", "fixed_loss_plus_scale"}:
        raw = m["intercept"] - m["loss_slope"] * x + m["scale_coefficient"] * np.log10(n / 10.0)
        return 100 * expit(raw)
    if kind == "scale_only_diagnostic":
        raw = m["intercept"] + m["scale_coefficient"] * np.log10(n / 10.0)
        return 100 * expit(raw)
    if kind == "gompertz":
        raw = np.clip(m["intercept"] + m["loss_slope"] * x, -30, 15)
        return 100 * np.exp(-np.exp(raw))
    raise ValueError(kind)


def main() -> None:
    source = ROOT / "real_attachments/C_efficiency_evolution/loss_benchmark_bridge_expanded.csv"
    d = pd.read_csv(source).dropna(subset=["Val_Loss", "LB_Average", "N_params_B"]).reset_index(drop=True)
    if len(d) != 75 or not (d.N_params_B > 0).all():
        raise AssertionError("C6 positive N and complete-case coverage changed")
    d["family"] = d.Model.map(family)
    d["high"] = d.Loss_Comparability.str.startswith("High")
    x = d.Val_Loss.to_numpy(float)
    y = d.LB_Average.to_numpy(float)
    n = d.N_params_B.to_numpy(float)
    w = np.where(d.high.to_numpy(), 1.0, 0.35)
    groups = d.family.to_numpy()
    folds = list(GroupKFold(n_splits=5).split(x, y, groups))

    prediction_rows = []
    fold_rows = []
    parameter_rows = []
    for kind in KINDS:
        for fold, (tr, te) in enumerate(folds):
            model = fit_model(kind, x[tr], y[tr], w[tr], n[tr])
            pred = predict(model, x[te], n[te])
            error = pred - y[te]
            parameter_rows.append({"kind": kind, "fold": fold, **model,
                                   "train_loss_min": float(x[tr].min()),
                                   "train_loss_max": float(x[tr].max()),
                                   "train_n_min_B": float(n[tr].min()),
                                   "train_n_max_B": float(n[tr].max())})
            fold_rows.append({"kind": kind, "fold": fold, "n_test": len(te),
                              "rmse": float(np.sqrt(np.mean(error ** 2))),
                              "mae": float(np.mean(np.abs(error))),
                              "test_families": "|".join(sorted(set(groups[te]))),
                              "disjoint": set(groups[tr]).isdisjoint(groups[te]),
                              "loss_outside_train_n": int(((x[te] < x[tr].min()) | (x[te] > x[tr].max())).sum()),
                              "N_outside_train_n": int(((n[te] < n[tr].min()) | (n[te] > n[tr].max())).sum())})
            for pos, p in zip(te, pred):
                prediction_rows.append({"kind": kind, "fold": fold, "row": int(pos),
                                        "model": d.Model.iloc[pos], "family": groups[pos],
                                        "high": bool(d.high.iloc[pos]),
                                        "loss": x[pos], "N_params_B": n[pos], "actual": y[pos],
                                        "prediction": float(p), "error": float(p - y[pos])})

    oof = pd.DataFrame(prediction_rows)
    f = pd.DataFrame(fold_rows)
    p = pd.DataFrame(parameter_rows)
    if not f.disjoint.all() or not (oof.groupby("kind").row.nunique() == len(d)).all():
        raise AssertionError("Families overlap or held-out rows are missing")
    old = pd.read_csv(ROOT / "problem4/outputs/tables/loss_benchmark_bridge_cv.csv")
    old = old[(old.source == "C6") & (old.comparability == "weighted")
              & (old.target == "LB_Average") & (old.kind == "logistic")]
    base = f[f.kind == "original_logistic"]
    if not np.allclose(old.rmse.to_numpy(), base.rmse.to_numpy(), atol=1e-8):
        raise AssertionError("Original logistic folds not reproduced")
    oof.to_csv(HERE / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    f.to_csv(HERE / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    p.to_csv(HERE / "fold_parameters.csv", index=False, encoding="utf-8-sig")

    score_rows = []
    for kind, z in oof.groupby("kind"):
        ff = f[f.kind == kind]
        row = {"kind": kind, "n": len(z), "families": z.family.nunique(),
               "oof_rmse": float(np.sqrt(np.mean(z.error ** 2))),
               "oof_mae": float(np.mean(np.abs(z.error))),
               "fold_rmse_sd": float(ff.rmse.std(ddof=1)),
               "worst_fold_rmse": float(ff.rmse.max()),
               "high_rmse": float(np.sqrt(np.mean(z.loc[z.high, "error"] ** 2))),
               "medium_rmse": float(np.sqrt(np.mean(z.loc[~z.high, "error"] ** 2)))}
        score_rows.append(row)
    pd.DataFrame(score_rows).sort_values("oof_rmse").to_csv(HERE / "scorecard.csv", index=False, encoding="utf-8-sig")

    # Paired resampling of model families, holding folds and predictions fixed.
    wide = oof.pivot(index="row", columns="kind", values="error")
    uniq = np.unique(groups)
    rng = np.random.default_rng(20260924)
    comparisons = []
    for kind in KINDS[1:]:
        diffs = []
        for _ in range(5000):
            sample = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([np.flatnonzero(groups == g) for g in sample])
            baseline = np.sqrt(np.mean(wide.loc[idx, "original_logistic"].to_numpy() ** 2))
            alternative = np.sqrt(np.mean(wide.loc[idx, kind].to_numpy() ** 2))
            diffs.append(baseline - alternative)
        diffs = np.asarray(diffs)
        comparisons.append({"kind": kind,
                            "rmse_improvement": float(np.sqrt(np.mean(wide.original_logistic ** 2)) - np.sqrt(np.mean(wide[kind] ** 2))),
                            "family_bootstrap_low95": float(np.quantile(diffs, 0.025)),
                            "family_bootstrap_high95": float(np.quantile(diffs, 0.975)),
                            "bootstrap_fraction_positive": float(np.mean(diffs > 0))})
    pd.DataFrame(comparisons).to_csv(HERE / "paired_family_bootstrap.csv", index=False, encoding="utf-8-sig")

    full_models = {kind: fit_model(kind, x, y, w, n) for kind in KINDS}
    grid_x = np.linspace(x.min(), x.max(), 101)
    grid_n = np.full_like(grid_x, 10.0)
    for kind, model in full_models.items():
        grid_pred = predict(model, grid_x, grid_n)
        if not np.all(np.diff(grid_pred) <= 1e-8) or not np.all((grid_pred >= 0) & (grid_pred <= 100)):
            raise AssertionError(f"Bounded monotonicity failure: {kind}")
    (HERE / "full_models.json").write_text(json.dumps(full_models, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata = {"input": str(source.relative_to(ROOT)), "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "rows": len(d), "families": len(uniq), "high_rows": int(d.high.sum()),
                "high_families": int(d.loc[d.high, "family"].nunique()),
                "N_complete_rows": int(d.N_params_B.notna().sum()),
                "D_complete_rows": int(d.D_tokens_B.notna().sum()),
                "fixed_fold_baseline_reproduced": True, "all_candidates_bounded_monotone_at_fixed_N": True,
                "seed": 20260924}
    (HERE / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

