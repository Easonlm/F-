"""Independent, reproducible Problem 4 frontier and bridge sensitivity experiments.

Run from the project root: python optimization_20260924/q4/run_experiments.py
Only files beside this script are written. Existing problem4 outputs are inputs.
"""
from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import expit, ndtr
from sklearn.model_selection import GroupKFold
import scipy
import sklearn


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "problem4"))
from bridge_model import family, fit as original_fit, predict as original_predict  # noqa: E402
from history_forecast import design as historical_design, fit_model as historical_fit, score as historical_score  # noqa: E402

QUARTERS = ["2024Q2", "2024Q3", "2024Q4", "2025Q1"]
METRICS = ("p90", "p95", "top3")
MIN_N = (1, 3, 5, 10)
WINDOWS = (1, 2, 3)


def metric(v: pd.Series, name: str) -> float:
    if name == "p90":
        return float(v.quantile(0.9))
    if name == "p95":
        return float(v.quantile(0.95))
    return float(v.nlargest(min(3, len(v))).mean())


def frontier_experiment() -> dict:
    source = ROOT / "problem4/outputs/tables/analysis_dataset.csv"
    a = pd.read_csv(source, low_memory=False)
    a["submission_date"] = pd.to_datetime(a.submission_date, errors="coerce")
    for c in ("complete", "open_w1", "open_w2"):
        a[c] = a[c].astype(str).str.lower().eq("true")
    a["ability"] = pd.to_numeric(a.ability, errors="coerce")
    a = a[a.complete & a.submission_date.notna() & a.ability.notna()].copy()
    a = a.sort_values(["submission_date", "ability"]).drop_duplicates("Model", keep="last")
    a["quarter"] = a.submission_date.dt.to_period("Q").astype(str)
    strict_cutoff = a.loc[a.open_w1, "submission_date"].max()

    subsets = {
        "strict_W1": a[a.open_w1 & a.submission_date.le(strict_cutoff)],
        "license_W2": a[a.open_w2 & a.submission_date.le(strict_cutoff)],
        "inherited_C2_aligned_cutoff": a[a.c2_open_weights.eq("Yes") & a.submission_date.le(strict_cutoff)],
        "inherited_C2_full_legacy_cutoff": a[a.c2_open_weights.eq("Yes")],
    }
    rows = []
    for subset, d in subsets.items():
        for window in WINDOWS:
            for qi, end in enumerate(QUARTERS):
                # A rolling window is reported only after all its calendar quarters exist.
                if qi + 1 < window:
                    continue
                included = QUARTERS[qi - window + 1 : qi + 1]
                values = d.loc[d.quarter.isin(included), "ability"]
                for name in METRICS:
                    rows.append(
                        dict(
                            subset=subset, window_quarters=window, end_quarter=end,
                            included_quarters="|".join(included), metric=name,
                            n=len(values), value=metric(values, name) if len(values) else np.nan,
                            n_min_1=len(values) >= 1, n_min_3=len(values) >= 3,
                            n_min_5=len(values) >= 5, n_min_10=len(values) >= 10,
                        )
                    )
    f = pd.DataFrame(rows)
    existing = pd.read_csv(ROOT / "problem4/outputs/tables/quarterly_frontier.csv")
    old = existing[(existing.subset == "open_w1") & existing.quarter.isin(QUARTERS)]
    fresh = f[(f.subset == "strict_W1") & (f.window_quarters == 1) & (f.metric == "p95")]
    merged = fresh.merge(old, left_on="end_quarter", right_on="quarter", validate="one_to_one")
    if len(merged) != 4 or not np.allclose(merged.value, merged.p95_ability, atol=1e-10):
        raise AssertionError("Independent W1 p95 does not reproduce existing quarterly baseline")
    f.to_csv(HERE / "frontier_variants.csv", index=False, encoding="utf-8-sig")

    changes = []
    for (subset, window, name), z in f.groupby(["subset", "window_quarters", "metric"]):
        z = z.sort_values("end_quarter")
        for n_min in MIN_N:
            eligible = z[z.n >= n_min]
            if len(eligible) < 2:
                continue
            first, last = eligible.iloc[0], eligible.iloc[-1]
            changes.append(
                dict(
                    subset=subset, window_quarters=window, metric=name, n_min=n_min,
                    first_quarter=first.end_quarter, last_quarter=last.end_quarter,
                    n_first=int(first.n), n_last=int(last.n),
                    first_value=first.value, last_value=last.value,
                    change=last.value - first.value,
                    same_span_as_strict_original=(first.end_quarter == "2024Q2" and last.end_quarter == "2025Q1"),
                )
            )
    pd.DataFrame(changes).to_csv(HERE / "frontier_changes.csv", index=False, encoding="utf-8-sig")

    # Persistence forecasts made at the end of each origin quarter. The actual
    # target uses ONLY models newly submitted in the next quarter. Thus rolling
    # windows do not get the same observations on both sides of an error.
    backtests = []
    for subset in subsets:
        one = f[(f.subset == subset) & (f.window_quarters == 1)]
        for window in WINDOWS:
            for name in METRICS:
                z = f[(f.subset == subset) & (f.window_quarters == window) & (f.metric == name)]
                for qi in range(len(QUARTERS) - 1):
                    origin = z[z.end_quarter == QUARTERS[qi]]
                    target = one[(one.metric == name) & (one.end_quarter == QUARTERS[qi + 1])]
                    if len(origin) != 1 or len(target) != 1:
                        continue
                    origin, target = origin.iloc[0], target.iloc[0]
                    if not np.isfinite(origin.value) or not np.isfinite(target.value):
                        continue
                    for n_min in MIN_N:
                        if origin.n < n_min or target.n < n_min:
                            continue
                        backtests.append(
                            dict(
                                subset=subset, window_quarters=window, metric=name,
                                n_min=n_min, origin_quarter=origin.end_quarter,
                                target_quarter=target.end_quarter, origin_n=int(origin.n),
                                target_n=int(target.n), prediction=origin.value,
                                actual=target.value, error=origin.value - target.value,
                                abs_error=abs(origin.value - target.value),
                            )
                        )
    bt = pd.DataFrame(backtests)
    bt.to_csv(HERE / "frontier_prequential_errors.csv", index=False, encoding="utf-8-sig")
    summary = (
        bt.groupby(["subset", "window_quarters", "metric", "n_min"])
        .agg(targets=("abs_error", "size"), mae=("abs_error", "mean"),
             max_abs_error=("abs_error", "max"))
        .reset_index()
    )
    summary.to_csv(HERE / "frontier_prequential_summary.csv", index=False, encoding="utf-8-sig")
    return {"input": str(source.relative_to(ROOT)), "strict_cutoff": str(strict_cutoff.date()),
            "strict_complete_models": len(subsets["strict_W1"]), "eligible_quarters": QUARTERS,
            "baseline_p95_reproduced": True}


def new_bridge_fit(x: np.ndarray, y: np.ndarray, w: np.ndarray, kind: str) -> dict:
    x, y, w = np.asarray(x, float), np.asarray(y, float), np.asarray(w, float)
    if kind == "clipped_linear":
        m = original_fit(x, y, w, "linear")
        return {"kind": kind, "a": m["a"], "b": m["b"]}
    if kind == "isotonic":
        return original_fit(x, y, w, "isotonic")
    if kind == "logistic":
        return original_fit(x, y, w, "logistic")
    if kind == "probit":
        fun = lambda p: 100 * ndtr(p[0] - np.exp(p[1]) * x)
        res = least_squares(lambda p: (fun(p) - y) * np.sqrt(w),
                            [3, 0], bounds=([-20, -8], [20, 5]), max_nfev=3000)
        return {"kind": kind, "a": float(res.x[0]), "b": float(-np.exp(res.x[1]))}
    if kind == "log_loss_logistic":
        if np.any(x <= 0):
            raise ValueError("Positive loss required")
        fun = lambda p: 100 * expit(p[0] - np.exp(p[1]) * np.log(x))
        res = least_squares(lambda p: (fun(p) - y) * np.sqrt(w),
                            [2, 1], bounds=([-20, -8], [20, 5]), max_nfev=3000)
        return {"kind": kind, "a": float(res.x[0]), "b": float(-np.exp(res.x[1]))}
    raise ValueError(kind)


def bridge_predict(m: dict, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    kind = m["kind"]
    if kind == "clipped_linear":
        return np.clip(m["a"] + m["b"] * x, 0, 100)
    if kind in {"logistic", "isotonic"}:
        return original_predict(m, x)
    if kind == "probit":
        return 100 * ndtr(m["a"] + m["b"] * x)
    if kind == "log_loss_logistic":
        return 100 * expit(m["a"] + m["b"] * np.log(np.maximum(x, 1e-12)))
    raise ValueError(kind)


def bridge_experiment() -> dict:
    source = ROOT / "real_attachments/C_efficiency_evolution/loss_benchmark_bridge_expanded.csv"
    d = pd.read_csv(source).dropna(subset=["Val_Loss", "LB_Average"]).copy()
    d["group"] = d.Model.map(family)
    d["high"] = d.Loss_Comparability.str.startswith("High")
    x = d.Val_Loss.to_numpy(float)
    y = d.LB_Average.to_numpy(float)
    w = np.where(d.high.to_numpy(), 1.0, 0.35)
    g = d.group.to_numpy()
    kinds = ("logistic", "clipped_linear", "isotonic", "probit", "log_loss_logistic")
    folds = list(GroupKFold(n_splits=min(5, d.group.nunique())).split(x, y, g))
    predictions, fold_rows = [], []
    for kind in kinds:
        for fold, (tr, te) in enumerate(folds):
            m = new_bridge_fit(x[tr], y[tr], w[tr], kind)
            yp = bridge_predict(m, x[te])
            err = yp - y[te]
            fold_rows.append(
                dict(kind=kind, fold=fold, n_test=len(te),
                     test_families="|".join(sorted(set(g[te]))),
                     rmse=float(np.sqrt(np.mean(err ** 2))),
                     mae=float(np.mean(np.abs(err))),
                     high_n=int(d.high.iloc[te].sum()),
                     train_families=len(set(g[tr])), test_family_count=len(set(g[te])),
                     family_disjoint=set(g[tr]).isdisjoint(g[te]))
            )
            for i, pred in zip(te, yp):
                predictions.append(
                    dict(kind=kind, fold=fold, row=int(i), model=d.Model.iloc[i],
                         family=g[i], high_comparability=bool(d.high.iloc[i]),
                         loss=x[i], actual=y[i], prediction=float(pred), error=float(pred - y[i]))
                )
    pred = pd.DataFrame(predictions)
    fd = pd.DataFrame(fold_rows)
    if not fd.family_disjoint.all() or not (pred.groupby("kind").row.nunique() == len(d)).all():
        raise AssertionError("Bridge family split or held-out row coverage invalid")
    pred.to_csv(HERE / "bridge_oof_predictions.csv", index=False, encoding="utf-8-sig")
    fd.to_csv(HERE / "bridge_folds.csv", index=False, encoding="utf-8-sig")

    result = []
    for kind, z in pred.groupby("kind"):
        fold_z = fd[fd.kind == kind]
        row = dict(kind=kind, n=len(z), families=z.family.nunique(),
                   oof_rmse=float(np.sqrt(np.mean(z.error ** 2))),
                   oof_mae=float(np.mean(np.abs(z.error))),
                   fold_rmse_mean=float(fold_z.rmse.mean()),
                   fold_rmse_sd=float(fold_z.rmse.std(ddof=1)),
                   fold_rmse_min=float(fold_z.rmse.min()),
                   fold_rmse_max=float(fold_z.rmse.max()))
        for label, zz in [("high", z[z.high_comparability]),
                          ("medium", z[~z.high_comparability])]:
            row[label + "_n"] = len(zz)
            row[label + "_rmse"] = float(np.sqrt(np.mean(zz.error ** 2))) if len(zz) else np.nan
        result.append(row)
    scores = pd.DataFrame(result).sort_values("oof_rmse")
    old_folds = pd.read_csv(ROOT / "problem4/outputs/tables/loss_benchmark_bridge_cv.csv")
    old_logistic = old_folds[(old_folds.source == "C6") & (old_folds.comparability == "weighted")
                             & (old_folds.target == "LB_Average") & (old_folds.kind == "logistic")]
    new_logistic = fd[fd.kind == "logistic"]
    if len(old_logistic) != 5 or not np.allclose(old_logistic.rmse.to_numpy(), new_logistic.rmse.to_numpy(), atol=1e-8):
        raise AssertionError("Bridge logistic did not reproduce original fold RMSE")
    scores.to_csv(HERE / "bridge_scorecard.csv", index=False, encoding="utf-8-sig")

    # Paired family bootstrap: all candidates share exactly the same folds and
    # test rows, and one model family is the resampling unit.
    wide = pred.pivot(index="row", columns="kind", values="error")
    fam = d.group.to_numpy()
    uniq = np.unique(fam)
    rng = np.random.default_rng(20260924)
    deltas = []
    for kind in kinds:
        if kind == "logistic":
            continue
        pair = []
        for _ in range(3000):
            sampled = rng.choice(uniq, len(uniq), replace=True)
            idx = np.concatenate([np.flatnonzero(fam == ff) for ff in sampled])
            base_rmse = np.sqrt(np.mean(wide.loc[idx, "logistic"].to_numpy() ** 2))
            alt_rmse = np.sqrt(np.mean(wide.loc[idx, kind].to_numpy() ** 2))
            pair.append(base_rmse - alt_rmse)
        pair = np.asarray(pair)
        deltas.append(dict(kind=kind, improvement_rmse=float(np.sqrt(np.mean(wide.logistic ** 2)) - np.sqrt(np.mean(wide[kind] ** 2))),
                           bootstrap_low95=float(np.quantile(pair, 0.025)),
                           bootstrap_high95=float(np.quantile(pair, 0.975)),
                           p_bootstrap_improvement=float(np.mean(pair > 0))))
    pd.DataFrame(deltas).to_csv(HERE / "bridge_paired_family_bootstrap.csv", index=False, encoding="utf-8-sig")
    return {"input": str(source.relative_to(ROOT)), "rows": len(d),
            "families": int(d.group.nunique()), "high_rows": int(d.high.sum()),
            "high_families": int(d.loc[d.high, "group"].nunique()),
            "cv_folds": len(folds), "baseline": "C6 weighted bounded logistic",
            "baseline_fold_rmse_reproduced": True}


def historical_experiment() -> dict:
    """Compare existing structural and compute-only models on identical family folds."""
    source = ROOT / "problem4/outputs/tables/historical_scale_sample.csv"
    d = pd.read_csv(source)
    d["submission_date"] = pd.to_datetime(d.submission_date)
    groups = d.organization.fillna(d.Model.str.split("/").str[0]).astype(str).to_numpy()
    folds = list(GroupKFold(n_splits=min(5, len(set(groups)))).split(d, groups=groups))
    kinds = ("time_only", "compute_only", "structural")
    rows = []
    for kind in kinds:
        for fold, (tr, te) in enumerate(folds):
            coef, _ = historical_fit(d.iloc[tr], kind)
            pred = historical_score(historical_design(d.iloc[te], kind) @ coef)
            for pos, value in zip(te, pred):
                rows.append(dict(kind=kind, fold=fold, row=int(pos),
                                 model=d.Model.iloc[pos], organization=groups[pos],
                                 actual=float(d.ability.iloc[pos]), prediction=float(value),
                                 error=float(value - d.ability.iloc[pos])))
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "historical_oof_predictions.csv", index=False, encoding="utf-8-sig")
    scorecard = out.groupby("kind").agg(n=("row", "size"), families=("organization", "nunique"),
                                         rmse=("error", lambda e: np.sqrt(np.mean(e ** 2))),
                                         mae=("error", lambda e: np.mean(np.abs(e)))).reset_index()
    old = pd.read_csv(ROOT / "problem4/outputs/tables/historical_model_comparison.csv")
    aligned = scorecard.merge(old[["model", "group_cv_rmse"]], left_on="kind", right_on="model", validate="one_to_one")
    if len(aligned) != 3 or not np.allclose(aligned.rmse, aligned.group_cv_rmse, atol=1e-8):
        raise AssertionError("Historical CV results do not reproduce original group CV")
    scorecard.to_csv(HERE / "historical_scorecard.csv", index=False, encoding="utf-8-sig")
    wide = out.pivot(index="row", columns="kind", values="error")
    uniq = np.unique(groups)
    rng = np.random.default_rng(20260924)
    paired = []
    for kind in ("time_only", "structural"):
        vals = []
        for _ in range(3000):
            sampled = rng.choice(uniq, len(uniq), replace=True)
            idx = np.concatenate([np.flatnonzero(groups == gg) for gg in sampled])
            base_rmse = np.sqrt(np.mean(wide.loc[idx, "compute_only"].to_numpy() ** 2))
            alt_rmse = np.sqrt(np.mean(wide.loc[idx, kind].to_numpy() ** 2))
            vals.append(base_rmse - alt_rmse)
        vals = np.asarray(vals)
        paired.append(dict(kind=kind,
                           improvement_over_compute_only=float(
                               np.sqrt(np.mean(wide.compute_only ** 2)) - np.sqrt(np.mean(wide[kind] ** 2))),
                           bootstrap_low95=float(np.quantile(vals, 0.025)),
                           bootstrap_high95=float(np.quantile(vals, 0.975)),
                           p_bootstrap_improvement=float(np.mean(vals > 0))))
    pd.DataFrame(paired).to_csv(HERE / "historical_paired_family_bootstrap.csv", index=False, encoding="utf-8-sig")
    boots = np.load(ROOT / "problem4/outputs/tables/historical_bootstrap_coefficients.npy")
    ci = np.quantile(boots[:, 2], [0.025, 0.5, 0.975]).tolist()
    return {"input": str(source.relative_to(ROOT)), "rows": len(d),
            "families": len(uniq), "baseline_cv_reproduced": True,
            "structural_time_coefficient_bootstrap_95": ci}


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    result = {"frontier": frontier_experiment(), "bridge": bridge_experiment(),
              "historical": historical_experiment(),
              "seed": 20260924,
              "input_sha256": {
                  str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (ROOT / "problem4/outputs/tables/analysis_dataset.csv",
                            ROOT / "real_attachments/C_efficiency_evolution/loss_benchmark_bridge_expanded.csv",
                            ROOT / "problem4/outputs/tables/historical_scale_sample.csv")
              },
              "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                           "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
              "interpretation": "Descriptive sensitivity only; sparse strict frontier and mixed-comparability bridge."}
    (HERE / "run_metadata.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
