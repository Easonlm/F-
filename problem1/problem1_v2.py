"""Train-only selection of a second mixture model and a locked v1/v2 comparison."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from scipy.linalg import helmert
from scipy.stats import spearmanr
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from mixture_pipeline import OUT, SEED, TABLES, alr, metric, pair


def transform(p: np.ndarray, family: str, eps: float, ref: int) -> np.ndarray:
    if family == "raw":
        return p[:, :-1]
    q = p + eps
    q = q / q.sum(axis=1, keepdims=True)
    logq = np.log(q)
    if family in ("alr", "poly_alr", "rbf_alr"):
        return logq[:, np.arange(p.shape[1]) != ref] - logq[:, ref, None]
    if family == "clr":
        return logq - logq.mean(axis=1, keepdims=True)
    if family == "ilr":
        return logq @ helmert(p.shape[1], full=False).T
    raise ValueError(family)


def estimator(family: str, alpha: float, gamma: float | None = None):
    if family == "poly_alr":
        return make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(), Ridge(alpha=alpha))
    if family == "rbf_alr":
        return make_pipeline(StandardScaler(), KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma))
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def candidates(p: np.ndarray):
    positive_min = float(p[p > 0].min())
    main_ref = p.shape[1] - 1
    refs = list(dict.fromkeys([main_ref, int(p.mean(axis=0).argmax()), int(p.mean(axis=0).argmin())]))
    for family in ["raw", "alr", "clr", "ilr", "poly_alr", "rbf_alr"]:
        ref_list = refs if family in ("alr", "poly_alr", "rbf_alr") else [main_ref]
        eps_list = [positive_min * q for q in (0.25, 0.5, 1.0)] if family != "raw" else [0.0]
        alphas = [0.1, 1.0, 10.0, 100.0] if family not in ("poly_alr", "rbf_alr") else [1.0, 10.0, 100.0]
        gammas = [0.01, 0.05, 0.2] if family == "rbf_alr" else [None]
        for ref in ref_list:
            for eps in eps_list:
                for alpha in alphas:
                    for gamma in gammas:
                        yield family, eps, ref, alpha, gamma


def evaluate_cv(p: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    splits = list(RepeatedKFold(n_splits=5, n_repeats=2, random_state=SEED).split(p))
    rows = []
    for family, eps, ref, alpha, gamma in candidates(p):
        x = transform(p, family, eps, ref)
        fold_mse = []
        for train_idx, val_idx in splits:
            m = estimator(family, alpha, gamma)
            m.fit(x[train_idx], y[train_idx])
            fold_mse.append(mean_squared_error(y[val_idx], m.predict(x[val_idx])))
        rows.append({"family": family, "eps": eps, "reference_index": ref,
                     "alpha": alpha, "gamma": gamma, "cv_rmse_pooled": float(np.sqrt(np.mean(fold_mse))),
                     "cv_rmse_fold_sd": float(np.std(np.sqrt(fold_mse))), "folds": len(splits)})
    return pd.DataFrame(rows).sort_values("cv_rmse_pooled").reset_index(drop=True)


def predictions(candidate: pd.Series, p_train, y_train, p_test):
    family, eps, ref, alpha = candidate["family"], float(candidate["eps"]), int(candidate["reference_index"]), float(candidate["alpha"])
    gamma = None if pd.isna(candidate["gamma"]) else float(candidate["gamma"])
    m = estimator(family, alpha, gamma)
    m.fit(transform(p_train, family, eps, ref), y_train)
    return m, m.predict(transform(p_test, family, eps, ref))


def summary_row(dataset: str, version: str, y: np.ndarray, pred: np.ndarray):
    return {"dataset": dataset, "version": version, "n_mixtures": len(y), **metric(y, pred)}


def run():
    TABLES.mkdir(parents=True, exist_ok=True)
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    _, p, y, pcols, ycols = pair("train", "1m")
    cv = evaluate_cv(p, y)
    cv.to_csv(TABLES / "v2_train_cv_search.csv", index=False)
    selected = cv.iloc[0]
    (TABLES / "v2_selection.json").write_text(json.dumps({
        "selection_rule": "lowest pooled RMSE in 2x5 repeated CV on A4/A5 only",
        "selected": selected.where(pd.notna(selected), None).to_dict(),
        "candidate_count": len(cv), "no_test_labels_used_for_selection": True,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    eps_v1 = float(p[p > 0].min() / 2)
    v1 = estimator("alr", 1.0)
    v1.fit(alr(p, eps_v1), y)
    v2 = estimator(selected.family, float(selected.alpha), None if pd.isna(selected.gamma) else float(selected.gamma))
    v2.fit(transform(p, selected.family, float(selected.eps), int(selected.reference_index)), y)
    joblib.dump({"model": v2, "family": selected.family, "eps": float(selected.eps),
                 "reference_index": int(selected.reference_index), "pcols": pcols, "ycols": ycols},
                OUT / "models" / "mixture_v2.joblib")

    comparison, per_domain, rows, support = [], [], [], []
    for label, prefix, scale in [("test_1m", "test", "1m"), ("test_60m", "test", "60m"),
                                 ("test_1B", "test", "1B"), ("est_10b", "est", "10b"),
                                 ("est_70b", "est", "70b")]:
        d, pt, yt, pc, yc = pair(prefix, scale)
        assert pc == pcols and yc == ycols
        pred1 = v1.predict(alr(pt, eps_v1))
        pred2 = v2.predict(transform(pt, selected.family, float(selected.eps), int(selected.reference_index)))
        for ver, pred in [("v1", pred1), ("v2", pred2)]:
            comparison.append(summary_row(label, ver, yt, pred))
            for j, col in enumerate(ycols):
                name = col.replace("metric/the_pile_", "").replace("_val_loss", "")
                per_domain.append({"dataset": label, "version": ver, "validation_domain": name,
                                   "rmse": float(np.sqrt(mean_squared_error(yt[:, j], pred[:, j]))),
                                   "mae": float(np.mean(np.abs(yt[:, j] - pred[:, j]))),
                                   "bias": float(np.mean(pred[:, j] - yt[:, j])),
                                   "spearman": float(spearmanr(yt[:, j], pred[:, j]).statistic)})
            if label == "test_1m":
                for i, idx in enumerate(d["index"]):
                    rows.append({"index": idx, "version": ver, "actual_mean_loss": float(yt[i].mean()),
                                 "predicted_mean_loss": float(pred[i].mean()),
                                 "mixture_mse": float(np.mean((pred[i] - yt[i]) ** 2))})
        if label == "test_1m":
            per_mix_1 = np.mean((pred1 - yt) ** 2, axis=1)
            per_mix_2 = np.mean((pred2 - yt) ** 2, axis=1)
            rng = np.random.default_rng(SEED)
            draws = rng.integers(0, len(yt), size=(3000, len(yt)))
            diffs = np.sqrt(per_mix_1[draws].mean(axis=1)) - np.sqrt(per_mix_2[draws].mean(axis=1))
            support = [{"dataset": label, "n_mixtures": len(yt), "v1_minus_v2_rmse": float(np.sqrt(per_mix_1.mean()) - np.sqrt(per_mix_2.mean())),
                        "bootstrap_ci_low": float(np.quantile(diffs, 0.025)), "bootstrap_ci_high": float(np.quantile(diffs, 0.975)),
                        "bootstrap_unit": "mixture row", "bootstrap_draws": len(diffs)}]
    pd.DataFrame(comparison).to_csv(TABLES / "v2_vs_v1_metrics.csv", index=False)
    pd.DataFrame(per_domain).to_csv(TABLES / "v2_vs_v1_per_domain.csv", index=False)
    pd.DataFrame(rows).to_csv(TABLES / "v2_vs_v1_1m_per_mixture.csv", index=False)
    pd.DataFrame(support).to_csv(TABLES / "v2_vs_v1_bootstrap.csv", index=False)
    # Robustness of descriptive substitution effects, using training data only.
    center = p.mean(axis=0)
    center /= center.sum()
    donor = int(center.argmax())
    delta = 0.01
    effect_rows = []
    positive_min = float(p[p > 0].min())
    for ref in list(dict.fromkeys([p.shape[1] - 1, donor, int(p.mean(axis=0).argmin())])):
        for mult in [0.25, 0.5, 1.0]:
            eps = positive_min * mult
            robust_model = estimator("poly_alr", 10.0)
            robust_model.fit(transform(p, "poly_alr", eps, ref), y)
            base = robust_model.predict(transform(center[None, :], "poly_alr", eps, ref))[0].mean()
            for i, name in enumerate(pcols):
                if i == donor:
                    continue
                shifted = center.copy()
                shifted[i] += delta
                shifted[donor] -= delta
                effect = robust_model.predict(transform(shifted[None, :], "poly_alr", eps, ref))[0].mean() - base
                effect_rows.append({"increase": name.replace("train_the_pile_", ""),
                                    "decrease": pcols[donor].replace("train_the_pile_", ""),
                                    "reference_index": ref, "eps_multiplier": mult,
                                    "predicted_mean_loss_change_1pp": effect})
    effects = pd.DataFrame(effect_rows)
    effects.to_csv(TABLES / "v2_effect_sensitivity.csv", index=False)
    effects.groupby(["increase", "decrease"]).predicted_mean_loss_change_1pp.agg(
        n_specs="size", min_effect="min", max_effect="max", median_effect="median",
        sign_consistency=lambda s: max((s > 0).mean(), (s < 0).mean())
    ).reset_index().to_csv(TABLES / "v2_effect_stability_summary.csv", index=False)
    print(cv.head(10).to_string(index=False))
    print(pd.DataFrame(comparison)[["dataset", "version", "rmse", "mae", "spearman_mean", "bias"]].to_string(index=False))
    print(pd.DataFrame(support).to_string(index=False))


if __name__ == "__main__":
    run()
