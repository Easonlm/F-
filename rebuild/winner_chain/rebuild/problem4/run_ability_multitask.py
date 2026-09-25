"""Independent ability-index checks and partial-pooling C6 bridge challenge."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "problem4"))
from bridge_model import family, fit, predict  # noqa: E402

TASKS = ["LB_IFEval", "LB_BBH", "LB_MATH", "LB_GPQA", "LB_MUSR", "LB_MMLU_PRO"]


def ability_check() -> list[dict]:
    detailed = pd.read_csv(ROOT / "problem4/outputs/tables/c8_vs_leaderboard.csv")
    indexes = pd.read_csv(ROOT / "problem4/outputs/tables/ability_index_comparison.csv")
    d = detailed.merge(indexes[["Model", "ability", "robust_ability", "pca_ability"]],
                       on="Model", suffixes=("", "_index"), validate="one_to_many")
    # C1 contains repeated submissions under one name; recover the exact C8
    # leaderboard row by its six-benchmark score before comparing indexes.
    d["ability_gap"] = (d.ability - d.ability_index).abs()
    d = d.sort_values("ability_gap").drop_duplicates("Model", keep="first")
    assert d.ability_gap.max() < 1e-8
    d["family"] = d.Model.map(family)
    out = []
    rng = np.random.default_rng(20260925)
    families = d.family.unique()
    for name, col in [("six_benchmark_mean", "ability"),
                      ("robust_standardized_mean", "robust_ability"),
                      ("PCA_latent", "pca_ability"),
                      ("C8_detailed_tasks", "detailed_ability")]:
        x = d[col].to_numpy(float)
        y = d.detailed_ability.to_numpy(float)
        valid = np.isfinite(x) & np.isfinite(y)
        rho = float(spearmanr(x[valid], y[valid]).statistic)
        boot = []
        # Cluster by model family, preserving within-family variants.
        family_idx = {g: np.flatnonzero(d.family.to_numpy() == g) for g in families}
        for _ in range(500):
            chosen = rng.choice(families, size=len(families), replace=True)
            ix = np.concatenate([family_idx[g] for g in chosen])
            v = np.isfinite(x[ix]) & np.isfinite(y[ix])
            if v.sum() > 10:
                boot.append(float(spearmanr(x[ix][v], y[ix][v]).statistic))
        lo, hi = np.quantile(boot, [0.025, 0.975])
        out.append(dict(candidate=name, n=int(valid.sum()), families=len(families),
                        c8_spearman=rho, cluster_ci_low=float(lo), cluster_ci_high=float(hi),
                        independent_validation=False,
                        note="C8 task data are related to the leaderboard; consistency check only"))
    pd.DataFrame(out).to_csv(HERE / "ability_index_tournament.csv", index=False, encoding="utf-8-sig")
    return out


def fit_multitask(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    # Six task intercepts, one shared nonnegative Loss slope. Seven effective
    # parameters are trained on 75 model rows, each containing six outcomes.
    def raw(p):
        return p[:6][None, :] - np.exp(p[6]) * x[:, None]
    p0 = np.r_[np.full(6, 2.0), 0.0]
    lower = np.r_[np.full(6, -20.0), -8.0]
    upper = np.r_[np.full(6, 20.0), 5.0]
    res = least_squares(lambda p: ((100 * expit(raw(p)) - y) * np.sqrt(w[:, None] / 6)).ravel(),
                        p0, bounds=(lower, upper), max_nfev=3000)
    return res.x


def bridge_challenge() -> dict:
    source = ROOT / "real_attachments/C_efficiency_evolution/loss_benchmark_bridge_expanded.csv"
    d = pd.read_csv(source).dropna(subset=["Val_Loss", "LB_Average", *TASKS]).reset_index(drop=True)
    assert len(d) == 75
    d["family"] = d.Model.map(family)
    x = d.Val_Loss.to_numpy(float)
    y = d.LB_Average.to_numpy(float)
    task_y = d[TASKS].to_numpy(float)
    w = np.where(d.Loss_Comparability.str.startswith("High"), 1.0, 0.35)
    g = d.family.to_numpy()
    folds = list(GroupKFold(n_splits=5).split(x, y, g))
    pred = np.full((len(d), 2), np.nan)
    params = []
    for k, (tr, te) in enumerate(folds):
        assert set(g[tr]).isdisjoint(g[te])
        m = fit(x[tr], y[tr], w[tr], "logistic")
        pred[te, 0] = predict(m, x[te])
        p = fit_multitask(x[tr], task_y[tr], w[tr])
        pred[te, 1] = (100 * expit(p[:6][None, :] - np.exp(p[6]) * x[te, None])).mean(axis=1)
        params.append(dict(fold=k, shared_slope=float(np.exp(p[6])),
                           train_n=len(tr), test_n=len(te),
                           train_families=len(set(g[tr])), test_families=len(set(g[te]))))
    assert np.isfinite(pred).all()
    rows = []
    for j, name in enumerate(["bounded_logistic", "partial_pool_multitask"]):
        err = pred[:, j] - y
        rows.append(dict(candidate=name, family_cv_rmse=float(np.sqrt(np.mean(err ** 2))),
                         family_cv_mae=float(np.mean(np.abs(err))), n=len(y), families=len(set(g)),
                         parameter_count=2 if j == 0 else 7,
                         min_loss_slope=min(z["shared_slope"] for z in params) if j else -m["b"]))
    pd.DataFrame(rows).to_csv(HERE / "multitask_bridge_tournament.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(params).to_csv(HERE / "multitask_fold_parameters.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(dict(Model=d.Model, family=g, actual=y, bounded_logistic=pred[:, 0],
                      partial_pool_multitask=pred[:, 1])).to_csv(
                          HERE / "multitask_oof_predictions.csv", index=False, encoding="utf-8-sig")
    rng = np.random.default_rng(20260925)
    families = np.unique(g)
    by_family = {z: np.flatnonzero(g == z) for z in families}
    gains = []
    for _ in range(3000):
        chosen = rng.choice(families, len(families), replace=True)
        ix = np.concatenate([by_family[z] for z in chosen])
        e = pred[ix] - y[ix, None]
        gains.append(float(np.sqrt(np.mean(e[:, 0] ** 2)) - np.sqrt(np.mean(e[:, 1] ** 2))))
    meta = dict(input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                n=len(d), families=len(families), high_rows=int((w == 1).sum()),
                baseline_reproduced=abs(rows[0]["family_cv_rmse"] - 11.86412981770644) < 1e-9,
                paired_family_bootstrap_gain_95=np.quantile(gains, [0.025, 0.975]).tolist(),
                note="Bootstrapped OOF residuals; no retraining in bootstrap. High-comparability observations come from one family.")
    (HERE / "multitask_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    assert meta["baseline_reproduced"]
    return meta


if __name__ == "__main__":
    ability_check()
    print(json.dumps(bridge_challenge(), indent=2))
