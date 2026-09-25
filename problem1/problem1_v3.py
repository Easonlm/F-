"""Conditional 1M RegMix candidate: additive CLR splines, preserving V2 artifact.

Run from the project root: python problem1/problem1_v3.py
Only A4/A5 training labels select and fit the model. A6-A11 and estimated A12-A15
are retrospective checks; they never select hyperparameters. This script writes
unique V3 tables and `mixture_v3_1m.joblib`, and leaves V1/V2 outputs intact.
"""
from __future__ import annotations

import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler

from mixture_pipeline import OUT, SEED, TABLES, metric, pair
from problem1_v2 import transform as v2_transform


FAMILIES = ("v2", "v3_additive_clr_spline")


def clr(p: np.ndarray, eps: float) -> np.ndarray:
    q = p + eps
    q = q / q.sum(axis=1, keepdims=True)
    logq = np.log(q)
    return logq - logq.mean(axis=1, keepdims=True)


def new_model(n_knots: int, alpha: float):
    return make_pipeline(
        SplineTransformer(n_knots=n_knots, degree=3, knots="quantile",
                          extrapolation="linear", include_bias=False),
        StandardScaler(), Ridge(alpha=alpha),
    )


def specs():
    return [dict(eps_mult=eps_mult, n_knots=n_knots, alpha=alpha)
            for eps_mult in (0.25, 0.5) for n_knots in (4, 6) for alpha in (1., 10., 100.)]


def fit_spec(spec, p, y):
    eps = float(p[p > 0].min() * spec["eps_mult"])
    model = new_model(spec["n_knots"], spec["alpha"])
    model.fit(clr(p, eps), y)
    return model, eps


def choose(p, y, splits):
    rows = []
    for spec in specs():
        errors = []
        for train, val in splits:
            model, eps = fit_spec(spec, p[train], y[train])
            errors.append(mean_squared_error(y[val], model.predict(clr(p[val], eps))))
        rows.append(dict(spec=spec, inner_mse=float(np.mean(errors))))
    return min(rows, key=lambda row: row["inner_mse"]), rows


def predict_v3(bundle: dict, p: np.ndarray) -> np.ndarray:
    return bundle["model"].predict(clr(p, float(bundle["eps"])))


def rank_mean(y, pred):
    return float(spearmanr(y.mean(axis=1), pred.mean(axis=1)).statistic)


def paired_gain(y, base, new, seed, draws=3000):
    old_err = np.mean((base - y) ** 2, axis=1)
    new_err = np.mean((new - y) ** 2, axis=1)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(y), size=(draws, len(y)))
    gains = np.sqrt(old_err[indices].mean(axis=1)) - np.sqrt(new_err[indices].mean(axis=1))
    return dict(v2_minus_v3_rmse=float(np.sqrt(old_err.mean()) - np.sqrt(new_err.mean())),
                ci_low=float(np.quantile(gains, .025)), ci_high=float(np.quantile(gains, .975)),
                unit="mixture row", draws=draws)


def add_metrics(rows, dataset, status, y, v2, v3):
    for name, pred in zip(FAMILIES, (v2, v3)):
        rows.append(dict(dataset=dataset, status=status, version=name, n_mixtures=len(y),
                         **metric(y, pred), mixture_mean_spearman=rank_mean(y, pred)))


def run():
    TABLES.mkdir(parents=True, exist_ok=True)
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    ids, p, y, pcols, ycols = pair("train", "1m")
    assert len(y) == 512 and y.shape[1] == 13
    old_fold_specs = pd.read_csv(TABLES / "v2_nested_cv_folds.csv").set_index("outer_fold")
    oof_v2, oof_v3 = np.full_like(y, np.nan), np.full_like(y, np.nan)
    nested, nested_search = [], []
    for fold, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=SEED + 1).split(p), 1):
        prior = old_fold_specs.loc[fold]
        assert prior.selected_family == "poly_alr"
        old = make_pipeline(PolynomialFeatures(2, include_bias=False),
                            StandardScaler(), Ridge(alpha=float(prior.selected_alpha)))
        old.fit(v2_transform(p[tr], "poly_alr", float(prior.selected_eps), int(prior.selected_ref)), y[tr])
        oof_v2[te] = old.predict(v2_transform(p[te], "poly_alr", float(prior.selected_eps), int(prior.selected_ref)))
        assert np.isclose(np.sqrt(mean_squared_error(y[te], oof_v2[te])), prior.v2_outer_rmse, atol=1e-10)

        split = list(KFold(3, shuffle=True, random_state=SEED + fold).split(tr))
        winner, searched = choose(p[tr], y[tr], split)
        for row in searched:
            nested_search.append(dict(outer_fold=fold, **row["spec"],
                                      inner_rmse=float(np.sqrt(row["inner_mse"])), selected=row is winner))
        v3, eps = fit_spec(winner["spec"], p[tr], y[tr])
        oof_v3[te] = v3.predict(clr(p[te], eps))
        nested.append(dict(outer_fold=fold, n=len(te), v2_rmse=float(np.sqrt(mean_squared_error(y[te], oof_v2[te]))),
                           v3_rmse=float(np.sqrt(mean_squared_error(y[te], oof_v3[te]))),
                           selected_spec=json.dumps(winner["spec"], sort_keys=True)))
    assert np.all(np.isfinite(oof_v2)) and np.all(np.isfinite(oof_v3))
    pd.DataFrame(nested).to_csv(TABLES / "v3_nested_cv_folds.csv", index=False)
    pd.DataFrame(nested_search).to_csv(TABLES / "v3_nested_inner_search.csv", index=False)
    full_splits = list(RepeatedKFold(n_splits=5, n_repeats=2, random_state=SEED).split(p))
    selected, searched = choose(p, y, full_splits)
    pd.DataFrame([dict(**row["spec"], cv_rmse=float(np.sqrt(row["inner_mse"])),
                       selected=row is selected) for row in searched]).to_csv(TABLES / "v3_train_cv_search.csv", index=False)
    model, eps = fit_spec(selected["spec"], p, y)
    old_path = OUT / "models" / "mixture_v2.joblib"
    old_hash = hashlib.sha256(old_path.read_bytes()).hexdigest()
    bundle = dict(model=model, family="additive_clr_spline", eps=eps, pcols=pcols, ycols=ycols,
                  selected_spec=selected["spec"], intended_scope="1M mixture prediction; 60M retrospective support only",
                  upstream_v2_sha256=old_hash)
    joblib.dump(bundle, OUT / "models" / "mixture_v3_1m.joblib")
    (TABLES / "v3_selection.json").write_text(json.dumps(dict(
        selection_rule="A4/A5 2x5 repeated CV minimum sqrt(mean fold MSE); 12 spline candidates",
        selected=selected["spec"], cv_rmse=float(np.sqrt(selected["inner_mse"])),
        model_artifact="outputs/models/mixture_v3_1m.joblib", upstream_v2_sha256=old_hash,
        no_test_labels_used_for_selection=True,
    ), ensure_ascii=False, indent=2), encoding="utf-8")
    v2_bundle = joblib.load(old_path)
    assert v2_bundle["pcols"] == pcols and v2_bundle["ycols"] == ycols
    comparisons, domain_rows, boots = [], [], []
    add_metrics(comparisons, "train_nested_oof", "observed_train_oof", y, oof_v2, oof_v3)
    boots.append(dict(dataset="train_nested_oof", n_mixtures=len(y), **paired_gain(y, oof_v2, oof_v3, SEED + 100)))
    for i, (label, prefix, scale) in enumerate((("test_1m", "test", "1m"),
                                                ("test_60m", "test", "60m"),
                                                ("test_1B", "test", "1B"),
                                                ("est_10b", "est", "10b"),
                                                ("est_70b", "est", "70b"))):
        _, pt, yt, pc, yc = pair(prefix, scale)
        assert pc == pcols and yc == ycols
        pred2 = v2_bundle["model"].predict(v2_transform(pt, v2_bundle["family"],
                                                          v2_bundle["eps"], v2_bundle["reference_index"]))
        pred3 = predict_v3(bundle, pt)
        add_metrics(comparisons, label, "observed" if prefix == "test" else "estimated_label", yt, pred2, pred3)
        boots.append(dict(dataset=label, n_mixtures=len(yt), **paired_gain(yt, pred2, pred3, SEED + 101 + i)))
        for version, pred in zip(FAMILIES, (pred2, pred3)):
            for j, col in enumerate(ycols):
                domain_rows.append(dict(dataset=label, version=version,
                                        validation_domain=col.replace("metric/the_pile_", "").replace("_val_loss", ""),
                                        rmse=float(np.sqrt(mean_squared_error(yt[:, j], pred[:, j]))),
                                        bias=float(np.mean(pred[:, j] - yt[:, j])),
                                        spearman=float(spearmanr(yt[:, j], pred[:, j]).statistic)))
    pd.DataFrame(comparisons).to_csv(TABLES / "v3_vs_v2_metrics.csv", index=False)
    pd.DataFrame(domain_rows).to_csv(TABLES / "v3_vs_v2_per_domain.csv", index=False)
    pd.DataFrame(boots).to_csv(TABLES / "v3_vs_v2_bootstrap.csv", index=False)
    print(pd.DataFrame(comparisons)[["dataset", "version", "rmse", "spearman_mean", "mixture_mean_spearman"]].to_string(index=False))


if __name__ == "__main__":
    run()
