"""Compare two new, train-selected P1 compositional regressors with frozen V2.

Run: python optimization_20260924/round2/q1/run_new_models.py
Only this directory is written. A4/A5 select hyperparameters; A6-A11 and estimated
A12-A15 are evaluated after fitting. The latter were previously inspected, so
their scores are retrospective evidence rather than a fresh blind test.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.linalg import helmert
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
A = ROOT / "real_attachments" / "A_data_value" / "regmix_tables"
P1 = ROOT / "problem1" / "outputs"
SEED = 20260923


def paired_a(prefix: str, scale: str):
    mix_path = A / f"{prefix}_mixture_{scale}.csv"
    loss_path = A / f"{prefix}_pile_loss_{scale}.csv"
    mix, loss = pd.read_csv(mix_path), pd.read_csv(loss_path)
    assert mix["index"].is_unique and loss["index"].is_unique
    x = mix.merge(loss, on="index", validate="one_to_one")
    assert len(x) == len(mix) == len(loss)
    pcols = [c for c in mix if c.startswith("train_the_pile_")]
    ycols = [c for c in loss if c.startswith("metric/the_pile_")]
    p = x[pcols].to_numpy(float)
    assert np.all(p >= 0) and np.allclose(p.sum(axis=1), 1, atol=.005)
    p = p / p.sum(axis=1, keepdims=True)
    y = x[ycols].to_numpy(float)
    assert np.all(np.isfinite(y))
    return x["index"].to_numpy(), p, y, pcols, ycols, [mix_path, loss_path]


class LogRatio(BaseEstimator, TransformerMixin):
    def __init__(self, geometry: str = "clr", eps_mult: float = 0.5):
        self.geometry = geometry
        self.eps_mult = eps_mult

    def fit(self, p, y=None):
        self.eps_ = float(np.min(p[p > 0]) * self.eps_mult)
        self.helmert_ = helmert(p.shape[1], full=False) if self.geometry == "ilr" else None
        return self

    def transform(self, p):
        q = p + self.eps_
        q = q / q.sum(axis=1, keepdims=True)
        z = np.log(q)
        if self.geometry == "clr":
            return z - z.mean(axis=1, keepdims=True)
        if self.geometry == "ilr":
            return z @ self.helmert_.T
        raise ValueError(self.geometry)


def specs(family: str):
    if family == "additive_clr_spline":
        return [dict(family=family, eps_mult=e, n_knots=k, alpha=a)
                for e, k, a in itertools.product((0.25, 0.5), (4, 6), (1., 10., 100.))]
    if family == "lowrank_ilr_quadratic":
        return [dict(family=family, eps_mult=e, n_components=k, alpha=a)
                for e, k, a in itertools.product((0.25, 0.5), (4, 8, 12), (1., 10., 100.))]
    raise ValueError(family)


def candidate_model(spec):
    if spec["family"] == "additive_clr_spline":
        return make_pipeline(
            LogRatio("clr", spec["eps_mult"]),
            SplineTransformer(n_knots=spec["n_knots"], degree=3, knots="quantile",
                              extrapolation="linear", include_bias=False),
            StandardScaler(), Ridge(alpha=spec["alpha"]),
        )
    if spec["family"] == "lowrank_ilr_quadratic":
        return make_pipeline(
            LogRatio("ilr", spec["eps_mult"]),
            PCA(n_components=spec["n_components"]),
            PolynomialFeatures(2, include_bias=False),
            StandardScaler(), Ridge(alpha=spec["alpha"]),
        )
    raise ValueError(spec["family"])


def v2_transform(p, eps, ref):
    q = p + eps
    q = q / q.sum(axis=1, keepdims=True)
    z = np.log(q)
    return z[:, np.arange(p.shape[1]) != ref] - z[:, ref, None]


def clr_with_eps(p, eps):
    q = p + eps
    q = q / q.sum(axis=1, keepdims=True)
    z = np.log(q)
    return z - z.mean(axis=1, keepdims=True)


def score(y, pred):
    means = [spearmanr(y[:, j], pred[:, j]).statistic for j in range(y.shape[1])]
    return dict(n_mixtures=len(y), rmse=float(np.sqrt(mean_squared_error(y, pred))),
                mae=float(np.mean(np.abs(y - pred))), bias=float(np.mean(pred - y)),
                domain_spearman_mean=float(np.nanmean(means)),
                mixture_mean_spearman=float(spearmanr(y.mean(axis=1), pred.mean(axis=1)).statistic),
                centered_mean_rmse=float(np.sqrt(np.mean(((pred.mean(axis=1) - pred.mean()) -
                                                          (y.mean(axis=1) - y.mean())) ** 2))))


def choose(p, y, family, splits):
    rows = []
    for spec in specs(family):
        mse = []
        for train, val in splits:
            model = candidate_model(spec)
            model.fit(p[train], y[train])
            mse.append(mean_squared_error(y[val], model.predict(p[val])))
        rows.append(dict(spec=spec, mean_mse=float(np.mean(mse))))
    return min(rows, key=lambda r: r["mean_mse"]), rows


def bootstrap_gain(y, base, candidate, seed, draws=3000):
    rng = np.random.default_rng(seed)
    mse_base = np.mean((base - y) ** 2, axis=1)
    mse_new = np.mean((candidate - y) ** 2, axis=1)
    idx = rng.integers(0, len(y), size=(draws, len(y)))
    gains = np.sqrt(mse_base[idx].mean(axis=1)) - np.sqrt(mse_new[idx].mean(axis=1))
    return float(np.sqrt(mse_base.mean()) - np.sqrt(mse_new.mean())), float(np.quantile(gains, .025)), float(np.quantile(gains, .975))


def rank_bootstrap_gain(y, base, candidate, seed, draws=2000):
    actual, old, new = y.mean(axis=1), base.mean(axis=1), candidate.mean(axis=1)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        ix = rng.integers(len(y), size=len(y))
        values.append(float(spearmanr(actual[ix], new[ix]).statistic -
                            spearmanr(actual[ix], old[ix]).statistic))
    point = float(spearmanr(actual, new).statistic - spearmanr(actual, old).statistic)
    return point, float(np.quantile(values, .025)), float(np.quantile(values, .975))


def support_summary(train_p, target_p, eps):
    train_clr = np.log(train_p + eps)
    train_clr -= train_clr.mean(axis=1, keepdims=True)
    test_clr = np.log(target_p + eps)
    test_clr -= test_clr.mean(axis=1, keepdims=True)
    outside = ((test_clr < train_clr.min(axis=0)) | (test_clr > train_clr.max(axis=0))).any(axis=1)
    nearest = cdist(target_p, train_p).min(axis=1)
    return dict(fraction_outside_train_clr_range=float(outside.mean()),
                median_nearest_train_raw_l2=float(np.median(nearest)),
                p90_nearest_train_raw_l2=float(np.quantile(nearest, .9)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ids, p, y, pcols, ycols, source_paths = paired_a("train", "1m")
    assert len(y) == 512 and y.shape[1] == 13
    families = ("additive_clr_spline", "lowrank_ilr_quadratic")
    outer = KFold(5, shuffle=True, random_state=SEED + 1)
    old_specs = pd.read_csv(P1 / "tables" / "v2_nested_cv_folds.csv").set_index("outer_fold")
    oof = {name: np.full_like(y, np.nan) for name in ("v2",) + families}
    outer_rows, inner_rows = [], []
    for fold, (tr, te) in enumerate(outer.split(p), 1):
        prior = old_specs.loc[fold]
        assert prior.selected_family == "poly_alr"  # Existing nested selection, not refitted on test.
        old = make_pipeline(PolynomialFeatures(2, include_bias=False),
                            StandardScaler(), Ridge(alpha=float(prior.selected_alpha)))
        old.fit(v2_transform(p[tr], float(prior.selected_eps), int(prior.selected_ref)), y[tr])
        oof["v2"][te] = old.predict(v2_transform(p[te], float(prior.selected_eps), int(prior.selected_ref)))
        assert np.isclose(score(y[te], oof["v2"][te])["rmse"], prior.v2_outer_rmse, atol=1e-10)
        outer_rows.append(dict(outer_fold=fold, model="v2", **score(y[te], oof["v2"][te])))
        splits = list(KFold(3, shuffle=True, random_state=SEED + fold).split(tr))
        for family in families:
            selection, all_scores = choose(p[tr], y[tr], family, splits)
            for row in all_scores:
                inner_rows.append(dict(outer_fold=fold, model=family,
                                       spec=json.dumps(row["spec"], sort_keys=True),
                                       inner_rmse=float(np.sqrt(row["mean_mse"])),
                                       selected=row is selection))
            model = candidate_model(selection["spec"])
            model.fit(p[tr], y[tr])
            oof[family][te] = model.predict(p[te])
            outer_rows.append(dict(outer_fold=fold, model=family, selected_spec=json.dumps(selection["spec"], sort_keys=True),
                                   **score(y[te], oof[family][te])))
        print("completed nested outer fold", fold, flush=True)
    assert all(np.all(np.isfinite(z)) for z in oof.values())
    assert np.isclose(score(y, oof["v2"])["rmse"], 0.3340097260320924, atol=1e-10)
    pd.DataFrame(outer_rows).to_csv(OUT / "nested_outer_folds.csv", index=False)
    pd.DataFrame(inner_rows).to_csv(OUT / "nested_inner_search.csv", index=False)
    all_metrics = []
    gain_rows = []
    rank_rows = []
    pred_rows = []
    domain_rows = []
    support_rows = []
    for name, pred in oof.items():
        all_metrics.append(dict(dataset="train_nested_oof", status="observed_train_oof", model=name, **score(y, pred)))
        pred_rows.extend(dict(dataset="train_nested_oof", index=ids[i], model=name,
                              actual_mean_loss=float(y[i].mean()), predicted_mean_loss=float(pred[i].mean()),
                              mixture_mse=float(np.mean((pred[i] - y[i]) ** 2))) for i in range(len(y)))
        if name != "v2":
            gain, lo, hi = bootstrap_gain(y, oof["v2"], pred, SEED + len(gain_rows))
            gain_rows.append(dict(dataset="train_nested_oof", model=name, n=len(y),
                                  v2_minus_candidate_rmse=gain, ci_low=lo, ci_high=hi))
            rgain, rlo, rhi = rank_bootstrap_gain(y, oof["v2"], pred, SEED + 500 + len(rank_rows))
            rank_rows.append(dict(dataset="train_nested_oof", model=name, n=len(y),
                                  candidate_minus_v2_mean_spearman=rgain, ci_low=rlo, ci_high=rhi))

    full_splits = list(RepeatedKFold(n_splits=5, n_repeats=2, random_state=SEED).split(p))
    full_rows, models = [], {}
    for family in families:
        selection, search = choose(p, y, family, full_splits)
        full_rows.extend(dict(model=family, spec=json.dumps(row["spec"], sort_keys=True),
                              cv_rmse=float(np.sqrt(row["mean_mse"])), selected=row is selection)
                         for row in search)
        models[family] = candidate_model(selection["spec"])
        models[family].fit(p, y)
    pd.DataFrame(full_rows).to_csv(OUT / "full_train_search.csv", index=False)
    spline_spec = json.loads(next(r["spec"] for r in full_rows if r["model"] == "additive_clr_spline" and r["selected"]))
    spline_eps = float(np.min(p[p > 0]) * spline_spec["eps_mult"])
    spline_fitted = models["additive_clr_spline"]
    spline_artifact = dict(
        model=Pipeline(spline_fitted.steps[1:]),
        geometry="clr", eps=spline_eps, pcols=pcols, ycols=ycols,
        selected_spec=spline_spec,
        selection="A4/A5 only, 2x5 repeated CV; use only within documented support",
    )
    assert np.allclose(spline_artifact["model"].predict(clr_with_eps(p, spline_eps)), spline_fitted.predict(p))
    joblib.dump(spline_artifact, OUT / "additive_clr_spline_candidate.joblib")
    saved_v2 = joblib.load(P1 / "models" / "mixture_v2.joblib")
    assert pcols == saved_v2["pcols"] and ycols == saved_v2["ycols"]
    for dataset, prefix, scale, status in (
        ("test_1m", "test", "1m", "observed"),
        ("test_60m", "test", "60m", "observed"),
        ("test_1B", "test", "1B", "observed"),
        ("est_10b", "est", "10b", "estimated_label"),
        ("est_70b", "est", "70b", "estimated_label"),
    ):
        ix, pt, yt, pc, yc, paths = paired_a(prefix, scale)
        source_paths.extend(paths)
        assert pc == pcols and yc == ycols
        support_rows.append(dict(dataset=dataset, n=len(pt), **support_summary(p, pt, spline_eps)))
        predictions = {"v2": saved_v2["model"].predict(v2_transform(pt, saved_v2["eps"], saved_v2["reference_index"]))}
        predictions.update((family, model.predict(pt)) for family, model in models.items())
        for name, pred in predictions.items():
            all_metrics.append(dict(dataset=dataset, status=status, model=name, **score(yt, pred)))
            pred_rows.extend(dict(dataset=dataset, index=ix[i], model=name,
                                  actual_mean_loss=float(yt[i].mean()), predicted_mean_loss=float(pred[i].mean()),
                                  mixture_mse=float(np.mean((pred[i] - yt[i]) ** 2))) for i in range(len(yt)))
            for j, col in enumerate(ycols):
                domain_rows.append(dict(dataset=dataset, model=name, validation_domain=col.replace("metric/the_pile_", "").replace("_val_loss", ""),
                                        n=len(yt), rmse=float(np.sqrt(mean_squared_error(yt[:, j], pred[:, j]))),
                                        bias=float(np.mean(pred[:, j] - yt[:, j])),
                                        spearman=float(spearmanr(yt[:, j], pred[:, j]).statistic)))
            if name != "v2":
                gain, lo, hi = bootstrap_gain(yt, predictions["v2"], pred, SEED + 100 + len(gain_rows))
                gain_rows.append(dict(dataset=dataset, model=name, n=len(yt),
                                      v2_minus_candidate_rmse=gain, ci_low=lo, ci_high=hi))
                rgain, rlo, rhi = rank_bootstrap_gain(yt, predictions["v2"], pred, SEED + 600 + len(rank_rows))
                rank_rows.append(dict(dataset=dataset, model=name, n=len(yt),
                                      candidate_minus_v2_mean_spearman=rgain, ci_low=rlo, ci_high=rhi))
    pd.DataFrame(all_metrics).to_csv(OUT / "metrics_by_scale.csv", index=False)
    pd.DataFrame(gain_rows).to_csv(OUT / "paired_bootstrap_gain.csv", index=False)
    pd.DataFrame(rank_rows).to_csv(OUT / "rank_bootstrap_gain.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(OUT / "predictions_by_mixture.csv", index=False)
    pd.DataFrame(domain_rows).to_csv(OUT / "per_domain_metrics.csv", index=False)
    pd.DataFrame(support_rows).to_csv(OUT / "composition_support.csv", index=False)
    manifest_paths = source_paths + [P1 / "tables" / "v2_nested_cv_folds.csv", P1 / "models" / "mixture_v2.joblib"]
    manifest = {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in manifest_paths}
    (OUT / "input_sha256.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(pd.DataFrame(all_metrics)[["dataset", "model", "rmse", "domain_spearman_mean", "mixture_mean_spearman"]].to_string(index=False))


if __name__ == "__main__":
    main()
