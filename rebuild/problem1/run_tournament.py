"""Reproducible, retrospective Problem 1 Champion–Challenger tournament.

Run from the project root: python rebuild/problem1/run_tournament.py
All writes are confined to rebuild/problem1. Test labels never enter selection.
"""
from __future__ import annotations

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import hashlib
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.linalg import helmert
from scipy.stats import spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import MultiTaskElasticNet, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = ROOT / "real_attachments" / "A_data_value" / "regmix_tables"
BASELINE = ROOT / "problem1" / "outputs"
SEED = 20260923
SCALES = (("test_1m", "test", "1m"), ("test_60m", "test", "60m"),
          ("test_1B", "test", "1B"), ("est_10b", "est", "10b"),
          ("est_70b", "est", "70b"))
FAMILIES = ("v2_champion", "alr_elasticnet", "ilr_ridge", "alr_pls", "clr_spline")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair(prefix: str, scale: str):
    pm = DATA / f"{prefix}_mixture_{scale}.csv"
    py = DATA / f"{prefix}_pile_loss_{scale}.csv"
    mix, loss = pd.read_csv(pm), pd.read_csv(py)
    if mix["index"].duplicated().any() or loss["index"].duplicated().any():
        raise ValueError("duplicate mixture index")
    d = mix.merge(loss, on="index", how="outer", validate="one_to_one", indicator=True)
    if len(d) != len(mix) or len(d) != len(loss) or not d._merge.eq("both").all():
        raise ValueError("mixture/loss rows do not pair exactly")
    pcols = [c for c in mix if c.startswith("train_the_pile_")]
    ycols = [c for c in loss if c.startswith("metric/the_pile_")]
    p, y = d[pcols].to_numpy(float), d[ycols].to_numpy(float)
    if len(pcols) != 17 or len(ycols) != 13 or np.any(p < 0) or not np.isfinite(p).all():
        raise ValueError("unexpected or invalid simplex")
    if not np.allclose(p.sum(axis=1), 1, atol=.005) or not np.isfinite(y).all():
        raise ValueError("invalid simplex or target")
    p /= p.sum(axis=1, keepdims=True)
    return d["index"].to_numpy(), p, y, pcols, ycols, (pm, py)


def coordinates(p: np.ndarray, family: str, eps: float, ref: int) -> np.ndarray:
    q = p + eps
    q /= q.sum(axis=1, keepdims=True)
    z = np.log(q)
    if family in ("v2_champion", "alr_elasticnet", "alr_pls"):
        return z[:, np.arange(p.shape[1]) != ref] - z[:, ref, None]
    if family == "ilr_ridge":
        return z @ helmert(p.shape[1], full=False).T
    if family == "clr_spline":
        return z - z.mean(axis=1, keepdims=True)
    raise ValueError(family)


def specs(family: str):
    if family == "alr_elasticnet":
        return [dict(alpha=a, l1_ratio=r) for a in (.001, .01, .1) for r in (.2, .7)]
    if family == "ilr_ridge":
        return [dict(eps_mult=e, alpha=a) for e in (.25, .5) for a in (1., 10., 100.)]
    if family == "alr_pls":
        return [dict(n_components=n) for n in (2, 4, 8, 12)]
    if family == "clr_spline":
        return [dict(eps_mult=e, n_knots=k, alpha=a)
                for e in (.25, .5) for k in (4, 6) for a in (1., 10., 100.)]
    raise ValueError(family)


def estimator(family: str, spec: dict):
    if family == "v2_champion":
        return make_pipeline(PolynomialFeatures(2, include_bias=False),
                             StandardScaler(), Ridge(alpha=float(spec["alpha"])))
    if family == "alr_elasticnet":
        return make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(),
                             MultiTaskElasticNet(alpha=float(spec["alpha"]),
                                                 l1_ratio=float(spec["l1_ratio"]),
                                                 max_iter=10000, tol=1e-4, random_state=SEED))
    if family == "ilr_ridge":
        return make_pipeline(PolynomialFeatures(2, include_bias=False),
                             StandardScaler(), Ridge(alpha=float(spec["alpha"])))
    if family == "alr_pls":
        return make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(),
                             PLSRegression(n_components=int(spec["n_components"]), scale=False))
    if family == "clr_spline":
        return make_pipeline(SplineTransformer(n_knots=int(spec["n_knots"]), degree=3,
                                               knots="quantile", extrapolation="linear",
                                               include_bias=False), StandardScaler(),
                             Ridge(alpha=float(spec["alpha"])))
    raise ValueError(family)


def fit(family: str, spec: dict, p: np.ndarray, y: np.ndarray,
        baseline_eps: float, baseline_ref: int):
    eps = (float(p[p > 0].min()) * float(spec["eps_mult"])
           if "eps_mult" in spec else baseline_eps)
    ref = int(spec.get("ref", baseline_ref))
    m = estimator(family, spec)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        m.fit(coordinates(p, family, eps, ref), y)
    return dict(family=family, spec=spec, eps=eps, ref=ref, model=m)


def predict(bundle: dict, p: np.ndarray) -> np.ndarray:
    return np.asarray(bundle["model"].predict(coordinates(
        p, bundle["family"], bundle["eps"], bundle["ref"])), float)


def rmse(y, pred) -> float:
    return float(np.sqrt(np.mean((pred-y)**2)))


def rank(x, y) -> float:
    r = spearmanr(x, y).statistic
    return float(r) if np.isfinite(r) else np.nan


def metrics(y, pred) -> dict:
    return dict(rmse=rmse(y, pred), mae=float(np.mean(np.abs(pred-y))),
                bias=float(np.mean(pred-y)),
                spearman_mean=float(np.nanmean([rank(y[:, j], pred[:, j])
                                                for j in range(y.shape[1])])),
                mixture_mean_spearman=rank(y.mean(axis=1), pred.mean(axis=1)))


def bootstrap_gain(y, base, new, seed: int, draws: int = 2000) -> dict:
    old = np.mean((base-y)**2, axis=1)
    alt = np.mean((new-y)**2, axis=1)
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(y), size=(draws, len(y)))
    delta = np.sqrt(old[ix].mean(axis=1)) - np.sqrt(alt[ix].mean(axis=1))
    return dict(v2_minus_candidate_rmse=float(np.sqrt(old.mean())-np.sqrt(alt.mean())),
                ci_low=float(np.quantile(delta, .025)),
                ci_high=float(np.quantile(delta, .975)),
                probability_positive=float(np.mean(delta > 0)),
                bootstrap_draws=draws, bootstrap_unit="mixture row")


def rank_gain(y, base, new, seed: int, draws: int = 600) -> dict:
    actual = y.mean(axis=1)
    b, c = base.mean(axis=1), new.mean(axis=1)
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(y), size=(draws, len(y)))
    delta = np.array([rank(actual[ii], c[ii])-rank(actual[ii], b[ii]) for ii in ix])
    return dict(candidate_minus_v2_rank=float(rank(actual, c)-rank(actual, b)),
                rank_ci_low=float(np.nanquantile(delta, .025)),
                rank_ci_high=float(np.nanquantile(delta, .975)))


def inner_choose(family: str, p, y, splits, base_eps, base_ref):
    rows = []
    for spec in specs(family):
        errors = []
        for tr, va in splits:
            m = fit(family, spec, p[tr], y[tr], base_eps, base_ref)
            errors.append(float(mean_squared_error(y[va], predict(m, p[va]))))
        rows.append(dict(family=family, spec=spec, mean_mse=float(np.mean(errors)),
                         fold_rmse_sd=float(np.std(np.sqrt(errors), ddof=1))))
    best = min(rows, key=lambda r: r["mean_mse"])
    return best, rows


def substitution(bundle, center, donor: int):
    base = float(predict(bundle, center[None, :]).mean())
    vals = np.full(len(center), np.nan)
    for j in range(len(center)):
        if j == donor:
            continue
        shifted = center.copy()
        shifted[j] += .01
        shifted[donor] -= .01
        vals[j] = float(predict(bundle, shifted[None, :]).mean()-base)
    return vals


def active_features(bundle) -> int:
    family = bundle["family"]
    if family == "alr_elasticnet":
        coef = bundle["model"].steps[-1][1].coef_
        return int(np.sum(np.linalg.norm(coef, axis=0) > 1e-8))
    if family == "clr_spline":
        return int(bundle["model"].steps[0][1].n_features_out_)
    if family == "alr_pls":
        return int(bundle["spec"]["n_components"])
    return 152


def write(df, filename):
    pd.DataFrame(df).to_csv(OUT / filename, index=False, encoding="utf-8-sig")


def main():
    OUT.mkdir(exist_ok=True)
    manifest = json.loads((ROOT / "model_baseline_manifest.json").read_text(encoding="utf-8"))
    old_path = BASELINE / "models" / "mixture_v2.joblib"
    old_rel = "problem1/outputs/models/mixture_v2.joblib"
    if sha256(old_path) != manifest["files"][old_rel]["sha256"]:
        raise RuntimeError("V2 artifact changed since baseline freeze")
    ids, p, y, pcols, ycols, files = pair("train", "1m")
    if p.shape != (512, 17) or y.shape != (512, 13):
        raise ValueError("unexpected training dimensions")
    old = joblib.load(old_path)
    if old["family"] != "poly_alr" or old["pcols"] != pcols or old["ycols"] != ycols:
        raise ValueError("V2 artifact incompatible with current data")
    base_eps, base_ref = float(old["eps"]), int(old["reference_index"])
    old_folds = pd.read_csv(BASELINE / "tables" / "v2_nested_cv_folds.csv").set_index("outer_fold")
    oof = {family: np.full_like(y, np.nan) for family in FAMILIES}
    nested_rows, search_rows = [], []
    outer_bundles = {f: [] for f in FAMILIES}
    center = p.mean(axis=0)
    center /= center.sum()
    donor = int(center.argmax())
    if center[donor] <= .01:
        raise ValueError("invalid 1pp substitution center")
    for fold, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=SEED+1).split(p), 1):
        prior = old_folds.loc[fold]
        if prior.selected_family != "poly_alr":
            raise ValueError("frozen V2 fold no longer quadratic ALR")
        v2 = fit("v2_champion", dict(alpha=float(prior.selected_alpha),
                                    ref=int(prior.selected_ref)), p[tr], y[tr],
                 float(prior.selected_eps), int(prior.selected_ref))
        oof["v2_champion"][te] = predict(v2, p[te])
        assert np.isclose(rmse(y[te], oof["v2_champion"][te]), prior.v2_outer_rmse, atol=1e-9)
        outer_bundles["v2_champion"].append(v2)
        nested_rows.append(dict(outer_fold=fold, family="v2_champion", n=len(te),
                                rmse=rmse(y[te], oof["v2_champion"][te]),
                                selected_spec=json.dumps(v2["spec"], sort_keys=True)))
        inner = list(KFold(3, shuffle=True, random_state=SEED+fold).split(tr))
        # ALR reference/pseudocount for this outer fold come solely from its
        # frozen V2 train-only inner selection, never from the full-data artifact.
        fold_eps, fold_ref = float(prior.selected_eps), int(prior.selected_ref)
        for family in FAMILIES[1:]:
            selected, searched = inner_choose(family, p[tr], y[tr], inner, fold_eps, fold_ref)
            for row in searched:
                search_rows.append(dict(outer_fold=fold, family=family,
                                        spec=json.dumps(row["spec"], sort_keys=True),
                                        inner_rmse=float(np.sqrt(row["mean_mse"])),
                                        fold_rmse_sd=row["fold_rmse_sd"],
                                        selected=row is selected))
            b = fit(family, selected["spec"], p[tr], y[tr], fold_eps, fold_ref)
            outer_bundles[family].append(b)
            oof[family][te] = predict(b, p[te])
            nested_rows.append(dict(outer_fold=fold, family=family, n=len(te),
                                    rmse=rmse(y[te], oof[family][te]),
                                    selected_spec=json.dumps(selected["spec"], sort_keys=True)))
        print(f"outer fold {fold}/5 finished", flush=True)
    if any(not np.isfinite(v).all() for v in oof.values()):
        raise RuntimeError("incomplete OOF predictions")
    write(nested_rows, "nested_outer_folds.csv")
    write(search_rows, "nested_inner_search.csv")

    full_splits = list(RepeatedKFold(n_splits=5, n_repeats=2, random_state=SEED).split(p))
    final = {}
    full_search = []
    for family in FAMILIES[1:]:
        selected, searched = inner_choose(family, p, y, full_splits, base_eps, base_ref)
        for row in searched:
            full_search.append(dict(family=family, spec=json.dumps(row["spec"], sort_keys=True),
                                    cv_rmse=float(np.sqrt(row["mean_mse"])),
                                    fold_rmse_sd=row["fold_rmse_sd"],
                                    selected=row is selected))
        final[family] = fit(family, selected["spec"], p, y, base_eps, base_ref)
        joblib.dump(dict(**final[family], pcols=pcols, ycols=ycols,
                         training_scope="A4/A5 1M only"), OUT / f"{family}.joblib")
        print(f"full selection {family}: {selected['spec']}", flush=True)
    write(full_search, "full_train_search.csv")
    final["v2_champion"] = dict(family="v2_champion", spec=dict(alpha=float(old["model"].steps[-1][1].alpha)),
                                eps=base_eps, ref=base_ref, model=old["model"])
    if not np.isclose(rmse(y, predict(final["v2_champion"], p)),
                      rmse(y, old["model"].predict(coordinates(p, "v2_champion", base_eps, base_ref)))):
        raise RuntimeError("V2 artifact prediction mismatch")

    metrics_rows, domain_rows, boot_rows, pred_rows = [], [], [], []
    datasets = [("train_nested_oof", ids, p, y, "train_only_nested")]
    input_files = [*files]
    for name, prefix, scale in SCALES:
        di, dp, dy, pc, yc, paths = pair(prefix, scale)
        if pc != pcols or yc != ycols:
            raise ValueError("cross-scale columns changed")
        datasets.append((name, di, dp, dy, "estimated_label" if prefix == "est" else "retrospective_observed"))
        input_files.extend(paths)
    dataset_preds = {}
    for di, (name, mix_ids, dp, dy, status) in enumerate(datasets):
        pp = {family: (oof[family] if name == "train_nested_oof" else predict(final[family], dp))
              for family in FAMILIES}
        dataset_preds[name] = (dy, pp)
        for family in FAMILIES:
            pred = pp[family]
            metrics_rows.append(dict(dataset=name, evidence_status=status, family=family,
                                     n_mixtures=len(dy), **metrics(dy, pred)))
            for j, col in enumerate(ycols):
                domain_rows.append(dict(dataset=name, family=family,
                                        validation_domain=col.replace("metric/the_pile_", "").replace("_val_loss", ""),
                                        rmse=rmse(dy[:, j], pred[:, j]),
                                        mae=float(np.mean(np.abs(pred[:, j]-dy[:, j]))),
                                        bias=float(np.mean(pred[:, j]-dy[:, j])),
                                        spearman=rank(dy[:, j], pred[:, j])))
            mse_row = np.mean((pred-dy)**2, axis=1)
            for i, idx in enumerate(mix_ids):
                pred_rows.append(dict(dataset=name, family=family, index=idx,
                                      actual_mean_loss=float(dy[i].mean()),
                                      predicted_mean_loss=float(pred[i].mean()),
                                      mixture_mse=float(mse_row[i])))
            if family != "v2_champion":
                gain = bootstrap_gain(dy, pp["v2_champion"], pred, SEED+1000+di*10+FAMILIES.index(family))
                rg = rank_gain(dy, pp["v2_champion"], pred, SEED+2000+di*10+FAMILIES.index(family))
                boot_rows.append(dict(dataset=name, family=family, n_mixtures=len(dy),
                                      **gain, **rg))
    write(metrics_rows, "metrics_by_scale.csv")
    write(domain_rows, "metrics_by_domain.csv")
    write(boot_rows, "paired_bootstrap_gain.csv")
    write(pred_rows, "predictions_by_mixture.csv")

    effect_rows, effect_summary = [], []
    all_effects = {}
    rng = np.random.default_rng(SEED+3000)
    for family in FAMILIES:
        full = substitution(final[family], center, donor)
        fold_effects = np.stack([substitution(b, center, donor) for b in outer_bundles[family]])
        boot_effects = []
        active = []
        for _ in range(40):
            ii = rng.integers(0, len(p), len(p))
            b = fit(family, final[family]["spec"], p[ii], y[ii],
                    final[family]["eps"], final[family]["ref"])
            boot_effects.append(substitution(b, center, donor))
            active.append(active_features(b))
        boot_effects = np.stack(boot_effects)
        all_effects[family] = full
        consistency = []
        for j, col in enumerate(pcols):
            if j == donor:
                continue
            sign_rate = float(np.mean(np.sign(boot_effects[:, j]) == np.sign(full[j])))
            fold_rate = float(np.mean(np.sign(fold_effects[:, j]) == np.sign(full[j])))
            consistency.append(sign_rate)
            effect_rows.append(dict(family=family, increase=col.replace("train_the_pile_", ""),
                                    decrease=pcols[donor].replace("train_the_pile_", ""),
                                    predicted_mean_loss_change_1pp=float(full[j]),
                                    bootstrap_ci_low=float(np.quantile(boot_effects[:, j], .025)),
                                    bootstrap_ci_high=float(np.quantile(boot_effects[:, j], .975)),
                                    bootstrap_same_sign_rate=sign_rate,
                                    outer_fold_same_sign_rate=fold_rate))
        effect_summary.append(dict(family=family, n_targets=len(consistency),
                                   n_bootstrap_fits=40,
                                   stable_sign_targets_ge_0_9=int(np.sum(np.array(consistency) >= .9)),
                                   mean_sign_consistency=float(np.mean(consistency)),
                                   active_features_full=active_features(final[family]),
                                   active_features_bootstrap_median=float(np.median(active))))
        print(f"substitution bootstrap {family} finished", flush=True)
    write(effect_rows, "substitution_effects.csv")
    write(effect_summary, "substitution_stability.csv")

    metric_df = pd.DataFrame(metrics_rows).set_index(["dataset", "family"])
    boot_df = pd.DataFrame(boot_rows).set_index(["dataset", "family"])
    effect_df = pd.DataFrame(effect_summary).set_index("family")
    base_folds = pd.DataFrame(nested_rows).query("family == 'v2_champion'").rmse.to_numpy()
    cv_se = float(base_folds.std(ddof=1)/np.sqrt(len(base_folds)))
    tournament = []
    for family in FAMILIES:
        row = dict(family=family, role="frozen_champion" if family == "v2_champion" else "challenger",
                   primary_test_1m_rmse=float(metric_df.loc[("test_1m", family), "rmse"]),
                   nested_oof_rmse=float(metric_df.loc[("train_nested_oof", family), "rmse"]),
                   test_60m_rmse=float(metric_df.loc[("test_60m", family), "rmse"]),
                   test_1B_rmse=float(metric_df.loc[("test_1B", family), "rmse"]),
                   test_1m_mae=float(metric_df.loc[("test_1m", family), "mae"]),
                   test_1m_spearman_mean=float(metric_df.loc[("test_1m", family), "spearman_mean"]),
                   test_1B_mixture_mean_spearman=float(metric_df.loc[("test_1B", family), "mixture_mean_spearman"]),
                   active_features=int(effect_df.loc[family, "active_features_full"]),
                   stable_1pp_targets=int(effect_df.loc[family, "stable_sign_targets_ge_0_9"]),
                   selected_spec=json.dumps(final[family]["spec"], sort_keys=True))
        base_effect = all_effects["v2_champion"]
        candidate_effect = all_effects[family]
        valid = np.isfinite(base_effect)
        row["effect_sign_agreement_with_v2"] = int(np.sum(
            np.sign(base_effect[valid]) == np.sign(candidate_effect[valid])))
        row["effect_rank_spearman_with_v2"] = rank(base_effect[valid], candidate_effect[valid])
        if family == "v2_champion":
            row.update(primary_gain_ci_low=np.nan, primary_gain_ci_high=np.nan,
                       scale_60m_gain_ci_high=np.nan, scale_1B_gain_ci_high=np.nan,
                       eligible_to_replace=False, decision="RETAIN_PENDING_CHALLENGER_COMPARISON")
        else:
            p1 = boot_df.loc[("test_1m", family)]
            p60 = boot_df.loc[("test_60m", family)]
            p1b = boot_df.loc[("test_1B", family)]
            row.update(primary_gain_ci_low=float(p1.ci_low), primary_gain_ci_high=float(p1.ci_high),
                       scale_60m_gain_ci_high=float(p60.ci_high),
                       scale_1B_gain_ci_high=float(p1b.ci_high))
            eligible = bool(p1.ci_low > 0 and
                            row["nested_oof_rmse"] <= float(metric_df.loc[("train_nested_oof", "v2_champion"), "rmse"])+cv_se and
                            p60.ci_high >= 0 and p1b.ci_high >= 0 and
                            boot_df.loc[("test_1B", family), "rank_ci_high"] >= 0 and
                            row["effect_sign_agreement_with_v2"] >= 13)
            row["eligible_to_replace"] = eligible
            row["decision"] = "ELIGIBLE" if eligible else "REJECT_CROSS_SCALE_UPGRADE"
        tournament.append(row)
    tournament_df = pd.DataFrame(tournament)
    eligible_df = tournament_df[tournament_df.eligible_to_replace]
    if len(eligible_df):
        # Predeclared one-SE simplification among eligible models.
        best_cv = float(eligible_df.nested_oof_rmse.min())
        candidates = eligible_df[eligible_df.nested_oof_rmse <= best_cv + cv_se]
        chosen = candidates.sort_values(["active_features", "nested_oof_rmse"]).iloc[0].family
        tournament_df.loc[tournament_df.family == chosen, "decision"] = "NEW_CROSS_QUESTION_CHAMPION"
        tournament_df.loc[tournament_df.family == "v2_champion", "decision"] = "REPLACED"
    else:
        chosen = "v2_champion"
        tournament_df.loc[tournament_df.family == "v2_champion", "decision"] = "RETAINED_AFTER_RECHALLENGE"
    tournament_df.to_csv(OUT / "problem1_model_tournament.csv", index=False, encoding="utf-8-sig")
    metadata = dict(frozen_v2_sha256=sha256(old_path), baseline_manifest_sha256=sha256(ROOT / "model_baseline_manifest.json"),
                    data_sha256={str(path.relative_to(ROOT)): sha256(path) for path in input_files},
                    old_fold_table_sha256=sha256(BASELINE / "tables" / "v2_nested_cv_folds.csv"),
                    primary_metric="A6/A7 1M pooled 13-output RMSE",
                    test_status="retrospective; no new blind test",
                    outer_split="KFold(5, shuffle=True, random_state=20260924)",
                    inner_split="KFold(3, shuffle=True, random_state=20260923+outer_fold)",
                    full_selection="RepeatedKFold(5, 2, random_state=20260923)",
                    baseline_outer_fold_rmse_se=cv_se,
                    winner=chosen, q_labels=int(len(pd.read_csv(BASELINE / "tables" / "manual_validation_results.csv"))))
    (OUT / "run_manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(tournament_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
