"""Independent, preregistered Problem 2 model tournament.

Run from any cwd: python rebuild/problem2/run_tournament.py
Only this rebuild/problem2 directory is written. All fitted numbers come from raw B
attachments, never from the frozen problem2/problem2_v2 output tables.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
INPUT = ROOT / "real_attachments" / "B_scaling_laws"
SEED = 20260925
FILES = {
    "B1": "pythia_training_log_existing.csv",
    "B2": "cerebras_training_log.csv",
    "B4": "scaling_baseline.csv",
    "B5": "published_scaling_data.csv",
    "B6": "supplementary_NQ_experiment.csv",
    "B7": "supplementary_NQ_experiment_expanded.csv",
    "B8": "supplementary_NQ_experiment_large.csv",
}
ND_KINDS = ("classic", "interaction", "broken_N")
Q_KINDS = ("additive", "effective_tokens", "Q_by_N", "Q_by_D", "Q_by_ND")
KEYS = ["N_params_B", "D_tokens_B", "Q_score"]


def save_csv(df: pd.DataFrame, name: str) -> None:
    df.to_csv(HERE / name, index=False, float_format="%.12g")


def data() -> dict[str, pd.DataFrame]:
    return {key: pd.read_csv(INPUT / name) for key, name in FILES.items()}


def novel(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    seen = old[KEYS].drop_duplicates().assign(_seen=1)
    joined = new.merge(seen, on=KEYS, how="left", validate="many_to_one")
    return joined.loc[joined._seen.isna()].drop(columns="_seen").copy()


def arrays(df: pd.DataFrame):
    return tuple(df[c].to_numpy(float) for c in ["N_params_B", "D_tokens_B", "val_loss"])


def quality_arrays(df: pd.DataFrame):
    return tuple(df[c].to_numpy(float) for c in ["N_params_B", "D_tokens_B", "Q_score", "val_loss"])


def rmse(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def measures(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    rho = float(spearmanr(y, p).statistic) if len(y) > 2 and np.std(y) and np.std(p) else np.nan
    return dict(n=len(y), rmse=rmse(y, p), mae=float(np.mean(np.abs(y - p))),
                bias=float(np.mean(p - y)), spearman=rho)


def predict_nd(kind, theta, n, d):
    n, d = np.broadcast_arrays(np.asarray(n, float), np.asarray(d, float))
    e, a, alpha, b, beta = theta[:5]
    core = e + a * n ** (-alpha) + b * d ** (-beta)
    if kind == "classic":
        return core
    if kind == "interaction":
        return core + theta[5] * n ** (-alpha) * d ** (-beta)
    if kind == "broken_N":
        slope_delta, nc = theta[5:7]
        return e + a * n ** (-alpha) * (1 + (n / nc) ** 2) ** (-slope_delta / 2) + b * d ** (-beta)
    raise KeyError(kind)


def fit_nd(kind, df):
    n, d, y = arrays(df)
    lower = [0, 1e-6, .005, 1e-6, .005]
    upper = [4, 100, 2, 100, 2]
    start = [1.69, .35, .34, 1.24, .28]
    if kind == "interaction":
        lower += [-5]; upper += [5]; start += [0]
    elif kind == "broken_N":
        lower += [-.25, .08]; upper += [.25, 12]; start += [0, 2.8]
    result = least_squares(lambda z: predict_nd(kind, z, n, d) - y, start,
                           bounds=(lower, upper), max_nfev=2500, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    if not result.success and np.linalg.norm(result.fun) > 1e-2:
        raise RuntimeError(f"{kind} N-D fit failed: {result.message}")
    return result


def predict_q(kind, theta, n, d, q, ndtheta):
    n, d, q = np.broadcast_arrays(np.asarray(n, float), np.asarray(d, float), np.asarray(q, float))
    base = predict_nd("classic", ndtheta, n, d)
    if kind == "effective_tokens":
        e, a, alpha, b, beta = ndtheta
        return e + a * n ** (-alpha) + b * (d * q ** theta[0]) ** (-beta)
    g, kappa = theta[:2]
    penalty = g * (1 - q) ** kappa
    if kind == "additive":
        return base + penalty
    if kind == "Q_by_N":
        return base + penalty * n ** (-theta[2])
    if kind == "Q_by_D":
        return base + penalty * (d / 100) ** (-theta[2])
    if kind == "Q_by_ND":
        return base + penalty * n ** (-theta[2]) * (d / 100) ** (-theta[3])
    raise KeyError(kind)


def fit_q(kind, df, ndtheta):
    n, d, q, y = quality_arrays(df)
    if kind == "effective_tokens":
        start, lower, upper = [1], [0], [15]
    elif kind == "additive":
        start, lower, upper = [.36, 1], [0, .1], [5, 4]
    elif kind in ("Q_by_N", "Q_by_D"):
        start, lower, upper = [.36, 1, .15], [0, .1, -1], [5, 4, 1]
    else:
        start, lower, upper = [.36, 1, .15, 0], [0, .1, -1, -1], [5, 4, 1, 1]
    result = least_squares(lambda z: predict_q(kind, z, n, d, q, ndtheta) - y, start,
                           bounds=(lower, upper), max_nfev=2500, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    if not result.success and np.linalg.norm(result.fun) > 1:
        raise RuntimeError(f"{kind} Q fit failed: {result.message}")
    return result


def bic(y, p, k):
    n = len(y)
    return float(n * np.log(np.mean((np.asarray(y) - np.asarray(p)) ** 2)) + k * np.log(n))


def cond(result):
    return float(np.linalg.cond(result.jac.T @ result.jac))


def grouped_delta_interval(oof: pd.DataFrame, challenger: str, base: str,
                           group_col: str, replicates=2000):
    """Paired group bootstrap of RMSE(base) - RMSE(challenger)."""
    a = oof[oof.model == base][["row_id", group_col, "observed", "predicted"]].rename(columns={"predicted": "base"})
    b = oof[oof.model == challenger][["row_id", "predicted"]].rename(columns={"predicted": "challenger"})
    joined = a.merge(b, on="row_id", validate="one_to_one")
    groups = sorted(joined[group_col].unique())
    by_group = {g: joined[joined[group_col] == g] for g in groups}
    rng = np.random.default_rng(SEED + sum(map(ord, challenger)))
    differences = []
    for _ in range(replicates):
        draw = rng.choice(groups, len(groups), replace=True)
        sample = pd.concat([by_group[g] for g in draw], ignore_index=True)
        differences.append(rmse(sample.observed, sample.base) - rmse(sample.observed, sample.challenger))
    return np.quantile(differences, [.025, .5, .975]).tolist()


def nd_tournament(d):
    b1 = d["B1"].reset_index(drop=True)
    groups = sorted(b1.N_params_B.unique())
    summary, out_of_fold, external, extrapolation, parameters = [], [], [], [], []
    fits = {}
    for kind in ND_KINDS:
        fit = fit_nd(kind, b1)
        fits[kind] = fit.x
        n, dtok, y = arrays(b1)
        pred = predict_nd(kind, fit.x, n, dtok)
        line = dict(model=kind, n_parameters=len(fit.x), B1_training_rmse=rmse(y, pred),
                    B1_bic=bic(y, pred, len(fit.x)), jacobian_condition=cond(fit),
                    parameters=json.dumps(fit.x.tolist()))
        for held in groups:
            tr = b1[b1.N_params_B != held]
            te = b1[b1.N_params_B == held]
            ff = fit_nd(kind, tr)
            pn = predict_nd(kind, ff.x, te.N_params_B, te.D_tokens_B)
            for rid, obs, pr in zip(te.index, te.val_loss, pn):
                out_of_fold.append(dict(model=kind, row_id=int(rid), held_N_B=float(held),
                                        observed=float(obs), predicted=float(pr)))
        for src in ("B2", "B4", "B5"):
            x = d[src]
            n, dtok, y = arrays(x)
            m = measures(y, predict_nd(kind, fit.x, n, dtok))
            external.append(dict(model=kind, source=src, protocol="zero_shot_raw", **m))
            line[f"{src}_raw_rmse"] = m["rmse"]
        for fraction in (.5, .75):
            ng = max(3, int(np.floor(len(groups) * fraction)))
            tr = b1[b1.N_params_B.isin(groups[:ng])]
            te = b1[~b1.N_params_B.isin(groups[:ng])]
            ff = fit_nd(kind, tr)
            pred = predict_nd(kind, ff.x, te.N_params_B, te.D_tokens_B)
            extrapolation.append(dict(model=kind, axis="small_N_to_large_N", fraction=fraction,
                                      max_train_N_B=max(groups[:ng]), **measures(te.val_loss, pred)))
            masks = []
            for _, g in b1.groupby("N_params_B"):
                masks.append(g.D_tokens_B <= g.D_tokens_B.quantile(fraction))
            mask = pd.concat(masks).sort_index()
            tr, te = b1[mask], b1[~mask]
            ff = fit_nd(kind, tr)
            pred = predict_nd(kind, ff.x, te.N_params_B, te.D_tokens_B)
            extrapolation.append(dict(model=kind, axis="early_D_to_late_D", fraction=fraction,
                                      max_train_N_B=np.nan, **measures(te.val_loss, pred)))
        parameters.append(dict(model=kind, **{f"theta_{i}": float(v) for i, v in enumerate(fit.x)},
                               jacobian_condition=cond(fit)))
        summary.append(line)
    oof = pd.DataFrame(out_of_fold)
    for row in summary:
        z = oof[oof.model == row["model"]]
        row["B1_group_cv_rmse"] = rmse(z.observed, z.predicted)
        if row["model"] != "classic":
            ci = grouped_delta_interval(oof, row["model"], "classic", "held_N_B")
            row["B1_paired_improvement_ci_low"], row["B1_paired_improvement_median"], row["B1_paired_improvement_ci_high"] = ci
        else:
            row["B1_paired_improvement_ci_low"] = row["B1_paired_improvement_median"] = row["B1_paired_improvement_ci_high"] = 0
    score = pd.DataFrame(summary)
    save_csv(score, "nd_scorecard.csv")
    save_csv(oof, "nd_group_oof.csv")
    save_csv(pd.DataFrame(external), "nd_external_raw.csv")
    save_csv(pd.DataFrame(extrapolation), "nd_extrapolation.csv")
    save_csv(pd.DataFrame(parameters), "nd_parameters.csv")
    return score, fits


def q_tournament(d, ndtheta):
    b6 = d["B6"].reset_index(drop=True)
    b7_new = novel(d["B6"], d["B7"])
    b8_new = novel(d["B7"], d["B8"])
    groups = sorted(b6.N_params_B.unique())
    summary, oof_rows, predictions, bootstrap_rows = [], [], [], []
    fits = {}
    for kind in Q_KINDS:
        fit = fit_q(kind, b6, ndtheta)
        fits[kind] = fit.x
        n, dtok, q, y = quality_arrays(b6)
        pred = predict_q(kind, fit.x, n, dtok, q, ndtheta)
        line = dict(model=kind, n_parameters=len(fit.x), B6_training_rmse=rmse(y, pred),
                    B6_bic=bic(y, pred, len(fit.x)), jacobian_condition=cond(fit),
                    parameters=json.dumps(fit.x.tolist()))
        for held in groups:
            tr = b6[b6.N_params_B != held]
            te = b6[b6.N_params_B == held]
            ff = fit_q(kind, tr, ndtheta)
            nn, dd, qq, yy = quality_arrays(te)
            pp = predict_q(kind, ff.x, nn, dd, qq, ndtheta)
            for rid, obs, pr in zip(te.index, yy, pp):
                oof_rows.append(dict(model=kind, row_id=int(rid), held_N_B=float(held),
                                     observed=float(obs), predicted=float(pr)))
        for src, x in (("B7_new", b7_new), ("B8_new_diagnostic", b8_new)):
            nn, dd, qq, yy = quality_arrays(x)
            pp = predict_q(kind, fit.x, nn, dd, qq, ndtheta)
            m = measures(yy, pp)
            predictions.append(dict(model=kind, source=src, **m))
            line[f"{src}_rmse"] = m["rmse"]
        summary.append(line)
    oof = pd.DataFrame(oof_rows)
    for row in summary:
        z = oof[oof.model == row["model"]]
        row["B6_leave_N_rmse"] = rmse(z.observed, z.predicted)
    # B7 group bootstrap compares full B6-fitted predictions for equal B7 rows.
    b7 = b7_new.reset_index(drop=True)
    for kind in Q_KINDS:
        n, dtok, q, y = quality_arrays(b7)
        b7[kind] = predict_q(kind, fits[kind], n, dtok, q, ndtheta)
    rng = np.random.default_rng(SEED + 771)
    b7_groups = sorted(b7.N_params_B.unique())
    pieces = {g: b7[b7.N_params_B == g] for g in b7_groups}
    for row in summary:
        kind = row["model"]
        if kind == "Q_by_N":
            row["B7_vs_champion_ci_low"] = row["B7_vs_champion_ci_median"] = row["B7_vs_champion_ci_high"] = 0
            continue
        deltas = []
        for _ in range(2000):
            draw = rng.choice(b7_groups, len(b7_groups), replace=True)
            sample = pd.concat([pieces[g] for g in draw], ignore_index=True)
            deltas.append(rmse(sample.val_loss, sample.Q_by_N) - rmse(sample.val_loss, sample[kind]))
        row["B7_vs_champion_ci_low"], row["B7_vs_champion_ci_median"], row["B7_vs_champion_ci_high"] = np.quantile(deltas, [.025, .5, .975])
    # Refit by N-cluster resampling to test added exponent stability.
    rng = np.random.default_rng(SEED + 661)
    for kind in ("Q_by_N", "Q_by_D", "Q_by_ND"):
        for rep in range(200):
            draw = rng.choice(groups, len(groups), replace=True)
            sample = pd.concat([b6[b6.N_params_B == g] for g in draw], ignore_index=True)
            ff = fit_q(kind, sample, ndtheta)
            rec = dict(model=kind, replicate=rep, G=float(ff.x[0]), kappa=float(ff.x[1]),
                       eta_N=float(ff.x[2]) if kind != "Q_by_D" else np.nan,
                       eta_D=float(ff.x[2]) if kind == "Q_by_D" else float(ff.x[3]) if kind == "Q_by_ND" else np.nan)
            bootstrap_rows.append(rec)
    score = pd.DataFrame(summary)
    bootstrap = pd.DataFrame(bootstrap_rows)
    intervals = []
    for kind, part in bootstrap.groupby("model"):
        for col in ("G", "kappa", "eta_N", "eta_D"):
            x = part[col].dropna()
            if not x.empty:
                intervals.append(dict(model=kind, parameter=col, ci_low=x.quantile(.025),
                                      median=x.median(), ci_high=x.quantile(.975)))
    save_csv(score, "quality_scorecard.csv")
    save_csv(oof, "quality_leave_N_oof.csv")
    save_csv(pd.DataFrame(predictions), "quality_transfer.csv")
    save_csv(bootstrap, "quality_cluster_bootstrap.csv")
    save_csv(pd.DataFrame(intervals), "quality_parameter_intervals.csv")
    return score, fits, b7_new, b8_new


def source_calibration(d, ndtheta):
    rows, transfer = [], []
    source_offsets = {}
    for src in ("B2", "B4", "B5"):
        x = d[src].copy()
        x["base"] = predict_nd("classic", ndtheta, x.N_params_B, x.D_tokens_B)
        group = "run_id" if src == "B2" else "family"
        sort_by = "D_tokens_B" if src == "B2" else "N_params_B"
        train_ids = []
        for _, part in x.groupby(group, sort=False):
            order = part.sort_values(sort_by)
            train_ids.extend(order.index[:max(1, int(np.floor(.2 * len(order))))].tolist())
        calibration = x.loc[sorted(set(train_ids))]
        holdout = x.drop(index=calibration.index)
        residual = calibration.val_loss - calibration.base
        offset = float(residual.mean())
        source_offsets[src] = offset
        slope, intercept = np.polyfit(calibration.base, calibration.val_loss, 1)
        d_slope, d_intercept = np.polyfit(np.log(calibration.D_tokens_B), residual, 1)
        variants = {
            "raw": holdout.base,
            "source_intercept_20pct": holdout.base + offset,
            "source_affine_20pct": intercept + slope * holdout.base,
            "source_logD_20pct": holdout.base + d_intercept + d_slope * np.log(holdout.D_tokens_B),
        }
        for protocol, pred in variants.items():
            rows.append(dict(source=src, protocol=protocol, calibration_rows=len(calibration),
                             holdout_rows=len(holdout), offset=offset, affine_intercept=intercept,
                             affine_slope=slope, logD_intercept=d_intercept, logD_slope=d_slope,
                             labels_required=(protocol != "raw"), **measures(holdout.val_loss, pred)))
    # B5 is entirely held out when applying the mean offset learned from B2+B4.
    b5 = d["B5"]
    raw = predict_nd("classic", ndtheta, b5.N_params_B, b5.D_tokens_B)
    source_mean = np.mean([source_offsets["B2"], source_offsets["B4"]])
    for protocol, pred in (("zero_shot_classic", raw), ("B2_B4_mean_intercept_to_heldout_B5", raw + source_mean)):
        transfer.append(dict(source="B5", protocol=protocol, B5_labels_used_for_fit=False,
                             training_sources="B2+B4" if protocol.startswith("B2_B4") else "B1",
                             **measures(b5.val_loss, pred)))
    save_csv(pd.DataFrame(rows), "source_conditional_holdout.csv")
    save_csv(pd.DataFrame(transfer), "source_leave_B5_out.csv")
    return pd.DataFrame(rows), pd.DataFrame(transfer)


def mechanism_diagnostic(d, b8_new):
    overlap = d["B6"][KEYS + ["val_loss"]].merge(
        d["B8"][KEYS + ["val_loss"]], on=KEYS, suffixes=("_B6", "_B8"), validate="one_to_one")
    overlap["B8_minus_B6"] = overlap.val_loss_B8 - overlap.val_loss_B6
    save_csv(overlap, "B6_B8_exact_overlap.csv")
    direction = []
    for source, x in (("B6", d["B6"]), ("B7_full", d["B7"]), ("B8_new", b8_new)):
        for (n, dtok), part in x.groupby(["N_params_B", "D_tokens_B"]):
            if part.Q_score.nunique() >= 3:
                rho = float(spearmanr(part.Q_score, part.val_loss).statistic)
                direction.append(dict(source=source, N_params_B=n, D_tokens_B=dtok,
                                      q_levels=part.Q_score.nunique(), rho_Q_L=rho))
    dirs = pd.DataFrame(direction)
    save_csv(dirs, "quality_direction_by_source.csv")
    stats = {key: dict(groups=len(z), fraction_loss_declines_with_Q=float(np.mean(z.rho_Q_L < 0)))
             for key, z in dirs.groupby("source")}
    stats["B6_B8_exact_overlap"] = dict(n=len(overlap), mean_difference=float(overlap.B8_minus_B6.mean()),
                                        q01_mean=float(overlap.loc[np.isclose(overlap.Q_score, .1), "B8_minus_B6"].mean()),
                                        q1_mean=float(overlap.loc[np.isclose(overlap.Q_score, 1), "B8_minus_B6"].mean()))
    (HERE / "mechanism_diagnostic.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def p_scenarios():
    rows = []
    for n in (.07, .7, 7, 70, 120):
        for name, value in (
            ("constant", 1),
            ("power_decay_rho_0.1", (n / .7) ** (-.1)),
            ("saturating_floor_0.5_rho_0.1", .5 + .5 * (n / .7) ** (-.1)),
            ("partial_identification_low", .5),
            ("partial_identification_high", 1.5),
        ):
            rows.append(dict(N_B=n, scenario=name, lambda_p=float(value), identified=False,
                             note="no independent joint N,D,Q,p loss labels"))
    save_csv(pd.DataFrame(rows), "lambda_p_scenarios.csv")


def choose(ndscore, qscore):
    classic = ndscore.set_index("model").loc["classic"]
    nd_checks = {}
    for _, row in ndscore.iterrows():
        if row.model == "classic":
            continue
        conditions = {
            "group_cv_5pct": row.B1_group_cv_rmse <= .95 * classic.B1_group_cv_rmse,
            "paired_group_ci_positive": row.B1_paired_improvement_ci_low > 0,
            "external_no_2pct_harm": all(row[f"{src}_raw_rmse"] <= 1.02 * classic[f"{src}_raw_rmse"] for src in ("B2", "B4", "B5")),
            "bic_no_material_harm": row.B1_bic <= classic.B1_bic + 10,
            "jacobian_condition_reasonable": row.jacobian_condition <= 1e8,
        }
        nd_checks[row.model] = {"conditions": {k: bool(v) for k, v in conditions.items()}, "passed": bool(all(conditions.values()))}
    nd_winner = "classic"
    valid_nd = [key for key, value in nd_checks.items() if value["passed"]]
    if valid_nd:
        nd_winner = min(valid_nd, key=lambda kind: ndscore.set_index("model").loc[kind, "B1_group_cv_rmse"])
    incumbent = qscore.set_index("model").loc["Q_by_N"]
    intervals = pd.read_csv(HERE / "quality_parameter_intervals.csv")
    q_checks = {}
    for _, row in qscore.iterrows():
        if row.model == "Q_by_N":
            continue
        extra = "eta_D" if row.model in ("Q_by_D", "Q_by_ND") else None
        parameter_identified = False
        if extra:
            z = intervals[(intervals.model == row.model) & (intervals.parameter == extra)]
            parameter_identified = bool(len(z) and (z.ci_low.iloc[0] > 0 or z.ci_high.iloc[0] < 0))
        conditions = {
            "leave_N_5pct": row.B6_leave_N_rmse <= .95 * incumbent.B6_leave_N_rmse,
            "B7_5pct": row.B7_new_rmse <= .95 * incumbent.B7_new_rmse,
            "B7_paired_ci_positive": row.B7_vs_champion_ci_low > 0,
            "new_exponent_identified": parameter_identified,
            "bic_no_material_harm": row.B6_bic <= incumbent.B6_bic + 10,
            "jacobian_condition_reasonable": row.jacobian_condition <= 1e8,
        }
        q_checks[row.model] = {"conditions": {k: bool(v) for k, v in conditions.items()}, "passed": bool(all(conditions.values()))}
    q_winner = "Q_by_N"
    valid_q = [key for key, value in q_checks.items() if value["passed"]]
    if valid_q:
        q_winner = min(valid_q, key=lambda kind: qscore.set_index("model").loc[kind, "B7_new_rmse"])
    return nd_winner, q_winner, nd_checks, q_checks


def tournament_table(ndscore, qscore, calib, transfer, nd_winner, q_winner):
    rows = []
    for _, r in ndscore.iterrows():
        for dataset, split, metric, col in (
            ("B1", "training", "rmse", "B1_training_rmse"),
            ("B1", "leave_N_group", "rmse", "B1_group_cv_rmse"),
            ("B1", "training", "bic", "B1_bic"),
            ("B2", "external_raw", "rmse", "B2_raw_rmse"),
            ("B4", "external_raw", "rmse", "B4_raw_rmse"),
            ("B5", "external_raw", "rmse", "B5_raw_rmse"),
        ):
            rows.append(dict(module="N-D", model=r.model, dataset=dataset, split=split,
                             metric=metric, value=float(r[col]), selected=(r.model == nd_winner),
                             n_parameters=int(r.n_parameters), evidence_level="same_source" if dataset == "B1" else "development_external_raw"))
    for _, r in qscore.iterrows():
        for dataset, split, metric, col in (
            ("B6", "training", "rmse", "B6_training_rmse"),
            ("B6", "leave_N_group", "rmse", "B6_leave_N_rmse"),
            ("B6", "training", "bic", "B6_bic"),
            ("B7", "new_points", "rmse", "B7_new_rmse"),
            ("B8", "mechanism_only", "rmse", "B8_new_diagnostic_rmse"),
        ):
            rows.append(dict(module="Q", model=r.model, dataset=dataset, split=split,
                             metric=metric, value=float(r[col]), selected=(r.model == q_winner),
                             n_parameters=int(r.n_parameters), evidence_level="semi_synthetic_diagnostic" if dataset == "B8" else "semi_synthetic"))
    for _, r in calib.iterrows():
        rows.append(dict(module="source_application", model=r.protocol, dataset=r.source,
                         split="20pct_labeled_early_to_80pct_holdout", metric="rmse", value=float(r.rmse),
                         selected=False, n_parameters=0 if r.protocol == "raw" else 1 if "intercept" in r.protocol else 2,
                         evidence_level="labeled_target_source_development"))
    for _, r in transfer.iterrows():
        rows.append(dict(module="source_transfer", model=r.protocol, dataset=r.source,
                         split="B5_fully_held_out", metric="rmse", value=float(r.rmse), selected=False,
                         n_parameters=0 if r.protocol == "zero_shot_classic" else 1,
                         evidence_level="zero_shot_external_source"))
    save_csv(pd.DataFrame(rows), "problem2_model_tournament.csv")


def main():
    d = data()
    manifest = {key: {"path": str(INPUT / name), "sha256": hashlib.sha256((INPUT / name).read_bytes()).hexdigest(),
                      "rows": len(d[key])} for key, name in FILES.items()}
    (HERE / "input_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    ndscore, ndfits = nd_tournament(d)
    qscore, qfits, b7_new, b8_new = q_tournament(d, ndfits["classic"])
    calib, source_transfer = source_calibration(d, ndfits["classic"])
    mechanism = mechanism_diagnostic(d, b8_new)
    p_scenarios()
    nd_winner, q_winner, nd_checks, q_checks = choose(ndscore, qscore)
    tournament_table(ndscore, qscore, calib, source_transfer, nd_winner, q_winner)
    selection = {
        "ND_winner": nd_winner, "Q_winner": q_winner,
        "ND_parameters": ndfits[nd_winner].tolist(), "Q_parameters": qfits[q_winner].tolist(),
        "N_unit": "billion_parameters", "D_unit": "billion_tokens", "Q_semantics": "B6/B7 synthetic mechanism only",
        "p_bridge": "lambda_p=1 scenario convention, no independently identified scale law",
        "source_calibration": "optional application layer requiring target-source labeled loss",
        "ND_challenger_checks": nd_checks, "Q_challenger_checks": q_checks,
        "B7_new_rows": len(b7_new), "B8_new_rows": len(b8_new),
        "B8_mechanism": mechanism,
        "selection_caveat": "B2/B4/B5/B7 seen in prior studies; repeated independent calculation, not fresh blind validation",
    }
    (HERE / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: selection[k] for k in ("ND_winner", "Q_winner", "ND_parameters", "Q_parameters", "B7_new_rows", "B8_new_rows")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
