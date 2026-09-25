"""Independent second-round experiments for question 2.

No main-project artifact is overwritten. Run from any directory with Python 3.11:
    python optimization_20260924/round2/q2/run_experiments.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / "real_attachments" / "B_scaling_laws"
sys.path.insert(0, str(ROOT / "problem2_v2"))
from models import fit_nd, fit_q, predict_nd, predict_q  # noqa: E402

SEED = 20260924


def read_data():
    files = {
        "B1": "pythia_training_log_existing.csv",
        "B2": "cerebras_training_log.csv",
        "B4": "scaling_baseline.csv",
        "B5": "published_scaling_data.csv",
        "B6": "supplementary_NQ_experiment.csv",
        "B7": "supplementary_NQ_experiment_expanded.csv",
        "B8": "supplementary_NQ_experiment_large.csv",
    }
    return {name: pd.read_csv(DATA / file) for name, file in files.items()}


def unseen(old, new):
    keys = ["N_params_B", "D_tokens_B", "Q_score"]
    return new.merge(old[keys].drop_duplicates().assign(_seen=1), on=keys, how="left").query("_seen != 1").drop(columns="_seen")


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def nd_generalized(z, n, d):
    """Generalized mean of the classic N and D terms; r=1 is exactly classic."""
    e, a, alpha, b, beta, r = z
    n, d = np.broadcast_arrays(np.asarray(n, float), np.asarray(d, float))
    u = a * n ** (-alpha)
    v = b * d ** (-beta)
    return e + (u ** r + v ** r) ** (1 / r)


def fit_nd_generalized(frame, start):
    lo = np.array([0, 1e-6, .005, 1e-6, .005, .25])
    hi = np.array([4, 100, 2, 100, 2, 3.0])
    n = frame.N_params_B.to_numpy(float)
    d = frame.D_tokens_B.to_numpy(float)
    y = frame.val_loss.to_numpy(float)
    fit = least_squares(lambda z: nd_generalized(z, n, d) - y,
                        np.clip(start, lo + 1e-8, hi - 1e-8),
                        bounds=(lo, hi), max_nfev=2000, xtol=1e-11, ftol=1e-11)
    return fit.x


def q_predict(kind, z, n, d, q, base):
    n, d, q = np.broadcast_arrays(np.asarray(n, float), np.asarray(d, float), np.asarray(q, float))
    nd = predict_nd("classic", base, n, d)
    g, kappa, eta = z[:3]
    h = g * np.maximum(1 - q, 1e-12) ** kappa
    if kind == "quality_joint_nd":
        return nd + h * n ** (-eta) * (d / 100) ** (-z[3])
    if kind == "quality_compute":
        return nd + h * (n * d / 100) ** (-eta)
    raise ValueError(kind)


def q_fit(kind, frame, base, start):
    n = frame.N_params_B.to_numpy(float)
    d = frame.D_tokens_B.to_numpy(float)
    q = frame.Q_score.to_numpy(float)
    y = frame.val_loss.to_numpy(float)
    lo = np.array([0, .1, -1] + ([-1] if kind == "quality_joint_nd" else []), float)
    hi = np.array([5, 4, 1] + ([1] if kind == "quality_joint_nd" else []), float)
    fit = least_squares(lambda z: q_predict(kind, z, n, d, q, base) - y,
                        np.clip(start, lo + 1e-8, hi - 1e-8), bounds=(lo, hi), max_nfev=2000)
    return fit.x


def write(frame, name):
    pd.DataFrame(frame).to_csv(HERE / name, index=False, float_format="%.12g")


def grouped_rmse_interval(frame, pred_a, pred_b, group, n_boot=1000):
    """Cluster bootstrap of RMSE(new) - RMSE(current), paired by held-out group."""
    rng = np.random.default_rng(SEED)
    groups = frame[group].drop_duplicates().to_numpy()
    by_group = {v: np.flatnonzero(frame[group].to_numpy() == v) for v in groups}
    y = frame.val_loss.to_numpy(float)
    a = np.asarray(pred_a, float)
    b = np.asarray(pred_b, float)
    effects = []
    for _ in range(n_boot):
        chosen = rng.choice(groups, len(groups), replace=True)
        index = np.concatenate([by_group[v] for v in chosen])
        effects.append(rmse(y[index], a[index]) - rmse(y[index], b[index]))
    return np.quantile(effects, [.025, .5, .975]).tolist()


def nd_experiment(data):
    b1 = data["B1"].reset_index(drop=True)
    current = fit_nd("classic", b1).x
    novel = fit_nd_generalized(b1, np.r_[current, 1.0])
    fits = {"classic_current": current, "generalized_ND_new": novel}
    fit_metrics = []
    for name, coef in fits.items():
        pred = predict_nd("classic", coef, b1.N_params_B, b1.D_tokens_B) if name == "classic_current" else nd_generalized(coef, b1.N_params_B, b1.D_tokens_B)
        rss = np.sum((pred - b1.val_loss.to_numpy(float)) ** 2)
        fit_metrics.append(dict(model=name, dataset="B1_fit", n=len(b1), rmse=rmse(b1.val_loss, pred), mae=mae(b1.val_loss, pred), bic=len(b1)*np.log(rss/len(b1))+len(coef)*np.log(len(b1)), parameters=json.dumps(coef.tolist())))
    predictions = []
    for held in sorted(b1.N_params_B.unique()):
        tr = b1[b1.N_params_B != held]
        te = b1[b1.N_params_B == held]
        c = fit_nd("classic", tr, start=current).x
        z = fit_nd_generalized(tr, np.r_[c, 1.0])
        for model, pred in [
            ("classic_current", predict_nd("classic", c, te.N_params_B, te.D_tokens_B)),
            ("generalized_ND_new", nd_generalized(z, te.N_params_B, te.D_tokens_B)),
        ]:
            predictions.extend(dict(model=model, source="B1", split="leave_N_out", row=int(i), N_params_B=float(held), observed=float(y), predicted=float(p)) for i, y, p in zip(te.index, te.val_loss, pred))
    for source in ["B2", "B4", "B5"]:
        x = data[source].reset_index(drop=True)
        for model, pred in [
            ("classic_current", predict_nd("classic", current, x.N_params_B, x.D_tokens_B)),
            ("generalized_ND_new", nd_generalized(novel, x.N_params_B, x.D_tokens_B)),
        ]:
            predictions.extend(dict(model=model, source=source, split="raw_no_target_labels", row=int(i), N_params_B=float(n), observed=float(y), predicted=float(p)) for i, n, y, p in zip(x.index, x.N_params_B, x.val_loss, pred))
    write(fit_metrics, "nd_fit.csv")
    write(predictions, "nd_predictions.csv")
    return current


def q_experiment(data, base):
    b6 = data["B6"].reset_index(drop=True)
    b7new = unseen(data["B6"], data["B7"]).reset_index(drop=True)
    b8new = unseen(data["B7"], data["B8"]).reset_index(drop=True)
    current = fit_q("quality_model", b6, base).x
    novel = {
        "quality_joint_nd_new": q_fit("quality_joint_nd", b6, base, np.r_[current, 0.0]),
        "quality_compute_new": q_fit("quality_compute", b6, base, np.array([current[0], current[1], current[2] / 2])),
    }
    params = {"quality_model_current": current, **novel}
    fit_metrics = []
    for model, coef in params.items():
        pred = predict_q("quality_model", coef, b6.N_params_B, b6.D_tokens_B, b6.Q_score, base) if model == "quality_model_current" else q_predict(model.replace("_new", ""), coef, b6.N_params_B, b6.D_tokens_B, b6.Q_score, base)
        rss = np.sum((pred - b6.val_loss.to_numpy(float)) ** 2)
        fit_metrics.append(dict(model=model, dataset="B6_fit", n=len(b6), rmse=rmse(b6.val_loss, pred), mae=mae(b6.val_loss, pred), bic=len(b6)*np.log(rss/len(b6))+len(coef)*np.log(len(b6)), parameters=json.dumps(coef.tolist())))
    predictions = []
    for held in sorted(b6.N_params_B.unique()):
        tr = b6[b6.N_params_B != held]
        te = b6[b6.N_params_B == held]
        c = fit_q("quality_model", tr, base).x
        fold_coefs = {"quality_model_current": c,
                      "quality_joint_nd_new": q_fit("quality_joint_nd", tr, base, np.r_[c, 0.0]),
                      "quality_compute_new": q_fit("quality_compute", tr, base, np.array([c[0], c[1], c[2] / 2]))}
        for model, coef in fold_coefs.items():
            pred = predict_q("quality_model", coef, te.N_params_B, te.D_tokens_B, te.Q_score, base) if model == "quality_model_current" else q_predict(model.replace("_new", ""), coef, te.N_params_B, te.D_tokens_B, te.Q_score, base)
            predictions.extend(dict(model=model, source="B6", split="leave_N_out", row=int(i), N_params_B=float(held), D_tokens_B=float(d), Q_score=float(q), observed=float(y), predicted=float(p)) for i, d, q, y, p in zip(te.index, te.D_tokens_B, te.Q_score, te.val_loss, pred))
    for source, frame in [("B7_new", b7new), ("B8_new", b8new)]:
        for model, coef in params.items():
            pred = predict_q("quality_model", coef, frame.N_params_B, frame.D_tokens_B, frame.Q_score, base) if model == "quality_model_current" else q_predict(model.replace("_new", ""), coef, frame.N_params_B, frame.D_tokens_B, frame.Q_score, base)
            predictions.extend(dict(model=model, source=source, split="never_fit_on_target", row=int(i), N_params_B=float(n), D_tokens_B=float(d), Q_score=float(q), observed=float(y), predicted=float(p)) for i, n, d, q, y, p in zip(frame.index, frame.N_params_B, frame.D_tokens_B, frame.Q_score, frame.val_loss, pred))
    write(fit_metrics, "quality_fit.csv")
    write(predictions, "quality_predictions.csv")
    return current, b8new


def b8_source_calibration(data, b8new, base, q_current):
    """Source-gated correction; B8 labels are required, so this is conditional transfer."""
    x = b8new.reset_index(drop=True).copy()
    x["ND_group"] = x.N_params_B.astype(str) + "|" + x.D_tokens_B.astype(str)
    x["base_prediction"] = predict_q("quality_model", q_current, x.N_params_B, x.D_tokens_B, x.Q_score, base)
    # Two of ten D groups per N are calibration; held-out (N,D) cells stay untouched.
    train_groups = []
    for _, g in x[["N_params_B", "D_tokens_B", "ND_group"]].drop_duplicates().sort_values(["N_params_B", "D_tokens_B"]).groupby("N_params_B"):
        train_groups.extend(g.iloc[[1, 6]]["ND_group"].tolist())
    x["partition"] = np.where(x.ND_group.isin(train_groups), "calibration", "heldout_ND")
    train = x[x.partition == "calibration"].copy()
    test = x[x.partition == "heldout_ND"].copy()
    def feature(frame):
        logn = np.log(frame.N_params_B.to_numpy(float))
        logd = np.log(frame.D_tokens_B.to_numpy(float))
        q = frame.Q_score.to_numpy(float) - .5
        return np.column_stack([q, logn, logd, q*logn, q*logd])
    scale = StandardScaler().fit(feature(train))
    X = scale.transform(feature(train))
    Xt = scale.transform(feature(test))
    target = train.val_loss.to_numpy(float) - train.base_prediction.to_numpy(float)
    groups = train.ND_group.to_numpy()
    cv = GroupKFold(n_splits=5)
    choices = []
    for alpha in [.001, .01, .1, 1, 10, 100]:
        pred = np.full(len(train), np.nan)
        for ti, vi in cv.split(X, target, groups):
            fold_scale = StandardScaler().fit(feature(train.iloc[ti]))
            fit = Ridge(alpha=alpha).fit(fold_scale.transform(feature(train.iloc[ti])), target[ti])
            pred[vi] = fit.predict(fold_scale.transform(feature(train.iloc[vi])))
        choices.append(dict(alpha=alpha, calibration_group_cv_rmse=rmse(target, pred)))
    selected = min(choices, key=lambda row: row["calibration_group_cv_rmse"])["alpha"]
    fit = Ridge(alpha=selected).fit(X, target)
    raw = test.base_prediction.to_numpy(float)
    offset = raw + target.mean()
    corrected = raw + fit.predict(Xt)
    rows = []
    for model, pred in [("quality_model_current_raw", raw), ("source_intercept_control", offset), ("source_quality_interaction_new", corrected)]:
        rows.extend(dict(model=model, source="B8_new", split="heldout_ND_after_2_cells_per_N_calibration", row=int(i), ND_group=group, observed=float(y), predicted=float(p)) for i, group, y, p in zip(test.index, test.ND_group, test.val_loss, pred))
    # Exact B6/B8 overlapping inputs were never among the B8-new calibration rows.
    keys = ["N_params_B", "D_tokens_B", "Q_score"]
    overlap = data["B8"].merge(data["B6"][keys].drop_duplicates(), on=keys, how="inner").reset_index(drop=True)
    overlap["ND_group"] = overlap.N_params_B.astype(str) + "|" + overlap.D_tokens_B.astype(str)
    overlap_base = predict_q("quality_model", q_current, overlap.N_params_B, overlap.D_tokens_B, overlap.Q_score, base)
    overlap_preds = [
        ("quality_model_current_raw", overlap_base),
        ("source_intercept_control", overlap_base + target.mean()),
        ("source_quality_interaction_new", overlap_base + fit.predict(scale.transform(feature(overlap)))),
    ]
    for model, pred in overlap_preds:
        rows.extend(dict(model=model, source="B8_overlap_B6", split="exact_inputs_heldout_from_B8_calibration", row=int(i), ND_group=group, observed=float(y), predicted=float(p)) for i, group, y, p in zip(overlap.index, overlap.ND_group, overlap.val_loss, pred))
    write(rows, "b8_conditional_predictions.csv")
    write(choices, "b8_calibration_inner_cv.csv")
    write([dict(calibration_rows=len(train), calibration_ND_groups=train.ND_group.nunique(), heldout_rows=len(test), heldout_ND_groups=test.ND_group.nunique(), ridge_alpha=selected, coefficient_intercept=float(fit.intercept_), standardized_coefficients=json.dumps(fit.coef_.tolist()), feature_names="Qcenter,logN,logD,Qcenter*logN,Qcenter*logD")], "b8_calibration_metadata.csv")


def summarize_predictions(file, key):
    frame = pd.read_csv(HERE / file)
    rows = []
    for (model, source, split), g in frame.groupby(["model", "source", "split"]):
        rows.append(dict(experiment=key, model=model, source=source, split=split, n=len(g), rmse=rmse(g.observed, g.predicted), mae=mae(g.observed, g.predicted), bias=float(np.mean(g.predicted - g.observed))))
    return rows


def conflict_audit(data):
    keys = ["N_params_B", "D_tokens_B", "Q_score"]
    overlap = data["B6"].merge(data["B8"], on=keys, suffixes=("_B6", "_B8"))
    gap = overlap.val_loss_B8 - overlap.val_loss_B6
    write([dict(exact_input_pairs=len(overlap), nonzero_loss_conflicts=int((np.abs(gap) > 1e-12).sum()), mean_B8_minus_B6=float(gap.mean()), min_B8_minus_B6=float(gap.min()), max_B8_minus_B6=float(gap.max()), rmse_irreducible_equal_source_weight=float(np.sqrt(np.mean((gap/2)**2))), source="B6_vs_B8")], "repeat_input_conflicts.csv")


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    data = read_data()
    base = nd_experiment(data)
    q_current, b8new = q_experiment(data, base)
    b8_source_calibration(data, b8new, base, q_current)
    conflict_audit(data)
    scores = []
    for file, key in [("nd_predictions.csv", "N-D law"), ("quality_predictions.csv", "quality law"), ("b8_conditional_predictions.csv", "source-conditioned transfer")]:
        scores += summarize_predictions(file, key)
    write(scores, "scorecard.csv")
    write([dict(seed=SEED, model_selection="Candidate classes fixed before evaluation; only B8 correction ridge alpha chosen by calibration-only GroupKFold", baseline="P2 V2 classic N-D and quality_model refitted on B1 and B6", B7="exact-key new rows relative to B6", B8="exact-key new rows relative to B7; source correction uses two (N,D) calibration cells per N and tests disjoint cells; B6/B8 exact overlaps held out from B8 calibration")], "protocol.csv")


if __name__ == "__main__":
    main()
