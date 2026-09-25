"""Paired clustered uncertainty and stricter B8 scale holdout for Q2 experiments."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from run_experiments import DATA, HERE, SEED, q_fit, read_data, rmse, unseen, write


def paired_effects(path, source, current, new, group_cols, n_boot=2000):
    z = pd.read_csv(HERE / path)
    z = z[z.source == source].copy()
    left = z[z.model == current].set_index("row")
    right = z[z.model == new].set_index("row")
    common = left.index.intersection(right.index)
    left = left.loc[common]
    right = right.loc[common]
    if not np.allclose(left.observed, right.observed):
        raise AssertionError("Not paired")
    groups = left[group_cols].astype(str).agg("|".join, axis=1).to_numpy()
    unique = np.unique(groups)
    positions = {g: np.flatnonzero(groups == g) for g in unique}
    y = left.observed.to_numpy(float)
    pc = left.predicted.to_numpy(float)
    pn = right.predicted.to_numpy(float)
    rng = np.random.default_rng(SEED)
    deltas = []
    for _ in range(n_boot):
        chosen = rng.choice(unique, len(unique), replace=True)
        idx = np.concatenate([positions[g] for g in chosen])
        deltas.append(rmse(y[idx], pn[idx]) - rmse(y[idx], pc[idx]))
    interval = np.quantile(deltas, [.025, .5, .975])
    return dict(dataset=source, current=current, candidate=new, n_rows=len(y), n_groups=len(unique), current_rmse=rmse(y, pc), candidate_rmse=rmse(y, pn), delta_new_minus_current=rmse(y, pn)-rmse(y, pc), relative_change_pct=100*(rmse(y, pn)/rmse(y, pc)-1), cluster_bootstrap_delta_lo95=interval[0], cluster_bootstrap_delta_median=interval[1], cluster_bootstrap_delta_hi95=interval[2], bootstrap_p_improvement=float(np.mean(np.asarray(deltas) < 0)))


def q_parameter_bootstrap(data, base, n_boot=300):
    b6 = data["B6"]
    rng = np.random.default_rng(SEED)
    groups = np.sort(b6.N_params_B.unique())
    current = np.array([.3568, .9915, .1621, .04])
    rows = []
    for i in range(n_boot):
        sampled = rng.choice(groups, len(groups), replace=True)
        frame = pd.concat([b6[b6.N_params_B == g] for g in sampled], ignore_index=True)
        coef = q_fit("quality_joint_nd", frame, base, current)
        rows.append(dict(draw=i, G=coef[0], kappa=coef[1], eta_N=coef[2], eta_D=coef[3]))
    write(rows, "q_joint_cluster_bootstrap.csv")
    return pd.DataFrame(rows)


def b8_leave_N_calibration(data, base, current):
    """Train B8 residual correction using two D cells from each other N; hold one N out."""
    b8 = unseen(data["B7"], data["B8"]).reset_index(drop=True).copy()
    b8["base_prediction"] = base[0] + base[1]*b8.N_params_B.to_numpy(float)**(-base[2]) + base[3]*b8.D_tokens_B.to_numpy(float)**(-base[4]) + current[0]*(1-b8.Q_score.to_numpy(float))**current[1]*b8.N_params_B.to_numpy(float)**(-current[2])
    def feature(frame):
        logn = np.log(frame.N_params_B.to_numpy(float))
        logd = np.log(frame.D_tokens_B.to_numpy(float))
        q = frame.Q_score.to_numpy(float) - .5
        return np.column_stack([q, logn, logd, q*logn, q*logd])
    rows = []
    for held in np.sort(b8.N_params_B.unique()):
        train = b8[b8.N_params_B != held].copy()
        selected = []
        for _, group in train[["N_params_B", "D_tokens_B"]].drop_duplicates().sort_values(["N_params_B", "D_tokens_B"]).groupby("N_params_B"):
            selected.extend(zip(group.iloc[[1, 6]].N_params_B, group.iloc[[1, 6]].D_tokens_B))
        selected = set(selected)
        train = train[[tuple(v) in selected for v in train[["N_params_B", "D_tokens_B"]].to_numpy()]].copy()
        test = b8[b8.N_params_B == held].copy()
        scale = StandardScaler().fit(feature(train))
        target = train.val_loss.to_numpy(float) - train.base_prediction.to_numpy(float)
        fit = Ridge(alpha=.1).fit(scale.transform(feature(train)), target)
        preds = {
            "quality_model_current_raw": test.base_prediction.to_numpy(float),
            "source_intercept_control": test.base_prediction.to_numpy(float) + target.mean(),
            "source_quality_interaction_new": test.base_prediction.to_numpy(float) + fit.predict(scale.transform(feature(test))),
        }
        for model, pred in preds.items():
            rows.extend(dict(model=model, source="B8_new", held_N_B=float(held), row=int(i), observed=float(y), predicted=float(p), calibration_rows=len(train), calibration_N_groups=train.N_params_B.nunique()) for i, y, p in zip(test.index, test.val_loss, pred))
    write(rows, "b8_leave_N_predictions.csv")
    return pd.DataFrame(rows)


def main():
    from run_experiments import fit_nd
    data = read_data()
    base = fit_nd("classic", data["B1"]).x
    current = np.asarray(json.loads(pd.read_csv(HERE / "quality_fit.csv").query("model == 'quality_model_current'").parameters.iloc[0]))
    comparisons = []
    comparisons.append(paired_effects("nd_predictions.csv", "B1", "classic_current", "generalized_ND_new", ["N_params_B"]))
    for source in ["B2", "B4", "B5"]:
        comparisons.append(paired_effects("nd_predictions.csv", source, "classic_current", "generalized_ND_new", ["N_params_B"]))
    for source in ["B6", "B7_new", "B8_new"]:
        group = ["N_params_B"] if source == "B6" else ["N_params_B", "D_tokens_B"]
        for candidate in ["quality_joint_nd_new", "quality_compute_new"]:
            comparisons.append(paired_effects("quality_predictions.csv", source, "quality_model_current", candidate, group))
    for source in ["B8_new", "B8_overlap_B6"]:
        for baseline in ["quality_model_current_raw", "source_intercept_control"]:
            comparisons.append(paired_effects("b8_conditional_predictions.csv", source, baseline, "source_quality_interaction_new", ["ND_group"]))
    boot = q_parameter_bootstrap(data, base)
    write([dict(parameter="eta_D", median=float(boot.eta_D.median()), lo95=float(boot.eta_D.quantile(.025)), hi95=float(boot.eta_D.quantile(.975)), share_positive=float((boot.eta_D > 0).mean()), clusters="B6 N groups")], "q_joint_parameter_summary.csv")
    b8 = b8_leave_N_calibration(data, base, current)
    for candidate in ["source_intercept_control", "source_quality_interaction_new"]:
        z = b8[b8.model == candidate]
        write([dict(model=candidate, held_N_groups=z.held_N_B.nunique(), n=len(z), rmse=rmse(z.observed, z.predicted), mae=float(np.mean(abs(z.observed-z.predicted))))], "b8_leave_N_"+candidate+"_score.csv")
    comparisons.append(paired_effects("b8_leave_N_predictions.csv", "B8_new", "source_intercept_control", "source_quality_interaction_new", ["held_N_B"]))
    write(comparisons, "paired_effects.csv")


if __name__ == "__main__":
    main()
