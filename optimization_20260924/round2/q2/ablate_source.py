"""Leakage-aware ablation of the labeled B8 quality/source correction."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from run_experiments import HERE, fit_nd, predict_q, q_predict, read_data, rmse, unseen, write

ALPHAS = [.001, .01, .1, 1, 10, 100]
FEATURES = {
    "source_Q_only": [0],
    "source_Q_N_D_main": [0, 1, 2],
    "source_Q_interactions": [0, 1, 2, 3, 4],
}


def raw_features(frame):
    logn = np.log(frame.N_params_B.to_numpy(float))
    logd = np.log(frame.D_tokens_B.to_numpy(float))
    q = frame.Q_score.to_numpy(float) - .5
    return np.column_stack([q, logn, logd, q * logn, q * logd])


def fit_calibrated(train, test, columns):
    x = raw_features(train)[:, columns]
    xt = raw_features(test)[:, columns]
    y = train.val_loss.to_numpy(float) - train.base_prediction.to_numpy(float)
    groups = train.ND_group.to_numpy()
    n_splits = min(5, len(np.unique(groups)))
    rows = []
    for alpha in ALPHAS:
        oof = np.full(len(y), np.nan)
        for tr, va in GroupKFold(n_splits=n_splits).split(x, y, groups):
            scale = StandardScaler().fit(x[tr])
            model = Ridge(alpha=alpha).fit(scale.transform(x[tr]), y[tr])
            oof[va] = model.predict(scale.transform(x[va]))
        rows.append((alpha, rmse(y, oof)))
    alpha, cv_score = min(rows, key=lambda z: z[1])
    scale = StandardScaler().fit(x)
    model = Ridge(alpha=alpha).fit(scale.transform(x), y)
    pred = test.base_prediction.to_numpy(float) + model.predict(scale.transform(xt))
    return pred, alpha, cv_score


def selected_cells(frame, candidate_N=None):
    keys = []
    source = frame if candidate_N is None else frame[frame.N_params_B.isin(candidate_N)]
    for _, g in source[["N_params_B", "D_tokens_B"]].drop_duplicates().sort_values(["N_params_B", "D_tokens_B"]).groupby("N_params_B"):
        keys += list(zip(g.iloc[[1, 6]].N_params_B, g.iloc[[1, 6]].D_tokens_B))
    return set(keys)


def main():
    data = read_data()
    base = fit_nd("classic", data["B1"]).x
    quality = np.asarray(json.loads(pd.read_csv(HERE / "quality_fit.csv").query("model == 'quality_model_current'").parameters.iloc[0]))
    b8 = unseen(data["B7"], data["B8"]).reset_index(drop=True).copy()
    b8["ND_group"] = b8.N_params_B.astype(str) + "|" + b8.D_tokens_B.astype(str)
    b8["base_prediction"] = predict_q("quality_model", quality, b8.N_params_B, b8.D_tokens_B, b8.Q_score, base)
    calibr_N = np.sort(b8.loc[b8.data_type == "calibrated", "N_params_B"].unique())
    all_cells = selected_cells(b8)
    calibrated_cells = selected_cells(b8, calibr_N)
    one_cell_per_N = set()
    for _, group in b8[["N_params_B", "D_tokens_B"]].drop_duplicates().sort_values(["N_params_B", "D_tokens_B"]).groupby("N_params_B"):
        one_cell_per_N.add(tuple(group.iloc[1][["N_params_B", "D_tokens_B"]]))
    is_all_cell = np.array([tuple(v) in all_cells for v in b8[["N_params_B", "D_tokens_B"]].to_numpy()])
    is_cal_cell = np.array([tuple(v) in calibrated_cells for v in b8[["N_params_B", "D_tokens_B"]].to_numpy()])
    is_one_cell = np.array([tuple(v) in one_cell_per_N for v in b8[["N_params_B", "D_tokens_B"]].to_numpy()])
    q_training = {0.05, .3, .5, .7, .95}
    q_test = {0.1, .2, .4, .6, .8, .9, 1.0}
    experiments = [
        ("ND_cells_disjoint", b8[is_all_cell], b8[~is_all_cell]),
        ("one_ND_cell_per_N", b8[is_one_cell], b8[~is_one_cell]),
        ("one_ND_cell_per_N_Q_disjoint", b8[is_one_cell & b8.Q_score.isin(q_training)], b8[(~is_one_cell) & b8.Q_score.isin(q_test)]),
        ("ND_and_Q_values_disjoint", b8[is_all_cell & b8.Q_score.isin(q_training)], b8[(~is_all_cell) & b8.Q_score.isin(q_test)]),
        ("calibrated_to_extrapolated_N", b8[is_cal_cell & (b8.data_type == "calibrated")], b8[b8.data_type == "extrapolated"]),
    ]
    pd.DataFrame({"experiment_id": experiments[0][1].experiment_id}).to_csv(HERE / "b8_calibration_ids.csv", index=False)
    pd.DataFrame({"experiment_id": experiments[0][2].experiment_id}).to_csv(HERE / "b8_evaluation_ids.csv", index=False)
    for protocol, train, test in experiments[1:3]:
        pd.DataFrame({"experiment_id": train.experiment_id}).to_csv(HERE / f"b8_{protocol}_calibration_ids.csv", index=False)
        pd.DataFrame({"experiment_id": test.experiment_id}).to_csv(HERE / f"b8_{protocol}_evaluation_ids.csv", index=False)
    metrics = []
    prediction_rows = []
    for protocol, train, test in experiments:
        train = train.copy()
        test = test.copy()
        assert set(train.ND_group).isdisjoint(set(test.ND_group))
        if protocol == "ND_and_Q_values_disjoint":
            assert set(train.Q_score).isdisjoint(set(test.Q_score))
        raw = test.base_prediction.to_numpy(float)
        intercept = raw + float((train.val_loss - train.base_prediction).mean())
        prediction_map = {"zero_shot_Model_C": (raw, np.nan, np.nan), "source_intercept": (intercept, np.nan, np.nan)}
        for name, columns in FEATURES.items():
            prediction_map[name] = fit_calibrated(train, test, columns)
        for name, (pred, alpha, cv_score) in prediction_map.items():
            metrics.append(dict(protocol=protocol, model=name, calibration_rows=len(train), calibration_ND_cells=train.ND_group.nunique(), calibration_N_groups=train.N_params_B.nunique(), calibration_Q_levels=train.Q_score.nunique(), heldout_rows=len(test), heldout_ND_cells=test.ND_group.nunique(), heldout_N_groups=test.N_params_B.nunique(), heldout_Q_levels=test.Q_score.nunique(), rmse=rmse(test.val_loss, pred), mae=float(np.mean(np.abs(test.val_loss.to_numpy(float)-pred))), alpha=alpha, calibration_group_cv_rmse=cv_score))
            prediction_rows.extend(dict(protocol=protocol, model=name, row=int(i), ND_group=nd, N_params_B=float(n), Q_score=float(q), data_type=typ, observed=float(y), predicted=float(p)) for i, nd, n, q, typ, y, p in zip(test.index, test.ND_group, test.N_params_B, test.Q_score, test.data_type, test.val_loss, pred))
    write(metrics, "b8_source_ablation.csv")
    write(prediction_rows, "b8_source_ablation_predictions.csv")
    joint = np.asarray(json.loads(pd.read_csv(HERE / "quality_fit.csv").query("model == 'quality_joint_nd_new'").parameters.iloc[0]))
    base_law_rows = []
    for protocol, train_original, test_original in experiments:
        for law in ["quality_model_current", "quality_joint_nd_new"]:
            train = train_original.copy()
            test = test_original.copy()
            for frame in [train, test]:
                frame["base_prediction"] = (predict_q("quality_model", quality, frame.N_params_B, frame.D_tokens_B, frame.Q_score, base)
                                            if law == "quality_model_current" else q_predict("quality_joint_nd", joint, frame.N_params_B, frame.D_tokens_B, frame.Q_score, base))
            pred, alpha, cv_score = fit_calibrated(train, test, FEATURES["source_Q_interactions"])
            base_law_rows.append(dict(protocol=protocol, base_quality_law=law, source_correction="Q_logN_logD_with_Q_interactions", calibration_rows=len(train), heldout_rows=len(test), rmse=rmse(test.val_loss, pred), mae=float(np.mean(abs(test.val_loss.to_numpy(float)-pred))), alpha=alpha, calibration_group_cv_rmse=cv_score))
    write(base_law_rows, "b8_source_base_law_comparison.csv")


if __name__ == "__main__":
    main()
