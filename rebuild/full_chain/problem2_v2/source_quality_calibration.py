"""Optional, explicitly labeled B8 source adaptation for the P2 N-D-Q model.

The canonical recommended_model.joblib is read only. This module never changes
the zero-shot Model C used by questions 3 and 4. It requires caller-provided
B8 calibration and evaluation experiment IDs, and refuses row or N-D cell
overlap between them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from models import predict_q

ALPHAS = (.001, .01, .1, 1.0, 10.0, 100.0)
FEATURE_NAMES = {
    "full_interaction": ("Q_centered", "log_N", "log_D", "Q_centered_x_log_N", "Q_centered_x_log_D"),
    "q_only": ("Q_centered",),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def features(frame: pd.DataFrame, mode: str = "full_interaction") -> np.ndarray:
    if mode not in FEATURE_NAMES:
        raise ValueError(f"Unsupported calibration mode: {mode}")
    n = frame.N_params_B.to_numpy(float)
    d = frame.D_tokens_B.to_numpy(float)
    q = frame.Q_score.to_numpy(float) - .5
    if np.any(n <= 0) or np.any(d <= 0) or np.any(~np.isfinite(q)):
        raise ValueError("N and D must be positive; N, D and Q must be finite")
    logn, logd = np.log(n), np.log(d)
    if mode == "q_only":
        return q.reshape(-1, 1)
    return np.column_stack((q, logn, logd, q * logn, q * logd))


def groups(frame: pd.DataFrame) -> np.ndarray:
    return frame.N_params_B.astype(str).to_numpy() + "|" + frame.D_tokens_B.astype(str).to_numpy()


def read_ids(path: Path) -> list[str]:
    frame = pd.read_csv(path, dtype={"experiment_id": str})
    if "experiment_id" not in frame or frame.experiment_id.isna().any() or frame.experiment_id.duplicated().any():
        raise ValueError(f"ID file needs unique, nonempty experiment_id values: {path}")
    return frame.experiment_id.tolist()


def select_disjoint(b8: pd.DataFrame, calibration_ids: list[str], evaluation_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "experiment_id" not in b8 or b8.experiment_id.duplicated().any():
        raise ValueError("B8 file must contain unique experiment_id values")
    if len(calibration_ids) != len(set(calibration_ids)) or len(evaluation_ids) != len(set(evaluation_ids)):
        raise ValueError("Duplicate IDs in calibration or evaluation list")
    if set(calibration_ids) & set(evaluation_ids):
        raise ValueError("Calibration and evaluation rows overlap")
    indexed = b8.set_index("experiment_id", drop=False)
    missing = (set(calibration_ids) | set(evaluation_ids)) - set(indexed.index)
    if missing:
        raise ValueError(f"Experiment IDs missing from B8 file: {sorted(missing)[:3]}")
    calibration = indexed.loc[calibration_ids].reset_index(drop=True)
    evaluation = indexed.loc[evaluation_ids].reset_index(drop=True)
    if not calibration_ids or not evaluation_ids:
        raise ValueError("Calibration and evaluation sets must both be nonempty")
    if set(groups(calibration)) & set(groups(evaluation)):
        raise ValueError("Calibration and evaluation N-D cells overlap")
    return calibration, evaluation


def base_prediction(frame: pd.DataFrame, base_artifact: dict) -> np.ndarray:
    if base_artifact["ND_model"] != "classic" or base_artifact["Q_model"] != "quality_model":
        raise ValueError("This adapter requires the existing P2 V2 classic + quality_model artifact")
    return predict_q("quality_model", base_artifact["Q_parameters"], frame.N_params_B,
                     frame.D_tokens_B, frame.Q_score, base_artifact["ND_parameters"])


def fit_b8_calibrator(calibration: pd.DataFrame, base_artifact: dict, base_sha256: str,
                      mode: str = "full_interaction") -> dict:
    """Fit only on labeled B8 calibration rows; tune alpha using grouped OOF."""
    if "val_loss" not in calibration or calibration.val_loss.isna().any():
        raise ValueError("Calibration rows require observed B8 val_loss")
    x = features(calibration, mode)
    target = calibration.val_loss.to_numpy(float) - base_prediction(calibration, base_artifact)
    cell = groups(calibration)
    n_cell = len(set(cell))
    if n_cell < 5:
        raise ValueError("At least five distinct B8 N-D calibration cells are required")
    if mode == "full_interaction" and (calibration.groupby("N_params_B").D_tokens_B.nunique() < 2).any():
        raise ValueError("Full B8 source interaction requires at least two D cells for each calibration N; explicitly select q_only mode for one-cell designs")
    cv_rows = []
    for alpha in ALPHAS:
        oof = np.full(len(calibration), np.nan)
        for tr, va in GroupKFold(n_splits=5).split(x, target, cell):
            scaler = StandardScaler().fit(x[tr])
            ridge = Ridge(alpha=alpha).fit(scaler.transform(x[tr]), target[tr])
            oof[va] = ridge.predict(scaler.transform(x[va]))
        cv_rows.append((alpha, float(np.sqrt(np.mean((target - oof) ** 2)))))
    alpha, cv_rmse = min(cv_rows, key=lambda row: row[1])
    scaler = StandardScaler().fit(x)
    ridge = Ridge(alpha=alpha).fit(scaler.transform(x), target)
    return dict(source="B8_labeled", mode=mode, base_artifact_sha256=base_sha256, feature_names=FEATURE_NAMES[mode],
                scaler=scaler, ridge=ridge, alpha=alpha, calibration_group_cv_rmse=cv_rmse,
                calibration_rows=len(calibration), calibration_ND_cells=n_cell,
                calibration_N_groups=int(calibration.N_params_B.nunique()),
                calibration_Q_levels=int(calibration.Q_score.nunique()),
                support={name: [float(calibration[name].min()), float(calibration[name].max())]
                         for name in ("N_params_B", "D_tokens_B", "Q_score")})


def predict_b8_calibrated(calibrator: dict, base_artifact_path: Path, frame: pd.DataFrame) -> np.ndarray:
    """Predict conditional B8 N-D-Q loss after checking the canonical artifact hash."""
    if calibrator.get("source") != "B8_labeled":
        raise ValueError("This calibrator is only for the labeled B8 source")
    if sha256(base_artifact_path) != calibrator["base_artifact_sha256"]:
        raise ValueError("Canonical P2 artifact differs from the artifact used for B8 calibration")
    base_artifact = joblib.load(base_artifact_path)
    mode = calibrator.get("mode", "full_interaction")
    return base_prediction(frame, base_artifact) + calibrator["ridge"].predict(calibrator["scaler"].transform(features(frame, mode)))


def run(base_model: Path, b8_file: Path, calibration_index: Path, evaluation_index: Path,
        output_dir: Path, mode: str = "full_interaction") -> dict:
    before = sha256(base_model)
    base_artifact = joblib.load(base_model)
    b8 = pd.read_csv(b8_file, dtype={"experiment_id": str})
    required = {"experiment_id", "N_params_B", "D_tokens_B", "Q_score", "val_loss", "data_type"}
    if required - set(b8.columns):
        raise ValueError(f"B8 file is missing required columns: {sorted(required - set(b8.columns))}")
    calibration, evaluation = select_disjoint(b8, read_ids(calibration_index), read_ids(evaluation_index))
    calibrator = fit_b8_calibrator(calibration, base_artifact, before, mode=mode)
    raw = base_prediction(evaluation, base_artifact)
    offset = float(np.mean(calibration.val_loss - base_prediction(calibration, base_artifact)))
    predictions = {
        "zero_shot_Model_C": raw,
        "source_intercept_control": raw + offset,
        ("source_quality_interaction" if mode == "full_interaction" else "source_q_only"):
            predict_b8_calibrated(calibrator, base_model, evaluation),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrator, output_dir / "b8_labeled_source_calibrator.joblib")
    rows = []
    for model, pred in predictions.items():
        rows += [dict(experiment_id=id_, N_params_B=n, D_tokens_B=d, Q_score=q, data_type=kind,
                      observed=y, predicted=p, model=model)
                 for id_, n, d, q, kind, y, p in zip(evaluation.experiment_id, evaluation.N_params_B,
                                                      evaluation.D_tokens_B, evaluation.Q_score,
                                                      evaluation.data_type, evaluation.val_loss, pred)]
    pd.DataFrame(rows).to_csv(output_dir / "b8_labeled_source_predictions.csv", index=False)
    metrics = {name: {"n": len(evaluation),
                      "rmse": float(np.sqrt(np.mean((evaluation.val_loss.to_numpy(float) - pred) ** 2))),
                      "mae": float(np.mean(np.abs(evaluation.val_loss.to_numpy(float) - pred)))}
               for name, pred in predictions.items()}
    meta = dict(protocol="B8 observed-label calibration on explicit N-D cells; disjoint held-out N-D cells", mode=mode,
                canonical_model_sha256_before=before, canonical_model_sha256_after=sha256(base_model),
                canonical_model_unchanged=(before == sha256(base_model)),
                calibration_index_sha256=sha256(calibration_index), evaluation_index_sha256=sha256(evaluation_index),
                calibration_rows=len(calibration), calibration_ND_cells=len(set(groups(calibration))),
                calibration_N_groups=int(calibration.N_params_B.nunique()),
                calibration_Q_levels=int(calibration.Q_score.nunique()),
                evaluation_rows=len(evaluation), evaluation_ND_cells=len(set(groups(evaluation))),
                evaluation_N_groups=int(evaluation.N_params_B.nunique()),
                evaluation_Q_levels=int(evaluation.Q_score.nunique()),
                calibration_fraction_of_selected_rows=len(calibration)/(len(calibration)+len(evaluation)),
                alpha=calibrator["alpha"], calibration_group_cv_rmse=calibrator["calibration_group_cv_rmse"],
                metrics=metrics, limitation="B8 data_type includes calibrated and extrapolated attachment labels; this is supervised within-source adaptation, not zero-shot cross-source generalization")
    (output_dir / "b8_labeled_source_metrics.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if before != sha256(base_model):
        raise RuntimeError("Canonical P2 artifact changed during optional calibration")
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--b8-file", required=True, type=Path)
    parser.add_argument("--calibration-index", required=True, type=Path)
    parser.add_argument("--evaluation-index", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=tuple(FEATURE_NAMES), default="full_interaction",
                        help="full_interaction requires two D cells per N; q_only supports one-cell designs")
    args = parser.parse_args()
    print(json.dumps(run(args.base_model, args.b8_file, args.calibration_index,
                         args.evaluation_index, args.output_dir, mode=args.mode), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
