"""Independent, read-only P1 scale transfer and B6/B8 contradiction audit.

Run from the repository root: python optimization_20260924/q12/run_p1_scale_and_b8.py
Inputs are frozen existing artifacts and the original A/B attachments. Outputs are
written only beside this script. The test sets are previously inspected project
sets, so this is a retrospective diagnostic, not a fresh blind validation.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
A = ROOT / "real_attachments" / "A_data_value" / "regmix_tables"
B = ROOT / "real_attachments" / "B_scaling_laws"
P1 = ROOT / "problem1" / "outputs"
SEED = 20260924


def paired_a(scale: str):
    mix = pd.read_csv(A / f"test_mixture_{scale}.csv")
    loss = pd.read_csv(A / f"test_pile_loss_{scale}.csv")
    assert mix["index"].is_unique and loss["index"].is_unique
    x = mix.merge(loss, on="index", validate="one_to_one")
    assert len(x) == len(mix) == len(loss)
    pcols = [c for c in mix if c.startswith("train_the_pile_")]
    ycols = [c for c in loss if c.startswith("metric/the_pile_")]
    p = x[pcols].to_numpy(float)
    assert np.all(p >= 0) and np.allclose(p.sum(axis=1), 1, atol=0.005)
    p = p / p.sum(axis=1, keepdims=True)
    return x["index"].to_numpy(), p, x[ycols].to_numpy(float), pcols, ycols


def alr(p, eps, ref):
    q = p + eps
    q /= q.sum(axis=1, keepdims=True)
    return np.log(q[:, np.arange(p.shape[1]) != ref] / q[:, ref, None])


def fit_line(x, y):
    slope = float(np.cov(x, y, ddof=0)[0, 1] / np.var(x))
    intercept = float(np.mean(y) - slope * np.mean(x))
    return intercept, slope


def rmse(x, y):
    return float(np.sqrt(np.mean((x - y) ** 2)))


def main_p1():
    v1 = joblib.load(P1 / "models" / "mixture_primary.joblib")
    v2 = joblib.load(P1 / "models" / "mixture_v2.joblib")
    metadata = json.loads((P1 / "tables" / "mixture_metadata.json").read_text(encoding="utf-8"))
    assert metadata["selected_model_by_train_cv"] == "alr_ridge"
    assert v2["family"] == "poly_alr"
    raw_rows, slope_rows, pred_rows, holdout_rows, sensitivity_rows = [], [], [], [], []
    for scale in ("1m", "60m", "1B"):
        ids, p, y, pcols, ycols = paired_a(scale)
        assert pcols == v2["pcols"] and ycols == v2["ycols"]
        pred_v1 = v1.predict(alr(p, metadata["alr_zero_pseudocount"], p.shape[1] - 1))
        pred_v2 = v2["model"].predict(alr(p, v2["eps"], v2["reference_index"]))
        actual_mean = y.mean(axis=1)
        for name, pred in (("v1", pred_v1), ("v2", pred_v2)):
            pred_mean = pred.mean(axis=1)
            raw_rows.append(dict(scale=scale, version=name, n=len(y), raw_rmse=rmse(pred, y),
                                 raw_bias=float(np.mean(pred - y)), mean_loss_rmse=rmse(pred_mean, actual_mean),
                                 centered_mean_rmse=rmse(pred_mean - pred_mean.mean(), actual_mean - actual_mean.mean()),
                                 mean_loss_spearman=float(spearmanr(pred_mean, actual_mean).statistic)))
            intercept, slope = fit_line(pred_mean, actual_mean)
            rng = np.random.default_rng(SEED + {"1m": 1, "60m": 2, "1B": 3}[scale])
            boots = []
            for _ in range(3000):
                ix = rng.integers(len(y), size=len(y))
                if np.var(pred_mean[ix]) > 1e-12:
                    boots.append(fit_line(pred_mean[ix], actual_mean[ix])[1])
            slope_rows.append(dict(scale=scale, version=name, n=len(y), centered_slope=slope,
                                   slope_ci_low=float(np.quantile(boots, 0.025)),
                                   slope_ci_high=float(np.quantile(boots, 0.975)),
                                   intercept=intercept,
                                   mean_loss_spearman=float(spearmanr(pred_mean, actual_mean).statistic)))
            pred_rows.extend(dict(scale=scale, version=name, index=ids[i], actual_mean_loss=actual_mean[i],
                                  predicted_mean_loss=pred_mean[i]) for i in range(len(y)))

            # Primary fixed protocol: 25% of target-scale mixtures calibrate; 75% are held out.
            # Compare a fixed unit effect (intercept only) against one added slope parameter.
            split_rng = np.random.default_rng(SEED + {"1m": 11, "60m": 12, "1B": 13}[scale])
            ix = split_rng.permutation(len(y))
            k = max(4, int(round(len(y) * 0.25)))
            cal, hold = ix[:k], ix[k:]
            intercept_only = float(np.mean(actual_mean[cal] - pred_mean[cal]))
            a, b = fit_line(pred_mean[cal], actual_mean[cal])
            q1 = intercept_only + pred_mean[hold]
            q2 = a + b * pred_mean[hold]
            # Paired bootstrap over held-out mixture rows, conditional on the fixed calibration.
            boot_rng = np.random.default_rng(SEED + {"1m": 21, "60m": 22, "1B": 23}[scale])
            diffs = []
            for _ in range(3000):
                j = boot_rng.integers(len(hold), size=len(hold))
                diffs.append(rmse(q1[j], actual_mean[hold][j]) - rmse(q2[j], actual_mean[hold][j]))
            holdout_rows.append(dict(scale=scale, version=name, calibration_rows=len(cal), holdout_rows=len(hold),
                                     calibrated_slope=b, intercept_only_rmse=rmse(q1, actual_mean[hold]),
                                     affine_rmse=rmse(q2, actual_mean[hold]),
                                     rmse_gain_intercept_minus_affine=rmse(q1, actual_mean[hold]) - rmse(q2, actual_mean[hold]),
                                     gain_ci_low=float(np.quantile(diffs, 0.025)),
                                     gain_ci_high=float(np.quantile(diffs, 0.975))))
            # Robustness to the arbitrary calibration sample; never select a split by its score.
            for rep in range(20):
                rr = np.random.default_rng(SEED + rep + 1000 + {"1m": 100, "60m": 200, "1B": 300}[scale])
                ij = rr.permutation(len(y))
                ci, hi = ij[:k], ij[k:]
                ai, bi = fit_line(pred_mean[ci], actual_mean[ci])
                offset = float(np.mean(actual_mean[ci] - pred_mean[ci]))
                base = rmse(offset + pred_mean[hi], actual_mean[hi])
                affine = rmse(ai + bi * pred_mean[hi], actual_mean[hi])
                sensitivity_rows.append(dict(scale=scale, version=name, repetition=rep, calibration_rows=k,
                                             calibrated_slope=bi, intercept_only_rmse=base,
                                             affine_rmse=affine, gain=base - affine))
    pd.DataFrame(raw_rows).to_csv(OUT / "p1_raw_metrics.csv", index=False)
    pd.DataFrame(slope_rows).to_csv(OUT / "p1_centered_slopes.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(OUT / "p1_scale_predictions.csv", index=False)
    pd.DataFrame(holdout_rows).to_csv(OUT / "p1_calibration_holdout.csv", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(OUT / "p1_calibration_sensitivity.csv", index=False)


def main_b8():
    b6 = pd.read_csv(B / "supplementary_NQ_experiment.csv")
    b8 = pd.read_csv(B / "supplementary_NQ_experiment_large.csv")
    keys = ["N_params_B", "D_tokens_B", "Q_score"]
    overlap = b6.merge(b8, on=keys, validate="one_to_one", suffixes=("_b6", "_b8"))
    overlap["loss_diff_b8_minus_b6"] = overlap["val_loss_b8"] - overlap["val_loss_b6"]
    overlap.to_csv(OUT / "b6_b8_exact_overlap.csv", index=False)
    overlap.groupby("Q_score", as_index=False).agg(
        n=("loss_diff_b8_minus_b6", "size"),
        mean_b8_minus_b6=("loss_diff_b8_minus_b6", "mean"),
        sd_b8_minus_b6=("loss_diff_b8_minus_b6", "std"),
    ).to_csv(OUT / "b6_b8_overlap_by_q.csv", index=False)
    rows = []
    for source, df in (("B6", b6), ("B8", b8)):
        for (n, d), z in df.groupby(["N_params_B", "D_tokens_B"]):
            if z.Q_score.nunique() < 3:
                continue
            rows.append(dict(source=source, N_params_B=n, D_tokens_B=d, n_q=len(z),
                             loss_vs_q_slope=fit_line(z.Q_score.to_numpy(), z.val_loss.to_numpy())[1],
                             loss_q_spearman=float(spearmanr(z.Q_score, z.val_loss).statistic)))
    sl = pd.DataFrame(rows)
    sl.to_csv(OUT / "b6_b8_quality_directions.csv", index=False)
    summary = {
        "overlap_points": len(overlap),
        "overlap_mean_b8_minus_b6": float(overlap.loss_diff_b8_minus_b6.mean()),
        "overlap_rmse_difference": rmse(overlap.val_loss_b8, overlap.val_loss_b6),
        "overlap_q01_mean_difference": float(overlap.loc[np.isclose(overlap.Q_score, 0.1), "loss_diff_b8_minus_b6"].mean()),
        "overlap_q10_mean_difference": float(overlap.loc[np.isclose(overlap.Q_score, 1.0), "loss_diff_b8_minus_b6"].mean()),
        "b6_fraction_negative_q_slope": float((sl.loc[sl.source == "B6", "loss_vs_q_slope"] < 0).mean()),
        "b8_fraction_negative_q_slope": float((sl.loc[sl.source == "B8", "loss_vs_q_slope"] < 0).mean()),
        "b6_nd_groups": int((sl.source == "B6").sum()),
        "b8_nd_groups": int((sl.source == "B8").sum()),
    }
    (OUT / "b6_b8_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def write_provenance():
    inputs = [P1 / "models" / "mixture_primary.joblib", P1 / "models" / "mixture_v2.joblib",
              P1 / "tables" / "mixture_metadata.json",
              B / "supplementary_NQ_experiment.csv", B / "supplementary_NQ_experiment_large.csv"]
    for scale in ("1m", "60m", "1B"):
        inputs.extend([A / f"test_mixture_{scale}.csv", A / f"test_pile_loss_{scale}.csv"])
    manifest = {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in inputs}
    (OUT / "input_sha256.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    main_p1()
    main_b8()
    write_provenance()
    print("Wrote independent P1 scale and B6/B8 results to", OUT)
