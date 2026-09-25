"""Leakage-free short-horizon frontier diagnostics on existing P4 backtest origins.

Run at project root: python optimization_20260924/round2/q4/run_short_term.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "problem4"))
from history_forecast import logit_score, score  # noqa: E402


def main() -> None:
    q = pd.read_csv(ROOT / "problem4/outputs/tables/quarterly_frontier.csv")
    q = q[q.subset.eq("open_w1")].sort_values("quarter").reset_index(drop=True)
    old = pd.read_csv(ROOT / "problem4/outputs/tables/forecast_backtest.csv")
    rows = []
    for h in (1, 2, 4, 8):
        for origin in range(1, len(q) - h):
            train = q.iloc[: origin + 1]
            target = q.iloc[origin + h]
            # Same logit trend fit as the existing baseline. Coefficients are
            # calculated using only quarterly observations available at origin.
            y = logit_score(train.p95_ability)
            slope = np.polyfit(np.arange(len(train)), y, 1)[0]
            for name, slope_factor in (("existing_direct_time", 1.0),
                                       ("half_damped_time", 0.5),
                                       ("persistence", 0.0)):
                prediction = float(score(y[-1] + slope_factor * slope * h))
                rows.append({"model": name, "horizon_quarters": h,
                             "origin_quarter": train.iloc[-1].quarter,
                             "target_quarter": target.quarter,
                             "train_quarters": len(train),
                             "origin_n": int(train.iloc[-1].n),
                             "target_n": int(target.n),
                             "prediction": prediction,
                             "actual": float(target.p95_ability),
                             "error": prediction - float(target.p95_ability),
                             "abs_error": abs(prediction - float(target.p95_ability)),
                             "target_n_ge_5": bool(target.n >= 5)})
    result = pd.DataFrame(rows)
    baseline = result[result.model.eq("existing_direct_time")]
    check = baseline.merge(old, on=["horizon_quarters", "origin_quarter", "target_quarter"],
                           suffixes=("_new", "_old"), validate="one_to_one")
    if len(check) != len(old) or not np.allclose(check.prediction_new, check.prediction_old, atol=1e-10):
        raise AssertionError("Original direct-time backtest was not reproduced")
    result.to_csv(HERE / "short_term_predictions.csv", index=False, encoding="utf-8-sig")
    metrics = (
        result.groupby(["model", "horizon_quarters"])
        .agg(targets=("error", "size"), mae=("abs_error", "mean"),
             rmse=("error", lambda x: np.sqrt(np.mean(x ** 2))),
             min_target_n=("target_n", "min"))
        .reset_index()
    )
    metrics.to_csv(HERE / "short_term_scorecard.csv", index=False, encoding="utf-8-sig")
    qualified = result[result.target_n_ge_5]
    qualified_metrics = (
        qualified.groupby(["model", "horizon_quarters"])
        .agg(targets=("error", "size"), mae=("abs_error", "mean"),
             rmse=("error", lambda x: np.sqrt(np.mean(x ** 2))))
        .reset_index()
    )
    qualified_metrics.to_csv(HERE / "short_term_target_n_ge_5.csv", index=False, encoding="utf-8-sig")
    metadata = {"original_backtest_rows": len(old), "strict_frontier_quarters": len(q),
                "baseline_predictions_reproduced": True,
                "candidate_factors_set_before_scoring": {"half_damped_time": 0.5, "persistence": 0.0},
                "long_horizon_backtest_available": False}
    (HERE / "short_term_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
