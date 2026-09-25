"""As-of 2025-01-31 held-out C4 compute-frontier check, not an ability backtest.

Run from the project root:
    python paper_work/compute_asof_20250131.py
Only paper_work/compute_asof_20250131_*.{csv,json} are written.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "real_attachments/C_efficiency_evolution/epoch_all_ai_models.csv"
ORIGIN = pd.Timestamp("2025-01-31")
TARGET = pd.Period("2026Q1", freq="Q")


def main() -> None:
    d = pd.read_csv(SOURCE, low_memory=False)
    d["publication_date"] = pd.to_datetime(d["Publication date"], errors="coerce")
    d["compute"] = pd.to_numeric(d["Training compute (FLOP)"], errors="coerce")
    d = d[
        d.Domain.eq("Language")
        & d["Open model weights?"].eq("Yes")
        & d.Confidence.isin(["Confident", "Likely"])
        & (d.compute > 0)
        & d.publication_date.notna()
    ].copy()
    d["quarter"] = d.publication_date.dt.to_period("Q")

    # The origin quarter was incomplete on 2025-01-31. The last visible full
    # quarter is 2024Q4, matching the project's current C4 compute anchor.
    last_complete = ORIGIN.to_period("Q") - 1
    train = d[(d.publication_date <= ORIGIN) & (d.quarter <= last_complete)].copy()
    target = d[d.quarter.eq(TARGET)].copy()
    if train.publication_date.max() > ORIGIN or target.publication_date.min() <= ORIGIN:
        raise AssertionError("As-of split is not chronological")

    q = (
        train.groupby("quarter")
        .agg(n=("Model", "size"), p90_compute=("compute", lambda s: s.quantile(0.9)))
        .reset_index()
    )
    q = q[(q.quarter >= pd.Period("2021Q1", freq="Q")) & (q.n >= 3)].copy()
    q = q.sort_values("quarter")
    if q.quarter.iloc[-1] != last_complete or len(q) < 8:
        raise AssertionError("Insufficient complete as-of compute quarters")
    q["quarter"] = q.quarter.astype(str)
    q.to_csv(HERE / "compute_asof_20250131_train_quarters.csv", index=False, encoding="utf-8-sig")

    target_n = len(target)
    actual = float(target.compute.quantile(0.9)) if target_n else np.nan
    target[["Model", "Publication date", "Confidence", "Training compute (FLOP)"]].to_csv(
        HERE / "compute_asof_20250131_target_models.csv", index=False, encoding="utf-8-sig"
    )

    periods = pd.PeriodIndex(q.quarter, freq="Q")
    t = (periods.asi8 - periods[0].ordinal) / 4.0
    log_p90 = np.log(q.p90_compute.to_numpy(float))
    recent_mask = periods >= pd.Period("2023Q1", freq="Q")
    long_slope = float(np.polyfit(t, log_p90, 1)[0])
    recent_slope = float(np.polyfit(t[recent_mask], log_p90[recent_mask], 1)[0])
    old = pd.read_csv(ROOT / "problem4/outputs/tables/compute_growth_models.csv")
    expected = float(old.loc[old.model == "recent_2023_2024", "slope_log_per_year"].iloc[0])
    if not np.isclose(recent_slope, expected, rtol=1e-12, atol=1e-12):
        raise AssertionError("Recent as-of slope does not reproduce the existing P4 input")

    anchor = float(q.p90_compute.iloc[-1])
    quarters_ahead = TARGET.ordinal - last_complete.ordinal
    years_ahead = quarters_ahead / 4.0
    rows = []
    for name, slope in [("last_complete_quarter", 0.0),
                        ("long_loglinear_2021", long_slope),
                        ("recent_loglinear_2023", recent_slope)]:
        pred = float(anchor * np.exp(slope * years_ahead))
        rows.append(
            dict(
                model=name,
                asof_date=str(ORIGIN.date()),
                last_complete_quarter=str(last_complete),
                target_quarter=str(TARGET),
                train_eligible_quarters=len(q),
                target_n=target_n,
                target_n_ge_5=bool(target_n >= 5),
                quarters_from_anchor=quarters_ahead,
                anchor_p90_compute=anchor,
                slope_log_per_year=slope,
                predicted_p90_compute=pred,
                observed_p90_compute=actual,
                predicted_over_observed=pred / actual if actual > 0 else np.nan,
                absolute_log_error=abs(np.log(pred / actual)) if actual > 0 else np.nan,
                status="diagnostic only: one held-out quarter with n<5"
                if target_n < 5 else "single held-out quarter",
            )
        )
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "compute_asof_20250131_results.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "input": str(SOURCE.relative_to(ROOT)),
        "input_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "origin": str(ORIGIN.date()),
        "last_complete_quarter": str(last_complete),
        "target_quarter": str(TARGET),
        "target_n": target_n,
        "target_n_ge_5": target_n >= 5,
        "recent_slope_reproduced": True,
        "ability_scores_used": False,
        "interpretation": "C4 compute-only held-out diagnostic; not a benchmark-ability validation",
    }
    (HERE / "compute_asof_20250131_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
