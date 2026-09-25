"""Prequential quantile frontier challenge on strictly matched open models."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
data = pd.read_csv(ROOT / "problem4/outputs/tables/analysis_dataset.csv", low_memory=False)
data["submission_date"] = pd.to_datetime(data.submission_date, errors="coerce")
for col in ("complete", "open_w1"):
    data[col] = data[col].astype(str).str.lower().eq("true")
data["ability"] = pd.to_numeric(data.ability, errors="coerce")
data = data[data.complete & data.open_w1 & data.submission_date.notna() & data.ability.notna()]
data = data.sort_values(["submission_date", "ability"]).drop_duplicates("Model", keep="last")
data["quarter"] = data.submission_date.dt.to_period("Q").astype(str)
quarters = ["2024Q2", "2024Q3", "2024Q4", "2025Q1"]
data = data[data.quarter.isin(quarters)]
assert [int((data.quarter == q).sum()) for q in quarters] == [22, 16, 5, 2]
rows = []
for qi in (1, 2):  # at least two observed quarters before an out-of-time target
    train = data[data.quarter.isin(quarters[:qi + 1])].copy()
    target = data[data.quarter == quarters[qi + 1]].copy()
    train["time"] = train.quarter.map({q: i for i, q in enumerate(quarters)})
    for quantile in (0.9, 0.95):
        actual = float(target.ability.quantile(quantile))
        persistence = float(train.loc[train.quarter == quarters[qi], "ability"].quantile(quantile))
        model = QuantileRegressor(quantile=quantile, alpha=0.1, solver="highs")
        model.fit(train[["time"]], train.ability)
        forecast = float(model.predict(pd.DataFrame({"time": [qi + 1]}))[0])
        for name, value in (("quarterly_persistence", persistence),
                            ("linear_quantile_regression", forecast)):
            rows.append(dict(candidate=name, quantile=quantile,
                             origin=quarters[qi], target=quarters[qi + 1],
                             train_n=len(train), target_n=len(target),
                             prediction=value, actual=actual,
                             abs_error=abs(value - actual),
                             alpha=0.1 if name == "linear_quantile_regression" else np.nan))
out = pd.DataFrame(rows)
out.to_csv(HERE / "quantile_frontier_prequential.csv", index=False, encoding="utf-8-sig")
score = out.groupby(["candidate", "quantile"]).agg(
    targets=("abs_error", "size"), mae=("abs_error", "mean"),
    min_target_n=("target_n", "min")).reset_index()
score.to_csv(HERE / "quantile_frontier_scorecard.csv", index=False, encoding="utf-8-sig")
print(score.to_string(index=False))
