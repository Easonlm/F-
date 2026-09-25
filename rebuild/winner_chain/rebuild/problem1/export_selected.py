"""Export the chosen mixture model in the frozen V2 downstream bundle schema."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from run_tournament import DATA, OUT, ROOT, coordinates, pair, sha256


def main():
    selection = pd.read_csv(OUT / "problem1_model_tournament.csv")
    chosen = selection.loc[selection.decision.eq("NEW_CROSS_QUESTION_CHAMPION"), "family"].tolist()
    if chosen != ["alr_elasticnet"]:
        raise RuntimeError(f"Unexpected selected family: {chosen}")
    source = joblib.load(OUT / "alr_elasticnet.joblib")
    bundle = dict(model=source["model"], family="poly_alr", eps=float(source["eps"]),
                  reference_index=int(source["ref"]), pcols=source["pcols"],
                  ycols=source["ycols"], challenger_family="alr_elasticnet",
                  selected_spec=source["spec"],
                  validation_status="retrospective_tournament_not_new_blind_test")
    path = OUT / "selected_model.joblib"
    joblib.dump(bundle, path)
    rng = np.random.default_rng(20260925)
    summary, preds = [], []
    for label, prefix, scale in (("test_1m", "test", "1m"),
                                 ("test_60m", "test", "60m"),
                                 ("test_1B", "test", "1B")):
        ix, p, y, pcols, ycols, _ = pair(prefix, scale)
        if pcols != bundle["pcols"] or ycols != bundle["ycols"]:
            raise ValueError("Input column mismatch")
        pred = bundle["model"].predict(coordinates(p, "alr_elasticnet",
                                                    bundle["eps"], bundle["reference_index"]))
        predicted_mean, actual_mean = pred.mean(axis=1), y.mean(axis=1)
        slope = float(np.cov(predicted_mean, actual_mean, ddof=0)[0, 1]
                      / np.var(predicted_mean))
        slopes = []
        for _ in range(3000):
            draw = rng.integers(len(y), size=len(y))
            xx, yy = predicted_mean[draw], actual_mean[draw]
            if np.var(xx) > 1e-12:
                slopes.append(float(np.cov(xx, yy, ddof=0)[0, 1] / np.var(xx)))
        summary.append(dict(dataset=label, version="elasticnet_selected", n_mixtures=len(y),
                            mean_loss_spearman=float(spearmanr(predicted_mean, actual_mean).statistic),
                            centered_slope=slope, slope_ci_low=float(np.quantile(slopes, .025)),
                            slope_ci_high=float(np.quantile(slopes, .975)),
                            mean_loss_bias=float(np.mean(predicted_mean-actual_mean)),
                            evidence_status="retrospective_test_diagnostic_not_new_blind_validation"))
        preds.extend(dict(dataset=label, version="elasticnet_selected", index=idx,
                          actual_mean_loss=float(actual_mean[i]),
                          predicted_mean_loss=float(predicted_mean[i]))
                     for i, idx in enumerate(ix))
    pd.DataFrame(summary).to_csv(OUT / "selected_scale_transfer_summary.csv", index=False)
    pd.DataFrame(preds).to_csv(OUT / "selected_scale_mixture_predictions.csv", index=False)
    metadata = dict(selected_model_sha256=sha256(path), family="poly_alr",
                    original_family="alr_elasticnet", reference_index=bundle["reference_index"],
                    eps=bundle["eps"], source_challenger_sha256=sha256(OUT / "alr_elasticnet.joblib"))
    (OUT / "selected_export_manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(metadata)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
