"""Independent integrity checks for the rebuilt Problem 2 tournament."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_tournament import FILES, HERE, INPUT, KEYS, data, novel, predict_nd, predict_q, rmse


def main():
    raw = data()
    manifest = json.loads((HERE / "input_manifest.json").read_text(encoding="utf-8"))
    selection = json.loads((HERE / "selection.json").read_text(encoding="utf-8"))
    nd = pd.read_csv(HERE / "nd_scorecard.csv").set_index("model")
    q = pd.read_csv(HERE / "quality_scorecard.csv").set_index("model")
    nd_oof = pd.read_csv(HERE / "nd_group_oof.csv")
    q_oof = pd.read_csv(HERE / "quality_leave_N_oof.csv")
    b7_new = novel(raw["B6"], raw["B7"])
    b8_new = novel(raw["B7"], raw["B8"])
    theta = np.asarray(selection["ND_parameters"], float)
    phi = np.asarray(selection["Q_parameters"], float)
    checks = {}
    checks["raw_inputs_unchanged"] = all(
        manifest[k]["sha256"] == hashlib.sha256((INPUT / name).read_bytes()).hexdigest()
        and manifest[k]["rows"] == len(raw[k]) for k, name in FILES.items())
    checks["B7_new_only"] = len(b7_new) == 90 and len(b7_new.merge(raw["B6"][KEYS], on=KEYS)) == 0
    checks["B8_new_only"] = len(b8_new) == 1480 and len(b8_new.merge(raw["B7"][KEYS], on=KEYS)) == 0
    checks["B1_group_oof_complete"] = all(
        set(z.row_id) == set(range(len(raw["B1"]))) and len(z) == len(raw["B1"])
        for _, z in nd_oof.groupby("model"))
    checks["B6_leave_N_oof_complete"] = all(
        set(z.row_id) == set(range(len(raw["B6"]))) and len(z) == len(raw["B6"])
        for _, z in q_oof.groupby("model"))
    checks["B1_champion_CV_recomputed"] = np.isclose(
        rmse(*[nd_oof[nd_oof.model == "classic"][c] for c in ("observed", "predicted")]),
        nd.loc["classic", "B1_group_cv_rmse"], atol=1e-12)
    checks["B6_champion_CV_recomputed"] = np.isclose(
        rmse(*[q_oof[q_oof.model == "Q_by_N"][c] for c in ("observed", "predicted")]),
        q.loc["Q_by_N", "B6_leave_N_rmse"], atol=1e-12)
    b7_pred = predict_q("Q_by_N", phi, b7_new.N_params_B, b7_new.D_tokens_B, b7_new.Q_score, theta)
    checks["B7_champion_new_recomputed"] = np.isclose(rmse(b7_new.val_loss, b7_pred),
                                                        q.loc["Q_by_N", "B7_new_rmse"], atol=1e-12)
    checks["Q_equals_one_reduces_to_ND"] = all(
        np.isclose(predict_q("Q_by_N", phi, n, d, 1, theta), predict_nd("classic", theta, n, d))
        for n in (.07, .7, 7, 70) for d in (10, 100, 1000))
    checks["NDQ_monotone_on_B6_domain"] = all(
        predict_q("Q_by_N", phi, n * 1.01, d, q, theta) < predict_q("Q_by_N", phi, n, d, q, theta)
        and predict_q("Q_by_N", phi, n, d * 1.01, q, theta) < predict_q("Q_by_N", phi, n, d, q, theta)
        and predict_q("Q_by_N", phi, n, d, min(q + .01, 1), theta) <= predict_q("Q_by_N", phi, n, d, q, theta)
        for n in (.07, .7, 7) for d in (10, 100, 1000) for q in (.1, .5, .9))
    mechanism = selection["B8_mechanism"]
    checks["B8_mechanism_conflict"] = (
        mechanism["B6_B8_exact_overlap"]["n"] == 160
        and mechanism["B6"]["fraction_loss_declines_with_Q"] == 1
        and mechanism["B8_new"]["fraction_loss_declines_with_Q"] == 0)
    transfer = pd.read_csv(HERE / "source_leave_B5_out.csv")
    checks["source_transfer_B5_fully_heldout"] = bool((~transfer.B5_labels_used_for_fit).all())
    checks["lambda_scenarios_not_claimed_identified"] = bool(
        (~pd.read_csv(HERE / "lambda_p_scenarios.csv").identified).all())
    checks["preregistered_winners"] = selection["ND_winner"] == "classic" and selection["Q_winner"] == "Q_by_N"
    checks["all_tournament_metrics_finite"] = bool(np.isfinite(pd.read_csv(HERE / "problem2_model_tournament.csv").value).all())
    result = {"all_passed": bool(all(checks.values())), "checks": {k: bool(v) for k, v in checks.items()}}
    (HERE / "verification_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["all_passed"]:
        raise AssertionError("Problem 2 tournament verification failed")


if __name__ == "__main__":
    main()
