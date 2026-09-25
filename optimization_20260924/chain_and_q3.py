"""Read-only cross-question interface audit and P3 cap sensitivity experiment.

Run from the project root: python optimization_20260924/chain_and_q3.py
All new results are written beside this script; existing model outputs stay intact.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "problem1"))
sys.path.insert(0, str(ROOT / "problem3_v2"))
from problem1_v2 import transform  # noqa: E402
from methods import Model, capped_solve, solve, total_cost  # noqa: E402


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p1_path = ROOT / "problem1/outputs/models/mixture_v2.joblib"
    p2_path = ROOT / "problem2_v2/outputs/models/recommended_model.joblib"
    interface_path = ROOT / "problem2_v2/outputs/tables/problem2_to_problem3_v2.json"
    p3_path = ROOT / "problem3_v2/outputs/tables/extrapolation_levels.csv"
    p4_path = ROOT / "problem4/outputs/tables/mechanistic_forecast.csv"
    p1 = joblib.load(p1_path)
    p2 = joblib.load(p2_path)
    interface = json.loads(interface_path.read_text(encoding="utf-8"))
    b1 = pd.read_csv(ROOT / "real_attachments/B_scaling_laws/pythia_training_log_existing.csv")
    p = pd.read_csv(ROOT / "real_attachments/A_data_value/regmix_tables/train_mixture_1m.csv")
    assert p1["pcols"] == p2["mixture_model"]["pcols"]
    x = p[p1["pcols"]].to_numpy(float)
    x /= x.sum(axis=1, keepdims=True)

    a, b = p1, p2["mixture_model"]
    y1 = a["model"].predict(transform(x, a["family"], a["eps"], a["reference_index"]))
    y2 = b["model"].predict(transform(x, b["family"], b["eps"], b["reference_index"]))
    mixture_max_difference = float(np.max(np.abs(y1 - y2)))
    assert mixture_max_difference < 1e-10, "P1 V2 and embedded P2 mixture predictions differ"
    assert (a["family"], a["eps"], a["reference_index"]) == (b["family"], b["eps"], b["reference_index"])

    names = ["E", "A", "alpha", "B", "beta", "G", "kappa", "eta_N"]
    p2_parameters = np.array([*p2["ND_parameters"], *p2["Q_parameters"]], float)
    json_parameters = np.array([interface["parameters"][key] for key in names], float)
    parameter_max_difference = float(np.max(np.abs(p2_parameters - json_parameters)))
    assert parameter_max_difference < 1e-12, "P2 artifact and P3/P4 interface parameters differ"
    assert float(p2["lambda_p"]) == float(interface["lambda_p_default"])
    model = Model(*p2_parameters)

    p3 = pd.read_csv(p3_path)
    p4 = pd.read_csv(p4_path)
    p4_ref = p4[p4.p_scenario == "P0_reference"].dropna(subset=["loss", "N_B", "D_B", "Q_B"])
    predicted_loss = np.array([model.loss(r.N_B, r.D_B, r.Q_B) for r in p4_ref.itertuples()])
    p4_loss_max_difference = float(np.max(np.abs(predicted_loss - p4_ref.loss.to_numpy(float))))
    assert p4_loss_max_difference < 1e-10, "P4 mechanism losses differ from the P2 artifact"
    p4_budget_violations = 0
    for r in p4_ref.itertuples():
        actual, *_ = total_cost(r.N_B, r.D_B, r.Q_B, .6, 4096, "exponential")
        p4_budget_violations += actual > r.compute_budget * (1 + 1e-8)
    assert p4_budget_violations == 0

    audit = {
        "status": "passed",
        "artifact_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): digest(path)
                            for path in [p1_path, p2_path, interface_path, p3_path, p4_path]},
        "p1_to_p2_training_prediction_max_abs_difference": mixture_max_difference,
        "p2_to_p3_p4_parameter_max_abs_difference": parameter_max_difference,
        "p4_mechanism_loss_max_abs_difference": p4_loss_max_difference,
        "p4_reference_rows_checked": int(len(p4_ref)),
        "p4_budget_violations": int(p4_budget_violations),
        "quality_scale_link_identified": False,
        "lambda_p_cross_scale_identified": False,
    }
    (OUT / "interface_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    n_bounds = (float(b1.N_params_B.min()), float(b1.N_params_B.max()))
    d_bounds = (float(b1.D_tokens_B.min()), float(b1.D_tokens_B.max()))
    rows = []
    for budget in [1e22, 1e24]:
        for factor in [1, 2, 3, 5, 10]:
            result = capped_solve(model, budget, .6, 4096, "exponential",
                                  (n_bounds[0], n_bounds[1] * factor),
                                  (d_bounds[0], d_bounds[1] * factor), global_search=True)
            rows.append({"budget": budget, "cap_factor": factor, "regime": "capped", **result})
        result = solve(model, budget, .6, 4096, "exponential", global_search=True)
        rows.append({"budget": budget, "cap_factor": np.nan, "regime": "free", **result})
    profile = pd.DataFrame(rows)
    profile.to_csv(OUT / "p3_cap_profile.csv", index=False, float_format="%.12g")
    old = p3[(p3.level == "E1_moderate_3x") & (p3.budget == 1e24)].iloc[0]
    new = profile[(profile.regime == "capped") & (profile.budget == 1e24) & (profile.cap_factor == 3)].iloc[0]
    assert abs(old.loss - new.loss) < 1e-8
    print(json.dumps({"interface": audit["status"], "p4_rows": len(p4_ref),
                      "p3_1e24": profile.loc[profile.budget == 1e24,
                                              ["regime", "cap_factor", "N_B", "D_B", "loss", "budget_ratio"]].to_dict("records")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
