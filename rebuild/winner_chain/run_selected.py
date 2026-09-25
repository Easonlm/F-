"""Inject the validated P1 challenger and rerun all downstream legacy stages.

All writes stay in this isolated copy. The old v2 filenames here are adapter
slots required by legacy code, not the final project interface.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "rebuild/problem1"
STAGES = [
    ("problem2_base", "problem2/run_problem2.py"),
    ("problem2_final", "problem2_v2/run_v2.py"),
    ("problem3_base", "problem3/run_problem3.py"),
    ("problem3_final", "problem3_v2/run_v2.py"),
    ("problem3_selected_challenge", "rebuild/problem3/run_decision_tournament.py"),
    ("problem3_selected_verify", "rebuild/problem3/verify_tournament.py"),
    ("p3_paper_assets_selected", "rebuild/problem3/build_paper_assets.py"),
    ("problem4_final", "problem4/run_problem4.py"),
    ("problem4_selected_frontier_bridge", "rebuild/problem4/run_frontier_bridge.py"),
    ("problem4_selected_bridge_structures", "rebuild/problem4/run_bridge_structures.py"),
    ("problem4_selected_multitask", "rebuild/problem4/run_ability_multitask.py"),
    ("problem4_selected_quantile", "rebuild/problem4/run_quantile_frontier.py"),
    ("problem4_selected_forecast", "rebuild/problem4/run_forecast_backtest.py"),
    ("problem4_selected_selection", "rebuild/problem4/select_models.py"),
]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    resume = "--resume" in sys.argv[1:]
    status_path = ROOT / "selected_rerun_status.json"
    previous = json.loads(status_path.read_text(encoding="utf-8")) if resume and status_path.exists() else []
    passed = {row["stage"]: row for row in previous if row["exit_code"] == 0}
    dest = ROOT / "problem1/outputs/models/mixture_v2.joblib"
    new = SOURCE / "selected_model.joblib"
    if not new.exists():
        raise SystemExit("Selected P1 bundle missing")
    provenance_path = ROOT / "adapter_provenance.json"
    original_hash = (json.loads(provenance_path.read_text(encoding="utf-8"))["old_slot_sha256"]
                     if resume and provenance_path.exists() else sha(dest))
    shutil.copy2(new, dest)
    # Legacy downstream code filters `version == v2`; keep that compatibility
    # label in this isolated adapter table while preserving explicit provenance.
    scale = pd.read_csv(SOURCE / "selected_scale_transfer_summary.csv")
    scale["version"] = "v2"
    scale["evidence_status"] = "selected_ALR_ElasticNet_retrospective_not_blind"
    old = pd.read_csv(ROOT / "problem1/outputs/tables/v2_scale_transfer_summary.csv")
    out = pd.concat([old[old.version.eq("v1")], scale], ignore_index=True)
    out.to_csv(ROOT / "problem1/outputs/tables/v2_scale_transfer_summary.csv", index=False)
    selected_predictions = SOURCE / "selected_scale_mixture_predictions.csv"
    shutil.copy2(selected_predictions, ROOT / "problem1/outputs/tables/selected_scale_mixture_predictions.csv")
    provenance_path.write_text(json.dumps({
        "legacy_slot": "problem1/outputs/models/mixture_v2.joblib",
        "meaning": "P1 selected ALR MultiTask ElasticNet, not old V2 Ridge",
        "old_slot_sha256": original_hash,
        "selected_model_sha256": sha(new),
        "selected_scale_source_sha256": sha(SOURCE / "selected_scale_transfer_summary.csv"),
    }, indent=2), encoding="utf-8")
    prior = ROOT / "problem2_v2/outputs/tables/v1_sha256.json"
    if prior.exists() and "problem2_final" not in passed:
        prior.unlink()
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    status = []
    for name, script in STAGES:
        if name in passed and (ROOT / f"selected_{name}.log").exists():
            status.append(passed[name])
            print(f"SKIP {name} previously passed", flush=True)
            continue
        print(f"START {name}", flush=True)
        t = time.time()
        with (ROOT / f"selected_{name}.log").open("w", encoding="utf-8") as f:
            p = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT,
                               env=env, stdout=f, stderr=subprocess.STDOUT)
        record = dict(stage=name, script=script, exit_code=p.returncode, seconds=round(time.time()-t, 2))
        status.append(record)
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(f"END {name} exit={p.returncode} seconds={record['seconds']}", flush=True)
        if p.returncode:
            raise SystemExit(f"Failed at {name}; see selected_{name}.log")

    sys.path.insert(0, str(ROOT / "problem1"))
    p1 = joblib.load(dest)
    p2_path = ROOT / "problem2_v2/outputs/models/recommended_model.joblib"
    p2 = joblib.load(p2_path)
    assert p1["family"] == p2["mixture_model"]["family"] == "poly_alr"
    assert p1["reference_index"] == p2["mixture_model"]["reference_index"]
    assert p1["pcols"] == p2["mixture_model"]["pcols"]
    assert np.allclose(p1["model"].steps[-1][1].coef_, p2["mixture_model"]["model"].steps[-1][1].coef_)
    p3 = json.loads((ROOT / "problem2_v2/outputs/tables/problem2_to_problem3_v2.json").read_text(encoding="utf-8"))
    verification = json.loads((ROOT / "problem4/outputs/tables/verification_results.json").read_text(encoding="utf-8"))
    lineage = {
        "P1_selected_sha256": sha(dest),
        "P2_recommended_sha256": sha(p2_path),
        "P2_embeds_selected_P1_coefficients": True,
        "P3_interface_sha256": sha(ROOT / "problem2_v2/outputs/tables/problem2_to_problem3_v2.json"),
        "P3_interface_model": p3.get("model", p3.get("model_type", "see interface")),
        "P3_robust_table_sha256": sha(ROOT / "problem3_v2/outputs/tables/robust_optimization.csv"),
        "P4_summary_sha256": sha(ROOT / "problem4/outputs/tables/final_project_summary.json"),
        "P4_verification_passed": sum(bool(x["passed"]) for x in verification.values()),
        "P4_verification_total": len(verification),
        "P4_failed_evidence_gates": [k for k, v in verification.items() if not v["passed"]],
    }
    (ROOT / "selected_lineage.json").write_text(json.dumps(lineage, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(lineage, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
