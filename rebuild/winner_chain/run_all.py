"""One-command independent reproduction of the provisional selected chain.

Includes historical baseline regeneration, all four retrospective challenges,
selected P1 injection, and a second P2-to-P4 dependency-chain rerun.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
STAGES = [
    ("baseline_p1_quality", "problem1/run_problem1.py"),
    ("baseline_p1_mixture", "problem1/problem1_v2.py"),
    ("baseline_p2", "problem2/run_problem2.py"),
    ("baseline_p2_model_c", "problem2_v2/run_v2.py"),
    ("baseline_p3", "problem3/run_problem3.py"),
    ("baseline_p3_robust", "problem3_v2/run_v2.py"),
    ("baseline_p4", "problem4/run_problem4.py"),
    ("p1_challenge", "rebuild/problem1/run_tournament.py"),
    ("p1_export", "rebuild/problem1/export_selected.py"),
    ("p1_verify", "rebuild/problem1/verify_tournament.py"),
    ("p1_q_sensitivity", "rebuild/problem1/run_q_sensitivity.py"),
    ("p1_assets", "rebuild/problem1/build_paper_assets.py"),
    ("p2_challenge", "rebuild/problem2/run_tournament.py"),
    ("p2_verify", "rebuild/problem2/verify_tournament.py"),
    ("p2_assets", "rebuild/problem2/build_deliverables.py"),
    ("p3_challenge", "rebuild/problem3/run_decision_tournament.py"),
    ("p3_verify", "rebuild/problem3/verify_tournament.py"),
    ("p3_assets", "rebuild/problem3/build_paper_assets.py"),
    ("p4_frontier_bridge", "rebuild/problem4/run_frontier_bridge.py"),
    ("p4_bridge_structures", "rebuild/problem4/run_bridge_structures.py"),
    ("p4_ability_multitask", "rebuild/problem4/run_ability_multitask.py"),
    ("p4_quantile_frontier", "rebuild/problem4/run_quantile_frontier.py"),
    ("p4_forecast", "rebuild/problem4/run_forecast_backtest.py"),
    ("p4_select", "rebuild/problem4/select_models.py"),
    ("selected_dependency_chain", "run_selected.py"),
]


def main():
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    resume = "--resume" in sys.argv[1:]
    status_path = ROOT / "one_click_status.json"
    previous = json.loads(status_path.read_text(encoding="utf-8")) if resume and status_path.exists() else []
    passed = {x["stage"]: x for x in previous if x["exit_code"] == 0}
    status = []
    for name, script in STAGES:
        if name in passed and (ROOT / passed[name]["log"]).exists():
            status.append(passed[name])
            print(f"SKIP {name} previously passed", flush=True)
            continue
        if name == "baseline_p2_model_c":
            old_sha = ROOT / "problem2_v2/outputs/tables/v1_sha256.json"
            if old_sha.exists():
                old_sha.unlink()
        print(f"START {name}", flush=True)
        started = time.time()
        log = ROOT / f"one_click_{name}.log"
        with log.open("w", encoding="utf-8") as f:
            p = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT,
                               env=env, stdout=f, stderr=subprocess.STDOUT)
        record = dict(stage=name, script=script, exit_code=p.returncode,
                      seconds=round(time.time()-started, 2), log=log.name)
        status.append(record)
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(f"END {name} exit={p.returncode} seconds={record['seconds']}", flush=True)
        if p.returncode:
            raise SystemExit(f"Stopped at {name}; see {log}")
    summary = json.loads((ROOT / "problem4/outputs/tables/final_project_summary.json").read_text(encoding="utf-8"))
    mix = pd.read_csv(ROOT / "problem3_v2/outputs/tables/mixture_support_diagnostics.csv")
    value = float(mix.loc[mix.strategy.eq("P1_observed_support"), "delta_p"].iloc[0])
    assert abs(summary["problem1"]["p1_delta_loss"] - value) < 1e-10
    assert all(x["exit_code"] == 0 for x in status)
    (ROOT / "one_click_run_summary.json").write_text(json.dumps({
        "all_passed": True,
        "resumed": resume,
        "stages": len(status),
        "executed_this_invocation": len(status) - sum(
            1 for row in status if row["stage"] in passed and (ROOT / row["log"]).exists()
        ),
        "selected_p1_delta_in_p4": value,
    }, indent=2), encoding="utf-8")
    print(f"DONE all stages; selected P1→P4 delta_p={value:.12f}", flush=True)


if __name__ == "__main__":
    main()
