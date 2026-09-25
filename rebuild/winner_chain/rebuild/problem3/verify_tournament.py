"""Check mathematical and provenance invariants of the P3 tournament."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "problem3"))
from core import total_cost  # noqa: E402


def main():
    metadata = json.loads((HERE / "run_metadata.json").read_text(encoding="utf-8"))
    oracles = pd.read_csv(HERE / "scenario_oracles.csv")
    policies = pd.read_csv(HERE / "policies.csv")
    scores = pd.read_csv(HERE / "scenario_policy_scores.csv")
    summary = pd.read_csv(HERE / "problem3_decision_tournament.csv")
    audit = pd.read_csv(HERE / "decision_audit.csv")
    raw = pd.read_csv(HERE / "nominal_cap_profile.csv")
    b1 = pd.read_csv(ROOT / "real_attachments/B_scaling_laws/pythia_training_log_existing.csv")
    key = ["budget", "split", "bootstrap_replicate", "cost_model", "Q0", "L_ctx"]
    wide = oracles.pivot(index=key, columns="cap", values="oracle_loss")
    monotone_error = max(float((wide[b] - wide[a]).max()) for a, b in
                         (("1x", "2x"), ("2x", "3x"), ("3x", "5x"), ("5x", "free")))
    cost_error = max(abs(total_cost(r.N_B, r.D_B, r.Q_B, r.Q0, r.L_ctx,
                                    r.cost_model)[0] / r.budget - r.cost_ratio)
                     for r in scores.itertuples(index=False))
    grouped = scores.groupby(["budget", "cap", "policy", "split"]).size()
    split_sizes_ok = all((size == (36 if index[3] == "train" else 144))
                         for index, size in grouped.items())
    raw_free = raw[raw.cap == "free"]
    model_path = ROOT / metadata["model_artifact"]
    provenance_ok = (hashlib.sha256(model_path.read_bytes()).hexdigest() ==
                     metadata["model_artifact_sha256"])
    # In the selected-chain copy, P3 outputs are intentionally regenerated.
    # Verify the actual frozen original workspace rather than this adapter copy.
    frozen_root = ROOT.parents[1] if (ROOT.parents[1] / "model_baseline_manifest.json").exists() else ROOT
    frozen_files = json.loads((frozen_root / "model_baseline_manifest.json").read_text(
        encoding="utf-8"))["files"]
    frozen_p3 = {name: info for name, info in frozen_files.items()
                 if name.startswith(("problem3/", "problem3_v2/"))}
    frozen_mismatch = [name for name, info in frozen_p3.items()
                       if not (frozen_root / name).is_file() or
                       hashlib.sha256((frozen_root / name).read_bytes()).hexdigest() != info["sha256"]]
    checks = {
        "scenario_oracle_rows": len(oracles),
        "policy_rows": len(policies),
        "scenario_score_rows": len(scores),
        "summary_rows": len(summary),
        "train_heldout_parameter_overlap": len(set(metadata["training_parameter_replicates"]) &
                                              set(metadata["heldout_parameter_replicates"])),
        "scenario_counts_pass": bool(len(oracles) == 2700 and len(policies) == 75 and
                                     len(scores) == 13500 and len(summary) == 150 and split_sizes_ok),
        "scenario_oracle_cap_monotonicity_max_error": monotone_error,
        "oracle_cap_monotonicity_pass": monotone_error <= 1e-6,
        "all_policy_scenarios_feasible": bool(scores.feasible.all()),
        "max_cost_ratio": float(scores.cost_ratio.max()),
        "cost_recompute_max_ratio_error": float(cost_error),
        "cost_recompute_pass": cost_error < 1e-10,
        "minimum_within_cap_regret": float(scores.within_cap_regret.min()),
        "regret_nonnegative_with_tolerance": bool(scores.within_cap_regret.min() >= -1e-7),
        "max_local_direction_improvement": float(policies.local_direction_improvement.max()),
        "local_direction_check_pass": bool(policies.local_direction_improvement.max() < 1e-7),
        "raw_free_kkt_max_rel_error": float(raw_free.raw_free_kkt_rel_error.max()),
        "raw_free_kkt_pass": bool((raw_free.raw_free_kkt_rel_error < 2e-3).all() and
                                  raw_free.raw_free_kkt_complementarity_ok.all()),
        "model_artifact_hash_pass": provenance_ok,
        "frozen_p3_files_checked": len(frozen_p3),
        "frozen_p3_sha_mismatches": frozen_mismatch,
        "frozen_p3_sha_pass": not frozen_mismatch,
        "no_challenger_passes_upgrade_gate": not audit.upgrade_eligible.any(),
        "max_policy_N_over_B1": float((policies.N_B / float(b1.N_params_B.max())).max()),
        "max_policy_D_over_B1": float((policies.D_B / float(b1.D_tokens_B.max())).max()),
        "interpretation": "Directions are local checks for nonsmooth robust policies, not complete KKT/global certificates",
    }
    checks["all_pass"] = bool(all(checks[name] for name in (
        "scenario_counts_pass", "oracle_cap_monotonicity_pass", "all_policy_scenarios_feasible",
        "cost_recompute_pass", "regret_nonnegative_with_tolerance",
        "local_direction_check_pass", "raw_free_kkt_pass", "model_artifact_hash_pass",
        "frozen_p3_sha_pass", "no_challenger_passes_upgrade_gate")))
    (HERE / "verification_results.json").write_text(json.dumps(checks, ensure_ascii=False,
                                                                 indent=2), encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    if not checks["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
