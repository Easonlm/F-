"""Freeze the exact pre-challenge working tree without modifying source models."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FROZEN_DIRS = ["problem1", "problem2", "problem2_v2", "problem3", "problem3_v2", "problem4"]
MAIN = {
    "problem1": {
        "model": "four-group equal-weight Q_A; quadratic ALR Ridge mixture V2",
        "parameters": "problem1/outputs/tables/v2_selection.json",
        "artifacts": ["problem1/outputs/models/mixture_v2.joblib", "problem1/outputs/tables/domain_quality_scores.csv"],
        "outputs": ["problem1/outputs/tables/v2_vs_v1_metrics.csv", "problem1/outputs/tables/v2_nested_cv_summary.csv"],
    },
    "problem2": {
        "model": "Model C: classic N-D + Q-by-N + lambda_p Delta_p (lambda_p=1 scenario)",
        "parameters": "problem2_v2/outputs/tables/selection.json",
        "artifacts": ["problem2_v2/outputs/models/recommended_model.joblib"],
        "outputs": ["problem2_v2/outputs/tables/nd_model_scorecard.csv", "problem2_v2/outputs/tables/quality_model_scorecard.csv"],
    },
    "problem3": {
        "model": "V2 E0/E1/E2 constrained optimizer and minimax-regret policy",
        "parameters": "problem3_v2/outputs/tables/independent_verification.json",
        "artifacts": ["problem3_v2/outputs/tables/extrapolation_levels.csv", "problem3_v2/outputs/tables/robust_optimization.csv"],
        "outputs": ["problem3_v2/outputs/tables/verification_results.json"],
    },
    "problem4": {
        "model": "six-benchmark ability; quarterly p95; structural decomposition; bounded logistic bridge; scenario forecast",
        "parameters": "problem4/outputs/tables/final_project_summary.json",
        "artifacts": ["problem4/outputs/tables/loss_benchmark_bridge_model.json", "problem4/outputs/tables/frontier_forecast_12m_24m.csv"],
        "outputs": ["problem4/outputs/tables/historical_model_comparison.csv", "problem4/outputs/tables/forecast_backtest.csv"],
    },
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str | None:
    p = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else None


def rows(path: str) -> list[dict[str, str]]:
    with (ROOT / path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    target = ROOT / "model_baseline_manifest.json"
    if target.exists():
        raise SystemExit("Baseline is frozen already; refusing to overwrite")
    files = {}
    for name in FROZEN_DIRS:
        for p in sorted((ROOT / name).rglob("*")):
            if p.is_file():
                rel = p.relative_to(ROOT).as_posix()
                files[rel] = {"sha256": sha(p), "bytes": p.stat().st_size}
    manifest = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git("rev-parse", "HEAD"),
        "git_status_porcelain": git("status", "--porcelain=v1", "--untracked-files=all"),
        "frozen_directories": FROZEN_DIRS,
        "files": files,
        "current_final_models": MAIN,
        "baseline_source": "Existing working tree, including tracked modifications and untracked V3 artifacts; historical values are retrospective, not new blind tests.",
    }
    for info in MAIN.values():
        p = ROOT / info["parameters"]
        info["parameter_sha256"] = files[p.relative_to(ROOT).as_posix()]["sha256"]
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    scores: list[dict[str, str]] = []

    def add(problem: str, model: str, dataset: str, split: str, metric: str,
            value: str, evidence_level: str, source: str, notes: str = "") -> None:
        scores.append(dict(problem=problem, model=model, dataset=dataset, split=split,
                           metric=metric, value=value, evidence_level=evidence_level,
                           notes=f"source={source}; {notes}".rstrip("; ")))

    src = "problem1/outputs/tables/v2_vs_v1_metrics.csv"
    for r in rows(src):
        if r["version"] == "v2":
            for metric in ("rmse", "mae", "spearman_mean", "bias"):
                add("problem1", "quadratic_ALR_Ridge_V2", r["dataset"], "retrospective_test",
                    metric, r[metric], "observed_or_estimated_per_dataset", src)
    src = "problem1/outputs/tables/v2_nested_cv_summary.csv"
    for r in rows(src):
        if r["version"] == "v2":
            add("problem1", "quadratic_ALR_Ridge_V2", "A4_A5", "nested_group_cv",
                "rmse", r["nested_oof_rmse"], "internal_cv", src)
    src = "problem2_v2/outputs/tables/nd_model_scorecard.csv"
    for r in rows(src):
        if r["model"] == "classic":
            for col, dataset, split in [("B1_group_cv_rmse", "B1", "group_cv"),
                                        ("B2_raw_rmse", "B2", "external_raw"),
                                        ("B4_raw_rmse", "B4", "external_raw"),
                                        ("B5_raw_rmse", "B5", "external_raw")]:
                add("problem2", "classic_ND", dataset, split, "rmse", r[col],
                    "semi_synthetic_or_external_source", src)
            add("problem2", "classic_ND", "B1", "training", "bic", r["bic"], "training", src)
    src = "problem2_v2/outputs/tables/quality_model_scorecard.csv"
    for r in rows(src):
        if r["model"] in ("quality_model", "Q_by_N", "q_n"):
            chosen = r
            break
    else:
        chosen = min(rows(src), key=lambda r: float(r["B6_leave_N_out_rmse"]))
    for col, dataset, split in [("B6_leave_N_out_rmse", "B6", "leave_N"),
                                ("B7_new_rmse", "B7", "new_point"),
                                ("B8_new_rmse", "B8", "mechanism_diagnostic")]:
        add("problem2", "Q_by_N", dataset, split, "rmse", chosen[col],
            "semi_synthetic", src, "B8 is diagnostic only")
    src = "problem3_v2/outputs/tables/robust_optimization.csv"
    for r in rows(src):
        if r["policy"] == "robust_minimax":
            add("problem3", "robust_minimax_E1", r["budget"], "36_scenarios",
                "max_regret", r["max_regret"], "conditional_decision", src)
            add("problem3", "robust_minimax_E1", r["budget"], "36_scenarios",
                "worst_cost_ratio", r["worst_cost_ratio"], "conditional_decision", src)
    src = "problem4/outputs/tables/historical_model_comparison.csv"
    for r in rows(src):
        if r["model"] in ("structural", "compute_only", "time_only"):
            add("problem4", r["model"], "C1_C4", "family_group_cv", "rmse",
                r["group_cv_rmse"], "small_observed_sample", src)
    src = "problem4/outputs/tables/short_term_baseline_summary.csv"
    for r in rows(src):
        if r["sample_rule"] == "all" and r["horizon_months"] == "3":
            add("problem4", r["model"], "strict_W1_frontier", "rolling_origin_3m",
                "mae", r["mae"], "two_backtest_targets", src,
                f"targets={r['targets']}; min_target_n={r['min_target_n']}")
    summary = json.loads((ROOT / "problem4/outputs/tables/final_project_summary.json").read_text(encoding="utf-8"))
    add("problem4", "bounded_logistic_bridge", "C6", "family_group_cv", "rmse",
        str(summary["problem4"]["bridge_cv_rmse"]), "mixed_comparability", "problem4/outputs/tables/final_project_summary.json")
    with (ROOT / "baseline_scorecard.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["problem", "model", "dataset", "split", "metric",
                                                "value", "evidence_level", "notes"])
        writer.writeheader()
        writer.writerows(scores)
    print(f"Frozen {len(files)} files; wrote {len(scores)} scorecard rows")


if __name__ == "__main__":
    main()
