"""Re-run the entire legacy dependency chain inside this isolated copy."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = [
    ("problem1_base", "problem1/run_problem1.py"),
    ("problem1_mixture", "problem1/problem1_v2.py"),
    ("problem2_base", "problem2/run_problem2.py"),
    ("problem2_final", "problem2_v2/run_v2.py"),
    ("problem3_base", "problem3/run_problem3.py"),
    ("problem3_final", "problem3_v2/run_v2.py"),
    ("problem4_final", "problem4/run_problem4.py"),
]
ARTIFACTS = [
    "problem1/outputs/models/mixture_v2.joblib",
    "problem2_v2/outputs/models/recommended_model.joblib",
    "problem3_v2/outputs/tables/robust_optimization.csv",
    "problem4/outputs/tables/final_project_summary.json",
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # This is an isolated copy. A prior run's V1 hash gate must be recomputed
    # against the freshly regenerated V1 outputs, never against the source tree.
    prior = ROOT / "problem2_v2/outputs/tables/v1_sha256.json"
    if prior.exists():
        prior.unlink()
    records = []
    for name, script in STAGES:
        print(f"START {name}", flush=True)
        started = time.time()
        log = ROOT / f"{name}.log"
        with log.open("w", encoding="utf-8") as f:
            result = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT,
                                    env=env, stdout=f, stderr=subprocess.STDOUT)
        row = dict(stage=name, script=script, exit_code=result.returncode,
                   seconds=round(time.time() - started, 2), log=log.name)
        records.append(row)
        (ROOT / "rerun_status.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"END {name} exit={result.returncode} seconds={row['seconds']}", flush=True)
        if result.returncode:
            raise SystemExit(f"Stopped at {name}; see {log}")
    provenance = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "stages": records,
        "artifacts": {p: sha(ROOT / p) for p in ARTIFACTS},
    }
    (ROOT / "rerun_manifest.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print("COMPLETE all seven dependency stages", flush=True)


if __name__ == "__main__":
    main()
