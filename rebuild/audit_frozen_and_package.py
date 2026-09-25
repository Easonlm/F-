"""Read-only SHA audit of the frozen original project and review package."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "rebuild" / "final_review" / "verification"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    package_only = "--package-only" in sys.argv[1:]
    frozen = ({} if package_only else json.loads((ROOT / "model_baseline_manifest.json").read_text(
        encoding="utf-8"))["files"])
    mismatches = []
    for relative, recorded in frozen.items():
        path = ROOT / relative
        if not path.is_file():
            mismatches.append(dict(path=relative, issue="missing"))
        elif sha(path) != recorded["sha256"]:
            mismatches.append(dict(path=relative, issue="sha256_changed"))
    package = pd.read_csv(ROOT / "rebuild/final_review/artifact_manifest.csv")
    package_mismatch = []
    for row in package.itertuples(index=False):
        target = ROOT / "rebuild/final_review" / row.packaged_as
        if not target.is_file() or sha(target) != row.sha256:
            package_mismatch.append(row.packaged_as)
    result = {
        "frozen_files_checked": len(frozen),
        "frozen_mismatches": mismatches,
        "frozen_all_passed": not mismatches,
        "packaged_artifacts_checked": len(package),
        "package_mismatches": package_mismatch,
        "package_all_passed": not package_mismatch,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ("package_sha_audit.json" if package_only else "frozen_and_package_sha_audit.json")).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, list)}, indent=2))
    if mismatches or package_mismatch:
        raise SystemExit("SHA audit failed; see frozen_and_package_sha_audit.json")


if __name__ == "__main__":
    main()
