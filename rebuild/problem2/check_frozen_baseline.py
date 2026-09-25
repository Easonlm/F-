"""Read-only SHA audit of the frozen Problem 2 directories."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / "model_baseline_manifest.json").read_text(encoding="utf-8"))
    entries = {name: info for name, info in manifest["files"].items()
               if name.startswith("problem2/") or name.startswith("problem2_v2/")}
    changed, missing = [], []
    for relative, info in entries.items():
        path = ROOT / relative
        if not path.is_file():
            missing.append(relative)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
            changed.append(relative)
    result = {"checked": len(entries), "unchanged": len(entries) - len(changed) - len(missing),
              "changed": changed, "missing": missing, "all_passed": not changed and not missing,
              "scope": "all manifest files under frozen problem2 and problem2_v2; read-only"}
    (HERE / "frozen_baseline_sha_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_passed"]:
        raise AssertionError("Frozen Problem 2 SHA mismatch")


if __name__ == "__main__":
    main()
