"""Create an isolated copy for a complete legacy dependency-chain rerun."""

from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "rebuild" / "full_chain"
SOURCES = ["real_attachments", "problem1", "problem2", "problem2_v2", "problem3", "problem3_v2", "problem4"]


def ignore(directory, names):
    return {n for n in names if n in {"node_modules", "__pycache__", ".git"}}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for name in SOURCES:
        source = ROOT / name
        target = DEST / name
        if target.exists():
            raise SystemExit(f"Refusing to overwrite existing isolated copy: {target}")
        shutil.copytree(source, target, ignore=ignore)
        print(f"Copied {name}", flush=True)


if __name__ == "__main__":
    main()
