"""Copy the clean full rerun into a second isolated selected-model pipeline."""

from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "rebuild" / "full_chain"
DEST = ROOT / "rebuild" / "winner_chain"


def ignore(directory, names):
    return {n for n in names if n in {"__pycache__", ".git"} or n.endswith(".log")}


if __name__ == "__main__":
    if DEST.exists():
        raise SystemExit("Refusing to overwrite winner_chain")
    shutil.copytree(SOURCE, DEST, ignore=ignore)
    print("Copied independent raw data and all seven rerun stages")
