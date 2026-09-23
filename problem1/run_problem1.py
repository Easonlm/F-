"""Reproduce every stage in a fixed order."""
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
SCRIPTS=["quality_pipeline.py","data_audit.py","evaluate_quality.py","sensitivity.py",
         "mixture_pipeline.py","effect_bootstrap.py","mixture_support.py","quality_mixture_link.py","build_report.py"]
for name in SCRIPTS:
    print(f"Running {name}",flush=True)
    subprocess.run([sys.executable,str(HERE/name)],check=True,cwd=HERE.parent)
