"""Small downstream interface for the selected Problem 2 law.

N and D are in billions. Q is the B6/B7 quality score. delta_p must come from
the selected Problem 1 mixture model, centered at its reference recipe.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def predict_loss(N_B, D_B, Q_B, delta_p=0.0, lambda_p=1.0, selection_path=None):
    path = Path(selection_path) if selection_path is not None else HERE / "selection.json"
    selected = json.loads(path.read_text(encoding="utf-8"))
    if selected["ND_winner"] != "classic" or selected["Q_winner"] != "Q_by_N":
        raise ValueError("This interface expects the verified classic + Q_by_N champion")
    N_B, D_B, Q_B, delta_p = np.broadcast_arrays(
        np.asarray(N_B, float), np.asarray(D_B, float), np.asarray(Q_B, float), np.asarray(delta_p, float))
    if not np.isfinite(N_B).all() or not np.isfinite(D_B).all() or not np.isfinite(Q_B).all():
        raise ValueError("Inputs must be finite")
    if (N_B <= 0).any() or (D_B <= 0).any() or ((Q_B < 0) | (Q_B > 1)).any():
        raise ValueError("Expected N_B,D_B > 0 and Q_B in [0,1]")
    e, a, alpha, b, beta = selected["ND_parameters"]
    g, kappa, eta_n = selected["Q_parameters"]
    result = e + a * N_B ** (-alpha) + b * D_B ** (-beta)
    result += g * (1 - Q_B) ** kappa * N_B ** (-eta_n)
    result += float(lambda_p) * delta_p
    return float(result) if result.ndim == 0 else result

