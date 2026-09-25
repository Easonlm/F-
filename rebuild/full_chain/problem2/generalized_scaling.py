"""Reusable N-D-Q-p prediction and exact quality/parameter equivalence.

N and D are billions. Q is the supplementary B quality scale, not the
unanchored quality proxy constructed in problem one.
"""
from __future__ import annotations
import numpy as np


def predict_ndq(N, D, Q, parameters, kind="exponential"):
    E, A, alpha, B, beta = [parameters[k] for k in ("E", "A", "alpha", "B", "beta")]
    N, D, Q = np.broadcast_arrays(np.asarray(N, float), np.asarray(D, float), np.asarray(Q, float))
    if np.any(N <= 0) or np.any(D <= 0) or np.any(Q <= 0):
        raise ValueError("N,D,Q must be positive")
    if kind == "exponential":
        data = B * D**(-beta) * np.exp(-beta * parameters["gamma"] * (Q-1))
    elif kind == "power":
        data = B * D**(-beta) * Q**(-beta * parameters["gamma"])
    elif kind == "additive":
        data = B * D**(-beta) + parameters["G"] * np.maximum(1-Q, 0)**parameters["kappa"]
    else:
        raise ValueError(kind)
    return E + A*N**(-alpha) + data


def derivatives(N, D, Q, parameters, kind="exponential"):
    N, D, Q = map(lambda v: np.asarray(v, float), (N, D, Q))
    A, alpha, B, beta = [parameters[k] for k in ("A", "alpha", "B", "beta")]
    dn = -alpha*A*N**(-alpha-1)
    if kind == "exponential":
        data = B*D**(-beta)*np.exp(-beta*parameters["gamma"]*(Q-1))
        return dn, -beta*data/D, -beta*parameters["gamma"]*data
    if kind == "power":
        data = B*D**(-beta)*Q**(-beta*parameters["gamma"])
        return dn, -beta*data/D, -beta*parameters["gamma"]*data/Q
    return dn, -beta*B*D**(-beta-1), -parameters["G"]*parameters["kappa"]*np.maximum(1-Q, 1e-12)**(parameters["kappa"]-1)


def equivalent_parameters(N, D, Q, delta_Q, parameters, kind="exponential"):
    """Exact N required at original Q to match the improvement at Q+delta_Q."""
    if Q + delta_Q > 1 or delta_Q <= 0:
        raise ValueError("Q+delta_Q must be in (0,1]")
    target = float(predict_ndq(N, D, Q+delta_Q, parameters, kind))
    baseline_without_N = float(predict_ndq(N, D, Q, parameters, kind) - parameters["A"]*N**(-parameters["alpha"]))
    remainder = target - baseline_without_N
    if remainder <= 0:
        return np.inf
    return float((parameters["A"]/remainder)**(1/parameters["alpha"]))


def predict_generalized(N, D, Q, p, parameters, mixture_model, reference_p, lambda_p=1, kind="exponential"):
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"problem1"))
    from problem1_v2 import transform
    m = mixture_model
    p = np.asarray(p, float).reshape(1, -1)
    ref = np.asarray(reference_p, float).reshape(1, -1)
    def mean_loss(v):
        return float(m["model"].predict(transform(v, m["family"], m["eps"], m["reference_index"])).mean())
    return float(predict_ndq(N,D,Q,parameters,kind)) + lambda_p*(mean_loss(p)-mean_loss(ref))
