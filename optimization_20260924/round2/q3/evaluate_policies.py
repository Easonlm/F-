"""Compare new risk objectives for Problem 3 using fixed scenario splits.

Run from repository root: python optimization_20260924/round2/q3/evaluate_policies.py
The original P3 model, fit, and optimizers are read only.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize, minimize_scalar

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "problem3_v2"))
from methods import Model, COST_MODELS, capped_solve, d_from_budget, total_cost  # noqa: E402

HERE = Path(__file__).resolve().parent
HERE.mkdir(parents=True, exist_ok=True)
T = ROOT / "problem3_v2" / "outputs" / "tables"
PARAMETER_NAMES = ["E", "A", "alpha", "B", "beta", "G", "kappa", "eta_N"]


def cvar(x: np.ndarray, frac: float = 0.1) -> float:
    return float(np.mean(np.sort(x)[-max(1, math.ceil(frac * len(x))):]))


def model_from_row(row: pd.Series) -> Model:
    return Model(*[float(row[k] if isinstance(row, pd.Series) else getattr(row, k))
                   for k in PARAMETER_NAMES])


def common_d(budget: float, n: float, q: float, d_hi: float) -> float:
    return min(d_hi, min(d_from_budget(budget, n, q, q0, ctx, kind)
                         for kind in COST_MODELS for q0 in (0.2, 0.8) for ctx in (2048, 32768)))


def scenario_arrays(rows: pd.DataFrame, par: pd.DataFrame):
    lut = par.set_index("replicate")
    params = np.array([[float(lut.loc[int(r.bootstrap_replicate), k]) for k in PARAMETER_NAMES]
                       for r in rows.itertuples(index=False)])
    oracle = rows.optimum_loss.to_numpy(float)
    return params, oracle


def losses(params: np.ndarray, n: float, d: float, q: float) -> np.ndarray:
    e, a, alpha, b, beta, g, kappa, eta = params.T
    return e + a * n ** (-alpha) + b * d ** (-beta) + g * (1 - q) ** kappa * n ** (-eta)


def optimize_policy(name: str, budget: float, nb: tuple[float, float], db: tuple[float, float],
                    params: np.ndarray, oracle: np.ndarray) -> tuple[float, float, float]:
    bounds = [tuple(np.log(nb)), (0.8, 1.0)]

    def objective(x):
        n, q = float(np.exp(x[0])), float(x[1])
        d = common_d(budget, n, q, db[1])
        if d < db[0]:
            return 1e6 + 1e3 * (db[0] - d)
        regret = losses(params, n, d, q) - oracle
        if name == "mean_regret":
            return float(np.mean(regret))
        if name == "cvar90_regret":
            return cvar(regret)
        raise ValueError(name)

    de = differential_evolution(objective, bounds, seed=20260924, popsize=14,
                                maxiter=130, tol=1e-10, polish=False)
    candidates = [de]
    for x0 in (de.x, [np.log(np.sqrt(nb[0] * nb[1])), 0.8],
               [np.log(np.sqrt(nb[0] * nb[1])), 1.0]):
        candidates.append(minimize(objective, x0, method="SLSQP", bounds=bounds,
                                   options={"ftol": 1e-12, "maxiter": 500}))
    for q in (0.8, 1.0):
        r = minimize_scalar(lambda logn: objective([logn, q]), bounds=bounds[0],
                            method="bounded", options={"xatol": 1e-11})
        candidates.append(r)
        r.x = np.array([r.x, q])
    best = min(candidates, key=lambda r: float(r.fun))
    n, q = float(np.exp(best.x[0])), float(best.x[1])
    return n, common_d(budget, n, q, db[1]), q


def policy_scores(rows: pd.DataFrame, policy: dict, split: str,
                  nb: tuple[float, float], db: tuple[float, float]) -> list[dict]:
    out = []
    n, d, q = policy["N_B"], policy["D_B"], policy["Q_B"]
    for r in rows.itertuples(index=False):
        mm = model_from_row(r)
        cost = total_cost(n, d, q, r.Q0, r.L_ctx, r.cost_model)[0]
        out.append(dict(split=split, budget=r.budget, policy=policy["policy"],
                        bootstrap_replicate=int(r.bootstrap_replicate),
                        cost_model=r.cost_model, Q0=r.Q0, L_ctx=r.L_ctx,
                        loss=mm.loss(n, d, q), oracle_loss=r.optimum_loss,
                        regret=mm.loss(n, d, q) - r.optimum_loss,
                        feasible=bool(q + 1e-10 >= r.Q0 and cost <= r.budget * (1 + 1e-8)
                                      and nb[0] - 1e-8 <= n <= nb[1] + 1e-8
                                      and db[0] - 1e-8 <= d <= db[1] + 1e-8),
                        cost_ratio=cost / r.budget))
    return out


def main():
    par = pd.read_csv(T / "joint_bootstrap_parameters.csv")
    train_oracle = pd.read_csv(T / "robust_scenario_optima.csv")
    baseline = pd.read_csv(T / "robust_optimization.csv")
    b1 = pd.read_csv(ROOT / "real_attachments/B_scaling_laws/pythia_training_log_existing.csv")
    nb = (float(b1.N_params_B.min()), float(b1.N_params_B.max()) * 3)
    db = (float(b1.D_tokens_B.min()), float(b1.D_tokens_B.max()) * 3)
    train_ids = set(train_oracle.bootstrap_replicate.astype(int))
    available = par[~par.replicate.isin(train_ids)].sort_values("replicate")
    rng = np.random.default_rng(20260924)
    heldout_ids = sorted(rng.choice(available.replicate.to_numpy(int), 12, replace=False).tolist())
    heldout_rows = []
    for budget in (1e19, 1e22, 1e24):
        for rid in heldout_ids:
            mm = model_from_row(par.set_index("replicate").loc[rid])
            for kind in COST_MODELS:
                for q0 in (0.2, 0.8):
                    for ctx in (2048, 32768):
                        opt = capped_solve(mm, budget, q0, ctx, kind, nb, db, global_search=False)
                        heldout_rows.append(dict(budget=budget, bootstrap_replicate=rid,
                                                 cost_model=kind, Q0=q0, L_ctx=ctx,
                                                 optimum_loss=opt["loss"]))
    heldout = pd.DataFrame(heldout_rows)
    heldout.to_csv(HERE / "heldout_scenario_oracles.csv", index=False, float_format="%.12g")
    detail = []
    policy_rows = []
    for budget in (1e19, 1e22, 1e24):
        train = train_oracle[train_oracle.budget == budget].copy()
        test = heldout[heldout.budget == budget].copy()
        params, oracle = scenario_arrays(train, par)
        model_columns = par[["replicate", *PARAMETER_NAMES]]
        train = train.merge(model_columns, left_on="bootstrap_replicate", right_on="replicate", validate="many_to_one")
        test = test.merge(model_columns, left_on="bootstrap_replicate", right_on="replicate", validate="many_to_one")
        policies = []
        for name in ("mean_regret", "cvar90_regret"):
            n, d, q = optimize_policy(name, budget, nb, db, params, oracle)
            policies.append(dict(policy=name, budget=budget, N_B=n, D_B=d, Q_B=q))
        z = baseline[(baseline.budget == budget) & baseline.policy.isin(
            ["robust_minimax", "nominal_moderate_repaired"])].copy()
        for r in z.itertuples(index=False):
            policies.append(dict(policy=r.policy, budget=budget, N_B=r.N_B, D_B=r.D_B, Q_B=r.Q_B))
        for p in policies:
            policy_rows.append(p)
            detail.extend(policy_scores(train, p, "train", nb, db))
            detail.extend(policy_scores(test, p, "heldout", nb, db))
    scores = pd.DataFrame(detail)
    assert not train_ids.intersection(heldout_ids)
    assert len(heldout) == 12 * 12 * 3
    assert bool(scores.feasible.all())
    assert float(scores.regret.min()) > -1e-7
    assert all(len(z) == 144 for _, z in scores[scores.split == "heldout"].groupby(["budget", "policy"]))
    scores.to_csv(HERE / "scenario_policy_scores.csv", index=False, float_format="%.12g")
    pd.DataFrame(policy_rows).to_csv(HERE / "policies.csv", index=False, float_format="%.12g")
    summary = []
    for (split, budget, policy), z in scores.groupby(["split", "budget", "policy"]):
        rr = z.regret.to_numpy(float)
        summary.append(dict(split=split, budget=budget, policy=policy,
                            scenario_count=len(z), replicate_count=z.bootstrap_replicate.nunique(),
                            mean_regret=float(np.mean(rr)), q90_regret=float(np.quantile(rr, .9)),
                            cvar90_regret=cvar(rr), max_regret=float(np.max(rr)),
                            mean_loss=float(z.loss.mean()),
                            feasible_fraction=float(z.feasible.mean()),
                            worst_cost_ratio=float(z.cost_ratio.max())))
    summary = pd.DataFrame(summary).sort_values(["budget", "split", "policy"])
    summary.to_csv(HERE / "policy_summary.csv", index=False, float_format="%.12g")
    (HERE / "run_metadata.json").write_text(json.dumps({"train_parameter_replicates": sorted(train_ids),
        "heldout_parameter_replicates": heldout_ids, "seed": 20260924,
        "important_limit": "heldout bootstrap fits reuse the same B1/B6 source data; this is parameter stability, not independent external validation"}, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
