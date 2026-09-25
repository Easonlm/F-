"""Recompute the P3 decision tournament without modifying frozen baselines.

Run from repository root: python rebuild/problem3/run_decision_tournament.py
An upstream P2 replacement can be supplied with --model-artifact and
--bootstrap-parameters; the latter must contain the eight Model C columns.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize, minimize_scalar

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "problem3_v2"))
from methods import (  # noqa: E402
    COST_MODELS, Model, capped_solve, d_from_budget, kkt_diagnostic, total_cost
)

PARAMS = ("E", "A", "alpha", "B", "beta", "G", "kappa", "eta_N")
BUDGETS = (1e19, 1e22, 1e24)
KINDS = ("nominal_shared", "minimax_regret", "balanced_mean_regret",
         "cvar90_regret", "worst_case_loss")
SEED = 20260924


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_model(row) -> Model:
    return Model(*[float(row[k]) for k in PARAMS])


def cv90(values: np.ndarray) -> float:
    return float(np.mean(np.sort(values)[-max(1, math.ceil(.1 * len(values))):]))


def losses(parameters: np.ndarray, n: float, d: float, q: float) -> np.ndarray:
    e, a, alpha, b, beta, g, kappa, eta = parameters.T
    return e + a * n ** (-alpha) + b * d ** (-beta) + g * (1 - q) ** kappa * n ** (-eta)


def common_d(budget: float, n: float, q: float, d_hi: float) -> float:
    return min(d_hi, min(d_from_budget(budget, n, q, q0, ctx, cost)
                         for cost in COST_MODELS for q0 in (.2, .8)
                         for ctx in (2048, 32768)))


def cap_bounds(b1: pd.DataFrame) -> dict[str, tuple[tuple[float, float], tuple[float, float]]]:
    n_lo, n_hi = float(b1.N_params_B.min()), float(b1.N_params_B.max())
    d_lo, d_hi = float(b1.D_tokens_B.min()), float(b1.D_tokens_B.max())
    return {**{f"{factor}x": ((n_lo, n_hi * factor), (d_lo, d_hi * factor))
               for factor in (1, 2, 3, 5)},
            "free": ((n_lo, 1e6), (d_lo, 1e7))}


def optimize_policy(kind: str, budget: float, nb: tuple[float, float],
                    db: tuple[float, float], parameters: np.ndarray,
                    oracle: np.ndarray, central: Model):
    bounds = [tuple(np.log(nb)), (.8, 1.)]

    def objective(x):
        n, q = float(np.exp(x[0])), float(x[1])
        d = common_d(budget, n, q, db[1])
        if d < db[0]:
            return 1e6 + 1e3 * (db[0] - d)
        if kind == "nominal_shared":
            return central.loss(n, d, q)
        value = losses(parameters, n, d, q)
        if kind == "worst_case_loss":
            return float(np.max(value))
        regret = value - oracle
        if kind == "minimax_regret":
            return float(np.max(regret))
        if kind == "balanced_mean_regret":
            return float(np.mean(regret))
        if kind == "cvar90_regret":
            return cv90(regret)
        raise ValueError(kind)

    de = differential_evolution(objective, bounds, seed=SEED, popsize=12,
                                maxiter=110, tol=1e-9, polish=False)
    candidates = [(float(de.fun), np.array(de.x, float))]
    for initial in (de.x, [np.log(np.sqrt(nb[0] * nb[1])), .8],
                    [np.log(np.sqrt(nb[0] * nb[1])), 1.]):
        result = minimize(objective, initial, method="SLSQP", bounds=bounds,
                          options={"ftol": 1e-12, "maxiter": 400})
        candidates.append((float(result.fun), result.x))
    for q in (.8, 1.):
        result = minimize_scalar(lambda logn: objective((logn, q)),
                                 bounds=bounds[0], method="bounded",
                                 options={"xatol": 1e-11})
        candidates.append((float(result.fun), np.array([result.x, q])))
        for n in (nb[0], nb[1]):
            x = np.array([np.log(n), q])
            candidates.append((float(objective(x)), x))
    value, x = min(candidates, key=lambda pair: pair[0])
    n, q = float(np.exp(x[0])), float(x[1])
    d = common_d(budget, n, q, db[1])
    if d < db[0] - 1e-8 or value >= 1e5:
        raise RuntimeError(f"Policy infeasible: {kind}, {budget:g}, {nb}, {db}")
    # Directional local certificate; nondifferentiable maxima/caps do not
    # admit the ordinary smooth KKT ratio used for free nominal solutions.
    h = 1e-4
    neighbors = []
    for a, b in ((-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)):
        y = np.array([np.clip(x[0] + h * a, *bounds[0]),
                      np.clip(x[1] + h * b, .8, 1.)])
        neighbors.append(float(objective(y)))
    return dict(N_B=n, D_B=d, Q_B=q, training_objective=value,
                local_direction_improvement=max(0., value - min(neighbors)),
                global_candidate_gap=max(0., value - float(de.fun)))


def make_scenarios(par: pd.DataFrame) -> tuple[pd.DataFrame, list[int], list[int]]:
    ranked = par.sort_values("G").reset_index(drop=True)
    training = [int(ranked.iloc[int((len(ranked) - 1) * fraction)].replicate)
                for fraction in (.05, .5, .95)]
    available = par[~par.replicate.isin(training)].sort_values("replicate")
    heldout = sorted(np.random.default_rng(SEED).choice(
        available.replicate.to_numpy(int), 12, replace=False).tolist())
    records = []
    for split, ids in (("train", training), ("heldout", heldout)):
        for rid in ids:
            row = par.set_index("replicate").loc[rid]
            for cost in COST_MODELS:
                for q0 in (.2, .8):
                    for ctx in (2048, 32768):
                        records.append(dict(split=split, bootstrap_replicate=rid,
                                            cost_model=cost, Q0=q0, L_ctx=ctx,
                                            **{k: float(row[k]) for k in PARAMS}))
    return pd.DataFrame(records), training, heldout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-artifact", type=Path,
                        default=ROOT / "problem2_v2/outputs/models/recommended_model.joblib")
    parser.add_argument("--bootstrap-parameters", type=Path,
                        default=ROOT / "problem3_v2/outputs/tables/joint_bootstrap_parameters.csv")
    args = parser.parse_args()
    artifact = joblib.load(args.model_artifact)
    if artifact.get("ND_model") != "classic" or artifact.get("Q_model") != "quality_model":
        raise ValueError("This P3 Model C runner requires an explicit adapter for the upgraded P2 model")
    central = Model(*map(float, [*artifact["ND_parameters"], *artifact["Q_parameters"]]))
    par = pd.read_csv(args.bootstrap_parameters)
    missing = set(("replicate", *PARAMS)) - set(par.columns)
    if missing or not par.replicate.is_unique or len(par) < 20:
        raise ValueError(f"Invalid joint bootstrap table; missing={missing}")
    b1_path = ROOT / "real_attachments/B_scaling_laws/pythia_training_log_existing.csv"
    b1 = pd.read_csv(b1_path)
    caps = cap_bounds(b1)
    scenarios, train_ids, heldout_ids = make_scenarios(par)
    parameter_lookup = par.set_index("replicate")
    oracle_rows = []
    for budget in BUDGETS:
        print(f"Recomputing scenario oracles for {budget:.0e} ...", flush=True)
        for scenario in scenarios.itertuples(index=False):
            mm = make_model(parameter_lookup.loc[int(scenario.bootstrap_replicate)])
            for cap, (nb, db) in caps.items():
                result = capped_solve(mm, budget, float(scenario.Q0), int(scenario.L_ctx),
                                      str(scenario.cost_model), nb, db, global_search=False)
                oracle_rows.append(dict(budget=budget, split=scenario.split,
                                        bootstrap_replicate=int(scenario.bootstrap_replicate),
                                        cost_model=scenario.cost_model, Q0=scenario.Q0,
                                        L_ctx=scenario.L_ctx, cap=cap,
                                        oracle_loss=result["loss"], oracle_N_B=result["N_B"],
                                        oracle_D_B=result["D_B"], oracle_Q_B=result["Q_B"],
                                        oracle_budget_ratio=result["budget_ratio"]))
    oracle_table = pd.DataFrame(oracle_rows)
    oracle_table.to_csv(HERE / "scenario_oracles.csv", index=False, float_format="%.12g")
    policies = []
    raw_nominal = []
    for budget in BUDGETS:
        for cap, (nb, db) in caps.items():
            print(f"Optimizing policies {budget:.0e} {cap} ...", flush=True)
            current = oracle_table[(oracle_table.budget == budget) &
                                   (oracle_table.cap == cap) &
                                   (oracle_table.split == "train")].copy()
            current = current.merge(scenarios[scenarios.split == "train"],
                                    on=["split", "bootstrap_replicate", "cost_model", "Q0", "L_ctx"],
                                    validate="one_to_one")
            pars = current[list(PARAMS)].to_numpy(float)
            oracle = current.oracle_loss.to_numpy(float)
            for kind in KINDS:
                result = optimize_policy(kind, budget, nb, db, pars, oracle, central)
                policies.append(dict(budget=budget, cap=cap, policy=kind, **result))
            raw = capped_solve(central, budget, .6, 4096, "exponential", nb, db,
                               global_search=True)
            structural = [(kind, q0, ctx) for kind in COST_MODELS
                          for q0 in (.2, .8) for ctx in (2048, 32768)]
            costs = [total_cost(raw["N_B"], raw["D_B"], raw["Q_B"], q0,
                                ctx, kind)[0] / budget
                     for kind, q0, ctx in structural]
            diagnostic = dict(budget=budget, cap=cap, N_B=raw["N_B"], D_B=raw["D_B"],
                              Q_B=raw["Q_B"], nominal_loss=raw["loss"],
                              nominal_budget_ratio=raw["budget_ratio"],
                              raw_common_feasible_fraction=float(np.mean([
                                  raw["Q_B"] >= q0 - 1e-10 and ratio <= 1 + 1e-8
                                  for (_, q0, _), ratio in zip(structural, costs)])),
                              raw_worst_cost_ratio=max(costs),
                              N_over_B1_max=raw["N_B"] / float(b1.N_params_B.max()),
                              D_over_B1_max=raw["D_B"] / float(b1.D_tokens_B.max()),
                              N_cap_active=bool(raw["N_cap_active"]),
                              D_cap_active=bool(raw["D_cap_active"]))
            if cap == "free":
                kkt = kkt_diagnostic(central, raw, .6, 4096, "exponential")
                diagnostic.update(raw_free_kkt_rel_error=kkt["KKT_rel_error"],
                                  raw_free_kkt_complementarity_ok=kkt["Q_complementarity_ok"])
            raw_nominal.append(diagnostic)
    policy_table = pd.DataFrame(policies)
    policy_table.to_csv(HERE / "policies.csv", index=False, float_format="%.12g")
    pd.DataFrame(raw_nominal).to_csv(HERE / "nominal_cap_profile.csv", index=False,
                                     float_format="%.12g")
    # Oracle coordinates and model parameters are recomputed above. Every
    # candidate is now scored against *the same* scenario and free oracle.
    scores = []
    for policy in policy_table.itertuples(index=False):
        sub = oracle_table[(oracle_table.budget == policy.budget) &
                           (oracle_table.cap == policy.cap)].copy()
        free = oracle_table[(oracle_table.budget == policy.budget) &
                            (oracle_table.cap == "free")][[
                                "split", "bootstrap_replicate", "cost_model", "Q0", "L_ctx",
                                "oracle_loss"]].rename(columns={"oracle_loss": "free_oracle_loss"})
        sub = sub.merge(free, on=["split", "bootstrap_replicate", "cost_model", "Q0", "L_ctx"],
                        validate="one_to_one")
        sub = sub.merge(scenarios, on=["split", "bootstrap_replicate", "cost_model", "Q0", "L_ctx"],
                        validate="one_to_one")
        values = losses(sub[list(PARAMS)].to_numpy(float), policy.N_B, policy.D_B, policy.Q_B)
        for j, row in enumerate(sub.itertuples(index=False)):
            cost_ratio = total_cost(policy.N_B, policy.D_B, policy.Q_B,
                                    row.Q0, row.L_ctx, row.cost_model)[0] / policy.budget
            feasible = (policy.Q_B >= row.Q0 - 1e-10 and cost_ratio <= 1 + 1e-8 and
                        caps[policy.cap][0][0] - 1e-8 <= policy.N_B <= caps[policy.cap][0][1] + 1e-8 and
                        caps[policy.cap][1][0] - 1e-8 <= policy.D_B <= caps[policy.cap][1][1] + 1e-8)
            scores.append(dict(budget=policy.budget, cap=policy.cap, policy=policy.policy,
                               split=row.split, bootstrap_replicate=row.bootstrap_replicate,
                               cost_model=row.cost_model, Q0=row.Q0, L_ctx=row.L_ctx,
                               N_B=policy.N_B, D_B=policy.D_B, Q_B=policy.Q_B,
                               loss=values[j], cap_oracle_loss=row.oracle_loss,
                               free_oracle_loss=row.free_oracle_loss,
                               within_cap_regret=values[j] - row.oracle_loss,
                               free_regret=values[j] - row.free_oracle_loss,
                               feasible=feasible, cost_ratio=cost_ratio))
    score_table = pd.DataFrame(scores)
    if not score_table.feasible.all() or score_table.within_cap_regret.min() < -1e-5:
        raise RuntimeError("A candidate is infeasible or beats a scenario oracle beyond tolerance")
    score_table.to_csv(HERE / "scenario_policy_scores.csv", index=False, float_format="%.12g")
    summary = []
    for (budget, cap, policy, split), z in score_table.groupby(
            ["budget", "cap", "policy", "split"], sort=False):
        rr, fr = z.within_cap_regret.to_numpy(float), z.free_regret.to_numpy(float)
        p = policy_table[(policy_table.budget == budget) & (policy_table.cap == cap) &
                         (policy_table.policy == policy)].iloc[0]
        summary.append(dict(budget=budget, cap=cap, policy=policy, split=split,
                            scenario_count=len(z), parameter_replicates=z.bootstrap_replicate.nunique(),
                            feasible_fraction=float(z.feasible.mean()),
                            mean_loss=float(z.loss.mean()),
                            mean_within_cap_regret=float(rr.mean()),
                            cvar90_within_cap_regret=cv90(rr),
                            max_within_cap_regret=float(rr.max()),
                            mean_free_regret=float(fr.mean()),
                            cvar90_free_regret=cv90(fr),
                            max_free_regret=float(fr.max()),
                            nominal_loss=central.loss(p.N_B, p.D_B, p.Q_B),
                            N_B=p.N_B, D_B=p.D_B, Q_B=p.Q_B,
                            N_over_B1_max=p.N_B / float(b1.N_params_B.max()),
                            D_over_B1_max=p.D_B / float(b1.D_tokens_B.max()),
                            worst_budget_ratio=float(z.cost_ratio.max()),
                            local_direction_improvement=p.local_direction_improvement,
                            KKT_kind="directional_nonsmooth_proxy"))
    summary_table = pd.DataFrame(summary).sort_values(["budget", "cap", "split", "policy"])
    summary_table.to_csv(HERE / "problem3_decision_tournament.csv", index=False,
                         float_format="%.12g")
    bootstrap = []
    rng = np.random.default_rng(SEED + 1)
    for budget in BUDGETS:
        base = score_table[(score_table.budget == budget) & (score_table.cap == "3x") &
                           (score_table.policy == "minimax_regret") &
                           (score_table.split == "heldout")]
        ids = np.array(sorted(base.bootstrap_replicate.unique()))
        base_by_id = base.groupby("bootstrap_replicate").free_regret.mean().reindex(ids).to_numpy()
        for (cap, policy), z in score_table[(score_table.budget == budget) &
                                             (score_table.split == "heldout")].groupby(["cap", "policy"]):
            candidate_by_id = z.groupby("bootstrap_replicate").free_regret.mean().reindex(ids).to_numpy()
            difference = base_by_id - candidate_by_id
            draws = rng.choice(len(ids), (2000, len(ids)), replace=True)
            boot = difference[draws].mean(axis=1)
            bootstrap.append(dict(budget=budget, cap=cap, policy=policy,
                                  mean_improvement_vs_3x_minimax=float(difference.mean()),
                                  improvement_ci05=float(np.quantile(boot, .05)),
                                  improvement_ci95=float(np.quantile(boot, .95)),
                                  bootstrap_fraction_improved=float(np.mean(boot > 0)),
                                  cluster_count=len(ids),
                                  note="same-source parameter bootstrap; not external validation"))
    bootstrap_table = pd.DataFrame(bootstrap)
    bootstrap_table.to_csv(HERE / "paired_bootstrap_vs_champion.csv", index=False,
                           float_format="%.12g")
    audit = []
    heldout_summary = summary_table[summary_table.split == "heldout"]
    for budget in BUDGETS:
        baseline = heldout_summary[(heldout_summary.budget == budget) &
                                   (heldout_summary.cap == "3x") &
                                   (heldout_summary.policy == "minimax_regret")].iloc[0]
        baseline_extrapolation = max(baseline.N_over_B1_max, baseline.D_over_B1_max)
        for row in heldout_summary[heldout_summary.budget == budget].itertuples(index=False):
            p = bootstrap_table[(bootstrap_table.budget == budget) &
                                (bootstrap_table.cap == row.cap) &
                                (bootstrap_table.policy == row.policy)].iloc[0]
            gain = float(baseline.mean_free_regret - row.mean_free_regret)
            relative_gain = gain / max(float(baseline.mean_free_regret), 1e-12)
            material = gain >= .005 and relative_gain >= .05
            extrapolation_non_worse = max(row.N_over_B1_max, row.D_over_B1_max) <= baseline_extrapolation + 1e-8
            is_champion = row.cap == "3x" and row.policy == "minimax_regret"
            eligible = (not is_champion and row.feasible_fraction == 1 and material and
                        p.bootstrap_fraction_improved > .9 and extrapolation_non_worse)
            audit.append(dict(budget=budget, cap=row.cap, policy=row.policy,
                              mean_free_regret_improvement=gain,
                              relative_improvement=relative_gain,
                              bootstrap_fraction_improved=p.bootstrap_fraction_improved,
                              material_improvement=material,
                              extrapolation_non_worse=extrapolation_non_worse,
                              fully_feasible=row.feasible_fraction == 1,
                              upgrade_eligible=eligible,
                              decision="CHAMPION" if is_champion else
                                       "ACCEPT" if eligible else "REJECT"))
    audit_table = pd.DataFrame(audit)
    audit_table.to_csv(HERE / "decision_audit.csv", index=False, float_format="%.12g")
    metadata = dict(model_artifact=str(args.model_artifact.relative_to(ROOT)) if args.model_artifact.is_relative_to(ROOT) else str(args.model_artifact),
                    model_artifact_sha256=digest(args.model_artifact),
                    bootstrap_parameters_sha256=digest(args.bootstrap_parameters),
                    b1_sha256=digest(b1_path), training_parameter_replicates=train_ids,
                    heldout_parameter_replicates=heldout_ids, seed=SEED,
                    budgets=list(BUDGETS), caps=list(caps), policies=list(KINDS),
                    scenario_probabilities_identified=False,
                    heldout_is_external_data=False,
                    champion="3x minimax_regret retained conditionally",
                    any_challenger_upgrade_eligible=bool(audit_table.upgrade_eligible.any()),
                    note="All scenario oracles and policies reoptimized in this run; shared feasibility across structural scenarios")
    (HERE / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False,
                                                          indent=2), encoding="utf-8")
    print(summary_table[summary_table.split == "heldout"][
        ["budget", "cap", "policy", "mean_within_cap_regret", "mean_free_regret",
         "max_free_regret", "N_over_B1_max", "D_over_B1_max"]].to_string(index=False))


if __name__ == "__main__":
    main()
