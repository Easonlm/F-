"""Create the predeclared P4 module tournament from independent rerun outputs."""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def read(name):
    return pd.read_csv(HERE / name)


def one(df, col, val):
    row = df[df[col].eq(val)]
    assert len(row) == 1, (col, val, len(row))
    return row.iloc[0]


def main():
    ability = read("ability_index_tournament.csv")
    historical = read("historical_scorecard.csv")
    bridge1 = read("bridge_scorecard.csv")
    bridge2 = read("scorecard.csv")
    bridge3 = read("multitask_bridge_tournament.csv")
    forecast = read("short_term_scorecard.csv")
    frontier = read("quantile_frontier_scorecard.csv")
    rows = []

    def add(module, candidate, dataset, split, metric, value, sample, decision, note):
        rows.append(dict(problem="problem4", module=module, candidate=candidate,
                         dataset=dataset, split=split, primary_metric=metric,
                         value=float(value), sample=sample, decision=decision, note=note))

    for _, r in ability.iterrows():
        add("ability", r.candidate, "C1_C8", "related_task_consistency", "spearman",
            r.c8_spearman, f"n={r.n}, families={r.families}",
            "retain_champion" if r.candidate == "six_benchmark_mean" else "sensitivity_only",
            "C8 derives from related benchmark tasks; no independent criterion for index upgrade")
    for _, r in historical.iterrows():
        add("scale_technology", r.kind, "C1_C4_strict", "organization_group_cv", "rmse",
            r.rmse, f"n={r.n}, families={r.families}",
            "predictive_reference" if r.kind == "compute_only" else "scenario_only" if r.kind == "structural" else "reject",
            "Time contribution not independently identified; structural time coefficient CI crosses zero")
    for _, r in bridge1.iterrows():
        add("bridge", r.kind, "C6", "family_group_cv", "rmse", r.oof_rmse,
            f"n={r.n}, families={r.families}",
            "retain_champion" if r.kind == "logistic" else "reject",
            "Candidate gain bootstrap CI crosses zero or extrapolation is unstable")
    for _, r in bridge2.iterrows():
        if r.kind == "original_logistic":
            continue
        add("bridge", r.kind, "C6", "family_group_cv", "rmse", r.oof_rmse,
            f"n={r.n}, families={r.families}", "diagnostic_only" if r.kind in ("loss_plus_scale", "scale_only_diagnostic") else "reject",
            "Loss+scale gain dominated by scale with degenerate Loss slope; no robust family gain")
    r = one(bridge3, "candidate", "partial_pool_multitask")
    add("bridge", r.candidate, "C6_six_tasks", "family_group_cv", "rmse",
        r.family_cv_rmse, f"n={r.n}, families={r.families}", "reject",
        "Seven parameters; family bootstrap gain 95% interval includes zero")
    for _, r in frontier.iterrows():
        add("frontier", f"{r.candidate}_q{r['quantile']}", "strict_W1", "rolling_origin",
            "mae", r.mae, f"targets={r.targets}, min_target_n={r.min_target_n}",
            "sensitivity_only", "Only two out-of-time targets, one with n>=5; no stable selection")
    for _, r in forecast.iterrows():
        if r.horizon_quarters != 1:
            continue
        add("forecast", r.model, "strict_W1", "rolling_origin_3m", "mae",
            r.mae, f"targets={r.targets}, min_target_n={r.min_target_n}",
            "short_horizon_reference" if r.model == "persistence" else "scenario_only",
            "No direct 12/24m target; 3m has only two targets, one with n>=5")
    pd.DataFrame(rows).to_csv(HERE / "problem4_model_tournament.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
