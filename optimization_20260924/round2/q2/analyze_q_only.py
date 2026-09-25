"""Paired N-D cell bootstrap for the explicit low-label B8 q_only interface."""
from __future__ import annotations

import numpy as np
import pandas as pd

from run_experiments import HERE, SEED, rmse, write


def effect(directory, protocol, baseline, candidate="source_q_only", n_boot=2000):
    predictions = pd.read_csv(HERE / directory / "b8_labeled_source_predictions.csv")
    left = predictions[predictions.model == baseline].set_index("experiment_id")
    right = predictions[predictions.model == candidate].set_index("experiment_id")
    ids = left.index.intersection(right.index)
    left = left.loc[ids]
    right = right.loc[ids]
    if not np.allclose(left.observed, right.observed):
        raise AssertionError("Prediction rows are not paired")
    cell = left.N_params_B.astype(str).to_numpy() + "|" + left.D_tokens_B.astype(str).to_numpy()
    unique = np.unique(cell)
    positions = {key: np.flatnonzero(cell == key) for key in unique}
    y = left.observed.to_numpy(float)
    p0 = left.predicted.to_numpy(float)
    p1 = right.predicted.to_numpy(float)
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(n_boot):
        chosen = rng.choice(unique, len(unique), replace=True)
        selected = np.concatenate([positions[key] for key in chosen])
        draws.append(rmse(y[selected], p1[selected]) - rmse(y[selected], p0[selected]))
    interval = np.quantile(draws, [.025, .5, .975])
    return dict(protocol=protocol, baseline=baseline, candidate=candidate, n_rows=len(y), n_ND_cells=len(unique),
                baseline_rmse=rmse(y, p0), candidate_rmse=rmse(y, p1),
                delta_new_minus_baseline=rmse(y, p1) - rmse(y, p0),
                delta_lo95=interval[0], delta_median=interval[1], delta_hi95=interval[2],
                bootstrap_share_improving=float(np.mean(np.asarray(draws) < 0)))


def main():
    rows = []
    for directory, protocol in [
        ("integrated_q_only_100", "one_ND_cell_per_N_100_1380"),
        ("integrated_q_only_q_disjoint", "one_ND_cell_per_N_unseen_Q_51_777"),
    ]:
        for baseline in ["zero_shot_Model_C", "source_intercept_control"]:
            rows.append(effect(directory, protocol, baseline))
    write(rows, "q_only_paired_effects.csv")


if __name__ == "__main__":
    main()
