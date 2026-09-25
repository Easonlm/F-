"""Unsupervised A1 Q sensitivity; never selects a new Q without human labels."""
from __future__ import annotations

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
TABLES = ROOT / "problem1" / "outputs" / "tables"


def main():
    scores = pd.read_parquet(TABLES / "sample_quality_scores.parquet")
    z = pd.read_parquet(TABLES / "quality_standardized.parquet")
    weights = pd.read_csv(TABLES / "quality_weights.csv")
    keys = ["source", "domain", "id"]
    if not scores[keys].equals(z[keys]):
        raise ValueError("Q score and standardized indicators are not row aligned")
    mask = scores.source.eq("A1").to_numpy()
    fields = weights.field.tolist()
    x = z.loc[mask, fields].to_numpy(float)
    a1 = scores.loc[mask, keys + ["Q_baseline"]].copy().reset_index(drop=True)
    if not np.allclose(x @ weights.baseline.to_numpy(float), a1.Q_baseline, atol=1e-9):
        raise ValueError("frozen Q baseline no longer reconstructs")
    base = a1.Q_baseline.to_numpy(float)
    scaler = StandardScaler()
    xs = scaler.fit_transform(x)
    pca = PCA(n_components=1, random_state=20260923)
    pca_score = pca.fit_transform(xs)[:, 0]
    fa = FactorAnalysis(n_components=1, random_state=20260923, max_iter=200)
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter("always", ConvergenceWarning)
        fa_score = fa.fit_transform(xs)[:, 0]
    fa_converged = not any(issubclass(m.category, ConvergenceWarning) for m in messages)
    if spearmanr(pca_score, base).statistic < 0:
        pca_score = -pca_score
    if spearmanr(fa_score, base).statistic < 0:
        fa_score = -fa_score
    # Entropy weighting: column mass on A1, without access to external labels.
    xp = np.clip(x, 1e-12, None)
    prop = xp / xp.sum(axis=0, keepdims=True)
    ent = -np.sum(prop * np.log(prop), axis=0) / np.log(len(xp))
    dispersion = np.maximum(1-ent, 0)
    ent_weights = dispersion / dispersion.sum()
    entropy_score = x @ ent_weights
    # Rank aggregate is robust to monotone recalibration of an indicator.
    xr = pd.DataFrame(x, columns=fields).rank(method="average", pct=True).to_numpy(float)
    group_ranks = []
    for group in weights.group.unique():
        group_ranks.append(xr[:, weights.group.eq(group).to_numpy()].mean(axis=1))
    rank_score = np.mean(group_ranks, axis=0)
    candidate = {"Q_baseline": base, "PCA_1": pca_score, "FA_1": fa_score,
                 "entropy_22": entropy_score, "group_equal_rank_aggregate": rank_score}
    frame = pd.DataFrame(dict(domain=a1.domain, **candidate))
    domain = frame.groupby("domain", sort=True).mean(numeric_only=True)
    base_order = set(domain.Q_baseline.nsmallest(3).index)
    rows = []
    for name, values in candidate.items():
        rows.append(dict(method=name, n_A1=len(a1), n_domains=len(domain),
                         sample_spearman_vs_Q=float(spearmanr(base, values).statistic),
                         domain_spearman_vs_Q=float(spearmanr(domain.Q_baseline,
                                                               domain[name]).statistic),
                         lower_three_domain_overlap=len(base_order.intersection(
                             set(domain[name].nsmallest(3).index))),
                         numerical_converged=(fa_converged if name == "FA_1" else True),
                         has_external_truth=False, eligible_to_replace_Q=False,
                         note=("FA did not converge; descriptive only" if name == "FA_1" and not fa_converged
                               else "Sensitivity only: no completed manual labels or independent quality outcome")))
    pd.DataFrame(rows).to_csv(OUT / "q_sensitivity_audit.csv", index=False, encoding="utf-8-sig")
    domain.reset_index().to_csv(OUT / "q_domain_scores_sensitivity.csv", index=False,
                                encoding="utf-8-sig")
    pd.DataFrame(dict(field=fields, entropy_weight=ent_weights,
                      pca_loading=pca.components_[0], fa_loading=fa.components_[0]
                      )).to_csv(OUT / "q_unsupervised_weights.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
