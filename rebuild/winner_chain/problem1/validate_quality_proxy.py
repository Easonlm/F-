"""Additive diagnostics for the existing quality proxy; never refits Q."""
from __future__ import annotations

import json
import lzma
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from quality_pipeline import FIELDS, input_files, scalar

HERE = Path(__file__).resolve().parent
TABLES = HERE / "outputs" / "tables"
FIGURES = HERE / "outputs" / "figures"
SEED = 20260923


def load_a1_scalar() -> pd.DataFrame:
    rows = []
    with lzma.open(input_files()[0][1], "rt", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            rows.append({"id": str(obj["id"]), **{f: scalar(f, obj.get(f)) for f in FIELDS}})
    return pd.DataFrame(rows)


def rho(a, b) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.unique(np.asarray(a)[ok]).size < 2 or np.unique(np.asarray(b)[ok]).size < 2:
        return np.nan
    return float(spearmanr(np.asarray(a)[ok], np.asarray(b)[ok]).statistic)


def internal_consistency() -> pd.DataFrame:
    raw = load_a1_scalar()
    z = pd.read_parquet(TABLES / "quality_standardized.parquet")
    score = pd.read_parquet(TABLES / "sample_quality_scores.parquet")
    z = z.loc[z.source.eq("A1")].reset_index(drop=True)
    score = score.loc[score.source.eq("A1")].reset_index(drop=True)
    if not (len(raw) == len(z) == len(score) == 51230):
        raise ValueError("A1 row counts do not match")
    if not (raw.id.to_numpy() == z.id.astype(str).to_numpy()).all():
        raise ValueError("A1 row order/id mismatch")
    if not (raw.id.to_numpy() == score.id.astype(str).to_numpy()).all():
        raise ValueError("A1 score row order/id mismatch")

    rules = pd.read_csv(TABLES / "quality_indicator_rules.csv").set_index("field")
    weights = pd.read_csv(TABLES / "quality_weights.csv").set_index("field")
    q = score.Q_baseline.to_numpy(float)
    rows = []
    for field in FIELDS:
        rule = rules.loc[field]
        direction = str(rule.raw_direction)
        weight = float(weights.loc[field, "baseline"])
        raw_value = raw[field].to_numpy(float)
        transformed = z[field].to_numpy(float)
        q_without = (q - weight * transformed) / (1 - weight)
        values = np.sign(raw_value) * np.log1p(np.abs(raw_value)) if rule.skew_transform == "signed_log1p" else raw_value
        calib = np.array(json.loads(rule.calibration_values), float)
        if direction == "适中最好；经验梯形效用":
            lo, left, right, hi = calib
            expected = np.minimum(
                np.clip((values - lo) / max(left - lo, 1e-12), 0, 1),
                np.clip((hi - values) / max(hi - right, 1e-12), 0, 1),
            )
            left_mask = np.isfinite(values) & (values > lo) & (values < left)
            right_mask = np.isfinite(values) & (values > right) & (values < hi)
            left_rho = rho(values[left_mask], transformed[left_mask])
            right_rho = rho(values[right_mask], transformed[right_mask])
            direction_ok = bool(
                np.nanmax(np.abs(expected - transformed)) < 1e-10
                and np.isfinite(left_rho) and left_rho > 0.99
                and np.isfinite(right_rho) and right_rho < -0.99
            )
            relation_match = np.nan
            shape_match = direction_ok
            expected_direction = "left increases; middle plateau; right decreases"
        else:
            lo, hi = calib
            expected = np.clip((values - lo) / max(hi - lo, 1e-12), 0, 1)
            if direction == "越低越好":
                expected = 1 - expected
            left_mask = right_mask = np.zeros(len(values), bool)
            left_rho = right_rho = np.nan
            sign = 1 if direction == "越高越好" else -1
            observed = rho(raw_value, q)
            direction_ok = bool(np.isfinite(observed) and sign * observed > 0)
            relation_match = direction_ok
            shape_match = np.nan
            expected_direction = "positive" if sign == 1 else "negative"
        finite = np.isfinite(expected)
        error = np.abs(expected[finite] - transformed[finite])
        rows.append({
            "field": field, "raw_type": rule.raw_type, "raw_direction": direction,
            "expected_relation_to_Q": expected_direction,
            "n_A1": int(len(values)), "n_raw_nonmissing": int(np.isfinite(raw_value).sum()),
            "weight_in_Q": weight,
            "spearman_raw_vs_Q": rho(raw_value, q),
            "spearman_raw_vs_Q_without_self": rho(raw_value, q_without),
            "spearman_raw_vs_transformed": rho(raw_value, transformed),
            "transform_max_abs_error": float(error.max()) if len(error) else np.nan,
            "transform_rule_check": bool(len(error) and error.max() < 1e-10),
            "left_ramp_n": int(left_mask.sum()),
            "left_ramp_spearman_raw_vs_score": left_rho,
            "right_ramp_n": int(right_mask.sum()),
            "right_ramp_spearman_raw_vs_score": right_rho,
            "plateau_fraction_A1": float(np.mean(np.isfinite(values) & (values >= calib[1]) & (values <= calib[2])))
            if len(calib) == 4 else np.nan,
            "relation_to_Q_matches_expected": relation_match,
            "hump_shape_check": shape_match,
            "direction_or_shape_check": direction_ok,
            "interpretation": "implementation/shape audit only; part-whole correlation is not external validity",
        })
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "quality_internal_consistency.csv", index=False, encoding="utf-8-sig")
    domain_rows = []
    domains = z.domain.astype(str).to_numpy()
    for field in FIELDS:
        raw_value = raw[field].to_numpy(float)
        transformed = z[field].to_numpy(float)
        weight = float(weights.loc[field, "baseline"])
        q_without = (q - weight * transformed) / (1 - weight)
        for domain in np.unique(domains):
            mask = domains == domain
            domain_rows.append({
                "field": field, "domain": domain, "n": int(mask.sum()),
                "spearman_raw_vs_Q": rho(raw_value[mask], q[mask]),
                "spearman_raw_vs_Q_without_self": rho(raw_value[mask], q_without[mask]),
            })
    pd.DataFrame(domain_rows).to_csv(
        TABLES / "quality_internal_consistency_by_domain.csv", index=False, encoding="utf-8-sig"
    )
    return out


def bootstrap_summary(q: np.ndarray, conflict: np.ndarray, rng: np.random.Generator, draws: int = 200) -> dict:
    n = len(q)
    means = np.empty(draws)
    rates = np.empty(draws)
    for i in range(draws):
        idx = rng.integers(0, n, size=n)
        means[i] = q[idx].mean()
        rates[i] = conflict[idx].mean()
    return {
        "n": n, "Q_mean": float(q.mean()),
        "Q_ci_low": float(np.quantile(means, 0.025)),
        "Q_ci_high": float(np.quantile(means, 0.975)),
        "conflict_rate": float(conflict.mean()),
        "conflict_ci_low": float(np.quantile(rates, 0.025)),
        "conflict_ci_high": float(np.quantile(rates, 0.975)),
        "bootstrap_draws": draws,
    }


def independent_extensions() -> pd.DataFrame:
    scores = pd.read_parquet(TABLES / "sample_quality_scores.parquet")
    rng = np.random.default_rng(SEED)
    rows = []
    for domain, source in [("arxiv", "A2"), ("github", "A3")]:
        a1 = scores.loc[scores.source.eq("A1") & scores.domain.eq(domain)].copy()
        ext = scores.loc[scores.source.eq(source) & scores.domain.eq(domain)].copy()
        ids = set(a1.id.astype(str))
        overlap = ext.id.astype(str).isin(ids)
        unique = ext.loc[~overlap]
        for label, frame in [("A1", a1), ("extension_full_existing", ext), ("extension_nonoverlap_new", unique)]:
            result = bootstrap_summary(frame.Q_baseline.to_numpy(float), frame.conflict_any.to_numpy(bool), rng)
            rows.append({"domain": domain, "subset": label, "source": "A1" if label == "A1" else source,
                         "overlap_with_A1_n": int(overlap.sum()) if label != "A1" else 0, **result})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "quality_extension_dedup.csv", index=False, encoding="utf-8-sig")
    return out


def figures(internal: pd.DataFrame, extension: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    mono = internal.loc[internal.raw_direction.ne("适中最好；经验梯形效用")].copy()
    mono = mono.sort_values("spearman_raw_vs_Q")
    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(mono))
    ax.barh(y - 0.18, mono.spearman_raw_vs_Q, height=0.35, label="Raw vs Q")
    ax.barh(y + 0.18, mono.spearman_raw_vs_Q_without_self, height=0.35, label="Raw vs Q excluding own term")
    ax.set_yticks(y, mono.field, fontsize=8)
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_xlabel("Spearman correlation on A1")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "quality_internal_consistency.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    colors = {"A1": "#294a6d", "extension_nonoverlap_new": "#d48b39"}
    for i, domain in enumerate(["arxiv", "github"]):
        part = extension.loc[extension.domain.eq(domain) & extension.subset.isin(colors)].set_index("subset")
        for j, label in enumerate(colors):
            row = part.loc[label]
            axes[0].errorbar(i + (j - .5) * .18, row.Q_mean,
                             yerr=[[row.Q_mean - row.Q_ci_low], [row.Q_ci_high - row.Q_mean]],
                             fmt="o", color=colors[label], capsize=3,
                             label=("A1" if j == 0 else "Extension without A1 IDs") if i == 0 else None)
            axes[1].errorbar(i + (j - .5) * .18, row.conflict_rate * 100,
                             yerr=[[(row.conflict_rate - row.conflict_ci_low) * 100],
                                   [(row.conflict_ci_high - row.conflict_rate) * 100]],
                             fmt="o", color=colors[label], capsize=3)
    axes[0].set(ylabel="Mean Q", title="Quality proxy")
    axes[1].set(ylabel="Conflict rate (%)", title="Predefined conflicts")
    for ax in axes:
        ax.set_xticks([0, 1], ["arxiv", "github"])
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "quality_extension_dedup.png", dpi=180)
    plt.close(fig)


def run() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    internal = internal_consistency()
    extension = independent_extensions()
    figures(internal, extension)
    print(internal[["field", "raw_direction", "spearman_raw_vs_Q",
                    "spearman_raw_vs_Q_without_self", "direction_or_shape_check"]].to_string(index=False))
    print(extension[["domain", "subset", "n", "overlap_with_A1_n",
                     "Q_mean", "conflict_rate"]].to_string(index=False))


if __name__ == "__main__":
    run()
