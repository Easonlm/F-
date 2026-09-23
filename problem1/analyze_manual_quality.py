"""Analyze completed blind ratings; blank templates yield no invented statistics."""
from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, spearmanr
from sklearn.metrics import cohen_kappa_score

HERE = Path(__file__).resolve().parent
TABLES = HERE / "outputs" / "tables"
FIGURES = HERE / "outputs" / "figures"
DEFAULT_TEMPLATE = HERE / "outputs" / "v2_review" / "人工质量盲评_140条.xlsx"
COLUMNS = ["sample_id", "text", "rater_id", "overall_quality", "fluency",
           "information_education", "cleanliness", "ad_or_spam", "notes"]
RESULT_COLUMNS = ["analysis", "comparison", "n_samples", "n_raters", "estimate",
                  "ci_low", "ci_high", "p_value", "method"]
SEED = 20260923


def _read_xlsx_stdlib(path: Path) -> pd.DataFrame:
    """Read the first worksheet without requiring a third-party Excel reader."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as zf:
        shared = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("m:si", ns):
                shared.append("".join(t.text or "" for t in si.findall(".//m:t", ns)))
        sheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        sparse = []
        for row in sheet.findall(".//m:sheetData/m:row", ns):
            cells = {}
            for cell in row.findall("m:c", ns):
                ref = cell.get("r", "")
                letters = re.match(r"[A-Z]+", ref)
                if not letters:
                    continue
                col = 0
                for char in letters.group():
                    col = col * 26 + ord(char) - 64
                cell_type = cell.get("t")
                val = cell.find("m:v", ns)
                if cell_type == "inlineStr":
                    value = "".join(t.text or "" for t in cell.findall(".//m:t", ns))
                elif val is None:
                    value = None
                elif cell_type == "s":
                    value = shared[int(val.text)]
                else:
                    value = val.text
                cells[col - 1] = value
            if cells:
                sparse.append([cells.get(i) for i in range(max(cells) + 1)])
    if not sparse:
        raise ValueError(f"Empty workbook: {path}")
    header = [str(x) if x is not None else "" for x in sparse[0]]
    return pd.DataFrame([r + [None] * (len(header) - len(r)) for r in sparse[1:]], columns=header)


def read_ratings(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        data = pd.read_csv(path, dtype=str, keep_default_na=False)
    elif path.suffix.lower() == ".xlsx":
        data = _read_xlsx_stdlib(path).fillna("")
    else:
        raise ValueError(f"Use .csv or .xlsx: {path}")
    missing = set(COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return data[COLUMNS].copy()


def bootstrap_ci(values: np.ndarray, fn, rng: np.random.Generator, draws: int = 2000):
    values = np.asarray(values)
    if len(values) < 3:
        return np.nan, np.nan
    estimates = []
    for _ in range(draws):
        picked = values[rng.integers(0, len(values), len(values))]
        result = fn(picked)
        if np.isfinite(result):
            estimates.append(result)
    if len(estimates) < 100:
        return np.nan, np.nan
    return float(np.quantile(estimates, .025)), float(np.quantile(estimates, .975))


def bootstrap_difference(a, b, rng, draws=2000):
    estimates = []
    for _ in range(draws):
        aa = a[rng.integers(0, len(a), len(a))]
        bb = b[rng.integers(0, len(b), len(b))]
        estimates.append(float(aa.mean() - bb.mean()))
    return float(np.quantile(estimates, .025)), float(np.quantile(estimates, .975))


def icc_two_way_absolute(matrix: np.ndarray) -> float:
    n, k = matrix.shape
    if n < 3 or k < 2:
        return np.nan
    grand = matrix.mean()
    row_mean = matrix.mean(axis=1)
    col_mean = matrix.mean(axis=0)
    ms_row = k * ((row_mean - grand) ** 2).sum() / (n - 1)
    ms_col = n * ((col_mean - grand) ** 2).sum() / (k - 1)
    residual = matrix - row_mean[:, None] - col_mean[None, :] + grand
    ms_error = (residual ** 2).sum() / ((n - 1) * (k - 1))
    denominator = ms_row + (k - 1) * ms_error + k * (ms_col - ms_error) / n
    return float((ms_row - ms_error) / denominator) if denominator != 0 else np.nan


def metric_row(analysis, comparison, n, raters, estimate, ci, p, method):
    return dict(analysis=analysis, comparison=comparison, n_samples=n, n_raters=raters,
                estimate=estimate, ci_low=ci[0], ci_high=ci[1], p_value=p, method=method)


def run(paths: list[Path]) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    frames = [read_ratings(path) for path in paths]
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    key = pd.DataFrame(json.loads((TABLES / "v2_blind_review_key.json").read_text(encoding="utf-8")))
    item = pd.DataFrame(json.loads((TABLES / "v2_blind_review_items.json").read_text(encoding="utf-8")))
    if len(key) != 140 or len(item) != 140 or set(key.review_id) != set(item.review_id):
        raise ValueError("Blind sample/key should contain the same 140 sample IDs")
    if not data.empty:
        data["sample_id"] = data.sample_id.astype(str).str.strip()
        data["rater_id"] = data.rater_id.astype(str).str.strip()
        if not set(data.sample_id).issubset(set(key.review_id)):
            raise ValueError("Unknown sample_id in ratings")
        expected_text = item.set_index("review_id").text.to_dict()
        for row in data.itertuples():
            if str(row.text).replace("\r\n", "\n") != expected_text[row.sample_id].replace("\r\n", "\n"):
                raise ValueError(f"Text differs from blind sample {row.sample_id}")
        for col in ["overall_quality", "fluency", "information_education", "cleanliness"]:
            data[col] = pd.to_numeric(data[col].astype(str).str.strip(), errors="coerce")
            invalid = data[col].notna() & (~data[col].between(1, 5) | (data[col] % 1 != 0))
            if invalid.any():
                raise ValueError(f"{col} must be an integer from 1 to 5")
        ad = data.ad_or_spam.astype(str).str.strip()
        if (~ad.isin(["", "0", "1"])).any():
            raise ValueError("ad_or_spam must be 0, 1, or blank")
        rated = data.loc[data.overall_quality.notna()].copy()
    else:
        rated = data
    output = TABLES / "manual_validation_results.csv"
    if rated.empty:
        pd.DataFrame(columns=RESULT_COLUMNS).to_csv(output, index=False, encoding="utf-8-sig")
        print("No human overall ratings entered. Wrote a header-only results file; no statistics or figures.")
        return
    if rated.rater_id.eq("").any():
        raise ValueError("Each scored row needs a rater_id")
    if rated.duplicated(["sample_id", "rater_id"]).any():
        raise ValueError("Duplicate (sample_id, rater_id); use one row per rater and sample")
    joined = rated.merge(key[["review_id", "Q_baseline", "conflict_any"]],
                         left_on="sample_id", right_on="review_id", validate="many_to_one")
    sample = joined.groupby("sample_id", as_index=False).agg(
        human_mean=("overall_quality", "mean"),
        Q=("Q_baseline", "first"),
        conflict=("conflict_any", "first"),
        n_raters=("rater_id", "nunique"),
    )
    if len(sample) < 10:
        raise ValueError("At least 10 distinct scored samples are needed for analysis")
    # The 140-item sample is deliberately stratified; groups are within this sample.
    all_q = key[["review_id", "Q_baseline"]].copy()
    all_q["Q_group"] = pd.qcut(all_q.Q_baseline.rank(method="first"), 3, labels=["low", "middle", "high"])
    sample = sample.merge(all_q[["review_id", "Q_group"]], left_on="sample_id", right_on="review_id",
                          validate="one_to_one")
    rng = np.random.default_rng(SEED)
    rows = []
    pairs = sample[["Q", "human_mean"]].to_numpy(float)
    corr = spearmanr(pairs[:, 0], pairs[:, 1])
    rho_ci = bootstrap_ci(pairs, lambda x: spearmanr(x[:, 0], x[:, 1]).statistic, rng)
    rows.append(metric_row("Q_human", "Spearman Q vs mean overall score", len(sample),
                           int(joined.rater_id.nunique()), float(corr.statistic), rho_ci,
                           float(corr.pvalue), "sample-level Spearman; bootstrap resamples sample IDs"))
    groups = {}
    for label in ["low", "middle", "high"]:
        v = sample.loc[sample.Q_group.eq(label), "human_mean"].to_numpy(float)
        groups[label] = v
        if len(v):
            ci = bootstrap_ci(v, np.mean, rng)
            rows.append(metric_row("Q_tertile", f"{label} mean human score", len(v),
                                   int(joined.rater_id.nunique()), float(v.mean()), ci, np.nan,
                                   "Q tertiles fixed on all 140 blind items"))
    if all(len(groups[g]) >= 2 for g in groups):
        test = kruskal(groups["low"], groups["middle"], groups["high"])
        rows.append(metric_row("Q_tertile", "omnibus three-group difference",
                               len(sample), int(joined.rater_id.nunique()),
                               float(test.statistic), (np.nan, np.nan), float(test.pvalue),
                               "Kruskal-Wallis H statistic; exploratory"))
        ci = bootstrap_difference(groups["high"], groups["low"], rng)
        rows.append(metric_row("Q_tertile", "high minus low mean human score",
                               len(groups["high"]) + len(groups["low"]), int(joined.rater_id.nunique()),
                               float(groups["high"].mean() - groups["low"].mean()), ci,
                               float(mannwhitneyu(groups["high"], groups["low"]).pvalue),
                               f"Mann-Whitney high vs low; omnibus Kruskal p={test.pvalue:.6g}"))
    conflict = sample.loc[sample.conflict, "human_mean"].to_numpy(float)
    ordinary = sample.loc[~sample.conflict, "human_mean"].to_numpy(float)
    for label, values in [("conflict", conflict), ("ordinary", ordinary)]:
        if len(values):
            rows.append(metric_row("conflict", f"{label} mean human score", len(values),
                                   int(joined.rater_id.nunique()), float(values.mean()),
                                   bootstrap_ci(values, np.mean, rng), np.nan,
                                   "selected blind sample; descriptive mean"))
    if len(conflict) >= 2 and len(ordinary) >= 2:
        ci = bootstrap_difference(conflict, ordinary, rng)
        rows.append(metric_row("conflict", "conflict minus ordinary mean human score",
                               len(sample), int(joined.rater_id.nunique()),
                               float(conflict.mean() - ordinary.mean()), ci,
                               float(mannwhitneyu(conflict, ordinary).pvalue),
                               f"sample-level Mann-Whitney; n_conflict={len(conflict)}; n_ordinary={len(ordinary)}"))
    matrix = joined.pivot(index="sample_id", columns="rater_id", values="overall_quality").dropna()
    if matrix.shape[1] >= 2 and matrix.shape[0] >= 3:
        array = matrix.to_numpy(float)
        icc = icc_two_way_absolute(array)
        ci = bootstrap_ci(array, icc_two_way_absolute, rng)
        rows.append(metric_row("rater_agreement", "ICC(2,1) absolute agreement",
                               len(matrix), matrix.shape[1], icc, ci, np.nan,
                               "complete common samples; two-way random single-rating ICC"))
        if matrix.shape[1] == 2:
            kappa = float(cohen_kappa_score(array[:, 0].astype(int), array[:, 1].astype(int),
                                            weights="quadratic"))
            kappa_ci = bootstrap_ci(
                array,
                lambda x: cohen_kappa_score(x[:, 0].astype(int), x[:, 1].astype(int),
                                          weights="quadratic"),
                rng,
            )
            rows.append(metric_row("rater_agreement", "quadratic weighted Cohen kappa",
                                   len(matrix), 2, kappa, kappa_ci, np.nan,
                                   "two complete raters; ordinal 1-5 scale"))
    result = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    result.to_csv(output, index=False, encoding="utf-8-sig")
    sample.to_csv(TABLES / "manual_validation_sample_summary.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].scatter(sample.Q, sample.human_mean, alpha=.55, s=18)
    axes[0].set(xlabel="Quality proxy Q", ylabel="Mean human overall score (1–5)",
                title=f"Spearman rho = {corr.statistic:.2f}")
    axes[1].boxplot([groups[g] for g in ["low", "middle", "high"]], tick_labels=["Low", "Middle", "High"])
    axes[1].set(xlabel="Q tertile in 140 blind items", ylabel="Mean human overall score (1–5)")
    fig.tight_layout()
    fig.savefig(FIGURES / "manual_quality_validation.png", dpi=180)
    plt.close(fig)
    print(result.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, nargs="*", default=[DEFAULT_TEMPLATE],
                        help="One or more completed .xlsx or .csv reviewer copies")
    arguments = parser.parse_args()
    run(arguments.ratings)
