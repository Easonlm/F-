"""Read-only independent checks of the problem-one deliverables.

Runs from raw A files plus saved outputs; does not refit or modify any model.
"""
from __future__ import annotations

import json
import lzma
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
T = HERE / "outputs" / "tables"
A = ROOT / "real_attachments" / "A_data_value"


def raw_quality_counts():
    paths = {
        "A1": A / "slimpajama_quality_signal_sample.jsonl.xz",
        "A2": next((A / "slimpajama_quality_extended").glob("arxiv_*.xz")),
        "A3": next((A / "slimpajama_quality_extended").glob("github_*.xz")),
    }
    result = {}
    for code, path in paths.items():
        with lzma.open(path, "rt", encoding="utf-8") as fh:
            result[code] = sum(1 for _ in fh)
    return result


def mixture_pair(prefix, scale):
    folder = A / "regmix_tables"
    p = pd.read_csv(folder / f"{prefix}_mixture_{scale}.csv")
    y = pd.read_csv(folder / f"{prefix}_pile_loss_{scale}.csv")
    if p["index"].duplicated().any() or y["index"].duplicated().any():
        raise AssertionError("duplicate mixture/loss index")
    merged = p.merge(y, on="index", validate="one_to_one")
    if len(merged) != len(p) or len(merged) != len(y):
        raise AssertionError("mixture/loss rows do not match")
    pcols = [c for c in p if c.startswith("train_the_pile_")]
    ycols = [c for c in y if c.startswith("metric/the_pile_")]
    x = merged[pcols].to_numpy(float)
    x /= x.sum(axis=1, keepdims=True)
    return merged["index"].to_numpy(), x, merged[ycols].to_numpy(float), ycols


def main():
    checks = []
    def record(name, ok, detail):
        checks.append((name, bool(ok), str(detail)))

    # Re-read every compressed source instead of trusting a saved file-count table.
    counts = raw_quality_counts()
    saved = pd.read_parquet(T / "sample_quality_scores.parquet")
    observed = saved.groupby("source").size().to_dict()
    record("A1–A3 全量读取", counts == observed and sum(counts.values()) == 272505,
           f"原始 {counts}；评分表 {observed}")

    weights = pd.read_csv(T / "quality_weights.csv")
    standardized = pd.read_parquet(T / "quality_standardized.parquet")
    fields = weights.field.tolist()
    reconstruction = standardized[fields].to_numpy(float) @ weights.baseline.to_numpy(float)
    q = saved.Q_baseline.to_numpy(float)
    q_err = float(np.max(np.abs(reconstruction - q)))
    record("22 指标及 Q 重构", len(fields) == 22 and np.isclose(weights.baseline.sum(), 1) and q_err < 1e-10,
           f"字段 {len(fields)}；权重和 {weights.baseline.sum():.12f}；最大重构误差 {q_err:.2e}")
    record("Q 的取值范围", np.isfinite(q).all() and (q >= 0).all() and (q <= 1).all(),
           f"最小 {q.min():.6f}；最大 {q.max():.6f}")

    domain = pd.read_csv(T / "domain_quality_scores.csv")
    baseline = domain[domain.method.eq("Q_baseline")].set_index(["source", "domain"])
    by_group = saved.groupby(["source", "domain"]).Q_baseline.mean()
    d_err = max(abs(float(by_group.loc[key]) - float(baseline.loc[key, "mean"])) for key in by_group.index)
    record("域级 Q 重构", d_err < 1e-10, f"最大均值差 {d_err:.2e}")

    # Count overlap directly by ID, because A1 is nested in the two extensions.
    a1 = saved[saved.source.eq("A1")]
    a2 = saved[saved.source.eq("A2")]
    a3 = saved[saved.source.eq("A3")]
    overlap2 = len(set(a1[a1.domain.eq("arxiv")].id) & set(a2.id))
    overlap3 = len(set(a1[a1.domain.eq("github")].id) & set(a3.id))
    record("A1 与扩展集重叠", overlap2 == 1419 and overlap3 == 10000,
           f"arxiv {overlap2}；github {overlap3}；不能当成独立重复实验")

    # Load the fitted model and compute predictions afresh from raw A6-A11.
    model = joblib.load(HERE / "outputs" / "models" / "mixture_primary.joblib")
    meta = json.loads((T / "mixture_metadata.json").read_text(encoding="utf-8"))
    eps = float(meta["alr_zero_pseudocount"])
    comparison = pd.read_csv(T / "mixture_model_comparison.csv")
    pred_saved = pd.read_csv(T / "test_predictions.csv")
    recomputed = {}
    for label, scale in [("test_1m", "1m"), ("test_60m", "60m"), ("test_1B", "1B")]:
        indices, p, y, ycols = mixture_pair("test", scale)
        xp = p + eps
        xp /= xp.sum(axis=1, keepdims=True)
        alr = np.log(xp[:, :-1] / xp[:, -1, None])
        prediction = model.predict(alr)
        rmse = float(np.sqrt(np.mean((prediction - y) ** 2)))
        reported = float(comparison[(comparison.dataset.eq(label)) & (comparison.model.eq("alr_ridge"))].rmse.iloc[0])
        saved_rows = pred_saved[(pred_saved.dataset.eq(label)) & (pred_saved.model.eq("alr_ridge"))]
        pivot = saved_rows.pivot(index="index", columns="validation_domain", values="predicted")
        domains = [c.replace("metric/the_pile_", "").replace("_val_loss", "") for c in ycols]
        from_csv = pivot.reindex(index=indices, columns=domains).to_numpy(float)
        max_pred_diff = float(np.max(abs(from_csv - prediction)))
        recomputed[label] = rmse
        record(f"{label} 原始文件复算", abs(rmse - reported) < 1e-10 and max_pred_diff < 1e-10,
               f"RMSE {rmse:.6f}；与预测表最大差 {max_pred_diff:.2e}")

    # Labels in A12-A15 are estimates, never used for model fitting.
    est = comparison[comparison.dataset.str.startswith("est_")]
    record("外推表标记", set(est.status) == {"estimated_extrapolation"},
           f"A12–A15 性质：{sorted(set(est.status))}")

    report = HERE / "problem1_report.md"
    text = report.read_text(encoding="utf-8")
    links = re.findall(r"\]\(([^)]+)\)", text)
    local = [x for x in links if not x.startswith(("http:", "https:"))]
    missing = [x for x in local if not (HERE / x).exists()]
    record("报告图表与本地链接", not missing and len(list((HERE / "outputs" / "figures").glob("*.png"))) >= 14,
           f"本地链接 {len(local)}；失效 {len(missing)}；PNG {len(list((HERE / 'outputs' / 'figures').glob('*.png')))}")

    status = "通过" if all(ok for _, ok, _ in checks) else "未通过"
    lines = ["# 问题一独立核验结果", "", f"核验状态：**{status}**。本脚本直接重读压缩质量文件与 A6–A11 原始 CSV，并用保存的模型复算预测。", "",
             "| 检查项 | 结果 | 证据 |", "|---|---|---|"]
    lines += [f"| {name} | {'通过' if ok else '失败'} | {detail} |" for name, ok, detail in checks]
    lines += ["", "## 如何判断研究质量", "",
              "上述检查确认数据管线、结果表和报告数值能够相互复核；它**不能证明质量分 Q 表示真实训练价值**，也不能证明模型可直接用于大规模配置。",
              "", "下一道独立检验是让至少两名不看模型得分的评审者对分层抽取的 A1 原文打质量标签，检查 Q 与标签的一致性及评审者一致性。",
              "再往后，应使用新的配比实验或真实更大模型训练点检验 17 域配比模型；现有 60M/1B 数据已显示绝对 Loss 迁移失败，10B/70B 的 Loss 是估算值。",
              "", "可运行 `python problem1/run_problem1.py` 从原始附件重建全部结果，然后再次运行 `python problem1/verify_problem1.py`。"]
    (HERE / "verification_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(status)
    for name, ok, detail in checks:
        print(("PASS" if ok else "FAIL"), name, detail)
    if not all(ok for _, ok, _ in checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
