"""Generate paper numbers and figures from corrected P1 tournament CSVs."""
from __future__ import annotations

import os
from pathlib import Path

OUT = Path(__file__).resolve().parent
os.environ["MPLCONFIGDIR"] = str(OUT / ".mplcache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json


FIG = OUT / "figures"


def main():
    verification = json.loads((OUT / "verification_results.json").read_text(encoding="utf-8"))
    if verification["status"] != "PASS":
        raise RuntimeError("P1 verification must pass before paper generation")
    FIG.mkdir(exist_ok=True)
    metrics = pd.read_csv(OUT / "metrics_by_scale.csv")
    gains = pd.read_csv(OUT / "paired_bootstrap_gain.csv")
    tournament = pd.read_csv(OUT / "problem1_model_tournament.csv")
    domain = pd.read_csv(OUT / "per_domain_paired_bootstrap.csv")
    q = pd.read_csv(OUT / "q_sensitivity_audit.csv")
    em = metrics.set_index(["dataset", "family"])
    eg = gains.set_index(["dataset", "family"])
    labels = ["train_nested_oof", "test_1m", "test_60m", "test_1B", "est_10b", "est_70b"]
    records = []
    for ds in labels:
        b = em.loc[(ds, "v2_champion")]
        n = em.loc[(ds, "alr_elasticnet")]
        g = eg.loc[(ds, "alr_elasticnet")]
        records.append(dict(dataset=ds, n_mixtures=int(n.n_mixtures),
                            status=n.evidence_status,
                            v2_rmse=float(b.rmse), elasticnet_rmse=float(n.rmse),
                            v2_minus_elasticnet_rmse=float(g.v2_minus_candidate_rmse),
                            gain_ci_low=float(g.ci_low), gain_ci_high=float(g.ci_high),
                            v2_mae=float(b.mae), elasticnet_mae=float(n.mae),
                            v2_spearman_mean=float(b.spearman_mean),
                            elasticnet_spearman_mean=float(n.spearman_mean),
                            v2_mixture_mean_spearman=float(b.mixture_mean_spearman),
                            elasticnet_mixture_mean_spearman=float(n.mixture_mean_spearman),
                            v2_bias=float(b.bias), elasticnet_bias=float(n.bias)))
    key = pd.DataFrame(records)
    key.to_csv(OUT / "paper_key_numbers.csv", index=False, encoding="utf-8-sig")
    sig_harm = domain[(domain.dataset == "test_1B") & (domain.ci_high < 0)]
    if len(sig_harm) != 4:
        raise AssertionError("Expected four 1B domains with adverse retrospective CIs")
    selected = tournament.set_index("family").loc["alr_elasticnet"]
    lines = ["# 问题一论文关键数字（脚本生成）", "",
             "生成源：`metrics_by_scale.csv`、`paired_bootstrap_gain.csv`、`problem1_model_tournament.csv`、`per_domain_paired_bootstrap.csv`、`q_sensitivity_audit.csv`。",
             "所有 A6/A7、60M、1B 结果均为**回顾性**；10B/70B 是估算标签。RMSE 差为 V2 减 ElasticNet，正数表示 ElasticNet 更低。", "",
             "| 数据 | n | V2 RMSE | ElasticNet RMSE | 差值 [配对 95% CI] |", "|---|---:|---:|---:|---:|"]
    for row in key.itertuples():
        lines.append(f"| {row.dataset} | {row.n_mixtures} | {row.v2_rmse:.4f} | {row.elasticnet_rmse:.4f} | {row.v2_minus_elasticnet_rmse:+.4f} [{row.gain_ci_low:+.4f}, {row.gain_ci_high:+.4f}] |")
    lines += ["", f"最终规格：ALR 二阶特征 + MultiTask ElasticNet，`alpha=0.01`、`l1_ratio=0.7`、参考域 `pile_cc`、`epsilon=0.000249500998`；激活特征 {int(selected.active_features)}/152。",
              f"修正外折参考域后的嵌套 OOF：V2 {key.iloc[0].v2_rmse:.6f}，ElasticNet {key.iloc[0].elasticnet_rmse:.6f}；五折均改善。",
              f"1pp 局部效应与 V2 符号一致 {int(selected.effect_sign_agreement_with_v2)}/16；但 Bootstrap 中达到 ≥90% 同号率的目标域只有 {int(selected.stable_1pp_targets)}/16。", "",
              "## 1B 逐域损害（事后描述，不用于调整门槛）", "",
              "| 验证域 | 新减旧 RMSE | 相对变化 | V2 减新模型 95% CI |", "|---|---:|---:|---:|"]
    dom_rmse = pd.read_csv(OUT / "metrics_by_domain.csv")
    old1b = dom_rmse[(dom_rmse.dataset == "test_1B") & (dom_rmse.family == "v2_champion")].set_index("validation_domain")
    for row in sig_harm.itertuples():
        worse = -row.v2_minus_selected_rmse
        pct = 100*worse/old1b.loc[row.validation_domain, "rmse"]
        lines.append(f"| {row.validation_domain} | +{worse:.4f} | +{pct:.2f}% | [{row.ci_low:+.4f}, {row.ci_high:+.4f}] |")
    lines += ["", "四个验证域各占 pooled RMSE 的 1/13 个输出维度；1B 总体 RMSE 与配方平均排序改善，并不表示各域均改善。", "",
              "## Q 证据边界", "",
              f"人工盲评有效标签 0 行；Q 四组等权保留。无监督 PCA 的 7 域均值排序与原 Q Spearman {q.set_index('method').loc['PCA_1','domain_spearman_vs_Q']:.2f}；熵权为 {q.set_index('method').loc['entropy_22','domain_spearman_vs_Q']:.2f}。FA 未数值收敛。上述值只度量敏感性，不能选出质量真值模型。", "",
              "![跨规模 RMSE 对照](figures/cross_scale_rmse.png)", "",
              "![配对 RMSE 改善区间](figures/paired_gain_forest.png)", "",
              "![1B 逐域权衡](figures/one_b_domain_tradeoffs.png)", ""]
    (OUT / "paper_key_numbers.md").write_text("\n".join(lines), encoding="utf-8")

    plt.rcParams.update({"font.size": 10, "figure.dpi": 120, "savefig.dpi": 180})
    colors = {"v2_champion": "#355c7d", "alr_elasticnet": "#42a78b", "clr_spline": "#d9a441"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0))
    for ax, ds, title in zip(axes, ("test_1m", "test_60m", "test_1B"),
                              ("A6/A7 1M", "60M", "1B")):
        b = float(em.loc[(ds, "v2_champion"), "rmse"])
        n = float(em.loc[(ds, "alr_elasticnet"), "rmse"])
        s = float(em.loc[(ds, "clr_spline"), "rmse"])
        values = [b, n, s]
        bars = ax.bar([0, 1, 2], values,
                      color=[colors[k] for k in ("v2_champion", "alr_elasticnet", "clr_spline")], width=.65)
        ax.set_xticks([0, 1, 2], ["V2", "ElasticNet", "CLR spline"], rotation=20, ha="right")
        ax.set_ylim(0, max(values)*1.17)
        ax.set_ylabel("Pooled RMSE")
        ax.set_title(f"{title} (retrospective)\nV2 - ENet = {b-n:+.3f}")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, value+max(values)*.018,
                    f"{value:.3f}", ha="center", va="bottom", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("One model trained on A4/A5 1M; no scale recalibration", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "cross_scale_rmse.png", bbox_inches="tight")
    plt.close(fig)

    ds_order = ["train_nested_oof", "test_1m", "test_60m", "test_1B"]
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.axvline(0, color="#49515b", linewidth=1)
    for i, family in enumerate(("alr_elasticnet", "clr_spline")):
        sub = gains[(gains.family == family) & gains.dataset.isin(ds_order)].set_index("dataset").loc[ds_order]
        yy = np.arange(len(ds_order)) + (-.14 if i == 0 else .14)
        val = sub.v2_minus_candidate_rmse.to_numpy(float)
        lo, hi = sub.ci_low.to_numpy(float), sub.ci_high.to_numpy(float)
        ax.errorbar(val, yy, xerr=[val-lo, hi-val], fmt="o", capsize=3,
                    color=colors[family], label="ElasticNet" if i == 0 else "CLR spline")
    ax.set_yticks(np.arange(len(ds_order)), ["A4/A5 nested OOF", "A6/A7 1M", "60M", "1B"])
    ax.invert_yaxis()
    ax.set_xlabel("V2 RMSE - challenger RMSE (positive favors challenger)")
    ax.set_title("Paired mixture-row bootstrap, conditional 95% intervals")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "paired_gain_forest.png", bbox_inches="tight")
    plt.close(fig)

    d = domain[domain.dataset == "test_1B"].copy()
    d["new_minus_old"] = -d.v2_minus_selected_rmse
    d = d.sort_values("new_minus_old")
    harm = d.ci_high < 0
    better = d.ci_low > 0
    bar_colors = np.where(harm, "#c45f5f", np.where(better, "#42a78b", "#9aa7ad"))
    fig, ax = plt.subplots(figsize=(9.6, 5.7))
    ax.barh(d.validation_domain, d.new_minus_old, color=bar_colors)
    ax.axvline(0, color="#2f3e46", linewidth=1)
    ax.set_xlabel("ElasticNet - V2 domain RMSE (positive = worse)")
    ax.set_title("1B: pooled gain masks four adverse validation domains")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.16)
    ax.spines[["top", "right"]].set_visible(False)
    old_lookup = old1b.rmse
    for row in d[harm].itertuples():
        pct = 100*row.new_minus_old/old_lookup.loc[row.validation_domain]
        ax.text(row.new_minus_old+.003, list(d.validation_domain).index(row.validation_domain),
                f"+{pct:.1f}%", va="center", color="#9a3838", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "one_b_domain_tradeoffs.png", bbox_inches="tight")
    plt.close(fig)
    print("paper assets generated:", *(str(p.name) for p in FIG.glob("*.png")), sep="\n")


if __name__ == "__main__":
    main()
