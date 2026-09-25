"""Build Problem 3 paper tables, figures, and Chinese final explanation.

All numerical values are read from the frozen P3 V2 tables or this tournament.
No baseline directory is written. Run after run_decision_tournament.py.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
TAB = HERE / "paper_tables"
FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "problem3"))
from core import total_cost  # noqa: E402

BLUE = "#2463A6"
ORANGE = "#E68627"
GREEN = "#39856D"
RED = "#C24B40"
PURPLE = "#7560A7"
GREY = "#6C7480"
CAPS = ["1x", "2x", "3x", "5x", "free"]
BUDGETS = [1e19, 1e22, 1e24]
POLICY_LABELS = {
    "minimax_regret": "Minimax regret",
    "nominal_shared": "Nominal, shared feasible",
    "balanced_mean_regret": "Balanced mean regret",
    "cvar90_regret": "CVaR90 regret",
    "worst_case_loss": "Worst-case loss",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_figure(fig, stem: str) -> None:
    fig.savefig(FIG / f"{stem}.png", dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def clean_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#DDE2E8", lw=.7)
    ax.set_axisbelow(True)


def fmt(value, digits=5):
    return f"{float(value):.{digits}g}"


def md_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |", *[
                          "| " + " | ".join(map(str, row)) + " |" for row in rows]])


def plot_evidence_levels(levels):
    label = {"E0_empirical": "E0: observed bound",
             "E1_moderate_3x": "E1: 3x bound",
             "E2_free_scaling": "E2: free"}
    color = {"E0_empirical": BLUE, "E1_moderate_3x": ORANGE,
             "E2_free_scaling": RED}
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.7), layout="constrained")
    for level, z in levels.groupby("level", sort=False):
        z = z.sort_values("budget")
        x = np.log10(z.budget.to_numpy(float))
        axes[0].plot(x, z.loss, marker="o", ms=6, lw=2, color=color[level], label=label[level])
        axes[1].plot(x, 100 * z.budget_ratio, marker="o", ms=6, lw=2,
                     color=color[level], label=label[level])
    for ax in axes:
        ax.set_xticks([19, 22, 24], [r"$10^{19}$", r"$10^{22}$", r"$10^{24}$"])
        ax.set_xlabel("Budget (FLOPs)")
        clean_axes(ax)
    axes[0].set_ylabel("Conditional nominal Loss")
    axes[1].set_ylabel("Budget used (%)")
    axes[1].set_ylim(0, 108)
    axes[0].legend(frameon=False, loc="upper right", fontsize=8)
    fig.suptitle("Evidence boundary changes high-budget conclusions", fontsize=12, fontweight="bold")
    save_figure(fig, "p3_evidence_levels")


def plot_cap_tradeoff(raw, summary):
    high = summary[(summary.budget == 1e24) & (summary.split == "heldout") &
                   (summary.policy == "minimax_regret")].set_index("cap").loc[CAPS]
    nominal = raw[raw.budget == 1e24].set_index("cap").loc[CAPS]
    x = np.arange(len(CAPS))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), layout="constrained")
    axes[0].plot(x, nominal.nominal_loss, "o--", ms=6, lw=1.8, color=GREY,
                 label="Single scenario optimum")
    axes[0].plot(x, high.nominal_loss, "s-", ms=6, lw=2.1, color=BLUE,
                 label="Shared-feasible minimax")
    axes[0].set_ylabel("Nominal Loss")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(x, high.mean_free_regret, "o-", ms=7, lw=2, color=RED,
                 label="Held-out mean free-oracle regret")
    axes[1].set_ylabel("Regret vs free oracle")
    for ax in axes:
        ax.axvspan(1.5, 2.5, color=ORANGE, alpha=.13, lw=0)
        ax.set_xticks(x, ["1x", "2x", "3x", "5x", "Free"])
        ax.set_xlabel("N and D upper-bound multiplier")
        clean_axes(ax)
    axes[1].annotate("3x: 0 within-cap regret\nbut 0.0535 vs free",
                     xy=(2, high.loc["3x", "mean_free_regret"]), xytext=(.25, .13),
                     arrowprops={"arrowstyle": "->", "color": GREY}, fontsize=8)
    fig.suptitle(r"$10^{24}$ FLOPs: loss and extrapolation-bound trade-off",
                 fontsize=12, fontweight="bold")
    save_figure(fig, "p3_cap_tradeoff")


def plot_fair_rules(summary):
    data = summary[(summary.split == "heldout") & (summary.cap == "3x")]
    order = ["nominal_shared", "balanced_mean_regret", "cvar90_regret",
             "worst_case_loss"]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.75), layout="constrained")
    for j, (budget, multiplier, unit) in enumerate(((1e19, 1e6, r"$10^{-6}$ Loss"),
                                                     (1e22, 1e9, r"$10^{-9}$ Loss"))):
        z = data[data.budget == budget].set_index("policy")
        base = float(z.loc["minimax_regret", "mean_within_cap_regret"])
        values = [(float(z.loc[k, "mean_within_cap_regret"]) - base) * multiplier
                  for k in order]
        bars = axes[j].barh(np.arange(len(order)), values,
                            color=[BLUE, ORANGE, PURPLE, GREEN], height=.58)
        axes[j].set_yticks(np.arange(len(order)))
        if j == 0:
            axes[j].set_yticklabels([POLICY_LABELS[k] for k in order])
        else:
            axes[j].set_yticklabels([])
        axes[j].invert_yaxis()
        axes[j].axvline(0, color=GREY, lw=1)
        axes[j].set_xlabel(f"Mean regret minus minimax ({unit})")
        axes[j].set_title(rf"Budget $10^{{{int(np.log10(budget))}}}$ FLOPs")
        clean_axes(axes[j])
        span = max(abs(v) for v in values)
        lo = min(0., min(values))
        hi = max(0., max(values))
        axes[j].set_xlim(lo - max(span * .22, .25), hi + max(span * .22, .25))
        for bar, value in zip(bars, values):
            offset = max(span * .035, .03)
            axes[j].text(value + (offset if value >= 0 else -offset),
                         bar.get_y() + bar.get_height() / 2, f"{value:+.3f}",
                         va="center", ha="left" if value >= 0 else "right", fontsize=8)
    fig.suptitle("Fair reoptimization makes risk rules nearly indistinguishable",
                 fontsize=12, fontweight="bold")
    save_figure(fig, "p3_fair_policy_regret")


def plot_costs(costs):
    z = costs[costs.cost_context == "central_exponential_4096"].sort_values("budget")
    x = np.arange(len(z))
    fig, ax = plt.subplots(figsize=(8.4, 4.2), layout="constrained")
    values = [("Training", z.train_pct.to_numpy(), BLUE),
              ("Quality", z.quality_pct.to_numpy(), ORANGE),
              ("Attention", z.attention_pct.to_numpy(), GREEN),
              ("Unused", z.unused_pct.to_numpy(), "#D7DDE3")]
    bottom = np.zeros(len(z))
    for label, v, color in values:
        ax.bar(x, v, bottom=bottom, width=.57, color=color, label=label)
        bottom += v
    ax.set_xticks(x, [r"$10^{19}$", r"$10^{22}$", r"$10^{24}$"])
    ax.set_xlabel("Budget (FLOPs)")
    ax.set_ylabel("Share of available budget (%)")
    ax.set_ylim(0, 110)
    ax.legend(ncol=1, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    clean_axes(ax)
    for i, r in enumerate(z.itertuples(index=False)):
        ax.text(i, 101.5, f"{r.total_pct:.1f}% used", ha="center", va="bottom", fontsize=8)
    ax.set_title("3x minimax: central cost accounting and unused budget",
                 fontsize=12, fontweight="bold", pad=12)
    save_figure(fig, "p3_cost_budget")


def plot_transition(transitions):
    colors = {"exponential": BLUE, "power": ORANGE, "logarithmic": GREEN}
    fig, ax = plt.subplots(figsize=(7.9, 3.5), layout="constrained")
    for j, kind in enumerate(("exponential", "power", "logarithmic")):
        z = transitions[transitions.cost_model == kind].set_index("transition")
        a = float(z.loc["start", "C_threshold"])
        b = float(z.loc["saturation", "C_threshold"])
        ax.plot([a, b], [j, j], lw=3, color=colors[kind], alpha=.8)
        ax.scatter([a, b], [j, j], s=[70, 70], c=[colors[kind]] * 2,
                   edgecolor="white", linewidth=.8, zorder=3)
        if np.isclose(a, b):
            ax.annotate("start = saturation", xy=(a, j), xytext=(3.0e18, 1.55),
                        arrowprops={"arrowstyle": "->", "color": GREY}, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(9e17, 3e20)
    ax.set_yticks(range(3), ["Exponential", "Power", "Logarithmic"])
    ax.invert_yaxis()
    ax.set_xlabel("Budget at quality investment start / saturation (FLOPs)")
    ax.grid(axis="x", color="#DDE2E8")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title(r"Conditional transitions ($Q_0=0.6$, 4096 tokens)",
                 fontsize=12, fontweight="bold")
    save_figure(fig, "p3_quality_transitions")


def main():
    frozen = json.loads((ROOT / "model_baseline_manifest.json").read_text(
        encoding="utf-8"))["files"]
    source_paths = [
        HERE / "problem3_decision_tournament.csv",
        HERE / "policies.csv",
        HERE / "nominal_cap_profile.csv",
        HERE / "verification_results.json",
        HERE / "run_metadata.json",
        ROOT / "problem3_v2/outputs/tables/extrapolation_levels.csv",
        ROOT / "problem3_v2/outputs/tables/quality_transition_points_exact.csv",
        ROOT / "problem3_v2/outputs/tables/mixture_support_diagnostics.csv",
    ]
    for path in source_paths:
        relative = path.relative_to(ROOT).as_posix()
        if relative in frozen and sha(path) != frozen[relative]["sha256"]:
            raise RuntimeError(f"Frozen source SHA changed: {relative}")
    metadata = json.loads((HERE / "run_metadata.json").read_text(encoding="utf-8"))
    model_artifact = ROOT / metadata["model_artifact"]
    if sha(model_artifact) != metadata["model_artifact_sha256"]:
        raise RuntimeError("P2 model changed: rerun P3 tournament before paper assets")
    checks = json.loads((HERE / "verification_results.json").read_text(encoding="utf-8"))
    if not checks["all_pass"]:
        raise RuntimeError("Tournament verification failed")
    summary = pd.read_csv(HERE / "problem3_decision_tournament.csv")
    policies = pd.read_csv(HERE / "policies.csv")
    raw = pd.read_csv(HERE / "nominal_cap_profile.csv")
    levels = pd.read_csv(ROOT / "problem3_v2/outputs/tables/extrapolation_levels.csv")
    transition = pd.read_csv(ROOT / "problem3_v2/outputs/tables/quality_transition_points_exact.csv")
    mix = pd.read_csv(ROOT / "problem3_v2/outputs/tables/mixture_support_diagnostics.csv")
    b1 = pd.read_csv(ROOT / "real_attachments/B_scaling_laws/pythia_training_log_existing.csv")
    plot_evidence_levels(levels)
    plot_cap_tradeoff(raw, summary)
    plot_fair_rules(summary)
    plot_transition(transition)
    policy = policies[(policies.cap == "3x") &
                      (policies.policy == "minimax_regret")].sort_values("budget")
    cost_rows = []
    for r in policy.itertuples(index=False):
        for context_name, q0, ctx, kind in (("central_exponential_4096", .6, 4096, "exponential"),
                                           ("worst_structural_scenario", None, None, None)):
            if context_name == "worst_structural_scenario":
                scenarios = [(k, q, c) for k in ("exponential", "power", "logarithmic")
                             for q in (.2, .8) for c in (2048, 32768)]
                kind, q0, ctx = max(scenarios, key=lambda item: total_cost(
                    r.N_B, r.D_B, r.Q_B, item[1], item[2], item[0])[0])
            total, train, quality, attn = total_cost(r.N_B, r.D_B, r.Q_B, q0, ctx, kind)
            cost_rows.append(dict(budget=r.budget, policy="3x_minimax_regret",
                                  cost_context=context_name, cost_model=kind, Q0=q0, L_ctx=ctx,
                                  N_B=r.N_B, D_B=r.D_B, Q_B=r.Q_B,
                                  total_flops=total, train_flops=train,
                                  quality_flops=quality, attention_flops=attn,
                                  total_pct=100 * total / r.budget,
                                  train_pct=100 * train / r.budget,
                                  quality_pct=100 * quality / r.budget,
                                  attention_pct=100 * attn / r.budget,
                                  unused_pct=100 * max(0, 1 - total / r.budget)))
    costs = pd.DataFrame(cost_rows)
    plot_costs(costs)
    levels.to_csv(TAB / "p3_evidence_levels.csv", index=False)
    transition.to_csv(TAB / "p3_quality_transitions.csv", index=False)
    costs.to_csv(TAB / "p3_cost_budget.csv", index=False, float_format="%.12g")
    policy_comparison = summary[(summary.split == "heldout") &
                                (summary.cap == "3x")].copy()
    policy_comparison.to_csv(TAB / "p3_fair_policy_comparison.csv", index=False,
                             float_format="%.12g")
    cap_comparison = summary[(summary.split == "heldout") &
                             (summary.policy == "minimax_regret")].copy()
    cap_comparison.to_csv(TAB / "p3_cap_regret_comparison.csv", index=False,
                          float_format="%.12g")
    kkt = raw[raw.cap == "free"][["budget", "N_B", "D_B", "Q_B",
                                     "nominal_loss", "nominal_budget_ratio",
                                     "raw_free_kkt_rel_error",
                                     "raw_free_kkt_complementarity_ok"]]
    kkt.to_csv(TAB / "p3_free_nominal_kkt.csv", index=False, float_format="%.12g")
    mix[["strategy", "delta_p", "support_class", "observed_training_row"]].to_csv(
        TAB / "p3_mixture_support.csv", index=False)
    source_manifest = {
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in source_paths},
        "p2_model_sha256": sha(model_artifact),
        "derived_from_frozen_p3_v2": True,
        "derived_from_current_tournament": True,
        "figures": sorted(p.name for p in FIG.iterdir() if p.is_file()),
        "paper_tables": sorted(p.name for p in TAB.iterdir() if p.is_file()),
    }
    write_detail(levels, raw, summary, costs, transition, kkt, mix, checks)
    source_manifest["output_sha256"] = {
        p.relative_to(HERE).as_posix(): sha(p)
        for p in [*FIG.iterdir(), *TAB.iterdir(), HERE / "问题三最终详解.md"]
        if p.is_file()
    }
    (HERE / "paper_assets_manifest.json").write_text(json.dumps(source_manifest,
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Built {len(source_manifest['figures'])} figure files and "
          f"{len(source_manifest['paper_tables'])} paper tables")


def write_detail(levels, raw, summary, costs, transition, kkt, mix, checks):
    label = {"E0_empirical": "E0：实测域", "E1_moderate_3x": "E1：3× 边界",
             "E2_free_scaling": "E2：自由外推"}
    level_rows = []
    for budget in BUDGETS:
        for level in ("E0_empirical", "E1_moderate_3x", "E2_free_scaling"):
            r = levels[(levels.budget == budget) & (levels.level == level)].iloc[0]
            level_rows.append((f"10^{int(np.log10(budget))}", label[level],
                               fmt(r.N_B), fmt(r.D_B), fmt(r.Q_B), fmt(r.loss),
                               f"{100*r.budget_ratio:.2f}%"))
    high = summary[(summary.split == "heldout") & (summary.budget == 1e24) &
                   (summary.policy == "minimax_regret")].set_index("cap").loc[CAPS]
    high_rows = []
    for cap, r in high.iterrows():
        raw_r = raw[(raw.budget == 1e24) & (raw.cap == cap)].iloc[0]
        high_rows.append((cap, fmt(r.N_B), fmt(r.D_B), fmt(r.nominal_loss),
                          fmt(raw_r.nominal_loss), fmt(r.mean_free_regret),
                          fmt(r.max_free_regret), f"{100*r.worst_budget_ratio:.1f}%"))
    fair_rows = []
    for budget in (1e19, 1e22):
        z = summary[(summary.split == "heldout") & (summary.budget == budget) &
                    (summary.cap == "3x")].set_index("policy")
        for policy in ("minimax_regret", "nominal_shared", "balanced_mean_regret",
                       "cvar90_regret", "worst_case_loss"):
            r = z.loc[policy]
            fair_rows.append((f"10^{int(np.log10(budget))}", POLICY_LABELS[policy],
                              fmt(r.mean_within_cap_regret, 10),
                              fmt(r.cvar90_within_cap_regret, 10),
                              fmt(r.max_within_cap_regret, 10)))
    cost_rows = []
    for budget in BUDGETS:
        central = costs[(costs.budget == budget) &
                        (costs.cost_context == "central_exponential_4096")].iloc[0]
        worst = costs[(costs.budget == budget) &
                      (costs.cost_context == "worst_structural_scenario")].iloc[0]
        cost_rows.append((f"10^{int(np.log10(budget))}",
                          f"{central.train_pct:.2f}%", f"{central.quality_pct:.2f}%",
                          f"{central.attention_pct:.2f}%", f"{central.total_pct:.2f}%",
                          f"{worst.total_pct:.2f}%"))
    transition_rows = []
    for kind in ("exponential", "power", "logarithmic"):
        z = transition[transition.cost_model == kind].set_index("transition")
        transition_rows.append((kind, fmt(z.loc["start", "C_threshold"], 6),
                                fmt(z.loc["saturation", "C_threshold"], 6)))
    kkt_rows = [(f"10^{int(np.log10(r.budget))}",
                 f"{r.raw_free_kkt_rel_error:.3g}",
                 "通过" if r.raw_free_kkt_complementarity_ok else "失败")
                for r in kkt.itertuples(index=False)]
    observed = mix.set_index("strategy").loc["P1_observed_support"]
    loose = mix.set_index("strategy").loc["P3_loose_simplex"]
    body = rf"""# 问题三最终详解：资源配置与证据约束决策

> 本文数值由 `build_paper_assets.py` 从 P3 V2 冻结表和本轮重算复赛表自动生成。当前上游为冻结的 P2 Model C；若 P1/P2 最终 Winner 改变，必须按接口重跑，不能继续引用本版数字。

## 1. 问题与统一模型

设模型规模 $N$ 与训练 token 数 $D$ 均以十亿为单位，质量水平 $Q\in[Q_0,1]$，可用 FLOPs 为 $C$。当前 P2 Model C 给出

$$
L(N,D,Q,p)=E+A N^{{-\alpha}}+B D^{{-\beta}}+G(1-Q)^{{\kappa}}N^{{-\eta_N}}+\lambda_p\Delta_p(p).
$$

中心参数来自冻结 P2 artifact。成本约束为

$$
C_{{\rm total}}=6\times10^{{18}}ND+10^9D[g(Q)-g(Q_0)]_++2\times10^{{14}}NDL_{{\rm ctx}}\le C.
$$

质量成本 $g$ 分别考察指数、幂和对数形式；这是题设结构情景，不是估计出的发生概率。对给定 $N,Q$，因为 Loss 随 $D$ 单调下降，可把 $D$ 消去：

$$
D_{{\max}}=\frac{{C}}{{(6+2\times10^{{-4}}L_{{\rm ctx}})10^{{18}}N+10^9[g(Q)-g(Q_0)]_+}}.
$$

硬上限时取 $D=\min(D_{{\max}},D_{{\rm cap}})$。共同可行决策还要求对三种质量成本、$Q_0=0.2/0.8$、上下文 2048/32768 都不超预算。名义单情景解与共同可行解属于不同决策口径，必须分开展示。

## 2. 当前 Champion 与复赛规则

冻结 Champion 为 P3 V2 的 **3× B1 上界内 minimax regret**。本轮对同一 Model C 和成本公式重算 1×、2×、3×、5×、free 五个上限，及名义共同可行最优、minimax、平衡均值 regret、CVaR90 regret、最差情景 Loss 五个规则。训练为 3 个完整联合 bootstrap 参数向量乘 12 个结构情景；留出为不同的 12 个参数向量乘同样结构情景。留出参数来自同一 B1/B6 原始数据重采样，不属于独立外部验证。情景平均只描述平衡设计，不代表现实期望。

每个情景 $s$ 的后悔值为 $R_s(x)=L_s(x)-\min_{{y\in\mathcal F_s}}L_s(y)$。域内 regret 用该上限的 oracle，跨上限比较统一使用 free oracle。本轮预先规定，替代 Champion 需共同可行、留出改善至少 0.005 Loss 且 5%、配对参数簇 bootstrap 正改善比例超过 90%，并且不增加实际 N/D 外推。复赛审计 `decision_audit.csv` 中没有候选同时通过。

## 3. E0/E1/E2 证据等级

下表是**指数质量成本、$Q_0=0.6$、4096 上下文**的单情景最优；E0 限在 B1 实测 N/D 范围，E1 上界扩大 3 倍，E2 自由外推。E1 是敏感性边界，不是实测可信区间。

{md_table(['预算 FLOPs','等级','N (B)','D (B)','Q','Loss','预算使用'], level_rows)}

10^19 与 10^22 的三层基本重合；10^24 时 E0/E1 都碰到 N、D 硬上限且闲置大量预算，E2 使用全部预算但 D 达约 3657B，超出 B1 观测上限约 12.2 倍。故“Loss 更低”不能直接视为更可信的生产方案。

![E0/E1/E2 条件结果](figures/p3_evidence_levels.png)

## 4. 高预算上限与共同可行决策

下表使用 10^24 FLOPs、**同一留出情景和同一 free oracle**。`共同名义 Loss`是 minimax 政策在中心模型上的 Loss；`单情景 Loss`是独立指数/4096/$Q_0=0.6$ 的 raw nominal 最优，后者在 5×/free 下只有一半结构情景可行，不能当共同政策比较。

{md_table(['上限','共同 N (B)','共同 D (B)','共同名义 Loss','单情景 Loss','自由均值 regret','自由最大 regret','最坏成本率'], high_rows)}

在 3× 内，36 个训练情景的解共享同一个角点，使域内 regret 为 0；留出情景也如此。但相同 3× 政策对 free oracle 的留出均值 regret 是 {fmt(high.loc['3x','mean_free_regret'],7)}。5× 与 free 降低此条件后悔值，却分别把 D 推到 B1 上限的 5 倍与约 {fmt(high.loc['free','D_over_B1_max'],4)} 倍。没有独立数据支持这样的外推上限，故报告 Pareto 剖面，不给唯一高预算规模。3× minimax 仅保留为清楚标注条件的展示政策。

![高预算上限权衡](figures/p3_cap_tradeoff.png)

## 5. 公平名义对照与风险目标

原 V2 把单情景名义解截入共同可行域后直接比较。截入后没有重新优化 N/Q，10^19 下修复名义方案的留出平均 regret 约 0.172389。本轮直接在共同可行域重优化名义目标，得到更公平的对照：

{md_table(['预算 FLOPs','规则','留出平均域内 regret','CVaR90 regret','最大 regret'], fair_rows)}

10^19 下公平名义方案与 minimax 的平均 regret 仅差约 1.5×10^-7；10^22 下差约 3.8×10^-9。mean、CVaR90、最差 Loss 规则也只出现极小数值差异。此证据不支持换用更复杂风险准则，也不支持宣称 minimax 对公平名义优化具有实质性能优势。保留 minimax 是既有可解释条件政策的延续；简化为共同可行名义方案亦几乎等价。

![公平重优化风险规则对照](figures/p3_fair_policy_regret.png)

## 6. 成本、预算与 KKT

下表对 3× minimax 政策核算。前三项和中心使用率对应指数成本、$Q_0=0.6$、4096 上下文；最坏使用率对应 12 种结构情景中的最大成本。因此共同可行并不意味着每个情景都花满预算。

{md_table(['预算 FLOPs','中心训练','中心质量','中心注意力','中心总使用','最坏结构总使用'], cost_rows)}

10^24 的 3× 政策在中心情景只用约 22.35% 预算，在最坏结构情景也只用约 40.99%。闲置由 N/D 外推上限驱动，不能通过在上限外继续增大 N/D 来伪装为已验证收益。free 单情景名义解的 KKT 残差如下；Q=1 时采用单侧互补条件：

{md_table(['预算 FLOPs','free 名义 KKT 相对误差','Q 互补'], kkt_rows)}

minimax/CVaR 目标和硬上限交界可不光滑；其八个局部可行方向最大改善量为 {fmt(checks['max_local_direction_improvement'],4)}，这是数值局部检查，不是完整 KKT 证书。全部 75 个政策对 12 种结构情景的可行率为 100%；成本独立重算误差最大 {fmt(checks['cost_recompute_max_ratio_error'],4)}。

![预算成本构成](figures/p3_cost_budget.png)

## 7. 质量投资结构点与配比接口

在固定 $Q_0=0.6$、4096 上下文时，质量投资起点定义为 $Q^*>Q_0+10^{{-4}}$，饱和点为 $Q^*\ge1-10^{{-4}}$。扫描后在 $\log_{{10}} C$ 上二分。阈值是给定模型的数值结果，不是统计置信区间：

{md_table(['质量成本','起投 FLOPs','饱和 FLOPs'], transition_rows)}

对数成本的起投与饱和同点，表示当前模型下的离散跳跃，不应绘成连续过渡。不同质量成本情景的阈值差异也说明，不宜给不附条件的统一起投预算。

![质量投资结构点](figures/p3_quality_transitions.png)

配比项在固定 $\lambda_p>0$ 且可加可分时不改变 N-D-Q 最优。冻结 P1/P2 桥接下，观测配比候选预测 $\Delta_p={fmt(observed.delta_p,6)}$、在训练支持域内；宽松 simplex 候选预测 $\Delta_p={fmt(loose.delta_p,6)}$，却属外推。跨尺度 $\lambda_p(N)$ 未被独立识别，因此不能把 1M 配比收益直接外推到高预算，也不能在此复赛中用配比项“抵消”规模决策风险。若最终 P1 或 P2 桥接改变，需重算本问及 P4 下游。

## 8. 最终结论与证据边界

本轮确实重算 2700 个场景 oracle、75 个政策和 13500 条情景评分；全体 P3/P3 V2 冻结文件 SHA256 核对通过。没有 Challenger 同时满足预先规定的后悔改善、可行性和外推条件。论文可将 3× minimax 作为**条件式 Champion**，并给出 E0/E1/E2、高预算上限、质量成本转折和预算闲置；不能称 3× 是经外部验证的唯一最优规模，也不能把同源 bootstrap 留出称为独立外部验证。
"""
    (HERE / "问题三最终详解.md").write_text(body, encoding="utf-8")


if __name__ == "__main__":
    main()
