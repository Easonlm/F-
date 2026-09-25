"""Generate the final Problem 2 model card, detailed explanation, numbers and plots.

All reported fitted quantities are loaded from this rebuild's CSV/JSON outputs.
Only rebuild/problem2/ is written.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 140, "savefig.dpi": 230,
})
BLUE, ORANGE, GRAY = "#2878B5", "#E38B2C", "#64748B"


def load():
    return {
        "selection": json.loads((HERE / "selection.json").read_text(encoding="utf-8")),
        "nd": pd.read_csv(HERE / "nd_scorecard.csv").set_index("model"),
        "q": pd.read_csv(HERE / "quality_scorecard.csv").set_index("model"),
        "qi": pd.read_csv(HERE / "quality_parameter_intervals.csv"),
        "calib": pd.read_csv(HERE / "source_conditional_holdout.csv"),
        "transfer": pd.read_csv(HERE / "source_leave_B5_out.csv"),
        "overlap": pd.read_csv(HERE / "B6_B8_exact_overlap.csv"),
        "direction": pd.read_csv(HERE / "quality_direction_by_source.csv"),
        "mechanism": json.loads((HERE / "mechanism_diagnostic.json").read_text(encoding="utf-8")),
        "verify": json.loads((HERE / "verification_results.json").read_text(encoding="utf-8")),
        "reproduction": json.loads((HERE / "reproduction.json").read_text(encoding="utf-8")),
        "frozen": json.loads((HERE / "frozen_baseline_sha_audit.json").read_text(encoding="utf-8")),
        "manifest": json.loads((HERE / "input_manifest.json").read_text(encoding="utf-8")),
    }


def fmt(x, digits=5):
    return f"{float(x):.{digits}f}"


def md_table(df, columns=None, formatters=None):
    if columns is not None:
        df = df[columns]
    formatters = formatters or {}
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in df.itertuples(index=False, name=None):
        cells = []
        for key, val in zip(headers, row):
            if key in formatters:
                cells.append(formatters[key](val))
            elif isinstance(val, (float, np.floating)):
                cells.append(f"{val:.5g}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def make_numbers(a):
    sel, nd, q = a["selection"], a["nd"], a["q"]
    numbers = []
    def add(name, value, unit, dataset, split, source_file, meaning):
        numbers.append(dict(number=name, value=float(value), unit=unit, dataset=dataset,
                            split=split, source_file=source_file, meaning=meaning))
    classic = nd.loc[sel["ND_winner"]]
    qwin = q.loc[sel["Q_winner"]]
    for key, val in zip(("E", "A", "alpha", "B", "beta"), sel["ND_parameters"]):
        add(key, val, "model_parameter", "B1", "full_fit", "selection.json", "N-D winner parameter")
    for key, val in zip(("G", "kappa", "eta_N"), sel["Q_parameters"]):
        add(key, val, "model_parameter", "B6", "full_fit", "selection.json", "Q winner parameter")
    for label, col, dataset, split in (
        ("B1_leave_N_RMSE", "B1_group_cv_rmse", "B1", "leave_N"),
        ("B2_raw_RMSE", "B2_raw_rmse", "B2", "zero_shot_raw"),
        ("B4_raw_RMSE", "B4_raw_rmse", "B4", "zero_shot_raw"),
        ("B5_raw_RMSE", "B5_raw_rmse", "B5", "zero_shot_raw"),
    ):
        add(label, classic[col], "loss_RMSE", dataset, split, "nd_scorecard.csv", "selected classic N-D")
    for label, col, dataset, split in (
        ("B6_leave_N_RMSE", "B6_leave_N_rmse", "B6", "leave_N"),
        ("B7_new_RMSE", "B7_new_rmse", "B7", "new_points"),
        ("B8_diagnostic_RMSE", "B8_new_diagnostic_rmse", "B8", "mechanism_only"),
    ):
        add(label, qwin[col], "loss_RMSE", dataset, split, "quality_scorecard.csv", "selected Q-by-N")
    add("Q_by_ND_B7_new_RMSE", q.loc["Q_by_ND", "B7_new_rmse"], "loss_RMSE", "B7", "new_points", "quality_scorecard.csv", "unselected Q-by-N-D")
    add("Q_by_ND_B7_gain_CI_low", q.loc["Q_by_ND", "B7_vs_champion_ci_low"], "loss_RMSE_difference", "B7", "N_cluster_bootstrap", "quality_scorecard.csv", "champion RMSE minus challenger RMSE")
    add("Q_by_ND_B7_gain_CI_high", q.loc["Q_by_ND", "B7_vs_champion_ci_high"], "loss_RMSE_difference", "B7", "N_cluster_bootstrap", "quality_scorecard.csv", "champion RMSE minus challenger RMSE")
    add("B6_B8_exact_overlap_n", a["mechanism"]["B6_B8_exact_overlap"]["n"], "rows", "B6+B8", "exact_N_D_Q_overlap", "mechanism_diagnostic.json", "conflicting loss labels")
    add("B6_B8_overlap_mean_difference", a["mechanism"]["B6_B8_exact_overlap"]["mean_difference"], "loss", "B6+B8", "exact_N_D_Q_overlap", "mechanism_diagnostic.json", "B8 minus B6")
    for src, protocol in (("B2", "source_logD_20pct"), ("B4", "source_affine_20pct"), ("B5", "source_affine_20pct")):
        row = a["calib"].query("source == @src and protocol == @protocol").iloc[0]
        add(f"{src}_{protocol}_conditional_RMSE", row.rmse, "loss_RMSE", src, "early_20pct_labeled_late_80pct_holdout", "source_conditional_holdout.csv", "descriptive labeled application; not zero shot")
    save = pd.DataFrame(numbers)
    save.to_csv(HERE / "paper_key_numbers.csv", index=False, float_format="%.12g")
    (HERE / "paper_key_numbers.json").write_text(json.dumps(numbers, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_cross_source(a):
    nd = a["nd"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), constrained_layout=True)
    names = ["classic", "interaction", "broken_N"]
    x = np.arange(len(names))
    axes[0].bar(x - .18, nd.loc[names, "B1_training_rmse"].to_numpy() * 1e4, .36, label="B1 train", color=GRAY)
    axes[0].bar(x + .18, nd.loc[names, "B1_group_cv_rmse"].to_numpy() * 1e4, .36, label="B1 leave-N", color=BLUE)
    axes[0].set_xticks(x, ["Classic", "N x D", "Broken N"])
    axes[0].set_ylabel("RMSE (x 1e-4)")
    axes[0].set_title("Same-source fit versus held-out N")
    axes[0].legend(loc="lower left", frameon=False, fontsize=8)
    for i, src in enumerate(("B2", "B4", "B5")):
        val = nd.loc["classic", f"{src}_raw_rmse"]
        axes[1].bar(i, val, width=.62, color=[ORANGE, BLUE, GRAY][i])
        axes[1].text(i, val + .025, f"{val:.3f}", ha="center", fontsize=9)
    axes[1].set_xticks(np.arange(3), ["B2", "B4", "B5"])
    axes[1].set_ylim(0, nd.loc["classic", "B2_raw_rmse"] * 1.23)
    axes[1].set_ylabel("Raw zero-shot RMSE")
    axes[1].set_title("External sources retain large shifts")
    fig.savefig(FIG / "p2_nd_source_shift.png", bbox_inches="tight")
    plt.close(fig)


def plot_quality(a):
    q = a["q"]
    names = ["additive", "effective_tokens", "Q_by_N", "Q_by_D", "Q_by_ND"]
    labels = ["Additive", "Effective\ntokens", "Q x N", "Q x D", "Q x N x D"]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9.2, 4.2), constrained_layout=True)
    ax.bar(x - .18, q.loc[names, "B6_leave_N_rmse"].to_numpy(), .36, label="B6 leave-N", color=BLUE)
    ax.bar(x + .18, q.loc[names, "B7_new_rmse"].to_numpy(), .36, label="B7 new points", color=ORANGE)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(q.B6_leave_N_rmse) * 1.18)
    ax.set_ylabel("Loss RMSE")
    ax.set_title("Quality-law validation; B8 excluded from selection")
    ax.legend(frameon=False)
    ax.axvspan(1.5, 2.5, color=BLUE, alpha=.07, zorder=-1)
    fig.savefig(FIG / "p2_quality_tournament.png", bbox_inches="tight")
    plt.close(fig)


def plot_b8(a):
    overlap = a["overlap"]
    direction = a["direction"]
    by_q = overlap.groupby("Q_score").B8_minus_B6.agg(["mean", "count"]).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(by_q.Q_score, by_q["mean"], marker="o", lw=2,
                 color=ORANGE)
    axes[0].axhline(0, color=GRAY, lw=1, ls="--")
    axes[0].set_xlabel("Shared Q score")
    axes[0].set_ylabel("B8 loss minus B6 loss")
    axes[0].set_title(f"Mean differences at {len(overlap)} identical inputs")
    direction_rate = direction.groupby("source").rho_Q_L.apply(lambda x: np.mean(x < 0))
    order = ["B6", "B7_full", "B8_new"]
    axes[1].bar(np.arange(3), direction_rate.loc[order].to_numpy(), color=[BLUE, BLUE, ORANGE], width=.55)
    axes[1].set_xticks(np.arange(3), ["B6", "B7", "B8 new"])
    axes[1].set_ylim(0, 1.08)
    axes[1].set_ylabel("Fraction of (N,D) cells where loss falls with Q")
    axes[1].set_title("Opposite quality response in B8")
    axes[1].text(2, .025, "0%", ha="center", color=ORANGE, fontsize=10)
    fig.savefig(FIG / "p2_b8_mechanism.png", bbox_inches="tight")
    plt.close(fig)


def plot_source_calibration(a):
    c = a["calib"]
    protocols = ["raw", "source_intercept_20pct", "source_affine_20pct", "source_logD_20pct"]
    labels = ["Raw", "Intercept", "Affine", "log D"]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7), constrained_layout=True)
    for ax, src in zip(axes, ("B2", "B4", "B5")):
        z = c[c.source == src].set_index("protocol").loc[protocols]
        bars = ax.bar(np.arange(4), z.rmse, color=[GRAY, "#9EB8C9", BLUE, ORANGE])
        ax.set_xticks(np.arange(4), labels, rotation=30, ha="right")
        ax.set_title(f"{src}: 20% labeled early data")
        ax.set_ylabel("Later holdout RMSE" if src == "B2" else "")
        ax.set_ylim(0, max(z.rmse) * 1.23)
        for bar, val in zip(bars, z.rmse):
            ax.text(bar.get_x() + bar.get_width()/2, val + max(z.rmse)*.025, f"{val:.3f}",
                    ha="center", fontsize=8)
    fig.savefig(FIG / "p2_labeled_source_application.png", bbox_inches="tight")
    plt.close(fig)


def plot_quality_scale(a):
    e, amp_n, alpha, amp_d, beta = a["selection"]["ND_parameters"]
    g, kappa, eta = a["selection"]["Q_parameters"]
    q_grid = np.linspace(.05, 1, 200)
    fig, ax = plt.subplots(figsize=(6.1, 4), constrained_layout=True)
    for n, color in ((.07, ORANGE), (.7, BLUE), (7, GRAY)):
        y = e + amp_n*n**(-alpha) + amp_d*300**(-beta) + g*(1-q_grid)**kappa*n**(-eta)
        ax.plot(q_grid, y, lw=2.2, color=color, label=f"N={n:g}B")
    ax.set_xlabel("B6/B7 quality score Q")
    ax.set_ylabel("Predicted validation loss (D=300B)")
    ax.set_title("Selected Q x N response within the studied N range")
    ax.legend(frameon=False)
    fig.savefig(FIG / "p2_quality_by_scale.png", bbox_inches="tight")
    plt.close(fig)


def reports(a):
    s, nd, q, qi = a["selection"], a["nd"], a["q"], a["qi"]
    e, aa, alpha, bb, beta = s["ND_parameters"]
    g, kappa, eta = s["Q_parameters"]
    qnd = q.loc["Q_by_ND"]
    qwin = q.loc[s["Q_winner"]]
    classic = nd.loc[s["ND_winner"]]
    nd_comp = nd.reset_index()[["model", "B1_training_rmse", "B1_group_cv_rmse", "B2_raw_rmse", "B4_raw_rmse", "B5_raw_rmse", "B1_bic", "jacobian_condition"]]
    q_comp = q.reset_index()[["model", "B6_training_rmse", "B6_leave_N_rmse", "B7_new_rmse", "B6_bic", "jacobian_condition"]]
    interval_n = qi.query("model == 'Q_by_N' and parameter == 'eta_N'").iloc[0]
    interval_d = qi.query("model == 'Q_by_ND' and parameter == 'eta_D'").iloc[0]
    gain_b6 = 100 * (1 - qnd.B6_leave_N_rmse/qwin.B6_leave_N_rmse)
    gain_b7 = 100 * (1 - qnd.B7_new_rmse/qwin.B7_new_rmse)
    source_best = a["calib"].query("protocol != 'raw'").sort_values("rmse").groupby("source", as_index=False).first()
    source_table = a["calib"][["source", "protocol", "calibration_rows", "holdout_rows", "rmse"]]
    form = (f"L={e:.10f}+{aa:.10f}N^{{-{alpha:.10f}}}+{bb:.10f}D^{{-{beta:.10f}}}"
            f"+{g:.10f}(1-Q_B)^{{{kappa:.10f}}}N^{{-{eta:.10f}}}+\\lambda_p\\Delta_p")

    card = f"""# 问题二最终模型卡

**模型状态：** 原 Model C 经复赛保留。N-D Winner=`{s['ND_winner']}`，Q Winner=`{s['Q_winner']}`。参数与指标均由本目录重跑产物自动生成。

## 使用方式与输入

`predict_final.predict_loss(N_B, D_B, Q_B, delta_p=0, lambda_p=1)`；`N_B` 是十亿参数，`D_B` 是十亿 token，`Q_B∈[0,1]` 是 B6/B7 半合成质量量表。`delta_p` 须由最终问题一配方模型相对同一参考配方中心化。默认 `lambda_p=1` 是未识别桥接情景。

主公式：`{form}`。

## 拟合数据与独立层级

N-D 参数仅 B1 拟合；B1 按 N 留组 RMSE **{classic.B1_group_cv_rmse:.8f}**。B2/B4/B5 零样本 raw RMSE 分别为 **{classic.B2_raw_rmse:.4f}/{classic.B4_raw_rmse:.4f}/{classic.B5_raw_rmse:.4f}**，显示跨来源误差。Q 参数仅 B6 拟合；B6 留 N RMSE **{qwin.B6_leave_N_rmse:.5f}**，B7 新增 {s['B7_new_rows']} 点 RMSE **{qwin.B7_new_rmse:.5f}**。这些外部文件曾用于旧轮次研究，属于可复查的开发证据，不是新封存盲测。

## 选择与不确定性

低维 N×D 交互及 broken-N 未改善 B1 组外预测；broken-N Jacobian 条件数 {nd.loc['broken_N','jacobian_condition']:.2e}。Q×N×D 的 B7 RMSE {qnd.B7_new_rmse:.5f}，相对 Champion 改善 {gain_b7:.2f}%；按 N 组自举的 RMSE 改善 95% 区间 [{qnd.B7_vs_champion_ci_low:.5f},{qnd.B7_vs_champion_ci_high:.5f}] 跨零，未通过预注册 5% 门槛。Q×N 的 `eta_N` B6 组自举区间 [{interval_n.ci_low:.4f},{interval_n.ci_high:.4f}]；此区间仅反映 B6 半合成机制。

## 适用边界

- B8 仅机制诊断：与 B6 有 {s['B8_mechanism']['B6_B8_exact_overlap']['n']} 个相同输入却不同 Loss，B8 中 Q 响应方向相反。不得把 B8 写成统一 Q 律的外部通过案例。
- 有目标来源 Loss 标签时，可以在独立的条件应用层估计截距、仿射或 `log D` 修正；这些系数不能由 B1/B6 零样本推得，且不自动作用于 Q 或配方项。
- `lambda_p(N)` 和 `Q_A→Q_B` 均未识别。B 数据无联合配方/规模 Loss 标签，两个质量量表无配对锚点。
- Q 模型拟合来自半合成附件；大 N 和真实异源训练外推应作为敏感性，不给精确政策承诺。

## 复现与完整性

运行 `python rebuild/problem2/reproduce.py`；{len(a['reproduction']['verified_files'])} 个输出两次逐字节一致，{len(a['verify']['checks'])}/{len(a['verify']['checks'])} 项验证通过。冻结问题二目录 {a['frozen']['unchanged']}/{a['frozen']['checked']} 个 SHA 不变。来源与完整指标见 `input_manifest.json`、`problem2_model_tournament.csv`、`paper_key_numbers.csv`。
"""
    (HERE / "model_card.md").write_text(card, encoding="utf-8")

    detail = f"""# 问题二最终详解

## 1. 研究任务与证据边界

问题二将预训练 Loss 表述为模型规模 `N`、token 数 `D`、质量分数 `Q_B` 及问题一配方效应 `Delta_p` 的函数。不同附件的验证 Loss 口径及数据生成机制未被证明等同，因此以 B1/B6 同源拟合、B1/B6 留组、B7 新点和 B2/B4/B5 raw 外部检验构成分层证据。B8 仅检验机制相容性。数据路径和 SHA 见 `input_manifest.json`。

## 2. 候选提出依据

B1 的经典加性律是 [Hoffmann 等，NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/c1e2faff6f588870935f114ebe04a3e5-Paper-Conference.pdf) 一类规模律的低维基线。B1 训练误差接近四位小数记录精度，所以只测试乘积交互与一个平滑 N 转折；后者借鉴 [Caballero 等，ICLR 2023](https://openreview.net/forum?id=BfGrlFuNyhJ)，须有组外改善才能采用。质量的有效 token 候选借鉴 [Chang 等，EMNLP Industry 2024](https://aclanthology.org/2024.emnlp-industry.8/)，但附件只含汇总 Q，不能复刻原文的多样性和合成度指标。具体先验门槛记录在 `why_this_candidate.md`。

## 3. N-D 建模与估计

比较 `classic: E+A N^-alpha+B D^-beta`、`interaction: classic+H N^-alpha D^-beta` 和 `broken-N`。仅 B1 的 {a['manifest']['B1']['rows']} 行用于最小二乘参数估计；按完整 N 组逐一留出，且将较小 N 或早期 D 作为训练集单独测试外推。所有模型对 B2/B4/B5 均不重新拟合，评价是原始零样本预测。

{md_table(nd_comp, formatters={'B1_training_rmse':lambda x:f'{x:.8f}','B1_group_cv_rmse':lambda x:f'{x:.8f}','B2_raw_rmse':lambda x:f'{x:.5f}','B4_raw_rmse':lambda x:f'{x:.5f}','B5_raw_rmse':lambda x:f'{x:.5f}','B1_bic':lambda x:f'{x:.2f}','jacobian_condition':lambda x:f'{x:.2e}'})}

交互与 broken-N 的训练误差略低，但 B1 留 N RMSE 从 {classic.B1_group_cv_rmse:.8f} 增至 {nd.loc['interaction','B1_group_cv_rmse']:.8f}、{nd.loc['broken_N','B1_group_cv_rmse']:.8f}。broken-N 条件数 {nd.loc['broken_N','jacobian_condition']:.2e}，在半数较小 N 训练的高 N 测试上明显失稳。故保留 classic；B2/B4/B5 原始 RMSE {classic.B2_raw_rmse:.4f}/{classic.B4_raw_rmse:.4f}/{classic.B5_raw_rmse:.4f}，外部来源偏移是主要误差，不可用 B1 近乎零的训练误差掩盖。

![问题二 N-D 与来源偏移](figures/p2_nd_source_shift.png)

## 4. Q 建模与复赛

固定重拟合的 classic N-D 参数，在 B6 {a['manifest']['B6']['rows']} 行上分别拟合加性 Q、`D_eff=D Q^gamma`、Q×N、Q×D 与低秩 Q×N×D。B6 按 N 留组；B7 相对 B6 去重后的 {s['B7_new_rows']} 个输入点评价。B8 的标签没有参与任何正常方向 Q 候选拟合和选择。

{md_table(q_comp, formatters={'B6_training_rmse':lambda x:f'{x:.5f}','B6_leave_N_rmse':lambda x:f'{x:.5f}','B7_new_rmse':lambda x:f'{x:.5f}','B6_bic':lambda x:f'{x:.2f}','jacobian_condition':lambda x:f'{x:.2e}'})}

Q×N×D 在 B6 留 N 和 B7 新点分别比 Q×N 改善 {gain_b6:.2f}% 和 {gain_b7:.2f}%，但均低于预注册 5% 门槛。按 B7 的 N 组配对自举，RMSE 改善 95% 区间 [{qnd.B7_vs_champion_ci_low:.5f},{qnd.B7_vs_champion_ci_high:.5f}] 覆盖 0。新增 D 指数在 B6 组自举区间 [{interval_d.ci_low:.4f},{interval_d.ci_high:.4f}]，表明可作机制敏感性；组外收益不够稳健，主模型继续采用 Q×N。

![问题二质量律复赛](figures/p2_quality_tournament.png)

## 5. 最终公式与参数

本轮从原始 B1/B6 重新估计的模型为：

$$
{form}
$$

N、D 都按十亿计。`Q_B=1` 时 Q 项为零，回到 N-D 律。`Q_B` 的训练机制限于 B6/B7；`Delta_p` 由最终问题一模型预测相对相同参考配方的中心化 Loss 差。预测代码 `predict_final.py` 从 `selection.json` 读取参数，对 N、D、Q 做输入范围检查。

![问题二 Q 与 N 响应](figures/p2_quality_by_scale.png)

## 6. B2/B4/B5 来源应用与零样本区分

`source_conditional_holdout.csv` 使用每条 B2 轨迹或 B4/B5 家族早期 20% 已知 Loss 拟合来源截距、仿射或 `log D` 残差修正，再评估后期 80%。此协议消耗目标来源标签，不能放入前述零样本分数中，也不能自动把来源修正乘到 Q/p 项。多种形式在本轮均已查看，以下最优形式只是描述性，不宣称全新封存验证：{', '.join(f"{r.source}={r.protocol}, RMSE {r.rmse:.4f}" for r in source_best.itertuples())}。

{md_table(source_table, formatters={'rmse':lambda x:f'{x:.5f}'})}

完全不使用 B5 标签估计时，从 B2+B4 平均来源偏移迁移到 B5 的 RMSE 为 {a['transfer'].query("protocol == 'B2_B4_mean_intercept_to_heldout_B5'").rmse.iloc[0]:.4f}，而原始 classic 为 {a['transfer'].query("protocol == 'zero_shot_classic'").rmse.iloc[0]:.4f}。来源修正不能作为通用无标签规律。

![问题二有标签来源应用](figures/p2_labeled_source_application.png)

## 7. B8 机制冲突

B6 与 B8 有 {s['B8_mechanism']['B6_B8_exact_overlap']['n']} 个相同 `(N,D,Q_B)` 输入，B8−B6 Loss 平均 {s['B8_mechanism']['B6_B8_exact_overlap']['mean_difference']:.5f}。在各 `(N,D)` 网格中，Q 增加而 Loss 降低的比例：B6={s['B8_mechanism']['B6']['fraction_loss_declines_with_Q']:.0%}，B7={s['B8_mechanism']['B7_full']['fraction_loss_declines_with_Q']:.0%}，B8 新点={s['B8_mechanism']['B8_new']['fraction_loss_declines_with_Q']:.0%}。同点不同标签及相反斜率证明不能把 B8 作为同一确定性质量律的验证通过。B8 包含 `calibrated` 与 `extrapolated` 类型；这些不是独立现实大模型实验。

![问题二 B8 机制诊断](figures/p2_b8_mechanism.png)

## 8. 配方桥、质量量表与下游

最终接口采用 `L(N,D,Q_B,p)=L_NDQ+lambda_p Delta_p(p)`。当前 `lambda_p=1` 是定义参考情景，不是根据 B 数据估出的跨规模参数。`lambda_p_scenarios.csv` 的幂衰减、带底值和上下界仅探索敏感性，所有 `identified=false`。B 数据不含联合 `(N,D,Q,p)` 多规模 Loss 标签，所以不能从问题一的单尺度配方响应或其他论文推断 `lambda_p(N)`。尽管 [Ye 等，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/cc84bfabe6389d8883fc2071c848f62a-Abstract-Conference.html) 研究了数据混合规律，其数据不能代替本题独立锚点。

问题一质量指数 `Q_A` 与 B6/B7 的 `Q_B` 不同，缺乏同一训练样本的成对测量与共同 Loss 锚点。故无法识别 `Q_A→Q_B` 映射，问题三若使用 Q_B 必须明示情景或另取标注。若问题一的最终配方模型发生变化，应重算 `Delta_p`，并顺序重跑问题三与问题四受影响的结果。

## 9. 模型选择、证据等级与复现

最终 N-D 与 Q Winner 分别为 `{s['ND_winner']}` 和 `{s['Q_winner']}`，即原 Model C 经重新挑战保留。最强结论是：B1 同机制经典律、B2/B4/B5 外部 raw 偏移、B6/B7 半合成 Q×N 响应及 B8 机制冲突。较弱结论是 B6 内 Q×N×D 的小 D 指数、来源有标签校准及高 N 等价计算情景。不得写成真实跨来源统一质量因果律或精确 `lambda_p(N)`。

从原始 CSV 运行 `python rebuild/problem2/reproduce.py`，20 个核心输出连续两次逐字节相同，{len(a['verify']['checks'])}/{len(a['verify']['checks'])} 项检查通过；只读核查冻结问题二目录 {a['frozen']['unchanged']}/{a['frozen']['checked']} 个 SHA 未变化。完整模型结果见 `problem2_model_tournament.csv`，论文可用数字见 `paper_key_numbers.csv`。
"""
    (HERE / "问题二最终详解.md").write_text(detail, encoding="utf-8")


def main():
    a = load()
    if not a["verify"]["all_passed"] or not a["reproduction"]["deterministic"] or not a["frozen"]["all_passed"]:
        raise RuntimeError("Core Problem 2 evidence is not verified")
    make_numbers(a)
    plot_cross_source(a)
    plot_quality(a)
    plot_b8(a)
    plot_source_calibration(a)
    plot_quality_scale(a)
    reports(a)
    manifest = pd.DataFrame([
        dict(file="figures/p2_nd_source_shift.png", role="main", caption="N-D same-source group CV and zero-shot source shift", evidence="B1/B2/B4/B5 from nd_scorecard.csv"),
        dict(file="figures/p2_quality_tournament.png", role="main", caption="Q candidate leave-N and B7-new comparison", evidence="B6/B7 from quality_scorecard.csv"),
        dict(file="figures/p2_b8_mechanism.png", role="main", caption="B6-B8 identical-input disagreement and reversed Q direction", evidence="diagnostic only; B6_B8_exact_overlap.csv"),
        dict(file="figures/p2_labeled_source_application.png", role="appendix", caption="Conditional calibration requiring target-source loss labels", evidence="source_conditional_holdout.csv; not zero shot"),
        dict(file="figures/p2_quality_by_scale.png", role="appendix", caption="Selected Q-by-N response inside B6 N/D range", evidence="scenario curve from selection.json"),
    ])
    manifest.to_csv(HERE / "paper_figure_manifest.csv", index=False)
    print("Wrote model card, final explanation, paper numbers, and 5 figures")


if __name__ == "__main__":
    main()
