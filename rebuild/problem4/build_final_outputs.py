"""Create P4 selected-scope outputs from the completed isolated winner chain.

This does not rewrite the frozen project or turn unavailable backtests into passes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CHAIN = ROOT / "rebuild" / "winner_chain"
P4 = CHAIN / "problem4" / "outputs" / "tables"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    status = json.loads((CHAIN / "one_click_status.json").read_text(encoding="utf-8"))
    if not status or any(row["exit_code"] for row in status):
        raise RuntimeError("Winner-chain reproduction has not completed")
    if status[-1]["stage"] != "selected_dependency_chain":
        raise RuntimeError("Selected dependency chain is not the final stage")
    lineage = json.loads((CHAIN / "selected_lineage.json").read_text(encoding="utf-8"))
    summary = json.loads((P4 / "final_project_summary.json").read_text(encoding="utf-8"))
    original_checks = json.loads((P4 / "verification_results.json").read_text(encoding="utf-8"))
    forecast = pd.read_csv(P4 / "frontier_forecast_12m_24m.csv")
    strict_frontier = pd.read_csv(P4 / "quarterly_frontier.csv")
    short = pd.read_csv(HERE / "short_term_scorecard.csv")
    historical = pd.read_csv(HERE / "historical_scorecard.csv")
    bridge = pd.read_csv(HERE / "bridge_scorecard.csv")
    p1delta = float(pd.read_csv(CHAIN / "problem3_v2" / "outputs" / "tables" /
                                 "mixture_support_diagnostics.csv").query(
                                     "strategy == 'P1_observed_support'")["delta_p"].iloc[0])
    failures = [name for name, result in original_checks.items() if not result["passed"]]
    expected_failures = {
        "12 month direct backtest", "24 month direct backtest",
        "Historical decomposition agrees with observed frontier",
    }
    checks = {
        "one_click_all_stages_exit_zero": all(row["exit_code"] == 0 for row in status),
        "selected_p1_embedded_in_p2": lineage["P2_embeds_selected_P1_coefficients"],
        "p4_summary_hash_matches_lineage": sha(P4 / "final_project_summary.json") == lineage["P4_summary_sha256"],
        "p4_uses_selected_p1_delta": abs(summary["problem1"]["p1_delta_loss"] - p1delta) < 1e-10,
        "original_three_evidence_gaps_preserved": set(failures) == expected_failures,
        "all_forecasts_scenario_labeled": forecast.status.str.contains("exploratory").all(),
        "all_forecasts_sparse_anchor_flagged": (~forecast.ability_anchor_meets_n5).all(),
        "all_forecasts_bounded_and_ordered": (
            forecast[["p95_low", "p80_low", "p50_low", "median_ability",
                      "p50_high", "p80_high", "p95_high"]].apply(
                          lambda row: 0 <= row.min() and row.max() <= 100 and
                          row.p95_low <= row.p80_low <= row.p50_low <= row.median_ability <=
                          row.p50_high <= row.p80_high <= row.p95_high, axis=1).all()),
        "two_horizons_nine_scenarios_each": forecast.groupby("horizon_months").size().to_dict() == {12: 9, 24: 9},
    }
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        raise AssertionError(checks)
    (HERE / "selected_scope_verification.json").write_text(json.dumps({
        "computational_integrity": checks,
        "original_p4_verification_passed": sum(v["passed"] for v in original_checks.values()),
        "original_p4_verification_total": len(original_checks),
        "unmet_evidence_gates": failures,
        "final_freeze_achieved": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    envelope = (forecast.groupby("horizon_months", as_index=False)
                .agg(target_date=("target_date", "first"),
                     scenario_median_min=("median_ability", "min"),
                     scenario_median_max=("median_ability", "max"),
                     conditional_p95_min=("p95_low", "min"),
                     conditional_p95_max=("p95_high", "max"),
                     scenario_count=("median_ability", "size"),
                     anchor_n=("ability_anchor_n", "first")))
    envelope["interpretation"] = "scenario span; conditional intervals are uncalibrated"
    envelope.to_csv(HERE / "p4_scenario_envelope.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7.0, 4.2), layout="constrained")
    x = [12, 24]
    lo = envelope.scenario_median_min.to_numpy()
    hi = envelope.scenario_median_max.to_numpy()
    mid = (lo + hi) / 2
    ax.errorbar(x, mid, yerr=[mid-lo, hi-mid], fmt="o", markersize=7,
                capsize=9, elinewidth=6, color="#2878B5", label="Nine scenario median span")
    ax.set(xlabel="Months after 2025-01-31 data origin", ylabel="Ability index (0-100)",
           xticks=x, ylim=(0, 100), title="Exploratory scenario envelope")
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    fig.savefig(HERE / "p4_scenario_envelope.png", dpi=300)
    fig.savefig(HERE / "p4_scenario_envelope.pdf")
    plt.close(fig)

    r12, r24 = envelope.itertuples(index=False)
    pers = short[(short.model == "persistence") & (short.horizon_quarters == 1)].iloc[0]
    direct = short[(short.model == "existing_direct_time") & (short.horizon_quarters == 1)].iloc[0]
    hist = historical.set_index("kind")
    br = bridge.set_index("kind")
    detail = f"""# 问题四：选定模型与证据边界

## 1 定义与数据

能力指数为 IFEval、BBH、MATH、GPQA、MUSR、MMLU-PRO 六项完整分数的等权均值。C1 有 4576 条榜单记录；严格核对 C4 发布实体及开放权重后，W1 主前沿有 45 个独立模型。C8 逐任务相关性是同源任务一致性，不能充当独立能力真值。

## 2 历史前沿

采用严格开放权重样本每季度能力 p95。四个季度样本数分别为 22、16、5、2；2025Q1 未达到预先设置的 n≥5 解释门槛。因此仅发布前沿的描述数值，不能断言最近季度涨跌。宽松继承匹配的 C2 样本给出相反方向，只作敏感性。

## 3 规模与时间

在同 32 个发布、13 个组织的分组交叉验证中，仅算力模型 RMSE={hist.loc['compute_only','rmse']:.3f}，算力加时间结构模型 RMSE={hist.loc['structural','rmse']:.3f}。因此历史预测对照用仅算力模型。结构模型可按固定假设生成算力/时间 Shapley 分解，但其隐含前沿变化 {summary['problem4']['historical_model_scale_points']+summary['problem4']['historical_model_technology_points']:.2f} 分，与观测变化 {summary['problem4']['observed_frontier_change']:.2f} 分方向相反；历史贡献份额不作为实证结论。

## 4 Loss 到能力的桥接

采用高可比权重 1、中可比权重 0.35 的单调有界 logistic 桥接。C6 的 75 条记录来自 23 个家族，其中高可比仅 7 条且同属一个家族。家族隔离五折 RMSE={br.loc['logistic','oof_rmse']:.3f} 分。isotonic 的 RMSE={br.loc['isotonic','oof_rmse']:.3f}，但增益的家族重抽样区间跨零，且端点平台不稳定；保留较简单 logistic。桥接输入须标注 Loss 来源及可比性。

## 5 决策与短期参照

问题三给出 E0/E1/E2 条件策略，默认展示 E1 的 3× 限制，但不能称其为已证明的唯一高预算最优。三个月短期目标只有两个，其中一个目标季度 n=2。持平基线 MAE={pers.mae:.3f}，直接时间趋势 MAE={direct.mae:.3f}；这支持把持平作为风险参照，不能据此升级长期预测器。

## 6 12/24 个月情景

原点为 2025-01-31，目标日期为 2026-01-31 和 2027-01-31。每个期限为三种算力增长 × 三种技术时间项设定，共九条。12 个月情景中位数范围 [{r12.scenario_median_min:.2f}, {r12.scenario_median_max:.2f}]；24 个月为 [{r24.scenario_median_min:.2f}, {r24.scenario_median_max:.2f}]。这是**情景包络**，不是经 12/24 个月回测校准的预测区间。各行条件 bootstrap 区间及锚点 n=2 的警示见 `../winner_chain/problem4/outputs/tables/frontier_forecast_12m_24m.csv`。以当前日期理解，2026-01-31 已是历史日期；本次严格使用冻结的 2025-01-31 数据截点，不把后见资料混入回测。

## 7 上下游与核验

P4 汇总读取选定 P1 配比增量 {p1delta:.12f}，与 P3 接口诊断逐字一致；P2 artifact 内嵌所选 P1 系数。P4 原验证 {len(original_checks)-len(failures)}/{len(original_checks)} 通过，未通过的是 12 个月直接回测、24 个月直接回测、历史分解与观测前沿一致性。这些是当前证据不足，不被改写为通过。计算完整性检查见 `selected_scope_verification.json`。

## 8 结论

最终可防守的分析是六基准能力定义、严格开放权重前沿、单调有界桥接、分组验证误差与情景范围。长期未来能力和算力/技术分量只能作为有明确假设的探索性输出。Final Freeze 尚未完成。
"""
    (HERE / "问题四最终详解.md").write_text(detail, encoding="utf-8")
    print("P4 selected-scope outputs written; Final Freeze remains false")


if __name__ == "__main__":
    main()
