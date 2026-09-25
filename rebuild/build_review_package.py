"""Assemble a reviewable selected-chain package while Final Freeze is unmet.

All paper numbers and figures are copied from or computed from selected-chain
outputs. Historical baseline directories are never written by this script.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "rebuild"
W = R / "winner_chain"
O = R / "final_review"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    run = json.loads((W / "one_click_run_summary.json").read_text(encoding="utf-8"))
    if not (run["all_passed"] and not run["resumed"] and run["executed_this_invocation"] == 25):
        raise RuntimeError("A clean 25-stage one-command run is required")
    checks = json.loads((R / "problem4" / "selected_scope_verification.json").read_text(encoding="utf-8"))
    selected_status = json.loads((W / "selected_rerun_status.json").read_text(encoding="utf-8"))
    if any(row["exit_code"] for row in selected_status) or selected_status[-1]["stage"] != "problem4_selected_selection":
        raise RuntimeError("Selected-source retrospective challenges are incomplete")
    if checks["final_freeze_achieved"] or checks["original_p4_verification_passed"] != 31:
        raise RuntimeError("P4 evidence status changed; review it before packaging")
    for sub in ("outputs/tables", "outputs/figures", "outputs/models", "verification"):
        (O / sub).mkdir(parents=True, exist_ok=True)
    provenance = []

    def take(source: Path, relative: str) -> None:
        destination = O / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        provenance.append(dict(source=str(source.relative_to(ROOT)),
                               packaged_as=relative, sha256=digest(source),
                               bytes=source.stat().st_size))

    for q in range(1, 5):
        selected_docs = W / "rebuild" / f"problem{q}" if q < 4 else R / "problem4"
        take(selected_docs / f"问题{('一','二','三','四')[q-1]}最终详解.md",
             f"问题{('一','二','三','四')[q-1]}最终详解.md")
        card_dir = W / "rebuild" / f"problem{q}"
        take(card_dir / "model_card.md", f"verification/problem{q}_model_card.md")
    take(W / "rebuild/problem1/paper_key_numbers.csv", "outputs/tables/problem1_paper_numbers.csv")
    take(W / "rebuild/problem1/metrics_by_scale.csv", "outputs/tables/problem1_scale_metrics.csv")
    take(W / "rebuild/problem1/per_domain_paired_bootstrap.csv", "outputs/tables/problem1_domain_bootstrap.csv")
    take(W / "rebuild/problem2/paper_key_numbers.csv", "outputs/tables/problem2_paper_numbers.csv")
    take(W / "rebuild/problem2/nd_scorecard.csv", "outputs/tables/problem2_nd_scorecard.csv")
    take(W / "rebuild/problem2/quality_scorecard.csv", "outputs/tables/problem2_quality_scorecard.csv")
    take(W / "rebuild/problem3/problem3_decision_tournament.csv", "outputs/tables/problem3_decision_tournament.csv")
    take(W / "rebuild/problem4/problem4_model_tournament.csv", "outputs/tables/problem4_model_tournament.csv")
    take(W / "problem4/outputs/tables/quarterly_frontier.csv", "outputs/tables/problem4_quarterly_frontier.csv")
    take(W / "problem4/outputs/tables/frontier_forecast_12m_24m.csv", "outputs/tables/problem4_scenarios.csv")
    take(R / "problem4/p4_scenario_envelope.csv", "outputs/tables/problem4_scenario_envelope.csv")
    take(W / "problem4/outputs/tables/verification_results.json", "verification/problem4_original_checks.json")
    take(R / "problem4/selected_scope_verification.json", "verification/problem4_selected_scope_checks.json")
    take(W / "selected_lineage.json", "verification/selected_lineage.json")
    take(W / "selected_rerun_status.json", "verification/selected_rerun_status.json")
    take(W / "one_click_run_summary.json", "verification/one_click_run_summary.json")
    take(ROOT / "model_baseline_manifest.json", "verification/model_baseline_manifest.json")
    take(ROOT / "baseline_scorecard.csv", "verification/baseline_scorecard.csv")
    take(W / "rebuild/problem1/selected_model.joblib", "outputs/models/problem1_selected.joblib")
    take(W / "problem2_v2/outputs/models/recommended_model.joblib", "outputs/models/problem2_selected.joblib")

    figures = {
        "mixture_effect.png": W / "rebuild/problem1/figures/one_b_domain_tradeoffs.png",
        "scaling_law.png": W / "rebuild/problem2/figures/p2_nd_source_shift.png",
        "quality_scale.png": W / "rebuild/problem2/figures/p2_quality_by_scale.png",
        "regime_diagram.png": W / "rebuild/problem3/figures/p3_evidence_levels.png",
        "trust_region.png": W / "rebuild/problem3/figures/p3_cap_tradeoff.png",
        "ability_frontier.png": W / "problem4/outputs/figures/open_frontier_over_time.png",
        "scale_technology_exploratory.png": W / "problem4/outputs/figures/scale_vs_technology_contribution.png",
        "forecast_scenarios_exploratory.png": R / "problem4/p4_scenario_envelope.png",
    }
    for name, source in figures.items():
        take(source, f"outputs/figures/{name}")
    fig, ax = plt.subplots(figsize=(11, 2.5), layout="constrained")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1)
    ax.axis("off")
    labels = ["P1: Q and mixture", "P2: N-D-Q Loss", "P3: robust decision", "P4: ability / scenario"]
    for i, label in enumerate(labels):
        ax.text(i+.5, .5, label, ha="center", va="center", fontsize=11,
                bbox=dict(boxstyle="round,pad=.7", fc="#EDF4FA", ec="#2878B5"))
        if i < 3:
            ax.annotate("", xy=(i+.92, .5), xytext=(i+.8, .5),
                        arrowprops=dict(arrowstyle="->", lw=1.8, color="#385A6A"))
    fig.savefig(O / "outputs/figures/overall_framework.png", dpi=300)
    plt.close(fig)
    provenance.append(dict(source="generated from selected lineage", packaged_as="outputs/figures/overall_framework.png",
                           sha256=digest(O / "outputs/figures/overall_framework.png"),
                           bytes=(O / "outputs/figures/overall_framework.png").stat().st_size))

    p1 = pd.read_csv(O / "outputs/tables/problem1_scale_metrics.csv").set_index(["dataset", "family"])
    p2 = json.loads((W / "rebuild/problem2/selection.json").read_text(encoding="utf-8"))
    p2nd = pd.read_csv(O / "outputs/tables/problem2_nd_scorecard.csv").set_index("model")
    p2q = pd.read_csv(O / "outputs/tables/problem2_quality_scorecard.csv").set_index("model")
    p3 = pd.read_csv(O / "outputs/tables/problem3_decision_tournament.csv")
    p3row = p3[(p3.budget == 1e24) & (p3.cap == "3x") &
               (p3.policy == "minimax_regret") & (p3.split == "heldout")].iloc[0]
    p4front = pd.read_csv(O / "outputs/tables/problem4_quarterly_frontier.csv")
    openfront = p4front[p4front.subset == "open_w1"].sort_values("quarter")
    p4env = pd.read_csv(O / "outputs/tables/problem4_scenario_envelope.csv").set_index("horizon_months")
    lineage = json.loads((W / "selected_lineage.json").read_text(encoding="utf-8"))
    original = json.loads((W / "problem4/outputs/tables/verification_results.json").read_text(encoding="utf-8"))
    if list(openfront.n) != [22, 16, 5, 2]:
        raise AssertionError("P4 strict frontier sample counts changed")

    selection = pd.DataFrame([
        dict(problem="P1", baseline="V2 polynomial ridge mixture", selected="ALR multitask ElasticNet",
             replacement="yes", primary_evidence="nested OOF and pooled external RMSE improve", limitation="four 1B domains worsen"),
        dict(problem="P2", baseline="classic N-D plus Q by N", selected="classic N-D plus Q by N",
             replacement="no", primary_evidence="no challenger passes preregistered external gates", limitation="B8 mechanism shift"),
        dict(problem="P3", baseline="conditional minimax regret", selected="conditional minimax regret",
             replacement="no", primary_evidence="fair nominal and robust alternatives do not materially improve", limitation="3x cap conditional; high-budget extrapolation"),
        dict(problem="P4", baseline="six-benchmark index, W1 p95, bounded logistic, structural scenarios",
             selected="same core; compute-only historical comparator and persistence risk reference",
             replacement="no validated long-horizon upgrade", primary_evidence="grouped CV and short rolling-origin checks",
             limitation="3 original evidence gates fail; no Final Freeze"),
    ])
    selection.to_csv(O / "model_replacement_log.csv", index=False, encoding="utf-8-sig")
    freeze = {
        "four_question_tournaments_complete": True,
        "selected_downstream_lineage_verified": bool(lineage["P2_embeds_selected_P1_coefficients"]),
        "selected_source_challenges_rerun": all(row["exit_code"] == 0 for row in selected_status),
        "paper_numbers_copied_from_selected_chain_csv": True,
        "figures_copied_from_selected_chain_or_selected_csv": True,
        "prior_25_stage_one_command_clean_run_passed": bool(run["all_passed"] and not run["resumed"]),
        "current_extended_one_command_path_clean_tested": False,
        "problem1_verification_passed": True,
        "problem2_verification_passed": True,
        "problem3_verification_passed": True,
        "problem4_original_verification_passed": all(v["passed"] for v in original.values()),
        "problem4_original_checks_passed": sum(v["passed"] for v in original.values()),
        "problem4_original_checks_total": len(original),
        "final_freeze_passed": False,
        "archive_created": False,
        "final_project_created": False,
        "original_models_deleted_or_replaced": False,
    }
    (O / "verification/final_freeze_checklist.json").write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8")

    report = f"""# 最终模型复赛技术审阅报告

本报告是 **Final Freeze 前的完整审阅件**。冻结的原始模型没有被替换、移动或删除；P4 原验证为 {freeze['problem4_original_checks_passed']}/{freeze['problem4_original_checks_total']}，故按任务规定尚不能建立 `final_project/` 或归档历史模型。全部数字来自本目录 `outputs/tables/` 或相应哈希记录。

## 1 Unified Framework

四问按 Q/配比 → N-D-Q Loss → 算力约束决策 → 能力前沿与情景链接。P1 的选定 artifact 哈希 `{lineage['P1_selected_sha256']}`；P2 artifact 内嵌其系数，P3 接口哈希 `{lineage['P3_interface_sha256']}`，P4 摘要哈希 `{lineage['P4_summary_sha256']}`。见 `verification/selected_lineage.json`。

## 2 Problem1

原 Champion 为 V2 多项式 Ridge 配比模型；最终选定 ALR 二阶特征、多任务 ElasticNet。嵌套 OOF RMSE {p1.loc[('train_nested_oof','v2_champion'),'rmse']:.6f}→{p1.loc[('train_nested_oof','alr_elasticnet'),'rmse']:.6f}；1M 外部 {p1.loc[('test_1m','v2_champion'),'rmse']:.6f}→{p1.loc[('test_1m','alr_elasticnet'),'rmse']:.6f}；60M {p1.loc[('test_60m','v2_champion'),'rmse']:.6f}→{p1.loc[('test_60m','alr_elasticnet'),'rmse']:.6f}；1B {p1.loc[('test_1B','v2_champion'),'rmse']:.6f}→{p1.loc[('test_1B','alr_elasticnet'),'rmse']:.6f}。这是既有测试集的回顾性比较，1B 有四个领域显著变差。Q 指数没有人工真值，保留四组等权。

## 3 Problem2

保留经典 N-D 标度律和 Q×N 机制：`E,A,alpha,B,beta={','.join(f'{x:.6f}' for x in p2['ND_parameters'])}`；`G,kappa,eta_N={','.join(f'{x:.6f}' for x in p2['Q_parameters'])}`。B1 组外 RMSE {p2nd.loc['classic','B1_group_cv_rmse']:.6f}；B6 leave-N {p2q.loc['Q_by_N','B6_leave_N_rmse']:.6f}，B7 新点 {p2q.loc['Q_by_N','B7_new_rmse']:.6f}。更复杂交互与破折线律未过外部/识别门槛。B8 同输入存在冲突机制，不作为同分布验证。`lambda_p=1` 仅情景约定，Q_A→Q_B 未校准。

## 4 Problem3

保留条件式 minimax regret，发布 1×、2×、3×、5×、free 的 Pareto 剖面。于 10^24 FLOPs、3× 上限，N={p3row.N_B:.4f}B、D={p3row.D_B:.4f}B、Q={p3row.Q_B:.4f}，名义 Loss={p3row.nominal_loss:.5f}，相对同一 free oracle 的留出平均后悔={p3row.mean_free_regret:.6f}。在同一可行域重新优化后，名义政策与 minimax 近乎持平；3× 是明确条件情景，不是唯一经证实上限。

## 5 Problem4

能力定义为六项基准等权均值；严格开放权重季度 p95 的样本数为 {', '.join(str(int(x)) for x in openfront.n)}。末季 n=2，不作方向断言。历史仅算力模型的组外误差低于加入时间项；C6 保留单调有界 logistic 桥接。12/24 个月九情景中位数包络分别为 [{p4env.loc[12,'scenario_median_min']:.2f}, {p4env.loc[12,'scenario_median_max']:.2f}]、[{p4env.loc[24,'scenario_median_min']:.2f}, {p4env.loc[24,'scenario_median_max']:.2f}]，原点为 2025-01-31，且缺少对应期限回测。这些不是预测置信区间。

## 6 Cross-question Interface

隔离 `winner_chain` 先完成四问历史基线和候选复赛，再注入 P1 所选模型并重跑 P2→P4。P4 中 P1 配比增量由 P3 诊断表动态读取，消除了旧代码硬编码沿用原 P1 数值的问题。所有下游产物与哈希见 `verification/selected_lineage.json`。

## 7 Validation

25 个顶层阶段在补充内层审阅表前曾单命令运行通过；其后选定来源的 P3/P4 审阅表从内层断点续跑完成，状态见 `verification/selected_rerun_status.json`。扩展后的完整入口未再从头执行，因此不把它记作已通过的干净一键复现。P1、P2、P3 的复赛验证通过。P4 原 34 项中 31 项通过；12、24 个月直接回测与历史分解方向一致性三项未通过。`verification/problem4_selected_scope_checks.json` 仅证明情景标识、范围、哈希等计算完整性，不能替代缺失的实证检验。因此 Final Freeze=false。

## 8 Robustness

P1 报告 1B 分领域损害与 10B/70B 估算弱点；P2 报告 B8 机制差异和外推；P3 报告共同 free oracle 与上限敏感性；P4 报告样本数门槛、桥接家族隔离、短期持平基线与长期情景范围。相关细节见四份问题详解和 `model_replacement_log.csv`。

## 9 Evidence Hierarchy

独立人工标签不存在；既有测试集已被查看，复赛增益属于回顾性证据。P3 留出参数由同一训练数据 bootstrap 得到，不是新外部实验。P4 直接历史只有四季度、最近严格前沿 n=2，不能把 12/24 月情景、结构分解或条件 bootstrap 区间提升为已校准结论。

## 10 Final Conclusions

可确认的模型替换只有 P1 配比模型。P2 和 P3 保留原 Champion；P4 保留解释受限的核心定义与桥接，并新增更简单的历史对照及短期风险参照。原工程未清理，`archive_pre_final/` 和 `final_project/` 尚未创建。要解除 Final Freeze，需要具备可核验的 12/24 月历史目标，且解决或限定历史分解与前沿方向冲突；在此之前正式论文只能使用本审阅件中标明证据等级的数字。
"""
    (O / "最终项目技术报告.md").write_text(report, encoding="utf-8")

    answers = """# 用户要求的 24 项结论

1. P1 原 Champion：V2 多项式 Ridge 配比模型。
2. P1 Challenger：ALR 多任务 ElasticNet、ALR PLS、ILR Ridge、CLR spline，以及 Q 的 PCA/FA/熵权/秩聚合敏感性。
3. P1 Winner：ALR 多任务 ElasticNet；Q 四组等权仍保留。
4. 原因：嵌套 OOF 和 1M/1B pooled 外部 RMSE 改善，预设总体门槛通过；1B 四域损害与 10B/70B 估算弱点须并列披露。
5. P2：原经典 N-D+Q×N；挑战交互、破折线、有效 token、Q×D、Q×N×D 等；原模型保留，复杂候选未过外部及识别门槛。
6. P3：原条件式 minimax regret；挑战公平重优化名义、平均后悔、CVaR90 和最坏 Loss；原框架保留，3× 只作条件上限。
7. P4：六基准均值、严格 W1 p95、有界 logistic 和条件情景核心保留；PCA/稳健指数、分位前沿、复杂桥接和长期预测没有取得可确认 Winner。历史预测对照改用 compute-only，持平作为短期风险参照。
8. 旧模型被替代：P1 V2 配比模型在隔离选定链内被 ALR 模型替代；原目录未覆盖。
9. 经挑战保留：P2 经典 N-D+Q×N、P3 条件 minimax、P4 能力主定义及 logistic 桥接。
10. 拒绝的复杂模型：P1 CLR spline 及弱外推候选、P2 交互/破折线、P3 CVaR/平均目标、P4 PCA/复杂桥接/分位前沿和未验证 ensemble。
11. 经外部验证改善：P1 既有 1M/1B pooled 测试集；均为回顾性，不是新盲测。
12. 只改善训练或局部拟合：P2/P4 部分复杂候选；它们未在预设独立口径上稳定改善。
13. 稳定性改善：P1 正确嵌套 OOF；P2 简约式保留；P3 公平名义对照和上限剖面；P4 样本数守门与短期持平参照。
14. 暂无进一步有据优化：P1 的 Q 权重、P2 Q_A→Q_B 与 p 桥、P3 唯一高预算上限、P4 长期预测及历史贡献份额。
15. 四问完全重跑：隔离链中历史基线与复赛均重跑，选定 P1 后 P2→P4 再重跑。
16. 下游来自最终上游：是；P2 内嵌所选 P1 系数，P4 的配比增量动态来自选定链，哈希见 lineage。
17. verification 全部通过：否；P4 原验证 31/34。
18. Final Project 独立一键运行：尚无 Final Project；先前隔离链 25 阶段单命令成功，扩展后的完整入口只做了断点续跑，未完成干净一键验证。
19. archive 完整：否；Final Freeze 不满足，尚未归档。
20. 删除/替换原模型：没有。
21. 删除依据：当前不存在；须先满足全部验证、归档和复现条件。
22. 真正提升：P1 pooled RMSE、选定上游传播、候选公平比较与证据边界；没有宣称 P4 长期预测准确性提升。
23. 论文数字：使用本包 `outputs/tables/`，尤其 P1、P2 key numbers，P3 tournament，P4 strict frontier 和 scenario envelope；P4 长期数字标为探索性。
24. 论文核心创新：以配比组成回归、质量条件标度律、证据分级的稳健决策和严格样本门槛的能力情景构成可追溯四问链；创新表述不得超出相应验证等级。
"""
    (O / "24项结论.md").write_text(answers, encoding="utf-8")
    readme = """# 复赛审阅包

本目录是 Final Freeze 之前的审阅件，不是可发布的 `final_project/`。原始六个模型目录未移动或删除。

- `最终项目技术报告.md`：10 节统一技术报告。
- `24项结论.md`：对任务最后 24 项问题逐项回答。
- `问题一至四最终详解.md`：选定模型及证据边界。
- `outputs/tables/`：从完成后的选定链 CSV 复制的论文数值。
- `outputs/figures/`：选定链重新生成的图，分解和预测图均标为探索性。
- `verification/final_freeze_checklist.json`：明确记录尚未满足的 P4 证据门槛。
- `artifact_manifest.csv`：每个打包产物的来源、SHA256 与大小。

隔离的一键入口在 `../winner_chain/run_all.py`；25 个顶层阶段的原始日志仍保留于 `../winner_chain/one_click_*.log`，扩展后的内层审阅表通过断点续跑完成。复赛结果和详细模型选择记录保留于 `../problem1/` 至 `../problem4/`。新增历史实测目标后，应重新挑战 P4 并更新本包，不能直接改写验证状态。
"""
    (O / "README.md").write_text(readme, encoding="utf-8")
    pd.DataFrame(provenance).to_csv(O / "artifact_manifest.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote review package: {O}; {len(provenance)} hashed artifacts; Final Freeze=false")


if __name__ == "__main__":
    main()
