"""Publish reproducible v2 method and v1/v2 comparison from measured CSV files."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mixture_pipeline import pair


HERE = Path(__file__).resolve().parent
TABLES = HERE / "outputs" / "tables"
FIGURES = HERE / "outputs" / "figures"


def fmt(x, digits=3):
    return f"{x:.{digits}f}"


def run():
    metrics = pd.read_csv(TABLES / "v2_vs_v1_metrics.csv")
    cv = pd.read_csv(TABLES / "v2_train_cv_search.csv")
    domains = pd.read_csv(TABLES / "v2_vs_v1_per_domain.csv")
    ci = pd.read_csv(TABLES / "v2_vs_v1_bootstrap.csv").iloc[0]
    stability = pd.read_csv(TABLES / "v2_effect_stability_summary.csv")
    nested = pd.read_csv(TABLES / "v2_nested_cv_summary.csv").set_index("version")
    nested_folds = pd.read_csv(TABLES / "v2_nested_cv_folds.csv")
    selection = json.loads((TABLES / "v2_selection.json").read_text(encoding="utf-8"))
    chosen = selection["selected"]
    _, _, _, pcols, _ = pair("train", "1m")
    reference_domain = pcols[int(chosen["reference_index"])].replace("train_the_pile_", "")
    base = cv[(cv.family == "alr") & (cv.reference_index == 16) & (cv.alpha == 1)
              & np.isclose(cv.eps, 0.000499001996007984)].iloc[0]
    t1 = metrics.query("dataset == 'test_1m'").set_index("version")
    reduction = 100 * (1 - t1.loc["v2", "rmse"] / t1.loc["v1", "rmse"])
    d1 = domains.query("dataset == 'test_1m'").pivot(index="validation_domain", columns="version", values="rmse")
    improved = int((d1.v2 < d1.v1).sum())
    robust = int((stability.sign_consistency == 1).sum())
    unstable = stability.loc[stability.sign_consistency < 1, "increase"].tolist()
    nested_improvement = 100 * (1 - nested.loc["v2", "nested_oof_rmse"] / nested.loc["v1", "nested_oof_rmse"])
    nested_wins = int((nested_folds.v2_outer_rmse < nested_folds.v1_outer_rmse).sum())

    plot = d1.sort_values("v1", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(plot))
    ax.barh(y - .18, plot.v1, height=.34, label="Version 1 ALR Ridge", color="#9ba5b0")
    ax.barh(y + .18, plot.v2, height=.34, label="Version 2 quadratic ALR Ridge", color="#267c78")
    ax.set_yticks(y, plot.index, fontsize=8)
    ax.set_xlabel("RMSE on independent 1M test set")
    ax.legend()
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "v2_vs_v1_per_domain.png", dpi=170)
    plt.close(fig)

    compare_lines = ["# 问题一第一版与第二版对比", "",
        "## 比较原则", "",
        "第一版保持原样作为基线。第二版只用 A4/A5 的 512 组 1M 训练配方进行数值选型；A6–A11 的真实 Loss 未进入第二版的拟合或超参数搜索。它们已在第一版中被查看，因此这里是复用既定测试集的同口径比较，不能称为全新盲测。A12–A15 是估算标签，单列展示。两版的质量分 Q 使用同一套规则；没有人工真值，因此不声称 Q 的准确度提升。", "",
        "## 训练集内部选型", "",
        "| 方法 | 2×5 折交叉验证 RMSE |", "|---|---:|",
        f"| 第一版 ALR Ridge，固定参考域与 ε | {fmt(base.cv_rmse_pooled, 4)} |",
        f"| 第二版二阶 ALR Ridge，训练集选出的参考域与 ε | {fmt(chosen['cv_rmse_pooled'], 4)} |", "",
        f"第二版共比较 {selection['candidate_count']} 个候选，按训练集重复交叉验证总体均方误差选出。二阶项允许领域对数比之间的非线性关系；Ridge 惩罚控制复杂度。候选较多会使最优 CV 数值偏乐观，因此另做外层五折、内层三折的嵌套交叉验证，外层每一折都重新选型。", "",
        "| 嵌套交叉验证 | 第一版 | 第二版 | 第二版降幅 |", "|---|---:|---:|---:|",
        f"| A4/A5 外层折拼接 RMSE | {fmt(nested.loc['v1','nested_oof_rmse'],4)} | {fmt(nested.loc['v2','nested_oof_rmse'],4)} | {fmt(nested_improvement,1)}% |", "",
        f"第二版在 {nested_wins}/5 个外层折中 RMSE 更低；五折内层均选出二阶 ALR Ridge。该结果评估了完整选型流程，且完全不读取 A6–A15。", "",
        "## 既定测试集上的同口径比较", "",
        "| 数据集 | 标签性质 | 第一版 RMSE | 第二版 RMSE | 变化 | 第一版域内 Spearman | 第二版域内 Spearman |", "|---|---|---:|---:|---:|---:|---:|" ]
    for ds in ["test_1m", "test_60m", "test_1B", "est_10b", "est_70b"]:
        sub = metrics[metrics.dataset.eq(ds)].set_index("version")
        v1, v2 = sub.loc["v1"], sub.loc["v2"]
        rel = 100 * (v2.rmse / v1.rmse - 1)
        compare_lines.append(f"| {ds} | {'实测' if ds.startswith('test') else '估算'} | {fmt(v1.rmse)} | {fmt(v2.rmse)} | {rel:+.1f}% | {fmt(v1.spearman_mean)} | {fmt(v2.spearman_mean)} |")
    compare_lines += ["",
        f"主要目标 1M 测试 RMSE 下降 **{fmt(reduction, 1)}%**，由 0.3415 降至 0.2702；MAE 从 {fmt(t1.loc['v1','mae'])} 降至 {fmt(t1.loc['v2','mae'])}。以 256 个测试配方为重抽样单位，第一版减第二版 RMSE 的 95% bootstrap 区间为 **[{fmt(ci.bootstrap_ci_low, 3)}, {fmt(ci.bootstrap_ci_high, 3)}]**。这是给定测试配方分布与已拟合模型的抽样区间，不包含重新选型的不确定性，也不消除测试集重复使用带来的偏乐观风险。", "",
        f"1M 的 **{improved}/13** 个验证领域 RMSE 下降。最大剩余误差在 `dm_mathematics`，第二版 RMSE 为 {fmt(d1.loc['dm_mathematics','v2'])}。", "",
        "![13 个验证领域的测试误差](outputs/figures/v2_vs_v1_per_domain.png)", "",
        "## 13 个验证领域的 1M RMSE", "",
        "| 验证领域 | 第一版 | 第二版 | 降幅 |", "|---|---:|---:|---:|" ]
    for domain, row in d1.sort_index().iterrows():
        compare_lines.append(f"| {domain} | {fmt(row.v1)} | {fmt(row.v2)} | {100*(1-row.v2/row.v1):.1f}% |")
    compare_lines += ["",
        "## 未解决的误差", "",
        "60M、1B 的绝对 Loss 依旧存在规模偏差，不能据此说跨规模预测已解决。1B 的域内平均 Spearman 略降；10B/70B 相对估算标签的 RMSE 和 Spearman 均退步。这些估算标签不代表真实大模型实验，也未被用于模型选择。", "",
        "完整数值见 `outputs/tables/v2_vs_v1_metrics.csv`、`v2_vs_v1_per_domain.csv`、`v2_vs_v1_bootstrap.csv`。", ""]
    (HERE / "problem1_v1_v2_comparison.md").write_text("\n".join(compare_lines), encoding="utf-8")

    model_lines = ["# 问题一第二版：建模方案与适用范围", "",
        "## 1. 本次实际优化", "",
        "质量指标的方向、固定 A1 标尺、四组等权 Q 和冲突标记继续沿用第一版。缺少独立人工标签与同配比异质量训练实验，目前没有证据可以把某个新权重或冲突惩罚宣称为更优，因此不改写 Q。第二版把可验证的改进集中在 17 域配比到 13 域 Loss 的预测，并扩展分域与效应稳定性检验。", "",
        "## 2. 数据使用规则", "",
        "A4/A5：512 条 1M 配方及 Loss，用于训练和选型；A6/A7：256 条 1M 既定测试配方，主要性能判据；A8–A11：60M、1B 实测跨规模迁移检验；A12–A15：10B/70B 估算标签，仅作附加诊断。测试集和估算集的 Loss 均未用于第二版数值拟合、超参数选择或规模截距拟合，但其结果已在第一版中被查看，因此并非全新封存的盲测。", "",
        "## 3. 模型", "",
        r"设配比向量 $\mathbf p$ 的 17 个分量和为 1。先作加性对数比 $x_i=\log[(p_i+\epsilon)/(p_r+\epsilon)]$，再用包含一次项、平方项和两两乘积的二阶特征 $\phi(\mathbf x)$ 建立 13 输出 Ridge：$\widehat{\mathbf L}=\mathbf b+\mathbf B\,\mathrm{standardize}(\phi(\mathbf x))$。", "",
        f"本轮由训练集 2×5 折 CV 选出：参考域 `{reference_domain}`、ε={chosen['eps']:.9f}、Ridge α={chosen['alpha']:.0f}。选择依据为训练集总体 RMSE，既定测试集结果见 [第一版与第二版对比](problem1_v1_v2_comparison.md)。", "",
        "## 4. 质量代理与人工核验", "",
        "`outputs/v2_review/人工盲评模板.xlsx` 包含 140 条 A1 文本；每个来源域按 Q 的四分位抽样，并额外纳入冲突样本。评审表只含随机编号和文本，不显示 Q、冲突状态或领域；`outputs/tables/v2_blind_review_key.json` 单独保存对照。两名评审者各复制模板独立填写五项 1–5 评分和问题标记。比较总体分与 Q 的 Spearman，并报告评审一致性；事实问题允许留空。该表目前未填写，因此人工有效性尚未获得。", "",
        "## 5. 领域置换结论的稳定性", "",
        f"在训练集上固定二阶 ALR Ridge 的 α=10，交叉比较 3 个参考域和 3 个 ε，共 9 种规格。以平均配比为中心，从 `pile_cc` 转移 1 个百分点至其他域时，{robust}/16 个目标域的预测 Loss 变化符号一致。`{', '.join(unstable)}` 的方向依赖规格，不宜写成稳定领域效应。置换只是模型的局部预测，不能解释为真实训练因果效应。", "",
        "## 6. 尚需的外部证据", "",
        "目前没有人工质量真值，无法证明 Q 的新旧优劣；没有同配比异质量的训练观测，无法识别质量对 Loss 的独立作用；仅有 1M 的训练标签，不能从 A8–A11 测试标签拟合规模校准后仍宣称独立测试。若补充规模训练实验或预先划分校准集，应另立验证协议。", "",
        "## 7. 复现", "",
        "依次运行 `python problem1/problem1_v2.py`、`python problem1/nested_v2_validation.py`、`python problem1/prepare_v2_review.py`、`python problem1/build_v2_report.py`、`python problem1/verify_problem1_v2.py`。质量原始处理仍按 `README.md` 中第一版流程执行。模型保存在 `outputs/models/mixture_v2.joblib`；候选与比较数据保存在 `outputs/tables/v2_*.csv`。盲评模板需要用 `.v2_artifact/build_review.mjs` 和 bundled artifact-tool 重新生成；现有文件已可直接供两名评审者复制填写。", ""]
    (HERE / "problem1_v2_report.md").write_text("\n".join(model_lines), encoding="utf-8")
    print(f"Wrote v2 report and comparison; 1M RMSE improvement {reduction:.1f}%, domains {improved}/13")


if __name__ == "__main__":
    run()
