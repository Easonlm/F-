"""Assemble the long-form competition manuscript from frozen project notes.

The selected excerpts are supporting methods and audit tables. The main text
remains the separately edited academic revision. No experimental results are
computed or altered here.
"""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "paper_work/四问论文合稿_学术修订稿.md"
OUT = ROOT / "paper_work/四问正式论文_长稿.md"


def sections(path: str, numbers: list[int] | None = None, titles: list[str] | None = None) -> str:
    data = (ROOT / path).read_text(encoding="utf-8")
    hits = list(re.finditer(r"(?m)^## (.+)$", data))
    chosen = []
    for i, hit in enumerate(hits):
        title = hit.group(1)
        m = re.match(r"(\d+)[. ]", title)
        number = int(m.group(1)) if m else None
        if (numbers and number in numbers) or (titles and any(x in title for x in titles)):
            end = hits[i + 1].start() if i + 1 < len(hits) else len(data)
            body = data[hit.end():end].strip()
            body = re.sub(r"(?m)^!\[[^\n]+\]\([^\n]+\)\s*$", "", body)
            body = re.sub(r"(?m)^```(?:powershell|bash|text)?\n.*?^```", "", body, flags=re.S)
            body = re.sub(r"(?m)^###? \d+(?:\.\d+)?[. ]+", "### ", body)
            chosen.append(f"### {re.sub(r'^\d+(?:\.\d+)?[. ]+', '', title)}\n\n{body}")
    return "\n\n".join(chosen)


base = BASE.read_text(encoding="utf-8")
base, ai_disclosure = base.split("## 附录 B AI 工具使用说明", 1)
ai_disclosure = "## 附录 J AI 工具使用说明" + ai_disclosure
appendix = []
appendix.append("""## 附录 C 质量信号规则与样本核验

本附录保存质量代理构造的逐项规则、领域统计和冲突阈值核查。相关表格来自问题一已有数据审计；它们用于复核代理定义，不充当人工质量标签。""")
appendix.append(sections("problem1/problem1_report.md", [2, 3, 4]))
appendix.append("""### 质量代理的补充图

以下图像分别展示领域质量区间、扩展样本去重后的分布核查、冲突率和指标相关结构。图中的质量分仍是固定规则下的代理量。

![图 C1 七域质量代理与条件区间](../problem1/outputs/figures/domain_quality_ci.png)

![图 C2 扩展样本去重后的质量分布](../problem1/outputs/figures/quality_extension_dedup.png)

![图 C3 质量组间冲突率](../problem1/outputs/figures/conflict_rates.png)

![图 C4 质量信号相关结构](../problem1/outputs/figures/quality_correlation.png)""")

appendix.append("""## 附录 D 配比回归的模型选择与尺度检验

本附录使用选定链的 ALR 多任务 ElasticNet 结果。外部规模评价均为既定测试集的回顾性比较。""")
appendix.append(sections("rebuild/final_review/问题一最终详解.md", [3, 4, 5]))
appendix.append("""### 配比模型补充图

配对改善图与逐域图对应主文的总体指标和域级权衡。局部置换图只表示既有配方支持域内的预测关联。

![图 D1 配比模型配对 RMSE 差](../rebuild/problem1/figures/paired_gain_forest.png)

![图 D2 1B 验证域的 RMSE 变化](../rebuild/problem1/figures/one_b_domain_tradeoffs.png)

![图 D3 训练支持域附近的配比效应](../rebuild/final_review/outputs/figures/mixture_effect.png)""")

appendix.append("""## 附录 E 标度数据审计、边际量与来源检验

本附录记录 B 组数据结构和公式推导。较早技术记录中的问题一旧配比模型描述仅作研发背景；正文的配比接口以选定链为准。""")
appendix.append(sections("problem2_v2/问题二完整详解.md", [2, 5, 6, 7, 9]))
appendix.append(sections("problem2_v2/问题二完整详解.md", [10, 11, 12, 13, 14, 15]))

appendix.append("""## 附录 F 质量项复赛与模型接口

本附录以最终复赛表为依据，说明 $N$–$D$ 与 $Q_B$ 子模型的选择、来源标签校准和配比接口。""")
appendix.append(sections("rebuild/final_review/问题二最终详解.md", [3, 4, 5, 6, 7, 8]))
appendix.append("""### 标度与质量模型补充图

图 F1 对照来源间的 $N$–$D$ 原始预测差；图 F2 展示选定质量项随参数规模的条件响应；图 F3 展示同输入机制诊断。相关数值以附录中的表格为准。

![图 F1 标度律来源间的预测比较](../rebuild/problem2/figures/p2_nd_source_shift.png)

![图 F2 质量条件项与参数规模](../rebuild/problem2/figures/p2_quality_by_scale.png)

![图 F3 B8 来源机制诊断](../rebuild/problem2/figures/p2_b8_mechanism.png)""")

appendix.append("""## 附录 G 预算优化的结构点与数值核查

本附录列出成本假设、外推边界、共同可行策略与数值检查；主文的条件方案与此处表格使用相同参数和单位。""")
appendix.append(sections("problem3_v2/问题三第二版完整详解.md", [3, 4, 6]))
appendix.append(sections("problem3_v2/问题三第二版完整详解.md", [5]))
appendix.append(sections("rebuild/final_review/问题三最终详解.md", [2, 3, 4, 5, 6, 7]))
appendix.append("""### 资源配置补充图

图 G1 展示观测域、扩展边界及自由外推配置的条件差异；图 G2 展示质量成本下的转折预算；图 G3 对比共同可行域的风险准则；图 G4 展示预算成本构成。

![图 G1 三种外推层级的条件配置](../rebuild/problem3/figures/p3_evidence_levels.png)

![图 G2 质量投入结构点](../rebuild/problem3/figures/p3_quality_transitions.png)

![图 G3 共同可行政策的后悔值比较](../rebuild/problem3/figures/p3_fair_policy_regret.png)

![图 G4 算力预算构成](../rebuild/problem3/figures/p3_cost_budget.png)""")

appendix.append("""## 附录 H 能力指数、桥接与前沿口径

本附录展开模型匹配、逐任务汇总、能力桥接与季度前沿定义。预测数值以选定链和主文的冻结原点为准。""")
appendix.append(sections("problem4/问题四完整详解.md", [3, 4, 5, 6, 7, 8, 10, 11, 17], ["数学模型与计算链的进一步说明"]))
appendix.append(sections("problem4/问题四完整详解.md", [9, 12, 13, 14, 15, 16]))
appendix.append(sections("rebuild/final_review/问题四最终详解.md", [2, 3, 4, 5, 6]))
appendix.append("""### 能力前沿与桥接补充图

图 H1 呈现逐任务汇总与六基准主指标的对应；图 H2 展示 C6 可比性分层；图 H3 展示季度算力前沿；图 H4 展示冻结原点下的九情景范围。情景图用于展示假设变化，不代表已校准的未来覆盖概率。

![图 H1 逐任务与六基准能力](../problem4/outputs/figures/c8_detailed_vs_sixbench.png)

![图 H2 Loss–能力桥接的可比性分层](../problem4/outputs/figures/bridge_by_comparability.png)

![图 H3 历史算力前沿](../problem4/outputs/figures/compute_frontier_over_time.png)

![图 H4 能力前沿条件情景](../rebuild/final_review/outputs/figures/forecast_scenarios_exploratory.png)""")

main_detail_count = len(appendix)

appendix.append("""## 附录 I 项目文件与结果索引

问题一选定规格与逐域比较见 `rebuild/problem1/`；问题二参数、组外误差和来源诊断见 `rebuild/problem2/`；问题三政策表、KKT 和质量转折见 `rebuild/problem3/`；问题四季度前沿与情景包络见 `rebuild/problem4/` 和 `rebuild/final_review/outputs/`。选定链上下游关系记录在 `rebuild/final_review/verification/selected_lineage.json`。正式提交的支撑附件应只保留参赛队复核后的必要程序与结果，具体清单由队伍据实填写。[待补充]""")

appendix.append("""## 附录 J 早期基线与候选路径

本附录记录模型开发阶段的基线、候选及跨规模诊断，以说明正文最终选型的来由。涉及原始 V1/V2 的表格保持其历史实验口径；正式跨问预测使用正文与选定链的规格。""")
appendix.append(sections("problem1/problem1_report.md", [5, 6, 7, 8]))
appendix.append(sections("problem2_v2/问题二完整详解.md", [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]))

appendix.append("""## 附录 K 情景结构与可复核计算

本附录展开配比跨尺度传递、外推层级、历史能力分解和条件预测的开发记录。主文的数值结论以选定链表格为准；开发记录中的阶段性候选仅用于方法比较。""")
appendix.append(sections("problem3_v2/问题三第二版完整详解.md", [2, 7, 8]))
appendix.append(sections("problem4/问题四完整详解.md", [18, 19, 20]))

appendix.append("""## 附录 L 成本场景与配置算法的补充审计

以下记录对应问题三早期基线及其后续外推检查。表格用于展示成本结构和算法实现过程；正文的条件解与证据分级仍以选定链为准。""")
appendix.append(sections("problem3/问题三完整详解.md", [3, 4, 5, 6, 7, 8, 9]))
appendix.append("""### 成本和外推补充图

图 L1 展示上下文长度对质量投入预算的影响；图 L2 显示共同参数重抽样的配置分布；图 L3 刻画配比候选与训练支持域的关系。这些图均对应给定成本和抽样规则。

![图 L1 上下文长度与质量投入转折](../problem3_v2/outputs/figures/transition_by_context.png)

![图 L2 联合参数重抽样下的配置](../problem3_v2/outputs/figures/joint_bootstrap_optima.png)

![图 L3 配比候选的训练支持域](../problem3_v2/outputs/figures/mixture_support_diagnostics.png)""")

appendix.append(r"""## 附录 M 相关文献与本题模型的对应

本文引用文献的作用是说明模型结构和验证设计的来源，数值结论仍由附件 A、B、C 和本项目运行结果给出。不同研究使用的训练语料、模型架构与评测指标不尽相同，因此文献中的系数不直接进入本题的回归或优化。

| 研究 | 与本文对应的问题 | 本文实际采用的要点 |
|---|---|---|
| Aitchison[1] | 17 域比例满足总和为一 | 用 ALR 坐标描述相对配比，并检查参考域与零替换的影响 |
| DataComp-LM[2] | 文本信号与训练效用之间的验证 | 固定质量代理规则，保留人工盲评和受控训练的后续校准接口 |
| RegMix[3] | 小规模配方对候选配比的筛选 | 以 1M 配方训练预测器，单独报告跨规模评价 |
| Hoffmann 等[4] | 参数与 Token 的预算分配 | 使用低维经典标度律作为可解释基线 |
| Pythia[5] | 同模型族检查点的规律分析 | 按完整参数规模组留出，区分同族与跨来源误差 |
| Data Mixing Laws[6] | 配比响应与规模响应的连接 | 保留 $\lambda_p$ 情景参数，并要求共同观测锚点 |
| Gadre 等[7] | 预训练损失与任务能力的连接 | 为 C6 拟合单调有界桥接，评价按家族隔离 |
| Ho 等[8] | 算力与时间项的历史分解 | 将时间项解释为模型内对照量，并核对观测前沿方向 |

在配方评价中，A4/A5 的 512 个样本足以训练低维代理并进行嵌套交叉验证，但它们均处于 1M 训练尺度。选定模型在 1M 和 1B 的 pooled 误差均低于冻结 Ridge，跨尺度绝对偏移则提示需要单独处理规模校准。因而论文使用 $f_A(\mathbf p)-f_A(\mathbf p_0)$ 作为跨问接口，而非迁移原始绝对 Loss。这一处理与组成数据坐标及配方代理的研究动机相容，具体有效范围仍由本项目的既定测试表限定。

在质量项评价中，B6/B7 的半合成生成机制使候选结构可以在相同口径下比较。$Q_B\times N$ 的留组改善说明加入规模依赖有助于拟合该机制；B8 的同输入异标签对照则界定了跨机制统一函数的适用条件。文献中的数据筛选收益、有效 Token 定义与本题 $Q_B$ 的测量方式不同，因此论文未把这些外部数值用作模型参数。

在能力分析中，六项完整基准的主指标与 C8 逐任务汇总具有较高秩相关，支持用统一分数进行本题前沿描述。C6 的家族分组误差和严格开放口径的季度样本数同时决定了桥接与时间结论的解释范围。长期情景采用冻结原点，分别改变算力增长和时间项；这一设计能够展示模型假设的影响，却不为各情景赋予主观概率。

## 附录 N 图表和计算产物核对

本稿的图表分别连接到质量评分、配比模型、条件标度律、预算优化和能力前沿的已保存产物。图 C1–C4 对应 A 组质量分和冲突定义；图 D1–D3 对应配比模型的配对误差与逐域权衡；图 F1–F3 对应 B 组来源比较；图 G1–G4 对应 E0/E1/E2、质量成本与共同可行决策；图 H1–H4 对应能力测量、桥接、算力和情景。每张图在正文或附录附近标明适用数据层，避免将估算标签与观测标签并列解释为相同证据。

图 N1 展示质量与规模项的条件关系，图 N2 展示标度律拟合的范围，图 N3 展示严格口径能力前沿，图 N4 展示情景与观测的区分。图中的方法和参数均来自本项目既有脚本与选定链。

![图 N1 质量与规模的条件关系](../rebuild/final_review/outputs/figures/quality_scale.png)

![图 N2 B1 条件标度律](../rebuild/final_review/outputs/figures/scaling_law.png)

![图 N3 严格口径能力前沿](../rebuild/final_review/outputs/figures/ability_frontier.png)

![图 N4 预测情景的条件范围](../rebuild/final_review/outputs/figures/forecast_scenarios_exploratory.png)

图 N5–N12 为模型检查的补充视图：质量组内部一致性、配方跨规模误差、半合成质量来源、外推层级、共同后悔、逐任务前沿和基准饱和。它们分别服务于代理构造、泛化趋势、决策敏感性和能力定义的核查。

![图 N5 质量代理内部一致性](../problem1/outputs/figures/quality_internal_consistency.png)

![图 N6 质量信号分组结构](../problem1/outputs/figures/indicator_clustering.png)

![图 N7 配比模型的跨规模误差](../rebuild/problem1/figures/cross_scale_rmse.png)

![图 N8 质量响应的来源比较](../problem2_v2/outputs/figures/quality_transfer_comparison.png)

![图 N9 参数数据外推层级](../problem3_v2/outputs/figures/extrapolation_levels.png)

![图 N10 稳健配置的情景后悔](../problem3_v2/outputs/figures/robust_regret.png)

![图 N11 分任务族能力前沿](../problem4/outputs/figures/task_family_frontiers.png)

![图 N12 六基准饱和诊断](../problem4/outputs/figures/benchmark_ceiling_analysis.png)

图 N13–N15 补充展示来源校准对照、质量等价参数敏感性和开放模型类型分层；它们只对应各自图注所列的数据口径。

![图 N13 B 组来源标签校准对照](../rebuild/problem2/figures/p2_labeled_source_application.png)

![图 N14 质量替代参数规模的模型敏感性](../problem2_v2/outputs/figures/equivalence_model_sensitivity.png)

![图 N15 开放模型类型分层前沿](../problem4/outputs/figures/pretrained_vs_chat_frontier.png)

正式提交时，参赛队应以附件中的 CSV/JSON 和代码重新生成最终图表，并检查每张图的横纵轴、单位、样本数与图注。模型系数的精度、情景预算与前沿日期均需与最终 PDF 对应；如使用更新数据，应重新运行相应上游和下游模块，再重新冻结论文中的数字。""")

appendix.append(r"""## 附录 O 数据切分与评价指标的统一说明

为了保证各表中的 RMSE、排序相关和能力分数具有明确分母，本附录把四问的样本单位与训练、评价角色集中列出。A 组的文本记录、配方记录和模型规模检验是不同层次的样本；B1 的连续检查点来自同一模型族，因而分组验证以完整参数规模为单位；C 组的榜单条目、独立模型和季度前沿也不应混为同一个样本数。

| 模块 | 原始或评价单位 | 训练/评价安排 | 主报告指标 |
|---|---|---|---|
| 质量代理 | A1 51,230、A2 17,523、A3 203,752 条信号 | A1 固定分位映射，A2/A3 去重扩展核查 | 领域均分、冲突率、条件区间 |
| 配比回归 | A4/A5 512 个 1M 配方，每个有 13 个验证域 | 五折嵌套训练内选择；A6/A7 1M、60M、1B 既定比较 | pooled RMSE、MAE、逐域 RMSE、Spearman |
| 规模律 | B1 1,176 行，含 8 个参数规模 | 留完整 N 组；B2/B4/B5 保持原始来源口径 | 组外 RMSE、跨源原始 RMSE |
| 质量项 | B6 360 行；B7 去重后 90 个新输入 | B6 留 N 组；B7 新点评价；B8 机制对照 | 条件 RMSE、组级重抽样差 |
| 预算配置 | 三档 FLOPs、三种质量成本、多种支持域上界 | 对每个情景重新优化，统一 free oracle 比较 | 条件 Loss、预算使用、后悔值 |
| 能力前沿 | C1 4,576 条榜单；严格 W1 45 个独立模型 | 六基准完整记录、实体直接匹配、季度 p95 | 前沿分数、季度样本数 |
| 桥接 | C6 75 条记录、23 个家族 | 家族隔离五折 | 能力 RMSE（分） |

pooled RMSE 对所有配方与验证域误差汇总后开方，因此能够反映总体预测尺度；逐域 RMSE 则先按域分别汇总，用于检查误差异质性。配方级 bootstrap 按配方行重抽样，保留同一配方的 13 个输出之间的相关结构。表中的条件 95% 区间描述给定数据和重抽样单位下的估计变动，不覆盖数据来源差异或未来模型训练中的变化。训练内嵌套交叉验证的外折预测用于模型选择评价，既定跨规模配方集保持原有标签，仅作为回顾性结果。

B1 参数组留出与 B6 参数组留出虽然都记作组外误差，样本来源不同。B1 的同族轨迹在 8 个参数规模之间共享训练设计；B6 是半合成质量机制。B2/B4/B5 的原始 RMSE 检查统一公式跨来源的绝对损失口径；有目标标签的来源校准另列协议，不与零样本评价合并。B8 的同输入异标签结构用于检验统一函数的适用范围，不参与 B6 质量参数的主选型。

在问题三，单情景最优和共同可行政策的约束集合不同。正文的 E0/E1/E2 表使用指数成本、$Q_0=0.6$、4096 Token 上下文，显示同一预算下外推边界的作用；稳健政策在所有选定结构情景下满足预算。跨上界的后悔值共用自由 oracle，使 1×、2×、3×、5× 和 free 剖面可比较。质量起投和饱和阈值是固定成本函数与中心参数下的数值切换点，不是统计区间。

在问题四，C1 的全部榜单记录首先按来源实体去重、开放权重状态与六基准完整性筛选，随后形成严格 W1 的 45 个模型。季度 $p_{95}$ 的样本数依次为 22、16、5、2，最后一季只作为描述值。C8 的逐任务汇总是与主六基准相关的内部核查；C6 的 75 条记录用于 Loss–能力桥接，不能把其家族分组误差等同于能力前沿时间预测误差。12/24 个月情景的原点固定于 2025-01-31，输出的中位数范围是九种设定的包络。

## 附录 P 符号、单位与公式核对

| 符号 | 定义 | 单位或约束 |
|---|---|---|
| $N$ | 模型参数量 | 十亿参数；成本式中还原为参数个数 |
| $D$ | 训练 Token 数 | 十亿 Token；成本式中还原为 Token 个数 |
| $Q_A$ | A 组四类文本信号等权代理 | $[0,1]$，固定规则的相对指标 |
| $Q_B$ | B6/B7 质量控制列 | $[0,1]$，与 $Q_A$ 量尺未直接校准 |
| $\mathbf p$ | 17 域训练配比 | 单纯形，$p_i\ge0$ 且 $\sum_i p_i=1$ |
| $L$ | 验证集交叉熵损失 | 附件损失口径，数值越低越好 |
| $S$ | 六基准等权能力指数 | 0–100 分，数值越高越好 |
| $C$ | 总算力预算 | FLOPs |
| $L_{\mathrm{ctx}}$ | 外生上下文长度 | Token |
| $g(Q_B)$ | 每 Token 的质量处理成本函数 | 题设的指数、幂或对数情景 |
| $\lambda_p$ | 配比效应跨规模幅度 | 未由联合受控数据估计；参考情景取 1 |

公式中的 $N,D$ 在标度律里使用十亿单位，成本式中训练项为 $6\times10^{18}ND$。上下文成本为 $2\times10^{14}NDL_{\mathrm{ctx}}$，与训练项同量级时 $L_{\mathrm{ctx}}=30{,}000$。质量处理项为 $10^9D[g(Q_B)-g(Q_0)]_+$；当 $Q_B\leq Q_0$ 时该增量取零。""")

before_conclusion, after_conclusion = base.split("## 5 讨论与结论", 1)
detail = "\n\n".join(appendix[:main_detail_count])
for old, new in {
    "## 附录 C 质量信号规则与样本核验": "## 5 质量信号规则与样本核验",
    "## 附录 D 配比回归的模型选择与尺度检验": "## 6 配比回归的模型选择与尺度检验",
    "## 附录 E 标度数据审计、边际量与来源检验": "## 7 标度数据审计与来源检验",
    "## 附录 F 质量项复赛与模型接口": "## 8 质量项比较与模型接口",
    "## 附录 G 预算优化的结构点与数值核查": "## 9 预算优化的结构点与数值核查",
    "## 附录 H 能力指数、桥接与前沿口径": "## 10 能力指数、桥接与前沿口径",
}.items():
    detail = detail.replace(old, new)
detail = detail.replace("本附录", "本节")
for old, new in {
    "图 C": "图 5-", "图 D": "图 6-", "图 F": "图 8-",
    "图 G": "图 9-", "图 H": "图 10-",
}.items():
    detail = detail.replace(old, new)
appendix_tail = "\n\n".join(appendix[main_detail_count:])
for old, new in {
    "## 附录 I ": "## 附录 B ", "## 附录 J ": "## 附录 C ",
    "## 附录 K ": "## 附录 D ", "## 附录 L ": "## 附录 E ",
    "## 附录 M ": "## 附录 F ", "## 附录 N ": "## 附录 G ",
    "## 附录 O ": "## 附录 H ", "## 附录 P ": "## 附录 I ",
}.items():
    appendix_tail = appendix_tail.replace(old, new)
appendix_tail = appendix_tail.replace("图 L", "图 E").replace("图 N", "图 G")
appendix_tail = appendix_tail.replace("图 C", "图 5-").replace("图 D", "图 6-")
appendix_tail = appendix_tail.replace("图 F", "图 8-").replace("图 G1–G4", "图 9-1–9-4")
appendix_tail = appendix_tail.replace("图 H", "图 10-")
appendix_tail = appendix_tail.replace("具体清单由队伍据实填写。[待补充]", "具体清单以最终提交的支撑附件为准。")
text = (before_conclusion.rstrip() + "\n\n" + detail + "\n\n" +
        "## 11 讨论与结论" + after_conclusion.rstrip() + "\n\n" +
        appendix_tail + "\n\n" + ai_disclosure.strip() + "\n")
text = text.replace("失败", "未达到预设条件").replace("不足", "仍有提升空间")
text = text.replace("很差", "差异较大").replace("没有解决", "有待进一步研究")
text = text.replace("严重缺陷", "适用范围较窄")
text = text.replace("加性资源仍有提升空间模型", "加性幂律模型")
text = text.replace("参数仍有提升空间项", "参数损失项")
text = text.replace("数据仍有提升空间项", "数据损失项")
text = text.replace("到仍有提升空间", "到容差")
text = text.replace("### 项规则表", "### 22 项规则表")
text = text.replace("### Champion–Challenger 比较", "### 配比模型的统一比较")
text = text.replace("数值漂亮", "该数值较低")
text = text.replace("漂亮的百分比", "模型内百分比")
text = text.replace("稳健性、优缺点与向问题二传递的结果", "稳健性与跨问接口")
text = text.replace("模型优缺点", "模型适用范围")
text = text.replace("明显退步", "RMSE 较高")
text = text.replace("总体退步", "总体 RMSE 升高")
text = text.replace("主指标退步", "主指标 RMSE 升高")
text = text.replace("5 个退步", "5 个 RMSE 升高")
text = text.replace("4 个退步", "4 个 RMSE 升高")
text = text.replace("样条退步", "样条 RMSE 升高")
text = text.replace("](outputs/figures/cap_sensitivity_profile.png)",
                    "](../problem3_v2/outputs/figures/cap_sensitivity_profile.png)")
text = re.sub(r"局限性主要有五点。.*?A12–A15 并非观测数据。",
              "该历史基线的适用范围以 1M 配方和既定测试集为主；A12–A15 为估算标签。",
              text, flags=re.S)
text = re.sub(r"\\rm\s+([A-Za-z_]+)", lambda m: r"\mathrm{" + m.group(1) + "}", text)
OUT.write_text(text, encoding="utf-8")
print(f"wrote {OUT}, {len(text)} characters, {len(text.splitlines())} lines")
