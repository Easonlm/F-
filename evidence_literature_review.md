# 与本项目已发现缺陷对应的正式文献核验

本表先依据项目残差、接口和样本量确定问题，再查正式论文。文献说明方法可能性，不替代本项目的留出验证。检索日期：2026-09-24。

| Title / Authors / Year / Venue / URL | Relevant problem and current weakness | 可借鉴 | 不能直接迁移 |
|---|---|---|---|
| [The Statistical Analysis of Compositional Data](https://academic.oup.com/jrsssb/article/44/2/139/7027742), John Aitchison, 1982, *JRSS B*, DOI [10.1111/j.2517-6161.1982.tb01195.x](https://doi.org/10.1111/j.2517-6161.1982.tb01195.x) | P1 的 17 域比例在单纯形上；原始比例回归有闭合约束 | ALR/对数比坐标及参照敏感性 | 不能证明二阶交互具有训练因果意义，也不能验证零替代常数 |
| [RegMix: Data Mixture as Regression for Language Model Pre-training](https://proceedings.iclr.cc/paper_files/paper/2025/file/5f67d864aae6115374fed7beddd119e0-Paper-Conference.pdf), Qian Liu 等, 2025, ICLR | P1 的小模型配比迁移到大模型，现有跨规模绝对 Loss 严重偏移 | 在受控配比训练中用代理模型筛选候选，并验证大模型排序 | 论文中有真实跨规模训练；本项目不能把 1M 模型局部差分自动当作 1B/高预算真实收益 |
| [Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance](https://proceedings.iclr.cc/paper_files/paper/2025/hash/cc84bfabe6389d8883fc2071c848f62a-Abstract-Conference.html), Jiasheng Ye 等, 2025, ICLR | P2 的配比效应跨规模系数 `lambda_p` 未识别 | 可把配比响应与规模响应分层建模并设计跨尺度验证 | 本项目没有该论文的受控联合实验；不能凭文献填入精确 `lambda_p(N)` |
| [An empirical analysis of compute-optimal large language model training](https://papers.nips.cc/paper_files/paper/2022/hash/c1e2faff6f588870935f114ebe04a3e5-Abstract-Conference.html), Jordan Hoffmann 等, 2022, NeurIPS, DOI [10.52202/068431-2176](https://doi.org/10.52202/068431-2176) | P2/P3 的 `N-D` 与预算优化 | 保留可解释的经典基线和预算约束推导 | 其经验比例来自不同训练设计；本项目的半合成 Q 成本与长上下文项不能由此获实证认证 |
| [Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling](https://proceedings.mlr.press/v202/biderman23a.html), Stella Biderman 等, 2023, ICML | P2 的 B1 同族误差约 0.00015，容易误当跨源泛化 | 把统一模型族的检查点用于内部动力学分析 | 相同数据顺序和训练设置意味着留模型组的误差不是新来源误差 |
| [Broken Neural Scaling Laws](https://openreview.net/pdf?id=sckjveqlCZ), Ethan Caballero、Kshitij Gupta、Irina Rish、David Krueger, 2023, ICLR | P2 曾考察 broken-N，但 B1 没有稳定残差折点 | 只有诊断出现转折时才把平滑折点作为候选 | 不能仅因论文存在就加参数；本项目 BIC 与可识别性目前支持经典式 |
| [DataComp-LM: In search of the next generation of training sets for language models](https://papers.nips.cc/paper_files/paper/2024/hash/19e4ea30dded58259665db375885e412-Abstract-Datasets_and_Benchmarks_Track.html), Jeffrey Li 等, 2024, NeurIPS Datasets and Benchmarks, DOI [10.52202/079017-0455](https://doi.org/10.52202/079017-0455) | P1 的指标 Q 尚未证明训练效用 | 数据质量主张最好经固定模型、受控数据筛选和下游任务验证 | 不能把该文的数据筛选收益转移到 SlimPajama 代理分或本项目的 17 域上 |
| [Language models scale reliably with over-training and on downstream tasks](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a91869936a63d814971b6423990ecf6e-Abstract-Conference.html), Samir Yitzhak Gadre 等, 2025, ICLR | P4 的 Loss–Benchmark 桥接组外 RMSE 约 11.86 | 在统一训练和评价口径下以简单函数联系 Loss 与多任务表现 | 该文受控模型和数据分布不等于 C6 跨族混合口径；不可代入其系数 |
| [Algorithmic progress in language models](https://proceedings.neurips.cc/paper_files/paper/2024/file/6b066da6a23bc55f9b887e7298102884-Paper-Conference.pdf), Anson Ho 等, 2024, NeurIPS | P4 把时间残差称技术贡献时存在样本选择和口径混杂 | 对固定能力水平的等效算力和技术趋势做敏感性分解 | 该研究有 2012–2023 的长期同类评价；本项目严格样本只有四季度，不能复制因果解释 |
| [A note on the validity of cross-validation for evaluating autoregressive time series prediction](https://talks.robjhyndman.com/publications/cv-time-series/), Christoph Bergmeir、Rob J. Hyndman、Bonsoo Koo, 2018, *Computational Statistics & Data Analysis*, DOI [10.1016/j.csda.2017.11.003](https://doi.org/10.1016/j.csda.2017.11.003) | P4 长期预测没有对应期限回测 | 预设按时间推进的验证原点，明确可验证期限 | 该文条件针对自回归预测，不证明本项目四季度资料足以评估 12/24 月 |

## 引用纠错与边界

- 根目录赛题 Markdown 和《数据说明2.pdf》将 RegMix 标为“ICML 2024”；正式会议论文首页标注 **ICLR 2025**。正式论文应以会议论文信息为准。问题一已有 Word 正文使用 ICLR 2025，需保证四问合稿保持一致。
- `Q_A→Q_B`、B8 机制冲突、E1 三倍边界目前主要依据项目数据与数学可识别性判断；未找到能替本项目标定具体映射或三倍阈值的正式文献。不得借外部论文伪装为已识别参数。
