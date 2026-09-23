# 第二轮文献与官方实现核查

本轮只把文献用于定义**可在本题数据识别**的候选形式；论文对其他任务的性能数字不移植成本题参数。以下网址均为作者论文页、正式会议/期刊或作者代码。

| 作者、年份、题名 | Venue 与链接 | 对本题的实际启示 |
|---|---|---|
| Kaplan et al. (2020), *Scaling Laws for Neural Language Models* | [arXiv:2001.08361](https://arxiv.org/abs/2001.08361) | 经验幂律依赖同口径 loss 和控制条件；不能把跨评估集 raw 截距视为统一常数。 |
| Hoffmann et al. (2022), *Training Compute-Optimal Large Language Models* | [NeurIPS 2022 论文](https://proceedings.neurips.cc/paper_files/paper/2022/file/c1e2faff6f588870935f114ebe04a3e5-Paper-Conference.pdf), [arXiv:2203.15556](https://arxiv.org/abs/2203.15556) | 加性 (E+A/N^\alpha+B/D^\beta) 为强基线；算力 (C\approx6ND) 只是约束，不会自动消除跨来源 loss 差异。 |
| Biderman et al. (2023), *Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling* | [ICML 2023](https://proceedings.mlr.press/v202/biderman23a.html), [官方代码](https://github.com/EleutherAI/pythia) | Pythia 模型共享数据顺序和大量检查点；同族模型留一仍不能代表不同训练/测评来源。 |
| Bahri et al. (2024), *Explaining Neural Scaling Laws* | [PNAS, DOI:10.1073/pnas.2311878121](https://doi.org/10.1073/pnas.2311878121) | 不同机制有不同 scaling regime；转折必须由同源数据的残差与阈值识别支持。 |
| Caballero et al. (2023), *Broken Neural Scaling Laws* | [ICLR 2023](https://openreview.net/pdf?id=sckjveqlCZ), [官方代码](https://github.com/ethancaballero/broken_neural_scaling_laws) | 平滑转折可外推非单一幂律；本题仅试少参数 broken-N，避免用 B8 选转折。 |
| Ye et al. (2025), *Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance* | [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/cc84bfabe6389d8883fc2071c848f62a-Abstract-Conference.html), [官方代码](https://github.com/yegcjs/mixinglaws) | 小模型配比规律可与规模律嵌套，但需要多规模配比实验才能识别交互；本题 B 不含 17 域配比。 |
| Chang et al. (2024), *Scaling Parameter-Constrained Language Models with Quality Data* | [EMNLP 2024 Industry Track](https://aclanthology.org/2024.emnlp-industry.8.pdf), [arXiv:2410.03083](https://arxiv.org/abs/2410.03083) | 使用多样性和合成度构成有效 token；本题 B 的单一 Q 分数不足以直接复制其微观指标。 |
| Goyal et al. (2024), *Scaling Laws for Data Filtering—Data Curation cannot be Compute Agnostic* | [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Goyal_Scaling_Laws_for_Data_Filtering--_Data_Curation_cannot_be_Compute_CVPR_2024_paper.html), [官方代码](https://github.com/locuslab/scaling_laws_data_filtering) | 质量收益可随数据量和重复次数变动；因此检验 Q×D，而不把质量惩罚一律设为独立加项。研究对象为 VLM，应用到本题属于候选假设。 |
| Hernandez et al. (2022), *Scaling Laws and Interpretability of Learning from Repeated Data* | [arXiv:2205.10487](https://arxiv.org/abs/2205.10487) | 重复数据可能使 token 数不等于独立有效信息量；附件无重复率，不能估计该机制。 |
| Subramanyam, Chen & Grossman (2026), *Scaling Laws Revisited: Modeling the Role of Data Quality in Language Model Pretraining* | [ICLR 2026 论文](https://openreview.net/pdf?id=x54wwB6QvL), [arXiv:2510.03313](https://arxiv.org/abs/2510.03313) | 质量的有效样本量解释与本题 effective-token 候选相关；其受控合成实验并不能证明 B6/B8 共用同一个 Q 方向。 |

## 实现核查

查看了 [BNSL 官方代码](https://github.com/ethancaballero/broken_neural_scaling_laws) 的拟合与外推脚本、[Data Mixing Laws 官方代码](https://github.com/yegcjs/mixinglaws) 的 `law.py` 和 pipeline 结构、[Pythia 官方仓库](https://github.com/EleutherAI/pythia) 的检查点说明，以及 [Data Filtering 官方仓库](https://github.com/locuslab/scaling_laws_data_filtering)。本题实现采用最少自由度、清晰边界和独立验证协议，不复制对方训练实验或声称复现其结果。
