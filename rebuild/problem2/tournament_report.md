# 问题二最终复赛报告

**结论：保留 Model C：classic N-D + Q×N。** 两个模块均从原始 B 文件独立重拟合；没有修改 `problem2/` 或 `problem2_v2/`。本轮 `Q×N×D` 候选带来小幅 B7 点预测改善，但组级不确定性覆盖零改善，故未通过[预注册判胜协议](why_this_candidate.md)。

## 数据与协议

- N-D 只在 B1 的 1176 条记录上拟合。B1 按 8 个 N 组留一；B2、B4、B5 保持原始 Loss 口径作零样本外部比较。B2/B4/B5 在旧轮次已被查看，本轮属于独立复算的开发性外部证据，不能再称全新盲测。
- Q 只在 B6 的 360 条记录上拟合；按 9 个 N 组留一。B7 与 B6 去重后仅以 90 个新增点评估。B8 与 B7 去重后 1480 个点只作机制诊断，不参与参数估计或 winner 筛选。
- 来源校准另用各目标来源轨迹或家族早期约 20% 的有标签 Loss，后期约 80% 是**条件 holdout**；与原始零样本指标严格分开。B5 完全留出，用于 B2+B4 来源平均修正的转移检验。
- 所有量纲：`N_B` 为十亿参数，`D_B` 为十亿 token；`Q_B` 只具有 B6/B7 机制下的语义。

## N-D 比赛

| 模型 | B1 训练 RMSE | B1 留 N RMSE | B2 raw | B4 raw | B5 raw | B1 BIC | Jacobian 条件数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| classic | 0.00014650 | **0.00015118** | 1.24308 | 0.29267 | 0.19760 | **-20729.31** | 1.14×10³ |
| N×D 交互 | 0.00014647 | 0.00015232 | 1.24308 | 0.29267 | 0.19760 | -20722.62 | 1.56×10³ |
| broken-N | **0.00014614** | 0.00015512 | 1.24308 | 0.29267 | 0.19757 | -20720.85 | 3.12×10¹⁰ |

交互的训练优势只有约 0.016%，留 N 反而变差。broken-N 训练降低约 0.24%，但留 N 变差约 2.6%，小 N→大 N 半数训练切分 RMSE 0.000669（classic 为 0.000151），转折参数弱识别。两个 Challenger 均不满足 5% 组外改善，故保留 classic。B1 的四位小数 Loss 与近确定性同源关系使微小训练差异缺乏应用意义。正式的平滑转折模型来自 [Caballero et al., ICLR 2023](https://openreview.net/forum?id=BfGrlFuNyhJ)，但本数据未支持其新增自由度；经典加性形式对应 [Hoffmann et al., NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/c1e2faff6f588870935f114ebe04a3e5-Paper-Conference.pdf)。

## Q 比赛

| 模型 | B6 训练 RMSE | B6 留 N RMSE | B7 新点 RMSE | B6 BIC |
| --- | ---: | ---: | ---: | ---: |
| additive | 0.07607 | 0.08144 | 0.05952 | -1843.04 |
| effective tokens | 0.11697 | 0.11957 | 0.10557 | -1539.10 |
| **Q×N（原 Champion）** | 0.05180 | 0.05235 | 0.04213 | -2113.75 |
| Q×D | 0.07508 | 0.08060 | 0.05890 | -1846.52 |
| Q×N×D | **0.05036** | **0.05106** | **0.04117** | **-2128.20** |

Q×N×D 相对 Q×N 的留 N 改善 2.46%、B7 新点改善 2.28%，均低于预注册 5% 门槛。B7 按 N 组配对自举，绝对 RMSE 改善的 95% 区间为 **[-0.00111, 0.00262]**，覆盖零。新增 D 指数在 B6 按 N 自举的 95% 区间 [0.0259, 0.0546]，说明该半合成机制内有微弱 D 调制，但不足以在新增 B7 组上稳定胜出。Q×N 指数区间 [0.1501, 0.1909]。有效 token 候选受 [Chang et al., EMNLP Industry 2024](https://aclanthology.org/2024.emnlp-industry.8/) 启发；其原文用多样性和合成度表征质量，本附件只有一维 Q，故失败不能反驳原文方法。

## 来源、B8 与 p 桥

零样本 classic 在 B2/B4/B5 的 raw RMSE 为 1.2431/0.2927/0.1976，远大于 B1 留组误差，显示来源或评价口径差异。在有来源标签的早期 20% 校准协议下，B2 的 `log D` 残差修正后期 RMSE 为 0.3674，B4 的仿射修正为 0.0916，B5 的仿射修正为 0.1619；这些是多个应用形式已查看后的描述性成绩，不能当零样本外推或全新独立模型选择。将 B2+B4 的平均截距修正直接零样本迁到完全留出的 B5，RMSE 从 0.1976 恶化至 1.0696，说明来源效应不能视为通用常数。

B6/B7 的各 `(N,D)` 网格均为 Q 增加、Loss 下降；B8 新点中符合该方向的网格比例为 0。B6 与 B8 有 **160** 个完全相同 `(N,D,Q)` 输入，平均 `B8−B6` Loss 为 **-1.15361**，其中 Q=0.1 平均差 -2.1583、Q=1 平均差 -0.0125。这是数据机制或 Q 语义冲突的直接证据。任何仅靠同一个单调质量律的 B8 拟合不能当统一质量因果规律。

`lambda_p=1` 只作为 P1 中心化 `Delta_p` 的接口约定。`lambda_p(N)` 的幂衰减、带底值衰减及区间已列在 `lambda_p_scenarios.csv`，全部标记为 `identified=false`。当前 B 数据没有联合 `N,D,Q,p` 的独立 Loss 标签；[Ye et al., ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/cc84bfabe6389d8883fc2071c848f62a-Abstract-Conference.html) 的配比建模结论不替代本题的识别数据。

## Winner 与下游接口

选择的主模型（参数来自本轮重拟合）为：

\[
L=1.6897975629+0.3539803207N^{-0.3399765819}
 +1.2403055836D^{-0.2798781285}
 +0.3568073584(1-Q_B)^{0.9915433891}N^{-0.1620729110}
 +\lambda_p\Delta_p.
\]

参数机器接口见 [`selection.json`](selection.json)，预测函数见 [`predict_final.py`](predict_final.py)。`Delta_p` 必须由最终问题一模型相对相同参考配方中心化后供给；如果问题一 Winner 更换，应重生成该输入及问题三、四依赖。目标来源标注 Loss 可另做条件校准，但不能自动缩放 Q/p 项，也不能把来源参数当作零样本已知。

## 复现与完整性

从项目根目录执行 `python rebuild/problem2/run_tournament.py`、`python rebuild/problem2/verify_tournament.py`。[`reproduce.py`](reproduce.py) 连续两次独立重跑并校验 20 个输出 SHA，`reproduction.json` 记录其逐字节一致；`verification_results.json` 的 15 项检查全部通过。只读 SHA 核对冻结的 `problem2/` 与 `problem2_v2/` 中 119 个已登记文件，119 个均与基线一致，详见 `frozen_baseline_sha_audit.json`。完整逐模型、逐数据集指标在 [`problem2_model_tournament.csv`](problem2_model_tournament.csv)。
