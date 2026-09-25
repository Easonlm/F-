# 问题二：候选与判胜协议（运行前登记）

登记日期：2026-09-25。所有重跑输出写入本目录；原 `problem2/` 与 `problem2_v2/` 是冻结基线。独立读取 `real_attachments/B_scaling_laws/`，不调用旧版拟合函数，也不把旧输出表当计算输入。

## 现象与可识别性

- B1 的经典律全样本残差标准差约 0.000146，`val_loss` 以四位小数为主，8 个 N 组共用 D 网格；组外误差仍只是同一 Pythia 机制。B2 的旧版原始 RMSE 约 1.243，B4 为 0.293，B5 为 0.198。需要把跨来源原始误差放在 B1 内部误差之前解释，不能从近乎确定性的 B1 再堆自由度。
- B2 的残差有 D 趋势；来源截距或仿射修正可作**有目标来源标签**的应用层。B4/B5 的 Loss 口径未证实等同于 B1。B1 拟合的零样本模型与使用目标来源标签校准的模型分轨比较。
- B6 与 B7 的 Q 增加对应 Loss 下降，B8 在相同 N,D,Q 处与 B6 冲突且 Q 方向反转。B8 只作机制诊断，不参与拟合或挑选正常方向的质量律。
- B 数据没有配方 `p` 的独立跨规模联合损失标签。`lambda_p(N)` 仅列情景，不做“估计”。

## 原始文献给出的候选边界

- [Hoffmann et al., NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/c1e2faff6f588870935f114ebe04a3e5-Paper-Conference.pdf) 支持 `E + A N^-alpha + B D^-beta` 作低维基线。
- [Caballero et al., ICLR 2023](https://openreview.net/forum?id=BfGrlFuNyhJ) 的平滑转折提醒检验 broken scaling；本题仅允许一个 N 转折与一个斜率变化，须经 BIC、组外和参数条件数约束。
- [Chang et al., EMNLP Industry 2024](https://aclanthology.org/2024.emnlp-industry.8/) 将质量解释为有效 token；本题只有汇总 Q，故仅拟合一参数 `D_eff=D Q^gamma`，不复制其多样性/合成度指标。
- [Ye et al., ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/cc84bfabe6389d8883fc2071c848f62a-Abstract-Conference.html) 说明混合配方影响可定量建模，但不提供本题 `lambda_p(N)` 的识别数据。

## 候选、数据隔离与规则

N-D 候选：`classic`、一个乘积交互 `classic+H N^-alpha D^-beta`、平滑 `broken_N`。仅 B1 拟合；B1 按 N 留组及小 N→大 N、早 D→晚 D 检验；B2/B4/B5 全部只做零样本原始预测。升级须同时：B1 留组 RMSE 至少下降 5%，按留组误差做配对自举的改善 95% 区间下界大于 0，三个外部源各自 RMSE 不增加超过 2%，并检查 BIC、Jacobian 条件数及外推。训练拟合改善单独列示，不能判胜。若候选已在旧轮次比较，B2/B4/B5 属透明的开发性外部复查，不称全新盲测。

Q 候选：`additive`、`effective_tokens`、`Q×N`（现任）、`Q×D`、`Q×N×D`。固定 B1 重拟合的 N-D 部分，仅 B6 拟合 Q；B6 按 N 留组，B7 只用相对 B6 新增的输入点验证。挑战 `Q×N` 须 B6 留 N 与 B7 新点 RMSE 均至少下降 5%，B7 按 N 配对自举改善区间下界大于 0，新增指数的 B6 按 N 自举 95% 区间排除 0，BIC 不明显变差，且不能以 B8 诊断成绩判胜。B8 只记录同点冲突、方向与预测残差。

来源应用模型：在每个来源每条轨迹/家族的早期 20% 标签上单独拟合截距、仿射或 `log D` 残差修正，剩余 80% 作条件 holdout；它不是零样本泛化。另把 B5 完全留出，检查 B2+B4 学到的平均来源修正能否零样本迁移；不能用 B5 标签调节规则后再称它独立。来源内条件 holdout 由于旧轮次已看过，仍属于开发证据。

`p` 桥：保留 `lambda_p=1` 作为接口约定，并列幂衰减、带底值衰减和区间情景；全部标记 `identified=false`。问题一最终上游若发生变化，必须重新生成 `Delta_p`，本次不把旧问题一标签用于 `lambda_p(N)` 的再训练。
