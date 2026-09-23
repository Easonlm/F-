# 问题一数据处理、质量评分与验证总结

## 1. 总体结论

问题一已经完成了以下主要工作：

1. 对 A1-A3 的 272,505 条质量信号记录进行预处理；
2. 将 22 个质量指标统一为 `[0,1]` 且“越高越好”；
3. 构造样本级综合质量代理分数 $Q_j$；
4. 按来源域聚合得到领域级质量分数 $Q_d$；
5. 定义并分析质量冲突，比较冲突惩罚方案；
6. 用 A4/A5 的 17 域配比预测 13 个验证域 Loss；
7. 在 A6-A11 上进行同规模和跨规模检验；
8. 尝试将领域质量 $Q$ 映射到训练配方。

当前结果可以证明：数据处理流程、综合评分模型和配比预测基线已经建立并且可复现；但尚不能证明 $Q$ 是真实训练价值，也不能将配比效应解释为因果效应。

---

## 2. 数据预处理在哪里完成

### 2.1 质量数据预处理

核心文件是：

- `problem1/quality_pipeline.py`

主要函数：

- `load_all()`：流式读取 A1-A3；
- `scalar()`：把列表型指标压缩为标量；
- `transform()`：长尾变换、方向统一、标准化和缺失值处理；
- `group_scores()`：计算四个质量维度；
- `scores()`：生成综合质量分；
- `domain_summary()`：从样本级分数聚合领域级结果。

### 2.2 配比数据预处理

核心文件是：

- `problem1/mixture_pipeline.py`

主要操作：

- 根据 `index` 配对训练配比和 Loss；
- 检查重复键、缺失匹配和非法数值；
- 检查配比非负且每行和接近 1；
- 对配比重新归一化；
- 用 ALR 对数比变换处理 17 个比例和为 1 的组合数据。

### 2.3 数据审计

文件：

- `problem1/data_audit.py`

它检查 A1-A15 的文件结构、行数、列数、缺失值、重复 `index` 和配比和偏差，并输出审计表。

---

## 3. A1-A3 的数据规模与输入

输入文件位于 `real_attachments/A_data_value/`：

| 数据集 | 记录数 | 数据性质 |
|---|---:|---|
| A1 | 51,230 | 质量信号抽样集，包含七个来源域 |
| A2 | 17,523 | arxiv 扩展集 |
| A3 | 203,752 | github 扩展集 |
| 合计 | 272,505 | 全量质量信号 |

审计结果见：

- `outputs/tables/quality_audit.json`
- `outputs/tables/quality_file_counts.csv`
- `outputs/tables/data_audit.csv`

A1 与扩展集存在重叠：

- A1 与 A2 重叠 1,419 个 ID；
- A1 与 A3 重叠 10,000 个 ID；
- 同一来源内部没有重复 ID。

因此在 A1 与 A2/A3 对照时，扩展集会排除已出现在 A1 中的 ID，避免重复记录被当成独立证据。

---

## 4. 22 个质量指标的预处理

每个样本的第 $k$ 个指标最终变为：

$$
z_{jk}\in[0,1],
$$

并统一解释为：数值越大，表示该项质量越好。

每个字段的详细规则见：

- `outputs/tables/quality_indicator_rules.csv`

### 4.1 列表型指标压缩

项目把列表型指标先压缩为一个标量。

#### 二分类 logits

用于 `fluency_en` 和 `ad_en`。

将两个类别的 logits 转换为正向类别概率：

$$
p_1=\frac{e^{l_1}}{e^{l_0}+e^{l_1}}.
$$

其中 `fluency_en` 的正向类别解释为流畅，`ad_en` 的正向类别解释为无广告。

#### 六级 ModernBERT logits

用于：

- `modernbert_cleanliness`；
- `modernbert_readability`；
- `modernbert_reasoning`；
- `modernbert_professionalism`。

先计算六个等级的 softmax 概率，再计算等级期望并除以 5：

$$
\tilde z=\frac{\sum_{r=0}^{5}r p_r}{5}.
$$

#### 单元素列表

`fineweb_edu` 直接取列表中的唯一值。

#### QuRating 四维列表

`qurater` 的四个维度取等权平均：

$$
qurater=\frac{q_1+q_2+q_3+q_4}{4}.
$$

### 4.2 长尾变换

对文档长度、句子数量和重复字符比例等长尾指标，先进行：

$$
x'=\operatorname{sign}(x)\log(1+|x|).
$$

这样可以压缩极端值，避免少量极大值主导评分。

### 4.3 指标方向统一

#### 越高越好

使用 A1 的 1% 和 99% 分位点做截尾 Min-Max：

$$
z=\operatorname{clip}\left(\frac{x-q_{0.01}}{q_{0.99}-q_{0.01}},0,1\right).
$$

#### 越低越好

先按同样方法归一化，再取反：

$$
z_{final}=1-z.
$$

主要负向指标包括：

- `rps_doc_frac_no_alph_words`；
- `rps_doc_frac_chars_top_2gram`；
- `rps_doc_frac_chars_top_3gram`。

#### 适中最好

对文长、句子数、大小写比例、数字比例和平均词长等指标采用梯形效用：低于合理区间或高于合理区间都会降低得分，中间区间得分较高。

### 4.4 缺失值处理

`modernbert_reasoning` 有 13 条缺失，`modernbert_professionalism` 有 6 条缺失，其他主要字段没有缺失。缺失的标准化值使用 A1 参考中位数填补，并在规则表中记录缺失数量。

### 4.5 统一标尺

所有分位点和梯形阈值都由 A1 计算，再固定应用于 A2 和 A3。这样三个来源处于同一评分尺度，避免每个文件单独标准化导致不可比。

---

## 5. 预处理结果在哪里

### 5.1 标准化指标结果

文件：

- `outputs/tables/quality_standardized.parquet`

内容包括：

- `source`；
- `domain`；
- `id`；
- 22 个统一方向后的标准化指标。

### 5.2 指标规则表

文件：

- `outputs/tables/quality_indicator_rules.csv`

记录每个指标的：

- 原始类型；
- 原始方向；
- 列表压缩方法；
- 是否使用 `signed_log1p`；
- 标准化方法；
- 校准分位点；
- 缺失数量；
- 最终含义。

这个文件是预处理规则的直接结果。

### 5.3 样本级质量评分

文件：

- `outputs/tables/sample_quality_scores.parquet`

每条记录包含：

- 来源和领域；
- `Q_baseline`；
- `Q_critic`；
- `Q_robust`；
- `Q_penalty`；
- `Q_hard`；
- 冲突标记；
- 冲突强度。

---

## 6. 综合质量评分 $Q$ 的具体计算

### 6.1 四个质量维度

22 个指标被分成：

1. 教育性；
2. 可读性；
3. 洁净度；
4. 文本结构。

三个高度相关的 DSIR 指标先合并：

$$
z_{DSIR}=\frac{z_{books}+z_{wiki}+z_{math}}{3}.
$$

教育性组实际计算为：

$$
S_{j,educational}
=\frac{z_{fineweb}+z_{reasoning}+z_{professionalism}+z_{qurater}+z_{DSIR}}{5}.
$$

其他三组分别对所属指标取均值。

### 6.2 样本级综合分

最终主评分为：

$$
Q_j=\frac{1}{4}
\left(
S_{j,educational}+
S_{j,readability}+
S_{j,cleanliness}+
S_{j,structure}
\right).
$$

该分数是当前规则下的可解释质量代理，不是已经由人工标签或训练 Loss 证明的质量真值。

### 6.3 对照评分方案

同时计算了：

- `Q_critic`：CRITIC 权重对照；
- `Q_robust`：去掉最高和最低组分后的鲁棒聚合；
- `Q_penalty`：连续冲突强度惩罚；
- `Q_hard`：极端广告风险硬惩罚。

主结果仍采用未惩罚的 `Q_baseline`，因为当前没有独立证据证明惩罚后分数更正确。

---

## 7. 从样本级到领域级的聚合

代码在 `quality_pipeline.py` 的 `domain_summary()` 中按：

```python
raw.groupby(["source", "domain"])
```

分组。因此 A1/arxiv、A2/arxiv 等被分别统计。

对于领域 $d$ 的样本集合 $D_d$，主领域评分为样本平均值：

$$
Q_d=\frac{1}{n_d}\sum_{j\in D_d}Q_j.
$$

同时计算：

- 中位数；
- 10% 截尾均值；
- Winsorized 均值；
- 200 次 bootstrap 的 95% 区间。

Bootstrap 只反映当前评分规则下的抽样波动，不包含指标方向、权重和效用函数本身的不确定性。

领域级结果见：

- `outputs/tables/domain_quality_scores.csv`

A1 主模型结果：

| 领域 | 平均 $Q$ | 排名 |
|---|---:|---:|
| arxiv | 0.7647 | 1 |
| commoncrawl | 0.7092 | 2 |
| stackexchange | 0.6959 | 3 |
| c4 | 0.6894 | 4 |
| wikipedia | 0.6477 | 5 |
| book | 0.6201 | 6 |
| github | 0.5750 | 7 |

直观图表：

- `outputs/figures/domain_quality_ci.png`

---

## 8. A1 与 A2/A3 扩展集对照

为避免重复 ID 造成虚假的独立验证，A2/A3 对照只使用不在 A1 中的记录。

结果见：

- `outputs/tables/quality_extension_agreement.csv`

| 领域 | A1 平均 $Q$ | 扩展集非重合平均 $Q$ | 差值 |
|---|---:|---:|---:|
| arxiv | 0.7647 | 0.7616 | -0.0031 |
| github | 0.5750 | 0.5741 | -0.0009 |

说明在当前评分规则下，arxiv 和 github 的扩展记录与 A1 同领域均值较接近，主要结论具有一定来源稳定性。

但 A2/A3 只对应 arxiv 和 github，不能据此证明另外五个 A1 领域也完成了扩展集验证。

---

## 9. 质量冲突及冲突处理

### 9.1 冲突定义

项目预先定义三类冲突：

1. 教育性高，但无广告概率低；
2. 流畅度高，但洁净度低；
3. 教育性高，但可读性低。

离散冲突使用 A1 的分位数阈值：一项高于 75% 分位，另一项低于 25% 分位。

连续冲突强度为：

$$
I_j=\max_gS_{jg}-\min_gS_{jg}.
$$

### 9.2 冲突处理候选

连续强度惩罚：

$$
Q_j^{pen}=\left[Q_j-0.1I_j\right]_{[0,1]}.
$$

极端广告风险硬惩罚：

$$
Q_j^{hard}=\begin{cases}
0.85Q_j,& ad\_en \text{低于 A1 第10分位},\\
Q_j,& \text{其他情况}.
\end{cases}
$$

A1 任一预定义冲突率为 7.83%。阈值改为 20%/80% 和 30%/70% 时，冲突率分别为 3.93% 和 13.62%。

### 9.3 冲突结果与证据边界

冲突分析覆盖 A1 全体，也在 A2 arxiv 和 A3 github 上进行了对照。冲突方法具有透明的应用层设计，但分位数阈值、极差强度和惩罚系数不是全新的理论方法。

由于没有人工标签或独立训练实验，项目没有把惩罚后的 $Q$ 宣称为更优模型，而是保留未惩罚 $Q$ 作为主结果，将冲突标签和强度作为辅助信息。

相关文件：

- `conflict_summary.csv`；
- `conflict_threshold_sensitivity.csv`；
- `conflict_penalty_sensitivity.csv`；
- `outputs/figures/conflict_rates.png`；
- `outputs/figures/conflict_matrix.png`。

---

## 10. 质量到训练配方的映射

### 10.1 映射文件和脚本

核心脚本：

- `problem1/quality_mixture_link.py`

映射表：

- `real_attachments/A_data_value/domain_mapping_guide.csv`

质量域和 RegMix 的 17 个训练域不一一对应，因此不是自动学习出完整映射，而是根据领域语义建立映射表。

只使用：

- `direct`：直接对应；
- `near_direct`：近似对应。

不把可信度较低的 `inferred` 映射作为主结果。

### 10.2 配方级质量特征

对于配方：

$$
\mathbf p=(p_1,\ldots,p_{17}),
$$

已映射训练域集合记为 $M$，相应质量分为 $Q_i$，则构造：

$$
Q_{mix}=\frac{\sum_{i\in M}p_iQ_i}{\sum_{i\in M}p_i},
$$

以及已映射领域的总配比：

$$
P_{mapped}=\sum_{i\in M}p_i.
$$

比较两个模型：

- `ALR_only`：只有配比特征；
- `ALR+mapped_Q`：ALR 配比加 $Q_{mix}$ 和 $P_{mapped}$。

### 10.3 结果和解释

质量增强模型在 1M 和 60M 上略有改善，但在 1B 上略有退步。因此不能说引入 $Q$ 后稳定提高了 Loss 预测效果。

更重要的是，$Q_i$ 对每个领域是固定常数，$Q_{mix}$ 由配比和固定领域质量计算出来，本质上与配比高度相关：

$$
Q_{mix}=g(\mathbf p;Q_1,\ldots,Q_{17}).
$$

因此这个实验是增量预测比较，不是质量的因果识别。现有数据缺少“同配比、不同质量”的训练观测，无法单独估计质量 $Q$ 对 Loss 的作用。

论文中应表述为：建立了部分可靠的质量域—训练域语义映射，并检验了映射质量特征的增量预测能力；不应写成证明了质量提高会导致训练效果提升。

---

## 11. 配比模型和第二版验证

虽然本总结从质量预处理开始，但问题一后续还包含 A4-A15 的配比建模。

### 11.1 第一版模型

第一版在 A4/A5 的 512 组 1M 训练配方上使用 ALR Ridge，预测 13 个验证域 Loss。

### 11.2 第二版模型

第二版使用二阶 ALR Ridge，加入平方项和两两交互项，并通过训练集内重复交叉验证选择：

- 参考域 `pile_cc`；
- $\epsilon=0.000249501$；
- Ridge $\alpha=10$。

第二版的选择没有读取 A6-A15 的 Loss 标签。

### 11.3 A6-A11 检验结果

A6-A11 的用途：

| 数据 | 配比 | Loss | 用途 |
|---|---|---|---|
| 1M | A6 | A7 | 同规模检验 |
| 60M | A8 | A9 | 跨规模检验 |
| 1B | A10 | A11 | 跨规模检验 |

第二版检验结果：

| 检验集 | RMSE | 平均偏差 | 平均 Spearman | 评价 |
|---|---:|---:|---:|---|
| 1M | 0.270 | 0.046 | 0.945 | 同规模预测较好 |
| 60M | 1.596 | 1.506 | 0.941 | 排序较好，绝对值有尺度偏差 |
| 1B | 2.702 | 2.619 | 0.877 | 数值误差较大，排序尚可 |

A6-A11 没有用于第二版拟合或调参，但第一版已经查看过这些数据，因此这不是全新封存的盲测。更准确地说，这是锁定模型在既定检验集上的同口径评估。

核验报告：

- `verification_report_v2.md`

---

## 12. 结果文件索引

| 结果内容 | 文件 |
|---|---|
| 22 项预处理规则 | `outputs/tables/quality_indicator_rules.csv` |
| 标准化后的 22 项指标 | `outputs/tables/quality_standardized.parquet` |
| 样本级质量分 | `outputs/tables/sample_quality_scores.parquet` |
| 域级质量分 | `outputs/tables/domain_quality_scores.csv` |
| A1-A3 质量审计 | `outputs/tables/quality_audit.json` |
| A1 与扩展集对照 | `outputs/tables/quality_extension_agreement.csv` |
| 冲突汇总 | `outputs/tables/conflict_summary.csv` |
| 冲突阈值敏感性 | `outputs/tables/conflict_threshold_sensitivity.csv` |
| 冲突惩罚敏感性 | `outputs/tables/conflict_penalty_sensitivity.csv` |
| 配比模型结果 | `outputs/tables/v2_vs_v1_metrics.csv` |
| 配比逐领域误差 | `outputs/tables/v2_vs_v1_per_domain.csv` |
| 第二版嵌套交叉验证 | `outputs/tables/v2_nested_cv_summary.csv` |
| 第二版核验结果 | `verification_report_v2.md` |

---

## 13. 最终评价

### 已经解决的部分

- A1-A3 全量质量信号已经读取和预处理；
- 22 个指标已经统一尺度和方向；
- 已建立样本级和领域级质量代理分数；
- 已对 A1 与 A2/A3 的部分同域扩展进行对照；
- 已定义质量冲突并完成覆盖抽样集的分析；
- 已建立配比到 Loss 的预测模型并在 A6-A11 上检验。

### 尚未完全解决的部分

- $Q$ 缺少独立人工质量真值；
- 冲突惩罚没有外部验证；
- 质量对训练 Loss 的独立作用无法识别；
- 60M 和 1B 的绝对 Loss 仍有明显尺度偏差；
- 10B/70B 标签是估算值，不能作为真实大模型验证；
- 配比置换效应属于模型条件关联，不是因果效应。

### 建议的论文结论

> 本文基于 A1-A3 的 272,505 条质量信号，完成了 22 个质量指标的标量化、方向统一和标准化处理，构造了样本级综合质量代理 $Q$，并通过来源域聚合得到领域级质量评分。A2/A3 的非重合扩展记录与 A1 同域均值较为接近，说明评分规则具有一定来源稳定性。进一步地，本文建立了 17 域配比到 13 个验证域 Loss 的 ALR-Ridge 基线，并通过二阶 ALR Ridge 改善了 1M 配比预测。但由于缺少人工质量真值和同配比异质量训练实验，$Q$ 尚不能解释为已验证的真实训练价值，也不能据此推出质量的独立因果效应。
