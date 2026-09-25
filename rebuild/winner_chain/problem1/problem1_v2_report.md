# 问题一第二版：建模方案与适用范围

## 1. 本次实际优化

质量指标的方向、固定 A1 标尺、四组等权 Q 和冲突标记继续沿用第一版。缺少独立人工标签与同配比异质量训练实验，目前没有证据可以把某个新权重或冲突惩罚宣称为更优，因此不改写 Q。第二版把可验证的改进集中在 17 域配比到 13 域 Loss 的预测，并扩展分域与效应稳定性检验。

## 2. 数据使用规则

A4/A5：512 条 1M 配方及 Loss，用于训练和选型；A6/A7：256 条 1M 既定测试配方，主要性能判据；A8–A11：60M、1B 实测跨规模迁移检验；A12–A15：10B/70B 估算标签，仅作附加诊断。测试集和估算集的 Loss 均未用于第二版数值拟合、超参数选择或规模截距拟合，但其结果已在第一版中被查看，因此并非全新封存的盲测。

## 3. 模型

设配比向量 $\mathbf p$ 的 17 个分量和为 1。先作加性对数比 $x_i=\log[(p_i+\epsilon)/(p_r+\epsilon)]$，再用包含一次项、平方项和两两乘积的二阶特征 $\phi(\mathbf x)$ 建立 13 输出 Ridge：$\widehat{\mathbf L}=\mathbf b+\mathbf B\,\mathrm{standardize}(\phi(\mathbf x))$。

本轮由训练集 2×5 折 CV 选出：参考域 `pile_cc`、ε=0.000249501、Ridge α=10。选择依据为训练集总体 RMSE，既定测试集结果见 [第一版与第二版对比](problem1_v1_v2_comparison.md)。

## 4. 质量代理与人工核验

`outputs/v2_review/人工盲评模板.xlsx` 包含 140 条 A1 文本；每个来源域按 Q 的四分位抽样，并额外纳入冲突样本。评审表只含随机编号和文本，不显示 Q、冲突状态或领域；`outputs/tables/v2_blind_review_key.json` 单独保存对照。两名评审者各复制模板独立填写五项 1–5 评分和问题标记。比较总体分与 Q 的 Spearman，并报告评审一致性；事实问题允许留空。该表目前未填写，因此人工有效性尚未获得。

## 5. 领域置换结论的稳定性

在训练集上固定二阶 ALR Ridge 的 α=10，交叉比较 3 个参考域和 3 个 ε，共 9 种规格。以平均配比为中心，从 `pile_cc` 转移 1 个百分点至其他域时，13/16 个目标域的预测 Loss 变化符号一致。`gutenberg_pg_19, pubmed_abstracts, pubmed_central` 的方向依赖规格，不宜写成稳定领域效应。置换只是模型的局部预测，不能解释为真实训练因果效应。

## 6. 尚需的外部证据

目前没有人工质量真值，无法证明 Q 的新旧优劣；没有同配比异质量的训练观测，无法识别质量对 Loss 的独立作用；仅有 1M 的训练标签。`v2_scale_transfer_summary.csv` 给出最终 V2 的回顾性幅度诊断，1B 的配方均值排序弱于 V1。不能从 A8–A11 测试标签拟合规模校准后仍宣称独立测试。若补充规模训练实验或预先划分校准集，应另立验证协议。

## 7. 复现

依次运行 `python problem1/problem1_v2.py`、`python problem1/nested_v2_validation.py`、`python problem1/prepare_v2_review.py`、`python problem1/build_v2_report.py`、`python problem1/verify_problem1_v2.py`。质量原始处理仍按 `README.md` 中第一版流程执行。模型保存在 `outputs/models/mixture_v2.joblib`；候选与比较数据保存在 `outputs/tables/v2_*.csv`。盲评模板需要用 `.v2_artifact/build_review.mjs` 和 bundled artifact-tool 重新生成；现有文件已可直接供两名评审者复制填写。
