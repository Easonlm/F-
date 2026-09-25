"""Build paper-ready figures and data-backed reports from generated tables."""
from __future__ import annotations

import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_pipeline import ROOT, OUT, BENCH, norm, save

HERE=ROOT/"problem4"
FIG=HERE/"outputs"/"figures"
FIG.mkdir(parents=True,exist_ok=True)


def tbl(name):return pd.read_csv(OUT/name)
def fnum(x,d=2):return "not estimable" if not np.isfinite(x) else f"{x:.{d}f}"


def _plot(name,title,draw):
    fig,ax=plt.subplots(figsize=(8,4.8));draw(ax);ax.set_title(title);ax.grid(alpha=.25);fig.tight_layout();fig.savefig(FIG/name,dpi=160);plt.close(fig)


def figures():
    a=tbl("analysis_dataset.csv");a["submission_date"]=pd.to_datetime(a.submission_date)
    q=tbl("quarterly_frontier.csv");cc=tbl("c4_open_compute_frontier.csv")
    cont=tbl("scale_technology_contributions.csv");fc=tbl("frontier_forecast_12m_24m.csv")
    bridge=tbl("loss_benchmark_bridge_models.csv");c6=pd.read_csv(ROOT/"real_attachments/C_efficiency_evolution/loss_benchmark_bridge_expanded.csv")
    c8=tbl("c8_vs_leaderboard.csv");tf=tbl("c8_task_family_frontiers.csv");me=tbl("data_vs_mechanistic_forecast.csv")
    def scatter_time(ax):
        for k,g in a.groupby("model_type"):ax.scatter(g.submission_date,g.ability,s=5,alpha=.35,label=k)
        ax.set_ylabel("Six benchmark mean, points");ax.legend(fontsize=7)
    _plot("ability_over_time.png","Leaderboard ability over time",scatter_time)
    def lines(ax,subsets,log=False):
        for k in subsets:
            z=q[q.subset==k].sort_values("quarter")
            if len(z):ax.plot(z.quarter,z.p95_ability,marker="o",label=k)
        ax.set_ylabel("Quarterly p95 ability");ax.legend(fontsize=8)
    _plot("open_frontier_over_time.png","Open-weight frontier and C2 sensitivity",lambda ax:lines(ax,["all","open_w1","open_w2","c2_inherited_open_sensitivity"]))
    _plot("pretrained_vs_chat_frontier.png","Verified open model types",lambda ax:lines(ax,["open_pretrained","open_chat_finetuned"]))
    def compute(ax):
        ax.plot(cc.quarter,cc.p90_compute,marker="o");ax.set_yscale("log");ax.tick_params(axis="x",rotation=70);ax.set_ylabel("C4 open language p90 training FLOP")
    _plot("compute_frontier_over_time.png","C4 training compute frontier",compute)
    valid=cont[cont.window.isin(["observed_span","last_two_quarters"])]
    def bars(ax):
        x=np.arange(len(valid));ax.bar(x-.18,valid.scale_points,width=.36,label="scale");ax.bar(x+.18,valid.technology_points,width=.36,label="residual time");ax.set_xticks(x,valid.window);ax.set_ylabel("Model implied ability points");ax.legend()
    _plot("scale_vs_technology_contribution.png","Exploratory decomposition",bars)
    ti=tbl("technology_trend.csv")
    _plot("technology_index_over_time.png","Residual time index",lambda ax:(ax.plot(ti.quarter,ti.technology_score_equivalent,marker="o"),ax.set_ylabel("Score equivalent at fixed compute")))
    _plot("compute_equivalent_technology_gain.png","Compute equivalent of time term",lambda ax:(ax.plot(ti.quarter,ti.compute_equivalent_multiplier,marker="o"),ax.set_ylabel("Compute multiplier")))
    bm=json.loads((OUT/"loss_benchmark_bridge_model.json").read_text())
    from bridge_model import predict
    xx=np.linspace(c6.Val_Loss.min(),c6.Val_Loss.max(),200)
    def bridgeplot(ax):
        ax.scatter(c6.Val_Loss,c6.LB_Average,s=15,alpha=.5);ax.plot(xx,predict(bm,xx),color="red");ax.set_xlabel("Validation loss");ax.set_ylabel("Six benchmark mean")
    _plot("loss_benchmark_bridge.png","C6 loss to ability bridge",bridgeplot)
    def bycomp(ax):
        for k,g in c6.groupby("Loss_Comparability"):ax.scatter(g.Val_Loss,g.LB_Average,s=18,alpha=.6,label=k.split(" ")[0])
        ax.legend();ax.set_xlabel("Validation loss");ax.set_ylabel("Six benchmark mean")
    _plot("bridge_by_comparability.png","Bridge comparability strata",bycomp)
    _plot("c8_detailed_vs_sixbench.png","Task detail versus six benchmark mean",lambda ax:(ax.scatter(c8.ability,c8.detailed_ability,s=5,alpha=.25),ax.set_xlabel("Six benchmark mean"),ax.set_ylabel("Mean leaf-task percentile")))
    def task(ax):
        for k,g in tf.groupby("family"):ax.plot(g.quarter,g.p95,marker="o",label=k)
        ax.legend(ncol=3,fontsize=7);ax.set_ylabel("Task family p95")
    _plot("task_family_frontiers.png","C8 task family frontiers",task)
    main=fc[fc.tech_scenario=="continuation"]
    def pred(ax):
        for k,g in main.groupby("compute_scenario"):
            g=g.sort_values("horizon_months");ax.plot(g.horizon_months,g.median_ability,marker="o",label=k)
            ax.fill_between(g.horizon_months.to_numpy(float),g.p95_low.to_numpy(float),g.p95_high.to_numpy(float),alpha=.08)
        ax.set_xlabel("Months after observed origin");ax.set_ylabel("Ability points");ax.legend(fontsize=8)
    _plot("forecast_12m_24m.png","Exploratory forecasts with bootstrap intervals",pred)
    def scenario(ax):
        for k,g in fc.groupby("tech_scenario"):
            g=g[g.compute_scenario=="moderate_slowdown"].sort_values("horizon_months")
            ax.plot(g.horizon_months,g.median_ability,marker="o",label=k)
        ax.set_xlabel("Months");ax.set_ylabel("Ability points");ax.legend(fontsize=8)
    _plot("forecast_scenarios.png","Technology trend assumptions",scenario)
    def compare(ax):
        m=me[me.compute_scenario=="moderate_slowdown"].sort_values("horizon_months")
        ax.plot(m.horizon_months,m.median_ability,marker="o",label="data driven")
        ax.plot(m.horizon_months,m.benchmark,marker="o",label="P1-3 mechanism + C6")
        ax.set_xlabel("Months");ax.set_ylabel("Ability points");ax.legend()
    _plot("data_vs_mechanistic_forecast.png","Two forecast pathways",compare)
    def ceiling(ax):
        ax.bar(BENCH,[(a[k]>=80).mean()*100 for k in BENCH]);ax.tick_params(axis="x",rotation=35);ax.set_ylabel("Share scoring 80+ (%)")
    _plot("benchmark_ceiling_analysis.png","Benchmark ceiling audit",ceiling)
    fig,axes=plt.subplots(1,3,figsize=(14,4.8))
    z=q[q.subset=="open_w1"].sort_values("quarter")
    zb=q[q.subset=="c2_inherited_open_sensitivity"].sort_values("quarter")
    axes[0].plot(z.quarter,z.p95_ability,marker="o",label="verified open")
    axes[0].plot(zb.quarter,zb.p95_ability,marker="s",label="C2 inherited sensitivity")
    axes[0].set_title("Observed frontier");axes[0].set_ylabel("Six benchmark mean");axes[0].legend(fontsize=7)
    de=cont[cont.window=="observed_span"].iloc[0]
    axes[1].bar(["scale","residual time","observed change"],[de.scale_points,de.technology_points,de.observed_frontier_change],color=["#e69f00","#56b4e9","#666666"])
    axes[1].axhline(0,color="black",lw=.8);axes[1].set_title("Decomposition diagnostic");axes[1].set_ylabel("Ability points");axes[1].tick_params(axis="x",rotation=25)
    for k,g in main.groupby("compute_scenario"):
        g=g.sort_values("horizon_months")
        axes[2].plot(g.horizon_months,g.median_ability,marker="o",label=k)
    mm=me[me.compute_scenario=="moderate_slowdown"].sort_values("horizon_months")
    axes[2].plot(mm.horizon_months,mm.benchmark,marker="D",ls="--",color="black",label="P1–3 E1 mechanism")
    axes[2].set_title("Exploratory future");axes[2].set_xlabel("Months after origin");axes[2].set_ylabel("Six benchmark mean");axes[2].legend(fontsize=7)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle("Historical decomposition and forecast: strict sample is too sparse for validation",fontsize=12)
    fig.tight_layout();fig.savefig(FIG/"historical_decomposition_and_forecast.png",dpi=160);plt.close(fig)


def architecture(a,c7):
    left=a.copy();left["key"]=left.Model.map(norm)
    right=c7.copy();right["key"]=right.model_name.map(norm)
    m=left.merge(right,on="key",how="inner").drop_duplicates("leaderboard_row")
    m["context"]=pd.to_numeric(m.max_position_embeddings,errors="coerce")
    if len(m)>=5:
        m["residual_ability"]=m.ability-m.groupby("model_type").ability.transform("mean")
    save(m[["Model","context","n_layers","n_heads","d_model","ability","model_type"]],"c7_architecture_matches.csv")
    return len(m)


def build(a,inputs,c8summary,c8matched,history,b,bridgecv,origin):
    figures();c7n=architecture(a,inputs["c7"])
    q=tbl("quarterly_frontier.csv");g=tbl("compute_growth_models.csv");c=tbl("scale_technology_contributions.csv")
    f=tbl("frontier_forecast_12m_24m.csv");m=tbl("data_vs_mechanistic_forecast.csv");bt=tbl("forecast_backtest.csv")
    short=tbl("short_term_baseline_comparison.csv")
    hc=tbl("historical_model_comparison.csv");task=tbl("c8_task_family_frontiers.csv")
    strict=a[a.open_w1].sort_values(["submission_date","ability"]).drop_duplicates("Model",keep="last");r=c[c.window=="observed_span"].iloc[0]
    cv=bridgecv[(bridgecv.source=="C6")&(bridgecv.comparability=="weighted")&(bridgecv.target=="LB_Average")&(bridgecv.kind=="logistic")]
    cv_rmse=float(np.sqrt(np.average(cv.rmse**2,weights=cv.n_test)))
    corr=c8matched[["ability","detailed_ability"]].corr().iloc[0,1]
    rank_corr=c8matched[["ability","detailed_ability"]].corr(method="spearman").iloc[0,1]
    c2n=int(a.c2_matched.sum())
    recent=g[g.model=="recent_2023_2024"].iloc[0]
    overall=g[g.model=="single_loglinear"].iloc[0]
    piece=g[g.model=="piecewise"].iloc[0]
    fmain=f[f.tech_scenario=="continuation"]
    forecast_lines=[]
    for _,z in fmain.iterrows():forecast_lines.append(f"| {z.horizon_months} | {z.compute_scenario} | {z.median_ability:.2f} | [{z.p80_low:.2f}, {z.p80_high:.2f}] | [{z.p95_low:.2f}, {z.p95_high:.2f}] |")
    mechanism=m[m.compute_scenario=="moderate_slowdown"]
    t0=task[task.quarter==task.quarter.min()].set_index("family").p95
    t1=task[task.quarter==task.quarter.max()].set_index("family").p95
    task_gain=((t1-t0)*100).sort_values(ascending=False)
    ability_q=q[q.subset=="open_w1"].sort_values("quarter")
    broad_q=q[q.subset=="c2_inherited_open_sensitivity"].sort_values("quarter")
    sensitivity=tbl("frontier_definition_sensitivity.csv")
    strict_one=sensitivity[(sensitivity.subset=="open_w1")&(sensitivity.window_quarters==1)].sort_values("end_quarter")
    strict_five=strict_one[strict_one.eligible_n5]
    strict_ten=strict_one[strict_one.eligible_n10]
    strict_three=sensitivity[(sensitivity.subset=="open_w1")&(sensitivity.window_quarters==3)].sort_values("end_quarter")
    broad_aligned=sensitivity[(sensitivity.subset=="c2_inherited_aligned")&(sensitivity.window_quarters==1)].sort_values("end_quarter")
    b12=bt[bt.horizon_months==12];b24=bt[bt.horizon_months==24]
    b3=bt[bt.horizon_months==3]
    b3rmse=np.sqrt(np.mean(b3.error**2)) if len(b3) else np.nan
    short_direct=short[short.model=="direct_time"]
    short_persistence=short[short.model=="persistence"]
    short_pair=short_direct.merge(short_persistence,on=["horizon_months","origin_quarter","target_quarter"],
                                  suffixes=("_direct","_persistence"),validate="one_to_one")
    short_pair=short_pair.sort_values(["horizon_months","origin_quarter"])
    short_lines=[f"| {row.origin_quarter}→{row.target_quarter} | {int(row.horizon_months)} | {int(row.target_n_direct)} | {row.abs_error_direct:.2f} | {row.abs_error_persistence:.2f} |"
                 for _,row in short_pair.iterrows()]
    persist3=short_persistence[short_persistence.horizon_months==3]
    direct3=short_direct[short_direct.horizon_months==3]
    qualified_persist=short_persistence[short_persistence.target_n_ge_5]
    pca_corr=a[["ability","pca_ability"]].corr().iloc[0,1]
    model_bank=json.loads((OUT/"loss_benchmark_all_models.json").read_text(encoding="utf-8"))
    from bridge_model import predict
    bridge_at_2={label:float(predict(model_bank[f"C6:{label}:LB_Average:logistic"],[2.0])[0]) for label in ["weighted","high_only","all_unweighted"]}
    date_delta=a.loc[a.accepted,"date_diff_days"].dropna()
    med_date_delta=float(date_delta.median()) if len(date_delta) else np.nan
    weak_corr=c8matched[["ability","weak_ability"]].corr().iloc[0,1]
    c3=tbl("c3_timeseries_provenance.csv")
    report=f"""# 问题四：技术演进分析与前沿预测

## 1 问题分析
用训练规模和控制规模后的时间项描述能力差异，并用开放权重模型的能力前沿构造算力放缓情景。由于直接榜单历史只有九个月，贡献与 12/24 月预测均是**探索性结构情景**，没有足够历史作对应期限的回测。

## 2 数据来源与可信度
C1 有 {len(inputs['c1'])} 行；C2 给出 {c2n} 个 Epoch 字段候选，但含大量微调模型继承基础模型元数据，不能当作直接实体匹配。严格核对模型名、Hugging Face 发布方、参数及时间后接受 {int(a.accepted.sum())} 行（{a.accepted.mean()*100:.2f}%）。C3 有 {len(inputs['c3'])} 行，其中 {int((inputs['c3'].Source=='Historical (papers/reports)').sum())} 行历史报告/补全记录；其年度 p95 见 `outputs/tables/c3_timeseries_provenance.csv`，与直接观测分开。C4 还有推测记录，主规模样本排除 Speculative。C7 直接名称对应 {c7n} 行，仅作架构描述。C5 为 {len(inputs['c5'])} 行，C6 为 {len(inputs['c6'])} 行。

## 3 开源、模型类型和时间口径
主口径 Open-W1：C4 `Open model weights?=Yes` 且经过严格发布方核对、六基准完整。得到 {len(strict)} 个独立模型，pretrained {int((strict.model_type=='pretrained').sum())}、chat/finetuned {int((strict.model_type=='chat_finetuned').sum())}、其他 {int((strict.model_type=='other').sum())}。Open-W2 另要求存在可明确归为 permissive 或 research-only 的许可证标签，共 {int(a.loc[a.open_w2,'Model'].nunique())} 个模型；法律许可仍须逐条核验。主时间为 Submission Date。开放主样本截止日/预测原点为 **{origin.date()}**，12/24 月目标日为 **{(origin+pd.DateOffset(months=12)).date()} / {(origin+pd.DateOffset(months=24)).date()}**。严格匹配记录的提交日期减发布日期中位数为 {fnum(med_date_delta,0)} 天；日期差值分布见 `open_source_filter_audit.csv`。C4 未来模型绝不进入历史。

## 4 C8 逐任务聚合
扫描 {c8summary['json_files']} 个 JSON、{c8summary['model_directories']} 个目录，成功解析 {c8summary['parsed_models']} 个模型，{c8summary['failed_models']} 个目录不可用，{c8summary['failed_files']} 个损坏文件记录于 `c8_parse_failures.csv`。逐任务使用 {c8summary['task_count']} 个叶任务、{c8summary['usable_task_model_pairs']} 个模型-任务分数；固定每类可比 metric，逐任务 percentile 后均值/第25百分位，覆盖至少半数任务及三个任务族才合格。与 C1 六维均值 Pearson={corr:.3f}、Spearman={rank_corr:.3f}；弱项指标 Pearson={weak_corr:.3f}。汇总有较高代表性，个别模型仍有排序偏差。六任务族前沿从首至末季度变化（百分点）：{', '.join(f'{k} {v:+.1f}' for k,v in task_gain.items())}。这些来自不断变化的提交模型集合，不是同一模型的纵向改进。

## 5 综合能力指标
主指标 `S=(IFEval+BBH+MATH Lvl 5+GPQA+MUSR+MMLU-PRO)/6`，0–100 分。六项均完整，无均值填补。稳健 z 分与 PCA 第一成分是敏感性，主分与 PCA 得分相关 {pca_corr:.3f}。预测使用 logit(S/100) 保证边界。

## 6 Loss–Benchmark 桥接
C6 高可比 7 行、中可比 68 行。比较线性、logistic、isotonic；按模型族 GroupKFold，主桥接用高权重1、中权重0.35的有界单调 logistic，组外 RMSE={cv_rmse:.2f} 分。权重是敏感性设定，非经验真值。在 Loss=2.0 处，weighted/high-only/all-unweighted 的 logistic 映射分别为 {bridge_at_2['weighted']:.2f}/{bridge_at_2['high_only']:.2f}/{bridge_at_2['all_unweighted']:.2f} 分。C5、各可比性层和六个单项模型均见桥接表；high-only 仅一个主要系列，无法可靠外推。桥接误差较大，机理预测不能解释为精确分数。

## 7 历史能力前沿
开放权重六维完整模型的季度 p95：{', '.join(f'{z.quarter}={z.p95_ability:.2f}(n={z.n})' for _,z in ability_q.iterrows())}。同时给出 all/W2、pretrained/chat 的 p90、p95、Top3、Top1，见 `quarterly_frontier.csv`。最后一季只有 {int(ability_q.iloc[-1].n)} 个严格匹配模型，前沿受样本选择影响很大。C2 继承匹配的宽口径敏感性共 {int(broad_q.n.sum())} 个模型，首末季 p95 从 {broad_q.iloc[0].p95_ability:.2f} 到 {broad_q.iloc[-1].p95_ability:.2f}；方向与严格口径相反，但其实体对应不能逐项确认。

`frontier_definition_sensitivity.csv` 将单季最低样本数预设为 5/10，并列 1–3 季完整滚动窗口及相同 W1 截止日期的 C2 宽口径。严格单季 n≥5 只比较至 {strict_five.iloc[-1].end_quarter}，相对首季变化 {strict_five.iloc[-1].p95_ability-strict_five.iloc[0].p95_ability:+.2f} 分；n≥10 只比较至 {strict_ten.iloc[-1].end_quarter}，变化 {strict_ten.iloc[-1].p95_ability-strict_ten.iloc[0].p95_ability:+.2f} 分；三季窗口 {strict_three.iloc[0].end_quarter}→{strict_three.iloc[-1].end_quarter} 为 {strict_three.iloc[-1].p95_ability-strict_three.iloc[0].p95_ability:+.2f} 分。统一截止日期的 C2 单季首末为 {broad_aligned.iloc[-1].p95_ability-broad_aligned.iloc[0].p95_ability:+.2f} 分。这些比较的起止窗口并不相同，只用于判断方向是否依赖定义；末季严格 n<5，**不据此解释当季前沿涨跌**。

## 8 规模变量与技术变量
主模型在 {len(history)} 个独立基础发布上拟合 `logit(S/100)=beta0+betaC(log10 C-23)+betaT t`，betaC={b[1]:.3f}、betaT={b[2]:.3f}。C4 参数 N、训练数据 D、reported compute 均已读取；N+D+time 替代模型仅使用 D 完整子样本，见比较表。三个主候选的跨组织/家族验证误差：{', '.join(f'{z.model} {z.group_cv_rmse:.2f}' for _,z in hc.iterrows())}。结构模型没有优于 compute-only，时间项的识别弱。

## 9 规模/技术贡献分解
以首末季度 p95 为锚，对固定算力和固定时间两条路径作对称 Shapley。模型隐含总变化 {r.total_points:.2f} 分，其中算力 {r.scale_points:.2f} 分（{r.scale_share*100:.1f}%）、剩余时间项 {r.technology_points:.2f} 分（{r.technology_share*100:.1f}%）；组织 cluster bootstrap 95% 区间分别 [{r.scale_points_lo95:.2f},{r.scale_points_hi95:.2f}] 和 [{r.technology_points_lo95:.2f},{r.technology_points_hi95:.2f}]。**实际观测 p95 变化 {r.observed_frontier_change:.2f} 分，与模型隐含方向不一致，因此不能把上述百分比当作稳健主结论或因果份额。** 历史不满12个月，最近12/24个月贡献无法计算。技术等效算力由 `10^(betaT Δt/betaC)` 求出，仅作为敏感性。

## 10 pretrained vs chat
开放权重主样本 pretrained {int((strict.model_type=='pretrained').sum())}、chat/finetuned {int((strict.model_type=='chat_finetuned').sum())}；同一模型在 C1 存在类型冲突时，按同日较高能力提交去重，分类本身存在不确定性。有 reported compute 的 chat 模型过少，不能可靠分开估计 chat 的规模/技术贡献。两类能力前沿已分别列出和绘图。

## 11 算力前沿及增长变化
C4 对所有公开权重 Language 模型筛选 Confident/Likely、reported FLOP、发布日期在原点以前，2025Q1 为不完整季度而排除。2021–2024 的单斜率年对数增长 {overall.slope_log_per_year:.3f}（约 {overall.annual_multiplier:.2f} 倍/年）；2023–2024 近期斜率 {recent.slope_log_per_year:.3f}（约 {recent.annual_multiplier:.2f} 倍/年）。piecewise BIC={piece.bic:.2f}，单斜率 BIC={overall.bic:.2f}，未形成可靠的已放缓证据；近期 p90 波动较大。

## 12 算力增长放缓情景
情景 S1/S2/S3 将近期对数增长率乘 1/0.5/0.25；技术 T1/T2/T3 将剩余时间斜率乘 1/0.5/0。倍率是假设而非概率。以 C4 最后完整季度的 p90 FLOP 为起点；情景包络不是统计置信区间。

## 13 未来12/24月预测
下表是 T1 下的探索性有界前沿；区间是参数 bootstrap 加样本残差的条件区间，未涵盖未来数据源与制度变化。能力锚取严格 W1 末季，仅 {int(ability_q.iloc[-1].n)} 个模型，`frontier_forecast_12m_24m.csv` 已逐行标记该样本不足状态。没有12/24个月直接观测回测，不得用作可信的点预测。

| 月数 | 算力情景 | 点估计 | 条件80%区间 | 条件95%区间 |
|---:|---|---:|---|---|
{chr(10).join(forecast_lines)}

## 14 前三问机制预测
问题一采用 P1 观测支持域配比作敏感性；问题二读取 V2 Model C 参数；问题三 V2 在 E0/E1/E2 约束下把未来算力转为 N/D/Q 与 Loss；C6 logistic 桥接转为分数。主对照使用 P0 基准配比、E1 3× 边界。E1 在这些算力情景达到 N/D 上限，故其能力预测近乎不随算力变化，属于资源边界假设的结果。P1、E2 及超出 C6 Loss 范围均已标记。

## 15 两类预测对照
中度放缓 12/24 月：数据驱动 {mechanism.iloc[0].median_ability:.2f}/{mechanism.iloc[1].median_ability:.2f}，机制 {mechanism.iloc[0].benchmark:.2f}/{mechanism.iloc[1].benchmark:.2f}，差 {mechanism.iloc[0].difference_data_minus_mechanism:.2f}/{mechanism.iloc[1].difference_data_minus_mechanism:.2f} 分。机理表另给桥接 CV RMSE 推出的误差带，此带没有覆盖率保证。主要不一致来自小样本时间趋势外推、C6 Loss 定义/后训练差异及 E1 N/D 上限；不人为校准相等。

## 16 不确定性与回测
组外桥接 RMSE {cv_rmse:.2f} 分。可用季度数据只有 {len(ability_q)} 季，直接时间模型 3 个月滚动回测 {len(b3)} 折，RMSE={fnum(b3rmse)} 分。另以每个预测原点当季观测 p95 不变为**朴素短期持平基线**，在相同原点与目标季度比较：

| 原点→目标 | 提前月数 | 目标季 n | 直接时间趋势绝对误差 | 持平基线绝对误差 |
|---|---:|---:|---:|---:|
{chr(10).join(short_lines)}

3 个月的 {len(direct3)} 个目标上，直接趋势 MAE={direct3.abs_error.mean():.2f} 分，持平 MAE={persist3.abs_error.mean():.2f} 分；但全部 {len(short_persistence)} 个回测对里，要求目标季 n≥5 后仅剩 {len(qualified_persist)} 个，无法据此升级长期模型。持平基线只作为短期预测的风险参照，**不替换 12/24 个月数据驱动或机制路径**。12个月 {len(b12)} 折、24个月 {len(b24)} 折；长期区间不是经历史覆盖率校准的预测区间。bootstrap 组织重采样、算力与技术情景、桥接分层、问题三外推分别列出，不能将混合范围叫作95%置信区间。逐行对照见 `short_term_baseline_comparison.csv`。

## 17 Benchmark 饱和与逐任务差异
六维榜单没有模型达到任一项90分，当前样本无直接 ceiling；任务族 IFEval 高于其他部分任务，未来有界 logit 仍必要。C8 的 math 子任务族前沿增长最明显，但季度模型构成变化可能造成选择偏差。

## 18 模型优缺点
优点：真实解析逐任务 JSON，C2 进行反核对，C4 仅接受高置信直接匹配，所有路径可复现，情景和条件区间分开。限制：榜单直接历史仅九个月；严格实体匹配率约1%；季度末开放样本只有{int(ability_q.iloc[-1].n)}个；C6 高可比仅7行；Loss 尺度未跨模型族校准；Q/p 跨尺度未识别；未来12/24个月无法回测。残差时间趋势包含选择、工程和评测适配，不能称严格因果效应。

## 19 结论
能稳健写入论文的是数据口径、逐任务验证、桥接误差、算力长期增长及无法验证长期预测的限制。{r.scale_points:.2f}/{r.technology_points:.2f} 分贡献与未来点预测只能列为明确标注的小样本敏感性情景，不能作为确定性的历史份额或未来承诺。
"""
    pred_lookup={(int(z.horizon_months),z.compute_scenario):z for _,z in fmain.iterrows()}
    last_tech=tbl("technology_trend.csv").iloc[-1].compute_equivalent_multiplier
    recent_decomp=c[c.window=="last_two_quarters"].iloc[0]
    qa=[
        "六项 Benchmark 0–100 等权平均；预测用 logit 尺度。",
        f"C8 成功 {c8summary['parsed_models']} 个模型目录；{c8summary['failed_models']} 个目录失败、{c8summary['failed_files']} 个损坏 JSON 已记录。",
        f"Pearson {corr:.3f}，Spearman {rank_corr:.3f}。",
        "Open-W1=C4 明确开放权重+原发布方严格匹配+六项完整；W2再要求明确许可标签。",
        f"{len(strict)} 个独立模型。",
        f"pretrained {int((strict.model_type=='pretrained').sum())}；chat/finetuned {int((strict.model_type=='chat_finetuned').sum())}；其他 {int((strict.model_type=='other').sum())}。",
        "Submission Date；Release Date 作匹配与敏感性。",
        f"严格直接匹配 {int(a.accepted.sum())}/{len(a)}={a.accepted.mean()*100:.2f}%；C2 候选 {c2n}/{len(a)}={c2n/len(a)*100:.2f}%，含继承关联。",
        f"C4 年度 compute 倍率：长期 {overall.annual_multiplier:.2f}×，2023–24 近期 {recent.annual_multiplier:.2f}×。",
        f"未发现稳健放缓：piecewise BIC {piece.bic:.2f}，单斜率 BIC {overall.bic:.2f}。",
        f"严格主样本首末季度 p95 数值差 {r.observed_frontier_change:.2f} 分；末季 n={int(ability_q.iloc[-1].n)}<5，不能判定趋势方向。",
        f"结构模型隐含算力贡献 {r.scale_points:.2f} 分，仅探索性。",
        f"结构模型隐含剩余时间贡献 {r.technology_points:.2f} 分，仅探索性。",
        f"模型内比例 {r.scale_share*100:.1f}%/{r.technology_share*100:.1f}%，但模型与观测方向冲突，不可作稳健份额。",
        f"最近12/24月无法估计；可观测近两季的模型项为算力 {recent_decomp.scale_points:.2f}、时间 {recent_decomp.technology_points:.2f} 分。",
        f"不同：chat/finetuned 有{int((strict.model_type=='chat_finetuned').sum())}个严格开放模型、reported compute 更少，无法单独识别贡献。",
        "C6 可比性加权、单调有界 logistic。",
        f"GroupKFold 的 C6 主桥接 RMSE {cv_rmse:.2f} 分。",
        f"不一致：Loss=2.0 的 weighted/high-only/all-unweighted 映射 {bridge_at_2['weighted']:.2f}/{bridge_at_2['high_only']:.2f}/{bridge_at_2['all_unweighted']:.2f} 分。",
        f"六维与逐任务相关高，但弱项相关 {weak_corr:.3f}，个别排名偏差见 `c8_vs_leaderboard.csv`。",
        str(origin.date()),str((origin+pd.DateOffset(months=12)).date()),str((origin+pd.DateOffset(months=24)).date()),
        f"延续算力：12月 {pred_lookup[(12,'continuation')].median_ability:.2f}、24月 {pred_lookup[(24,'continuation')].median_ability:.2f} 分。",
        f"中度放缓：12月 {pred_lookup[(12,'moderate_slowdown')].median_ability:.2f}、24月 {pred_lookup[(24,'moderate_slowdown')].median_ability:.2f} 分。",
        f"强放缓：12月 {pred_lookup[(12,'strong_slowdown')].median_ability:.2f}、24月 {pred_lookup[(24,'strong_slowdown')].median_ability:.2f} 分。",
        "各情景50/80/95%条件区间详见 `frontier_forecast_12m_24m.csv`；未经12/24月覆盖率校准。",
        f"12/24月无有效滚动回测；3月有 {len(b3)} 折，原趋势 RMSE {fnum(b3rmse)} 分。持平短期基线见 `short_term_baseline_comparison.csv`，合格目标 n≥5 仅 {len(qualified_persist)} 个。",
        f"P1/P2-V2/P3-V2 在中度放缓、E1/P0 下12/24月均 {mechanism.iloc[0].benchmark:.2f} 分，受资源上限钳制。",
        f"中度放缓 12/24 月数据驱动减机理：{mechanism.iloc[0].difference_data_minus_mechanism:.2f}/{mechanism.iloc[1].difference_data_minus_mechanism:.2f} 分。",
        "C6 Loss 尺度与后训练能力不完全可比，且 E1 达 N/D 上限；数据趋势超出九个月观测。",
        f"本窗口的剩余时间项约等效 {last_tech:.2f}× compute；不稳定，仅敏感性。",
        "榜单六项均无人达90分；未观测到明显 ceiling，logit 用于防止未来越界。",
        f"C8 p95 增量最大任务族为 {task_gain.index[0]}（{task_gain.iloc[0]:+.1f} 个百分点）；样本构成会变化。",
        "可作主结论：数据来源审计、C8逐任务代表性、桥接误差、C4长期算力趋势、不可回测的限制。",
        "规模/技术百分比、C2继承匹配前沿、12/24月分数、P1配比收益和E2均只能作敏感性。",
        "直接榜单九个月且严格匹配稀疏；最后季度仅2个开放模型，长期预测无回测。",
        "验证中未通过的是12月/24月直接回测与分解方向一致性；具体数量见 verification_results.json。",
    ]
    assert len(qa)==38
    qa_text="\n## 20 对 38 个交付问题的逐项回答\n\n"+"\n".join(f"{i}. {answer}" for i,answer in enumerate(qa,1))+"\n"
    report+=qa_text
    (HERE/"problem4_report.md").write_text(report,encoding="utf-8")
    summary=f"""# 问题四摘要

- 主能力：六项榜单任务等权均值（0–100）；预测在 logit 尺度上进行。
- 开放主口径：C4 公开权重、原发布方直接匹配、六项完整；{len(strict)} 个，pretrained {int((strict.model_type=='pretrained').sum())}、chat/finetuned {int((strict.model_type=='chat_finetuned').sum())}。
- 时间：Submission Date；原点 {origin.date()}；目标 {(origin+pd.DateOffset(months=12)).date()} / {(origin+pd.DateOffset(months=24)).date()}。
- C8：{c8summary['parsed_models']} 个模型目录解析成功，{c8summary['failed_models']} 个失败；逐任务指标与六维均值 Pearson {corr:.3f}。
- C6 桥接：加权单调 logistic，家族分组 CV RMSE {cv_rmse:.2f} 分；high-only 样本仅7行。
- 严格开放前沿末季仅 {int(ability_q.iloc[-1].n)} 个模型；n<5 不解释季度涨跌，窗口与宽口径敏感性见 `frontier_definition_sensitivity.csv`。
- 持平前沿已加入朴素短期回测对照；3月目标 {len(direct3)} 个，持平 MAE {persist3.abs_error.mean():.2f} 分，但目标季 n≥5 仅 {len(qualified_persist)} 个，不支持长期模型升级。
- 规模/技术：模型隐含 {r.scale_points:.2f}/{r.technology_points:.2f} 分，观测前沿变化 {r.observed_frontier_change:.2f} 分，方向不一致，份额不可作为主结论。
- C4 算力：单斜率 {overall.annual_multiplier:.2f} 倍/年，近两年 {recent.annual_multiplier:.2f} 倍/年；未有稳健放缓证据。
- 12/24月中度放缓+技术延续：{mechanism.iloc[0].median_ability:.2f}/{mechanism.iloc[1].median_ability:.2f}；机制 E1：{mechanism.iloc[0].benchmark:.2f}/{mechanism.iloc[1].benchmark:.2f}；两者差异大。
- 最大限制：直接榜单仅九个月，无12/24个月回测；长期预测与区间均为探索性情景。

完整方法、图表、限制见 [完整报告](problem4_report.md)。
"""
    (HERE/"problem4_summary.md").write_text(summary,encoding="utf-8")
    detailed="# 问题四完整详解\n\n## 从题目到数据、模型和验证\n\n## 1 问题分析"+report.split("## 1 问题分析",1)[1]
    detailed += r"""

## 数学模型与计算链的进一步说明

### A. 逐任务能力

同一叶任务只使用预先固定的评价指标：BBH、GPQA、MUSR 为 `acc_norm,none`，MATH 为 `exact_match,none`，IFEval 为 `inst_level_strict_acc,none`，MMLU-Pro 为 `acc,none`。先在相同 task/metric 内计算百分位 r_ij，再令 `DetailedAbility_i=mean_j r_ij`、`WeakAbility_i=Q25_j r_ij`。父任务与子任务不重复计数；至少50%任务覆盖与三个任务族。百分位是样本相对指标，不能与榜单0–100原始分直接减法比较。用同模型的相关性和名次差检查它是否支持六维主指标。

### B. 桥接模型

模型族分组折中，同一系列绝不同时出现在训练与测试。线性桥接为 `S=a+bL, b<=0`；有界桥接为 `S=100 sigmoid(a+bL), b<0`；isotonic 为单调非增的经验阶梯函数，在训练Loss范围外为常数。主报告选择有界桥接以保证预测量纲与0–100边界，并以分组CV报告其泛化误差。High-only样本集中于少数系列，不能把其拟合优度看成跨模型族性能。C5只作对照，不增补C6主表。

### C. 规模与剩余时间趋势

主回归在验证过的独立发布上为 `Y_i=beta0+betaC(log10(C_i)-23)+betaT*t_i+error_i`，`Y_i=logit(S_i/100)`，t 以2024-01-01起算年数。训练算力来自C4 reported FLOP；C4的N、D用于参数冲突检查、proxy和 `Y~log10(N)+log10(D)+t` 替代模型。独立发布去重避免一次模型多次提交扩大样本量。用组织分组CV检查拟合。由于时间与算力同时变化，betaT还包含未观测数据质量、后训练、评测适配与样本选择；它不是特定算法的因果效应。

### D. 对称反事实分解

以历史首季度观测p95能力为 `S00`，把算力从首季p90切到末季p90、把时间从首季切到末季，构造 `S10`、`S01`、`S11`。算力贡献为 `[(S10-S00)+(S11-S01)]/2`；剩余时间贡献为 `[(S01-S00)+(S11-S10)]/2`。两者精确加到 `S11-S00`。组织聚类bootstrap重拟合回归，再逐次分解。模型隐含变化与实际 p95 变化方向相反的诊断被保留；这比只报告漂亮的百分比更重要。等效算力倍数为 `10^(betaT*Δt/betaC)`，仅在 betaC>0 时解释。

### E. 未来情景

C4开放语言模型季度p90算力序列作为外生资源路径；排除不完整季度。其2023–24对数年斜率记为g。未来第h年资源为 `C_h=C_0*exp(g*h*scale_factor)`，scale_factor 为1、0.5、0.25。技术残差 `betaT*h*tech_factor`，tech_factor 为1、0.5、0。能力预测在最后观测p95的 logit 上累加两项再反变换。组织bootstrap的系数分布加历史残差形成条件区间；不同情景形成包络，与条件区间概念不同。因为历史开源能力前沿只有四季，12/24个月滚动原点没有可用折，表中的长期数值不可视作已校准预测。

### F. 问题一至三的机制链

读取问题二 V2 的 Model C 参数与问题三 V2 的求解函数，在给定C下得到E0、E1、E2的N、D、Q和 Loss。问题一的P1观测支持域配比通过问题三V2已审计的 `delta_p` 传入；主对照采用P0参考配比，P1仅作lambda=0.25敏感性。再将Loss输入C6桥接。E1的3倍N/D上限在所有未来算力情景中都被触及，所以E1出现平坦的机理预测；这揭示的是机制约束与外推范围，而非前沿必定停滞。C6分组CV误差带和强外推标记与机制结果一同输出。

## 复现命令与文件

在项目根目录运行 `python problem4/run_problem4.py`。全部中间表在 `problem4/outputs/tables/`，图表在 `problem4/outputs/figures/`。`data_pipeline.py` 负责审计、实体匹配、C8解析；`bridge_model.py` 负责桥接；`history_forecast.py` 负责反事实分解、算力和预测；`mechanism.py` 只读问题一至三已有成果；`verify_problem4.py` 输出机器检查，数量见 `verification_results.json`。未通过的三项属于数据证据不足与模型诊断，不应通过放宽验证标准消除。
"""
    (HERE/"问题四完整详解.md").write_text(detailed,encoding="utf-8")
    final={"problem1":{"quality_proxy":"four-group index; no independent human gold standard","mixture":"P1 observed support","p1_delta_loss":-0.502243797638},
           "problem2":{"model":"Model C: classic + quality_model","interface":"problem2_v2/outputs/tables/problem2_to_problem3_v2.json"},
           "problem3":{"version":"V2","regime":"E1_moderate_3x","source":"problem3_v2/methods.py","extrapolation":"strong above 1e24 FLOP"},
           "problem4":{"forecast_origin_date":str(origin.date()),"open_models":len(strict),"strict_match_rate":float(a.accepted.mean()),
                       "historical_model_scale_points":float(r.scale_points),"historical_model_technology_points":float(r.technology_points),
                       "observed_frontier_change":float(r.observed_frontier_change),"c2_inherited_sensitivity_frontier_change":float(broad_q.iloc[-1].p95_ability-broad_q.iloc[0].p95_ability),"bridge_cv_rmse":cv_rmse,
                       "c8_pearson":float(corr),"compute_recent_log_growth":float(recent.slope_log_per_year),
                       "forecast":fmain.to_dict("records"),"evidence_level":"exploratory: no 12/24 month rolling backtest"}}
    (OUT/"final_project_summary.json").write_text(json.dumps(final,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    return final
