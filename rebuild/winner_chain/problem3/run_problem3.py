"""Reproduce all problem-3 tables, figures, reports, and verification.

Run from repository root: python problem3/run_problem3.py
"""
from __future__ import annotations
import json,sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from core import COST_MODELS,Model,g,dg,total_cost,cost_gradient,solve
from load_inputs import ROOT,HERE,TABLES,FIGURES,load_all,write_interface,context_summary

BUDGETS=[1e19,1e22,1e24]

def save(df,name):
    df=pd.DataFrame(df)
    df.fillna('not_applicable').to_csv(TABLES/name,index=False,float_format='%.12g')
    return df

def tag(v,lo,hi):
    if lo<=v<=hi:return 'in_domain'
    return 'moderate_extrapolation' if lo/3<=v<=hi*3 else 'strong_extrapolation'

def mixture_analysis(artifact,p,pcols):
    sys.path.insert(0,str(ROOT/'problem1'))
    from problem1_v2 import transform
    mix=artifact['mixture_model'];ref=artifact['reference_p']
    def pred(a):
        a=np.asarray(a,float).reshape(-1,len(ref))
        return mix['model'].predict(transform(a,mix['family'],mix['eps'],mix['reference_index'])).mean(axis=1)
    ref_loss=float(pred(ref)[0]);scores=pred(p);idx=int(np.argmin(scores));support_p=p[idx]
    # Loose simplex optimization is diagnostic only. Softmax guarantees simplex.
    def soft(z):
        z=np.r_[z,0.];z=z-np.max(z);e=np.exp(z);return e/e.sum()
    starts=[np.log(np.maximum(ref[:-1],1e-5)/ref[-1]),np.log((support_p[:-1]+.001)/(support_p[-1]+.001))]
    rng=np.random.default_rng(2026);starts.extend([rng.normal(0,1,16) for _ in range(2)])
    loose=[]
    for z in starts:
        r=minimize(lambda a:float(pred(soft(a))[0]),z,method='L-BFGS-B',bounds=[(-12,12)]*16,options={'maxiter':250})
        loose.append((float(r.fun),soft(r.x)))
    loose_loss,loose_p=min(loose,key=lambda x:x[0])
    eps=mix['eps'];X=transform(p,mix['family'],eps,mix['reference_index']);mean=X.mean(axis=0);sd=X.std(axis=0);sd[sd==0]=1
    XT=(X-mean)/sd;tree=cKDTree(XT);threshold=float(np.quantile(tree.query(XT,k=2)[0][:,1],.95))
    def dist(a):return float(tree.query((transform(np.asarray(a).reshape(1,-1),mix['family'],eps,mix['reference_index'])-mean)/sd,k=1)[0][0])
    rows=[]
    for name,v,loss in [('P0_reference',ref,ref_loss),('P1_observed_support',support_p,float(scores[idx])),('P1_loose_simplex',loose_p,loose_loss)]:
        rows.append(dict(strategy=name,mixture_loss=loss,delta_p=loss-ref_loss,distance_to_train=dist(v),training_local_support_threshold=threshold,
                         outside_support=dist(v)>threshold,max_share=float(max(v)),training_row=idx if name=='P1_observed_support' else -1,
                         **{str(k):float(v[j]) for j,k in enumerate(pcols)}))
    df=save(rows,'mixture_optimization.csv')
    return df,ref_loss,support_p,loose_p

def plots(base,grid,bycost,context,mix,unc):
    x=np.log10(grid.budget.to_numpy());styles={'N_B':('optimal_N_vs_budget.png','N (billion)'), 'D_B':('optimal_D_vs_budget.png','D (billion)'),
        'Q_B':('optimal_Q_vs_budget.png','Q_B'), 'loss':('optimal_loss_vs_budget.png','Loss'),'D_over_N':('optimal_nd_ratio.png','D/N')}
    for col,(fn,ylabel) in styles.items():
        fig,ax=plt.subplots(figsize=(7,4));ax.plot(x,grid[col],lw=2);ax.set(xlabel='log10 budget (FLOPs)',ylabel=ylabel);ax.grid(alpha=.25)
        if col in ('N_B','D_B','D_over_N'):ax.set_yscale('log')
        fig.tight_layout();fig.savefig(FIGURES/fn,dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));ax.stackplot(x,grid.s_train,grid.s_Q,grid.s_attn,labels=['train','quality','attention']);ax.legend();ax.set(xlabel='log10 budget',ylabel='FLOP share');fig.tight_layout();fig.savefig(FIGURES/'budget_share_vs_budget.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4));axs[0].plot(x,grid.Q_B);axs[0].axhline(.6,color='gray',ls=':');axs[0].set(xlabel='log10 budget',ylabel='Q_B');axs[1].stackplot(x,grid.s_train,grid.s_Q,grid.s_attn,labels=['train','quality','attention']);axs[1].legend();axs[1].set(xlabel='log10 budget',ylabel='FLOP share');fig.tight_layout();fig.savefig(FIGURES/'structural_shift_phase_diagram.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(12,3.8));
    for kind,z in bycost.groupby('cost_model'):
        z=z.sort_values('budget');u=np.log10(z.budget);[axs[j].plot(u,z[c],marker='o',label=kind) for j,c in enumerate(['N_B','D_B','Q_B'])]
    for j,c in enumerate(['N_B','D_B','Q_B']):axs[j].set(xlabel='log10 budget',ylabel=c);axs[j].legend(fontsize=7)
    fig.tight_layout();fig.savefig(FIGURES/'quality_cost_comparison.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(9,3.8));
    for b,z in context.groupby('budget'):
        z=z.sort_values('L_ctx');axs[0].plot(z.L_ctx,z.loss,marker='o',label=f'{b:.0e}');axs[1].plot(z.L_ctx,z.N_B,marker='o',label=f'{b:.0e}')
    for a,y in zip(axs,['Loss','N_B']):a.set(xlabel='context tokens',ylabel=y);a.set_xscale('log');a.legend(fontsize=7)
    fig.tight_layout();fig.savefig(FIGURES/'context_length_sensitivity.png',dpi=180);plt.close(fig)
    z=base.sort_values('budget');fig,axs=plt.subplots(1,2,figsize=(10,4));xx=np.arange(len(z));axs[0].bar(xx,z.s_train,label='train');axs[0].bar(xx,z.s_Q,bottom=z.s_train,label='quality');axs[0].bar(xx,z.s_attn,bottom=z.s_train+z.s_Q,label='attention');axs[0].set_xticks(xx,[f'{v:.0e}' for v in z.budget]);axs[0].legend();axs[0].set(ylabel='FLOP share');axs[1].plot(xx,z.N_B,marker='o',label='N_B');axs[1].plot(xx,z.D_B,marker='o',label='D_B');axs[1].set_yscale('log');axs[1].set_xticks(xx,[f'{v:.0e}' for v in z.budget]);axs[1].legend();fig.tight_layout();fig.savefig(FIGURES/'resource_allocation_low_medium_high.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4));ax.bar(mix.strategy,mix.delta_p);ax.tick_params(axis='x',rotation=15);ax.set(ylabel='Loss shift from p reference');fig.tight_layout();fig.savefig(FIGURES/'mixture_optimization.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));
    for b,z in unc.groupby('budget'):
        ax.errorbar(np.log10(b),z.Q_B.median(),yerr=[[z.Q_B.median()-z.Q_B.quantile(.05)],[z.Q_B.quantile(.95)-z.Q_B.median()]],fmt='o')
    ax.set(xlabel='log10 budget',ylabel='Q_B, 5–95% scenario range');fig.tight_layout();fig.savefig(FIGURES/'uncertainty_intervals.png',dpi=180);plt.close(fig)
    q=np.linspace(.1,1,100);fig,ax=plt.subplots(figsize=(7,4));
    for kind in COST_MODELS:ax.plot(q,[g(v,kind)/1e9 for v in q],label=kind)
    ax.set(xlabel='Q_B',ylabel='g(Q), 10^9 FLOPs/token');ax.legend();fig.tight_layout();fig.savefig(FIGURES/'quality_cost_functions.png',dpi=180);plt.close(fig)
    ctx=np.array([2048,4096,8192,30000,32768,131072]);fig,ax=plt.subplots(figsize=(7,4));ax.plot(ctx,2e-4*ctx/6,marker='o');ax.axhline(1,color='gray',ls=':');ax.set(xlabel='context tokens',ylabel='C_attention / C_train');fig.tight_layout();fig.savefig(FIGURES/'attention_train_ratio.png',dpi=180);plt.close(fig)

def main():
    artifact,model,b1,b6,c7,p,pcols=load_all();interface=write_interface(artifact,model,b1,b6);contexts,csummary=context_summary(c7)
    q0s={'low':.2,'mid':.6,'high':.8};q0=q0s['mid'];ctx=contexts['medium'];kind='exponential'
    audit=[dict(item='model',value='problem2_v2 Model C',source=interface['model_file']),dict(item='B1 rows',value=len(b1),source='B1'),dict(item='B6 Q grid',value=','.join(map(str,sorted(b6.Q_score.unique()))),source='B6'),dict(item='C7 rows',value=len(c7),source='C7'),dict(item='p train rows',value=len(p),source='A4/A5')]
    save(audit,'input_interface_audit.csv')
    save([dict(cost_model=k,Q_B=float(q),g_flop_per_token=g(q,k),dg_dQ=dg(q,k)) for k in COST_MODELS for q in [.2,.6,.8,1.]],'quality_cost_functions.csv')
    # Main results and cost-function comparison. Theoretical bounds permit extrapolation.
    bycost=[]
    for k in COST_MODELS:
        for b in BUDGETS:
            bycost.append(dict(cost_model=k,Q0=q0,L_ctx=ctx,**solve(model,b,q0,ctx,k,global_search=True)))
    bycost=save(bycost,'optimal_allocations_by_cost.csv');save(bycost,'quality_cost_comparison.csv')
    base=bycost[bycost.cost_model==kind].copy();save(base,'optimal_allocations.csv')
    # Dense budget path, with boundary crossings and elasticity diagnostics.
    budgets=np.unique(np.r_[np.logspace(18,25,71),BUDGETS]);grid=[]
    for b in budgets:
        grid.append(dict(cost_model=kind,Q0=q0,L_ctx=ctx,**solve(model,float(b),q0,ctx,kind)))
    grid=save(grid,'budget_grid_results.csv');grid['D_over_N']=grid.D_B/grid.N_B
    grid['e_N']=np.gradient(np.log(grid.N_B),np.log(grid.budget));grid['e_D']=np.gradient(np.log(grid.D_B),np.log(grid.budget));grid['e_Q']=np.gradient(grid.Q_B,np.log(grid.budget))
    save(grid[['budget','N_B','D_B','D_over_N','e_N','e_D','e_Q']],'optimal_nd_ratio.csv')
    dshare=np.r_[np.nan,np.abs(np.diff(grid[['s_train','s_Q','s_attn']],axis=0)).sum(axis=1)]
    transitions=[]
    for label,mask in [('Q_start',~grid.Q_at_Q0),('Q_saturation',grid.Q_at_1),('share_jump_tau_0.05',dshare>.05),('share_jump_tau_0.10',dshare>.10)]:
        ids=np.flatnonzero(mask);i=int(ids[0]) if len(ids) else None
        transitions.append(dict(criterion=label,threshold=.05 if label.endswith('0.05') else .10 if label.endswith('0.10') else np.nan,
                                C_crit=float(grid.budget.iloc[i]) if i is not None else np.nan,previous_C=float(grid.budget.iloc[i-1]) if i and i>0 else np.nan,
                                observed=bool(i is not None),note='grid crossing; not a fitted physical phase transition'))
    shift=save(transitions,'structural_shift_points.csv')
    # Context, Q0, and cost model sensitivity.
    cs=[]
    for label,c in contexts.items():
        for b in BUDGETS:cs.append(dict(context_scenario=label,L_ctx=c,Q0=q0,cost_model=kind,**solve(model,b,q0,c,kind)))
    cs=save(cs,'context_sensitivity.csv')
    qs=[]
    for label,qbase in q0s.items():
        for b in BUDGETS:qs.append(dict(Q0_scenario=label,Q0=qbase,L_ctx=ctx,cost_model=kind,**solve(model,b,qbase,ctx,kind)))
    qs=save(qs,'q0_sensitivity.csv')
    # p is separable in Model C: optimizing it leaves N,D,Q unchanged at fixed budget.
    mix,ref_loss,support_p,loose_p=mixture_analysis(artifact,p,pcols)
    lrows=[]
    for lam in [.5,.75,1,1.25,1.5]:
        for b in BUDGETS:
            r=base[base.budget==b].iloc[0]
            for _,m in mix.iterrows():lrows.append(dict(lambda_p=lam,budget=b,strategy=m.strategy,N_B=r.N_B,D_B=r.D_B,Q_B=r.Q_B,loss=r.loss+lam*m.delta_p,delta_p=lam*m.delta_p,outside_support=m.outside_support))
    save(lrows,'lambda_p_sensitivity.csv')
    # Marginal benefits per FLOP. Boundary Q values have one-sided KKT inequalities.
    margins=[]
    for _,r in base.iterrows():
        n,d,q=r.N_B,r.D_B,r.Q_B;gl=model.gradient(n,d,q);gc=cost_gradient(n,d,q,q0,ctx,kind);m=-gl/gc
        h=1e-6
        if r.Q_at_1:
            dldq=(model.loss(n,d,1)-model.loss(n,d,1-h))/h
            muq=-dldq/gc[2]
        elif r.Q_at_Q0:
            dldq=(model.loss(n,d,q+h)-model.loss(n,d,q))/h
            muq=-dldq/gc[2]
        else:muq=m[2]
        boundary='upper' if r.Q_at_1 else 'lower' if r.Q_at_Q0 else 'interior'
        margins.append(dict(budget=r.budget,N_B=n,D_B=d,Q_B=q,mu_N=m[0],mu_D=m[1],mu_Q=m[2] if np.isfinite(m[2]) else np.nan,
                            mu_Q_one_sided=muq,relative_N_D_gap=abs(m[0]-m[1])/max(m[0],m[1]),Q_boundary=boundary,
                            KKT_Q_satisfied=(muq>=m[0]*(1-1e-3) if boundary=='upper' else muq<=m[0]*(1+1e-3) if boundary=='lower' else abs(muq-m[0])/m[0]<1e-3)))
    margins=save(margins,'marginal_loss_reduction_per_flop.csv')
    # Empirical-range solution: capped N,D and B6 Q, with remaining budget potentially unspent.
    empirical=[]
    nb=(float(b1.N_params_B.min()),float(b1.N_params_B.max()));db=(float(b1.D_tokens_B.min()),float(b1.D_tokens_B.max()))
    for b in BUDGETS:
        r=solve(model,b,q0,ctx,kind,n_bounds=nb,d_bounds=db,global_search=True)
        if r['D_B']>db[1] or r['D_B']<db[0] or r['C_total']>b*(1+1e-7):
            # Direct constrained fallback in capped N,D,Q space.
            def f(x):return model.loss(np.exp(x[0]),np.exp(x[1]),x[2])
            def con(x):return 1-total_cost(np.exp(x[0]),np.exp(x[1]),x[2],q0,ctx,kind)[0]/b
            best=None
            for qstart in [q0,(q0+1)/2,1.]:
                rr=minimize(f,[np.log(np.sqrt(nb[0]*nb[1])),np.log(np.sqrt(db[0]*db[1])),qstart],method='SLSQP',
                  bounds=[tuple(np.log(nb)),tuple(np.log(db)),(q0,1)],constraints=[{'type':'ineq','fun':con}],options={'ftol':1e-12,'maxiter':1000})
                if con(rr.x)>=-1e-7 and (best is None or rr.fun<best.fun):best=rr
            if best is None:raise RuntimeError('No feasible empirical-range optimum')
            n,d,q=np.exp(best.x[0]),np.exp(best.x[1]),best.x[2];tot,tr,cq,att=total_cost(n,d,q,q0,ctx,kind)
            r=dict(budget=b,N_B=n,D_B=d,Q_B=q,loss=model.loss(n,d,q),C_total=tot,C_train=tr,C_Q=cq,C_attn=att,budget_ratio=tot/b)
        empirical.append(dict(solution_type='empirical_range',**r))
    empirical=save(empirical,'empirical_range_allocations.csv')
    flags=[]
    for _,r in pd.concat([base.assign(solution_type='unconstrained_scaling'),empirical],ignore_index=True).iterrows():
        flags.append(dict(solution_type=r.solution_type,budget=r.budget,N_extrapolation=tag(r.N_B,*nb),D_extrapolation=tag(r.D_B,*db),
                          Q_extrapolation=tag(r.Q_B,float(b6.Q_score.min()),float(b6.Q_score.max())),
                          strong_extrapolation=tag(r.N_B,*nb)=='strong_extrapolation' or tag(r.D_B,*db)=='strong_extrapolation'))
    flags=save(flags,'extrapolation_flags.csv')
    # Scenario ensemble: intervals for G,kappa,eta_N are bootstrap marginal bounds;
    # independent sampling is an explicit approximation, not a joint bootstrap.
    intervals=pd.read_csv(ROOT/'problem2_v2/outputs/tables/quality_model_intervals.csv').set_index('parameter')
    rng=np.random.default_rng(20260924);unc=[]
    for i in range(90):
        par={name:float(np.clip(rng.normal(intervals.loc[name,'median'],(intervals.loc[name,'ci_high']-intervals.loc[name,'ci_low'])/3.92),intervals.loc[name,'ci_low'],intervals.loc[name,'ci_high'])) for name in ['G','kappa','eta_N']}
        mm=Model(model.E,model.A,model.alpha,model.B,model.beta,par['G'],par['kappa'],par['eta_N'])
        qq=float(rng.choice([.2,.6,.8]));cc=int(rng.choice(list(contexts.values())));kk=str(rng.choice(COST_MODELS))
        for b in BUDGETS:unc.append(dict(draw=i,Q0=qq,L_ctx=cc,cost_model=kk,G=par['G'],kappa=par['kappa'],eta_N=par['eta_N'],**solve(mm,b,qq,cc,kk)))
    unc=save(unc,'optimization_uncertainty.csv')
    plots(base,grid,bycost,cs,mix,unc)
    # Machine-readable next-stage interface, including all assumption labels.
    p4={'model':'problem2_v2 Model C','cost_model':kind,'Q0':q0,'context_scenario':{'label':'medium','L_ctx':ctx},
        'budgets':BUDGETS,'recommended_robust_optimum':empirical.to_dict(orient='records'),
        'mathematical_optimum':base.to_dict(orient='records'),'structural_shift_points':shift.to_dict(orient='records'),
        'optimal_trajectory_file':'problem3/outputs/tables/budget_grid_results.csv',
        'uncertainty_file':'problem3/outputs/tables/optimization_uncertainty.csv','extrapolation_flags':flags.to_dict(orient='records'),
        'warnings':['Q_A to Q_B not identified','lambda_p not identified','high budgets extrapolate B1 N,D','quality cost functions are stipulated scenarios']}
    def json_clean(v):
        if isinstance(v,dict):return {k:json_clean(x) for k,x in v.items()}
        if isinstance(v,list):return [json_clean(x) for x in v]
        if isinstance(v,np.generic):return json_clean(v.item())
        if isinstance(v,float) and not np.isfinite(v):return None
        return v
    (TABLES/'problem3_to_problem4.json').write_text(json.dumps(json_clean(p4),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    build_reports(model,base,bycost,cs,mix,shift,flags,empirical,csummary,contexts,unc,grid,q0,ctx)
    from verify_problem3 import verify
    verify()
    print(base[['budget','N_B','D_B','Q_B','loss','s_train','s_Q','s_attn']].to_string(index=False))

def build_reports(model,base,bycost,cs,mix,shift,flags,empirical,csummary,contexts,unc,grid,q0,ctx):
    fmt=lambda x:f'{x:.4g}'
    baselines='\n'.join(f"| {r.budget:.0e} | {fmt(r.N_B)} | {fmt(r.D_B)} | {fmt(r.Q_B)} | {fmt(r.loss)} | {r.s_train:.1%} | {r.s_Q:.1%} | {r.s_attn:.1%} |" for _,r in base.iterrows())
    costlines='\n'.join(f"| {r.cost_model} | {r.budget:.0e} | {fmt(r.N_B)} | {fmt(r.D_B)} | {fmt(r.Q_B)} | {fmt(r.loss)} |" for _,r in bycost.iterrows())
    trans='\n'.join(f"| {r.criterion} | {fmt(r.C_crit) if r.observed else 'not observed'} |" for _,r in shift.iterrows())
    mixlines='\n'.join(f"| {r.strategy} | {fmt(r.delta_p)} | {r.outside_support} | {fmt(r.max_share)} |" for _,r in mix.iterrows())
    strong=', '.join(f"{r.budget:.0e}" for _,r in flags[(flags.solution_type=='unconstrained_scaling') & flags.strong_extrapolation].iterrows()) or 'none'
    report=f'''# 问题三：算力约束下的联合资源优化

## 1 问题与输入

第一问提供 17 领域配比效应；第二问 V2 的 Model C 提供 N、D、Q_B、p 到 Loss 的预测。读取 `recommended_model.joblib` 与 `selection.json`，没有调用 V1 的 JSON。B1 支持 N_B ∈ [0.070542,11.965825]、D_B ∈ [0.134,299.893]；B6 的 Q_B 网格为 0.1、0.2、0.3、0.4、0.6、0.8、0.9、1.0。

## 2 预测模型

$$L={model.E:.9f}+{model.A:.9f}N_B^{{-{model.alpha:.9f}}}+{model.B:.9f}D_B^{{-{model.beta:.9f}}}+{model.G:.9f}(1-Q_B)^{{{model.kappa:.9f}}}N_B^{{-{model.eta_N:.9f}}}+\\lambda_p\\Delta_p.$$

其中 p 修正是问题一模型预测的领域平均 Loss 相对均匀参考配比的差。默认 λ_p=1 是情景假设；Q_A 到 Q_B 未识别。

## 3 算力成本与单位

N_raw=10^9 N_B，D_raw=10^9 D_B。$$C_{{train}}=6\\times10^{{18}}N_BD_B,\\quad C_Q=10^9D_B[g(Q_B)-g(Q_0)]_+,\\quad C_{{attn}}=2\\times10^{{14}}N_BD_BL_{{ctx}}.$$

指数、幂、对数三种 g 分别为 $10^7e^{{6Q}}$、$5\\times10^9Q^4$、$2\\times10^9\\ln(1+10Q)$。三者均完整求解；指数函数仅作为叙述主情景，因为其递增边际成本给出内点 Q，便于展示起投和饱和两类机制，并无实证证据证明它比另外两者更真实。主情景 Q0=0.6、C7 中位上下文 {ctx}；Q0=0.2/0.8 及另两成本为敏感性情景。B6 没有唯一可识别的 Q0。

$$C_{{attn}}/C_{{train}}=\\eta L_{{ctx}}/6,\\quad L_{{ctx}}^{{crit}}=6/(2\\times10^{{-4}})=30000.$$

C7 共 {csummary['rows']} 行，缺失 {csummary['missing']}；中位 {csummary['median']:.0f}，最小 {csummary['minimum']}、最大 {csummary['maximum']}。实际值 {csummary['unique_values']}；低于 30000 的 {csummary['below_critical']} 行，高于的 {csummary['above_critical']} 行；临界点两侧最近值 {csummary['closest_below']} 与 {csummary['closest_above']}。131072 是单个长尾值，不作为主情景。短/中/长选 {contexts['short']}/{contexts['medium']}/{contexts['long']}。

## 4 优化与数值方法

在 $C_{{total}}\\le C$、$Q_0\\le Q_B\\le1$、p 为 simplex 下最小化 Model C。第一阶段固定 p=p_ref；将 D 用绑定预算解析消元，优化 log N_B 与有界 Q_B。三档主预算均运行 differential evolution、SLSQP 多初值及 Q 边界一维搜索。密集轨迹为 10^18 至 10^25 的 71 个对数点。p 单独求解，因为成本与 N-D-Q 主损失对 p 可分。

## 5 三档预算主结果

| FLOPs | N_B | D_B | Q_B | Loss | 训练 | 质量 | attention |
|---:|---:|---:|---:|---:|---:|---:|---:|
{baselines}

图见 `outputs/figures/resource_allocation_low_medium_high.png`。占比是成本除预算，不是 token 数量占比。

## 6 三种质量成本

| g | FLOPs | N_B | D_B | Q_B | Loss |
|---|---:|---:|---:|---:|---:|
{costlines}

三函数是题定的成本情景，不是数据拟合。其差异只能解释为成本假设敏感性。

## 7 上下文敏感性与结构性转移

短、中、长上下文均逐预算重新优化，完整结果见 `context_sensitivity.csv`。上下文超 30000 后，attention 成本超过基础训练成本。结构性转移定义为 Q 下界退出、Q 上界到达，或相邻 log 预算步的份额 L1 变化超过 0.05/0.10。它们是可复现的离散网格判据，不是物理相变。

| 判据 | 首次触发 FLOPs |
|---|---:|
{trans}

N、D 的预算弹性与 D/N 见 `optimal_nd_ratio.csv`。若主情景没有 Q 起投点或饱和点，按未观察到报告，不能人为指定。Model C 中 Q 罚项随 N 增加而减弱，因此 Q 不必单调随预算上升。

## 8 配比优化与 KKT

P0 使用均匀参考配比；P1 在 512 条问题一实际训练配比中选择预测最优者，这是严格训练支持域方案；Loose 用 softmax 连续优化，仅作数学外推诊断。

| 策略 | 相对 Loss 改变 | 超出局部支持 | 最大单域占比 |
|---|---:|---|---:|
{mixlines}

p 不消耗额外 FLOPs，因而同一 λ_p>0 下，N-D-Q 最优值不随 λ_p 改变；绝对 Loss 改变量与 λ_p 成比例。`lambda_p_sensitivity.csv` 枚举 0.5、0.75、1、1.25、1.5。内点 KKT 要求三项 $-\\partial L/\\partial x \,/\, \\partial C/\\partial x$ 相等；Q 在上下界时改用不等式。逐预算数值见 `marginal_loss_reduction_per_flop.csv`。

## 9 不确定性、外推与推荐

90 个情景抽取 Q0、C7 上下文、成本函数，并按问题二边际 bootstrap 区间近似抽取 G、κ、η_N。边际独立抽样不保留参数相关性，不是严格联合置信区间。结果见 `optimization_uncertainty.csv`。外推档位按 B1 区间的三倍外侧标为 strong；主理论方案强外推预算：{strong}。`empirical_range_allocations.csv` 提供 N、D 限在 B1 内的稳健对照；预算可能不绑定，表示数据支持范围内没有理由强行用完大预算。数学最优是无限制 Scaling Law 解；稳健推荐应优先引用实证范围方案，并明确其模型配比与 Q0 仍有假设。

## 10 模型评价与问题四接口

优势：统一单位、可复算、成本可分解、允许端点、显式支持域。限制：质量成本及 Q0 是情景假设；QA→QB 与 λ_p 未联合识别；B1 外推及问题一 p 模型支持域有限；C7 架构长度是可用情景，不等于训练时实际使用长度。问题四读取 `outputs/tables/problem3_to_problem4.json`，同时读取轨迹、情景集合和外推标签，不应只提取单一理论最优点。
'''
    contextlines='\n'.join(f"| {r.budget:.0e} | {r.context_scenario} ({int(r.L_ctx)}) | {fmt(r.N_B)} | {fmt(r.D_B)} | {fmt(r.Q_B)} | {fmt(r.loss)} | {r.s_attn:.1%} |" for _,r in cs.iterrows())
    empiricallines='\n'.join(f"| {r.budget:.0e} | {fmt(r.N_B)} | {fmt(r.D_B)} | {fmt(r.Q_B)} | {fmt(r.loss)} | {r.budget_ratio:.1%} |" for _,r in empirical.iterrows())
    uncertaintylines='\n'.join(f"| {b:.0e} | {fmt(z.N_B.quantile(.05))}–{fmt(z.N_B.quantile(.95))} | {fmt(z.D_B.quantile(.05))}–{fmt(z.D_B.quantile(.95))} | {fmt(z.Q_B.quantile(.05))}–{fmt(z.Q_B.quantile(.95))} |" for b,z in unc.groupby('budget'))
    report+=f'''\n## 11 实际决策解释与对照表\n\n低预算 10^19 FLOPs 时，Q_B={fmt(base.iloc[0].Q_B)}，质量支出 {base.iloc[0].s_Q:.1%}；在 10^20 附近，质量支出占比达约 {grid.iloc[(grid.budget-1e20).abs().argmin()].s_Q:.1%}。中预算 Q 已达上界，新增预算转向 N 与 D。高预算下 D/N={base.iloc[2].D_B/base.iloc[2].N_B:.1f}，明显高于低预算的 {base.iloc[0].D_B/base.iloc[0].N_B:.1f}；这由拟合指数及质量饱和后的边际平衡决定，不应机械套用固定 Chinchilla 比率。\n\n| FLOPs | C7 情景 | N_B | D_B | Q_B | Loss | attention 占比 |\n|---:|---|---:|---:|---:|---:|---:|\n{contextlines}\n\n在 10^19 下，短到长上下文令 N 从 {fmt(cs[(cs.budget==1e19)&(cs.context_scenario=='short')].iloc[0].N_B)} 降至 {fmt(cs[(cs.budget==1e19)&(cs.context_scenario=='long')].iloc[0].N_B)}，D 从 {fmt(cs[(cs.budget==1e19)&(cs.context_scenario=='short')].iloc[0].D_B)} 降至 {fmt(cs[(cs.budget==1e19)&(cs.context_scenario=='long')].iloc[0].D_B)}；Q 则从 {fmt(cs[(cs.budget==1e19)&(cs.context_scenario=='short')].iloc[0].Q_B)} 升至 {fmt(cs[(cs.budget==1e19)&(cs.context_scenario=='long')].iloc[0].Q_B)}。质量提升在此情景下部分替代 N-D 投入，Loss 仍恶化。\n\n| FLOPs | 实证范围 N_B | D_B | Q_B | Loss | 已用预算 |\n|---:|---:|---:|---:|---:|---:|\n{empiricallines}\n\n10^24 的实证范围方案仅能用去约 {empirical.iloc[2].budget_ratio:.1%} 的预算；它是证据保守对照，并非可把剩余算力有效利用的完整工程计划。\n\n| FLOPs | N_B 情景 5–95% | D_B 情景 5–95% | Q_B 情景 5–95% |\n|---:|---:|---:|---:|\n{uncertaintylines}\n\n这些是跨 Q0、上下文、成本函数与参数的情景分布区间，不能解释为频率学置信区间。指数成本在低预算会投资部分 Q；幂成本低预算停在 Q0；对数成本低预算直接到 Q=1。中高预算三者都到 Q=1，差别主要体现在质量成本份额和 N-D 细调。\n\nP1 训练样本支持方案使模型预测 Loss 相对 p_ref 下降 {abs(mix.iloc[1].delta_p):.3f}，但这一数量级受未识别 λ_p 及跨模型尺度转移支配，论文建议把 p_ref 作为主预测基线，将 P1 作为同算力配比重分配候选。Loose 的 {abs(mix.iloc[2].delta_p):.3f} 改善来自连续 simplex 数学优化；即便最近邻指标未越过 95% 局部阈值，该点并非观测训练配比，仍需新的联合实验验证。\n'''
    (HERE/'problem3_report.md').write_text(report,encoding='utf-8')
    detail=report+'''\n## 11 文件与复现详解\n\n从仓库根目录运行 `python problem3/run_problem3.py`。`load_inputs.py` 读取 V2 artifact、B1/B6/C7/A4，并新增 V2 接口 JSON 与导数 CSV。`core.py` 实现 Model C、三种成本、解析成本梯度以及 D 消元优化。`run_problem3.py` 批量生成表、图、报告与问题四接口。`verify_problem3.py` 检查单位、导数、预算、模型预测、场景覆盖和报告一致性。所有输出位于 `problem3/outputs`；源模型文件未修改。\n\n预算表中的 N_B 与 D_B 均为十亿单位。质量成本按原始 token 数计算。上下文是外生变量，不参与优化。质量 Q_B 属于 B6 量尺，不应直接赋予问题一 Q_A 的语义。\n\n结构性转移解释请结合 `structural_shift_points.csv` 与 `budget_grid_results.csv`，并查看 `optimal_Q_vs_budget.png`。转折位置是 0.1 对数步网格精度，可在相邻区间内再次二分精化；若某判据未触发，就不能声称该转移存在。\n'''
    (HERE/'问题三完整详解.md').write_text(detail,encoding='utf-8')
    summary=f'''# 问题三摘要\n\nModel C：$L=E+AN_B^{{-α}}+BD_B^{{-β}}+G(1-Q_B)^κN_B^{{-η}}+λ_pΔ_p$。成本：$C=6\\times10^{{18}}N_BD_B+10^9D_B[g(Q)-g(Q0)]_++2\\times10^{{14}}N_BD_BL_{{ctx}}$。临界上下文为 30000 tokens；C7 中 {csummary['below_critical']}/{csummary['rows']} 在下侧，{csummary['above_critical']}/{csummary['rows']} 在上侧。主情景 Q0={q0}、Lctx={ctx}、指数成本。\n\n| FLOPs | N_B | D_B | Q_B | Loss | 训练 | 质量 | attention |\n|---:|---:|---:|---:|---:|---:|---:|---:|\n{baselines}\n\n结构点：{'; '.join(f'{r.criterion}: {fmt(r.C_crit) if r.observed else "未观察到"}' for _,r in shift.iterrows())}。上下文增长使 N-D 可用乘积成本增大；三成本函数结论见比较表。p 的支持域方案相对 Loss 改变 {fmt(mix.iloc[1].delta_p)}；Loose 改变 {fmt(mix.iloc[2].delta_p)}，超支持域={mix.iloc[2].outside_support}。最大限制是成本与 Q0 未实证识别、V2 Scaling Law 强外推。\n'''
    (HERE/'problem3_summary.md').write_text(summary,encoding='utf-8')
    (HERE/'assumptions.md').write_text('# 问题三假设\n\n- 三种 g 是题目规定的情景函数。\n- Q0=.6 是 B6 实测网格点，但不是唯一估计；.2/.8 做敏感性。\n- 主上下文取 C7 中位数 4096；长度元数据不保证训练时实用长度。\n- p 修正按 V2 artifact 的均匀参考配比与 λp=1。\n- 质量成本按 D_raw 计；p 不另计 FLOPs。\n- 理论外推结果需要与 B1 范围内方案同时阅读。\n',encoding='utf-8')
    (HERE/'README.md').write_text('# Problem 3\n\nRun `python problem3/run_problem3.py` from repository root. See `problem3_report.md` for methods and results, `问题三完整详解.md` for implementation details, and `outputs/tables/verification_results.json` for checks.\n',encoding='utf-8')

if __name__=='__main__':main()
