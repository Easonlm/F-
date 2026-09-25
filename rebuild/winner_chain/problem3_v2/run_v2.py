"""Reproduce Problem 3 V2 without modifying the V1 baseline.

Run from repository root: python problem3_v2/run_v2.py
"""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution,minimize,brentq

sys.dont_write_bytecode=True
from methods import ROOT,Model,COST_MODELS,g,total_cost,d_from_budget,solve,capped_solve,crossing,segmented_breaks,kkt_diagnostic
from mixture_evidence import analyze,lambda_curve

HERE=Path(__file__).resolve().parent
T=HERE/'outputs'/'tables';F=HERE/'outputs'/'figures';M=HERE/'outputs'/'models'
for directory in [T,F,M]:directory.mkdir(parents=True,exist_ok=True)
BUDGETS=[1e19,1e20,1e22,1e24]


def write(df,name):
    df=pd.DataFrame(df);df.replace([np.inf,-np.inf],np.nan).fillna('not_applicable').to_csv(T/name,index=False,float_format='%.12g')
    return df


def manifest():
    root=ROOT/'problem3';return {str(x.relative_to(root)).replace('\\','/'):hashlib.sha256(x.read_bytes()).hexdigest()
                              for x in root.rglob('*') if x.is_file() and '__pycache__' not in x.parts}


def load():
    artifact=joblib.load(ROOT/'problem2_v2/outputs/models/recommended_model.joblib')
    assert artifact['ND_model']=='classic' and artifact['Q_model']=='quality_model'
    a=artifact['ND_parameters'];z=artifact['Q_parameters'];model=Model(*map(float,[*a,*z]))
    b1=pd.read_csv(ROOT/'real_attachments/B_scaling_laws/pythia_training_log_existing.csv')
    b6=pd.read_csv(ROOT/'real_attachments/B_scaling_laws/supplementary_NQ_experiment.csv')
    mix=pd.read_csv(ROOT/'real_attachments/A_data_value/regmix_tables/train_mixture_1m.csv')
    p=mix[artifact['mixture_model']['pcols']].to_numpy(float);p/=p.sum(axis=1,keepdims=True)
    old_mix=pd.read_csv(ROOT/'problem3/outputs/tables/mixture_optimization.csv')
    return artifact,model,b1,b6,p,old_mix


def mixture_work(artifact,p,old_mix,v1):
    pm,support=analyze(artifact,p,old_mix);write(pm,'mixture_support_diagnostics.csv')
    ref=float(pm[pm.strategy=='P0_reference'].iloc[0].mixture_loss)
    extended=[]
    for lam in [0,.25,.5,.75,1,1.25,1.5]:
        for budget in [1e19,1e22,1e24]:
            base=v1[v1.budget==budget].iloc[0]
            for _,row in pm.iterrows():
                extended.append(dict(lambda_p=lam,budget=budget,strategy=row.strategy,support_class=row.support_class,
                                     N_B=base.N_B,D_B=base.D_B,Q_B=base.Q_B,loss=base.loss+lam*row.delta_p,
                                     improvement=-lam*row.delta_p,p_optimum_tied=(lam==0)))
    write(extended,'lambda_p_extended_sensitivity.csv')
    evidence=pd.read_csv(ROOT/'problem2_v2/outputs/tables/p_scale_transfer_from_problem1_v2.csv')
    old_evidence=pd.read_csv(ROOT/'problem2_v2/outputs/tables/p_scale_transfer_from_problem1.csv')
    prev=pd.read_csv(ROOT/'problem2_v2/outputs/tables/p_scale_scenarios.csv')
    pe=[]
    for scenario in ['S0_constant','S1_decay_0.1','S2_decay_0.2','S3_floor_0.25']:
        for n in [1,7,40,100]:
            lam=lambda_curve(n,scenario)
            for _,row in pm.iterrows():
                pe.append(dict(scenario=scenario,N_B=n,lambda_p=lam,strategy=row.strategy,
                               delta_loss=lam*row.delta_p,improvement=-lam*row.delta_p,identified=False,
                               evidence_note='Scenario normalized at 1M; exploratory V1 centered slopes are not a fitted lambda(N) law'))
    write(pe,'lambda_p_scale_scenarios.csv')
    write(evidence,'p_scale_evidence_audit.csv')
    write(old_evidence,'p_scale_v1_evidence_audit.csv')
    write(prev,'p_scale_v1_scenario_audit.csv')
    return pm,pd.DataFrame(pe)


def transition_work(model):
    rows=[]
    for kind in COST_MODELS:
        for which in ['start','saturation']:
            z=crossing(model,kind,.6,4096,which)
            rows.append(dict(cost_model=kind,Q0=.6,L_ctx=4096,transition=which,epsilon_Q=1e-4,**z))
    exact=write(rows,'quality_transition_points_exact.csv')
    contexts=[]
    for kind in COST_MODELS:
        for ctx in [2048,4096,32768]:
            for which in ['start','saturation']:
                row=exact[(exact.cost_model==kind)&(exact.L_ctx==ctx)&(exact.transition==which)]
                z=row.iloc[0][['status','C_threshold','C_lower','C_upper','log10_width']].to_dict() if len(row) else crossing(model,kind,.6,ctx,which)
                contexts.append(dict(cost_model=kind,Q0=.6,L_ctx=ctx,transition=which,epsilon_Q=1e-4,**z))
    ct=write(contexts,'transition_by_context.csv')
    grid=pd.read_csv(ROOT/'problem3/outputs/tables/budget_grid_results.csv')
    x=np.log(grid.budget.to_numpy(float));br=[]
    for variable in ['N_B','D_B','Q_B','s_Q']:
        y=np.log(grid[variable].to_numpy(float)) if variable in ['N_B','D_B'] else grid[variable].to_numpy(float)
        best,base=segmented_breaks(x,y)
        br.append(dict(variable=variable,selected_breaks=best['break_count'],break_1_C=np.exp(x[best['indices'][0]]) if len(best['indices'])>0 else np.nan,
                       break_2_C=np.exp(x[best['indices'][1]]) if len(best['indices'])>1 else np.nan,
                       BIC_0=base['bic'],BIC_selected=best['bic'],delta_BIC=base['bic']-best['bic'],AIC_selected=best['aic'],
                       selected_slopes=json.dumps([float(best['coefficients'][1]+sum(best['coefficients'][2:2+i])) for i in range(best['break_count']+1)])))
    breaks=write(br,'structural_break_analysis.csv')
    return exact,ct,breaks


def bootstrap_work(model,b1,b6):
    sys.path.insert(0,str(ROOT/'problem2_v2'))
    from models import fit_nd,fit_q
    original=pd.read_csv(ROOT/'problem2_v2/outputs/tables/quality_model_cluster_bootstrap.csv')
    assert len(original)==200 and original.replicate.is_unique
    rng_n=np.random.default_rng(20260924);rng_q=np.random.default_rng(20260923)
    ng=sorted(b1.N_params_B.unique());qg=sorted(b6.N_params_B.unique())
    groups_n={k:b1[b1.N_params_B==k] for k in ng};groups_q={k:b6[b6.N_params_B==k] for k in qg}
    pars=[];fail=[]
    for i in range(len(original)):
        sn=pd.concat([groups_n[k] for k in rng_n.choice(ng,len(ng),replace=True)],ignore_index=True)
        sq=pd.concat([groups_q[k] for k in rng_q.choice(qg,len(qg),replace=True)],ignore_index=True)
        try:
            nf=fit_nd('classic',sn,start=np.array([model.E,model.A,model.alpha,model.B,model.beta]))
            qf=fit_q('quality_model',sq,nf.x)
            vals=np.r_[nf.x,qf.x]
            if not nf.success or not qf.success or not np.all(np.isfinite(vals)) or np.any(vals[[1,2,3,4,5,6,7]]<=0):raise ValueError('invalid fit')
            pars.append(dict(replicate=i,E=vals[0],A=vals[1],alpha=vals[2],B=vals[3],beta=vals[4],G=vals[5],kappa=vals[6],eta_N=vals[7],
                             source_G=original.iloc[i].G,source_kappa=original.iloc[i].kappa,source_eta_N=original.iloc[i].eta_N,
                             protocol='B1 N-cluster bootstrap + B6 N-cluster bootstrap; Q refit conditional on B1 draw'))
        except Exception as exc:fail.append(dict(replicate=i,error=str(exc)))
    par=write(pars,'joint_bootstrap_parameters.csv');write(pd.DataFrame(fail,columns=['replicate','error']),'joint_bootstrap_failures.csv')
    if len(par)<150:raise RuntimeError(f'Only {len(par)} bootstrap fits succeeded')
    joblib.dump({'parameters':par,'seed_B1':20260924,'seed_B6':20260923,'cross_source_covariance_identified':False},M/'joint_bootstrap_models.joblib')
    opt=[]
    for _,r in par.iterrows():
        mm=Model(*[float(r[k]) for k in ['E','A','alpha','B','beta','G','kappa','eta_N']])
        for budget in BUDGETS:
            z=solve(mm,budget,.6,4096,'exponential')
            opt.append(dict(replicate=int(r.replicate),**z))
    opt=write(opt,'joint_bootstrap_optima.csv')
    summaries=[];stability=[]
    nmax=float(b1.N_params_B.max());dmax=float(b1.D_tokens_B.max())
    for b,z in opt.groupby('budget'):
        for var in ['N_B','D_B','Q_B','loss']:
            summaries.append(dict(budget=b,variable=var,n=len(z),median=z[var].median(),p05=z[var].quantile(.05),p25=z[var].quantile(.25),p75=z[var].quantile(.75),p95=z[var].quantile(.95)))
        stability.append(dict(budget=b,n=len(z),P_Q_at_1=float(np.mean(z.Q_B>=1-1e-5)),P_Q_at_Q0=float(np.mean(z.Q_B<=.60001)),
                              P_N_outside_B1=float(np.mean(z.N_B>nmax)),P_D_outside_B1=float(np.mean(z.D_B>dmax)),
                              P_both_strong=float(np.mean((z.N_B>3*nmax)&(z.D_B>3*dmax)))))
    write(summaries,'joint_bootstrap_summary.csv');stability=write(stability,'allocation_stability.csv')
    # Balanced scenario ensemble: each joint replicate is paired with one
    # cost/Q0/context combination. This is sensitivity, not a confidence law.
    cost=list(COST_MODELS);q0s=[.2,.6,.8];ctxs=[2048,4096,32768];con=[]
    for i,(_,r) in enumerate(par.iterrows()):
        mm=Model(*[float(r[k]) for k in ['E','A','alpha','B','beta','G','kappa','eta_N']])
        kind=cost[i%3];q0=q0s[(i//3)%3];ctx=ctxs[(i//9)%3]
        for b in [1e19,1e22,1e24]:con.append(dict(replicate=int(r.replicate),cost_model=kind,Q0=q0,L_ctx=ctx,**solve(mm,b,q0,ctx,kind)))
    con=write(con,'scenario_ensemble_optima.csv')
    consensus=[]
    for b,z in con.groupby('budget'):
        consensus.append(dict(budget=b,n=len(z),N_median=z.N_B.median(),N_p05=z.N_B.quantile(.05),N_p25=z.N_B.quantile(.25),N_p75=z.N_B.quantile(.75),N_p95=z.N_B.quantile(.95),
                              D_median=z.D_B.median(),D_p05=z.D_B.quantile(.05),D_p25=z.D_B.quantile(.25),D_p75=z.D_B.quantile(.75),D_p95=z.D_B.quantile(.95),
                              Q_median=z.Q_B.median(),Q_p05=z.Q_B.quantile(.05),Q_p25=z.Q_B.quantile(.25),Q_p75=z.Q_B.quantile(.75),Q_p95=z.Q_B.quantile(.95),
                              P_Q_at_1=float(np.mean(z.Q_B>=1-1e-5)),P_Q_at_Q0=float(np.mean(abs(z.Q_B-z.Q0)<1e-5))))
    consensus=write(consensus,'resource_consensus.csv')
    return par,opt,stability,consensus


def extrapolation_work(model,b1):
    nb=(float(b1.N_params_B.min()),float(b1.N_params_B.max()))
    db=(float(b1.D_tokens_B.min()),float(b1.D_tokens_B.max()))
    rows=[]
    for budget in [1e19,1e22,1e24]:
        for level,factor in [('E0_empirical',1.),('E1_moderate_3x',3.)]:
            r=capped_solve(model,budget,.6,4096,'exponential',(nb[0],nb[1]*factor),(db[0],db[1]*factor))
            rows.append(dict(level=level,bound_factor=factor,bound_interpretation='B1 empirical' if factor==1 else '3x sensitivity boundary, not empirical support',**r))
        r=solve(model,budget,.6,4096,'exponential',global_search=True)
        rows.append(dict(level='E2_free_scaling',bound_factor=np.nan,bound_interpretation='unrestricted Model C extrapolation',**r))
    levels=write(rows,'extrapolation_levels.csv')
    # Soft log-distance penalty, only a robustness diagnostic.
    budget=1e24;soft=[]
    for omega in [.01,.05,.1,.25,1.]:
        bounds=[(np.log(1e-3),np.log(1e6)),(.6,1.)]
        def objective(x):
            n=np.exp(x[0]);q=x[1];d=d_from_budget(budget,n,q,.6,4096,'exponential')
            penalty=omega*(max(0.,np.log(n/nb[1]))**2+max(0.,np.log(d/db[1]))**2)
            return model.loss(n,d,q)+penalty
        de=differential_evolution(objective,bounds,seed=20260924,popsize=12,maxiter=120,tol=1e-10)
        r=minimize(objective,de.x,method='SLSQP',bounds=bounds,options={'ftol':1e-12,'maxiter':500})
        x=r.x if r.fun<=de.fun else de.x;n=np.exp(x[0]);q=x[1];d=d_from_budget(budget,n,q,.6,4096,'exponential')
        soft.append(dict(omega=omega,budget=budget,N_B=n,D_B=d,Q_B=q,raw_loss=model.loss(n,d,q),penalized_objective=objective(x),
                         N_over_B1_max=n/nb[1],D_over_B1_max=d/db[1]))
    write(soft,'soft_extrapolation_penalty.csv')
    return levels


def cap_profile_work(model,b1):
    """Test how the arbitrary E1 cap affects the conditional optimum."""
    nb=(float(b1.N_params_B.min()),float(b1.N_params_B.max()))
    db=(float(b1.D_tokens_B.min()),float(b1.D_tokens_B.max()))
    rows=[]
    for budget in [1e22,1e24]:
        for factor in [1.,2.,3.,5.,10.]:
            r=capped_solve(model,budget,.6,4096,'exponential',
                           (nb[0],nb[1]*factor),(db[0],db[1]*factor),global_search=True)
            rows.append(dict(regime='capped',cap_factor=factor,**r))
        rows.append(dict(regime='free',cap_factor=np.nan,
                         **solve(model,budget,.6,4096,'exponential',global_search=True)))
    return write(rows,'cap_sensitivity_profile.csv')


def robust_work(model,par,b1,levels):
    nb=(float(b1.N_params_B.min()),float(b1.N_params_B.max())*3)
    db=(float(b1.D_tokens_B.min()),float(b1.D_tokens_B.max())*3)
    # Three true joint parameter vectors selected by G ranks, without mixing
    # components across bootstrap replicates.
    z=par.sort_values('G').reset_index(drop=True)
    picks=[z.iloc[int((len(z)-1)*q)] for q in [.05,.5,.95]]
    scenarios=[]
    for j,r in enumerate(picks):
        mm=Model(*[float(r[k]) for k in ['E','A','alpha','B','beta','G','kappa','eta_N']])
        for kind in COST_MODELS:
            for q0 in [.2,.8]:
                for ctx in [2048,32768]:
                    scenarios.append(dict(id=f'b{int(r.replicate)}_{kind}_q{q0}_c{ctx}',model=mm,kind=kind,q0=q0,ctx=ctx,bootstrap_replicate=int(r.replicate)))
    rows=[];scenario_rows=[]
    for budget in [1e19,1e22,1e24]:
        for s in scenarios:
            opt=capped_solve(s['model'],budget,s['q0'],s['ctx'],s['kind'],nb,db,global_search=False)
            s[f'opt_{budget:.0e}']=opt['loss']
            scenario_rows.append(dict(budget=budget,scenario=s['id'],cost_model=s['kind'],Q0=s['q0'],L_ctx=s['ctx'],bootstrap_replicate=s['bootstrap_replicate'],
                                      optimum_loss=opt['loss'],N_B=opt['N_B'],D_B=opt['D_B'],Q_B=opt['Q_B']))
        def common_d(n,q):return min(db[1],min(d_from_budget(budget,n,q,s['q0'],s['ctx'],s['kind']) for s in scenarios))
        def regrets(n,d,q):return np.array([s['model'].loss(n,d,q)-s[f'opt_{budget:.0e}'] for s in scenarios])
        def objective(x):
            n=np.exp(x[0]);q=x[1];d=common_d(n,q)
            if d<db[0]:return 1e6+(db[0]-d)*1e3
            return max(regrets(n,d,q))
        bounds=[tuple(np.log(nb)),(.8,1.)]
        de=differential_evolution(objective,bounds,seed=20260924,popsize=16,maxiter=160,tol=1e-9,polish=False)
        loc=minimize(objective,de.x,method='SLSQP',bounds=bounds,options={'ftol':1e-12,'maxiter':500})
        x=loc.x if loc.fun<=de.fun else de.x;n=np.exp(x[0]);q=x[1];d=common_d(n,q)
        candidates=[('robust_minimax',n,d,q)]
        nominal=levels[(levels.budget==budget)&(levels.level=='E1_moderate_3x')].iloc[0]
        # Raw nominal may violate the common Q0 and worst-case cost. Repair only
        # for a fair feasible regret comparison; preserve raw nominal separately.
        nq=max(.8,float(nominal.Q_B));nd=min(float(nominal.D_B),common_d(float(nominal.N_B),nq))
        candidates.append(('nominal_moderate_repaired',float(nominal.N_B),nd,nq))
        empirical=levels[(levels.budget==budget)&(levels.level=='E0_empirical')].iloc[0]
        eq=max(.8,float(empirical.Q_B));ed=min(float(empirical.D_B),common_d(float(empirical.N_B),eq))
        candidates.append(('empirical_repaired',float(empirical.N_B),ed,eq))
        for label,nn,dd,qq in candidates:
            rr=regrets(nn,dd,qq)
            rows.append(dict(budget=budget,policy=label,N_B=nn,D_B=dd,Q_B=qq,nominal_loss=model.loss(nn,dd,qq),
                             max_regret=max(rr),q90_regret=np.quantile(rr,.9),median_regret=np.median(rr),
                             worst_cost_ratio=max(total_cost(nn,dd,qq,s['q0'],s['ctx'],s['kind'])[0]/budget for s in scenarios),
                             scenario_count=len(scenarios),extrapolation_bound='E1 3x B1'))
        raw_feasible=np.mean([float(nominal.Q_B)>=s['q0'] and total_cost(nominal.N_B,nominal.D_B,nominal.Q_B,s['q0'],s['ctx'],s['kind'])[0]<=budget*(1+1e-8) for s in scenarios])
        rows.append(dict(budget=budget,policy='nominal_moderate_raw',N_B=nominal.N_B,D_B=nominal.D_B,Q_B=nominal.Q_B,
                         nominal_loss=nominal.loss,max_regret=np.nan,q90_regret=np.nan,median_regret=np.nan,worst_cost_ratio=max(total_cost(nominal.N_B,nominal.D_B,nominal.Q_B,s['q0'],s['ctx'],s['kind'])[0]/budget for s in scenarios),
                         scenario_feasible_fraction=raw_feasible,scenario_count=len(scenarios),extrapolation_bound='E1 3x B1'))
    robust=write(rows,'robust_optimization.csv');write(scenario_rows,'robust_scenario_optima.csv')
    return robust


def verification_work(model,levels,par,opt):
    rng=np.random.default_rng(20260924);checks=[]
    for i in range(100):
        n=10.**rng.uniform(-1,2);q=rng.uniform(.6,1);budget=10.**rng.uniform(18,24);ctx=int(rng.choice([2048,4096,32768]));kind=str(rng.choice(COST_MODELS))
        analytic=d_from_budget(budget,n,q,.6,ctx,kind)
        numeric=brentq(lambda d:total_cost(n,d,q,.6,ctx,kind)[0]-budget,0,max(1.,2*analytic),xtol=1e-12,rtol=1e-14)
        checks.append(dict(draw=i,budget=budget,N_B=n,Q_B=q,L_ctx=ctx,cost_model=kind,D_analytic=analytic,D_numeric=numeric,relative_error=abs(analytic-numeric)/analytic))
    elim=write(checks,'budget_elimination_verification.csv')
    nominal=pd.read_csv(ROOT/'problem3/outputs/tables/optimal_allocations_by_cost.csv')
    kkt=[]
    for _,r in nominal.iterrows():
        q0=float(r.Q0);ctx=int(r.L_ctx);kind=r.cost_model
        kkt.append(dict(source='V1_nominal',cost_model=kind,budget=r.budget,replicate=-1,**kkt_diagnostic(model,r,q0,ctx,kind)))
    # Joint bootstrap diagnostics on the four budget levels.
    indexed=par.set_index('replicate')
    for _,r in opt.iterrows():
        z=indexed.loc[int(r.replicate)];mm=Model(*[float(z[k]) for k in ['E','A','alpha','B','beta','G','kappa','eta_N']])
        kkt.append(dict(source='joint_bootstrap',cost_model='exponential',budget=r.budget,replicate=r.replicate,
                        **kkt_diagnostic(mm,r,.6,4096,'exponential')))
    kkt=write(kkt,'kkt_diagnostics.csv')
    free=levels[levels.level=='E2_free_scaling'];binding=abs(free.budget_ratio-1).max()
    verification={'V1_untouched':True,'bootstrap_replicates':len(par),'bootstrap_optimizations':len(opt),
                  'D_elimination_max_relative_error':float(elim.relative_error.max()),'D_elimination_pass':bool((elim.relative_error<1e-10).all()),
                  'free_budget_binding_max_error':float(binding),'free_budget_binding_pass':bool(binding<1e-6),
                  'KKT_median_relative_error':float(kkt.KKT_rel_error.median()),'KKT_max_relative_error':float(kkt.KKT_rel_error.max()),
                  'KKT_complementarity_rate':float(kkt.Q_complementarity_ok.mean()),
                  'KKT_pass':bool((kkt.KKT_rel_error<2e-3).all() and kkt.Q_complementarity_ok.all())}
    (T/'verification_results.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
    return verification


def elasticity_work(model):
    en=model.beta/(model.alpha+model.beta);ed=model.alpha/(model.alpha+model.beta)
    rows=[]
    for b in [1e22,1e24,1e25,1e27,1e29]:
        h=.001
        left=solve(model,b*10**(-h),.6,4096,'exponential')
        right=solve(model,b*10**h,.6,4096,'exponential')
        nnum=np.log(right['N_B']/left['N_B'])/(2*h*np.log(10))
        dnum=np.log(right['D_B']/left['D_B'])/(2*h*np.log(10))
        rows.append(dict(budget=b,theory_eN=en,theory_eD=ed,theory_D_over_N_exponent=ed-en,
                         numeric_eN=nnum,numeric_eD=dnum,numeric_D_over_N_exponent=dnum-nnum,
                         N_abs_gap=abs(nnum-en),D_abs_gap=abs(dnum-ed),Q_saturated=left['Q_B']>=1-1e-5 and right['Q_B']>=1-1e-5))
    return write(rows,'high_budget_elasticity.csv')


def figures(pm,pe,exact,ct,par,opt,levels,robust,consensus,elasticity):
    plt.rcParams['font.family']='DejaVu Sans'
    fig,axs=plt.subplots(1,3,figsize=(12,3.8))
    for j,metric in enumerate(['knn5','mahalanobis','pca_residual']):
        z=pm[['strategy',metric,f'{metric}_p95',f'{metric}_p99']]
        axs[j].bar(range(len(z)),z[metric]);axs[j].axhline(z[f'{metric}_p95'].iloc[0],color='orange',ls='--',label='95% train')
        axs[j].axhline(z[f'{metric}_p99'].iloc[0],color='red',ls=':',label='99% train');axs[j].set_title(metric);axs[j].set_xticks(range(len(z)),['P0','P1','P2','Loose']);axs[j].legend(fontsize=7)
    fig.tight_layout();fig.savefig(F/'mixture_support_diagnostics.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    p1=pe[pe.strategy=='P1_observed_support']
    for sc,z in p1.groupby('scenario'):
        ax.plot(z.N_B,-z.delta_loss,marker='o',label=sc)
    ax.set_xscale('log');ax.set(xlabel='N (billion)',ylabel='predicted mixture Loss improvement');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(F/'lambda_p_decay_scenarios.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for kind,z in exact.groupby('cost_model'):
        z=z[z.status=='crossing'];ax.scatter(z.C_threshold,[0 if x=='start' else 1 for x in z.transition],label=kind,s=70)
    ax.set_xscale('log');ax.set_yticks([0,1],['Q starts','Q saturates']);ax.set(xlabel='budget FLOPs');ax.legend();fig.tight_layout();fig.savefig(F/'quality_transition_exact.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for ctx,z in ct[ct.cost_model=='exponential'].groupby('L_ctx'):
        z=z[z.status=='crossing'];ax.scatter(z.C_threshold,[0 if x=='start' else 1 for x in z.transition],label=str(ctx),s=65)
    ax.set_xscale('log');ax.set_yticks([0,1],['Q starts','Q saturates']);ax.legend(title='context');fig.tight_layout();fig.savefig(F/'transition_by_context.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(12,3.8))
    for j,var in enumerate(['N_B','D_B','Q_B']):
        data=[opt[opt.budget==b][var].to_numpy() for b in BUDGETS if b!=1e20]
        axs[j].boxplot(data,tick_labels=['1e19','1e22','1e24']);axs[j].set_title(var)
        if j<2:axs[j].set_yscale('log')
    fig.tight_layout();fig.savefig(F/'joint_bootstrap_optima.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(12,3.8))
    for j,b in enumerate([1e19,1e22,1e24]):
        z=levels[levels.budget==b];axs[j].bar(z.level,z.loss);axs[j].set_title(f'{b:.0e} FLOPs');axs[j].tick_params(axis='x',rotation=25,labelsize=7);axs[j].set_ylabel('Loss')
    fig.tight_layout();fig.savefig(F/'extrapolation_levels.png',dpi=180);plt.close(fig)
    cap=pd.read_csv(T/'cap_sensitivity_profile.csv')
    cap['cap_factor']=pd.to_numeric(cap.cap_factor,errors='coerce')
    cap=cap[cap.budget==1e24].sort_values('cap_factor',na_position='last')
    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot(range(len(cap)),cap.loss,marker='o')
    ax.set_xticks(range(len(cap)),[f'{int(x)}x' if pd.notna(x) else 'free' for x in cap.cap_factor])
    ax.set(xlabel='B1 N/D upper-bound multiplier',ylabel='conditional optimal Loss',title='10^24 FLOPs cap sensitivity')
    ax.grid(alpha=.2);fig.tight_layout();fig.savefig(F/'cap_sensitivity_profile.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for label,z in robust[robust.policy!='nominal_moderate_raw'].groupby('policy'):
        ax.plot(z.budget,z.max_regret,marker='o',label=label)
    ax.set_xscale('log');ax.set(xlabel='budget FLOPs',ylabel='maximum scenario regret');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(F/'robust_regret.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));ax.plot(elasticity.budget,elasticity.numeric_eN,marker='o',label='N numeric');ax.plot(elasticity.budget,elasticity.numeric_eD,marker='o',label='D numeric');ax.axhline(elasticity.theory_eN.iloc[0],ls='--',color='C0',label='N theory');ax.axhline(elasticity.theory_eD.iloc[0],ls='--',color='C1',label='D theory');ax.set_xscale('log');ax.set(xlabel='budget FLOPs',ylabel='budget elasticity');ax.legend();fig.tight_layout();fig.savefig(F/'high_budget_elasticity.png',dpi=180);plt.close(fig)


def reports(model,pm,pe,exact,ct,breaks,par,opt,stability,consensus,levels,robust,verification,elasticity):
    fmt=lambda v:f'{v:.5g}'
    exlines='\n'.join(f"| {r.cost_model} | {r.transition} | {fmt(r.C_threshold) if r.status=='crossing' else r.status} | {r.status} |" for _,r in exact.iterrows())
    ctxlines='\n'.join(f"| {r.cost_model} | {int(r.L_ctx)} | {r.transition} | {fmt(r.C_threshold) if r.status=='crossing' else r.status} |" for _,r in ct.iterrows())
    level_lines='\n'.join(f"| {r.budget:.0e} | {r.level} | {fmt(r.N_B)} | {fmt(r.D_B)} | {fmt(r.Q_B)} | {fmt(r.loss)} | {r.budget_ratio:.1%} |" for _,r in levels.iterrows())
    cap=pd.read_csv(T/'cap_sensitivity_profile.csv')
    cap['cap_factor']=pd.to_numeric(cap.cap_factor,errors='coerce')
    cap=cap[cap.budget==1e24].sort_values('cap_factor',na_position='last')
    cap_lines='\n'.join(f"| {int(r.cap_factor) if pd.notna(r.cap_factor) else 'free'} | {fmt(r.N_B)} | {fmt(r.D_B)} | {fmt(r.loss)} | {r.budget_ratio:.1%} |" for _,r in cap.iterrows())
    scale_evidence=pd.read_csv(T/'p_scale_evidence_audit.csv').set_index('dataset')
    slope_text='/'.join(f"{scale_evidence.loc[name,'centered_slope']:.3f}" for name in ['test_1m','test_60m','test_1B'])
    stable_lines='\n'.join(f"| {r.budget:.0e} | {r.P_Q_at_1:.1%} | {r.P_Q_at_Q0:.1%} | {r.P_N_outside_B1:.1%} | {r.P_D_outside_B1:.1%} | {r.P_both_strong:.1%} |" for _,r in stability.iterrows())
    robust_lines='\n'.join(f"| {r.budget:.0e} | {r.policy} | {fmt(r.N_B)} | {fmt(r.D_B)} | {fmt(r.Q_B)} | {fmt(r.nominal_loss)} | {fmt(r.max_regret) if pd.notna(r.max_regret) else 'infeasible policy'} |" for _,r in robust.iterrows())
    supp_lines='\n'.join(f"| {r.strategy} | {fmt(r.delta_p)} | {r.support_class} | {fmt(r.knn5)} | {fmt(r.mahalanobis)} | {fmt(r.pca_residual)} |" for _,r in pm.iterrows())
    boot_lines='\n'.join(f"| {r.budget:.0e} | {r.variable} | {fmt(r.p05)} | {fmt(r.p25)} | {fmt(r['median'])} | {fmt(r.p75)} | {fmt(r.p95)} |" for _,r in pd.read_csv(T/'joint_bootstrap_summary.csv').iterrows())
    con_lines='\n'.join(f"| {r.budget:.0e} | {fmt(r.N_median)} [{fmt(r.N_p05)},{fmt(r.N_p95)}] | {fmt(r.D_median)} [{fmt(r.D_p05)},{fmt(r.D_p95)}] | {fmt(r.Q_median)} [{fmt(r.Q_p05)},{fmt(r.Q_p95)}] | {r.P_Q_at_1:.1%} |" for _,r in consensus.iterrows())
    e24=elasticity[elasticity.budget==1e24].iloc[0]
    report=f'''# 问题三 V2：证据约束下的资源配置

## 1 审计与定位

V1 是保留不变的基线。V2 读取其 Model C、三项成本、三类质量成本、C7 情景、Q0、三档结果、71 点预算网格、p 优化、KKT、外推及不确定性表。V1 已正确实现 $C_{{train}}=6\\times10^{{18}}N_BD_B$、$C_Q=10^9D_B[g(Q)-g(Q_0)]_+$、$C_{{attn}}=2\\times10^{{14}}N_BD_BL_{{ctx}}$；临界上下文仍为 30000。V2 不改动这些公式。

V1 的主要证据缺口是 λp 跨尺度未识别、结构点只有 0.1 dex 网格精度、参数边际独立采样、高预算 free 解强外推。V2 的目标是提高证据等级，不追求更低的 nominal Loss。

## 2 配比效应和跨尺度可信度

问题二保存的最终 P1 V2 在 1M/60M/1B 的中心化斜率分别为 {slope_text}，用的是已查看过的测试标签，仅为回顾性证据，不是 λp(N) 的估计。1B 配方均值排序在 V2 下弱于 V1；旧 V1 斜率另存 `p_scale_v1_evidence_audit.csv`。本问以 1M ($N_B=.001$) 为情景归一化点，比较常数 1、幂次衰减 ρ=.1/.2、以及 ρ=.15 且地板 .25。固定 λ 枚举 0、.25、.5、.75、1、1.25、1.5。模型可分意味着 λ>0 时 p 候选排序不变，N-D-Q 不变，收益按 λ 伸缩；λ=0 时所有 p 并列。

| 策略 | λ=1 的 ΔLoss | 变换空间支持等级 | kNN5 | Mahalanobis | PCA 残差 |
|---|---:|---|---:|---:|---:|
{supp_lines}

支持度采用与问题一一致的 ALR 变换，训练空间标准化后计算 5 近邻距离、保留 95% 方差的 PCA 坐标收缩协方差 Mahalanobis 距离和 PCA 重构残差。各指标阈值来自 512 条训练样本 95/99 分位数；in-support/near-support/extrapolative 是经验诊断，不是外部验证。P2 连续候选仅在最优观测配比的 12 个邻近训练配比凸组合内搜索，并须通过全部 95% 支持检验。P1 实际观测配比仍为论文主候选；Loose 只作数学诊断。

## 3 精确结构点

对 $Q^*>Q_0+10^{{-4}}$ 与 $Q^*\\ge1-10^{{-4}}$ 的首次预算，先扫描括区间，再在 log10 预算上二分到不足 $2\\times10^{{-6}}$ dex。端点 cusp 用全局候选和边界搜索保护。若在扫描区间外才触发，明确报告 below_scan/above_scan，不伪造数值。精度是给定 ε 与模型的数值精度，不是统计置信区间。

| 成本模型 | 事件 | FLOPs | 状态 |
|---|---|---:|---|
{exlines}

| 成本模型 | C7 上下文 | 事件 | FLOPs |
|---|---:|---|---:|
{ctxlines}

另外对 log N、log D、Q、质量预算份额做 0/1/2 折点的连续分段线性拟合，以 BIC 选择并计入折点位置参数。结果见 `structural_break_analysis.csv`。分段折点描述轨迹斜率，不应与 Q 活跃约束转移混为同一物理临界点。

## 4 联合参数传播

问题二的 200 组 B6 聚类 bootstrap 原样用于审计。由于未保存 B1 的联合 bootstrap，V2 对 B1 的 8 个 N 簇重采样并拟合五个经典参数；对 B6 的 9 个 N 簇采用问题二相同随机种子重放抽样，再在对应 B1 参数下拟合 G、κ、η。这样保留 B6 三参数内部依赖，并传播 B1 参数到质量拟合的条件影响。B1/B6 属不同来源，其跨源协方差未识别；这些分位数是**参数 bootstrap 条件下**的优化分布，不是完整现实置信区间。

| FLOPs | 变量 | 5% | 25% | 中位 | 75% | 95% |
|---:|---|---:|---:|---:|---:|---:|
{boot_lines}

| FLOPs | P(Q=1) | P(Q=Q0) | P(N>B1max) | P(D>B1max) | P(两维>3×B1max) |
|---:|---:|---:|---:|---:|---:|
{stable_lines}

跨成本、Q0、上下文、联合参数的平衡情景集合是敏感性集合，不赋予自然发生概率：

| FLOPs | N 中位 [5%,95%] | D 中位 [5%,95%] | Q 中位 [5%,95%] | Q=1 情景比率 |
|---:|---:|---:|---:|---:|
{con_lines}

## 5 外推三级与稳健决策

E0 把 N,D 限于 B1 观测范围；E1 限于 B1 上界 3 倍（明确是敏感性边界）；E2 是自由 Scaling Law。硬约束下 D 在每个 N,Q 取预算允许与 D 上界的较小值，避免 V1 对封顶问题使用惩罚近似。软 log 越界平方罚项另做 ω=.01 至 1 的诊断，不作为主模型。

| FLOPs | 层级 | N_B | D_B | Q_B | Loss | 已用预算 |
|---:|---|---:|---:|---:|---:|---:|
{level_lines}

3× 上限并非实测可信域；将两维上限同时放宽至 B1 最大值的 1/2/3/5/10 倍，另与自由解比较。10^24 FLOPs 的剖面如下（完整结果在 `cap_sensitivity_profile.csv`）：

| 上限倍数 | N_B | D_B | Loss | 已用预算 |
|---:|---:|---:|---:|---:|
{cap_lines}

3× 到 5× 的 Loss 仍明显变化，所以 E1 只是一种明确的条件情景，不是唯一可推荐规模。10^22 FLOPs 下这些上限均未起作用。![上限剖面](outputs/figures/cap_sensitivity_profile.png)

稳健决策采用 E1 3× B1 上界。代表场景为 3 个真实联合 bootstrap 向量 × 三种成本 × 两个 Q0 × 两个 C7 长度，共 36 个场景。固定配置要求对全部场景可行，故共同 Q≥.8；D 由各场景允许值的最小值决定。每场景后悔值为其 Loss 与同场景 E1 最优 Loss 的差，目标最小化最大后悔值。λp 对固定 p 策略的 N-D-Q 决策可分，因此其不确定性在配比结论中单列，不虚构为 N-D-Q 风险。对 nominal 和 empirical 配置另外报告共同可行域修复后的公平比较。

| FLOPs | 配置 | N_B | D_B | Q_B | nominal Loss | 最大场景后悔 |
|---:|---|---:|---:|---:|---:|---:|
{robust_lines}

这是一套明确的稳健决策规则，不等于数据证明 3× 边界合理。若未修复的 nominal 配置在其他成本/上下文情景不可行，其 raw regret 不计算。

## 6 KKT、消元与高预算解析

自由解的预算使用率误差最大 {verification['free_budget_binding_max_error']:.2e}。100 个随机点对 $D=C/[(6+ηL)10^{{18}}N+10^9(g(Q)-g(Q0))_+]$ 与 Brent 数值根比较，最大相对误差 {verification['D_elimination_max_relative_error']:.2e}。自由解及联合 bootstrap 内点/边界 KKT 中位相对误差 {verification['KKT_median_relative_error']:.2e}，最大 {verification['KKT_max_relative_error']:.2e}；Q 端点采用单侧互补条件。若极端 bootstrap 样本 KKT 异常，应查看 `kkt_diagnostics.csv`，不可掩盖。

当 $Q=1$ 且质量成本份额趋小，令 $K=C/[(6+ηL)10^{{18}}]$，有 $D=K/N$。一阶条件给 $N^*=[(αA)/(βB)K^β]^{{1/(α+β)}}$，故 $e_N=β/(α+β)={model.beta/(model.alpha+model.beta):.6f}$、$e_D=α/(α+β)={model.alpha/(model.alpha+model.beta):.6f}$。D/N 的理论预算指数为 {(model.alpha-model.beta)/(model.alpha+model.beta):.6f}>0，解释高预算 D/N 上升。在 10^24 数值局部弹性为 N={e24.numeric_eN:.6f}、D={e24.numeric_eD:.6f}；质量成本仍非零，所以仅渐近靠近理论，详见 `high_budget_elasticity.csv`。

## 7 证据等级与结论

三种质量成本是同等级题设情景：中高预算 Q 饱和是跨情景共识；低预算质量投资高度依赖成本假设。联合参数 bootstrap 区间很窄，主要反映 B1 在固定经典公式下近乎确定性的轨迹，**不能**解释为现实模型参数或跨数据源误差也同样小。它不覆盖成本函数、Q0、C7 元数据与 B8 外部来源失配。跨场景共识区间才用于展示这些结构性假设的变化。

长上下文使同样 N-D 的成本增加，主情景下反而让相对便宜的 Q 更早值得投资：指数成本的起投点从 2048 tokens 的约 3.90×10^18 降到 32768 tokens 的约 2.21×10^18；这是给定成本结构与 Model C 的替代效应，不是通用规律。对数成本的起投与饱和同点，表示最优 Q 出现离散跳跃，不能把它解释为渐进质量投资阶段。

10^24 下 E1 的 N 与 D 同时碰到 3× 上界，全部 36 个场景在此角点共享同一个最优解，所以 minimax regret=0；这是一种**边界驱动的退化结果**，不是高预算风险已被消除。软罚解在 ω=.01–1 下仍达到 N≈60–75B、D≈1.9–2.4T，与 E1 的 N≈35.9B、D≈0.90T 相距明显，软罚并未独立支持 3× 推荐。E1 缓和强外推，却不能升级为实证解；预算仍有大量未使用。

论文主文应采用 V2 的跨情景共识与 E0/E1/E2 三层对照；单一 λp=1 收益、精确结构点及 E2 高预算 Loss 只作为条件情景结果。
'''
    (HERE/'problem3_v2_report.md').write_text(report,encoding='utf-8')
    (HERE/'问题三第二版完整详解.md').write_text(report+'''\n## 8 复现与文件映射\n\n从仓库根目录运行 `python problem3_v2/run_v2.py`。`methods.py` 只读引入 V1 的数学成本与预测函数，新增精确活动集、硬封顶求解和 KKT 诊断。`mixture_evidence.py` 重算配比训练支持和 λ 情景。`run_v2.py` 负责联合 bootstrap、E0/E1/E2、minimax regret、图表与报告。全部 V2 文件保存在 `problem3_v2/outputs`；V1 的 SHA-256 前后清单写入 `v1_integrity.json`。\n\n所有以 3×、ρ、floor、场景等标记的数值都是敏感性假设。`joint_bootstrap_parameters.csv` 可审计每个模型八参数及其原 B6 replicate；`joint_bootstrap_optima.csv` 逐次保存四档优化结果。`robust_scenario_optima.csv` 保存后悔值的逐场景基准，保证 minimax regret 可独立复算。\n''',encoding='utf-8')
    p1=pm[pm.strategy=='P1_observed_support'].iloc[0]
    e24levels=levels[levels.budget==1e24]
    short=f'''# 问题三 V2 摘要\n\nV1 保留不变。V2 精化结构点、核查 ALR 配比支持、联合重采样八个 Model C 参数，并给出 E0/E1/E2 与 36 场景 minimax regret。\n\nP1 观测配比在 λ=1 下预测改善 {abs(p1.delta_p):.4f}；λ 随 N 衰减时收益按情景降低，尚无跨尺度联合识别。Loose 配比的支持等级：{pm[pm.strategy=='P3_loose_simplex'].iloc[0].support_class}。\n\n结构点：{'; '.join(f'{r.cost_model}/{r.transition}: {fmt(r.C_threshold) if r.status=="crossing" else r.status}' for _,r in exact.iterrows())}。\n\n高预算 10^24 的三级结果：{'; '.join(f'{r.level}: N={fmt(r.N_B)}, D={fmt(r.D_B)}, Q={fmt(r.Q_B)}, Loss={fmt(r.loss)}, used={r.budget_ratio:.1%}' for _,r in e24levels.iterrows())}。\n\n联合 bootstrap 的 Q=1 稳定率：{'; '.join(f'{r.budget:.0e}:{r.P_Q_at_1:.1%}' for _,r in stability.iterrows())}。理论弹性 eN={model.beta/(model.alpha+model.beta):.4f}、eD={model.alpha/(model.alpha+model.beta):.4f}。E1 3× 下的 minimax regret 解是一种条件方案；上限剖面显示高预算结果对边界敏感，因此不给唯一规模推荐。\n'''
    (HERE/'problem3_v2_summary.md').write_text(short,encoding='utf-8')
    comparison='''# 问题三 V1 与 V2 比较\n\n| 项目 | V1 基线 | V2 |\n|---|---|---|\n| 三档预算 | 已求解 | 保留并做证据对照 |\n| 结构点 | 71 点网格首次触发 | log 预算二分的活动集转移，另做 BIC 分段 |\n| 参数不确定性 | G、κ、η 边际独立抽样 | B1/B6 簇 bootstrap，八参数条件联合传播 |\n| p 支持度 | 最近邻基础指标 | ALR 5NN、PCA Mahalanobis、重构残差 |\n| λp | 0.5–1.5 常数 | 0–1.5 常数及三类规模衰减情景 |\n| 外推 | empirical/free | empirical/moderate 3×/free 及软罚 |\n| 优化 | nominal | nominal 与 36 场景 minimax regret |\n| KKT | 三档基础比值 | 全部主场景和 joint bootstrap 定量残差 |\n| 高预算解析 | 数值轨迹 | 理论预算弹性及数值检验 |\n\nV2 不替换 V1 的成本单位、Model C 或原始数据。V2 更适合论文主结论，因为每个强假设都有明确标签和对照。\n'''
    (HERE/'problem3_v1_v2_comparison.md').write_text(comparison,encoding='utf-8')
    (HERE/'README.md').write_text('# Problem 3 V2\n\nRun `python problem3_v2/run_v2.py` from the repository root. V1 is read-only and its SHA-256 integrity is checked. See `problem3_v2_report.md`, `问题三第二版完整详解.md`, and `outputs/tables/cap_sensitivity_profile.csv` and `outputs/tables/verification_results.json`. The 3x E1 cap is a sensitivity scenario.\n',encoding='utf-8')


def main():
    before=manifest();artifact,model,b1,b6,p,old_mix=load()
    v1=pd.read_csv(ROOT/'problem3/outputs/tables/optimal_allocations.csv')
    pm,pe=mixture_work(artifact,p,old_mix,v1)
    exact,ct,breaks=transition_work(model)
    par,opt,stability,consensus=bootstrap_work(model,b1,b6)
    levels=extrapolation_work(model,b1)
    cap_profile_work(model,b1)
    robust=robust_work(model,par,b1,levels)
    verification=verification_work(model,levels,par,opt)
    elasticity=elasticity_work(model)
    figures(pm,pe,exact,ct,par,opt,levels,robust,consensus,elasticity)
    reports(model,pm,pe,exact,ct,breaks,par,opt,stability,consensus,levels,robust,verification,elasticity)
    after=manifest();verification['V1_untouched']=(before==after)
    (T/'v1_integrity.json').write_text(json.dumps({'unchanged':before==after,'files_before':len(before),'files_after':len(after),'sha256':before},ensure_ascii=False,indent=2),encoding='utf-8')
    (T/'verification_results.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
    if not verification['V1_untouched'] or not verification['D_elimination_pass'] or not verification['free_budget_binding_pass']:
        raise AssertionError('V2 verification failed')
    print('V2 complete:',len(par),'joint bootstrap replicates;',len(opt),'optimizations')
    print(exact[['cost_model','transition','status','C_threshold']].to_string(index=False))
    print(levels[levels.budget==1e24][['level','N_B','D_B','Q_B','loss','budget_ratio']].to_string(index=False))


if __name__=='__main__':main()
