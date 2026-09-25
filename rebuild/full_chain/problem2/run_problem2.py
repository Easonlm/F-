"""Run the complete, reproducible analysis for contest problem two."""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from generalized_scaling import predict_ndq, derivatives, equivalent_parameters, predict_generalized

ROOT=Path(__file__).resolve().parents[1]
BROOT=ROOT/'real_attachments'/'B_scaling_laws'
P1=ROOT/'problem1'/'outputs'
OUT=Path(__file__).resolve().parent/'outputs'
T=OUT/'tables'; F=OUT/'figures'; M=OUT/'models'
SEED=20260923
PKEY=['E','A','alpha','B','beta']
LOW=np.array([0.0,1e-5,.005,1e-5,.005])
HIGH=np.array([4.0,100,2,100,2])
STARTS=[np.array([1.5,.5,.2,.5,.2]),np.array([1.0,1,.1,1,.1]),np.array([2,.3,.4,.3,.4]),np.array([.5,2,.05,2,.05])]

def save(df,name):
    pd.DataFrame(df).to_csv(T/name,index=False,float_format='%.10g')

def metrics(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float)
    good=np.isfinite(y)&np.isfinite(p)
    y=y[good];p=p[good]
    rho=spearmanr(y,p).statistic if len(y)>2 and np.std(y)>0 and np.std(p)>0 else np.nan
    return dict(n=len(y),rmse=float(np.sqrt(np.mean((p-y)**2))),mae=float(np.mean(abs(p-y))),
                r2=float(r2_score(y,p)) if len(y)>1 else np.nan,spearman=float(rho),
                bias=float(np.mean(p-y)),relative_error=float(np.mean(abs(p-y)/np.maximum(abs(y),1e-9))))

def classic(x,N,D):
    E,A,a,B,b=x
    return E+A*np.exp(-a*np.log(N))+B*np.exp(-b*np.log(D))

def fit_classic(df,starts=STARTS,loss='linear',weights=None):
    N=df.N_params_B.to_numpy(float);D=df.D_tokens_B.to_numpy(float);y=df.val_loss.to_numpy(float)
    w=np.ones(len(y)) if weights is None else np.asarray(weights)
    fits=[]
    for s in starts:
        f=least_squares(lambda x:(classic(x,N,D)-y)*w,np.clip(s,LOW+1e-8,HIGH-1e-8),
                        bounds=(LOW,HIGH),loss=loss,max_nfev=1200,xtol=1e-10,ftol=1e-10)
        fits.append(f)
    best=min(fits,key=lambda f:np.sum(((classic(f.x,N,D)-y)*w)**2))
    return best, fits

def getp(x):return dict(zip(PKEY,map(float,x)))

def audit():
    rows=[]; datasets={}
    for path in sorted(BROOT.rglob('*.csv')):
        d=pd.read_csv(path);key=str(path.relative_to(BROOT));datasets[path.name]=d
        row=dict(file=key,rows=len(d),columns=len(d.columns),column_names='|'.join(d.columns),
                 missing_cells=int(d.isna().sum().sum()),duplicate_rows=int(d.duplicated().sum()),
                 duplicate_N_D=int(d.duplicated(['N_params_B','D_tokens_B']).sum()) if {'N_params_B','D_tokens_B'}<=set(d) else 0,
                 duplicate_checkpoint=int(d.duplicated(['run_id','steps']).sum()) if {'run_id','steps'}<=set(d) else 0,
                 model_families='|'.join(map(str,d.family.unique())) if 'family' in d else '',
                 units='N and D in billions' if 'N_params_B' in d else '',
                 provenance='real trajectory' if path.name.startswith('pythia_training_log') else
                   'interpolated Pythia checkpoints' if 'trajectory' in key else
                   'semi-synthetic' if path.name.startswith(('cerebras','supplementary_NQ')) else
                   'estimated loss' if path.name=='supplementary_large_baseline.csv' else 'metadata or literature/baseline')
        for col,lab in [('N_params_B','N'),('D_tokens_B','D'),('val_loss','Loss'),('Q_score','Q')]:
            row[lab+'_min']=float(d[col].min()) if col in d else np.nan
            row[lab+'_max']=float(d[col].max()) if col in d else np.nan
        rows.append(row)
    save(rows,'data_audit_b.csv')
    b6,b7,b8=[datasets[s] for s in ['supplementary_NQ_experiment.csv','supplementary_NQ_experiment_expanded.csv','supplementary_NQ_experiment_large.csv']]
    key=['N_params_B','D_tokens_B','Q_score']
    overlap=lambda a,b:len(a[key].merge(b[key].drop_duplicates(),on=key))
    rel=[dict(left='B6',right='B7',overlap_points=overlap(b6,b7),left_rows=len(b6),right_rows=len(b7)),
         dict(left='B7',right='B8',overlap_points=overlap(b7,b8),left_rows=len(b7),right_rows=len(b8)),
         dict(left='B6',right='B8',overlap_points=overlap(b6,b8),left_rows=len(b6),right_rows=len(b8))]
    save(rel,'quality_dataset_overlap.csv')
    relationships=rel+[dict(left='B1',right='B3',overlap_points=np.nan,left_rows=len(datasets['pythia_training_log_existing.csv']),right_rows=sum(len(d) for n,d in datasets.items() if n.endswith('_trajectory.csv')),note='B3 interpolates Pythia checkpoints; not independent'),
                       dict(left='B1',right='B4/B5',overlap_points=np.nan,left_rows=1176,right_rows=101,note='Validation loss evaluator and corpus not documented as identical; raw comparisons are distribution shift diagnostics')]
    save(relationships,'data_relationships_b.csv')
    return datasets

def classical(d):
    b1=d['pythia_training_log_existing.csv'].copy()
    assert len(b1)==1176
    group='N_params_B'
    result,fits=fit_classic(b1)
    p=getp(result.x)
    pred=classic(result.x,b1.N_params_B,b1.D_tokens_B)
    save([dict(parameter=k,estimate=v,lower_bound=LOW[i],upper_bound=HIGH[i],multistart_max_deviation=float(max(abs(f.x[i]-v) for f in fits)),jacobian_condition=float(np.linalg.cond(result.jac.T@result.jac))) for i,(k,v) in enumerate(p.items())],'classical_scaling_params.csv')
    covariance=np.linalg.pinv(result.jac.T@result.jac)*np.sum(result.fun**2)/max(len(b1)-5,1)
    std=np.sqrt(np.diag(covariance))
    corr=covariance/np.outer(std,std)
    save([dict(parameter_i=PKEY[i],parameter_j=PKEY[j],correlation=float(corr[i,j])) for i in range(5) for j in range(i+1,5)],'classical_parameter_correlations.csv')
    b1['prediction']=pred;b1['residual']=pred-b1.val_loss
    save(b1,'classical_scaling_predictions.csv')
    cv=[]
    for v in sorted(b1[group].unique()):
        tr=b1[b1[group]!=v];te=b1[b1[group]==v]
        f,_=fit_classic(tr)
        for name,pp in [('M1_ND',classic(f.x,te.N_params_B,te.D_tokens_B)),('M0_mean',np.full(len(te),tr.val_loss.mean()))]:
            cv.append(dict(model=name,held_out_N_B=v,**metrics(te.val_loss,pp)))
    save(cv,'classical_scaling_cv.csv')
    return p,b1,pd.DataFrame(cv)

def external(d,p,b1):
    rows=[]; predictions=[]
    for label,file in [('B2','cerebras_training_log.csv'),('B4','scaling_baseline.csv'),('B5','published_scaling_data.csv'),('B10','supplementary_large_baseline.csv')]:
        x=d[file].copy();pred=classic(np.array(list(p.values())),x.N_params_B,x.D_tokens_B)
        rows.append(dict(dataset=label,comparison='raw',provenance={'B2':'semi-synthetic','B4':'cross-family; loss protocol uncertain','B5':'published; loss protocol uncertain','B10':'estimated, not observation'}[label],**metrics(x.val_loss,pred)))
        if label in ('B4','B5'):
            offset=float(np.mean(x.val_loss-pred))
            rows.append(dict(dataset=label,comparison='descriptive_intercept_calibration_in_sample',provenance='not independent validation',**metrics(x.val_loss,pred+offset)))
        x['predicted']=pred;x['dataset']=label;predictions.append(x)
        if label=='B2':
            save([dict(run_id=str(g),**metrics(z.val_loss,z.predicted)) for g,z in x.groupby('run_id')],'b2_by_model_metrics.csv')
    traj=[]
    for name,x in d.items():
        if name.endswith('_trajectory.csv'):
            xx=x.copy();xx['predicted']=classic(np.array(list(p.values())),xx.N_params_B,xx.D_tokens_B);traj.append(xx)
    b3=pd.concat(traj,ignore_index=True)
    rows.append(dict(dataset='B3',comparison='interpolated_trajectory',provenance='Pythia interpolation, not independent',**metrics(b3.val_loss,b3.predicted)))
    save(rows,'external_validation_metrics.csv')
    save(b3,'b3_trajectory_predictions.csv')
    return pd.DataFrame(rows),pd.concat(predictions,ignore_index=True)

def qpred(x,N,D,Q,kind):
    p=getp(x[:5]);
    if kind in ('exponential','power'):p['gamma']=float(x[5])
    else:p.update(G=float(x[5]),kappa=float(x[6]))
    return predict_ndq(N,D,Q,p,kind)

def fit_quality(df,base,kind,mode,b1=None,loss='linear'):
    y=df.val_loss.to_numpy(float);N=df.N_params_B.to_numpy(float);D=df.D_tokens_B.to_numpy(float);Q=df.Q_score.to_numpy(float)
    extra=[.5] if kind!='additive' else [.3,1.0]
    lo=np.r_[LOW,[0.0] if kind!='additive' else [0.0,.1]]
    hi=np.r_[HIGH,[15.0] if kind!='additive' else [10.0,4.0]]
    if mode=='two_stage':
        def r(z):return qpred(np.r_[list(base.values()),z],N,D,Q,kind)-y
        f=least_squares(r,extra,bounds=(lo[5:],hi[5:]),loss=loss,max_nfev=1000)
        x=np.r_[list(base.values()),f.x]
    else:
        bn=b1.N_params_B.to_numpy(float);bd=b1.D_tokens_B.to_numpy(float);by=b1.val_loss.to_numpy(float)
        # Equal total squared-error weight for the real and synthetic sources.
        def r(z):return np.r_[(classic(z[:5],bn,bd)-by)/np.sqrt(len(by)),(qpred(z,N,D,Q,kind)-y)/np.sqrt(len(y))]
        trials=[]
        for factor in [.6,1,1.5]:
            s=np.clip(np.r_[list(base.values()),np.array(extra)*factor],lo+1e-8,hi-1e-8)
            trials.append(least_squares(r,s,bounds=(lo,hi),loss=loss,max_nfev=1500))
        f=min(trials,key=lambda z:sum(r(z.x)**2));x=f.x
    pp=qpred(x,N,D,Q,kind);rss=sum((pp-y)**2);k=len(x) if mode=='joint' else len(x)-5
    return x,dict(kind=kind,mode=mode,aic=float(len(y)*np.log(rss/len(y)+1e-30)+2*k),bic=float(len(y)*np.log(rss/len(y)+1e-30)+k*np.log(len(y))),**metrics(y,pp))

def quality(d,base,b1):
    b6=d['supplementary_NQ_experiment.csv'];b7=d['supplementary_NQ_experiment_expanded.csv'];b8=d['supplementary_NQ_experiment_large.csv']
    key=['N_params_B','D_tokens_B','Q_score']
    novel7=b7.merge(b6[key].drop_duplicates().assign(_old=1),on=key,how='left').query('_old != 1').drop(columns='_old')
    novel8=b8.merge(b7[key].drop_duplicates().assign(_old=1),on=key,how='left').query('_old != 1').drop(columns='_old')
    save([dict(dataset='B6_fit',n=len(b6)),dict(dataset='B7_new',n=len(novel7)),dict(dataset='B8_new',n=len(novel8))],'quality_split_counts.csv')
    comp=[];candidates={};vals=[]
    for kind in ['exponential','power','additive']:
      for mode in ['two_stage','joint']:
        x,row=fit_quality(b6,base,kind,mode,b1)
        candidates[(kind,mode)]=x;comp.append(row)
        for label,dat in [('B7_new',novel7),('B8_new',novel8)]:
            if len(dat):vals.append(dict(kind=kind,mode=mode,dataset=label,**metrics(dat.val_loss,qpred(x,dat.N_params_B,dat.D_tokens_B,dat.Q_score,kind))))
    comparison=pd.DataFrame(comp);validation=pd.DataFrame(vals)
    save(comparison,'quality_model_comparison.csv');save(validation,'quality_validation_metrics.csv')
    # Selection uses B6 fit + complexity + model semantics; B7/B8 remain validation only.
    stage=comparison[comparison['mode']=='two_stage'].sort_values('bic')
    selected=stage.iloc[0]['kind'];mode='two_stage'
    x=candidates[(selected,mode)]
    names=PKEY+(['gamma'] if selected!='additive' else ['G','kappa'])
    pars=dict(zip(names,map(float,x)))
    save([dict(parameter=k,estimate=v,source='B1 real baseline; B6 semi-synthetic quality',model=selected) for k,v in pars.items()],'quality_scaling_params.csv')
    return selected,mode,pars,comparison,validation,candidates

def problem1_bridge(pars,kind):
    m=joblib.load(P1/'models'/'mixture_v2.joblib')
    # Model metadata has no saved training centroid. Use the reproducible simplex
    # barycentre and explicitly record this reference assumption.
    pref=np.full(len(m['pcols']),1/len(m['pcols']))
    effects=pd.read_csv(P1/'tables'/'domain_effect_bootstrap.csv')
    stable=pd.read_csv(P1/'tables'/'v2_effect_stability_summary.csv')
    mix=effects.merge(stable[['increase','decrease','sign_consistency','median_effect']],on=['increase','decrease'],how='left')
    mix['robust']=((mix.ci_low*mix.ci_high>0)&(mix.sign_stability>=.95)&(mix.sign_consistency>=.9))
    save(mix,'mixture_effect_summary.csv')
    q=pd.read_csv(P1/'tables'/'domain_quality_scores.csv')
    save(q,'problem1_quality_proxy_import.csv')
    v2=pd.read_csv(P1/'tables'/'v2_vs_v1_metrics.csv')
    save(v2[v2.version=='v2'],'problem1_scale_transfer_metrics.csv')
    assumptions=dict(model='separable bridge',reference_p='uniform 17-domain simplex; training centroid not saved in problem1 model',lambda_p=1.0,
                     lambda_sensitivity=[.5,.75,1,1.25,1.5],Q_B='B6 quality mechanism scale',Q_A='problem1 proxy; no empirical anchor to Q_B')
    (T/'bridge_assumptions.json').write_text(json.dumps(assumptions,ensure_ascii=False,indent=2),encoding='utf-8')
    return m,pref,mix

def analyses(d,pars,kind,m,pref,mix):
    marg=[];elast=[];equiv=[]
    for N in [.1,1,7,70,120,300]:
      for D in [20,300,2000]:
       for Q in [.3,.6,.9]:
        L=float(predict_ndq(N,D,Q,pars,kind));dn,dd,dq=[float(z) for z in derivatives(N,D,Q,pars,kind)]
        vals={'N':(N,dn),'D':(D,dd),'Q':(Q,dq)}
        for factor,(v,der) in vals.items():
            marg.append(dict(N_B=N,D_B=D,Q=Q,factor=factor,derivative=der,improvement_rate=-der))
            elast.append(dict(N_B=N,D_B=D,Q=Q,factor=factor,loss_elasticity=v*der/L,reducible_loss_elasticity=v*der/(L-pars['E'])))
        if Q<=.9:
            ne=equivalent_parameters(N,D,Q,.1,pars,kind)
            equiv.append(dict(N_B=N,D_B=D,Q=Q,delta_Q=.1,N_equivalent_B=ne,delta_N_B=ne-N,multiplier=ne/N,
                              local_dN_dQ=-dq/dn,quality_loss_gain=float(L-predict_ndq(N,D,Q+.1,pars,kind))))
    save(marg,'marginal_effects.csv');save(elast,'elasticities.csv');save(equiv,'quality_parameter_equivalence.csv')
    sensitivity=[]
    for lp in [.5,.75,1,1.25,1.5]:
        sensitivity.append(dict(source='lambda_p',setting=lp,top_mixture_effect=float(mix.loc[mix.robust,'loss_change'].min()*lp) if mix.robust.any() else np.nan))
    for form in ['exponential','power','additive']:
        sensitivity.append(dict(source='Q_form',setting=form,top_mixture_effect=np.nan))
    save(sensitivity,'sensitivity_summary.csv')
    large=d['supplementary_large_baseline.csv'].merge(
        d['supplementary_large_models.csv'],left_on=['family','N_params_B','D_tokens_B'],
        right_on=['model_name','N_params_B','D_tokens_B'],how='left',validate='one_to_one')
    assert large.model_name.notna().all(), 'B9/B10 metadata linkage incomplete'
    large['predicted']=predict_ndq(large.N_params_B,large.D_tokens_B,1,pars,kind)
    large['residual']=large.predicted-large.val_loss
    large['scale_band']=pd.cut(large.N_params_B,[99,300,1000,np.inf],labels=['100-300B','300B-1T','>1T'])
    save(large,'large_model_extrapolation.csv')
    save([dict(scale_band=str(k),**metrics(g.val_loss,g.predicted)) for k,g in large.groupby('scale_band',observed=True)],'large_model_metrics_by_band.csv')
    return pd.DataFrame(equiv),pd.DataFrame(elast),large

def bootstrap(b1,b6,base,kind,rep=200):
    rng=np.random.default_rng(SEED);groups=sorted(b1.N_params_B.unique());rows=[];eq=[]
    for i in range(rep):
        draw=rng.choice(groups,len(groups),replace=True)
        sample=pd.concat([b1[b1.N_params_B==g] for g in draw],ignore_index=True)
        try:
            f,_=fit_classic(sample,starts=[np.array(list(base.values()))]);bp=getp(f.x)
            x,_=fit_quality(b6,bp,kind,'two_stage')
            keys=PKEY+(['gamma'] if kind!='additive' else ['G','kappa'])
            p=dict(zip(keys,map(float,x)))
            rows.append(dict(replicate=i,**p))
            eq.append(dict(replicate=i,N_equivalent_B=equivalent_parameters(7,300,.6,.1,p,kind)))
        except (ValueError,FloatingPointError):continue
    boot=pd.DataFrame(rows);save(boot,'bootstrap_parameters.csv');save(eq,'bootstrap_equivalence.csv')
    intervals=[dict(parameter=k,ci_low=float(boot[k].quantile(.025)),ci_high=float(boot[k].quantile(.975)),median=float(boot[k].median())) for k in boot if k!='replicate']
    save(intervals,'parameter_intervals.csv')
    return boot,pd.DataFrame(intervals)

def plots(d,b1,external,kind,pars,equiv,elast,large,mix):
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':110})
    def finish(name):plt.tight_layout();plt.savefig(F/name,dpi=240,bbox_inches='tight');plt.close()
    plt.figure(figsize=(5,4));plt.scatter(b1.val_loss,b1.prediction,s=5,alpha=.4);a=[b1.val_loss.min(),b1.val_loss.max()];plt.plot(a,a,'k--');plt.xlabel('B1 observed validation loss');plt.ylabel('Predicted loss');finish('classical_scaling_fit.png')
    plt.figure(figsize=(7,4));
    for n,g in b1.groupby('N_params_B'):
        z=g.sort_values('D_tokens_B');plt.plot(z.D_tokens_B,z.val_loss,alpha=.5,label=f'{n:g}B');plt.plot(z.D_tokens_B,z.prediction,'--',alpha=.7)
    plt.xscale('log');plt.xlabel('D, billion tokens');plt.ylabel('Validation loss');plt.legend(ncol=4,fontsize=7);finish('scaling_trajectory_fit.png')
    fig,ax=plt.subplots(1,3,figsize=(10,3.2))
    for a,lab in zip(ax,['B2','B4','B5']):
        g=external[external.dataset==lab]
        a.scatter(g.val_loss,g.predicted,s=7,alpha=.5);lims=[min(g.val_loss.min(),g.predicted.min()),max(g.val_loss.max(),g.predicted.max())];a.plot(lims,lims,'k--');a.set_title(lab);a.set_xlabel('Reference loss');a.set_ylabel('Raw prediction')
    finish('external_validation.png')
    D=np.geomspace(5,2000,70);Q=np.linspace(.1,1,60);dd,qq=np.meshgrid(D,Q)
    fig,ax=plt.subplots(1,3,figsize=(11,3.3))
    for a,n in zip(ax,[.1,7,120]):
        z=predict_ndq(n,dd,qq,pars,kind);im=a.contourf(dd,qq,z,levels=22);a.set_xscale('log');a.set_title(f'N={n:g}B');a.set_xlabel('D, billion tokens');a.set_ylabel('Q (B scale)');fig.colorbar(im,ax=a)
    finish('quality_effect_surface.png')
    plt.figure(figsize=(6,4))
    for q in [.2,.5,.8,1]:plt.plot(D,predict_ndq(7,D,q,pars,kind),label=f'Q={q:g}')
    plt.xscale('log');plt.xlabel('D, billion tokens');plt.ylabel('Predicted loss, N=7B');plt.legend();finish('quality_scaling_curves.png')
    e=elast[(elast.D_B==300)&(elast.Q==.6)];fig,ax=plt.subplots(figsize=(6,4))
    for factor,g in e.groupby('factor'):ax.plot(g.N_B,abs(g.reducible_loss_elasticity),marker='o',label=factor)
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('N, billion parameters');ax.set_ylabel('Magnitude of reducible-loss elasticity');ax.legend();finish('elasticity_N_D_Q.png')
    fig,ax=plt.subplots(figsize=(6,4))
    for q,g in equiv[equiv.D_B==300].groupby('Q'):ax.plot(g.N_B,g.multiplier,marker='o',label=f'Q={q:g} to {q+.1:g}')
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('Initial N, billion');ax.set_ylabel('Equivalent parameter multiplier');ax.legend();finish('quality_parameter_equivalence.png')
    fig,ax=plt.subplots(figsize=(6,4));ax.scatter(large.N_params_B,large.val_loss,s=10,label='B10 estimated');ax.scatter(large.N_params_B,large.predicted,s=10,label='Model');ax.set_xscale('log');ax.set_xlabel('N, billion parameters');ax.set_ylabel('Loss');ax.legend();finish('large_scale_extrapolation.png')
    z=mix.sort_values('loss_change').head(12);fig,ax=plt.subplots(figsize=(7,4));lab=z['decrease']+'→'+z['increase'];ax.barh(lab,z.loss_change,color=np.where(z.robust,'#2a788e','#aaa'));ax.invert_yaxis();ax.set_xlabel('Mean loss change per 1 percentage point');finish('mixture_effects.png')
    N=np.geomspace(.07,700,100);fig,ax=plt.subplots(figsize=(6,4));ax.plot(N,predict_ndq(N,300,.6,pars,kind));ax.set_xscale('log');ax.set_xlabel('N, billion parameters');ax.set_ylabel('Predicted loss (D=300B, Q=0.6)');finish('marginal_N_curve.png')
    fig,ax=plt.subplots(figsize=(6,4));ax.plot(D,-derivatives(7,D,.6,pars,kind)[1]);ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('D, billion tokens');ax.set_ylabel('Loss improvement per billion tokens');finish('marginal_D_curve.png')
    fig,ax=plt.subplots(figsize=(6,4));q=np.linspace(.1,1,100);ax.plot(q,-derivatives(7,300,q,pars,kind)[2]);ax.set_xlabel('Q (B scale)');ax.set_ylabel('Loss improvement per Q unit');finish('marginal_Q_curve.png')

def report(base,cv,ex,kind,mode,pars,comparison,validation,equiv,elast,large,mix,intervals):
    cv1=cv[cv.model=='M1_ND'];cv0=cv[cv.model=='M0_mean'];pooled=lambda z:float(np.sqrt(np.average(z.rmse**2,weights=z.n)))
    e7=equiv[(equiv.N_B==7)&(equiv.D_B==300)&(equiv.Q==.6)].iloc[0]
    ext=lambda k:ex[(ex.dataset==k)&(ex.comparison.isin(['raw','interpolated_trajectory']))].iloc[0]
    ptext=', '.join(f'{k}={v:.6g}' for k,v in pars.items())
    qrow=comparison[(comparison.kind==kind)&(comparison['mode']==mode)].iloc[0]
    valtxt='\n'.join(f"- {r.dataset}: RMSE {r.rmse:.4f}, MAE {r.mae:.4f}, bias {r.bias:.4f} ({r.kind}, {r['mode']})." for _,r in validation[(validation.kind==kind)&(validation['mode']==mode)].iterrows())
    v8=validation[(validation.kind==kind)&(validation['mode']==mode)&(validation.dataset=='B8_new')].iloc[0]
    robust=mix[mix.robust].sort_values('loss_change')
    mixtext='; '.join(f"{r.decrease}→{r.increase}: {r.loss_change:+.4f}" for _,r in robust.head(5).iterrows()) or '无足够稳健的方向性结果'
    r7=elast[(elast.N_B==7)&(elast.D_B==300)&(elast.Q==.6)]
    elasttext=', '.join(f"{r.factor}: {r.loss_elasticity:.4f} (可约损失 {r.reducible_loss_elasticity:.4f})" for _,r in r7.iterrows())
    large_m=metrics(large.val_loss,large.predicted)
    formula={'exponential':r'L=E+A N^{-\alpha}+B[D\exp\{\gamma(Q-1)\}]^{-\beta}+\lambda_p\Delta_p(p)',
             'power':r'L=E+A N^{-\alpha}+B(DQ^\gamma)^{-\beta}+\lambda_p\Delta_p(p)',
             'additive':r'L=E+A N^{-\alpha}+B D^{-\beta}+G(1-Q)^\kappa+\lambda_p\Delta_p(p)'}[kind]
    report=f'''# 问题二：跨维度数据融合与广义标度律

## 1 问题分析

经典标度律刻画参数规模 N 和训练 token 数 D 对验证损失的关系。本文用 B1 的 1,176 个真实 Pythia 轨迹点估计基准律，再用半合成 B6 的质量实验估计 Q 机制，最后读取问题一第二版的领域配比模型，以中心化损失效应连接 p。N 与 D 均以 **billion (10^9)** 为单位。融合依赖可分离性假设，并非 B 数据直接识别出 17 域配比系数。

## 2 数据与预处理

B1 是真实训练轨迹；B2 是半合成族外轨迹；B3 是 Pythia 检查点插值，不能视为独立实验。B4 是跨族模型收敛点，B5 是公开文献尺度数据。B6–B8 是基于真实规律校准的半合成 N-D-Q 数据，B6 为拟合集，B7 去掉 B6 重叠点、B8 去掉 B7 重叠点后只做检验。B9 是大模型元数据，B10 的 Loss 是估算值。详见数据审计表和重叠表。B4/B5 的验证损失评价语料和口径未证实完全相同，故其 raw 误差只能作分布迁移诊断；另列描述性截距校准，绝不称为独立验证。

## 3 经典 N-D 标度律

$L=E+A N^{{-\alpha}}+B D^{{-\beta}}$。有界非线性最小二乘、多初值、正参数约束。最终：E={base['E']:.6g}, A={base['A']:.6g}, α={base['alpha']:.6g}, B={base['B']:.6g}, β={base['beta']:.6g}。B1 按 8 个模型参数规模 leave-one-model-out，M1 汇总 RMSE={pooled(cv1):.4f}，均值 M0={pooled(cv0):.4f}。参数多初值偏差和 Jacobian 条件数见参数表。cluster bootstrap 区间见 parameter_intervals.csv。

## 4 外部与轨迹验证

{chr(10).join(f'- {k}: raw RMSE={ext(k).rmse:.4f}, MAE={ext(k).mae:.4f}, bias={ext(k).bias:.4f}, Spearman={ext(k).spearman:.4f}。' for k in ['B2','B3','B4','B5'])}

B2 的逐模型误差见 b2_by_model_metrics.csv。B2 raw RMSE 显著高于 B1 留模型 CV，说明族外损失口径或生成机制有强偏移，不能称模型已在族外准确泛化。B3 主要检验插值轨迹形状。B4/B5 未参与任何拟合，其 raw 偏差不能用事后校准掩盖。

## 5 数据质量广义标度律

比较指数有效数据量、幂函数有效数据量、加性惩罚三种模型；分别实行固定 B1 基准参数的两阶段估计和按来源均衡加权的联合估计。选型只依赖 B6 的 BIC、误差、参数数目和单调性，B7/B8 只检验，避免测试集反向选型。选中 {kind} / {mode}，B6 RMSE={qrow.rmse:.4f}，BIC={qrow.bic:.2f}。参数：{ptext}。其余形式及联合拟合的差异完整保存在模型比较表。新增质量系数非负，Q=1 退化为经典律。

{valtxt}

B8 新增点的 RMSE={v8.rmse:.4f}、bias={v8.bias:.4f}，显示大尺度半合成机制与 B6/B7 显著不一致；保留这一失败结果，不将 B8 称为通过验证。

## 6 领域配比融合与 Q 跨源可比性

问题一当前模型是第二版二阶 ALR Ridge，读取 mixture_v2.joblib，不重估附件 A。定义 Δ_p(p)=13 域预测验证损失均值 f_p(p) 减去参考配比 f_p(p0)。模型文件未保存 A4 配比均值，故明确选 p0 为 17 域均匀配比；在局部效应和差分中参考常数会抵消。λ_p=1 为损失单位可迁移的基准假设；0.5、0.75、1、1.25、1.5 做敏感性。问题一在 1M 绝对损失预测较好，在 60M/1B 存在整体尺度偏移，故仅迁移中心化配比效应，不能将绝对预测直接搬到大尺度。问题一只有 6 个 direct/near_direct 域可可靠映射；mapped_Q 曾将 1M RMSE 从 0.3415 略降至 0.3374，但 1B 略退步，不能独立识别 Q 的真实训练效应。Q_A 是 22 指标构造的质量代理，Q_B 是 B6 半合成机制变量；二者没有同配比不同质量的真实锚点，不宣称数值同尺度。本文计算的 Q+0.1 是 Q_B 情景结论，应用到 Q_A 必须另行校准。

## 7 最终 N-D-Q-p 广义标度律

$$
{formula}
$$

代入参数：{ptext}；λ_p=1 为基准，非 B 数据可识别参数。N,D 以 billion 为单位，Q_B 在 (0,1]；应用于问题一的 Q_A 只能作情景映射。p 在 17 维概率单纯形。

## 8 边际效用和弹性

所有边际效用是对应变量增加时的 -∂L/∂x。所选加性模型给出 ∂L/∂N=-α A N^(-α-1)，∂L/∂D=-β B D^(-β-1)，∂L/∂Q=-Gκ(1-Q)^(κ-1)（Q<1）。代码 derivatives() 与中心差分数值核验一致。弹性定义 ε_x=x∂L/(L∂x)，可约损失弹性将分母替换为 L-E。N=7B、D=300B、Q=0.6 时：{elasttext}。完整多场景表见表。p 效应严格采用保持总和为 1 的 1 个百分点 j→i 转移。

## 9 质量与规模替代

固定 D,p，保持损失不变有 dN/dQ=-(∂L/∂Q)/(∂L/∂N)=-Gκ(1-Q)^(κ-1)N^(α+1)/(α A)，为负值；质量提高允许减少 N。性能提升等价定义为 L(N_eq,D,Q)=L(N,D,Q+0.1)。对加性模型可解析求得 N_eq=[A/(A N^(-α)+G((1-Q-0.1)^κ-(1-Q)^κ))]^(1/α)；若括号分母非正，有限参数增长不可达到该质量收益。代码以目标损失精确反解并核验。7B/300B/Q=0.6 时，Q+0.1 的 Loss 改善 {e7.quality_loss_gain:.5f}，等价 N={e7.N_equivalent_B:.3g}B，增加 {e7.delta_N_B:.3g}B，倍数 {e7.multiplier:.3g}。其余 N=0.1、1、70、120、300B 及多 D/Q 情景见等价表；不可达时记为无穷大。此加性模型中固定 N,Q 的质量损失改善与 D 无关，等价参数倍数也不随 D 变；这是一项结构性假设。

## 10 领域替代与互补

结合问题一 bootstrap 95% 区间、符号稳定性和第二版 9 个规格的方向一致性，较稳健的局部替代为：{mixtext}。这些仅是局部、配比保持单纯形的预测差分。interaction_effects.csv 中的交互项没有同时满足 bootstrap、多规格和外部多重检验支持；因此尚不足确认稳定互补对。

## 11 大模型外推

B9/B10 按模型名、N、D 一对一关联，用于 100B 以上诊断。B10 是估算标签而非真实实验。最终 Q 模型在 Q=1、p=p0 的 raw 对比 RMSE={large_m['rmse']:.4f}，MAE={large_m['mae']:.4f}，bias={large_m['bias']:.4f}，Spearman={large_m['spearman']:.4f}，平均相对误差={large_m['relative_error']:.2%}。这种近乎精确的符合很可能反映估算表与经典标度律共享生成规律，不能视为独立验证。分规模指标见 large_model_metrics_by_band.csv。B9/B10 若缺质量和领域配比实测，只能使用参考情景，不能将差异都归于模型误差。

## 12 稳健性与不确定性

以 Pythia 模型为簇实施 200 次 bootstrap，每次重估经典参数及 B6 质量参数；区间在 parameter_intervals.csv，等价参数区间在 bootstrap_equivalence.csv。Jacobian 参数相关性见 classical_parameter_correlations.csv。三种 Q 形式、两阶段与联合拟合、B7/B8 非重合验证、B1 robust/equal-model 权重及 λ_p 情景均保存。B8 中存在 0.5 下界与高规模结构变化，验证误差显著升高，应视为半合成生成机制迁移，不能删去坏结果。bootstrap 区间只刻画同一数据生成机制下的抽样不确定性，不包含 B2/B8 迁移偏差。

## 13 适用范围和局限

真实证据支持 B1 上的 N-D 关系；Q 机制只由半合成数据支持；p 项由问题一 A4/A5 学得并在 A6–A11 有跨尺度排序证据，但跨规模幅度 λ_p 不可识别。没有真实同配比不同质量实验，所以 Q 与 p 的独立因果作用不能由附件 A/B 联合识别。B4/B5 评价口径差异和 B10 估算性质限制了外推验证。所有结果是预测关系，不能称为因果效应。
'''
    (Path(__file__).parent/'problem2_report.md').write_text(report,encoding='utf-8')
    summary=f'''# 问题二摘要

- 经典律：E={base['E']:.6g}, A={base['A']:.6g}, α={base['alpha']:.6g}, B={base['B']:.6g}, β={base['beta']:.6g}（N、D 为 billion）。B1 模型留一 CV RMSE={pooled(cv1):.4f}。
- 广义律：{formula}。质量模型 {kind}，{ptext}；λ_p=1 是敏感性基准。
- 外部 raw RMSE：B2 {ext('B2').rmse:.4f}，B3 插值 {ext('B3').rmse:.4f}，B4 {ext('B4').rmse:.4f}，B5 {ext('B5').rmse:.4f}。
- N=7B/D=300B/Q_B=0.6 的损失弹性：{elasttext}。
- 同情景 Q_B+0.1 等效参数 {e7.N_equivalent_B:.3g}B，倍数 {e7.multiplier:.3g}；Q_A 不可直接等同 Q_B。
- p：复用问题一第二版 17 域模型和中心化效应；λ_p 不可由 B 识别。较稳健替代：{mixtext}。尚无充分的互补证据。
- 100B+：对 B10 **估算标签** 的 RMSE {large_m['rmse']:.4f}，非真实外部验证。
- 稳健性：200 次模型簇 bootstrap；质量形式和拟合方式对照见表。最重要限制是 Q 的半合成机制、跨源标尺无锚点、B4/B5 损失口径和 λ_p 未识别。
- 问题三读取 `outputs/tables/problem2_to_problem3.json`、`outputs/models/final_generalized_scaling.joblib` 和 `generalized_scaling.py` 中的 `predict_generalized`、`equivalent_parameters`。
'''
    (Path(__file__).parent/'problem2_summary.md').write_text(summary,encoding='utf-8')
    return dict(cv_rmse=pooled(cv1),b2_rmse=float(ext('B2').rmse),b4_rmse=float(ext('B4').rmse),b5_rmse=float(ext('B5').rmse),equiv_7b=float(e7.N_equivalent_B),large_rmse=large_m['rmse'])

def main():
    for path in [T,F,M]:path.mkdir(parents=True,exist_ok=True)
    warnings.filterwarnings('ignore',category=RuntimeWarning)
    d=audit();base,b1,cv=classical(d);ex,external_predictions=external(d,base,b1)
    kind,mode,pars,comparison,validation,candidates=quality(d,base,b1)
    m,pref,mix=problem1_bridge(pars,kind)
    equiv,elast,large=analyses(d,pars,kind,m,pref,mix)
    robust,_=fit_classic(b1,starts=[np.array(list(base.values()))],loss='soft_l1')
    balance=1/np.sqrt(b1.groupby('N_params_B').N_params_B.transform('size').to_numpy(float))
    balanced,_=fit_classic(b1,starts=[np.array(list(base.values()))],weights=balance)
    sensitiv=pd.read_csv(T/'sensitivity_summary.csv')
    sensitiv=pd.concat([sensitiv,pd.DataFrame([dict(source='B1_fit',setting='robust_soft_l1',max_relative_parameter_change=float(np.max(abs(robust.x-np.array(list(base.values())))/np.array(list(base.values()))))),dict(source='B1_fit',setting='equal_model_weight',max_relative_parameter_change=float(np.max(abs(balanced.x-np.array(list(base.values())))/np.array(list(base.values())))))])],ignore_index=True)
    save(sensitiv,'sensitivity_summary.csv')
    boot,intervals=bootstrap(b1,d['supplementary_NQ_experiment.csv'],base,kind)
    plots(d,b1,external_predictions,kind,pars,equiv,elast,large,mix)
    save([dict(parameter=k,estimate=v,ci_low=float(intervals.set_index('parameter').loc[k,'ci_low']),ci_high=float(intervals.set_index('parameter').loc[k,'ci_high']),origin='B1' if k in PKEY else 'B6 semi-synthetic') for k,v in pars.items()]+[dict(parameter='lambda_p',estimate=1,ci_low=np.nan,ci_high=np.nan,origin='assumption; sensitivity 0.5-1.5')],'generalized_scaling_params.csv')
    joblib.dump(dict(kind=kind,parameters=pars,mixture_model=m,reference_p=pref,lambda_p=1),M/'final_generalized_scaling.joblib')
    interface=dict(model_type=kind,parameters=pars,parameter_intervals=intervals.to_dict('records'),p_correction='lambda_p * [mean_13_domain_loss_v2(p)-mean_13_domain_loss_v2(uniform_p)]',lambda_p=1,
                   Q_range=[.1,1],Q_scale='B6 semi-synthetic; not directly equal to problem1 Q proxy',N_B_fit_range=[float(b1.N_params_B.min()),float(b1.N_params_B.max())],D_B_fit_range=[float(b1.D_tokens_B.min()),float(b1.D_tokens_B.max())],
                   recommended_extrapolation='100B+ diagnostic only; compare B10 estimated losses, not observations',prediction_function='problem2/generalized_scaling.py:predict_generalized',equivalence_function='problem2/generalized_scaling.py:equivalent_parameters')
    (T/'problem2_to_problem3.json').write_text(json.dumps(interface,ensure_ascii=False,indent=2),encoding='utf-8')
    numbers=report(base,cv,ex,kind,mode,pars,comparison,validation,equiv,elast,large,mix,intervals)
    (T/'report_numbers.json').write_text(json.dumps(numbers,indent=2),encoding='utf-8')
    import verify_problem2
    verify_problem2.verify()
    print(json.dumps(dict(selected_quality_model=kind,parameters=pars,report_numbers=numbers,bootstrap_replicates=len(boot)),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
