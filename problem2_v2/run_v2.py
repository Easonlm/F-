"""Research round two: diagnostics, restricted model search and honest transfer tests."""
from __future__ import annotations
import json,sys,warnings,hashlib
from pathlib import Path
import numpy as np,pandas as pd,joblib
from scipy.stats import spearmanr,linregress
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models import fit_nd,predict_nd,fit_q,predict_q,equivalent_N,predict_recommended

ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
B=ROOT/'real_attachments'/'B_scaling_laws';V1=ROOT/'problem2'/'outputs'
T=HERE/'outputs'/'tables';F=HERE/'outputs'/'figures';M=HERE/'outputs'/'models';DIAG=HERE/'diagnostics'
KINDS=['classic','interaction','compute_ratio','broken_N'];QKINDS=['additive','effective_power','effective_exp','saturating','quality_data','quality_model']
SEED=20260923

def save(x,name,where=T):pd.DataFrame(x).to_csv(where/name,index=False,float_format='%.10g')
def metric(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float);good=np.isfinite(y)&np.isfinite(p);y=y[good];p=p[good]
    return dict(n=len(y),rmse=float(np.sqrt(np.mean((y-p)**2))),mae=float(np.mean(abs(y-p))),bias=float(np.mean(p-y)),
      relative_error=float(np.mean(abs(y-p)/np.maximum(abs(y),1e-12))),spearman=float(spearmanr(y,p).statistic) if len(y)>2 and np.std(y)>0 and np.std(p)>0 else np.nan)
def read():
    files={'B1':'pythia_training_log_existing.csv','B2':'cerebras_training_log.csv','B4':'scaling_baseline.csv','B5':'published_scaling_data.csv',
           'B6':'supplementary_NQ_experiment.csv','B7':'supplementary_NQ_experiment_expanded.csv','B8':'supplementary_NQ_experiment_large.csv'}
    return {k:pd.read_csv(B/f) for k,f in files.items()}
def novel(a,b):
    key=['N_params_B','D_tokens_B','Q_score']
    return b.merge(a[key].drop_duplicates().assign(_old=1),on=key,how='left').query('_old != 1').drop(columns='_old')

def diagnostics(d,base,qbase):
    b1=d['B1'];n=b1.N_params_B.to_numpy(float);D=b1.D_tokens_B.to_numpy(float);y=b1.val_loss.to_numpy(float)
    pred=predict_nd('classic',base,n,D);res=y-pred
    scale=pd.DataFrame([dict(source='B1',**metric(y,pred),residual_sd=float(np.std(res)),logN_logD_corr=float(np.corrcoef(np.log(n),np.log(D))[0,1]),n_groups=b1.N_params_B.nunique())])
    for src in ['B2','B4','B5']:
        x=d[src];pr=predict_nd('classic',base,x.N_params_B,x.D_tokens_B);scale=pd.concat([scale,pd.DataFrame([dict(source=src,**metric(x.val_loss,pr),residual_sd=float(np.std(x.val_loss-pr)))])],ignore_index=True)
    save(scale,'source_shift.csv',DIAG)
    # Rounded B1 values versus residuals: tests near-deterministic relation.
    precision=pd.DataFrame([dict(rounding_digits=int(k),rows=int(v)) for k,v in b1.val_loss.astype(str).str.split('.').str[1].str.len().value_counts().items()]);save(precision,'b1_loss_decimal_precision.csv',DIAG)
    # Profile E by refitting the remaining parameters on a grid; profile cannot use external sets.
    from scipy.optimize import least_squares
    e_grid=np.unique(np.r_[np.linspace(base[0]-.02,base[0]+.02,81),np.linspace(base[0]-.002,base[0]+.002,201)]);prof=[]
    for E in e_grid:
        f=least_squares(lambda z:E+z[0]*n**(-z[1])+z[2]*D**(-z[3])-y,base[1:],bounds=([1e-6,.005,1e-6,.005],[100,2,100,2]),max_nfev=350)
        prof.append(dict(E=E,rmse=float(np.sqrt(np.mean(f.fun**2))),A=f.x[0],alpha=f.x[1],B=f.x[2],beta=f.x[3]))
    save(prof,'profile_E.csv',DIAG)
    profdf=pd.DataFrame(prof);rmse_min=float(profdf.rmse.min())
    within=profdf[len(b1)*np.log((profdf.rmse/rmse_min)**2)<=3.84146]
    save([dict(method='within-B1 Gaussian profile LR, ignores source shift and trajectory correlation',
               E_low=float(within.E.min()),E_high=float(within.E.max()),rmse_min=rmse_min)],'profile_E_interval.csv',DIAG)
    b8=novel(d['B7'],d['B8']);b7=novel(d['B6'],d['B7'])
    def monotonic(src,df):
        rows=[]
        for (N,Dt),g in df.groupby(['N_params_B','D_tokens_B']):
            if g.Q_score.nunique()>=3:
                rho=float(spearmanr(g.Q_score,g.val_loss).statistic)
                rows.append(dict(source=src,N_B=N,D_B=Dt,n_Q=g.Q_score.nunique(),rho_Q_L=rho,quality_improves_loss=rho<0))
        return rows
    # B7's 90 novel rows are isolated new Q levels per (N,D); use the full B7
    # grid only for direction diagnostics, never as an independent fit score.
    mono=pd.DataFrame(monotonic('B6',d['B6'])+monotonic('B7_full',d['B7'])+monotonic('B8_new',b8));save(mono,'quality_monotonicity.csv',DIAG)
    # Same N,D,Q but discrepant outcomes: direct evidence of generation shift.
    key=['N_params_B','D_tokens_B','Q_score'];over=d['B6'].merge(d['B8'],on=key,suffixes=('_B6','_B8'))
    over['loss_difference_B8_minus_B6']=over.val_loss_B8-over.val_loss_B6
    save(over,'b6_b8_exact_overlap_conflicts.csv',DIAG)
    # B8 residual against variables requested by the brief.
    z=b8.copy();z['prediction_v1']=predict_q('additive',qbase,z.N_params_B,z.D_tokens_B,z.Q_score,base)
    z['residual_observed_minus_predicted']=z.val_loss-z.prediction_v1
    z['ND']=z.N_params_B*z.D_tokens_B;z['D_over_N']=z.D_tokens_B/z.N_params_B
    save(z,'b8_new_residuals.csv',DIAG)
    regress=[]
    for col in ['N_params_B','D_tokens_B','Q_score','ND','D_over_N']:
        xx=np.log(z[col]) if col!='Q_score' else z[col]
        lr=linregress(xx,z.residual_observed_minus_predicted)
        regress.append(dict(feature=col,transform='log' if col!='Q_score' else 'raw',slope=lr.slope,r=lr.rvalue,pvalue=lr.pvalue))
    save(regress,'b8_residual_trends.csv',DIAG)
    source_trends=[]
    for src in ['B2','B4','B5']:
        x=d[src].copy();x['raw_pred']=predict_nd('classic',base,x.N_params_B,x.D_tokens_B);x['residual_observed_minus_predicted']=x.val_loss-x.raw_pred
        save(x,f'{src.lower()}_raw_residuals.csv',DIAG)
        for col in ['N_params_B','D_tokens_B']:
            lr=linregress(np.log(x[col]),x.residual_observed_minus_predicted)
            source_trends.append(dict(source=src,feature='log_'+col,slope=lr.slope,r=lr.rvalue,pvalue=lr.pvalue))
    save(source_trends,'source_residual_trends.csv',DIAG)
    # Figures
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(1,5,figsize=(14,3))
    for a,col in zip(ax,['N_params_B','D_tokens_B','Q_score','ND','D_over_N']):
        a.scatter(z[col],z.residual_observed_minus_predicted,s=5,alpha=.4)
        if col!='Q_score':a.set_xscale('log')
        a.set_xlabel(col);a.set_ylabel('B8 observed - V1')
    fig.tight_layout();fig.savefig(F/'b8_residual_structure.png',dpi=240);plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,3.5));ax.plot(e_grid,pd.DataFrame(prof).rmse);ax.set_xlabel('Fixed E');ax.set_ylabel('B1 profile RMSE');fig.tight_layout();fig.savefig(F/'profile_E.png',dpi=240);plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,3.5));
    for src,g in mono.groupby('source'):ax.hist(g.rho_Q_L,bins=np.linspace(-1,1,21),alpha=.5,label=src)
    ax.set_xlabel('Within-(N,D) Spearman(Q, Loss)');ax.set_ylabel('Count');ax.legend();fig.tight_layout();fig.savefig(F/'quality_direction_by_source.png',dpi=240);plt.close(fig)
    return b7,b8,scale,mono,over,z

def nd_experiments(d,base):
    b1=d['B1'];fits={};fitrows=[];cv=[];extrap=[];external=[]
    for kind in KINDS:
        fit=fit_nd(kind,b1,start=base if kind=='classic' else None)
        fits[kind]=fit.x
        p=predict_nd(kind,fit.x,b1.N_params_B,b1.D_tokens_B);rss=np.sum((p-b1.val_loss)**2);k=len(fit.x)
        fitrows.append(dict(model=kind,n_parameters=k,train_rmse=metric(b1.val_loss,p)['rmse'],bic=float(len(b1)*np.log(rss/len(b1))+k*np.log(len(b1))),jacobian_condition=float(np.linalg.cond(fit.jac.T@fit.jac)),parameters=json.dumps(fit.x.tolist())))
        for held in sorted(b1.N_params_B.unique()):
            tr=b1[b1.N_params_B!=held];te=b1[b1.N_params_B==held]
            ff=fit_nd(kind,tr,start=fit.x)
            cv.append(dict(model=kind,held_N_B=held,**metric(te.val_loss,predict_nd(kind,ff.x,te.N_params_B,te.D_tokens_B))))
        groups=sorted(b1.N_params_B.unique())
        for frac in [.5,.6,.7,.8]:
            ntrain=max(3,int(np.floor(len(groups)*frac)))
            tr=b1[b1.N_params_B.isin(groups[:ntrain])];te=b1[~b1.N_params_B.isin(groups[:ntrain])]
            ff=fit_nd(kind,tr,start=fit.x)
            extrap.append(dict(model=kind,axis='N_small_to_large',train_fraction=frac,max_train_N_B=max(groups[:ntrain]),**metric(te.val_loss,predict_nd(kind,ff.x,te.N_params_B,te.D_tokens_B))))
            parts=[]
            for _,g in b1.groupby('N_params_B'):
                cut=g.D_tokens_B.quantile(frac)
                parts.append(g.D_tokens_B<=cut)
            mask=pd.concat(parts).sort_index();tr=b1[mask];te=b1[~mask]
            ff=fit_nd(kind,tr,start=fit.x)
            extrap.append(dict(model=kind,axis='D_early_to_late',train_fraction=frac,max_train_N_B=np.nan,**metric(te.val_loss,predict_nd(kind,ff.x,te.N_params_B,te.D_tokens_B))))
        for src in ['B2','B4','B5']:
            x=d[src];external.append(dict(model=kind,source=src,comparison='raw',**metric(x.val_loss,predict_nd(kind,fit.x,x.N_params_B,x.D_tokens_B))))
    # Robust targets only for the simple law. Strictly B1-trained.
    for obj in ['huber','soft_l1','relative','log']:
        kind='classic_'+obj;f=fit_nd('classic',b1,start=base,objective=obj);fits[kind]=f.x
        p=predict_nd('classic',f.x,b1.N_params_B,b1.D_tokens_B);rss=sum((p-b1.val_loss)**2)
        fitrows.append(dict(model=kind,n_parameters=5,train_rmse=metric(b1.val_loss,p)['rmse'],bic=float(len(b1)*np.log(rss/len(b1))+5*np.log(len(b1))),jacobian_condition=float(np.linalg.cond(f.jac.T@f.jac)),parameters=json.dumps(f.x.tolist())))
        for held in sorted(b1.N_params_B.unique()):
            tr=b1[b1.N_params_B!=held];te=b1[b1.N_params_B==held];ff=fit_nd('classic',tr,start=f.x,objective=obj)
            cv.append(dict(model=kind,held_N_B=held,**metric(te.val_loss,predict_nd('classic',ff.x,te.N_params_B,te.D_tokens_B))))
        groups=sorted(b1.N_params_B.unique())
        for frac in [.5,.6,.7,.8]:
            ntrain=max(3,int(np.floor(len(groups)*frac)))
            tr=b1[b1.N_params_B.isin(groups[:ntrain])];te=b1[~b1.N_params_B.isin(groups[:ntrain])]
            ff=fit_nd('classic',tr,start=f.x,objective=obj)
            extrap.append(dict(model=kind,axis='N_small_to_large',train_fraction=frac,max_train_N_B=max(groups[:ntrain]),**metric(te.val_loss,predict_nd('classic',ff.x,te.N_params_B,te.D_tokens_B))))
            parts=[]
            for _,g in b1.groupby('N_params_B'):
                cut=g.D_tokens_B.quantile(frac);parts.append(g.D_tokens_B<=cut)
            mask=pd.concat(parts).sort_index();tr=b1[mask];te=b1[~mask]
            ff=fit_nd('classic',tr,start=f.x,objective=obj)
            extrap.append(dict(model=kind,axis='D_early_to_late',train_fraction=frac,max_train_N_B=np.nan,**metric(te.val_loss,predict_nd('classic',ff.x,te.N_params_B,te.D_tokens_B))))
        for src in ['B2','B4','B5']:
            x=d[src];external.append(dict(model=kind,source=src,comparison='raw',**metric(x.val_loss,predict_nd('classic',f.x,x.N_params_B,x.D_tokens_B))))
    save(fitrows,'nd_fit_comparison.csv');save(cv,'nd_group_cv.csv');save(extrap,'nd_scale_extrapolation.csv');save(external,'nd_external_raw.csv')
    # Breakpoint profile fitted only on B1; comparison to baseline uses BIC and extrapolation.
    breakrows=[]
    for Nc in [.16,.41,1.04,2.78,6.86]:
        f=fit_nd('broken_N',b1,fixed_Nc=Nc)
        pred=predict_nd('broken_N',f.x,b1.N_params_B,b1.D_tokens_B)
        breakrows.append(dict(Nc=Nc,delta=f.x[5],rmse=metric(b1.val_loss,pred)['rmse']))
    save(breakrows,'broken_threshold_profile.csv')
    # Source calibration is a separate evaluation protocol, never a raw score.
    calib=[];selected_calibration=[]
    for kind in ['classic','interaction','compute_ratio','broken_N']:
      for src in ['B2','B4','B5']:
        x=d[src].copy();x['prediction']=predict_nd(kind,fits[kind],x.N_params_B,x.D_tokens_B)
        group='run_id' if src=='B2' else 'family'
        flags=[]
        for _,g in x.groupby(group,sort=False):
            g=g.sort_values('D_tokens_B' if src=='B2' else 'N_params_B')
            k=max(1,int(np.floor(.2*len(g))));flags.extend(g.index[:k].tolist())
        train=x.loc[sorted(set(flags))];test=x.drop(index=train.index)
        offset=float((train.val_loss-train.prediction).mean())
        if train.prediction.std()>1e-8:
            slope,intercept=np.polyfit(train.prediction,train.val_loss,1)
        else:slope,intercept=1,offset
        d_slope,d_intercept=np.polyfit(np.log(train.D_tokens_B),train.val_loss-train.prediction,1)
        predictions={'raw':test.prediction,'intercept_20pct':test.prediction+offset,
                     'affine_20pct':intercept+slope*test.prediction,
                     'logD_20pct':test.prediction+d_intercept+d_slope*np.log(test.D_tokens_B)}
        for name,pred in predictions.items():
            calib.append(dict(model=kind,source=src,protocol=name,calibration_rows=len(train),holdout_rows=len(test),offset=offset,
                              affine_intercept=intercept,affine_slope=slope,logD_intercept=d_intercept,logD_slope=d_slope,**metric(test.val_loss,pred)))
        if kind=='classic':
            def apply(method,tr,te):
                if method=='intercept_20pct':
                    return te.prediction+(tr.val_loss-tr.prediction).mean()
                if method=='affine_20pct':
                    c,a=np.polyfit(tr.prediction,tr.val_loss,1)
                    return a+c*te.prediction
                c,a=np.polyfit(np.log(tr.D_tokens_B),tr.val_loss-tr.prediction,1)
                return te.prediction+a+c*np.log(te.D_tokens_B)
            scores={}
            for method in ['intercept_20pct','affine_20pct','logD_20pct']:
                errors=[]
                for idx in train.index:
                    hold=train.loc[[idx]];tr=train.drop(index=idx)
                    errors.append(float(hold.val_loss.iloc[0]-apply(method,tr,hold).iloc[0]))
                scores[method]=float(np.sqrt(np.mean(np.square(errors))))
            chosen=min(scores,key=scores.get)
            selected_calibration.append(dict(source=src,selected_method=chosen,calibration_rows=len(train),
                                             loo_rmse_intercept=scores['intercept_20pct'],loo_rmse_affine=scores['affine_20pct'],
                                             loo_rmse_logD=scores['logD_20pct'],**metric(test.val_loss,predictions[chosen])))
            calib.append(dict(model=kind,source=src,protocol='nested_selected_20pct',calibration_rows=len(train),holdout_rows=len(test),
                              offset=offset,affine_intercept=intercept,affine_slope=slope,logD_intercept=d_intercept,logD_slope=d_slope,
                              selected_method=chosen,**metric(test.val_loss,predictions[chosen])))
    save(calib,'source_calibration_holdout.csv')
    save(selected_calibration,'source_calibration_model_selection.csv')
    frac_rows=[]
    for src in ['B2','B4','B5']:
        x=d[src].copy();x['prediction']=predict_nd('classic',fits['classic'],x.N_params_B,x.D_tokens_B)
        group='run_id' if src=='B2' else 'family'
        for fraction in [.1,.2,.3]:
            selected=[]
            for _,g in x.groupby(group,sort=False):
                g=g.sort_values('D_tokens_B' if src=='B2' else 'N_params_B')
                selected+=g.index[:max(1,int(np.floor(fraction*len(g))))].tolist()
            train=x.loc[sorted(set(selected))];test=x.drop(index=train.index)
            slope,intercept=np.polyfit(train.prediction,train.val_loss,1)
            d_slope,d_intercept=np.polyfit(np.log(train.D_tokens_B),train.val_loss-train.prediction,1)
            for protocol,pred in [('raw',test.prediction),('affine',intercept+slope*test.prediction),
                                  ('logD',test.prediction+d_intercept+d_slope*np.log(test.D_tokens_B))]:
                frac_rows.append(dict(source=src,fraction=fraction,protocol=protocol,calibration_rows=len(train),**metric(test.val_loss,pred)))
    save(frac_rows,'calibration_fraction_sensitivity.csv')
    return pd.DataFrame(fitrows),pd.DataFrame(cv),pd.DataFrame(extrap),pd.DataFrame(external),pd.DataFrame(calib),fits

def q_experiments(d,b7,b8,base):
    fits={};rows=[];val=[];equiv=[];qcv=[]
    for kind in QKINDS:
        f=fit_q(kind,d['B6'],base);fits[kind]=f.x
        pr=predict_q(kind,f.x,d['B6'].N_params_B,d['B6'].D_tokens_B,d['B6'].Q_score,base)
        rss=float(sum((pr-d['B6'].val_loss)**2));k=len(f.x)
        rows.append(dict(model=kind,parameters=json.dumps(f.x.tolist()),n_parameters=k,b6_rmse=metric(d['B6'].val_loss,pr)['rmse'],
                         bic=float(len(pr)*np.log(rss/len(pr))+k*np.log(len(pr))),jacobian_condition=float(np.linalg.cond(f.jac.T@f.jac))))
        for src,x in [('B7_new',b7),('B8_new',b8)]:
            val.append(dict(model=kind,source=src,**metric(x.val_loss,predict_q(kind,f.x,x.N_params_B,x.D_tokens_B,x.Q_score,base))))
        for held in sorted(d['B6'].N_params_B.unique()):
            tr=d['B6'][d['B6'].N_params_B!=held];te=d['B6'][d['B6'].N_params_B==held]
            ff=fit_q(kind,tr,base)
            qcv.append(dict(model=kind,held_N_B=held,**metric(te.val_loss,predict_q(kind,ff.x,te.N_params_B,te.D_tokens_B,te.Q_score,base))))
        for N in [.1,1,7,70,120]:
            for D in [20,300,2000]:
                ne=equivalent_N(kind,f.x,base,N,D,.6,.1)
                equiv.append(dict(model=kind,N_B=N,D_B=D,Q=.6,N_equiv_B=ne,multiplier=ne/N))
    save(rows,'quality_fit_comparison.csv');save(val,'quality_transfer.csv');save(equiv,'quality_equivalence_by_model.csv');save(qcv,'quality_leave_N_out.csv')
    rng=np.random.default_rng(SEED);sizes=sorted(d['B6'].N_params_B.unique());boot=[]
    for i in range(200):
        draw=rng.choice(sizes,len(sizes),replace=True)
        sample=pd.concat([d['B6'][d['B6'].N_params_B==s] for s in draw],ignore_index=True)
        f=fit_q('quality_model',sample,base)
        boot.append(dict(replicate=i,G=f.x[0],kappa=f.x[1],eta_N=f.x[2]))
    save(boot,'quality_model_cluster_bootstrap.csv')
    zb=pd.DataFrame(boot)
    save([dict(parameter=k,ci_low=float(zb[k].quantile(.025)),median=float(zb[k].median()),ci_high=float(zb[k].quantile(.975))) for k in ['G','kappa','eta_N']],'quality_model_intervals.csv')
    return pd.DataFrame(rows),pd.DataFrame(val),pd.DataFrame(equiv),fits

def q_mapping_and_p():
    # These maps are declared scenarios, not empirical calibration.
    maps=[]
    for name,slope,center in [('conservative',.5,.6),('neutral',1,.6),('optimistic',2,.6)]:
        qa0,qa1=.6,.7;qb0=center;qb1=center+slope*(qa1-qa0)
        maps.append(dict(scenario=name,Q_A_start=qa0,Q_A_end=qa1,Q_B_start=qb0,Q_B_end=qb1,quality_scale_multiplier=slope,empirical_anchor=False))
    save(maps,'QA_QB_scenarios.csv')
    p=[]
    for N in [.001,.06,1,7,70,120]:
        for law in ['constant','decay_N_power_0.1','grow_N_power_0.1']:
            lam=1 if law=='constant' else (N/.001)**(-.1 if law.startswith('decay') else .1)
            p.append(dict(N_B=N,scenario=law,lambda_p=lam,identified=False))
    save(p,'p_scale_scenarios.csv')

def p_scale_evidence():
    """Exploratory centered-effect amplitude from saved P1 v1 predictions only."""
    path=ROOT/'problem1'/'outputs'/'tables'/'test_predictions.csv'
    x=pd.read_csv(path)
    x=x[(x.model=='alr_ridge')&x.dataset.isin(['test_1m','test_60m','test_1B'])]
    g=x.groupby(['dataset','index'],as_index=False)[['actual','predicted']].mean()
    rng=np.random.default_rng(SEED);rows=[]
    for name,z in g.groupby('dataset'):
        a=z.actual.to_numpy();p=z.predicted.to_numpy()
        slope=float(np.cov(p,a,ddof=0)[0,1]/np.var(p))
        boots=[]
        for _ in range(500):
            ii=rng.integers(0,len(z),len(z));pb=p[ii];ab=a[ii]
            if np.var(pb)>1e-12:boots.append(float(np.cov(pb,ab,ddof=0)[0,1]/np.var(pb)))
        rows.append(dict(dataset=name,n_mixtures=len(z),centered_slope=slope,ci_low=float(np.quantile(boots,.025)),
                         ci_high=float(np.quantile(boots,.975)),spearman=float(spearmanr(p,a).statistic),
                         source='problem1 v1 alr_ridge saved predictions; exploratory, not selected v2 effect'))
    save(rows,'p_scale_transfer_from_problem1.csv')
    return pd.DataFrame(rows)

def summary_tables(ndfit,cv,extrap,external,qfit,qval):
    def pool(df):return float(np.sqrt(np.average(df.rmse**2,weights=df.n))) if len(df) else np.nan
    rows=[]
    for model in ndfit.model:
        c=cv[cv.model==model];ext=external[external.model==model];ex=extrap[(extrap.model==model)&(extrap.axis=='N_small_to_large')]
        row=dict(model=model,B1_group_cv_rmse=pool(c),N_extrap_rmse=pool(ex),parameters=int(ndfit.loc[ndfit.model==model,'n_parameters'].iloc[0]),bic=float(ndfit.loc[ndfit.model==model,'bic'].iloc[0]))
        for source in ['B2','B4','B5']:
            z=ext[ext.source==source];row[source+'_raw_rmse']=float(z.rmse.iloc[0]) if len(z) else np.nan
        rows.append(row)
    frame=pd.DataFrame(rows);save(frame,'nd_model_scorecard.csv')
    qrows=[]
    qcv=pd.read_csv(T/'quality_leave_N_out.csv')
    for _,r in qfit.iterrows():
        z=qval[qval.model==r.model]
        zcv=qcv[qcv.model==r.model]
        qrows.append(dict(model=r.model,B6_fit_rmse=r.b6_rmse,B6_leave_N_out_rmse=pool(zcv),
                          B7_new_rmse=float(z[z.source=='B7_new'].rmse.iloc[0]),B8_new_rmse=float(z[z.source=='B8_new'].rmse.iloc[0]),parameters=r.n_parameters,bic=r.bic))
    save(qrows,'quality_model_scorecard.csv')
    return frame,pd.DataFrame(qrows)

def charts(ndscore,qscore,extrap,calib,eq):
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(1,3,figsize=(11,3.3))
    for a,col in zip(ax,['B2_raw_rmse','B4_raw_rmse','B5_raw_rmse']):
        x=ndscore.sort_values(col);a.barh(x.model,x[col]);a.set_title(col)
    fig.tight_layout();fig.savefig(F/'nd_external_comparison.png',dpi=240);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4))
    for name,g in extrap[extrap.axis=='N_small_to_large'].groupby('model'):ax.plot(g.train_fraction,g.rmse,marker='o',label=name)
    ax.set_xlabel('Fraction of smallest N groups in training');ax.set_ylabel('RMSE on larger N');ax.legend();fig.tight_layout();fig.savefig(F/'small_to_large_extrapolation.png',dpi=240);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(9,3.5))
    for a,col in zip(ax,['B7_new_rmse','B8_new_rmse']):
        x=qscore.sort_values(col);a.barh(x.model,x[col]);a.set_title(col)
    fig.tight_layout();fig.savefig(F/'quality_transfer_comparison.png',dpi=240);plt.close(fig)
    x=calib[(calib.model=='classic')&calib.protocol.isin(['raw','affine_20pct'])];fig,ax=plt.subplots(figsize=(6,4))
    for j,name in enumerate(['raw','affine_20pct']):
        g=x[x.protocol==name].set_index('source').loc[['B2','B4','B5']]
        ax.bar(np.arange(3)+(j-.5)*.34,g.rmse,width=.34,label=name)
    ax.set_xticks(np.arange(3),['B2','B4','B5']);ax.set_ylabel('Holdout RMSE');ax.legend();fig.tight_layout();fig.savefig(F/'source_calibration.png',dpi=240);plt.close(fig)
    x=eq[(eq.D_B==300)&(eq.N_B<=120)];fig,ax=plt.subplots(figsize=(6,4))
    for name,g in x.groupby('model'):ax.plot(g.N_B,g.multiplier,marker='o',label=name)
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('Initial N (billion)');ax.set_ylabel('Equivalent multiplier for Q+0.1');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(F/'equivalence_model_sensitivity.png',dpi=240);plt.close(fig)

def report(d,base,scale,mono,over,b8res,ndfit,cv,extrap,external,calib,qfit,qval,eq,ndscore,qscore,fits,qfits):
    B1=ndscore[ndscore.model=='classic'].iloc[0];qbase=qscore[qscore.model=='additive'].iloc[0]
    mono_s=mono.groupby('source').quality_improves_loss.mean().to_dict()
    overlap_mean=float(over.loss_difference_B8_minus_B6.mean())
    overlap_q1=float(over.loc[np.isclose(over.Q_score,1),'loss_difference_B8_minus_B6'].mean())
    overlap_q01=float(over.loc[np.isclose(over.Q_score,.1),'loss_difference_B8_minus_B6'].mean())
    b2cal=calib[(calib.model=='classic')&(calib.source=='B2')]
    b2trend=pd.read_csv(DIAG/'source_residual_trends.csv')
    b2trend=b2trend[(b2trend.source=='B2')&(b2trend.feature=='log_D_tokens_B')].iloc[0]
    eprof=pd.read_csv(DIAG/'profile_E_interval.csv').iloc[0]
    classic_cond=float(ndfit.loc[ndfit.model=='classic','jacobian_condition'].iloc[0])
    broken_cond=float(ndfit.loc[ndfit.model=='broken_N','jacobian_condition'].iloc[0])
    ext_best=ndscore.sort_values(['N_extrap_rmse','B2_raw_rmse']).iloc[0]
    # A purported source-agnostic winner must improve two external sources and B1 extrapolation.
    qualify=[]
    for _,r in ndscore.iterrows():
        if r.model=='classic':continue
        wins=sum(r[s+'_raw_rmse']<B1[s+'_raw_rmse']*.95 for s in ['B2','B4','B5'])
        if wins>=2 and r.N_extrap_rmse<=B1.N_extrap_rmse*1.1 and r.B1_group_cv_rmse<=B1.B1_group_cv_rmse*2:qualify.append(r.model)
    model_B='classic' if not qualify else min(qualify,key=lambda m:ndscore.set_index('model').loc[m,'N_extrap_rmse'])
    model_C='classic'
    qbest=qscore.sort_values('B7_new_rmse').iloc[0]
    q_selected='additive'
    if (qbest.model!='additive' and qbest.B7_new_rmse<qbase.B7_new_rmse*.95
        and qbest.B8_new_rmse<=qbase.B8_new_rmse and qbest.bic<qbase.bic-6):
        q_selected=qbest.model
    # B8 contradiction prohibits selecting any Q law by its B8 score.
    mapping=pd.read_csv(T/'QA_QB_scenarios.csv')
    qmap=[]
    for _,r in mapping.iterrows():
        for kind in QKINDS:
            # Generalized equivalence for arbitrary Q_B delta by direct inversion.
            tar=float(predict_q(kind,qfits[kind],7,300,r.Q_B_end,base));f=lambda n:float(predict_q(kind,qfits[kind],n,300,r.Q_B_start,base))-tar
            from scipy.optimize import brentq
            ne=np.inf if f(1e12)>0 else brentq(f,7,1e12)
            qmap.append(dict(mapping=r.scenario,Q_model=kind,N_B=7,D_B=300,Q_B_start=r.Q_B_start,Q_B_end=r.Q_B_end,N_equiv_B=ne,multiplier=ne/7))
    save(qmap,'QA_QB_equivalence_sensitivity.csv')
    # Full comparison crossing ND law and Q law; no spurious joint fitting.
    cross=[]
    for nd in ['classic',model_B]:
        for q in ['additive',q_selected]:
            cross.append(dict(ND_model=nd,Q_model=q,B1_group_cv=float(ndscore.set_index('model').loc[nd,'B1_group_cv_rmse']),
                scale_extrap=float(ndscore.set_index('model').loc[nd,'N_extrap_rmse']),B2=float(ndscore.set_index('model').loc[nd,'B2_raw_rmse']),
                B4=float(ndscore.set_index('model').loc[nd,'B4_raw_rmse']),B5=float(ndscore.set_index('model').loc[nd,'B5_raw_rmse']),
                B7_new=float(qscore.set_index('model').loc[q,'B7_new_rmse']),B8_new=float(qscore.set_index('model').loc[q,'B8_new_rmse']),
                parameter_count=int(ndscore.set_index('model').loc[nd,'parameters']+qscore.set_index('model').loc[q,'parameters']),
                bic_B1=float(ndscore.set_index('model').loc[nd,'bic']),bic_B6=float(qscore.set_index('model').loc[q,'bic']),protocol='raw_all'))
    for q in ['additive',q_selected]:
        qs=qscore.set_index('model').loc[q]
        for proto in ['raw','affine_20pct']:
            row=dict(ND_model='classic',Q_model=q,B1_group_cv=float(B1.B1_group_cv_rmse),scale_extrap=float(B1.N_extrap_rmse),
                     parameter_count=int(5+qs.parameters+(2 if proto=='affine_20pct' else 0)),
                     bic_B1=float(B1.bic),bic_B6=float(qs.bic),protocol='source_holdout_'+proto)
            for src in ['B2','B4','B5']:
                row[src]=float(calib[(calib.model=='classic')&(calib.source==src)&(calib.protocol==proto)].rmse.iloc[0])
            row['B7_new']=float(qs.B7_new_rmse);row['B8_new']=float(qs.B8_new_rmse)
            cross.append(row)
    cross=pd.DataFrame(cross).drop_duplicates();save(cross,'v1_v2_comparison.csv')
    qpar=qfits[q_selected]
    qints=pd.read_csv(T/'quality_model_intervals.csv').set_index('parameter')
    eq_v2=eq[(eq.model==q_selected)&(eq.D_B==300)].set_index('N_B').multiplier
    q_decision_text=(f'质量×N 的 B6 按 N 留一 RMSE 从 {qbase.B6_leave_N_out_rmse:.5f} 降至 {qbest.B6_leave_N_out_rmse:.5f}，'
                     f'B7 RMSE 从 {qbase.B7_new_rmse:.5f} 降至 {qbest.B7_new_rmse:.5f}，'
                     f'B8 从 {qbase.B8_new_rmse:.4f} 降至 {qbest.B8_new_rmse:.4f}，'
                     f'BIC 从 {qbase.bic:.1f} 改为 {qbest.bic:.1f}；多 1 个参数，'
                     f'η_N={qpar[2]:.4f}，按 B6 的 N 簇 bootstrap 95% 区间 '
                     f'[{qints.loc["eta_N","ci_low"]:.4f},{qints.loc["eta_N","ci_high"]:.4f}]。'
                     '它仍不能解释 B8 的质量方向反转。'
                     if q_selected=='quality_model' else '复杂 Q 候选未达到预设 B7/B8/BIC 门槛，保留加性 Q。')
    result=dict(model_A='V1 classic + additive',model_B='V2 quality_model + prespecified source affine calibration (20% source labels)',model_C=model_C+' + '+q_selected,
                recommended_ND=model_C,recommended_Q=q_selected,B8_direction_consistent_fraction=mono_s,
                B6_B8_overlap_mean_difference=overlap_mean,
                selection_note='B2/B4/B5/B7 are comparative development evaluations, not fresh final tests; B8 used for diagnosis only; no external labels enter B1/B6 parameter fitting')
    (T/'selection.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    v1_artifact=joblib.load(V1/'models'/'final_generalized_scaling.joblib')
    joblib.dump(dict(ND_model=model_C,ND_parameters=fits[model_C],Q_model=q_selected,Q_parameters=qfits[q_selected],
                     mixture_model=v1_artifact['mixture_model'],reference_p=v1_artifact['reference_p'],lambda_p=1,
                     baseline_v1_unchanged=True),M/'recommended_model.joblib')
    coefficients={r.source:dict(affine_intercept=float(r.affine_intercept),affine_slope=float(r.affine_slope),
                                logD_intercept=float(r.logD_intercept),logD_slope=float(r.logD_slope),offset=float(r.offset),
                                calibration_rows=int(r.calibration_rows),holdout_rows=int(r.holdout_rows))
                  for _,r in calib[(calib.model=='classic')&(calib.protocol=='affine_20pct')].iterrows()}
    joblib.dump(dict(base_parameters=fits['classic'],Q_model=q_selected,Q_parameters=qfits[q_selected],source_calibration=coefficients,
                    condition='Use only after obtaining labeled calibration observations from target source'),M/'model_B_source_calibrated.joblib')
    def mdtable(df):
        cols=list(df.columns)
        def cell(v):
            if isinstance(v,(float,np.floating)):
                return '—' if not np.isfinite(v) else f'{v:.5g}'
            return str(v).replace('|','/')
        rows=['| '+' | '.join(map(str,cols))+' |','| '+' | '.join(['---']*len(cols))+' |']
        rows+=['| '+' | '.join(cell(v) for v in rec)+' |' for rec in df.itertuples(index=False,name=None)]
        return '\n'.join(rows)
    diag=f'''# 第二轮诊断报告

## 1. 为什么 B1/B3 几乎完美

B1 的 N、D 对数相关为 {scale.loc[scale.source=='B1','logN_logD_corr'].iloc[0]:.4f}，并非 N-D 共线导致的伪拟合；8 个 N 组共享接近相同的 D 网格。V1 全样本残差标准差为 {scale.loc[scale.source=='B1','residual_sd'].iloc[0]:.6f}，val_loss 主要保留四位小数，因此该附件在给定 N,D 下近似确定性。B3 由同族检查点插值，几乎完美并不提供新的独立验证。LOMO 排除了同模型检查点直接泄漏，但由于所有组共享 Pythia 数据顺序、评价方式和近同一函数族，其误差会对跨源风险虚假乐观。经典模型 Jacobian 条件数约 {classic_cond:.0f}，B1 内部 profile-LR 的 E 区间为 [{eprof.E_low:.6f},{eprof.E_high:.6f}]；broken-N 条件数约 {broken_cond:.2e}，显示额外转折参数弱识别。极窄区间只反映同源噪声，绝不覆盖跨来源偏移。

## 2. B2 来源偏移

V1 对 B2 raw bias 为 {scale.loc[scale.source=='B2','bias'].iloc[0]:.4f}（prediction-observation），显示明显的整体水平差。观测减预测的残差与 log D 的相关系数为 {b2trend.r:.3f}，说明截距之外还有训练阶段形状差异。20% 早期轨迹校准、80% 后期 holdout 的严格协议如下；校准成绩不能写作 raw 泛化；10/20/30% 校准量敏感性见 `calibration_fraction_sensitivity.csv`：

{mdtable(b2cal[['protocol','calibration_rows','holdout_rows','rmse','bias']])}

## 3. B8 不是简单的大 N 指数改变

在相同 N,D 下，Q 增加使 Loss 下降的比例：B6={mono_s.get('B6',float('nan')):.2%}，B7 全网格={mono_s.get('B7_full',float('nan')):.2%}，B8 新点={mono_s.get('B8_new',float('nan')):.2%}。B7 的新增 90 点对单个 N,D 通常不足三档 Q，故方向诊断用全 B7 网格，预测验证只用新增点。B6 与 B8 有 {len(over)} 个完全相同的 N,D,Q 点，但 B8-B6 Loss 平均差为 {overlap_mean:.4f}；在 Q=1 时差约 {overlap_q1:.4f}，Q=0.1 时差约 {overlap_q01:.4f}，因此不是单纯 source intercept。B8 新点 residual（观测-预测）均值={b8res.residual_observed_minus_predicted.mean():.4f}，按 Q 的系统变化明显；按 N 分层均值变化较弱。质量方向反转与同点冲突意味着源的 Q 语义/生成机制不相容。B8 的低 Loss 截断 0.5 进一步影响低 Q 区。不能通过平滑 broken-N 指数解释这些证据，也不能拿 B8 标签拟合后再称独立外推。

## 4. Loss 口径与结构参数

B4/B5 的验证语料和单位口径与 B1 未证实相同；保留 raw 指标。Profile E 与参数相关性检验见 diagnostics 与 V1 表。极窄同源置信区间不包含 B2/B8 的 source uncertainty；跨来源误差应另列，而非并入点估计 CI。
'''
    (HERE/'diagnostic_report.md').write_text(diag,encoding='utf-8')
    brief=f'''# 问题二第二轮研究摘要

- V1 保持原样。B1 近乎确定性，LOMO 检验的是同一 Pythia 生成规律；B2 raw RMSE={B1.B2_raw_rmse:.4f}。
- B8 质量方向与 B6 相反，且同一 N,D,Q 有 {len(over)} 个 Loss 冲突点；任何保持 Q 越高 Loss 越低的单一规律都无法同时准确拟合两者。
- 候选预测型：{result['model_B']}；论文主模型：{result['model_C']}。具体 raw / 外推对照见 `outputs/tables/v1_v2_comparison.csv`。
- B2 的来源校准须先提供少量该来源 Loss，不能算零样本泛化。B4/B5 raw 指标单列。
- Q×N 质量项在 B7 新点和 B8 诊断上均优于加性 Q；新模型 Q_B:0.6→0.7、D=300B 的等价参数倍数：7B {eq_v2.loc[7]:.3f}×，70B {eq_v2.loc[70]:.3f}×，120B {eq_v2.loc[120]:.3f}×。B8 仍存在反向机制，不能视为统一实测结论。
- p 的规模系数与 Q_A→Q_B 映射没有共同锚点，仅给情景，不给精确估计。
'''
    (HERE/'problem2_v2_summary.md').write_text(brief,encoding='utf-8')
    comp=f'''# 问题二 V1 与第二轮候选对照

V1 的代码、模型、报告和输出原样保存在 `problem2/`。以下数值由 `problem2_v2/run_v2.py` 从原始 B 和 V1 参数重新计算；B2/B4/B5/B7 已用于本轮多候选比较，**没有全新封存的外部测试集**，因此数值是透明的开发性比较。B8 只用于失败诊断，未进入参数拟合。两个 BIC 分别对应 B1 N-D 与 B6 Q，不能相加当成同一似然。

{mdtable(cross)}

**Model A**：V1 经典 N-D + 加性 Q + 可分离配比。

**Model B**：条件预测型 {result['model_B']}。校准形式预先固定为仿射，只用该源约 20% 标签估计截距和斜率；剩余 holdout 只评估一次。它适用于 B2/B4/B5 可观测的 N-D、Q 未测、参考 p 情景；不能据此声称 Q 或 p 的跨源幅度也应同倍缩放。它不能与全量 raw 测试集直接混列或宣称零样本泛化。另做的校准集内 LOO 形式选择在 B5 误选截距并使 holdout 恶化，因此不采用自适应形式选择作为主结果。

**Model C**：论文主模型 {result['model_C']}。{q_decision_text}源校准作为附加应用层，B8 机制冲突明确保留。
'''
    (HERE/'problem2_v1_v2_comparison.md').write_text(comp,encoding='utf-8')
    pscale=pd.read_csv(T/'p_scale_transfer_from_problem1.csv')
    eqtable=eq[eq.D_B==300].pivot(index='model',columns='N_B',values='multiplier').reset_index()
    brief_body=brief.partition('\n\n')[2]
    full=f'''# 问题二第二轮研究报告

## 研究目标与协议

保持 V1 不变；B1 模型留一与小→大 N、早→晚 D 截断训练评估结构外推。所有 N-D 候选只在 B1 拟合。B2/B4/B5 raw 比较在本轮用于模型研究与结论判断，因此不再是全新封存的最终测试。source 校准取各来源约 20% 的标签并在剩余约 80% 上评估；多个校准形式均已查看，其最终仿射结果属于**描述性条件 holdout**，不能称为无偏的全新外部验证。Q 参数只从 B6 拟合；B7 去重作为开发验证，B8 去重仅作机制诊断。B10 是估算表，未用于 V2 选型。

候选结构依据 [Hoffmann et al., NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/c1e2faff6f588870935f114ebe04a3e5-Paper-Conference.pdf)、[Caballero et al., ICLR 2023](https://openreview.net/pdf?id=sckjveqlCZ)、[Ye et al., ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/cc84bfabe6389d8883fc2071c848f62a-Abstract-Conference.html) 和 [Chang et al., EMNLP 2024](https://aclanthology.org/2024.emnlp-industry.8.pdf)。文献适用边界与官方代码记录在 `literature_review.md`。

## N-D 候选比较

{mdtable(ndscore)}

Interaction、compute/ratio、broken-N、ordinary/Huber/soft-L1/relative/log 残差目标均实际拟合。broken law 仅加有限自由参数；转折点 profile、BIC 与截断外推共同判断其可识别性。Compute-ratio 候选的 C≈6ND 是重参数化探索，不自动带来新信息。

## Q 候选比较

{mdtable(qscore)}

加性、幂/指数有效 token、饱和质量项、质量×D 和质量×N 均拟合；B8 误差对所有保持正常质量方向的模型都不能单独作为选型依据。交互系数和 B7 新点结果在模型参数表中。固定 D=300B、Q_B:0.6→0.7 时等价参数倍数如下；其他 D 见 `quality_equivalence_by_model.csv`。70B/120B 已远超 B1 支持，应限制为敏感性分析。

{mdtable(eqtable)}

## 来源校准、p、Q 跨源

20% 校准 holdout 指标见 `source_calibration_holdout.csv`，校准形式的内部 LOO 诊断见 `source_calibration_model_selection.csv`。来源特定 E/截距可以解释部分绝对 Loss 口径偏移，B2 的 log D 修正还能单独改善其后期轨迹，但不在 B4/B5 同时占优；两者都需要该来源实测锚点，不可能由 B1 单独识别。问题一**已有输出**中的旧版 ALR 预测可用于探索中心化配比幅度的跨尺度变化（未重读附件 A）：

{mdtable(pscale[['dataset','n_mixtures','centered_slope','ci_low','ci_high','spearman']])}

这些斜率依赖问题一旧版配比模型及各尺度配方的支持域，不能直接视为最终第二版模型的 λ_p(N) 精确估计。第二版未保存同样的跨尺度逐配方预测，故 additive 与乘法桥接只做情景。乘法形式在参考点一阶可与 additive 对齐，无法用 B 数据区分。Q_A 与 Q_B 无共同锚点，三种单调情景列于 `QA_QB_scenarios.csv`；它们不是估计出的映射。

## 最终判断

{brief_body}

Model A 保留 V1 的完整公式：

$$
L=1.689798+0.353980N^{{-0.339977}}+1.240306D^{{-0.279878}}
 +0.362175(1-Q_B)^{{0.990735}}+\lambda_p\Delta_p(\mathbf p).
$$

Model C 的完整公式为：

$$
L=1.689798+0.353980N^{{-0.339977}}+1.240306D^{{-0.279878}}
 +{qpar[0]:.6f}(1-Q_B)^{{{qpar[1]:.6f}}}N^{{-{qpar[2]:.6f}}}
 +\lambda_p\Delta_p(\mathbf p).
$$

Model B 在有目标来源标签时只修正可观测 N-D 部分：$L_s^{{cal}}(N,D)=a_s+c_sL_{{ND}}(N,D)$，随后可在参考情景附加 Model C 的 Q 与 p 项；这些项的跨源幅度未由 B2/B4/B5 验证。各来源的 $a_s,c_s$ 由预留校准集单独估计，不能对 Q 与 p 的跨源效应作同样缩放。

## 证据等级

论文主结论：B1 同族经典律、外部 raw 误差、B2 来源偏移、B8 同点冲突和 Q 方向反转。敏感性：复杂 N-D、质量×D、Q+0.1 的高 N 等价倍数、λ_p(N) 与 Q_A→Q_B 映射。不可写成强结论：B10 的真实验证、Q 的真实因果效应、稳定互补、单一质量律可同时适配 B6/B8。

## 对关键研究问题的逐项回答

1. **V1 最大问题**：同源几乎零误差掩盖了 source loss 水平偏移和 B8 质量机制冲突。
2. **B1 极小误差原因**：8 个 N 与共同 D 网格上近确定性、四位小数 Loss；V1 raw 全样本残差 SD {scale.loc[scale.source=='B1','residual_sd'].iloc[0]:.6f}。
3. **虚假乐观**：LOMO 没有随机行泄漏，但仍是同一 Pythia 来源；对 B2 的风险极度乐观。
4. **B2 失败**：raw bias {scale.loc[scale.source=='B2','bias'].iloc[0]:.4f}，来源截距及训练阶段形状共同变化；20% 早期校准后的后期 holdout affine RMSE {float(b2cal[b2cal.protocol=='affine_20pct'].rmse.iloc[0]):.4f}，仍有偏差。
5. **B8 失败**：B6/B8 同点冲突 {len(over)} 个，B8 在各 N,D 内高 Q 对应更高 Loss，违反 B6 方向；Q residual 相关远大于 N residual。
6. **文献方案**：Chinchilla、BNSL、有效 token、质量×D、Data Mixing Laws 均列入候选库，只有可识别的低维版本拟合。
7. **实际测试**：经典、交互、compute-ratio、broken-N、四种替代残差目标、六种 Q 形式及来源校准，见全部 CSV。
8. **最大可用改善**：有该源 20% 标签时，预设仿射校准改善三个来源 holdout；零样本结构候选没有多数来源稳定获益。校准集 LOO 自适应选择在 B5 失败，已排除。
9. **具体指标**：V1 和候选的 B1/B2/B4/B5 在 `nd_model_scorecard.csv`；B7/B8 在 `quality_model_scorecard.csv`；同 holdout 校准对照在 `v1_v2_comparison.csv`。这些外部集已被本轮研究查看，非全新封存测试。
10. **统一跨源模型**：没有。B8 的 Q 符号冲突使单一单调 Q 律不可成立。
11. **最终 Q 是否 additive**：否。{q_decision_text}
12. **Q×D / Q×N**：B6 半合成数据上的 Q×D 系数 η_D={qfits['quality_data'][2]:.5f}，Q×N 系数 η_N={qfits['quality_model'][2]:.5f}；B7 的最好改善约 {(1-qbest.B7_new_rmse/qbase.B7_new_rmse)*100:.2f}% 且 B8 仍失败。这不能证实真实训练的交互。
13. **Broken scaling**：不同阈值的 δ 接近 0，BIC 受罚，未获充分支持。
14. **source-specific E**：可作为有源标签时的应用层；B2 的单截距校准不充分，不能从 B1 推知目标源 E。
15. **p 是否 additive**：当前可分离 bridge 是可识别范围内的合理近似；缺 N,D,Q,p 联合实验，乘法模型不能被证实更好。
16. **λ_p(N)**：旧版问题一保存的逐配方输出提供探索性中心化斜率，见表；由于最终第二版缺跨尺度逐配方预测，不估计最终模型的精确指数。
17. **Q+0.1 等价参数**：V1 加性模型 7B、70B、120B 分别约 1.916×、5.332×、8.726×；V2 质量×N 分别为 {eq_v2.loc[7]:.3f}×、{eq_v2.loc[70]:.3f}×、{eq_v2.loc[120]:.3f}×。V1 大规模倍数主要来自 N 边际收益递减而 Q 收益保持常数的结构。
18. **论文主结论**：B1 经典律、外部 raw 失效、source 校准条件、B8 同点冲突与质量方向反转。
19. **仅敏感性**：Q×D、broken-N、λ_p(N)、Q_A→Q_B 映射和 70B+ 参数等价倍数。
20. **不能写成强结论**：B10 实测外推、跨源统一 Q 因果律、已识别的 p×规模系数、稳定领域互补。
'''
    (HERE/'problem2_v2_report.md').write_text(full,encoding='utf-8')
    return result

def verify(d,mono,over,ndscore,qscore):
    checks={}
    checks['V1_model_exists']=(V1/'models'/'final_generalized_scaling.joblib').exists()
    manifest=json.loads((T/'v1_sha256.json').read_text(encoding='utf-8'))
    checks['V1_hashes_unchanged']=all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==digest for f,digest in manifest.items())
    checks['B1_only_ND_training']=len(d['B1'])==1176
    checks['B7_B8_not_fit_for_Q']=len(novel(d['B6'],d['B7']))==90 and len(novel(d['B7'],d['B8']))==1480
    checks['B8_overlap_conflict_recorded']=len(over)>0
    checks['all_candidates_fit']=set(KINDS)<=set(ndscore.model) and set(QKINDS)<=set(qscore.model)
    model=joblib.load(M/'recommended_model.joblib')
    bp=model['ND_parameters'];qp=model['Q_parameters'];qkind=model['Q_model']
    checks['Q1_reduces_to_ND']=all(np.isclose(predict_q(qkind,qp,n,300,1,bp),predict_nd('classic',bp,n,300)) for n in [.1,1,7,120])
    checks['N_D_Q_monotone']=bool(predict_q(qkind,qp,8,300,.6,bp)<predict_q(qkind,qp,7,300,.6,bp)
                                  and predict_q(qkind,qp,7,400,.6,bp)<predict_q(qkind,qp,7,300,.6,bp)
                                  and predict_q(qkind,qp,7,300,.7,bp)<predict_q(qkind,qp,7,300,.6,bp))
    ne=equivalent_N(qkind,qp,bp,7,300,.6,.1)
    checks['equivalence_exact']=bool(np.isclose(predict_q(qkind,qp,ne,300,.6,bp),predict_q(qkind,qp,7,300,.7,bp)))
    checks['p_bridge_reference_zero']=bool(np.isclose(predict_recommended(model,7,300,1,model['reference_p']),
                                                     predict_nd('classic',bp,7,300)))
    checks['reports_exist']=all((HERE/f).exists() for f in ['literature_review.md','diagnostic_report.md','model_candidates.md','problem2_v2_report.md','problem2_v1_v2_comparison.md','problem2_v2_summary.md'])
    checks['no_nonfinite_core']=bool(np.isfinite(ndscore[['B1_group_cv_rmse','B2_raw_rmse','B4_raw_rmse','B5_raw_rmse']]).all().all() and np.isfinite(qscore[['B6_fit_rmse','B7_new_rmse','B8_new_rmse']]).all().all())
    result=dict(all_passed=all(checks.values()),checks=checks)
    (T/'verification_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    if not result['all_passed']:raise AssertionError(result)

def main():
    for path in [T,F,M,DIAG]:path.mkdir(parents=True,exist_ok=True)
    v1_files=['problem2/run_problem2.py','problem2/generalized_scaling.py','problem2/problem2_report.md',
              'problem2/problem2_summary.md','problem2/outputs/models/final_generalized_scaling.joblib']
    manifest={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in v1_files}
    manifest_path=T/'v1_sha256.json'
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text(encoding='utf-8'))
        if previous!=manifest:raise AssertionError('V1 changed since previous V2 run; inspect before proceeding')
    else:manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    warnings.filterwarnings('ignore',category=RuntimeWarning)
    d=read();v1=joblib.load(V1/'models'/'final_generalized_scaling.joblib')
    bp=v1['parameters'];base=np.array([bp[k] for k in ['E','A','alpha','B','beta']],float)
    qbase=np.array([bp['G'],bp['kappa']],float)
    b7,b8,scale,mono,over,b8res=diagnostics(d,base,qbase)
    ndfit,cv,extrap,external,calib,fits=nd_experiments(d,base)
    qfit,qval,eq,qfits=q_experiments(d,b7,b8,base)
    q_mapping_and_p();p_scale_evidence();ndscore,qscore=summary_tables(ndfit,cv,extrap,external,qfit,qval)
    charts(ndscore,qscore,extrap,calib,eq)
    result=report(d,base,scale,mono,over,b8res,ndfit,cv,extrap,external,calib,qfit,qval,eq,ndscore,qscore,fits,qfits)
    verify(d,mono,over,ndscore,qscore)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
