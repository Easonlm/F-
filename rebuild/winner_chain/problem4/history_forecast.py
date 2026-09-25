"""Small-sample historical decomposition and explicitly exploratory frontier scenarios."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold

from data_pipeline import OUT, save


def logit_score(s):return logit(np.clip(np.asarray(s,float)/100,1e-4,1-1e-4))
def score(y):return 100*expit(y)


def frontier(a):
    d=a[a.complete&a.submission_date.notna()].copy()
    d=d.sort_values(["submission_date","ability"]).drop_duplicates("Model",keep="last")
    d["quarter"]=d.submission_date.dt.to_period("Q").astype(str)
    rows=[]
    for subset,sel in [("all",d), ("open_w1",d[d.open_w1]),("open_w2",d[d.open_w2]),
                       ("c2_inherited_open_sensitivity",d[d.c2_open_weights=="Yes"]),
                       ("open_pretrained",d[d.open_w1&(d.model_type=="pretrained")]),
                       ("open_chat_finetuned",d[d.open_w1&(d.model_type=="chat_finetuned")])]:
        for q,g in sel.groupby("quarter"):
            c=g.compute.dropna();n=g.epoch_params.dropna();dd=g.training_data.dropna()
            rows.append(dict(subset=subset,quarter=q,start_date=str(g.submission_date.min().date()),end_date=str(g.submission_date.max().date()),
                             n=len(g),median_ability=g.ability.median(),p90_ability=g.ability.quantile(.9),p95_ability=g.ability.quantile(.95),
                             top3_ability=g.ability.nlargest(min(3,len(g))).mean(),top1_ability=g.ability.max(),
                             reported_compute_n=len(c),median_compute=c.median(),p90_compute=c.quantile(.9) if len(c) else np.nan,
                             parameter_frontier=n.quantile(.9) if len(n) else np.nan,training_data_frontier=dd.quantile(.9) if len(dd) else np.nan))
    return save(pd.DataFrame(rows),"quarterly_frontier.csv")


def frontier_sensitivity(a):
    """Descriptive frontier variants; all windows use the same W1 end date."""
    d=a[a.complete&a.submission_date.notna()&a.ability.notna()].copy()
    d=d.sort_values(["submission_date","ability"]).drop_duplicates("Model",keep="last")
    d["quarter"]=d.submission_date.dt.to_period("Q").astype(str)
    cutoff=d.loc[d.open_w1,"submission_date"].max()
    quarters=sorted(d.loc[d.open_w1,"quarter"].unique())
    subsets={"open_w1":d[d.open_w1],
             "c2_inherited_aligned":d[(d.c2_open_weights=="Yes")&(d.submission_date<=cutoff)]}
    rows=[]
    for subset,source in subsets.items():
        for window in [1,2,3]:
            for end_index in range(window-1,len(quarters)):
                end=quarters[end_index]
                included=quarters[end_index-window+1:end_index+1]
                values=source.loc[source.quarter.isin(included),"ability"]
                if len(values)==0:continue
                rows.append(dict(subset=subset,window_quarters=window,end_quarter=end,
                                 included_quarters="|".join(included),n=len(values),
                                 p90_ability=values.quantile(.9),p95_ability=values.quantile(.95),
                                 top3_ability=values.nlargest(min(3,len(values))).mean(),
                                 eligible_n5=len(values)>=5,eligible_n10=len(values)>=10,
                                 W1_cutoff=str(cutoff.date())))
    return save(pd.DataFrame(rows),"frontier_definition_sensitivity.csv")


def design(d, kind="structural"):
    t=(d.submission_date-pd.Timestamp("2024-01-01")).dt.days.to_numpy(float)/365.25
    c=np.log10(d.compute.to_numpy(float))
    if kind=="structural":return np.c_[np.ones(len(d)),c-23,t]
    if kind=="compute_only":return np.c_[np.ones(len(d)),c-23]
    if kind=="time_only":return np.c_[np.ones(len(d)),t]
    if kind=="nd_time":return np.c_[np.ones(len(d)),np.log10(d.epoch_params.to_numpy(float)/1e9),np.log10(d.training_data.to_numpy(float)/1e9),t]
    raise ValueError(kind)


def fit_model(d,kind="structural"):
    x=design(d,kind); y=logit_score(d.ability)
    # Weak ridge regularization only on slopes; keeps the extrapolation finite with collinearity.
    penalty=np.eye(x.shape[1])*.04;penalty[0,0]=0
    b=np.linalg.solve(x.T@x+penalty,x.T@y)
    resid=y-x@b
    return b,float(np.std(resid,ddof=min(len(b),len(resid)-1)))


def historical(a):
    d=a[a.open_w1&a.complete&a.submission_date.notna()&(a.compute>0)&(a.epoch_params>0)].copy()
    d=d[d.epoch_confidence.isin(["Confident","Likely"])]
    # Direct owner verified matches only. One row per underlying release.
    d=d.drop_duplicates("epoch_row")
    if len(d)<12:raise ValueError(f"Only {len(d)} strict scale observations")
    rows=[]
    groups=d.organization.fillna(d.Model.str.split("/").str[0]).astype(str)
    for kind in ["time_only","compute_only","structural","nd_time"]:
        sample=d[d.training_data>0].copy() if kind=="nd_time" else d
        sample_groups=sample.organization.fillna(sample.Model.str.split("/").str[0]).astype(str)
        if len(sample)<8:continue
        b,resid=fit_model(sample,kind)
        x=design(sample,kind)
        rmse=np.sqrt(np.mean((score(x@b)-sample.ability.to_numpy(float))**2))
        fold_errors=[]
        if sample_groups.nunique()>=3:
            for tr,te in GroupKFold(n_splits=min(5,sample_groups.nunique())).split(sample,groups=sample_groups):
                train=sample.iloc[tr];test=sample.iloc[te]
                if len(train)<len(b)+2:continue
                coef,_=fit_model(train,kind)
                fold_errors.extend((score(design(test,kind)@coef)-test.ability.to_numpy(float)).tolist())
        rows.append(dict(model=kind,n=len(sample),families=sample_groups.nunique(),train_rmse=rmse,group_cv_rmse=float(np.sqrt(np.mean(np.square(fold_errors)))) if fold_errors else np.nan,
                         coefficient_json=json.dumps(b.tolist()),logit_residual_sd=resid))
    save(pd.DataFrame(rows),"historical_model_comparison.csv")
    b,resid=fit_model(d)
    rng=np.random.default_rng(20260924)
    orgs=groups.unique();boots=[]
    for _ in range(400):
        sampled=rng.choice(orgs,len(orgs),replace=True)
        q=pd.concat([d.loc[groups==g] for g in sampled],ignore_index=True)
        if len(q)<10:continue
        try:boots.append(fit_model(q)[0])
        except np.linalg.LinAlgError:pass
    boots=np.asarray(boots)
    np.save(OUT/"historical_bootstrap_coefficients.npy",boots)
    save(d[["Model","ability","submission_date","compute","training_data","epoch_params","organization","model_type","release_date"]],"historical_scale_sample.csv")
    # C7 is too sparse for a primary term; quantify coverage and a residual association only.
    return d,b,resid,boots


def compute_growth(c4,origin):
    """C4-wide open language compute series; exclude incomplete origin quarter."""
    z=c4.copy()
    z["date"]=pd.to_datetime(z["Publication date"],errors="coerce")
    z["compute"]=pd.to_numeric(z["Training compute (FLOP)"],errors="coerce")
    z=z[z.Domain.eq("Language")&z["Open model weights?"].eq("Yes")&z.Confidence.isin(["Confident","Likely"])&(z.compute>0)&z.date.between("2019-01-01",origin)]
    z["quarter"]=z.date.dt.to_period("Q").astype(str)
    o=z.groupby("quarter").agg(n=("Model","size"),p90_compute=("compute",lambda v:v.quantile(.9)),top3_compute=("compute",lambda v:v.nlargest(min(3,len(v))).mean())).reset_index()
    # Last quarter is incomplete unless origin is its final calendar day.
    if origin != origin.to_period("Q").end_time.normalize():o=o[o.quarter!=str(origin.to_period("Q"))]
    save(o,"c4_open_compute_frontier.csv")
    o=o[(o.n>=3)&(o.quarter>="2021Q1")].copy().sort_values("quarter")
    o["t"]=(pd.PeriodIndex(o.quarter,freq="Q").asi8-pd.Period(o.quarter.iloc[0],freq="Q").ordinal)/4
    y=np.log(o.p90_compute.to_numpy(float));t=o.t.to_numpy(float)
    slope=np.polyfit(t,y,1)[0]
    recent_o=o[o.quarter>="2023Q1"]
    recent=np.polyfit(recent_o.t.to_numpy(float),np.log(recent_o.p90_compute.to_numpy(float)),1)[0]
    n=len(o);rss=float(np.sum((y-np.polyval(np.polyfit(t,y,1),t))**2))
    rows=[dict(model="single_loglinear",quarters=n,slope_log_per_year=slope,annual_multiplier=np.exp(slope),bic=n*np.log(max(rss/n,1e-12))+2*np.log(n),break_quarter=""),
          dict(model="recent_2023_2024",quarters=len(recent_o),slope_log_per_year=recent,annual_multiplier=np.exp(recent),bic=np.nan,break_quarter="")]
    best=None
    for i in range(4,n-4):
        X=np.c_[np.ones(n),t,np.maximum(0,t-t[i])]
        beta=np.linalg.lstsq(X,y,rcond=None)[0]
        r=float(np.sum((y-X@beta)**2));bic=n*np.log(max(r/n,1e-12))+4*np.log(n)
        if best is None or bic<best[0]:best=(bic,i,beta)
    if best:
        bic,i,beta=best
        rows.append(dict(model="piecewise",quarters=n,slope_log_per_year=beta[1]+beta[2],prior_slope_log_per_year=beta[1],annual_multiplier=np.exp(beta[1]+beta[2]),bic=bic,break_quarter=o.iloc[i].quarter))
    else: rows.append(dict(model="piecewise",quarters=n,slope_log_per_year=np.nan,annual_multiplier=np.nan,bic=np.nan,break_quarter=""))
    save(pd.DataFrame(rows),"compute_growth_models.csv")
    return o,float(recent)


def shapley(y0,c0,c1,t0,t1,b):
    f=lambda c,t:score(y0+b[1]*(c-c0)+b[2]*(t-t0))
    s00=f(c0,t0);s10=f(c1,t0);s01=f(c0,t1);s11=f(c1,t1)
    scale=.5*((s10-s00)+(s11-s01));tech=.5*((s01-s00)+(s11-s10));total=s11-s00
    return dict(S00=s00,S10=s10,S01=s01,S11=s11,scale_points=scale,technology_points=tech,total_points=total,
                scale_share=scale/total if abs(total)>1e-8 else np.nan,technology_share=tech/total if abs(total)>1e-8 else np.nan)


def contributions(q,b,boots):
    z=q[(q.subset=="open_w1")&q.p90_compute.notna()].sort_values("quarter")
    rows=[]
    if len(z)>=2:
        pairs=[("observed_span",z.iloc[0],z.iloc[-1])]
        if len(z)>=3:pairs.append(("last_two_quarters",z.iloc[-3],z.iloc[-1]))
        for label,start,end in pairs:
            c0=np.log10(start.p90_compute);c1=np.log10(end.p90_compute)
            t0=pd.Period(start.quarter).ordinal/4;t1=pd.Period(end.quarter).ordinal/4
            y0=logit_score(start.p95_ability)
            main=shapley(y0,c0,c1,t0,t1,b)
            dist=[shapley(y0,c0,c1,t0,t1,bb) for bb in boots]
            row=dict(subset="open_w1",window=label,start_quarter=start.quarter,end_quarter=end.quarter,n_start=int(start.n),n_end=int(end.n),
                     observed_frontier_change=end.p95_ability-start.p95_ability,**main)
            for col in ["scale_points","technology_points","total_points","scale_share","technology_share"]:
                vals=np.array([x[col] for x in dist]);vals=vals[np.isfinite(vals)]
                row[col+"_lo95"],row[col+"_hi95"]=np.quantile(vals,[.025,.975]) if len(vals) else (np.nan,np.nan)
            rows.append(row)
    for label in ["last_12_months","last_24_months"]:
        rows.append(dict(subset="open_w1",window=label,status="unavailable: direct observed history < window"))
    out=save(pd.DataFrame(rows),"scale_technology_contributions.csv")
    if len(z)>=2:
        c0=np.log10(z.iloc[0].p90_compute);base=z.iloc[0].p95_ability
        index=[]
        for _,r in z.iterrows():
            dt=(pd.Period(r.quarter).ordinal-pd.Period(z.iloc[0].quarter).ordinal)/4
            index.append(dict(quarter=r.quarter,technology_logit=b[2]*dt,technology_score_equivalent=float(score(logit_score(base)+b[2]*dt)-base),
                              compute_equivalent_multiplier=float(np.exp(b[2]*dt/(b[1]/np.log(10)))) if b[1]>0 else np.nan))
        save(pd.DataFrame(index),"technology_trend.csv")
    return out


def backtest(q):
    z=q[q.subset=="open_w1"].sort_values("quarter")
    rows=[]
    for h in [1,2,4,8]:
        for origin in range(1,len(z)-h):
            train=z.iloc[:origin+1];test=z.iloc[origin+h]
            # Only the direct time baseline can be refit from tiny quarterly samples.
            t=np.arange(len(train));ys=logit_score(train.p95_ability)
            slope=np.polyfit(t,ys,1)[0]
            pred=score(ys[-1]+slope*h)
            rows.append(dict(horizon_quarters=h,horizon_months=h*3,origin_quarter=train.iloc[-1].quarter,target_quarter=test.quarter,
                             model="direct_time",prediction=pred,actual=test.p95_ability,error=pred-test.p95_ability,
                             future_leakage=False))
    if not rows:
        out=pd.DataFrame(columns=["horizon_quarters","horizon_months","origin_quarter","target_quarter","model","prediction","actual","error","future_leakage"])
    else:out=pd.DataFrame(rows)
    return save(out,"forecast_backtest.csv")


def short_term_baseline(q, observed_backtest):
    """Compare the existing direct-time backtest with a no-trend persistence rule.

    Persistence uses only the observed frontier at each existing origin. The
    same target quarters and horizon definitions are used for both methods.
    """
    z=q[q.subset=="open_w1"].sort_values("quarter").reset_index(drop=True)
    lookup={row.quarter:(i,row) for i,row in z.iterrows()}
    rows=[]
    for item in observed_backtest.itertuples(index=False):
        origin_i,origin=lookup[item.origin_quarter]
        target_i,target=lookup[item.target_quarter]
        if target_i-origin_i != int(item.horizon_quarters) or not np.isclose(float(target.p95_ability),float(item.actual)):
            raise ValueError("Short-term baseline target does not match existing backtest")
        for model,prediction in [("direct_time",float(item.prediction)),("persistence",float(origin.p95_ability))]:
            rows.append(dict(model=model,horizon_quarters=int(item.horizon_quarters),horizon_months=int(item.horizon_months),
                             origin_quarter=item.origin_quarter,target_quarter=item.target_quarter,
                             available_train_quarters=origin_i+1,origin_n=int(origin.n),target_n=int(target.n),
                             target_n_ge_5=bool(target.n>=5),prediction=prediction,actual=float(target.p95_ability),
                             error=prediction-float(target.p95_ability),abs_error=abs(prediction-float(target.p95_ability)),
                             future_leakage=False))
    columns=["model","horizon_quarters","horizon_months","origin_quarter","target_quarter","available_train_quarters",
             "origin_n","target_n","target_n_ge_5","prediction","actual","error","abs_error","future_leakage"]
    out=save(pd.DataFrame(rows,columns=columns),"short_term_baseline_comparison.csv")
    summaries=[]
    for rule,subset in [("all",out),("target_n_ge_5",out[out.target_n_ge_5])]:
        for (model,months),group in subset.groupby(["model","horizon_months"]):
            summaries.append(dict(sample_rule=rule,model=model,horizon_months=int(months),targets=len(group),
                                  mae=float(group.abs_error.mean()),rmse=float(np.sqrt(np.mean(group.error**2))),
                                  min_target_n=int(group.target_n.min())))
    save(pd.DataFrame(summaries,columns=["sample_rule","model","horizon_months","targets","mae","rmse","min_target_n"]),
         "short_term_baseline_summary.csv")
    return out


def forecast(q,b,boots,g,resid_sd,origin,compute_frontier):
    z=q[(q.subset=="open_w1")&q.p90_compute.notna()].sort_values("quarter")
    base=z.iloc[-1];base_y=logit_score(base.p95_ability)
    base_compute=float(compute_frontier.iloc[-1].p90_compute);base_c=np.log10(base_compute)
    rng=np.random.default_rng(20260924)
    rows=[];unc=[];scenarios=[]
    for months in [12,24]:
        yrs=months/12
        target=(origin+pd.DateOffset(months=months)).date().isoformat()
        for c_name,c_factor in [("continuation",1.),("moderate_slowdown",.5),("strong_slowdown",.25)]:
            future_c=float(np.exp(np.log(base_compute)+g*yrs*c_factor))
            scenarios.append(dict(horizon_months=months,scenario=c_name,growth_log_per_year=g*c_factor,base_compute=base_compute,future_compute=future_c,forecast_origin_date=str(origin.date()),target_date=target))
            for t_name,t_factor in [("continuation",1.),("half_rate",.5),("flat",0.)]:
                delta_c=np.log10(future_c)-base_c
                pred=float(score(base_y+b[1]*delta_c+b[2]*yrs*t_factor))
                if len(boots):
                    choices=boots[rng.integers(0,len(boots),size=3000)]
                    sims=score(base_y+choices[:,1]*delta_c+choices[:,2]*yrs*t_factor+rng.normal(0,resid_sd,size=3000))
                else:sims=np.array([pred])
                qu=np.quantile(sims,[.025,.1,.25,.5,.75,.9,.975])
                row=dict(horizon_months=months,target_date=target,forecast_origin_date=str(origin.date()),compute_scenario=c_name,tech_scenario=t_name,
                         predicted_compute=future_c,median_ability=pred,p50_low=qu[2],p50_high=qu[4],p80_low=qu[1],p80_high=qu[5],p95_low=qu[0],p95_high=qu[6],
                         ability_anchor_quarter=base.quarter,ability_anchor_n=int(base.n),ability_anchor_meets_n5=bool(base.n>=5),
                         status="exploratory; sparse ability anchor; no 12/24-month direct-observation backtest" if base.n<5
                         else "exploratory; no 12/24-month direct-observation backtest")
                rows.append(row)
                unc.append({**row,"bootstrap_median":qu[3],"model_residual_sd_logit":resid_sd,"n_bootstrap":len(boots)})
    save(pd.DataFrame(scenarios),"compute_slowdown_scenarios.csv")
    save(pd.DataFrame(unc),"forecast_uncertainty.csv")
    return save(pd.DataFrame(rows),"frontier_forecast_12m_24m.csv")
