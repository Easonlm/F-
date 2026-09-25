"""A4-A15 compositional mixture models, held-out tests, extrapolation and effects."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from matplotlib.colors import SymLogNorm
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "real_attachments" / "A_data_value" / "regmix_tables"
OUT = Path(__file__).resolve().parent / "outputs"
TABLES, FIGURES = OUT / "tables", OUT / "figures"
SEED=20260923


def pair(prefix, scale):
    mix=pd.read_csv(DATA/f"{prefix}_mixture_{scale}.csv")
    loss=pd.read_csv(DATA/f"{prefix}_pile_loss_{scale}.csv")
    if mix["index"].duplicated().any() or loss["index"].duplicated().any(): raise ValueError("duplicate index")
    d=mix.merge(loss,on="index",validate="one_to_one",indicator=True)
    if not d._merge.eq("both").all() or len(d)!=len(mix) or len(d)!=len(loss): raise ValueError("unmatched mixture/loss")
    d=d.drop(columns="_merge")
    pcols=[c for c in mix if c.startswith("train_the_pile_")]
    ycols=[c for c in loss if c.startswith("metric/the_pile_")]
    p=d[pcols].to_numpy(float)
    if np.any(p<0) or np.any(~np.isfinite(p)) or not np.allclose(p.sum(axis=1),1,atol=.005):raise ValueError("invalid simplex")
    p=p/p.sum(axis=1,keepdims=True)
    y=d[ycols].to_numpy(float)
    if np.any(~np.isfinite(y)):raise ValueError("invalid loss")
    return d,p,y,pcols,ycols


def alr(p,eps):
    x=p+eps;x=x/x.sum(axis=1,keepdims=True)
    # Last component is the explicit reference, avoids CLR rank deficiency.
    return np.log(x[:,:-1]/x[:,-1,None])


def features(p,kind,eps):
    if kind=="raw": return p[:,:-1]
    if kind=="alr": return alr(p,eps)
    if kind=="quadratic": return p[:,:-1]
    if kind=="tree":return p
    if kind=="dummy":return p
    raise KeyError(kind)


def model(name,alpha=1):
    if name=="dummy":return DummyRegressor(strategy="mean")
    if name=="raw_ridge":return make_pipeline(StandardScaler(),Ridge(alpha=alpha))
    if name=="alr_ridge":return make_pipeline(StandardScaler(),Ridge(alpha=alpha))
    if name=="quadratic_ridge":return make_pipeline(PolynomialFeatures(2,include_bias=False),StandardScaler(),Ridge(alpha=alpha))
    if name=="extra_trees":return ExtraTreesRegressor(n_estimators=250,min_samples_leaf=3,max_features=.8,n_jobs=-1,random_state=SEED)
    raise KeyError(name)


def metric(y,pred):
    residual=pred-y
    sp=[]
    for j in range(y.shape[1]):
        v=spearmanr(y[:,j],pred[:,j]).statistic
        if np.isfinite(v):sp.append(v)
    return {"mae":mean_absolute_error(y,pred),"rmse":np.sqrt(mean_squared_error(y,pred)),
            "r2_flat":r2_score(y,pred,multioutput="variance_weighted"),
            "r2_domain_mean":r2_score(y,pred,multioutput="uniform_average"),
            "mape":np.mean(np.abs(residual)/np.maximum(y,1e-8)),
            "spearman_mean":np.mean(sp) if sp else np.nan,
            "bias":np.mean(residual),
            "mean_loss_mae":mean_absolute_error(y.mean(axis=1),pred.mean(axis=1))}


def fit_models(p,y,eps):
    names=["dummy","raw_ridge","alr_ridge","quadratic_ridge","extra_trees"]
    kinds={"dummy":"dummy","raw_ridge":"raw","alr_ridge":"alr","quadratic_ridge":"quadratic","extra_trees":"tree"}
    cv=KFold(5,shuffle=True,random_state=SEED)
    models={}; cvrows=[]
    for name in names:
        alphas=[1] if name in ("dummy","extra_trees") else [.1,1,10,100,1000]
        for alpha in alphas:
            errs=[]
            X=features(p,kinds[name],eps)
            for tr,va in cv.split(X):
                m=model(name,alpha);m.fit(X[tr],y[tr]);errs.append(np.sqrt(mean_squared_error(y[va],m.predict(X[va]))))
            cvrows.append({"model":name,"alpha":alpha,"cv_rmse":np.mean(errs),"cv_rmse_sd":np.std(errs)})
        best=min((r for r in cvrows if r["model"]==name),key=lambda r:r["cv_rmse"])
        m=model(name,best["alpha"]);m.fit(features(p,kinds[name],eps),y)
        models[name]=(m,kinds[name],best["alpha"])
    pd.DataFrame(cvrows).to_csv(TABLES/"mixture_cv.csv",index=False)
    return models,cvrows


def summarize_effects(primary, p, pcols, ycols, eps):
    m,kind,alpha=primary
    center=p.mean(axis=0);center/=center.sum()
    effects=[]; matrix=np.full((len(pcols),len(pcols)),np.nan)
    for i in range(len(pcols)):
        for j in range(len(pcols)):
            if i==j:continue
            delta=min(.01,center[j]*.5)
            new=center.copy();new[i]+=delta;new[j]-=delta
            base=m.predict(features(center[None,:],kind,eps))[0]
            pred=m.predict(features(new[None,:],kind,eps))[0]
            e=float(np.mean(pred-base))/delta
            matrix[i,j]=e
            effects.append({"increase":pcols[i].replace("train_the_pile_",""),"decrease":pcols[j].replace("train_the_pile_",""),
                            "delta":delta,"direct_change_in_mean_loss":float(np.mean(pred-base)),"local_slope_per_1pct":e*.01,"mean_loss_slope":e})
    ef=pd.DataFrame(effects);ef.to_csv(TABLES/"domain_effects.csv",index=False,encoding="utf-8-sig")
    limit=np.nanpercentile(abs(matrix),95)
    fig,ax=plt.subplots(figsize=(10,8));im=ax.imshow(matrix,cmap="RdBu_r",norm=SymLogNorm(linthresh=.5,vmin=-limit,vmax=limit));names=[c.replace("train_the_pile_","") for c in pcols];ax.set_xticks(range(len(names)),names,rotation=90,fontsize=6);ax.set_yticks(range(len(names)),names,fontsize=6);ax.set(xlabel="Decrease",ylabel="Increase");fig.colorbar(im,ax=ax,label="Loss slope, symlog colors");fig.tight_layout();fig.savefig(FIGURES/"domain_substitution.png",dpi=160);plt.close(fig)
    ref=int(np.argmax(center))
    fig,ax=plt.subplots(figsize=(10,5));v=matrix[:,ref];ax.bar(range(len(names)-1),v[:-1]);ax.set_xticks(range(len(names)-1),names[:-1],rotation=90,fontsize=6);ax.set(ylabel="Change in mean Loss / unit mixture",title=f"Substitute from {names[ref]}");fig.tight_layout();fig.savefig(FIGURES/"domain_marginal_effects.png",dpi=160);plt.close(fig)


def interaction_table(models,p,pcols,eps):
    # Interactions are descriptive candidate coefficients of the regularized quadratic model.
    m,kind,alpha=models["quadratic_ridge"]
    poly,scale,ridge=m.steps[0][1],m.steps[1][1],m.steps[2][1]
    names=poly.get_feature_names_out([c.replace("train_the_pile_","") for c in pcols[:-1]])
    c=ridge.coef_/scale.scale_[None,:]
    rows=[]
    for k,name in enumerate(names):
        if " " in name:
            rows.append({"interaction":name,"mean_loss_coefficient":float(c[:,k].mean()),"ridge_alpha":alpha,"stable_evidence":False,
                         "note":"Exploratory coefficient; not identified as stable synergy without bootstrap and external support"})
    out=pd.DataFrame(rows).sort_values("mean_loss_coefficient")
    out.to_csv(TABLES/"interaction_effects.csv",index=False,encoding="utf-8-sig")
    top=pd.concat([out.head(8),out.tail(8)])
    fig,ax=plt.subplots(figsize=(9,6));ax.barh(top.interaction,top.mean_loss_coefficient,color=np.where(top.mean_loss_coefficient<0,"#357a5b","#a45150"));ax.set(xlabel="Exploratory interaction coefficient");fig.tight_layout();fig.savefig(FIGURES/"interaction_effects.png",dpi=160);plt.close(fig)


def run():
    TABLES.mkdir(parents=True,exist_ok=True);FIGURES.mkdir(parents=True,exist_ok=True)
    train,p,y,pcols,ycols=pair("train","1m")
    positive=p[p>0];eps=float(np.min(positive)/2)
    models,cvrows=fit_models(p,y,eps)
    best=min(cvrows,key=lambda x:x["cv_rmse"])["model"]
    # Model selection is solely from A4/A5 CV, before opening any test labels.
    sets=[("test_1m","test","1m"),("test_60m","test","60m"),("test_1B","test","1B"),("est_10b","est","10b"),("est_70b","est","70b")]
    comparisons=[];predrows=[]
    for label,prefix,scale in sets:
        d,pt,yt,pc,yc=pair(prefix,scale)
        assert pc==pcols and yc==ycols
        for name,(m,kind,alpha) in models.items():
            pred=m.predict(features(pt,kind,eps))
            comparisons.append({"dataset":label,"status":"observed" if prefix=="test" else "estimated_extrapolation", "model":name,"alpha":alpha,**metric(yt,pred)})
            for row,idx in enumerate(d["index"]):
                for j,col in enumerate(ycols):
                    predrows.append({"dataset":label,"model":name,"index":idx,"validation_domain":col.replace("metric/the_pile_","").replace("_val_loss",""),
                                     "actual":yt[row,j],"predicted":pred[row,j],"residual":pred[row,j]-yt[row,j]})
    comp=pd.DataFrame(comparisons)
    comp.to_csv(TABLES/"mixture_model_comparison.csv",index=False)
    pred=pd.DataFrame(predrows)
    pred.to_csv(TABLES/"test_predictions.csv",index=False)
    comp[comp.dataset.str.startswith("est_")].to_csv(TABLES/"extrapolation_results.csv",index=False)
    primary=models[best]
    (OUT/"models").mkdir(parents=True,exist_ok=True)
    joblib.dump(primary[0],OUT/"models"/"mixture_primary.joblib")
    summarize_effects(primary,p,pcols,ycols,eps)
    interaction_table(models,p,pcols,eps)
    # Plots separate observed 1M validation from cross-scale and estimated extrapolation.
    a=pred[(pred.dataset=="test_1m")&(pred.model==best)]
    fig,ax=plt.subplots(figsize=(6,6));ax.scatter(a.actual,a.predicted,s=4,alpha=.25);mn=min(a.actual.min(),a.predicted.min());mx=max(a.actual.max(),a.predicted.max());ax.plot([mn,mx],[mn,mx],color="red");ax.set(xlabel="Observed Loss",ylabel="Predicted Loss",title=best);fig.tight_layout();fig.savefig(FIGURES/"actual_vs_predicted.png",dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4));ax.scatter(a.predicted,a.residual,s=4,alpha=.25);ax.axhline(0,color="red");ax.set(xlabel="Predicted Loss",ylabel="Residual");fig.tight_layout();fig.savefig(FIGURES/"residuals.png",dpi=170);plt.close(fig)
    cc=comp[comp.dataset=="test_1m"].sort_values("rmse")
    fig,ax=plt.subplots(figsize=(8,4));ax.barh(cc.model,cc.rmse);ax.set(xlabel="RMSE on observed 1M test");fig.tight_layout();fig.savefig(FIGURES/"model_performance.png",dpi=170);plt.close(fig)
    cc=comp[comp.model==best]
    fig,ax=plt.subplots(figsize=(8,4));ax.bar(cc.dataset,cc.rmse,color=np.where(cc.status=="observed","#426a91","#a57636"));ax.set(ylabel="Absolute RMSE",title="Cross-scale transport without recalibration");fig.tight_layout();fig.savefig(FIGURES/"extrapolation_error.png",dpi=170);plt.close(fig)
    metadata={"selected_model_by_train_cv":best,"alr_zero_pseudocount":eps,"reference_domain":pcols[-1],"train_rows":len(train),"mean_train_loss":float(y.mean()),"test_scale_shift_expected":True}
    (TABLES/"mixture_metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(metadata,ensure_ascii=False))

if __name__=="__main__":run()
