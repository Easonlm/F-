"""Bootstrap the primary substitution effects at a feasible 1 percentage point shift."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from mixture_pipeline import pair, features, model, TABLES, SEED


def run():
    d,p,y,pcols,ycols=pair("train","1m")
    meta=json.loads((TABLES/"mixture_metadata.json").read_text(encoding="utf-8"))
    assert meta["selected_model_by_train_cv"]=="alr_ridge"
    eps=meta["alr_zero_pseudocount"]
    cv=pd.read_csv(TABLES/"mixture_cv.csv")
    alpha=float(cv[cv.model.eq("alr_ridge")].sort_values("cv_rmse").iloc[0].alpha)
    center=p.mean(axis=0);center/=center.sum(); ref=int(np.argmax(center)); delta=.01
    if center[ref]<delta:raise ValueError("reference domain cannot supply 1pp")
    names=[c.replace("train_the_pile_","") for c in pcols]
    def effect(m,i):
        shifted=center.copy();shifted[i]+=delta;shifted[ref]-=delta
        b=m.predict(features(center[None,:],"alr",eps))[0].mean()
        a=m.predict(features(shifted[None,:],"alr",eps))[0].mean()
        return a-b
    rng=np.random.default_rng(SEED)
    draws=[]
    for k in range(300):
        idx=rng.integers(0,len(p),len(p))
        m=model("alr_ridge",alpha);m.fit(features(p[idx],"alr",eps),y[idx])
        draws.append([effect(m,i) if i!=ref else np.nan for i in range(len(pcols))])
    draws=np.asarray(draws)
    m=model("alr_ridge",alpha);m.fit(features(p,"alr",eps),y)
    rows=[]
    for i,n in enumerate(names):
        if i==ref:continue
        v=draws[:,i];point=effect(m,i)
        rows.append({"increase":n,"decrease":names[ref],"delta":delta,"loss_change":point,
                     "ci_low":np.quantile(v,.025),"ci_high":np.quantile(v,.975),
                     "sign_stability":max(np.mean(v<0),np.mean(v>0))})
    pd.DataFrame(rows).sort_values("loss_change").to_csv(TABLES/"domain_effect_bootstrap.csv",index=False,encoding="utf-8-sig")
    per_domain=[]
    base=m.predict(features(center[None,:],"alr",eps))[0]
    for i,n in enumerate(names):
        if i==ref:continue
        shifted=center.copy();shifted[i]+=delta;shifted[ref]-=delta
        change=m.predict(features(shifted[None,:],"alr",eps))[0]-base
        for j,col in enumerate(ycols):
            per_domain.append({"increase":n,"decrease":names[ref],"delta":delta,
                               "validation_domain":col.replace("metric/the_pile_","").replace("_val_loss",""),
                               "predicted_loss_change":change[j]})
    pd.DataFrame(per_domain).to_csv(TABLES/"domain_effects_by_validation_domain.csv",index=False,encoding="utf-8-sig")
    print(f"Bootstrap complete; reference={names[ref]}, delta={delta}")

if __name__=="__main__":run()
