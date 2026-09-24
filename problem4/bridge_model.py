"""Loss to benchmark bridge with family grouped validation and comparability audit."""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_pipeline import OUT, save

TARGETS = ["LB_Average","LB_IFEval","LB_BBH","LB_MATH","LB_GPQA","LB_MUSR","LB_MMLU_PRO"]


def family(name):
    s = str(name).lower().split("/")[-1]
    s = re.sub(r"\d+(?:\.\d+)?[bmt]", "", s)
    s = re.sub(r"[-_ ]?(instruct|chat|base|sft|dpo|rlhf).*", "", s)
    return re.sub(r"[^a-z]", "", s)[:18] or str(name)


def fit(x, y, w, kind):
    x,y,w=np.asarray(x,float),np.asarray(y,float),np.asarray(w,float)
    if kind == "linear":
        X=np.c_[np.ones(len(x)),x]
        b=np.linalg.lstsq(X*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0]
        b[1]=min(0.,b[1]); b[0]=np.average(y-b[1]*x,weights=w)
        return {"kind":kind,"a":float(b[0]),"b":float(b[1])}
    if kind == "logistic":
        def pred(z):return 100*expit(z[0]-np.exp(z[1])*x)
        r=least_squares(lambda z:(pred(z)-y)*np.sqrt(w),x0=[4.,1.],bounds=([-20,-8],[20,5]),max_nfev=3000)
        return {"kind":kind,"a":float(r.x[0]),"b":float(-np.exp(r.x[1]))}
    if kind == "isotonic":
        iso=IsotonicRegression(increasing=False,out_of_bounds="clip").fit(x,y,sample_weight=w)
        return {"kind":kind,"x":iso.X_thresholds_.tolist(),"y":iso.y_thresholds_.tolist()}
    raise ValueError(kind)


def predict(model, x):
    x=np.asarray(x,float)
    if model["kind"]=="linear":return model["a"]+model["b"]*x
    if model["kind"]=="logistic":return 100*expit(model["a"]+model["b"]*x)
    return np.interp(x,model["x"],model["y"])


def run_bridge(c5,c6):
    all_rows=[];cv=[];models={}
    for target in TARGETS:
        for label,source in [("C6",c6),("C5",c5)]:
            d=source.dropna(subset=["Val_Loss",target]).copy()
            d["group"]=d.Model.map(family)
            for comparability in ["weighted","high_only","all_unweighted"]:
                q=d[d.Loss_Comparability.str.startswith("High")].copy() if comparability=="high_only" else d.copy()
                if len(q)<4:continue
                w=np.where(q.Loss_Comparability.str.startswith("High"),1.,.35) if comparability=="weighted" else np.ones(len(q))
                x=q.Val_Loss.to_numpy(float); y=q[target].to_numpy(float);g=q.group.to_numpy()
                for kind in ["linear","logistic","isotonic"]:
                    mod=fit(x,y,w,kind);yp=predict(mod,x)
                    key=f"{label}:{comparability}:{target}:{kind}";models[key]=mod
                    all_rows.append({"source":label,"comparability":comparability,"target":target,"kind":kind,"n":len(q),"families":len(set(g)),"train_rmse":float(np.sqrt(mean_squared_error(y,yp))),"monotonic":bool(np.all(np.diff(predict(mod,np.linspace(x.min(),x.max(),100)))<=1e-8)),"loss_min":float(x.min()),"loss_max":float(x.max())})
                    if len(set(g))<3:continue
                    splitter=GroupKFold(n_splits=min(5,len(set(g))))
                    for fold,(tr,te) in enumerate(splitter.split(x,y,g)):
                        m=fit(x[tr],y[tr],w[tr],kind)
                        pr=predict(m,x[te])
                        cv.append({"source":label,"comparability":comparability,"target":target,"kind":kind,"fold":fold,"n_test":len(te),"train_families":len(set(g[tr])),"test_families":len(set(g[te])),"families_disjoint":set(g[tr]).isdisjoint(g[te]),"mae":mean_absolute_error(y[te],pr),"rmse":np.sqrt(mean_squared_error(y[te],pr))})
    summary=save(pd.DataFrame(all_rows),"loss_benchmark_bridge_models.csv")
    cvs=save(pd.DataFrame(cv),"loss_benchmark_bridge_cv.csv")
    # Bounded monotone logistic is the deployable bridge; isotonic has flat extrapolation.
    main=models["C6:weighted:LB_Average:logistic"]
    (OUT/"loss_benchmark_bridge_model.json").write_text(json.dumps(main,indent=2),encoding="utf-8")
    (OUT/"loss_benchmark_all_models.json").write_text(json.dumps(models,indent=2),encoding="utf-8")
    return main,summary,cvs,models
