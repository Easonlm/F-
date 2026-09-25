"""Separate mixture-space support from model-size extrapolation."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from mixture_pipeline import pair, alr, TABLES, FIGURES, SEED


def run():
    _,train,_,_,_=pair("train","1m")
    meta=json.loads((TABLES/"mixture_metadata.json").read_text(encoding="utf-8"))
    eps=meta["alr_zero_pseudocount"]
    x=alr(train,eps)
    mean=x.mean(axis=0);std=x.std(axis=0);std[std==0]=1
    xt=(x-mean)/std
    tree=cKDTree(xt)
    own=tree.query(xt,k=2)[0][:,1]
    threshold=float(np.quantile(own,.95))
    rows=[]
    for label,prefix,scale in [("test_1m","test","1m"),("test_60m","test","60m"),("test_1B","test","1B"),("est_10b","est","10b"),("est_70b","est","70b")]:
        _,p,_,_,_=pair(prefix,scale)
        distance=tree.query((alr(p,eps)-mean)/std,k=1)[0]
        rows.append({"dataset":label,"n":len(p),"training_loo_95pct_distance":threshold,
                     "median_nearest_train_distance":np.median(distance),"p95_nearest_train_distance":np.quantile(distance,.95),
                     "outside_training_local_support_rate":np.mean(distance>threshold),
                     "exact_training_mixture_match_rate":np.mean(distance<1e-10),
                     "axis_of_extrapolation":"parameter-scale; labels estimated" if prefix=="est" else "parameter-scale" if scale!="1m" else "mixture-space"})
    pd.DataFrame(rows).to_csv(TABLES/"mixture_support.csv",index=False)
    print(pd.DataFrame(rows)[["dataset","outside_training_local_support_rate","exact_training_mixture_match_rate"]].to_string(index=False))

if __name__=="__main__":run()
