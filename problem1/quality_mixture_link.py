"""Test whether defensible mapped-domain Q features help mixture prediction.

The Q features are deterministic functions of p because mapped domain qualities
are constants across RegMix runs. Consequently this is prediction comparison,
not independent identification of a causal Q effect.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from mixture_pipeline import pair, alr, TABLES, ROOT, SEED


def run():
    domain=pd.read_csv(TABLES/"domain_quality_scores.csv")
    dom=domain[domain.method.eq("Q_baseline")].set_index(["source","domain"])["mean"]
    q={d:dom.loc[("A1",d)] for d in domain.domain.unique() if ("A1",d) in dom.index}
    q["arxiv"]=dom.loc[("A2","arxiv")];q["github"]=dom.loc[("A3","github")]
    mapping=pd.read_csv(ROOT/"real_attachments"/"A_data_value"/"domain_mapping_guide.csv")
    mapped=mapping[mapping.mapping_type.isin(["direct","near_direct"])].copy()
    d,p,y,pcols,ycols=pair("train","1m")
    eps=float(np.min(p[p>0])/2)
    names=[c.replace("train_the_pile_","") for c in pcols]
    qvec=np.array([q.get(mapped.set_index("mixture_domain").quality_domain.get(n,""),np.nan) for n in names])
    known=np.isfinite(qvec)
    def features(p,extra):
        base=alr(p,eps)
        if not extra:return base
        mass=p[:,known].sum(axis=1)
        quality=(p[:,known]@qvec[known])/np.maximum(mass,1e-9)
        quality=np.where(mass>0,quality,np.mean(qvec[known]))
        return np.column_stack([base,quality,mass])
    cv=KFold(5,shuffle=True,random_state=SEED)
    records=[]
    for extra in [False,True]:
        X=features(p,extra)
        for alpha in [.1,1,10,100]:
            errors=[]
            for tr,va in cv.split(X):
                m=make_pipeline(StandardScaler(),Ridge(alpha=alpha));m.fit(X[tr],y[tr]);errors.append(np.sqrt(mean_squared_error(y[va],m.predict(X[va]))))
            records.append({"model":"ALR+mapped_Q" if extra else "ALR_only","alpha":alpha,"cv_rmse":np.mean(errors),
                            "test_1m_rmse":np.nan,"mapped_domains":int(known.sum())})
        best=min((r for r in records if r["model"]==("ALR+mapped_Q" if extra else "ALR_only")),key=lambda r:r["cv_rmse"])
        m=make_pipeline(StandardScaler(),Ridge(alpha=best["alpha"]));m.fit(X,y)
        for scale in ["1m","60m","1B"]:
            _,pt,yt,_,_=pair("test",scale)
            best[f"test_{scale}_rmse"]=float(np.sqrt(mean_squared_error(yt,m.predict(features(pt,extra)))))
    pd.DataFrame(records).to_csv(TABLES/"quality_mixture_comparison.csv",index=False)
    print(pd.DataFrame(records).sort_values("cv_rmse").head(4).to_string(index=False))

if __name__=="__main__":run()
