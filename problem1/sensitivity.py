"""Predeclared conflict-threshold and covariance diagnostics."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from quality_pipeline import group_scores

HERE=Path(__file__).resolve().parent
T=HERE/"outputs"/"tables"


def run():
    z=pd.read_parquet(T/"quality_standardized.parquet")
    s=pd.read_parquet(T/"sample_quality_scores.parquet")
    a=z[z.source.eq("A1")].reset_index(drop=True)
    qs=s[s.source.eq("A1")].reset_index(drop=True)
    fields=pd.read_csv(T/"quality_weights.csv").field.tolist()
    corr=a[fields].corr(method="spearman").fillna(0).to_numpy()
    eig=np.linalg.eigvalsh(corr)
    inv=np.linalg.pinv(corr)
    pd.DataFrame({"field":fields,"vif_from_spearman_correlation":np.diag(inv),
                  "max_abs_other_correlation":[np.max(np.abs(np.delete(corr[i],i))) for i in range(len(fields))]}).to_csv(T/"quality_collinearity.csv",index=False)
    weights=pd.read_csv(T/"quality_weights.csv");groups=weights.group.unique().tolist()
    group=group_scores(a[fields])[groups]
    rows=[]
    for lo,hi in [(.2,.8),(.25,.75),(.3,.7)]:
        f=lambda col,q:np.quantile(a[col],q)
        edu=(a.fineweb_edu>f("fineweb_edu",hi))&(a.ad_en<f("ad_en",lo))
        flu=(a.fluency_en>f("fluency_en",hi))&(a.modernbert_cleanliness<f("modernbert_cleanliness",lo))
        info=(group.educational>group.educational.quantile(hi))&(group.readability<group.readability.quantile(lo))
        rows.append({"low_quantile":lo,"high_quantile":hi,"any_rate":float((edu|flu|info).mean()),
                     "education_ads_rate":float(edu.mean()),"fluency_cleanliness_rate":float(flu.mean()),"information_readability_rate":float(info.mean())})
    pd.DataFrame(rows).to_csv(T/"conflict_threshold_sensitivity.csv",index=False)
    rows=[]
    base=qs.groupby("domain").Q_baseline.mean()
    for lam in [0,.05,.10,.20]:
        v=np.clip(qs.Q_baseline-lam*qs.conflict_intensity,0,1)
        mean=v.groupby(qs.domain).mean().reindex(base.index)
        rows.append({"lambda":lam,"domain_rank_spearman_vs_no_penalty":spearmanr(base,mean).statistic,
                     "mean_quality_decrease":float((qs.Q_baseline-v).mean())})
    pd.DataFrame(rows).to_csv(T/"conflict_penalty_sensitivity.csv",index=False)
    print(f"Condition number of Spearman matrix: {eig.max()/max(eig.min(),1e-12):.2f}")

if __name__=="__main__":run()
