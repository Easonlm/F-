"""Sensitivity and source agreement diagnostics for A1-A3 quality scores."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from quality_pipeline import group_scores

HERE=Path(__file__).resolve().parent
T=HERE/"outputs"/"tables"
RNG=np.random.default_rng(20260923)


def run():
    s=pd.read_parquet(T/"sample_quality_scores.parquet")
    z=pd.read_parquet(T/"quality_standardized.parquet")
    weights=pd.read_csv(T/"quality_weights.csv")
    a=s[s.source.eq("A1")]
    methods=["Q_baseline","Q_critic","Q_robust","Q_penalty","Q_hard"]
    domain_method=a.groupby("domain")[methods].mean()
    correlations=[]
    for m in methods:
        correlations.append({"method":m,"rank_spearman_vs_baseline":float(spearmanr(domain_method.Q_baseline,domain_method[m]).statistic),
                             "mean_absolute_sample_difference":float(np.mean(abs(a.Q_baseline-a[m]))),
                             "range_of_domain_means":float(domain_method[m].max()-domain_method[m].min())})
    pd.DataFrame(correlations).to_csv(T/"quality_model_comparison.csv",index=False)
    rows=[]
    for src,dom in [("A2","arxiv"),("A3","github")]:
        sub=s[(s.source.eq(src))&(s.domain.eq(dom))]
        sampled=a[a.domain.eq(dom)]
        independent=sub[~sub.id.isin(sampled.id)]
        rows.append({"domain":dom,"A1_n":len(sampled),"extended_total_n":len(sub),"extended_nonoverlap_n":len(independent),
                     "A1_mean_Q":sampled.Q_baseline.mean(),"nonoverlap_mean_Q":independent.Q_baseline.mean(),
                     "difference_nonoverlap_minus_A1":independent.Q_baseline.mean()-sampled.Q_baseline.mean()})
    pd.DataFrame(rows).to_csv(T/"quality_extension_agreement.csv",index=False)
    # Group-weight perturbation, preserving within-group equal weights. Not a ground-truth validation.
    fields=weights.field.tolist(); groups=weights.group.tolist(); unique=list(dict.fromkeys(groups))
    X=z.loc[z.source.eq("A1"),fields].to_numpy();dom=z.loc[z.source.eq("A1"),"domain"].to_numpy();names=sorted(set(dom))
    gmean=group_scores(z.loc[z.source.eq("A1"),fields].reset_index(drop=True))[unique].to_numpy();baseline=np.array([gmean[dom==d].mean(axis=0) for d in names])
    ranks=[]
    for _ in range(500):
        gw=RNG.dirichlet(np.full(len(unique),8.0))
        means=baseline@gw
        order=np.argsort(np.argsort(-means))+1
        ranks.append(order)
    ranks=np.array(ranks)
    pd.DataFrame({"domain":names,"baseline_rank":np.argsort(np.argsort(-baseline.mean(axis=1)))+1,
                  "perturbed_rank_mean":ranks.mean(axis=0),"rank_p05":np.quantile(ranks,.05,axis=0),"rank_p95":np.quantile(ranks,.95,axis=0),
                  "p_top3":(ranks<=3).mean(axis=0)}).to_csv(T/"quality_weight_sensitivity.csv",index=False)
    # Mean/median robustness of domain ordering.
    aggs=a.groupby("domain").Q_baseline.agg(["mean","median"])
    result={"records_A1":len(a),"spearman_domain_mean_vs_median":float(spearmanr(aggs["mean"],aggs["median"]).statistic),
            "method_rank":correlations,"extension":rows,
            "limitations":"No independent human-labeled quality ground truth; excerpt inspection is qualitative only."}
    (T/"quality_model_diagnostics.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False))

if __name__=="__main__":run()
