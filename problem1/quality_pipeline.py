"""Full-record A1-A3 quality audit, scoring, conflict analysis and figures."""
from __future__ import annotations

import json
import lzma
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "real_attachments" / "A_data_value"
OUT = Path(__file__).resolve().parent / "outputs"
TABLES, FIGURES = OUT / "tables", OUT / "figures"
RNG = np.random.default_rng(20260923)

FIELDS = [
    "fineweb_edu", "fluency_en", "modernbert_cleanliness", "modernbert_readability",
    "modernbert_reasoning", "modernbert_professionalism", "dsir_books", "dsir_wiki",
    "dsir_math", "qurater", "ad_en", "rps_doc_word_count",
    "rps_doc_num_sentences", "rps_doc_unigram_entropy", "rps_doc_frac_unique_words",
    "rps_doc_frac_no_alph_words", "rps_doc_frac_chars_top_2gram",
    "rps_doc_frac_chars_top_3gram", "rps_lines_uppercase_letter_fraction",
    "rps_lines_ending_with_terminal_punctution_mark",
    "rps_lines_numerical_chars_fraction", "rps_doc_mean_word_length",
]
GROUPS = {
    "educational": ["fineweb_edu", "modernbert_reasoning", "modernbert_professionalism", "dsir_books", "dsir_wiki", "dsir_math", "qurater"],
    "readability": ["fluency_en", "modernbert_readability", "rps_lines_ending_with_terminal_punctution_mark"],
    "cleanliness": ["modernbert_cleanliness", "ad_en", "rps_doc_frac_no_alph_words", "rps_doc_frac_chars_top_2gram", "rps_doc_frac_chars_top_3gram"],
    "structure": ["rps_doc_word_count", "rps_doc_num_sentences", "rps_doc_unigram_entropy", "rps_doc_frac_unique_words", "rps_lines_uppercase_letter_fraction", "rps_lines_numerical_chars_fraction", "rps_doc_mean_word_length"],
}
DSIR = ["dsir_books", "dsir_wiki", "dsir_math"]
HUMP = {"rps_doc_word_count", "rps_doc_num_sentences", "rps_lines_uppercase_letter_fraction", "rps_lines_numerical_chars_fraction", "rps_doc_mean_word_length"}
NEGATIVE = {"rps_doc_frac_no_alph_words", "rps_doc_frac_chars_top_2gram", "rps_doc_frac_chars_top_3gram"}
LOG = {"rps_doc_word_count", "rps_doc_num_sentences", "rps_doc_frac_chars_top_2gram", "rps_doc_frac_chars_top_3gram"}


def softmax_expected(v):
    a = np.asarray(v, dtype=float)
    if a.shape != (6,) or not np.all(np.isfinite(a)):
        return np.nan
    p = np.exp(a - a.max()); p /= p.sum()
    return float(p @ np.arange(6) / 5)


def binary_prob(v):
    a = np.asarray(v, dtype=float)
    if a.shape != (2,) or not np.all(np.isfinite(a)):
        return np.nan
    d = np.clip(a[1] - a[0], -35, 35)
    return float(1 / (1 + np.exp(-d)))


def scalar(field, v):
    try:
        if field.startswith("modernbert_"):
            return softmax_expected(v)
        if field in ("ad_en", "fluency_en"):
            return binary_prob(v)
        if field == "fineweb_edu":
            return float(v[0]) if isinstance(v, list) and len(v) == 1 else np.nan
        if field == "qurater":
            # Four documented dimensions; equal average is a conservative, explicit assumption.
            return float(np.mean(v)) if isinstance(v, list) and len(v) == 4 else np.nan
        x = float(v)
        return x if np.isfinite(x) else np.nan
    except (TypeError, ValueError, IndexError):
        return np.nan


def input_files():
    return [("A1", DATA / "slimpajama_quality_signal_sample.jsonl.xz", None),
            ("A2", *[(next((DATA / "slimpajama_quality_extended").glob("arxiv_*.xz"))), "arxiv"]),
            ("A3", *[(next((DATA / "slimpajama_quality_extended").glob("github_*.xz"))), "github"]) ]


def load_all():
    records, snippets = [], {}
    counts = []
    for source, path, inferred_domain in input_files():
        n = 0; invalid = 0
        with lzma.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                n += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    invalid += 1
                    continue
                dom = inferred_domain or obj.get("_source_domain") or obj.get("sub_path")
                row = {"source": source, "domain": dom, "id": obj.get("id"), "sub_path": obj.get("sub_path")}
                row.update({k: scalar(k, obj.get(k)) for k in FIELDS})
                records.append(row)
                if source == "A1":
                    snippets[(source, n)] = str(obj.get("content", ""))[:800].replace("\n", " ")
        counts.append({"dataset": source, "file": str(path.relative_to(ROOT)), "records": n, "invalid_json": invalid})
    raw = pd.DataFrame.from_records(records)
    # Position matches the preserved A1 stream order, permitting a small audit excerpt export.
    a1_snips = pd.Series([snippets.get(("A1", i+1), "") for i in range(counts[0]["records"])])
    pd.DataFrame(counts).to_csv(TABLES / "quality_file_counts.csv", index=False)
    return raw, a1_snips


def transform(raw):
    # The same calibration from A1 is applied to A2/A3; A1 contains all seven domains.
    x = raw[FIELDS].astype(float).copy()
    z = pd.DataFrame(index=raw.index)
    rules = []
    calibration = raw.source.eq("A1")
    for field in FIELDS:
        s = x[field].to_numpy(float)
        if field in LOG:
            s = np.sign(s) * np.log1p(np.abs(s))
        ref = s[calibration.to_numpy()]
        ref = ref[np.isfinite(ref)]
        if not len(ref):
            raise ValueError(f"No calibration values for {field}")
        if field in HUMP:
            q = np.quantile(ref, [.01, .20, .80, .99])
            lo, left, right, hi = q
            up = np.clip((s-lo)/max(left-lo, 1e-12), 0, 1)
            down = np.clip((hi-s)/max(hi-right, 1e-12), 0, 1)
            zz = np.minimum(up, down)
            direction = "适中最好；经验梯形效用"
        else:
            lo, hi = np.quantile(ref, [.01, .99])
            zz = np.clip((s-lo)/max(hi-lo, 1e-12), 0, 1)
            if field in NEGATIVE:
                zz = 1-zz
            direction = "越低越好" if field in NEGATIVE else "越高越好"
            q = [lo, hi]
        missing = ~np.isfinite(zz)
        zz[missing] = np.nanmedian(zz[calibration.to_numpy()])
        z[field] = zz
        compress = ("六级 logits 的 softmax 期望/5" if field.startswith("modernbert_") else
                    "两类 logits 中正向类概率" if field in ("ad_en", "fluency_en") else
                    "四个 QuRating 维度均值" if field == "qurater" else
                    "单元素取值" if field == "fineweb_edu" else "原始标量")
        rules.append({"field": field, "raw_type": "list" if field in ["fineweb_edu", "fluency_en", "ad_en", "qurater"] or field.startswith("modernbert_") else "number",
                      "raw_direction": direction, "missing_count": int(x[field].isna().sum()), "missing_rate": float(x[field].isna().mean()),
                      "outlier_below_1pct": int(np.sum(s < np.quantile(ref,.01))), "outlier_above_99pct": int(np.sum(s > np.quantile(ref,.99))),
                      "compression": compress, "skew_transform": "signed_log1p" if field in LOG else "none",
                      "scaling": "A1固定分位点梯形效用" if field in HUMP else "A1固定1%-99%截尾MinMax",
                      "calibration_values": json.dumps([float(v) for v in q]), "direction_unification": direction,
                      "final_meaning": "0-1，越高表示该项质量越好；启发式方向见假设文件"})
    pd.DataFrame(rules).to_csv(TABLES / "quality_indicator_rules.csv", index=False, encoding="utf-8-sig")
    return z


def group_scores(z):
    group = pd.DataFrame({g:z[cols].mean(axis=1) for g,cols in GROUPS.items()})
    # The three DSIR signals are near duplicates; count them as one subdimension.
    group["educational"] = (z[["fineweb_edu","modernbert_reasoning","modernbert_professionalism","qurater"]].sum(axis=1)
                            + z[DSIR].mean(axis=1))/5
    return group


def scores(raw, z):
    group = group_scores(z)
    baseline = group.mean(axis=1).to_numpy()
    corr = z.loc[raw.source.eq("A1")].corr().fillna(0).to_numpy()
    crit = np.nanstd(z.loc[raw.source.eq("A1")].to_numpy(),axis=0) * np.sum(1-corr,axis=0)
    crit = np.maximum(crit,0); crit /= crit.sum()
    equal = np.array([(1/(len(GROUPS)*5*3) if f in DSIR else
                       1/(len(GROUPS)*5) if f in GROUPS["educational"] else
                       1/(len(GROUPS)*len(GROUPS[g])))
                      for f in FIELDS for g in GROUPS if f in GROUPS[g]])
    # robust model clips the least and greatest group score before averaging.
    robust = np.sort(group.to_numpy(),axis=1)[:,1:3].mean(axis=1)
    raw["Q_baseline"] = baseline
    raw["Q_critic"] = z.to_numpy() @ crit
    raw["Q_robust"] = robust
    pd.DataFrame({"field": FIELDS, "group": [next(g for g,v in GROUPS.items() if f in v) for f in FIELDS],
                  "baseline": equal, "critic": crit}).to_csv(TABLES/"quality_weights.csv",index=False,encoding="utf-8-sig")
    return group


def conflicts(raw,z,group):
    a1 = raw.source.eq("A1")
    high = z.loc[a1].quantile(.75); low = z.loc[a1].quantile(.25)
    types = {
        "education_vs_ads": (z.fineweb_edu>high.fineweb_edu) & (z.ad_en<low.ad_en),
        "fluency_vs_cleanliness": (z.fluency_en>high.fluency_en) & (z.modernbert_cleanliness<low.modernbert_cleanliness),
        "importance_vs_readability": (group.educational>group.loc[a1,"educational"].quantile(.75)) & (group.readability<group.loc[a1,"readability"].quantile(.25)),
    }
    for k,v in types.items(): raw[k]=v.to_numpy()
    raw["conflict_any"] = np.column_stack([v.to_numpy() for v in types.values()]).any(axis=1)
    raw["conflict_intensity"] = group.max(axis=1)-group.min(axis=1)
    raw["Q_penalty"] = np.clip(raw.Q_baseline-.10*raw.conflict_intensity,0,1)
    raw["Q_hard"] = raw.Q_baseline * np.where(z.ad_en<z.loc[a1,"ad_en"].quantile(.10),.85,1.0)
    summary=[]
    for (source,dom), d in raw.groupby(["source","domain"]):
        row={"source":source,"domain":dom,"n":len(d),"any_rate":d.conflict_any.mean(),"mean_intensity":d.conflict_intensity.mean()}
        row.update({k+"_rate":d[k].mean() for k in types})
        summary.append(row)
    pd.DataFrame(summary).to_csv(TABLES/"conflict_summary.csv",index=False,encoding="utf-8-sig")
    return list(types)


def domain_summary(raw):
    rows=[]
    for (source,dom), d in raw.groupby(["source","domain"]):
        for method in ["Q_baseline","Q_critic","Q_robust","Q_penalty","Q_hard"]:
            a=d[method].to_numpy()
            # Full-size nonparametric bootstrap: even the 203,752-row A3 domain is resampled at its actual size.
            draws=np.array([np.mean(RNG.choice(a,len(a),replace=True)) for _ in range(200)])
            rows.append({"source":source,"domain":dom,"method":method,"n":len(a),"mean":np.mean(a),"median":np.median(a),
                         "trimmed_mean":np.mean(np.sort(a)[int(.1*len(a)):max(int(.9*len(a)),1)]),
                         "winsor_mean":np.mean(np.clip(a,*np.quantile(a,[.1,.9]))),
                         "ci_low":np.quantile(draws,.025),"ci_high":np.quantile(draws,.975)})
    out=pd.DataFrame(rows)
    out["rank_within_source"] = out.groupby(["source","method"])["mean"].rank(ascending=False,method="min")
    out.to_csv(TABLES/"domain_quality_scores.csv",index=False,encoding="utf-8-sig")
    return out


def figures(raw,z,dom,group):
    a1=z.loc[raw.source.eq("A1")]
    plt.rcParams.update({"font.size":9})
    fig,axes=plt.subplots(4,6,figsize=(17,10))
    for ax,f in zip(axes.flat,FIELDS): ax.hist(a1[f],bins=30,color="#426a91");ax.set_title(f.replace("rps_",""),fontsize=7)
    for ax in list(axes.flat)[len(FIELDS):]: ax.axis("off")
    fig.tight_layout();fig.savefig(FIGURES/"quality_distributions.png",dpi=160);plt.close(fig)
    corr=a1.corr(method="spearman").fillna(0)
    corr.to_csv(TABLES/"quality_spearman.csv",encoding="utf-8-sig")
    fig,ax=plt.subplots(figsize=(11,9));im=ax.imshow(corr,cmap="RdBu_r",vmin=-1,vmax=1)
    ax.set_xticks(range(len(FIELDS)),FIELDS,rotation=90,fontsize=6);ax.set_yticks(range(len(FIELDS)),FIELDS,fontsize=6);fig.colorbar(im,ax=ax)
    fig.tight_layout();fig.savefig(FIGURES/"quality_correlation.png",dpi=170);plt.close(fig)
    X=a1.to_numpy()-a1.to_numpy().mean(axis=0); _,sv,_=np.linalg.svd(X[:15000],full_matrices=False)
    ev=(sv**2)/(sv**2).sum()
    pd.DataFrame({"component":np.arange(1,len(ev)+1),"explained_variance":ev,"cumulative":np.cumsum(ev)}).to_csv(TABLES/"pca_variance.csv",index=False)
    fig,ax=plt.subplots();ax.bar(np.arange(1,len(ev)+1),ev);ax.plot(np.arange(1,len(ev)+1),np.cumsum(ev),color="red");ax.set(xlabel="PCA component",ylabel="Variance ratio");fig.tight_layout();fig.savefig(FIGURES/"pca_variance.png",dpi=170);plt.close(fig)
    dist=np.sqrt(np.maximum(0,2*(1-corr.to_numpy()))); tri=dist[np.triu_indices(len(FIELDS),1)]
    fig,ax=plt.subplots(figsize=(11,6));dendrogram(linkage(tri,method="average"),labels=FIELDS,leaf_rotation=90,ax=ax);fig.tight_layout();fig.savefig(FIGURES/"indicator_clustering.png",dpi=170);plt.close(fig)
    w=pd.read_csv(TABLES/"quality_weights.csv")
    fig,ax=plt.subplots(figsize=(11,5));x=np.arange(len(w));ax.bar(x-.2,w.baseline,.4,label="Group equal");ax.bar(x+.2,w.critic,.4,label="CRITIC");ax.set_xticks(x,w.field,rotation=90,fontsize=6);ax.legend();fig.tight_layout();fig.savefig(FIGURES/"quality_weights.png",dpi=170);plt.close(fig)
    q=dom.query("method=='Q_baseline'")
    fig,ax=plt.subplots(figsize=(9,5));srcs=q.source.unique();domains=sorted(q.domain.unique());
    for i,src in enumerate(srcs):
        a=q[q.source.eq(src)].set_index("domain").reindex(domains)
        ax.errorbar(np.arange(len(domains))+i*.12,a["mean"],yerr=[a["mean"]-a.ci_low,a.ci_high-a["mean"]],fmt="o",label=src,capsize=2)
    ax.set_xticks(np.arange(len(domains)),domains,rotation=35,ha="right");ax.set(ylabel="Quality Q");ax.legend();fig.tight_layout();fig.savefig(FIGURES/"domain_quality_ci.png",dpi=170);plt.close(fig)
    cs=pd.read_csv(TABLES/"conflict_summary.csv")
    pivot=cs.pivot(index="domain",columns="source",values="any_rate")
    fig,ax=plt.subplots(figsize=(7,4));pivot.plot.bar(ax=ax);ax.set(ylabel="Any conflict rate");fig.tight_layout();fig.savefig(FIGURES/"conflict_rates.png",dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5));m=np.array([[np.mean(raw.loc[raw.source.eq("A1"),a]&raw.loc[raw.source.eq("A1"),b]) for b in ["education_vs_ads","fluency_vs_cleanliness","importance_vs_readability"]] for a in ["education_vs_ads","fluency_vs_cleanliness","importance_vs_readability"]]);im=ax.imshow(m,cmap="Blues");ax.set_xticks(range(3),["Edu/ad","Fluent/clean","Info/read"],rotation=30);ax.set_yticks(range(3),["Edu/ad","Fluent/clean","Info/read"]);fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(FIGURES/"conflict_matrix.png",dpi=170);plt.close(fig)
    fig,ax=plt.subplots();ax.hist(raw.loc[raw.source.eq("A1"),"conflict_intensity"],bins=40);ax.set(xlabel="Group range",ylabel="Count");fig.tight_layout();fig.savefig(FIGURES/"conflict_intensity.png",dpi=170);plt.close(fig)


def run():
    TABLES.mkdir(parents=True,exist_ok=True);FIGURES.mkdir(parents=True,exist_ok=True)
    raw,snips=load_all()
    z=transform(raw)
    z.assign(source=raw.source,domain=raw.domain,id=raw.id).to_parquet(TABLES/"quality_standardized.parquet",index=False)
    group=scores(raw,z)
    conflicts(raw,z,group)
    dom=domain_summary(raw)
    raw[["source","domain","id","Q_baseline","Q_critic","Q_robust","Q_penalty","Q_hard","conflict_any","conflict_intensity"]].to_parquet(TABLES/"sample_quality_scores.parquet",index=False)
    # Export only short A1 excerpts, selected by score and conflict for qualitative inspection.
    a1=raw.loc[raw.source.eq("A1")].copy();a1["snippet"]=snips.to_numpy()
    pick=pd.concat([a1.nlargest(10,"Q_baseline"),a1.nsmallest(10,"Q_baseline"),a1[a1.conflict_any].head(20)])
    pick[["id","domain","Q_baseline","conflict_any","snippet"]].to_csv(TABLES/"manual_review_candidates.csv",index=False,encoding="utf-8-sig")
    figures(raw,z,dom,group)
    audit={"records":len(raw),"source_counts":raw.source.value_counts().to_dict(),"domain_counts":{f"{s}/{d}":int(n) for (s,d),n in raw.groupby(["source","domain"]).size().items()},
           "duplicate_id_within_source":int(raw.duplicated(["source","id"]).sum()),"A1_A2_id_overlap":int(len(set(raw.loc[raw.source.eq("A1") & raw.domain.eq("arxiv"),"id"])&set(raw.loc[raw.source.eq("A2"),"id"]))),
           "A1_A3_id_overlap":int(len(set(raw.loc[raw.source.eq("A1") & raw.domain.eq("github"),"id"])&set(raw.loc[raw.source.eq("A3"),"id"])))}
    (TABLES/"quality_audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    print(json.dumps(audit,ensure_ascii=False,default=str))

if __name__=="__main__": run()
