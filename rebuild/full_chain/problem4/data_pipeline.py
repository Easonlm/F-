"""Audited data preparation for Problem 4. All paths are relative to the project root."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "real_attachments" / "C_efficiency_evolution"
OUT = ROOT / "problem4" / "outputs" / "tables"
BENCH = ["IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"]


def save(df, name):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False, encoding="utf-8-sig")
    return df


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).split("/")[-1].lower())


def audit():
    rows = []
    for f in sorted(DATA.glob("*.csv")):
        d = pd.read_csv(f, low_memory=False)
        dates = []
        for col in ["Submission Date", "Publication date", "Epoch_AI_Publication_Date"]:
            if col in d:
                v = pd.to_datetime(d[col], errors="coerce").dropna()
                if len(v): dates.extend([str(v.min().date()), str(v.max().date())])
        ranges = {}
        for col in ["#Params (B)", "Parameters", "Training compute (FLOP)", "Training dataset size (total)", "Average ⬆️", "LB_Average"]:
            if col in d:
                v = pd.to_numeric(d[col], errors="coerce").dropna()
                ranges[col] = [float(v.min()), float(v.max())] if len(v) else []
        rows.append(dict(file=f.name, rows=len(d), columns=len(d.columns), column_names=json.dumps(list(d.columns), ensure_ascii=False),
                         missing_cells=int(d.isna().sum().sum()), duplicates=int(d.duplicated().sum()),
                         earliest_date=min(dates) if dates else "", latest_date=max(dates) if dates else "",
                         numeric_ranges=json.dumps(ranges, ensure_ascii=False),
                         source_counts=json.dumps(d["Source"].value_counts(dropna=False).to_dict() if "Source" in d else {}, ensure_ascii=False),
                         confidence_counts=json.dumps(d["Confidence"].value_counts(dropna=False).to_dict() if "Confidence" in d else {}, ensure_ascii=False)))
    save(pd.DataFrame(rows), "data_audit_c.csv")
    c3 = pd.read_csv(DATA / "leaderboard_extended_timeseries.csv")
    save(c3.groupby(["Year", "Source"], dropna=False).agg(n=("Model", "size"), p95_average=("Average", lambda v: v.quantile(.95))).reset_index(), "c3_timeseries_provenance.csv")
    return {"c1": pd.read_csv(DATA / "leaderboard_cleaned.csv"), "c2": pd.read_csv(DATA / "leaderboard_enhanced.csv"),
            "c3": c3, "c4": pd.read_csv(DATA / "epoch_all_ai_models.csv", low_memory=False),
            "c5": pd.read_csv(DATA / "loss_benchmark_bridge.csv"),
            "c6": pd.read_csv(DATA / "loss_benchmark_bridge_expanded.csv"),
            "c7": pd.read_csv(DATA / "model_architecture_metadata.csv")}


def license_table(c1):
    permissive = {"apache-2.0", "mit", "bsd-3-clause-clear"}
    research = {"cc-by-nc-4.0", "cc-by-nc-sa-4.0", "cc-by-nc-nd-4.0"}
    records = []
    for val, count in c1["Hub License"].fillna("<missing>").value_counts().items():
        cls = "missing" if val == "<missing>" else "permissive" if val in permissive else "research-only" if val in research else "unclear" if val in {"other", "unknown"} else "custom"
        records.append({"license": val, "classification": cls, "count": count,
                        "note": "Label-based research filter; legal permissions require reading license text."})
    return save(pd.DataFrame(records), "license_classification.csv")


def match_entities(c1, c2, c4):
    """Accept owner-verified direct model matches; keep inherited/fuzzy candidates for audit only."""
    lookup = defaultdict(list)
    for i, row in c4.iterrows():
        lookup[norm(row["Model"])].append(i)
    out = []
    for i, row in c1.iterrows():
        name = str(row["Model"])
        owner = name.split("/")[0].lower() if "/" in name else ""
        cand = lookup.get(norm(name), [])
        best = None
        for j in cand:
            z = c4.iloc[j]
            dev = str(z.get("Hugging Face developer id", "")).lower()
            if dev == owner and owner:
                best = (j, "normalized_exact_owner", .99, True)
                break
        if best is None and cand:
            best = (cand[0], "normalized_exact_owner_mismatch", .35, False)
        if best is None:
            best = (None, "unmatched", 0., False)
        j, method, conf, accepted = best
        z = c4.iloc[j] if j is not None else None
        n1 = pd.to_numeric(row["#Params (B)"], errors="coerce")
        n2 = pd.to_numeric(z["Parameters"], errors="coerce") / 1e9 if z is not None else np.nan
        dt1 = pd.to_datetime(row["Submission Date"], errors="coerce")
        dt2 = pd.to_datetime(z["Publication date"], errors="coerce") if z is not None else pd.NaT
        pdiff = abs(n1-n2)/max(n1,n2) if np.isfinite(n1) and np.isfinite(n2) and max(n1,n2)>0 else np.nan
        ddiff = (dt1-dt2).days if pd.notna(dt1) and pd.notna(dt2) else np.nan
        if accepted and np.isfinite(pdiff) and pdiff>.35: accepted=False; method="parameter_conflict";conf=.2
        if accepted and np.isfinite(ddiff) and ddiff < -30: accepted=False; method="publication_after_submission";conf=.2
        if accepted and z["Confidence"] == "Speculative": accepted=False; method="speculative_metadata";conf=.2
        out.append(dict(leaderboard_row=i, leaderboard_model=name, epoch_row=j if j is not None else np.nan,
                        epoch_model=z["Model"] if z is not None else "", method=method, confidence=conf,
                        parameter_difference=pdiff, date_difference=ddiff, accepted=accepted,
                        c2_open_weights=c2.iloc[i]["Epoch_AI_Open_Weights"],
                        c2_matched=c2.iloc[i]["Epoch_AI_Publication_Date"]==c2.iloc[i]["Epoch_AI_Publication_Date"]))
    return save(pd.DataFrame(out), "model_entity_matches.csv")


def parse_c8():
    """Take newest parseable JSON per model and use leaf task + fixed metric pairs."""
    all_files = list((DATA / "detailed_results").rglob("*.json"))
    by_dir = defaultdict(list)
    for f in all_files: by_dir[f.parent.name].append(f)
    failures, selected, task_rows = [], [], []
    for folder, files in by_dir.items():
        choice = None
        for f in sorted(files, key=lambda p: p.name, reverse=True):
            try:
                x = json.loads(f.read_text(encoding="utf-8"))
                if not isinstance(x.get("results"), dict): raise ValueError("no results object")
                choice = (f,x)
                break
            except Exception as exc:
                failures.append({"folder":folder,"file":str(f.relative_to(DATA)),"error":str(exc)[:250]})
        if choice is None: continue
        f,x = choice
        selected.append({"folder":folder,"file":str(f.relative_to(DATA)),"model_name":x.get("model_name", ""),"result_date":x.get("date", ""),"tasks":len(x["results"])})
        results = x["results"]
        names = list(results)
        for task, vals in results.items():
            if not task.startswith("leaderboard_") or not isinstance(vals,dict): continue
            # A parent has children with a longer task prefix and is counted only once at leaf level.
            if any(other.startswith(task+"_") for other in names): continue
            family = task.split("_")[1]
            if family == "bbh": metric="acc_norm,none"
            elif family == "math": metric="exact_match,none"
            elif family == "ifeval": metric="inst_level_strict_acc,none"
            elif family in {"gpqa","musr"}: metric="acc_norm,none"
            elif family == "mmlu": metric="acc,none"
            else: continue
            value = vals.get(metric)
            if isinstance(value,(int,float)) and np.isfinite(value) and 0 <= value <= 1:
                task_rows.append({"folder":folder,"task":task,"family":family,"metric":metric,"score":float(value)})
    save(pd.DataFrame(failures,columns=["folder","file","error"]), "c8_parse_failures.csv")
    save(pd.DataFrame(selected), "c8_selected_files.csv")
    raw = pd.DataFrame(task_rows)
    if raw.empty: raise ValueError("C8 task parser produced no leaf tasks")
    counts = raw.groupby(["task","metric"]).size()
    keep = counts[counts>=50].index
    raw = raw.set_index(["task","metric"]).loc[keep].reset_index()
    raw["percentile"] = raw.groupby(["task","metric"])["score"].rank(pct=True)
    agg = raw.groupby("folder").agg(detailed_ability=("percentile","mean"),weak_ability=("percentile",lambda s:s.quantile(.25)),task_count=("task","nunique"),family_count=("family","nunique")).reset_index()
    agg["coverage"] = agg.task_count / raw.task.nunique()
    agg["eligible"] = (agg.coverage>=.5)&(agg.family_count>=3)
    save(raw, "c8_task_scores_long.csv")
    save(agg, "c8_task_aggregation.csv")
    save(raw.groupby(["family","task"]).agg(n=("folder","nunique"),mean_score=("score","mean"),p95_score=("score",lambda x:x.quantile(.95))).reset_index(), "c8_task_summary.csv")
    summary = {"json_files":len(all_files),"model_directories":len(by_dir),"parsed_models":len(selected),"failed_models":len(by_dir)-len(selected),"failed_files":len(failures),"usable_task_model_pairs":len(raw),"eligible_aggregates":int(agg.eligible.sum()),"task_count":int(raw.task.nunique())}
    (OUT / "c8_parse_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return agg,raw,summary


def prepare(inputs):
    c1,c2,c4=inputs["c1"],inputs["c2"],inputs["c4"]
    license_table(c1)
    matches=match_entities(c1,c2,c4)
    a=c1.copy().reset_index(names="leaderboard_row")
    a["submission_date"]=pd.to_datetime(a["Submission Date"],errors="coerce")
    a["ability"]=a[BENCH].mean(axis=1)
    a["complete"]=a[BENCH].notna().all(axis=1)
    med=a[BENCH].median();iqr=a[BENCH].quantile(.75)-a[BENCH].quantile(.25)
    a["robust_ability"]=((a[BENCH]-med)/iqr.replace(0,np.nan)).mean(axis=1)
    a["pca_ability"]=PCA(n_components=1).fit_transform(StandardScaler().fit_transform(a[BENCH])).ravel()
    a["model_type"]=np.select([a.Type.str.contains("pretrained",case=False,na=False),a.Type.str.contains("chat|fine-tuned",case=False,na=False)], ["pretrained","chat_finetuned"],default="other")
    a["license_class"] = a["Hub License"].fillna("<missing>").map(dict(zip(pd.read_csv(OUT/"license_classification.csv").license,pd.read_csv(OUT/"license_classification.csv").classification)))
    a=a.merge(matches,on=["leaderboard_row"],how="left",suffixes=("","_match"))
    for col,new in [("Parameters","epoch_params"),("Training compute (FLOP)","compute"),("Training dataset size (total)","training_data"),("Open model weights?","open_weights"),("Publication date","release_date"),("Organization","organization"),("Confidence","epoch_confidence")]:
        a[new]=a.epoch_row.map(c4[col])
    for col in ["epoch_params","compute","training_data"]:a[col]=pd.to_numeric(a[col],errors="coerce")
    a.loc[~a.accepted,["epoch_params","compute","training_data","open_weights","release_date","organization","epoch_confidence"]]=np.nan
    a["compute_source"]=np.where(a.compute.notna(),"reported",np.where(a.epoch_params.notna()&a.training_data.notna(),"proxy","missing"))
    proxy=6*a.epoch_params*a.training_data
    a["compute_proxy"]=np.where(a.compute.notna(),a.compute,proxy)
    a["open_w1"]=a.open_weights.eq("Yes")
    a["open_w2"]=a.open_w1&a.license_class.isin(["permissive","research-only"])
    a["date_diff_days"]=(a.submission_date-pd.to_datetime(a.release_date,errors="coerce")).dt.days
    save(a[["Model","ability","robust_ability","pca_ability","complete","model_type","submission_date","open_w1","open_w2"]],"ability_index_comparison.csv")
    save(a[["Model","accepted","method","open_weights","open_w1","open_w2","license_class","model_type","compute_source","submission_date","release_date","date_diff_days"]],"open_source_filter_audit.csv")
    save(a,"analysis_dataset.csv")
    return a


def attach_c8(a,agg,raw):
    x=a[["Model","ability","submission_date"]].copy()
    x["folder"]=x.Model.str.replace("/","_",regex=False)
    x=x.sort_values(["submission_date","ability"]).drop_duplicates("folder",keep="last")
    v=x.merge(agg,on="folder",how="inner")
    v=v[v.eligible]
    v["rank_difference"]=(v.ability.rank(pct=True)-v.detailed_ability.rank(pct=True)).abs()
    save(v,"c8_vs_leaderboard.csv")
    raw2=raw.merge(x,on="folder",how="inner").dropna(subset=["submission_date"])
    raw2["quarter"]=raw2.submission_date.dt.to_period("Q").astype(str)
    f=raw2.groupby(["family","quarter"]).agg(n=("folder","nunique"),p95=("score",lambda s:s.quantile(.95)),median=("score","median")).reset_index()
    save(f,"c8_task_family_frontiers.csv")
    return v,f
