"""Machine-readable checks; unsupported long-horizon validation remains an explicit failure."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline import OUT, ROOT, BENCH


def verify():
    def t(name):return pd.read_csv(OUT/name)
    a=t("analysis_dataset.csv");audit=t("data_audit_c.csv");c8=json.loads((OUT/"c8_parse_summary.json").read_text())
    fc=t("frontier_forecast_12m_24m.csv");co=t("scale_technology_contributions.csv");cv=t("loss_benchmark_bridge_cv.csv");bt=t("forecast_backtest.csv")
    short=t("short_term_baseline_comparison.csv");short_summary=t("short_term_baseline_summary.csv")
    origin=pd.to_datetime(a.loc[a.open_w1,"submission_date"]).max()
    checks={}
    def check(label,passed,detail=""):checks[label]={"passed":bool(passed),"detail":str(detail)}
    files=set(audit.file)
    for key,file in [("C1","leaderboard_cleaned.csv"),("C2","leaderboard_enhanced.csv"),("C3","leaderboard_extended_timeseries.csv"),
                     ("C4","epoch_all_ai_models.csv"),("C5","loss_benchmark_bridge.csv"),("C6","loss_benchmark_bridge_expanded.csv"),("C7","model_architecture_metadata.csv")]:check(key+" audited",file in files)
    check("C8 parsed",c8["parsed_models"]>0,c8)
    check("C8 task aggregation",len(t("c8_task_aggregation.csv"))>0)
    check("C8 failures recorded",(OUT/"c8_parse_failures.csv").exists() and len(t("c8_parse_failures.csv"))==c8["failed_files"])
    check("Open weight rule",a.open_w1.sum()>0 and all(a.loc[a.open_w1,"open_weights"]=="Yes"))
    check("Type separated",{"pretrained","chat_finetuned"}.issubset(set(a.model_type)))
    check("Time axis",a.submission_date.notna().sum()>0)
    q=t("quarterly_frontier.csv")
    s=t("frontier_definition_sensitivity.csv")
    strict=s[(s.subset=="open_w1")&(s.window_quarters==1)].sort_values("end_quarter")
    original=q[q.subset=="open_w1"].sort_values("quarter")
    matched=strict.merge(original,left_on="end_quarter",right_on="quarter",suffixes=("_sensitivity","_original"))
    check("Frontier sample sensitivity",len(matched)==len(original) and
          np.allclose(matched.p95_ability_sensitivity,matched.p95_ability_original,atol=1e-10) and
          not bool(strict.iloc[-1].eligible_n5))
    check("Origin from sample",str(origin.date())==fc.forecast_origin_date.iloc[0])
    check("Forecast dates",all(pd.to_datetime(fc.target_date)==pd.to_datetime(fc.forecast_origin_date)+pd.to_timedelta(fc.horizon_months*365.25/12,unit="D")) if False else
          all(pd.to_datetime(row.target_date)==pd.to_datetime(row.forecast_origin_date)+pd.DateOffset(months=int(row.horizon_months)) for _,row in fc.iterrows()))
    z=co[co.window=="observed_span"].iloc[0]
    check("Shapley additivity",abs(z.scale_points+z.technology_points-z.total_points)<1e-8)
    check("Family CV leakage",cv.families_disjoint.all())
    check("Rolling CV leakage",(~bt.future_leakage).all() if len(bt) else False)
    direct=short[short.model=="direct_time"]
    persistence=short[short.model=="persistence"]
    matched=direct.merge(bt,on=["horizon_quarters","horizon_months","origin_quarter","target_quarter"],
                         suffixes=("_short","_original"),validate="one_to_one")
    origins=original[["quarter","p95_ability"]].rename(columns={"quarter":"origin_quarter","p95_ability":"origin_p95"})
    persistence_check=persistence.merge(origins,on="origin_quarter",validate="many_to_one")
    check("Short-term persistence baseline",len(matched)==len(bt)==len(persistence) and
          np.allclose(matched.prediction_short,matched.prediction_original,atol=1e-10) and
          np.allclose(persistence_check.prediction,persistence_check.origin_p95,atol=1e-10) and
          (~short.future_leakage).all())
    qualified_targets=int(persistence.target_n_ge_5.sum())
    check("Short-term sample guard",(short.target_n_ge_5==(short.target_n>=5)).all() and
          int(short_summary[(short_summary.sample_rule=="target_n_ge_5") & (short_summary.model=="persistence")].targets.sum())==qualified_targets,
          f"{qualified_targets} current backtest target(s) have at least five strict open models")
    check("Bridge monotonic",t("loss_benchmark_bridge_models.csv").monotonic.all())
    check("Comparability strata",set(t("loss_benchmark_bridge_models.csv").comparability)=={"weighted","high_only","all_unweighted"})
    check("Forecast bounded",fc[["median_ability","p95_low","p95_high"]].ge(0).all().all() and fc[["median_ability","p95_low","p95_high"]].le(100).all().all())
    check("Forecast anchor sample recorded",fc.ability_anchor_n.notna().all() and
          (fc.ability_anchor_n<5).eq(~fc.ability_anchor_meets_n5.astype(bool)).all())
    check("C4 compute/data/open weights used",a.compute.notna().sum()>0 and a.training_data.notna().sum()>0 and a.open_w1.sum()>0)
    check("P3 V2 interface",t("mechanistic_forecast.csv").regime.nunique()==3 and (ROOT/"problem2_v2/outputs/tables/problem2_to_problem3_v2.json").exists())
    check("Two forecast paths",len(t("data_vs_mechanistic_forecast.csv"))==6)
    check("Reports generated",all((ROOT/"problem4"/f).exists() for f in ["problem4_report.md","problem4_summary.md","问题四完整详解.md"]))
    check("Figures generated",len(list((ROOT/"problem4/outputs/figures").glob("*.png")))>=16)
    check("Forecast finite and ordered",np.isfinite(fc[["median_ability","p95_low","p95_high"]].to_numpy(float)).all() and (fc.p95_low<=fc.p95_high).all())
    check("12 month direct backtest",len(bt[bt.horizon_months==12])>=3,"Unavailable: direct open-weight history has four quarters")
    check("24 month direct backtest",len(bt[bt.horizon_months==24])>=3,"Unavailable: direct open-weight history has four quarters")
    check("Historical decomposition agrees with observed frontier",np.sign(z.total_points)==np.sign(z.observed_frontier_change),"Disagreement makes contribution shares exploratory")
    (OUT/"verification_results.json").write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding="utf-8")
    return checks


if __name__=="__main__":
    r=verify();print(json.dumps({"passed":sum(v["passed"] for v in r.values()),"total":len(r)},ensure_ascii=False))
