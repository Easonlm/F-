"""Read-only P1/P2-V2/P3-V2 mechanism connection to the C6 bridge."""
from __future__ import annotations

import json
import sys

import joblib
import numpy as np
import pandas as pd

from data_pipeline import ROOT, OUT, save
from bridge_model import predict

sys.path.insert(0,str(ROOT/"problem3_v2"))
from methods import Model, capped_solve, solve


def run_mechanism(forecasts,bridge):
    interface=json.loads((ROOT/"problem2_v2/outputs/tables/problem2_to_problem3_v2.json").read_text(encoding="utf-8"))
    if interface["model"]!="Model C: classic + quality_model":raise ValueError("V2 Model C interface missing")
    artifact=joblib.load(ROOT/"problem2_v2/outputs/models/recommended_model.joblib")
    vals=interface["parameters"]
    parameter_names=("E","A","alpha","B","beta","G","kappa","eta_N")
    artifact_parameters=np.asarray([*artifact["ND_parameters"],*artifact["Q_parameters"]],dtype=float)
    interface_parameters=np.asarray([vals[name] for name in parameter_names],dtype=float)
    if artifact["ND_model"]!="classic" or artifact["Q_model"]!="quality_model" or not np.allclose(
        artifact_parameters,interface_parameters,rtol=1e-12,atol=1e-12
    ) or not np.isclose(float(artifact["lambda_p"]),float(interface["lambda_p_default"]),rtol=1e-12,atol=1e-12):
        raise ValueError("P2 artifact and P3/P4 interface differ; regenerate the interface before forecasting")
    model=Model(**vals)
    mix=pd.read_csv(ROOT/"problem3_v2/outputs/tables/mixture_support_diagnostics.csv")
    p1=float(mix.loc[mix.strategy=="P1_observed_support","delta_p"].iloc[0])
    cv=pd.read_csv(OUT/"loss_benchmark_bridge_cv.csv")
    cc=cv[(cv.source=="C6")&(cv.comparability=="weighted")&(cv.target=="LB_Average")&(cv.kind=="logistic")]
    bridge_rmse=float(np.sqrt(np.average(cc.rmse**2,weights=cc.n_test)))
    nlo,nhi=interface["B1_N_B_range"];dlo,dhi=interface["B1_D_B_range"]
    scenario=forecasts[(forecasts.tech_scenario=="continuation")].copy()
    rows=[]
    for _,r in scenario.iterrows():
        budget=float(r.predicted_compute)
        for regime in ["E0_empirical","E1_moderate_3x","E2_free_scaling"]:
            try:
                if regime=="E0_empirical":res=capped_solve(model,budget,.6,4096,"exponential",(nlo,nhi),(dlo,dhi),global_search=False)
                elif regime=="E1_moderate_3x":res=capped_solve(model,budget,.6,4096,"exponential",(nlo,3*nhi),(dlo,3*dhi),global_search=False)
                else:res=solve(model,budget,.6,4096,"exponential",n_bounds=(nlo,1e4),d_bounds=(dlo,1e5),global_search=False)
                loss=float(res["loss"])
                for p_scenario,delta in [("P0_reference",0.),("P1_observed_support_lambda0.25",.25*p1)]:
                    adjusted=loss+delta
                    benchmark=float(predict(bridge,[adjusted])[0])
                    rows.append(dict(horizon_months=r.horizon_months,target_date=r.target_date,compute_scenario=r.compute_scenario,regime=regime,p_scenario=p_scenario,
                                     compute_budget=budget,N_B=res["N_B"],D_B=res["D_B"],Q_B=res["Q_B"],
                                     loss=adjusted,benchmark=benchmark,bridge_cv_rmse=bridge_rmse,
                                     bridge_error_band_low=max(0.,benchmark-1.96*bridge_rmse),bridge_error_band_high=min(100.,benchmark+1.96*bridge_rmse),
                                     bridge_loss_range_status="inside" if 1.65<=adjusted<=2.84 else "outside_C6_range",
                                     extrapolation="strong" if regime=="E2_free_scaling" or budget>1e24 else "moderate_or_supported"))
            except Exception as e:
                rows.append(dict(horizon_months=r.horizon_months,target_date=r.target_date,compute_scenario=r.compute_scenario,regime=regime,error=str(e)))
    out=save(pd.DataFrame(rows),"mechanistic_forecast.csv")
    m=out[(out.regime=="E1_moderate_3x")&(out.p_scenario=="P0_reference")][["horizon_months","compute_scenario","benchmark","loss"]]
    j=scenario.merge(m,on=["horizon_months","compute_scenario"],how="left")
    j["difference_data_minus_mechanism"]=j.median_ability-j.benchmark
    save(j,"data_vs_mechanistic_forecast.csv")
    return out,j
