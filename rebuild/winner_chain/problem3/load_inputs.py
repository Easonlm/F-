"""Audited inputs from problem 1, problem 2 V2, B1/B6, and C7."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from core import Model

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
TABLES=HERE/'outputs'/'tables'
FIGURES=HERE/'outputs'/'figures'


def load_all():
    for p in (TABLES,FIGURES,HERE/'outputs'/'models'):p.mkdir(parents=True,exist_ok=True)
    artifact=joblib.load(ROOT/'problem2_v2/outputs/models/recommended_model.joblib')
    selection=json.loads((ROOT/'problem2_v2/outputs/tables/selection.json').read_text(encoding='utf-8'))
    assert artifact['ND_model']=='classic' and artifact['Q_model']=='quality_model' and selection['model_C']=='classic + quality_model'
    model=Model(*map(float,[*artifact['ND_parameters'],*artifact['Q_parameters']]))
    b1=pd.read_csv(ROOT/'real_attachments/B_scaling_laws/pythia_training_log_existing.csv')
    b6=pd.read_csv(ROOT/'real_attachments/B_scaling_laws/supplementary_NQ_experiment.csv')
    c7=pd.read_csv(ROOT/'real_attachments/C_efficiency_evolution/model_architecture_metadata.csv')
    train=pd.read_csv(ROOT/'real_attachments/A_data_value/regmix_tables/train_mixture_1m.csv')
    pcols=artifact['mixture_model']['pcols']; p=train[pcols].to_numpy(float);p=p/p.sum(axis=1,keepdims=True)
    return artifact,model,b1,b6,c7,p,pcols


def write_interface(artifact,model,b1,b6):
    vals=dict(E=model.E,A=model.A,alpha=model.alpha,B=model.B,beta=model.beta,G=model.G,kappa=model.kappa,eta_N=model.eta_N)
    obj={'model':'Model C: classic + quality_model','parameters':vals,'lambda_p_default':artifact['lambda_p'],
         'Q_B_range':[float(b6.Q_score.min()),float(b6.Q_score.max())],
         'B1_N_B_range':[float(b1.N_params_B.min()),float(b1.N_params_B.max())],
         'B1_D_B_range':[float(b1.D_tokens_B.min()),float(b1.D_tokens_B.max())],
         'p_correction':'lambda_p * (mean mixture_model prediction at p - mean prediction at reference_p)',
         'reference_p':artifact['reference_p'].tolist(),
         'mixture_model_file':'problem1/outputs/models/mixture_v2.joblib (embedded in recommended_model.joblib)',
         'model_file':'problem2_v2/outputs/models/recommended_model.joblib',
         'prediction_function':'problem2_v2.models.predict_recommended',
         'derivatives':{'dL_dN_B':'-alpha*A*N_B^(-alpha-1)-eta_N*G*(1-Q_B)^kappa*N_B^(-eta_N-1)',
                        'dL_dD_B':'-beta*B*D_B^(-beta-1)',
                        'dL_dQ_B':'-kappa*G*(1-Q_B)^(kappa-1)*N_B^(-eta_N)'},
         'warnings':['B1 support is bounded; extrapolation is not validation','Q_A to Q_B mapping is not identified','lambda_p is not jointly identified']}
    target=ROOT/'problem2_v2/outputs/tables/problem2_to_problem3_v2.json'
    target.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=[]
    for n,d,q in [(0.1,1,0.2),(1,50,0.6),(10,300,0.9)]:
        grad=model.gradient(n,d,q)
        rows.append(dict(N_B=n,D_B=d,Q_B=q,dL_dN_B=grad[0],dL_dD_B=grad[1],dL_dQ_B=grad[2]))
    pd.DataFrame(rows).to_csv(ROOT/'problem2_v2/outputs/tables/model_c_derivatives.csv',index=False)
    return obj


def context_summary(c7):
    s=pd.to_numeric(c7.max_position_embeddings,errors='coerce').dropna().astype(int)
    unique=np.sort(s.unique());low=unique[unique<30000];high=unique[unique>30000]
    scenarios={'short':int(s.quantile(.25)),'medium':int(s.median()),'long':int(s.quantile(.75))}
    summary=dict(rows=len(c7),valid=len(s),missing=len(c7)-len(s),minimum=int(s.min()),median=float(s.median()),
                 p10=float(s.quantile(.1)),p25=float(s.quantile(.25)),p75=float(s.quantile(.75)),p90=float(s.quantile(.9)),
                 p95=float(s.quantile(.95)),maximum=int(s.max()),unique_values=','.join(map(str,unique)),
                 modes=','.join(f'{k}:{v}' for k,v in s.value_counts().items()),
                 below_critical=int((s<30000).sum()),above_critical=int((s>30000).sum()),
                 closest_below=int(low[-1]) if len(low) else np.nan,closest_above=int(high[0]) if len(high) else np.nan,
                 outlier_iqr_count=int(((s<s.quantile(.25)-1.5*(s.quantile(.75)-s.quantile(.25))) | (s>s.quantile(.75)+1.5*(s.quantile(.75)-s.quantile(.25)))).sum()))
    pd.DataFrame([summary]).to_csv(TABLES/'context_length_summary.csv',index=False)
    return scenarios,summary
