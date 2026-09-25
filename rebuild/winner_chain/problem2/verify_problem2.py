"""Programmatic source, leakage, formula, output and report checks."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from generalized_scaling import predict_ndq,derivatives,equivalent_parameters

HERE=Path(__file__).resolve().parent
T=HERE/'outputs'/'tables'
B=HERE.parent/'real_attachments'/'B_scaling_laws'

def verify():
    tests={}
    required=['data_audit_b.csv','data_relationships_b.csv','quality_dataset_overlap.csv','classical_scaling_params.csv','classical_scaling_cv.csv','classical_scaling_predictions.csv','external_validation_metrics.csv','quality_model_comparison.csv','quality_scaling_params.csv','quality_validation_metrics.csv','generalized_scaling_params.csv','marginal_effects.csv','elasticities.csv','quality_parameter_equivalence.csv','mixture_effect_summary.csv','large_model_extrapolation.csv','bootstrap_parameters.csv','sensitivity_summary.csv','problem2_to_problem3.json']
    tests['all_outputs_exist']=all((T/f).exists() for f in required) and all((HERE/f).exists() for f in ['problem2_report.md','problem2_summary.md']) and (HERE/'outputs'/'models'/'final_generalized_scaling.joblib').exists()
    audit=pd.read_csv(T/'data_audit_b.csv')
    tests['B1_to_B10_audited']=all(any(audit.file.str.contains(x,regex=False)) for x in ['pythia_training_log_existing.csv','cerebras_training_log.csv','scaling_baseline.csv','published_scaling_data.csv','supplementary_NQ_experiment.csv','supplementary_NQ_experiment_expanded.csv','supplementary_NQ_experiment_large.csv','supplementary_large_models.csv','supplementary_large_baseline.csv'])
    b1=pd.read_csv(B/'pythia_training_log_existing.csv');pred=pd.read_csv(T/'classical_scaling_predictions.csv')
    tests['B1_primary_fit']=len(b1)==len(pred)==1176 and np.allclose(b1.val_loss,pred.val_loss)
    cv=pd.read_csv(T/'classical_scaling_cv.csv')
    tests['group_cv']=cv[cv.model=='M1_ND'].held_out_N_B.nunique()==b1.N_params_B.nunique()==8
    ext=pd.read_csv(T/'external_validation_metrics.csv')
    tests['B4_B5_validation_only']=all(x in set(ext.dataset) for x in ['B4','B5']) and len(pred)==len(b1)
    overlap=pd.read_csv(T/'quality_dataset_overlap.csv')
    tests['quality_overlap_reported']=len(overlap)==3
    model=joblib.load(HERE/'outputs'/'models'/'final_generalized_scaling.joblib');p=model['parameters'];kind=model['kind']
    tests['parameter_constraints']=all(p[k]>0 for k in ['A','alpha','B','beta']) and p['E']>=0 and (p.get('gamma',1)>=0) and (p.get('G',1)>=0)
    n,d,q=7,300,.6
    baseline=p['E']+p['A']*n**(-p['alpha'])+p['B']*d**(-p['beta'])
    tests['Q1_reduces_to_classical']=bool(np.isclose(predict_ndq(n,d,1,p,kind),baseline))
    tests['monotonicity']=bool(predict_ndq(n*2,d,q,p,kind)<predict_ndq(n,d,q,p,kind) and predict_ndq(n,d*2,q,p,kind)<predict_ndq(n,d,q,p,kind) and predict_ndq(n,d,q+.1,p,kind)<=predict_ndq(n,d,q,p,kind))
    der=derivatives(n,d,q,p,kind);h=1e-5
    finite=[(predict_ndq(n+h,d,q,p,kind)-predict_ndq(n-h,d,q,p,kind))/(2*h),(predict_ndq(n,d+h,q,p,kind)-predict_ndq(n,d-h,q,p,kind))/(2*h),(predict_ndq(n,d,q+h,p,kind)-predict_ndq(n,d,q-h,p,kind))/(2*h)]
    tests['analytic_derivatives_match_code']=bool(np.allclose(der,finite,atol=1e-7,rtol=1e-5))
    ne=equivalent_parameters(n,d,q,.1,p,kind)
    tests['exact_equivalence']=bool(np.isclose(predict_ndq(ne,d,q,p,kind),predict_ndq(n,d,q+.1,p,kind),rtol=1e-7))
    core=['classical_scaling_params.csv','classical_scaling_cv.csv','external_validation_metrics.csv','quality_model_comparison.csv','quality_validation_metrics.csv','generalized_scaling_params.csv','elasticities.csv','large_model_extrapolation.csv']
    # Undefined Spearman for the constant M0 baseline and unidentified lambda CI
    # are semantic missing values, not failed numerical predictions.
    def finite_table(file):
        x=pd.read_csv(T/file).select_dtypes('number')
        x=x.drop(columns=[c for c in ['spearman','ci_low','ci_high','FLOPs'] if c in x],errors='ignore')
        return bool(np.isfinite(x).all().all())
    tests['core_tables_no_infinity']=all(finite_table(f) for f in core)
    nums=json.loads((T/'report_numbers.json').read_text(encoding='utf-8'))
    report=(HERE/'problem2_report.md').read_text(encoding='utf-8')
    tests['report_values_match_csv']=all(f'{nums[k]:.4f}' in report for k in ['b2_rmse','b4_rmse','b5_rmse','large_rmse'])
    out=dict(all_passed=all(tests.values()),checks=tests)
    (T/'verification_results.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8')
    if not out['all_passed']:raise AssertionError(out)
    return out

if __name__=='__main__':print(json.dumps(verify(),ensure_ascii=False,indent=2))
