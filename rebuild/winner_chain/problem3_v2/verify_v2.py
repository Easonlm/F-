"""Read-only audit of completed V2 results and V1 integrity."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
sys.dont_write_bytecode=True
from methods import ROOT,Model,solve,total_cost

HERE=Path(__file__).resolve().parent
T=HERE/'outputs'/'tables'

def run():
    checks={}
    def test(name,value):checks[name]=bool(value)
    integrity=json.loads((T/'v1_integrity.json').read_text(encoding='utf-8'))
    v1=ROOT/'problem3'
    current={str(x.relative_to(v1)).replace('\\','/'):hashlib.sha256(x.read_bytes()).hexdigest() for x in v1.rglob('*') if x.is_file() and '__pycache__' not in x.parts}
    test('V1 SHA256 unchanged',integrity['unchanged'] and current==integrity['sha256'])
    required=['lambda_p_extended_sensitivity.csv','lambda_p_scale_scenarios.csv','p_scale_evidence_audit.csv','p_scale_v1_evidence_audit.csv',
              'mixture_support_diagnostics.csv','quality_transition_points_exact.csv',
              'transition_by_context.csv','structural_break_analysis.csv','joint_bootstrap_parameters.csv','joint_bootstrap_optima.csv','allocation_stability.csv',
              'extrapolation_levels.csv','cap_sensitivity_profile.csv','soft_extrapolation_penalty.csv','robust_optimization.csv','resource_consensus.csv','kkt_diagnostics.csv',
              'budget_elimination_verification.csv','high_budget_elasticity.csv']
    test('required tables',all((T/x).exists() for x in required))
    pm=pd.read_csv(T/'mixture_support_diagnostics.csv')
    test('observed p in support',pm.set_index('strategy').loc['P1_observed_support','support_class']=='in-support')
    test('loose p flagged',pm.set_index('strategy').loc['P3_loose_simplex','support_class']=='extrapolative')
    test('lambda zero and full range',set(pd.read_csv(T/'lambda_p_extended_sensitivity.csv').lambda_p)=={0,.25,.5,.75,1,1.25,1.5})
    p2_scale=pd.read_csv(ROOT/'problem2_v2/outputs/tables/p_scale_transfer_from_problem1_v2.csv').sort_values('dataset')
    p3_scale=pd.read_csv(T/'p_scale_evidence_audit.csv').sort_values('dataset')
    test('final P1 V2 scale evidence carried',len(p2_scale)==len(p3_scale)==3 and
         np.allclose(p2_scale.centered_slope.to_numpy(float),p3_scale.centered_slope.to_numpy(float),atol=1e-9))
    tr=pd.read_csv(T/'quality_transition_points_exact.csv');test('six transitions',len(tr)==6 and (tr.status=='crossing').all())
    a=joblib.load(ROOT/'problem2_v2/outputs/models/recommended_model.joblib');m=Model(*list(a['ND_parameters'])+list(a['Q_parameters']))
    active=[]
    for _,r in tr.iterrows():
        below=solve(m,r.C_threshold*.999,.6,4096,r.cost_model,global_search=True)['Q_B']
        above=solve(m,r.C_threshold*1.001,.6,4096,r.cost_model,global_search=True)['Q_B']
        if r.transition=='start':active.append(below<=.6001 and above>.6001)
        else:active.append(below<.9999 and above>=.9999)
    test('transition brackets independently checked',all(active))
    par=pd.read_csv(T/'joint_bootstrap_parameters.csv');opt=pd.read_csv(T/'joint_bootstrap_optima.csv')
    test('joint bootstrap replicate count',len(par)==200 and len(opt)==800 and opt.replicate.nunique()==200)
    test('joint parameter vectors finite',np.isfinite(par[['E','A','alpha','B','beta','G','kappa','eta_N']].to_numpy()).all())
    test('bootstrap budget binding',np.max(abs(opt.budget_ratio-1))<1e-6)
    levels=pd.read_csv(T/'extrapolation_levels.csv')
    test('three extrapolation levels',set(levels.level)=={'E0_empirical','E1_moderate_3x','E2_free_scaling'} and len(levels)==9)
    test('free budget binding',np.max(abs(levels[levels.level=='E2_free_scaling'].budget_ratio-1))<1e-6)
    cap=pd.read_csv(T/'cap_sensitivity_profile.csv')
    cap['cap_factor']=pd.to_numeric(cap.cap_factor,errors='coerce')
    e1=levels[(levels.level=='E1_moderate_3x')&(levels.budget==1e24)].iloc[0]
    cap3=cap[(cap.regime=='capped')&(cap.cap_factor==3)&(cap.budget==1e24)].iloc[0]
    test('cap profile reproduces E1',len(cap)==12 and abs(cap3.loss-e1.loss)<1e-8 and
         abs(cap3.N_B-e1.N_B)<1e-6 and abs(cap3.D_B-e1.D_B)<1e-6)
    robust=pd.read_csv(T/'robust_optimization.csv');rr=robust[robust.policy=='robust_minimax']
    test('robust feasible all scenarios',len(rr)==3 and (rr.worst_cost_ratio<=1+1e-7).all())
    regret_robust=pd.to_numeric(rr.set_index('budget').max_regret)
    regret_nominal=pd.to_numeric(robust[robust.policy=='nominal_moderate_repaired'].set_index('budget').max_regret)
    test('robust regret improves repaired nominal',bool((regret_robust<=regret_nominal+1e-8).all()))
    vv=json.loads((T/'verification_results.json').read_text(encoding='utf-8'))
    test('elimination and KKT',vv['D_elimination_pass'] and vv['KKT_pass'])
    test('four reports',all((HERE/x).exists() for x in ['problem3_v2_report.md','problem3_v2_summary.md','problem3_v1_v2_comparison.md','问题三第二版完整详解.md']))
    result={'passed':sum(checks.values()),'total':len(checks),'all_passed':all(checks.values()),'checks':checks}
    (T/'independent_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    if not result['all_passed']:raise AssertionError([k for k,v in checks.items() if not v])
    return result

if __name__=='__main__':print(run())
