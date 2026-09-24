"""Independent numerical and provenance checks for generated problem-3 outputs."""
from __future__ import annotations
import json,sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from core import COST_MODELS,Model,total_cost,cost_gradient,g,dg
from load_inputs import ROOT,TABLES,load_all

def verify():
    artifact,m,b1,b6,c7,p,pcols=load_all();checks={}
    def check(name,condition):checks[name]=bool(condition)
    check('V2 Model C',artifact['Q_model']=='quality_model' and artifact['ND_model']=='classic')
    interface=json.loads((ROOT/'problem2_v2/outputs/tables/problem2_to_problem3_v2.json').read_text(encoding='utf-8'))
    check('V2 interface, no V1 path',interface['model_file'].startswith('problem2_v2/') and 'problem2/outputs' not in json.dumps(interface))
    n,d,q,q0,c=2.,3.,.8,.6,4096
    t,tr,cq,att=total_cost(n,d,q,q0,c,'exponential')
    check('N D raw units',np.isclose(tr,6*(n*1e9)*(d*1e9)))
    check('quality raw D',np.isclose(cq,d*1e9*(g(q,'exponential')-g(q0,'exponential'))))
    check('attention raw ND',np.isclose(att,2e-4*n*d*1e18*c))
    check('critical context',6/2e-4==30000 and np.isclose(att/tr,2e-4*c/6))
    check('cost component sum',np.isclose(t,tr+cq+att))
    for k in COST_MODELS:
        eps=1e-6
        check('quality derivative '+k,np.isclose(dg(q,k),(g(q+eps,k)-g(q-eps,k))/(2*eps),rtol=1e-7))
        cg=cost_gradient(n,d,q,q0,c,k)
        cn=(total_cost(n+eps,d,q,q0,c,k)[0]-total_cost(n-eps,d,q,q0,c,k)[0])/(2*eps)
        cd=(total_cost(n,d+eps,q,q0,c,k)[0]-total_cost(n,d-eps,q,q0,c,k)[0])/(2*eps)
        cq=(total_cost(n,d,q+eps,q0,c,k)[0]-total_cost(n,d,q-eps,q0,c,k)[0])/(2*eps)
        check('cost gradient '+k,np.allclose(cg,[cn,cd,cq],rtol=1e-7))
    grad=m.gradient(n,d,q);eps=1e-6
    num=np.array([(m.loss(n+eps,d,q)-m.loss(n-eps,d,q))/(2*eps),(m.loss(n,d+eps,q)-m.loss(n,d-eps,q))/(2*eps),(m.loss(n,d,q+eps)-m.loss(n,d,q-eps))/(2*eps)])
    check('Model C analytic gradient',np.allclose(grad,num,rtol=1e-7))
    sys.path.insert(0,str(ROOT));from problem2_v2.models import predict_recommended
    check('Model C prediction matches V2',np.isclose(m.loss(n,d,q),predict_recommended(artifact,n,d,q,artifact['reference_p']),atol=1e-10))
    base=pd.read_csv(TABLES/'optimal_allocations.csv');by=pd.read_csv(TABLES/'optimal_allocations_by_cost.csv');ctx=pd.read_csv(TABLES/'context_sensitivity.csv');mix=pd.read_csv(TABLES/'mixture_optimization.csv')
    check('three budgets',set(base.budget.astype(float))=={1e19,1e22,1e24})
    check('three cost functions',set(by.cost_model)==set(COST_MODELS) and len(by)==9)
    check('C7 read',int(pd.read_csv(TABLES/'context_length_summary.csv').iloc[0].rows)==len(c7))
    check('p simplex',all(abs(mix[pcols].sum(axis=1)-1)<1e-8) and (mix[pcols].to_numpy()>=0).all())
    check('Q range',((by.Q_B>=by.Q0-1e-8)&(by.Q_B<=1+1e-8)).all())
    check('budget feasibility',((by.C_total/by.budget)<=1+1e-7).all() and ((ctx.C_total/ctx.budget)<=1+1e-7).all())
    check('budget binding main',((base.C_total/base.budget)>1-1e-6).all())
    check('multi-start stability',bool((by.optimizer_spread<1e-4).all()))
    check('global refinement',bool((by.global_gap.abs()<1e-5).all()))
    marginal=pd.read_csv(TABLES/'marginal_loss_reduction_per_flop.csv')
    check('KKT marginal balance',bool((marginal.relative_N_D_gap<1e-3).all() and marginal.KKT_Q_satisfied.all()))
    check('extrapolation labels',len(pd.read_csv(TABLES/'extrapolation_flags.csv'))>=6)
    tables=list(TABLES.glob('*.csv'))
    def finite_present(path):
        table=pd.read_csv(path,keep_default_na=False)
        if table.isna().any().any():return False
        numeric=table.select_dtypes(include='number')
        values=numeric.to_numpy(dtype=float)
        return bool(np.isfinite(values[~np.isnan(values)]).all())
    check('tables finite',all(finite_present(f) for f in tables))
    report=(ROOT/'problem3/problem3_report.md').read_text(encoding='utf-8')
    check('report matches CSV',all(f'{r.N_B:.4g}' in report and f'{r.D_B:.4g}' in report for _,r in base.iterrows()))
    p4text=(TABLES/'problem3_to_problem4.json').read_text(encoding='utf-8')
    check('next-stage interface', bool(json.loads(p4text)) and not any(x in p4text for x in ['NaN','Infinity']))
    result={'passed':sum(checks.values()),'total':len(checks),'all_passed':all(checks.values()),'checks':checks}
    (TABLES/'verification_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    if not result['all_passed']:raise AssertionError('Problem 3 verification failed: '+str([k for k,v in checks.items() if not v]))
    return result

if __name__=='__main__':print(verify())
