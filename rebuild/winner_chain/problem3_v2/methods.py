"""Evidence-oriented methods for Problem 3 V2; imports V1 math read-only."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution, minimize, minimize_scalar, brentq

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'problem3'))
from core import Model,COST_MODELS,g,dg,total_cost,cost_gradient,d_from_budget,solve


def capped_solve(model,budget,q0,ctx,kind,n_bounds,d_bounds,global_search=True):
    """Exact hard N,D caps: at fixed N,Q, choose maximal feasible D."""
    nlo,nhi=n_bounds;dlo,dhi=d_bounds
    def values(x):
        n=np.exp(x[0]);q=x[1]
        d=min(dhi,d_from_budget(budget,n,q,q0,ctx,kind))
        return n,d,q
    def objective(x):
        n,d,q=values(x)
        if d<dlo:return 1e6+(dlo-d)*1e3
        return model.loss(n,d,q)
    bounds=[tuple(np.log(n_bounds)),(q0,1.)]
    starts=[[np.log(nlo),q0],[np.log(np.sqrt(nlo*nhi)),min(1.,q0+.2)],[np.log(nhi),1.]]
    if global_search:
        de=differential_evolution(objective,bounds,seed=20260924,popsize=12,maxiter=120,tol=1e-10,polish=False)
        starts.append(de.x)
    res=[minimize(objective,x,method='SLSQP',bounds=bounds,options={'ftol':1e-12,'maxiter':500}) for x in starts]
    for q in (q0,1.):
        r=minimize_scalar(lambda z:objective([z,q]),bounds=bounds[0],method='bounded',options={'xatol':1e-11})
        res.append(type('R',(),{'x':np.array([r.x,q]),'fun':r.fun})())
    best=min(res,key=lambda r:r.fun);n,d,q=values(best.x)
    total,train,quality,attention=total_cost(n,d,q,q0,ctx,kind)
    if d<dlo-1e-8 or total>budget*(1+1e-8):raise RuntimeError('capped solve infeasible')
    return dict(budget=budget,N_B=n,D_B=d,Q_B=q,loss=model.loss(n,d,q),C_total=total,C_train=train,C_Q=quality,C_attn=attention,
                budget_ratio=total/budget,s_train=train/budget,s_Q=quality/budget,s_attn=attention/budget,
                N_cap_active=n>=nhi*(1-1e-5),D_cap_active=d>=dhi*(1-1e-5))


def crossing(model,kind,q0,ctx,which,scan_log=(15,25),eps_q=1e-4):
    """First active-set crossing, log-budget bisection, with explicit absence states."""
    powers=np.linspace(*scan_log,101)
    values=[]
    for z in powers:
        r=solve(model,10.**z,q0,ctx,kind)
        active=(r['Q_B']>q0+eps_q) if which=='start' else (r['Q_B']>=1-eps_q)
        values.append(active)
    def global_active(j):
        r=solve(model,10.**powers[j],q0,ctx,kind,global_search=True)
        return (r['Q_B']>q0+eps_q) if which=='start' else (r['Q_B']>=1-eps_q)
    # A local scan only proposes a bracket. Verify both sides with a global
    # search; a local endpoint solution can otherwise create a false crossing.
    i=next((j for j,v in enumerate(values) if v),len(values)-1)
    while i<len(powers) and not global_active(i):i+=1
    if i>=len(powers):return dict(status='above_scan',C_threshold=np.nan,C_lower=10.**powers[-1],C_upper=np.nan,log10_width=np.nan)
    while i>0 and global_active(i-1):i-=1
    if i==0:return dict(status='below_scan',C_threshold=np.nan,C_lower=np.nan,C_upper=10.**powers[0],log10_width=np.nan)
    a,b=float(powers[i-1]),float(powers[i])
    for _ in range(30):
        m=(a+b)/2;r=solve(model,10.**m,q0,ctx,kind,global_search=True)
        active=(r['Q_B']>q0+eps_q) if which=='start' else (r['Q_B']>=1-eps_q)
        if active:b=m
        else:a=m
        if b-a<2e-6:break
    return dict(status='crossing',C_threshold=10.**b,C_lower=10.**a,C_upper=10.**b,log10_width=b-a)


def segmented_breaks(x,y,max_breaks=2,min_gap=5):
    """Continuous hinge regression, BIC counts break locations as parameters."""
    n=len(x)
    def fit(indices):
        X=[np.ones(n),x]
        for j in indices:X.append(np.maximum(0.,x-x[j]))
        X=np.column_stack(X);coef=np.linalg.lstsq(X,y,rcond=None)[0];rss=max(float(np.sum((y-X@coef)**2)),1e-20)
        k=2+2*len(indices)
        return dict(break_count=len(indices),indices=indices,rss=rss,bic=n*np.log(rss/n)+k*np.log(n),aic=n*np.log(rss/n)+2*k,coefficients=coef)
    allfits=[fit(())]
    candidates=list(range(min_gap,n-min_gap))
    allfits.extend(fit((i,)) for i in candidates)
    if max_breaks>=2:
        allfits.extend(fit((i,j)) for i in candidates for j in candidates if j>=i+min_gap)
    best=min(allfits,key=lambda z:z['bic'])
    base=allfits[0]
    return best,base


def kkt_diagnostic(model,row,q0,ctx,kind,h=1e-6):
    n,d,q=row['N_B'],row['D_B'],row['Q_B']
    gl=model.gradient(n,d,min(q,1-1e-12));gc=cost_gradient(n,d,q,q0,ctx,kind)
    rN=-gl[0]/gc[0];rD=-gl[1]/gc[1]
    if q>=1-1e-5:
        dL=(model.loss(n,d,1.)-model.loss(n,d,1.-h))/h;state='upper'
    elif q<=q0+1e-5:
        dL=(model.loss(n,d,q+h)-model.loss(n,d,q))/h;state='lower'
    else:
        dL=gl[2];state='interior'
    rQ=-dL/gc[2]
    if state=='interior':
        rr=np.array([rN,rD,rQ]);err=(rr.max()-rr.min())/np.mean(np.abs(rr));okay=err<1e-3
    elif state=='upper':err=abs(rN-rD)/np.mean([rN,rD]);okay=rQ>=min(rN,rD)*(1-1e-3)
    else:err=abs(rN-rD)/np.mean([rN,rD]);okay=rQ<=max(rN,rD)*(1+1e-3)
    return dict(r_N=rN,r_D=rD,r_Q=rQ,Q_state=state,KKT_rel_error=err,Q_complementarity_ok=bool(okay),budget_ratio=row['budget_ratio'])
