"""Problem 3: Model C, FLOP accounting, and reduced constrained optimization.

N and D are in billions in the predictive model. All costs are in raw FLOPs.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import differential_evolution, minimize

SCALE = 1e9
ETA_ATTN = 2e-4
COST_MODELS = ('exponential', 'power', 'logarithmic')


def g(q, kind):
    if kind == 'exponential': return 1e7 * np.exp(6*q)
    if kind == 'power': return 5e9 * q**4
    if kind == 'logarithmic': return 2e9 * np.log1p(10*q)
    raise ValueError(kind)


def dg(q, kind):
    if kind == 'exponential': return 6e7 * np.exp(6*q)
    if kind == 'power': return 2e10 * q**3
    if kind == 'logarithmic': return 2e10 / (1+10*q)
    raise ValueError(kind)


def total_cost(n, d, q, q0, ctx, kind):
    """Return total, train, quality, attention FLOPs, in this order."""
    train = 6e18*n*d
    quality = 1e9*d*max(0., g(q,kind)-g(q0,kind))
    attention = ETA_ATTN*1e18*n*d*ctx
    return train+quality+attention,train,quality,attention


def cost_gradient(n,d,q,q0,ctx,kind):
    base=(6+ETA_ATTN*ctx)*1e18
    surcharge=1e9*max(0.,g(q,kind)-g(q0,kind))
    return np.array([base*d, base*n+surcharge, 1e9*d*dg(q,kind)])


@dataclass(frozen=True)
class Model:
    E: float
    A: float
    alpha: float
    B: float
    beta: float
    G: float
    kappa: float
    eta_N: float

    def loss(self,n,d,q,delta_p=0.):
        return self.E+self.A*n**(-self.alpha)+self.B*d**(-self.beta)+self.G*(1-q)**self.kappa*n**(-self.eta_N)+delta_p

    def gradient(self,n,d,q):
        dn=-self.alpha*self.A*n**(-self.alpha-1)-self.eta_N*self.G*(1-q)**self.kappa*n**(-self.eta_N-1)
        dd=-self.beta*self.B*d**(-self.beta-1)
        dq=-self.kappa*self.G*(1-q)**(self.kappa-1)*n**(-self.eta_N) if q<1 else -np.inf
        return np.array([dn,dd,dq])


def d_from_budget(budget,n,q,q0,ctx,kind):
    return budget/((6+ETA_ATTN*ctx)*1e18*n+1e9*max(0.,g(q,kind)-g(q0,kind)))


def solve(model,budget,q0,ctx,kind,n_bounds=(1e-3,1e6),d_bounds=(1e-3,1e7),global_search=False,seed=2026):
    """Eliminate D with binding budget, then optimize log N and bounded Q.

    Differential evolution generates a global candidate for main scenarios;
    several SLSQP starts check stability for every scenario.
    """
    lo,hi=np.log(n_bounds)
    qlo,qhi=q0,1.
    def objective(x):
        n=np.exp(x[0]);q=x[1]
        d=d_from_budget(budget,n,q,q0,ctx,kind)
        if d<d_bounds[0]: return model.loss(n,d_bounds[0],q)+1e3*(d_bounds[0]-d)
        if d>d_bounds[1]: return model.loss(n,d_bounds[1],q)+1e3*(d-d_bounds[1])/d_bounds[1]
        return model.loss(n,d,q)
    bounds=[(lo,hi),(qlo,qhi)]
    starts=[[np.log(np.sqrt(n_bounds[0]*n_bounds[1])),q0],
            [np.log(max(n_bounds[0],min(n_bounds[1],budget/1e21))),min(1.,q0+.2)],
            [np.log(max(n_bounds[0],min(n_bounds[1],np.sqrt(budget/1e19)))),1.]]
    de_result=None
    if global_search:
        de_result=differential_evolution(objective,bounds,seed=seed,popsize=12,maxiter=100,tol=1e-10,polish=False)
        starts.append(de_result.x)
    candidates=[]
    local_values=[]
    for x in starts:
        r=minimize(objective,x,method='SLSQP',bounds=bounds,options={'ftol':1e-12,'maxiter':400})
        candidates.append(r)
        local_values.append(float(r.fun))
    if de_result is not None: candidates.append(de_result)
    # Explicit boundary search protects the Q=1 cusp and Q=Q0 active set.
    from scipy.optimize import minimize_scalar
    for q in [qlo,qhi]:
        r=minimize_scalar(lambda z:objective([z,q]),bounds=(lo,hi),method='bounded',options={'xatol':1e-11})
        candidates.append(type('Boundary',(),{'x':np.array([r.x,q]),'fun':r.fun,'success':r.success})())
    candidates=sorted(candidates,key=lambda r:r.fun)
    best=candidates[0];n=float(np.exp(best.x[0]));q=float(best.x[1]);d=float(d_from_budget(budget,n,q,q0,ctx,kind))
    total,train,quality,attention=total_cost(n,d,q,q0,ctx,kind)
    vals=[float(r.fun) for r in candidates]
    local=local_values
    return dict(budget=budget,N_B=n,D_B=d,Q_B=q,loss=model.loss(n,d,q),
                C_total=total,C_train=train,C_Q=quality,C_attn=attention,
                s_train=train/budget,s_Q=quality/budget,s_attn=attention/budget,
                budget_ratio=total/budget,optimizer_spread=max(local)-min(local),
                global_checked=global_search,global_gap=(min(vals)-de_result.fun if de_result is not None else np.nan),
                N_at_lower_bound=np.isclose(n,n_bounds[0],rtol=1e-4),N_at_upper_bound=np.isclose(n,n_bounds[1],rtol=1e-4),
                D_at_lower_bound=np.isclose(d,d_bounds[0],rtol=1e-4),D_at_upper_bound=np.isclose(d,d_bounds[1],rtol=1e-4),
                Q_at_Q0=abs(q-q0)<1e-5,Q_at_1=abs(q-1)<1e-5)
