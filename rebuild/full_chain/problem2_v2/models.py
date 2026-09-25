"""Small, constrained candidate library for second-round scaling-law tests."""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares, brentq

BASE_LO=np.array([0,1e-6,.005,1e-6,.005],float)
BASE_HI=np.array([4,100,2,100,2],float)

def predict_nd(kind,x,N,D):
    N,D=np.broadcast_arrays(np.asarray(N,float),np.asarray(D,float))
    E,A,a,B,b=x[:5] if kind!='compute_ratio' else [0,0,0,0,0]
    if kind=='classic':return E+A*N**(-a)+B*D**(-b)
    if kind=='interaction':return E+A*N**(-a)+B*D**(-b)+x[5]*N**(-a)*D**(-b)
    if kind=='broken_N':
        Nc=x[6]
        return E+A*N**(-a)*(1+(N/Nc)**2)**(-x[5]/2)+B*D**(-b)
    if kind=='compute_ratio':
        E,K,c,t=x
        return E+K*(6*N*D)**(-c)*(N/D)**(c*t)
    raise ValueError(kind)

def bounds(kind):
    if kind=='classic':return BASE_LO,BASE_HI
    if kind=='interaction':return np.r_[BASE_LO,-5],np.r_[BASE_HI,5]
    if kind=='broken_N':return np.r_[BASE_LO,-.25,.08],np.r_[BASE_HI,.25,12]
    if kind=='compute_ratio':return np.array([0,1e-6,.005,-.95]),np.array([4,100,2,.95])
    raise ValueError(kind)

def fit_nd(kind,df,start=None,objective='ordinary',fixed_Nc=None):
    n=df.N_params_B.to_numpy(float);d=df.D_tokens_B.to_numpy(float);y=df.val_loss.to_numpy(float)
    lo,hi=bounds(kind)
    if start is None:
        start=np.array([1.6898,.35398,.33998,1.2403,.27988]+([0] if kind=='interaction' else [0,2.8] if kind=='broken_N' else []))
        if kind=='compute_ratio':start=np.array([1.5,1.5,.12,0])
    if fixed_Nc is not None and kind=='broken_N':
        lo=lo.copy();hi=hi.copy();lo[6]=fixed_Nc-1e-8;hi[6]=fixed_Nc+1e-8;start=start.copy();start[6]=fixed_Nc
    start=np.clip(start,lo+1e-9,hi-1e-9)
    def res(x):
        p=predict_nd(kind,x,n,d);r=p-y
        if objective=='relative':return r/np.maximum(y-1.68,.1)
        if objective=='log':return np.log(np.maximum(p,1e-8))-np.log(y)
        return r
    loss='huber' if objective=='huber' else 'soft_l1' if objective=='soft_l1' else 'linear'
    f=least_squares(res,start,bounds=(lo,hi),loss=loss,max_nfev=1200)
    return f

def predict_q(kind,z,N,D,Q,base):
    N,D,Q=np.broadcast_arrays(np.asarray(N,float),np.asarray(D,float),np.asarray(Q,float))
    E,A,a,B,b=base
    nd=E+A*N**(-a)+B*D**(-b)
    if kind=='additive':return nd+z[0]*(1-Q)**z[1]
    if kind=='effective_power':return E+A*N**(-a)+B*(D*Q**z[0])**(-b)
    if kind=='effective_exp':return E+A*N**(-a)+B*(D*np.exp(z[0]*(Q-1)))**(-b)
    if kind=='saturating':
        G,t=z
        if t<1e-7:return nd+G*(1-Q)
        return nd+G*(np.exp(-t*Q)-np.exp(-t))/(1-np.exp(-t))
    if kind=='quality_data':return nd+z[0]*(1-Q)**z[1]*(D/100)**(-z[2])
    if kind=='quality_model':return nd+z[0]*(1-Q)**z[1]*N**(-z[2])
    raise ValueError(kind)

def fit_q(kind,df,base):
    n=df.N_params_B.to_numpy(float);d=df.D_tokens_B.to_numpy(float);q=df.Q_score.to_numpy(float);y=df.val_loss.to_numpy(float)
    specs={'additive':([.36,1],[0,.1],[5,4]),'effective_power':([1],[0],[15]),
           'effective_exp':([1],[0],[15]),'saturating':([.4,1],[0,0],[5,15]),
           'quality_data':([.36,1,.1],[0,.1,-1],[5,4,1]),
           'quality_model':([.36,1,.1],[0,.1,-1],[5,4,1])}
    st,lo,hi=[np.array(v,float) for v in specs[kind]]
    f=least_squares(lambda z:predict_q(kind,z,n,d,q,base)-y,st,bounds=(lo,hi),max_nfev=1200)
    return f

def equivalent_N(kind,z,base,N,D,Q,delta=.1):
    target=float(predict_q(kind,z,N,D,Q+delta,base))
    f=lambda n:float(predict_q(kind,z,n,D,Q,base))-target
    if f(1e12)>0:return np.inf
    return brentq(f,N,1e12)

def predict_calibrated_ND(artifact,source,N,D):
    """Model B conditional N-D prediction after target-source calibration.

    Q and p effects were not identified on B2/B4/B5 and are deliberately not
    rescaled by this source correction.
    """
    coeff=artifact['source_calibration'][source]
    raw=predict_nd('classic',artifact['base_parameters'],N,D)
    return coeff['affine_intercept']+coeff['affine_slope']*raw

def predict_recommended(artifact,N,D,Q,p):
    """Model C N-D-Q-p prediction; Q is on the B6 mechanism scale."""
    from pathlib import Path
    import sys
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'problem1'))
    from problem1_v2 import transform
    mix=artifact['mixture_model']
    def mean_loss(vec):
        arr=np.asarray(vec,float).reshape(1,-1)
        x=transform(arr,mix['family'],mix['eps'],mix['reference_index'])
        return float(mix['model'].predict(x).mean())
    delta=mean_loss(p)-mean_loss(artifact['reference_p'])
    return float(predict_q(artifact['Q_model'],artifact['Q_parameters'],N,D,Q,artifact['ND_parameters']))+artifact['lambda_p']*delta
