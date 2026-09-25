"""ALR support diagnostics and cross-scale mixture scenarios."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'problem1'))
from problem1_v2 import transform


class Support:
    def __init__(self,p,mix,k=5):
        self.p=np.asarray(p,float);self.mix=mix;self.k=k
        x=transform(self.p,mix['family'],mix['eps'],mix['reference_index'])
        self.mean=x.mean(axis=0);self.sd=x.std(axis=0);self.sd[self.sd==0]=1
        self.xt=(x-self.mean)/self.sd
        self.tree=cKDTree(self.xt)
        self.knn_train=self.tree.query(self.xt,k=k+1)[0][:,k]
        self.pca=PCA(n_components=.95,svd_solver='full').fit(self.xt)
        self.z=self.pca.transform(self.xt)
        self.cov=LedoitWolf().fit(self.z)
        self.md_train=self.cov.mahalanobis(self.z)
        self.resid_train=np.linalg.norm(self.xt-self.pca.inverse_transform(self.z),axis=1)
        self.threshold={name:{q:float(np.quantile(v,q)) for q in [.95,.99]} for name,v in
                        [('knn5',self.knn_train),('mahalanobis',self.md_train),('pca_residual',self.resid_train)]}

    def evaluate(self,p):
        p=np.asarray(p,float).reshape(-1,self.p.shape[1]);x=transform(p,self.mix['family'],self.mix['eps'],self.mix['reference_index'])
        xt=(x-self.mean)/self.sd;z=self.pca.transform(xt)
        distances={'knn5':np.asarray(self.tree.query(xt,k=self.k)[0])[:,-1],
                   'mahalanobis':self.cov.mahalanobis(z),
                   'pca_residual':np.linalg.norm(xt-self.pca.inverse_transform(z),axis=1)}
        labels=[]
        for i in range(len(p)):
            above95=sum(distances[name][i]>self.threshold[name][.95] for name in distances)
            above99=sum(distances[name][i]>self.threshold[name][.99] for name in distances)
            labels.append('in-support' if above95==0 else 'near-support' if above99==0 and above95<=2 else 'extrapolative')
        return distances,labels


def analyze(artifact,p,old_mix):
    mix=artifact['mixture_model'];support=Support(p,mix);ref=artifact['reference_p']
    def predict(a):
        arr=np.asarray(a,float).reshape(-1,p.shape[1]);return mix['model'].predict(transform(arr,mix['family'],mix['eps'],mix['reference_index'])).mean(axis=1)
    observed=p[int(old_mix[old_mix.strategy=='P1_observed_support'].iloc[0].training_row)]
    loose=old_mix[old_mix.strategy=='P1_loose_simplex'][mix['pcols']].to_numpy(float)[0]
    # A continuous search strictly within the local simplex spanned by the 12
    # nearest observed training mixtures. Candidates also pass three support tests.
    obs_x=support.xt[int(old_mix[old_mix.strategy=='P1_observed_support'].iloc[0].training_row)]
    ids=support.tree.query(obs_x,k=12)[1];local=p[ids]
    rng=np.random.default_rng(20260924);weights=rng.dirichlet(np.r_[12.,np.full(11,.7)],size=2500)
    candidates=np.vstack([local[0],weights@local])
    losses=predict(candidates);_,labels=support.evaluate(candidates)
    good=np.array([v=='in-support' for v in labels]);ix=int(np.argmin(np.where(good,losses,np.inf)))
    p_cont=candidates[ix]
    # Local SLSQP refinement; retain it only if all support checks still pass.
    def f(w):return float(predict(w@local)[0])
    w0=np.zeros(12);w0[0]=1
    rr=minimize(f,w0,method='SLSQP',bounds=[(0,1)]*12,constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],options={'ftol':1e-11,'maxiter':400})
    if rr.success:
        cand=rr.x@local
        if support.evaluate(cand)[1][0]=='in-support' and f(rr.x)<float(predict(p_cont)[0]):p_cont=cand
    candidates={'P0_reference':ref,'P1_observed_support':observed,'P2_continuous_support':p_cont,'P3_loose_simplex':loose}
    ref_loss=float(predict(ref)[0]);rows=[]
    for name,v in candidates.items():
        dist,lab=support.evaluate(v)
        rows.append(dict(strategy=name,mixture_loss=float(predict(v)[0]),delta_p=float(predict(v)[0]-ref_loss),support_class=lab[0],
                         observed_training_row=int(old_mix[old_mix.strategy=='P1_observed_support'].iloc[0].training_row) if name=='P1_observed_support' else -1,
                         max_share=float(v.max()),**{k:float(val[0]) for k,val in dist.items()},
                         **{f'{k}_p95':support.threshold[k][.95] for k in dist},
                         **{f'{k}_p99':support.threshold[k][.99] for k in dist},
                         **{col:float(v[j]) for j,col in enumerate(mix['pcols'])}))
    return pd.DataFrame(rows),support


def lambda_curve(n,scenario):
    n0=.001 # 1 million parameters in B units
    if scenario=='S0_constant':return 1.
    if scenario=='S1_decay_0.1':return min(1.,(n/n0)**(-.1))
    if scenario=='S2_decay_0.2':return min(1.,(n/n0)**(-.2))
    if scenario=='S3_floor_0.25':return max(.25,min(1.,(n/n0)**(-.15)))
    raise ValueError(scenario)
