from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from math import log
import numpy as np
import pandas as pd

@dataclass
class ConsumerParams:
    name: str; d0: float; d1: float; b0: float; b1: float; mu0: float; mu1: float; fit_accuracy: float

@dataclass
class TechParams:
    c0: float; c1: float; phi0: float; phi1: float; gamma: float

def load_data(path="data.csv"):
    return pd.read_csv(path)

def _positive_mean(x, fallback=1.0):
    v=x[x>0]
    return float(v.mean()) if len(v) else fallback

def _estimate_mu_delta(q0,q1,p0,p1,b0,b1,s=2.0):
    d0,d1=_positive_mean(q0),_positive_mean(q1)
    y=(q1>0).astype(float); rel=p1*d1-p0*d0
    def loss(delta):
        z=((b1-b0)+delta*s-rel)/10.0
        return float(np.sum(np.logaddexp(0,-z)*y+np.logaddexp(0,z)*(1-y))+0.05*delta**2)
    # Deterministic grid search keeps the deployed app lightweight and
    # avoids a SciPy runtime dependency in Streamlit Community Cloud.
    grid = np.linspace(-100.0, 100.0, 4001)
    losses = np.array([loss(float(x)) for x in grid])
    delta = float(grid[int(np.argmin(losses))])
    pred=((b1-b0)+delta*s-rel)>=0; active=(q0+q1)>0
    acc=float((pred[active]==(q1>0)[active]).mean()) if active.any() else 0.0
    return delta,acc

def estimate_consumers(df,s=2.0):
    p0=df.p_regular.to_numpy(float); p1=df.p_premium.to_numpy(float); out=[]
    for i in range(1,4):
        q0=df[f"h{i}_regular"].to_numpy(float); q1=df[f"h{i}_premium"].to_numpy(float)
        d0,d1=_positive_mean(q0),_positive_mean(q1)
        b0=float(np.max(p0*q0)); b1=float(np.max(p1*q1))
        mu,acc=_estimate_mu_delta(q0,q1,p0,p1,b0,b1,s)
        out.append(ConsumerParams(f"Household {i}",d0,d1,b0,b1,0.0,mu,acc))
    return out

def hyperplanes(consumers,s):
    rows=[]
    for c in consumers:
        rows += [
            {"kind":"Bounding","household":c.name,"equation":"p₀=(b₀+μ₀s)/d₀","a":c.d0,"b":0.0,"rhs":c.b0+c.mu0*s},
            {"kind":"Bounding","household":c.name,"equation":"p₁=(b₁+μ₁s)/d₁","a":0.0,"b":c.d1,"rhs":c.b1+c.mu1*s},
            {"kind":"Indifference","household":c.name,"equation":"p₀d₀−p₁d₁=(b₀−b₁)+(μ₀−μ₁)s","a":c.d0,"b":-c.d1,"rhs":(c.b0-c.b1)+(c.mu0-c.mu1)*s},
        ]
    return pd.DataFrame(rows)

def price_vertices(consumers,s):
    hp=hyperplanes(consumers,s); pts=[]
    for i,j in combinations(range(len(hp)),2):
        a,b=hp.iloc[i],hp.iloc[j]; A=np.array([[a.a,a.b],[b.a,b.b]],float)
        if abs(np.linalg.det(A))<1e-9: continue
        p=np.linalg.solve(A,np.array([a.rhs,b.rhs],float))
        if np.all(np.isfinite(p)) and np.all(p>=-1e-9):
            pts.append({"p_regular":max(float(p[0]),0.0),"p_premium":max(float(p[1]),0.0),"hyperplane_1":f"{a.kind} — {a.household}","hyperplane_2":f"{b.kind} — {b.household}"})
    if not pts: return pd.DataFrame(columns=["p_regular","p_premium","hyperplane_1","hyperplane_2"])
    return pd.DataFrame(pts).drop_duplicates(["p_regular","p_premium"]).sort_values(["p_regular","p_premium"]).reset_index(drop=True)

def consumer_choice(p0,p1,c,s):
    u0=c.b0-p0*c.d0+c.mu0*s; u1=c.b1-p1*c.d1+c.mu1*s
    k=int(np.argmax([0.0,u0,u1])); return k-1,float(max(0.0,[0.0,u0,u1][k]))

def market_outcome(p0,p1,consumers,s,tech,labels=(0,1)):
    d0=d1=0.0; choices={}; utilities={}
    for c in consumers:
        price0=p0 if labels[0]==0 else p1; price1=p0 if labels[1]==0 else p1
        mu0=c.mu0 if labels[0]==0 else c.mu1; mu1=c.mu0 if labels[1]==0 else c.mu1
        u0=c.b0-price0*c.d0+mu0*s; u1=c.b1-price1*c.d1+mu1*s
        k=int(np.argmax([0.0,u0,u1]))
        if k==1: d0+=c.d0; choices[c.name]="Regular"; utilities[c.name]=max(0.0,u0)
        elif k==2: d1+=c.d1; choices[c.name]="Premium"; utilities[c.name]=max(0.0,u1)
        else: choices[c.name]="Exit"; utilities[c.name]=0.0
    q0price=p0 if labels[0]==0 else p1; q1price=p0 if labels[1]==0 else p1
    profit=(q0price-tech.c0)*d0+(q1price-tech.c1)*d1
    footprint=tech.phi0*d0+tech.phi1*d1
    cu=sum(log(u+1.0) for u in utilities.values()); welfare=cu+profit-tech.gamma*footprint
    return {"p_regular":p0,"p_premium":p1,"demand_regular":d0,"demand_premium":d1,"profit":profit,"footprint":footprint,"consumer_utility":cu,"welfare":welfare,"choices":choices,"utilities":utilities,"labels":labels}

def empirical_regions(df):
    return pd.DataFrame([
        ["A — High Premium Price / Market Exit","p₁ > 81","7, 8, 9","H1 + H2 Regular; H3 Nothing","8 units"],
        ["B — Intermediate Premium Price","80 ≤ p₁ ≤ 81 and p₀ ≤ 55","2, 6, 11","H1 + H2 Regular; H3 Premium","12 units"],
        ["C — Low Premium Price / Eco-adoption","p₁ ≤ 75","1, 4, 5, 10","Premium adoption rises; H1 switches at high p₀","15–17 units"],
    ],columns=["Region","Condition","Observed weeks","Observed behavior","Total demand"])

def solve_scenario(scenario,consumers,tech,s=2.0):
    v=price_vertices(consumers,s); v=v[(v.p_regular>0)&(v.p_premium>0)]
    if v.empty: raise ValueError("No finite price vertices were generated.")
    labels=[(0,1)] if scenario in {"C","CI"} else [(0,1),(0,0)]
    def outcomes(lab): return [market_outcome(r.p_regular,r.p_premium,consumers,s,tech,lab) for r in v.itertuples(index=False)]
    if scenario=="C": best=max(outcomes((0,1)),key=lambda x:(x["welfare"],x["profit"]))
    elif scenario=="L":
        candidates=[]
        for r in v.itertuples(index=False):
            opts=[market_outcome(r.p_regular,r.p_premium,consumers,s,tech,l) for l in labels]
            candidates.append(max(opts,key=lambda x:(x["profit"],x["welfare"])))
        best=max(candidates,key=lambda x:(x["welfare"],x["profit"]))
    elif scenario=="CI": best=max(outcomes((0,1)),key=lambda x:(x["profit"],x["welfare"]))
    elif scenario=="LI": best=max([max(outcomes(l),key=lambda x:(x["profit"],x["welfare"])) for l in labels],key=lambda x:(x["welfare"],x["profit"]))
    else: raise ValueError(f"Unknown scenario: {scenario}")
    best=dict(best); best.update(scenario=scenario,stringency=s,delimiter=min(tech.phi0,tech.phi1)); return best

def observed_market_outcome(df,consumers,tech,s=2.0):
    rows=[]
    for r in df.itertuples(index=False):
        o=market_outcome(r.p_regular,r.p_premium,consumers,s,tech,(0,1))
        rows.append({"week":int(r.T),"p_regular":r.p_regular,"p_premium":r.p_premium,"profit":o["profit"],"welfare":o["welfare"],"footprint":o["footprint"],"demand_regular":o["demand_regular"],"demand_premium":o["demand_premium"],"choices":"; ".join(f"{k}: {v}" for k,v in o["choices"].items())})
    return pd.DataFrame(rows)
