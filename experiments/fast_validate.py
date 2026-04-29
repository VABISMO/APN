#!/usr/bin/env python3
"""
Fast APN validation — streamlined experiments for CPU.
Tests the hypothesis: APN needs fewer layers than SwiGLU.
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, time, json, os, sys

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── APN ──────────────────────────────────────────────────────────────
class APNFunc(nn.Module):
    NFUNCS = 6
    NAMES = ["identity", "sq-tanh", "s-sqrt", "b-prod", "sin", "relu"]
    def __init__(self, H, tau=3.0):
        super().__init__(); self.H = H; self.tau = tau
        self.logits = nn.Parameter(torch.randn(H, 6) * 0.1)
    def forward(self, p1, p2):
        a, b = p1, p2
        fvals = torch.stack([a, torch.tanh(a*a), a.sign()*(a.abs()+1e-4).sqrt(),
            (a*b)/((a*b).pow(2)+1).sqrt(), torch.sin(a), F.leaky_relu(a, 0.01)], dim=-1)
        alpha = F.softmax(self.logits/(self.tau+1e-7), dim=-1)
        return (fvals * alpha).sum(dim=-1)
    def anneal(self, p, tau0=3.0, tau1=0.05): self.tau = tau0*(tau1/tau0)**p

class APNLayer(nn.Module):
    def __init__(self, d, h, o, tau=3.0):
        super().__init__()
        self.W1=nn.Linear(d,h,bias=True); self.W2=nn.Linear(d,h,bias=True)
        self.apn=APNFunc(h,tau); self.Wo=nn.Linear(h,o,bias=True)
        s=math.sqrt(2/d)
        nn.init.normal_(self.W1.weight,0,s); nn.init.zeros_(self.W1.bias)
        nn.init.normal_(self.W2.weight,0,s*0.5); nn.init.zeros_(self.W2.bias)
        nn.init.normal_(self.Wo.weight,0,math.sqrt(2/h)); nn.init.zeros_(self.Wo.bias)
    def forward(self, x): return self.Wo(self.apn(self.W1(x), self.W2(x)))
    def anneal(self, p, t0=3.0, t1=0.05): self.apn.anneal(p,t0,t1)

class SwiGLU(nn.Module):
    def __init__(self, d, h, o):
        super().__init__()
        self.gate=nn.Linear(d,h,bias=False); self.up=nn.Linear(d,h,bias=False); self.down=nn.Linear(h,o,bias=False)
    def forward(self, x): return self.down(self.gate(x)*F.silu(self.up(x)))

class GELUFFN(nn.Module):
    def __init__(self, d, h, o):
        super().__init__()
        self.fc1=nn.Linear(d,h,bias=True); self.fc2=nn.Linear(h,o,bias=True)
    def forward(self, x): return self.fc2(F.gelu(self.fc1(x)))

# ── Exp 1: Single-layer function approximation ───────────────────────
def exp1():
    print("\n" + "="*70)
    print("  EXP 1: Single-layer function approximation")
    print("="*70)
    N=2000; H=64; torch.manual_seed(42)
    X=torch.randn(N,4).to(DEVICE).abs()*0.5+0.1
    tasks={
        "ratio x0/x1":   lambda x:(x[:,0]/(x[:,1]+0.05)).unsqueeze(1),
        "product x0*x1": lambda x:(x[:,0]*x[:,1]).unsqueeze(1),
        "sqrt |x0|":     lambda x:torch.sqrt(x[:,0].abs()+1e-4).unsqueeze(1),
        "sin(x0)":        lambda x:torch.sin(x[:,0]*3.14).unsqueeze(1),
        "x0^2":          lambda x:(x[:,0]**2).unsqueeze(1),
        "1/(1+|x0|)":    lambda x:(1/(1+x[:,0].abs())).unsqueeze(1),
        "x0^2+x1^2":     lambda x:(x[:,0]**2+x[:,1]**2).unsqueeze(1),
        "bprod x0*x1/sqrt(1+(x0*x1)^2)": lambda x:((x[:,0]*x[:,1])/(1+(x[:,0]*x[:,1])**2).sqrt()).unsqueeze(1),
    }
    Xtr,Xte=X[:1600],X[1600:]
    print(f"  {'Task':<40} {'APN-1L':>9} {'SwiGLU-1L':>10} {'GELU-1L':>9} {'APN-2L':>9} {'SwiGLU-2L':>10} {'Best 1L':>8}")
    print(f"  {'─'*96}")
    results={}
    for name,fn in tasks.items():
        y=fn(X); y=(y-y.mean())/(y.std()+1e-8)
        ytr,yte=y[:1600],y[1600:]
        res={}
        for mname,mods in [
            ("APN-1L",  [APNLayer(4,H,1)]),
            ("SwiGLU-1L",[SwiGLU(4,H,1)]),
            ("GELU-1L", [GELUFFN(4,H,1)]),
            ("APN-2L",  [APNLayer(4,H,H), APNLayer(H,H,1)]),
            ("SwiGLU-2L",[SwiGLU(4,H,H), SwiGLU(H,H,1)]),
        ]:
            model=nn.Sequential(*[m.to(DEVICE) for m in mods])
            opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=0.01)
            for step in range(800):
                prog=step/800
                for m in model.modules():
                    if isinstance(m,APNFunc): m.anneal(prog,3.0,0.05)
                opt.zero_grad(); loss=F.mse_loss(model(Xtr),ytr); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            with torch.no_grad(): mse=F.mse_loss(model(Xte),yte).item()
            res[mname]=mse
        results[name]=res
        best=min(("APN" if res["APN-1L"]<res["SwiGLU-1L"]*0.9 else
                  "SwiGLU" if res["SwiGLU-1L"]<res["APN-1L"]*0.9 else
                  "GELU" if res["GELU-1L"]<min(res["APN-1L"],res["SwiGLU-1L"])*0.9 else "tie"))
        print(f"  {name:<40} {res['APN-1L']:>9.5f} {res['SwiGLU-1L']:>10.5f} {res['GELU-1L']:>9.5f} {res['APN-2L']:>9.5f} {res['SwiGLU-2L']:>10.5f} {best:>8}")
    apn_wins=sum(1 for r in results.values() if r["APN-1L"]<r["SwiGLU-1L"])
    print(f"\n  APN-1L wins: {apn_wins}/{len(tasks)} tasks vs SwiGLU-1L")
    # Check depth efficiency
    depth_eq=sum(1 for r in results.values() if abs(r["APN-1L"]-r["SwiGLU-2L"])/max(r["APN-1L"],r["SwiGLU-2L"],1e-8)<0.2)
    print(f"  APN-1L ≈ SwiGLU-2L (within 20%): {depth_eq}/{len(tasks)} tasks")
    return results

# ── Exp 2: Per-layer capacity ────────────────────────────────────────
def exp2():
    print("\n" + "="*70)
    print("  EXP 2: Per-layer learning capacity (mixed nonlinear target)")
    print("="*70)
    torch.manual_seed(42); N=4000; D=16
    X=torch.randn(N,D).to(DEVICE).abs()*0.5+0.1
    Y=torch.zeros(N,D).to(DEVICE)
    Y[:,0]=X[:,0]/(X[:,1].abs()+0.1); Y[:,1]=X[:,2]*X[:,3]
    Y[:,2]=torch.sqrt(X[:,4].abs()+1e-4); Y[:,3]=torch.sin(X[:,5]*3.14)
    Y[:,4]=X[:,6]**2; Y[:,5]=X[:,7].abs()
    Y[:,6]=X[:,8]/(1+X[:,9].abs()); Y[:,7]=X[:,10]*X[:,11]/(1+(X[:,10]*X[:,11]).abs())
    for i in range(8,D): Y[:,i]=X[:,i%D]*0.5+X[:,(i+1)%D]*0.3
    Y=(Y-Y.mean(0))/(Y.std(0)+1e-8)
    Xtr,Ytr,Xte,Yte=X[:3200],Y[:3200],X[3200:],Y[3200:]
    H=128
    configs=[
        ("APN-1L",  nn.Sequential(APNLayer(D,H,D))),
        ("SwiGLU-1L",nn.Sequential(SwiGLU(D,H,D))),
        ("GELU-1L", nn.Sequential(GELUFFN(D,H,D))),
        ("APN-2L",  nn.Sequential(APNLayer(D,H,H),APNLayer(H,H,D))),
        ("SwiGLU-2L",nn.Sequential(SwiGLU(D,H,H),SwiGLU(H,H,D))),
        ("Linear",  nn.Sequential(nn.Linear(D,H),nn.Linear(H,D))),
    ]
    print(f"  {'Model':<14} {'Params':>8} {'MSE':>10} {'R²':>8}")
    print(f"  {'─'*44}")
    res={}
    for name,model in configs:
        model=model.to(DEVICE); npar=sum(p.numel() for p in model.parameters())
        opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=0.01)
        for step in range(1500):
            prog=step/1500
            for m in model.modules():
                if isinstance(m,APNFunc): m.anneal(prog,3.0,0.05)
            opt.zero_grad(); loss=F.mse_loss(model(Xtr),Ytr); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        with torch.no_grad():
            pred=model(Xte); mse=F.mse_loss(pred,Yte).item()
            ss_res=((Yte-pred)**2).sum(0); ss_tot=((Yte-Yte.mean(0))**2).sum(0)
            r2=(1-ss_res/(ss_tot+1e-8)).mean().item()
        res[name]={"params":npar,"mse":round(mse,6),"r2":round(r2,4)}
        print(f"  {name:<14} {npar:>8} {mse:>10.6f} {r2:>8.4f}")
    # Key comparison
    print(f"\n  APN-1L MSE improvement over SwiGLU-1L: {(1-res['APN-1L']['mse']/res['SwiGLU-1L']['mse'])*100:.1f}%")
    print(f"  APN-2L MSE improvement over SwiGLU-2L: {(1-res['APN-2L']['mse']/res['SwiGLU-2L']['mse'])*100:.1f}%")
    ratio=res['APN-1L']['mse']/res['SwiGLU-2L']['mse']
    print(f"  APN-1L MSE / SwiGLU-2L MSE ratio: {ratio:.3f}")
    if ratio<1.1: print(f"  → APN-1L APPROXIMATELY MATCHES SwiGLU-2L!")
    elif ratio<1.5: print(f"  → APN-1L is between SwiGLU-1L and SwiGLU-2L")
    else: print(f"  → APN-1L does NOT approach SwiGLU-2L")
    return res

# ── Exp 3: Depth scaling (language model) ────────────────────────────
class TinyBlock(nn.Module):
    def __init__(self,d,h,ffn_type="apn",tau=3.0):
        super().__init__()
        self.ln1=nn.LayerNorm(d); self.attn=nn.MultiheadAttention(d,h,batch_first=True,bias=False)
        self.ln2=nn.LayerNorm(d)
        ffn_h=d*4
        if ffn_type=="apn": self.ffn=APNLayer(d,ffn_h,d,tau)
        elif ffn_type=="swiglu": self.ffn=SwiGLU(d,ffn_h,d)
        elif ffn_type=="gelu": self.ffn=GELUFFN(d,ffn_h,d)
    def forward(self,x):
        T=x.shape[1]; msk=torch.triu(torch.ones(T,T,device=x.device),1).bool()
        h,_=self.attn(self.ln1(x),self.ln1(x),self.ln1(x),attn_mask=msk,need_weights=False)
        x=x+h; x=x+self.ffn(self.ln2(x)); return x

class GPT(nn.Module):
    def __init__(self,V,d,nl,nh,ft="apn",tau=3.0):
        super().__init__()
        self.emb=nn.Embedding(V,d); self.pos=nn.Embedding(256,d)
        self.blocks=nn.ModuleList([TinyBlock(d,nh,ft,tau) for _ in range(nl)])
        self.ln_f=nn.LayerNorm(d); self.head=nn.Linear(d,V,bias=False)
        self.head.weight=self.emb.weight
        for m in self.modules():
            if isinstance(m,nn.Linear): nn.init.normal_(m.weight,0,0.02)
            if isinstance(m,nn.Embedding): nn.init.normal_(m.weight,0,0.02)
    def forward(self,idx,tgt=None):
        B,T=idx.shape; x=self.emb(idx)+self.pos(torch.arange(T,device=idx.device))
        for b in self.blocks: x=b(x)
        lg=self.head(self.ln_f(x))
        loss=F.cross_entropy(lg.view(-1,lg.size(-1)),tgt.view(-1)) if tgt is not None else None
        return lg,loss

def exp3():
    print("\n" + "="*70)
    print("  EXP 3: Language model depth scaling")
    print("  APN with fewer layers vs SwiGLU with more layers")
    print("="*70)
    import random; random.seed(42)
    lines=[]
    for i in range(1,20):
        for j in range(1,20):
            lines+=["{}+{}={}".format(i,j,i+j),"{}*{}={}".format(i,j,i*j)]
    words="the cat dog ran jumped over under big small fast slow and or but a an is was".split()
    for _ in range(2000): lines.append(" ".join(random.choices(words,k=random.randint(4,9))))
    random.shuffle(lines); text="\n".join(lines)
    chars=sorted(set(text)); c2id={c:i for i,c in enumerate(chars)}; V=len(chars)
    ids=torch.tensor([c2id[c] for c in text],dtype=torch.long)
    SEQ=64; B=16; D=128; H=4
    configs=[
        ("SwiGLU-4L","swiglu",4), ("SwiGLU-6L","swiglu",6),
        ("APN-2L","apn",2), ("APN-3L","apn",3), ("APN-4L","apn",4),
    ]
    print(f"  {'Model':<14} {'Params':>8} {'PPL':>7} {'Loss':>8}")
    print(f"  {'─'*40}")
    res={}
    for name,ft,nl in configs:
        torch.manual_seed(42)
        model=GPT(V,D,nl,H,ft).to(DEVICE)
        npar=sum(p.numel() for p in model.parameters())
        opt=torch.optim.AdamW(model.parameters(),lr=3e-3,weight_decay=0.1)
        STEPS=600
        for step in range(STEPS):
            prog=step/STEPS
            for m in model.modules():
                if isinstance(m,APNFunc): m.anneal(prog,3.0,0.05)
            starts=torch.randint(0,len(ids)-SEQ-1,(B,))
            x=torch.stack([ids[s:s+SEQ] for s in starts]).to(DEVICE)
            y=torch.stack([ids[s+1:s+SEQ+1] for s in starts]).to(DEVICE)
            _,loss=model(x,y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        # Evaluate
        model.eval(); elosses=[]
        with torch.no_grad():
            for i in range(0,min(1000,len(ids)-SEQ-1),SEQ):
                x=ids[i:i+SEQ].unsqueeze(0).to(DEVICE); y=ids[i+1:i+SEQ+1].unsqueeze(0).to(DEVICE)
                _,l=model(x,y); elosses.append(l.item())
        ppl=math.exp(min(sum(elosses)/len(elosses),10))
        res[name]={"ppl":round(ppl,2),"loss":round(sum(elosses)/len(elosses),4),"params_M":round(npar/1e6,2)}
        print(f"  {name:<14} {npar/1e6:.2f}M  {ppl:>7.2f} {sum(elosses)/len(elosses):>8.4f}")

    # Compare depth efficiency
    sg4=res.get("SwiGLU-4L",{}).get("ppl",999)
    sg6=res.get("SwiGLU-6L",{}).get("ppl",999)
    a2=res.get("APN-2L",{}).get("ppl",999)
    a3=res.get("APN-3L",{}).get("ppl",999)
    a4=res.get("APN-4L",{}).get("ppl",999)
    print(f"\n  Depth efficiency:")
    for aname,ap in [("APN-2L",a2),("APN-3L",a3),("APN-4L",a4)]:
        for sname,sp in [("SwiGLU-4L",sg4),("SwiGLU-6L",sg6)]:
            if sp>0 and ap>0:
                diff=abs(ap-sp)/max(ap,sp)*100
                if diff<15: print(f"    {aname} (ppl={ap:.2f}) ≈ {sname} (ppl={sp:.2f})  diff={diff:.1f}%")
    return res

# ── Exp 4: Specialization ────────────────────────────────────────────
def exp4():
    print("\n" + "="*70)
    print("  EXP 4: APN neuron specialization after training")
    print("="*70)
    torch.manual_seed(42); N=3000; D=8
    X=torch.randn(N,D).to(DEVICE).abs()*0.5+0.1
    Y=(X[:,0]/(X[:,1]+0.05)+torch.sin(X[:,2])+X[:,3]**2).unsqueeze(1)
    Y=(Y-Y.mean())/(Y.std()+1e-8)
    H=32; model=APNLayer(D,H,1).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
    for step in range(1500):
        prog=step/1500; model.apn.anneal(prog,3.0,0.05)
        opt.zero_grad(); loss=F.mse_loss(model(X[:2500]),Y[:2500]); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    with torch.no_grad():
        alpha=F.softmax(model.apn.logits/(model.apn.tau+1e-7),dim=-1)
        dom=alpha.argmax(dim=-1)
    counts={}
    for name,idx in zip(APNFunc.NAMES,range(6)):
        c=(dom==idx).sum().item()
        if c>0: counts[name]=100*c//H
    for name,pct in sorted(counts.items(),key=lambda x:-x[1]):
        print(f"    {name:<12}: {pct}%")
    print(f"  Final tau: {model.apn.tau:.4f}")
    with torch.no_grad(): mse=F.mse_loss(model(X[2500:]),Y[2500:]).item()
    print(f"  Test MSE: {mse:.6f}")
    return counts

# ── Exp 5: Speed ─────────────────────────────────────────────────────
def exp5():
    print("\n" + "="*70)
    print("  EXP 5: Forward+backward speed per layer")
    print("="*70)
    D,H=512,2048; B=8
    x=torch.randn(B,D,device=DEVICE)
    layers={"APN":APNLayer(D,H,D).to(DEVICE),"SwiGLU":SwiGLU(D,H,D).to(DEVICE),"GELU":GELUFFN(D,H,D).to(DEVICE)}
    # Warmup
    for name,layer in layers.items():
        for _ in range(10): y=layer(x); y.sum().backward()
    N=500
    for name,layer in layers.items():
        torch.cuda.synchronize() if DEVICE.type=="cuda" else None
        t0=time.time()
        with torch.no_grad():
            for _ in range(N): y=layer(x)
        fw=(time.time()-t0)/N*1000
        torch.cuda.synchronize() if DEVICE.type=="cuda" else None
        t0=time.time()
        for _ in range(N): y=layer(x); y.sum().backward()
        fwb=(time.time()-t0)/N*1000
        npar=sum(p.numel() for p in layer.parameters())
        print(f"  {name:<10} params={npar/1e6:.2f}M  fwd={fw:.2f}ms  fwd+bwd={fwb:.2f}ms")

def main():
    print("="*70)
    print("  APN DEPTH HYPOTHESIS — FAST VALIDATION")
    print("="*70)
    r={}
    r["exp1"]=exp1()
    r["exp2"]=exp2()
    r["exp3"]=exp3()
    r["exp4"]=exp4()
    exp5()
    p=os.path.join(os.path.dirname(__file__) or ".", "validation_results.json")
    with open(p,"w") as f: json.dump(r,f,indent=2)
    print(f"\n  Results saved to {p}")
    print("\n"+"="*70)
    print("  FINAL VERDICT")
    print("="*70)
    e1=r["exp1"]; e2=r["exp2"]
    apn_wins=sum(1 for v in e1.values() if v["APN-1L"]<v["SwiGLU-1L"])
    total=len(e1)
    depth_eq=sum(1 for v in e1.values() if abs(v["APN-1L"]-v["SwiGLU-2L"])/max(v["APN-1L"],v["SwiGLU-2L"],1e-8)<0.2)
    r2_adv=e2.get("APN-1L",{}).get("r2",0)-e2.get("SwiGLU-1L",{}).get("r2",0)
    mse_ratio=e2.get("APN-1L",{}).get("mse",1)/max(e2.get("SwiGLU-2L",{}).get("mse",1),1e-8)
    print(f"  Single-layer: APN wins {apn_wins}/{total} tasks vs SwiGLU")
    print(f"  Depth: APN-1L ≈ SwiGLU-2L on {depth_eq}/{total} tasks (within 20%)")
    print(f"  R² advantage APN-1L over SwiGLU-1L: {r2_adv:+.4f}")
    print(f"  MSE ratio APN-1L/SwiGLU-2L: {mse_ratio:.3f}")
    if apn_wins>total//2 and mse_ratio<1.5:
        print("\n  HYPOTHESIS SUPPORTED: APN has higher per-layer expressivity.")
        print("  Fewer APN layers can match more SwiGLU layers on nonlinear tasks.")
    elif apn_wins>total//2:
        print("\n  HYPOTHESIS PARTIALLY SUPPORTED: APN-1L outperforms SwiGLU-1L")
        print("  but does NOT match SwiGLU-2L — still needs depth, just less.")
    else:
        print("\n  HYPOTHESIS NOT SUPPORTED: APN shows no clear per-layer advantage.")
        print("  The function bank doesn't compensate for depth.")

if __name__=="__main__": main()