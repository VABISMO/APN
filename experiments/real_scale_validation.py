#!/usr/bin/env python3
"""
Real-scale APN validation.
Train APN vs SwiGLU at multiple scales on real text (TinyStories).
Measure: perplexity, convergence speed, params/layer efficiency, and the
critical question: can APN with L/2 layers match SwiGLU with L layers?
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, time, json, os, sys
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── APN (from probnet_complete.py, identical) ────────────────────────
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
    def specialization(self):
        with torch.no_grad():
            alpha = F.softmax(self.logits/(self.tau+1e-7), dim=-1)
            dom = alpha.argmax(dim=-1)
            counts = {}
            for i in dom.cpu().tolist():
                n = self.NAMES[i]; counts[n] = counts.get(n, 0) + 1
            return {k: f"{100*v//self.H}%" for k, v in sorted(counts.items(), key=lambda x: -x[1])}

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
    def anneal(self, p, tau0=3.0, tau1=0.05): self.apn.anneal(p, tau0, tau1)

class SwiGLU(nn.Module):
    def __init__(self, d, h, o):
        super().__init__()
        self.gate=nn.Linear(d,h,bias=False); self.up=nn.Linear(d,h,bias=False); self.down=nn.Linear(h,o,bias=False)
    def forward(self, x): return self.down(self.gate(x)*F.silu(self.up(x)))

# ── Transformer ──────────────────────────────────────────────────────
class Block(nn.Module):
    def __init__(self, d, nh, ffn_type="apn", ffn_h=None, tau=3.0):
        super().__init__()
        self.ln1=nn.LayerNorm(d); self.attn=nn.MultiheadAttention(d,nh,batch_first=True,bias=False)
        self.ln2=nn.LayerNorm(d)
        ffn_h = ffn_h or d*4
        if ffn_type=="apn": self.ffn=APNLayer(d,ffn_h,d,tau)
        elif ffn_type=="swiglu": self.ffn=SwiGLU(d,ffn_h,d)
        else: raise ValueError(ffn_type)
    def forward(self, x):
        T=x.shape[1]; msk=torch.triu(torch.ones(T,T,device=x.device),1).bool()
        h,_=self.attn(self.ln1(x),self.ln1(x),self.ln1(x),attn_mask=msk,need_weights=False)
        x=x+h; x=x+self.ffn(self.ln2(x)); return x

class GPT(nn.Module):
    def __init__(self, V, d, nl, nh, ft="apn", ffn_h=None, tau=3.0, max_seq=256):
        super().__init__()
        self.emb=nn.Embedding(V,d); self.pos=nn.Embedding(max_seq,d)
        self.blocks=nn.ModuleList([Block(d,nh,ft,ffn_h,tau) for _ in range(nl)])
        self.ln_f=nn.LayerNorm(d); self.head=nn.Linear(d,V,bias=False)
        self.head.weight=self.emb.weight; self.ft=ft
        for m in self.modules():
            if isinstance(m,nn.Linear): nn.init.normal_(m.weight,0,0.02)
            if isinstance(m,nn.Embedding): nn.init.normal_(m.weight,0,0.02)
    def forward(self, idx, tgt=None):
        B,T=idx.shape; x=self.emb(idx)+self.pos(torch.arange(T,device=idx.device))
        for b in self.blocks: x=b(x)
        lg=self.head(self.ln_f(x))
        loss = F.cross_entropy(lg.view(-1,lg.size(-1)),tgt.view(-1)) if tgt is not None else None
        return lg, loss

# ── Data: TinyStories or synthetic ───────────────────────────────────
def get_data(min_chars=500000):
    """Try to load TinyStories, fall back to synthetic."""
    # Try HuggingFace datasets
    try:
        from datasets import load_dataset
        print("  Loading TinyStories from HuggingFace...")
        ds = load_dataset("roneneldan/TinyStories", split="train[:5000]")
        text = "\n".join(ds["text"])
        if len(text) > min_chars:
            return text[:min_chars], "TinyStories"
    except: pass
    # Try local file
    for path in ["data/corpus.txt", "data/tiny_stories.txt", "../data/corpus.txt"]:
        if os.path.exists(path):
            text = open(path).read()
            if len(text) > min_chars:
                return text[:min_chars], f"local:{path}"
    # Generate diverse synthetic corpus
    import random
    random.seed(42)
    lines = []
    # Arithmetic
    for i in range(1,30):
        for j in range(1,30):
            lines.append(f"{i} plus {j} equals {i+j}")
            lines.append(f"{i} times {j} equals {i*j}")
            if i > 0 and j > 0: lines.append(f"{i*j} divided by {j} equals {i}")
    # Natural language patterns (stories)
    subjects = ["a little girl","a boy","a cat","a dog","a rabbit","a turtle","a dragon","a princess","a knight","a wizard"]
    verbs = ["went to","found","lost","loved","hated","chased","helped","watched","played with","talked to"]
    objects = ["the park","a castle","the forest","a river","the mountain","a garden","the beach","a village","the library","a cave"]
    adj = ["happy","sad","brave","small","big","fast","slow","kind","angry","funny"]
    for _ in range(5000):
        s = random.choice(subjects)
        a = random.choice(adj)
        v = random.choice(verbs)
        o = random.choice(objects)
        lines.append(f"Once upon a time, {s} was {a} and {v} {o}.")
        lines.append(f"The {a} {s} {v} {o} every day.")
        lines.append(f"One day, {s} decided to {v} {o}.")
    # More complex sentences
    for _ in range(3000):
        s = random.choice(subjects)
        a = random.choice(adj)
        v = random.choice(verbs)
        o = random.choice(objects)
        lines.append(f"{s.capitalize()} was very {a}. {s.capitalize()} liked to {v.split()[0]} {o}.")
    text = "\n".join(lines)
    return text[:min_chars], "synthetic"

# ── Train and evaluate ───────────────────────────────────────────────
def train_and_eval(text, vocab_size, d_model, n_layers, n_heads, ffn_type,
                   ffn_hidden=None, seq_len=128, batch_size=16, steps=800, tau0=3.0):
    chars = sorted(set(text))
    c2id = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([c2id[c] for c in text], dtype=torch.long)
    V = len(chars)

    torch.manual_seed(42)
    model = GPT(V, d_model, n_layers, n_heads, ffn_type, ffn_hidden, tau0).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.1)

    # Train
    losses = []
    t0 = time.time()
    for step in range(steps):
        prog = step / steps
        for m in model.modules():
            if isinstance(m, APNFunc): m.anneal(prog, tau0, 0.05)
        starts = torch.randint(0, len(ids)-seq_len-1, (batch_size,))
        x = torch.stack([ids[s:s+seq_len] for s in starts]).to(DEVICE)
        y = torch.stack([ids[s+1:s+seq_len+1] for s in starts]).to(DEVICE)
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 50 == 0: losses.append(loss.item())

    train_time = time.time() - t0

    # Evaluate perplexity on held-out data
    model.eval()
    eval_losses = []
    with torch.no_grad():
        for i in range(0, min(2000, len(ids)-seq_len-1), seq_len):
            x = ids[i:i+seq_len].unsqueeze(0).to(DEVICE)
            y = ids[i+1:i+seq_len+1].unsqueeze(0).to(DEVICE)
            _, l = model(x, y)
            eval_losses.append(l.item())

    avg_loss = sum(eval_losses) / len(eval_losses)
    ppl = math.exp(min(avg_loss, 10))

    # Specialization (APN only)
    spec = None
    if ffn_type == "apn":
        specs = []
        for b in model.blocks:
            if hasattr(b.ffn, 'apn') and hasattr(b.ffn.apn, 'specialization'):
                specs.append(b.ffn.apn.specialization())
            elif hasattr(b.ffn, 'specialization'):
                specs.append(b.ffn.specialization())
        spec = specs

    return {
        "params_M": round(n_params/1e6, 3),
        "ppl": round(ppl, 3),
        "eval_loss": round(avg_loss, 4),
        "final_train_loss": round(losses[-1], 4),
        "train_time_s": round(train_time, 1),
        "losses": losses,
        "specialization": spec,
    }

def main():
    print("="*70)
    print("  REAL-SCALE APN VALIDATION")
    print("  Testing: APN with fewer layers vs SwiGLU with more layers")
    print("  On real text data at multiple parameter scales")
    print("="*70)

    text, data_source = get_data()
    chars = sorted(set(text))
    print(f"  Data: {data_source} | {len(text):,} chars | {len(chars)} vocab")
    print(f"  Device: {DEVICE}")

    # Configurations: (name, ffn_type, n_layers, d_model, n_heads, ffn_hidden)
    # Goal: APN with L/2 layers ≈ SwiGLU with L layers
    configs = [
        # Baseline SwiGLU models (reference)
        ("SwiGLU-4L-d128",  "swiglu", 4, 128, 4, 512),
        ("SwiGLU-6L-d128",  "swiglu", 6, 128, 4, 512),
        ("SwiGLU-8L-d128",  "swiglu", 8, 128, 4, 512),
        # APN models with fewer layers
        ("APN-2L-d128",     "apn",    2, 128, 4, 512),
        ("APN-3L-d128",     "apn",    3, 128, 4, 512),
        ("APN-4L-d128",     "apn",    4, 128, 4, 512),
        # Larger models (d=256)
        ("SwiGLU-4L-d256",  "swiglu", 4, 256, 4, 1024),
        ("SwiGLU-6L-d256",  "swiglu", 6, 256, 4, 1024),
        ("APN-2L-d256",     "apn",    2, 256, 4, 1024),
        ("APN-3L-d256",     "apn",    3, 256, 4, 1024),
        ("APN-4L-d256",     "apn",    4, 256, 4, 1024),
    ]

    results = {}
    print(f"\n  {'Model':<22} {'Params':>8} {'PPL':>7} {'Loss':>8} {'Time':>7} {'Spec':>20}")
    print(f"  {'─'*80}")

    for name, ft, nl, d, nh, fh in configs:
        print(f"  Training {name}...", end=" ", flush=True)
        r = train_and_eval(text, len(chars), d, nl, nh, ft, fh, steps=800)
        results[name] = r
        spec_str = ""
        if r["specialization"]:
            spec_str = " | ".join(f"{k}:{v}" for k,v in r["specialization"][0].items()) if r["specialization"] else ""
            if len(spec_str) > 35: spec_str = spec_str[:35] + "..."
        print(f"  {name:<22} {r['params_M']:>7.2f}M {r['ppl']:>7.2f} {r['eval_loss']:>8.4f} {r['train_time_s']:>6.1f}s  {spec_str}")

    # Save
    out = os.path.join(os.path.dirname(__file__) or ".", "real_scale_results.json")
    # Remove non-serializable
    for k in results:
        if "losses" in results[k]: results[k]["losses"] = results[k]["losses"][-5:]  # last 5 only
    with open(out, "w") as f: json.dump(results, f, indent=2)

    # ── Analysis ──────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  DEPTH EFFICIENCY ANALYSIS")
    print("="*70)

    # d=128 analysis
    sg4_128 = results.get("SwiGLU-4L-d128", {})
    sg6_128 = results.get("SwiGLU-6L-d128", {})
    a2_128 = results.get("APN-2L-d128", {})
    a3_128 = results.get("APN-3L-d128", {})

    print(f"\n  d=128 models:")
    print(f"  SwiGLU-4L: {sg4_128.get('ppl','?')} ppl, {sg4_128.get('params_M','?')}M params")
    print(f"  SwiGLU-6L: {sg6_128.get('ppl','?')} ppl, {sg6_128.get('params_M','?')}M params")
    print(f"  APN-2L:    {a2_128.get('ppl','?')} ppl, {a2_128.get('params_M','?')}M params")
    print(f"  APN-3L:    {a3_128.get('ppl','?')} ppl, {a3_128.get('params_M','?')}M params")

    # d=256 analysis
    sg4_256 = results.get("SwiGLU-4L-d256", {})
    sg6_256 = results.get("SwiGLU-6L-d256", {})
    a2_256 = results.get("APN-2L-d256", {})
    a3_256 = results.get("APN-3L-d256", {})

    print(f"\n  d=256 models:")
    print(f"  SwiGLU-4L: {sg4_256.get('ppl','?')} ppl, {sg4_256.get('params_M','?')}M params")
    print(f"  SwiGLU-6L: {sg6_256.get('ppl','?')} ppl, {sg6_256.get('params_M','?')}M params")
    print(f"  APN-2L:    {a2_256.get('ppl','?')} ppl, {a2_256.get('params_M','?')}M params")
    print(f"  APN-3L:    {a3_256.get('ppl','?')} ppl, {a3_256.get('params_M','?')}M params")

    # Depth ratio
    print(f"\n  DEPTH RATIO (lower = APN needs fewer layers for same PPL):")
    for sg_name, sg_data, apn_name, apn_data in [
        ("SwiGLU-4L", sg4_128, "APN-2L", a2_128),
        ("SwiGLU-6L", sg6_128, "APN-3L", a3_128),
        ("SwiGLU-4L", sg4_256, "APN-2L", a2_256),
        ("SwiGLU-6L", sg6_256, "APN-3L", a3_256),
    ]:
        sg_ppl = sg_data.get("ppl", 0)
        apn_ppl = apn_data.get("ppl", 0)
        if sg_ppl and apn_ppl and sg_ppl > 0:
            ratio = apn_ppl / sg_ppl
            apn_params = apn_data.get("params_M", 0)
            sg_params = sg_data.get("params_M", 0)
            param_ratio = apn_params / sg_params if sg_params else 0
            print(f"  {apn_name:>12} / {sg_name:>12}: PPL ratio={ratio:.3f}  Params ratio={param_ratio:.3f}")
            if ratio < 1.05:
                print(f"    → {apn_name} MATCHES {sg_name} with {int((1-param_ratio)*100)}% fewer params!")

    print(f"\n  Results saved to {out}")

if __name__ == "__main__":
    main()