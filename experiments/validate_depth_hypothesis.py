#!/usr/bin/env python3
"""
APN Depth Hypothesis Validation
================================

Hypothesis: APN layers can express more per layer than SwiGLU,
so APN needs fewer layers to achieve the same capability.

This script validates empirically:
  1. Function approximation: can 1 APN layer solve what SwiGLU needs 2+ for?
  2. Depth scaling: APN with L/2 layers ≈ SwiGLU with L layers?
  3. Convergence speed and parameter efficiency
  4. Real language modeling perplexity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import json
import os
import sys

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ================================================================
# 1. APN Function Bank (identical to apn_complete.py)
# ================================================================

class APNFunction(nn.Module):
    NFUNCS = 6
    NAMES = ["identity", "sq-tanh", "s-sqrt", "b-prod", "sin", "relu"]

    def __init__(self, hidden, tau=3.0):
        super().__init__()
        self.hidden = hidden
        self.tau = tau
        self.logits = nn.Parameter(torch.randn(hidden, self.NFUNCS) * 0.1)

    def forward(self, p1, p2):
        a, b = p1, p2
        f0 = a
        f1 = torch.tanh(a * a)
        f2 = a.sign() * (a.abs() + 1e-4).sqrt()
        f3 = (a * b) / ((a * b).pow(2) + 1.0).sqrt()
        f4 = torch.sin(a)
        f5 = F.leaky_relu(a, 0.01)
        fvals = torch.stack([f0, f1, f2, f3, f4, f5], dim=-1)
        alpha = F.softmax(self.logits / (self.tau + 1e-7), dim=-1)
        return (fvals * alpha).sum(dim=-1)

    def anneal(self, progress, tau0=3.0, tau1=0.05):
        self.tau = tau0 * (tau1 / tau0) ** progress


class APNLayer(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, tau=3.0):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.out_dim = out_dim
        self.W1 = nn.Linear(in_dim, hidden, bias=True)
        self.W2 = nn.Linear(in_dim, hidden, bias=True)
        self.apn = APNFunction(hidden, tau)
        self.W_out = nn.Linear(hidden, out_dim, bias=True)
        std = math.sqrt(2.0 / in_dim)
        nn.init.normal_(self.W1.weight, 0, std)
        nn.init.zeros_(self.W1.bias)
        nn.init.normal_(self.W2.weight, 0, std * 0.5)
        nn.init.zeros_(self.W2.bias)
        nn.init.normal_(self.W_out.weight, 0, math.sqrt(2.0 / hidden))
        nn.init.zeros_(self.W_out.bias)

    def forward(self, x):
        return self.W_out(self.apn(self.W1(x), self.W2(x)))

    def anneal(self, progress, tau0=3.0, tau1=0.05):
        self.apn.anneal(progress, tau0, tau1)


# ================================================================
# 2. Baseline architectures
# ================================================================

class SwiGLULayer(nn.Module):
    """Standard SwiGLU FFN: gate * silu(up) -> down"""
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.gate = nn.Linear(in_dim, hidden, bias=False)
        self.up = nn.Linear(in_dim, hidden, bias=False)
        self.down = nn.Linear(hidden, out_dim, bias=False)

    def forward(self, x):
        return self.down(self.gate(x) * F.silu(self.up(x)))


class GELULayer(nn.Module):
    """Standard GELU FFN: GELU(fc1) -> fc2"""
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, out_dim, bias=True)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


# ================================================================
# 3. Transformer blocks for language modeling
# ================================================================

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_type="apn", ffn_hidden=None, tau=3.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, bias=False)
        self.ln2 = nn.LayerNorm(d_model)
        ffn_hidden = ffn_hidden or d_model * 4
        if ffn_type == "apn":
            self.ffn = APNLayer(d_model, ffn_hidden, d_model, tau)
        elif ffn_type == "swiglu":
            self.ffn = SwiGLULayer(d_model, ffn_hidden, d_model)
        elif ffn_type == "gelu":
            self.ffn = GELULayer(d_model, ffn_hidden, d_model)
        else:
            raise ValueError(ffn_type)

    def forward(self, x):
        T = x.shape[1]
        msk = torch.triu(torch.ones(T, T, device=x.device), 1).bool()
        h, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=msk, need_weights=False)
        x = x + h
        x = x + self.ffn(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, ffn_type="apn",
                 ffn_hidden=None, tau=3.0, max_seq=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_type, ffn_hidden, tau)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.emb.weight  # tie weights
        self.ffn_type = ffn_type
        self.n_layers = n_layers
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.emb(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        x = self.ln_f(x)
        lg = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(lg.view(-1, lg.size(-1)), targets.view(-1))
        return lg, loss


# ================================================================
# EXPERIMENT 1: Single-layer function approximation
# ================================================================

def exp_single_layer_approximation():
    """
    Can 1 APN layer approximate functions that 1 SwiGLU layer cannot?
    Test on: ratio, product, sqrt, sin, x^2, |x|, composition
    """
    print("\n" + "="*70)
    print("  EXPERIMENT 1: Single-layer function approximation")
    print("  Can APN express in 1 layer what SwiGLU needs 2+ for?")
    print("="*70)

    N = 2000
    torch.manual_seed(42)
    X = torch.randn(N, 4).to(DEVICE)
    X = X.abs() * 0.5 + 0.1  # positive values for ratio/sqrt

    tasks = {
        "ratio (x0/x1)": lambda x: (x[:, 0] / (x[:, 1] + 0.05)).unsqueeze(1),
        "product (x0*x1)": lambda x: (x[:, 0] * x[:, 1]).unsqueeze(1),
        "sqrt (sqrt(x0))": lambda x: torch.sqrt(x[:, 0].abs() + 1e-4).unsqueeze(1),
        "sin (sin(x0))": lambda x: torch.sin(x[:, 0] * 3.14).unsqueeze(1),
        "x^2": lambda x: (x[:, 0] ** 2).unsqueeze(1),
        "|x|": lambda x: x[:, 0].abs().unsqueeze(1),
        "1/x": lambda x: (1.0 / (x[:, 0].abs() + 0.1)).unsqueeze(1),
        "x0^2+x1^2": lambda x: (x[:, 0]**2 + x[:, 1]**2).unsqueeze(1),
    }

    H = 64  # hidden dimension
    results = {}

    for name, fn in tasks.items():
        y = fn(X)
        y = (y - y.mean()) / (y.std() + 1e-8)  # normalize
        X_tr, y_tr = X[:1600], y[:1600]
        X_te, y_te = X[1600:], y[1600:]

        # Train each model
        models = {
            "APN-1L": APNLayer(4, H, 1),
            "SwiGLU-1L": SwiGLULayer(4, H, 1),
            "GELU-1L": GELULayer(4, H, 1),
            "APN-2L": nn.Sequential(APNLayer(4, H, H), APNLayer(H, H, 1)),
            "SwiGLU-2L": nn.Sequential(SwiGLULayer(4, H, H), SwiGLULayer(H, H, 1)),
        }

        task_results = {}
        for mname, model in models.items():
            model = model.to(DEVICE)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

            # Anneal schedule for APN
            n_steps = 1000
            for step in range(n_steps):
                prog = step / n_steps
                for m in model.modules():
                    if isinstance(m, APNFunction):
                        m.anneal(prog, 3.0, 0.05)
                opt.zero_grad()
                pred = model(X_tr)
                loss = F.mse_loss(pred, y_tr)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            with torch.no_grad():
                test_loss = F.mse_loss(model(X_te), y_te).item()

            # Count params
            n_params = sum(p.numel() for p in model.parameters())
            task_results[mname] = {"mse": round(test_loss, 6), "params": n_params}

        results[name] = task_results

        # Print comparison
        apn1 = task_results["APN-1L"]["mse"]
        sgl1 = task_results["SwiGLU-1L"]["mse"]
        gelu1 = task_results["GELU-1L"]["mse"]
        apn2 = task_results["APN-2L"]["mse"]
        sgl2 = task_results["SwiGLU-2L"]["mse"]

        winner_1l = "APN" if apn1 < min(sgl1, gelu1) * 0.9 else ("SwiGLU" if sgl1 < apn1 else "GELU")
        depth_win = "APN-1L ≈ SwiGLU-2L" if abs(apn1 - sgl2) / max(apn1, sgl2, 1e-8) < 0.15 else ""

        print(f"  {name:<20}  APN-1L={apn1:.4f}  SwiGLU-1L={sgl1:.4f}  GELU-1L={gelu1:.4f}  "
              f"APN-2L={apn2:.4f}  SwiGLU-2L={sgl2:.4f}  1L-winner={winner_1l}  {depth_win}")

    return results


# ================================================================
# EXPERIMENT 2: Depth scaling — fewer APN layers ≈ more SwiGLU layers?
# ================================================================

def exp_depth_scaling():
    """
    Train language models with:
      - SwiGLU: 4, 6, 8 layers
      - APN: 2, 3, 4, 6, 8 layers
    Compare perplexity vs parameters.
    If hypothesis holds: APN-3L ≈ SwiGLU-6L
    """
    print("\n" + "="*70)
    print("  EXPERIMENT 2: Depth scaling — APN with fewer layers vs SwiGLU")
    print("="*70)

    # Generate a more challenging corpus
    import random
    random.seed(42)
    lines = []
    # Math patterns
    for i in range(1, 20):
        for j in range(1, 20):
            lines.append(f"{i}+{j}={i+j}")
            lines.append(f"{i}*{j}={i*j}")
    # Natural language patterns
    subjects = ["the cat", "a dog", "the bird", "my friend", "the king"]
    verbs = ["sat on", "ran to", "jumped over", "looked at", "walked past"]
    objects = ["the mat", "the door", "a tree", "the river", "the house"]
    for _ in range(500):
        s, v, o = random.choice(subjects), random.choice(verbs), random.choice(objects)
        lines.append(f"{s} {v} {o}")
    # Code-like patterns
    for _ in range(300):
        lines.append(f"x = {random.randint(0,9)} ; y = {random.randint(0,9)} ; z = x + y")
    # Repeated structures (tests hierarchical learning)
    for _ in range(200):
        lines.append(f"if x > {random.randint(0,5)} then {random.choice(['yes','no','true','false'])}")

    text = "\n".join(lines) + "\n" + (lines[0] + " ") * 100
    # Build vocab
    chars = sorted(set(text))
    char2id = {c: i for i, c in enumerate(chars)}
    vocab_size = len(chars)
    ids = torch.tensor([char2id[c] for c in text], dtype=torch.long)

    SEQ = 128
    B = 32
    D = 128
    HEADS = 4

    configs = [
        # (name, ffn_type, n_layers)
        ("SwiGLU-4L", "swiglu", 4),
        ("SwiGLU-6L", "swiglu", 6),
        ("SwiGLU-8L", "swiglu", 8),
        ("APN-2L",    "apn",    2),
        ("APN-3L",    "apn",    3),
        ("APN-4L",    "apn",    4),
        ("APN-6L",    "apn",    6),
        ("APN-8L",    "apn",    8),
    ]

    results = {}
    for name, ffn_type, n_layers in configs:
        torch.manual_seed(42)
        model = TinyGPT(vocab_size, D, n_layers, HEADS, ffn_type).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.1)

        STEPS = 1500
        best_loss = float('inf')
        losses = []

        for step in range(STEPS):
            prog = step / STEPS
            # Anneal APN
            for m in model.modules():
                if isinstance(m, APNFunction):
                    m.anneal(prog, 3.0, 0.05)

            starts = torch.randint(0, len(ids) - SEQ - 1, (B,))
            x = torch.stack([ids[s:s+SEQ] for s in starts]).to(DEVICE)
            y = torch.stack([ids[s+1:s+SEQ+1] for s in starts]).to(DEVICE)

            _, loss = model(x, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            losses.append(loss.item())
            if loss.item() < best_loss:
                best_loss = loss.item()

        # Evaluate final perplexity
        model.eval()
        with torch.no_grad():
            eval_losses = []
            for i in range(0, min(2000, len(ids) - SEQ - 1), SEQ):
                x = ids[i:i+SEQ].unsqueeze(0).to(DEVICE)
                y = ids[i+1:i+SEQ+1].unsqueeze(0).to(DEVICE)
                _, l = model(x, y)
                eval_losses.append(l.item())
            avg_loss = sum(eval_losses) / len(eval_losses)
            ppl = math.exp(min(avg_loss, 10))

        results[name] = {
            "params_M": round(n_params / 1e6, 2),
            "best_train_loss": round(best_loss, 4),
            "eval_loss": round(avg_loss, 4),
            "eval_ppl": round(ppl, 2),
            "final_train_loss": round(losses[-1], 4),
        }
        print(f"  {name:<14}  params={n_params/1e6:.2f}M  eval_ppl={ppl:.2f}  "
              f"eval_loss={avg_loss:.4f}  best_train={best_loss:.4f}")

    # Summary: can APN with fewer layers match SwiGLU?
    print("\n  ── Depth Efficiency Summary ──")
    sg4 = results.get("SwiGLU-4L", {}).get("eval_ppl", 999)
    sg6 = results.get("SwiGLU-6L", {}).get("eval_ppl", 999)
    sg8 = results.get("SwiGLU-8L", {}).get("eval_ppl", 999)
    a2 = results.get("APN-2L", {}).get("eval_ppl", 999)
    a3 = results.get("APN-3L", {}).get("eval_ppl", 999)
    a4 = results.get("APN-4L", {}).get("eval_ppl", 999)
    a6 = results.get("APN-6L", {}).get("eval_ppl", 999)

    # Find closest matches
    for apn_name, apn_ppl in [("APN-2L", a2), ("APN-3L", a3), ("APN-4L", a4)]:
        for sg_name, sg_ppl in [("SwiGLU-4L", sg4), ("SwiGLU-6L", sg6), ("SwiGLU-8L", sg8)]:
            diff_pct = abs(apn_ppl - sg_ppl) / max(apn_ppl, sg_ppl) * 100
            if diff_pct < 15:
                print(f"  {apn_name} (ppl={apn_ppl:.2f}) ≈ {sg_name} (ppl={sg_ppl:.2f})  diff={diff_pct:.1f}%")

    return results


# ================================================================
# EXPERIMENT 3: Per-layer expressivity test
# ================================================================

def exp_per_layer_capacity():
    """
    Measure how much each layer type can learn from data.
    Same architecture, just swap FFN type. More learning per layer = fewer layers needed.
    """
    print("\n" + "="*70)
    print("  EXPERIMENT 3: Per-layer learning capacity")
    print("  How much does each individual layer learn?")
    print("="*70)

    # Create a dataset with known structure
    torch.manual_seed(42)
    N, D_in, D_out = 4000, 16, 16

    # Target: a nonlinear transformation
    X = torch.randn(N, D_in).to(DEVICE)

    # Complex target: ratio + product + composition
    Y = torch.zeros(N, D_out).to(DEVICE)
    Y[:, 0] = X[:, 0] / (X[:, 1].abs() + 0.1)  # ratio
    Y[:, 1] = X[:, 2] * X[:, 3]  # product
    Y[:, 2] = X[:, 4].abs().sqrt()  # sqrt
    Y[:, 3] = torch.sin(X[:, 5] * 3.14)  # sin
    Y[:, 4] = X[:, 6] ** 2  # square
    Y[:, 5] = X[:, 7].abs()  # abs
    Y[:, 6] = X[:, 8] / (1 + X[:, 9].abs())  # saturation
    Y[:, 7] = X[:, 10] * X[:, 11] / (1 + (X[:, 10] * X[:, 11]).abs())  # bounded product
    # rest are linear combinations
    for i in range(8, D_out):
        Y[:, i] = X[:, i % D_in] * 0.5 + X[:, (i + 1) % D_in] * 0.3
    # Normalize
    Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-8)

    X_tr, Y_tr = X[:3200], Y[:3200]
    X_te, Y_te = X[3200:], Y[3200:]

    H = 128
    layer_types = {
        "APN-1L": nn.Sequential(APNLayer(D_in, H, D_out)),
        "SwiGLU-1L": nn.Sequential(SwiGLULayer(D_in, H, D_out)),
        "GELU-1L": nn.Sequential(GELULayer(D_in, H, D_out)),
        "APN-2L": nn.Sequential(APNLayer(D_in, H, H), APNLayer(H, H, D_out)),
        "SwiGLU-2L": nn.Sequential(SwiGLULayer(D_in, H, H), SwiGLULayer(H, H, D_out)),
        "Linear": nn.Sequential(nn.Linear(D_in, H), nn.Linear(H, D_out)),
    }

    results = {}
    for name, model in layer_types.items():
        model = model.to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2000)

        for step in range(2000):
            prog = step / 2000
            for m in model.modules():
                if isinstance(m, APNFunction):
                    m.anneal(prog, 3.0, 0.05)
            opt.zero_grad()
            pred = model(X_tr)
            loss = F.mse_loss(pred, Y_tr)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()

        with torch.no_grad():
            test_mse = F.mse_loss(model(X_te), Y_te).item()
            # Per-dimension R^2
            pred = model(X_te)
            ss_res = ((Y_te - pred) ** 2).sum(0)
            ss_tot = ((Y_te - Y_te.mean(0)) ** 2).sum(0)
            r2 = (1 - ss_res / (ss_tot + 1e-8)).mean().item()

        results[name] = {
            "params": n_params,
            "mse": round(test_mse, 6),
            "r2": round(r2, 4),
        }
        print(f"  {name:<14}  params={n_params:>8}  MSE={test_mse:.6f}  R²={r2:.4f}")

    # Key comparison: APN-1L vs SwiGLU-1L
    apn1_mse = results["APN-1L"]["mse"]
    sg1_mse = results["SwiGLU-1L"]["mse"]
    apn1_r2 = results["APN-1L"]["r2"]
    sg1_r2 = results["SwiGLU-1L"]["r2"]

    print(f"\n  Per-layer advantage:")
    print(f"    APN-1L MSE improvement over SwiGLU-1L: {(1-apn1_mse/sg1_mse)*100:.1f}%")
    print(f"    APN-1L R² advantage: {apn1_r2 - sg1_r2:.4f}")

    apn2_mse = results["APN-2L"]["mse"]
    sg2_mse = results["SwiGLU-2L"]["mse"]
    print(f"    APN-2L MSE improvement over SwiGLU-2L: {(1-apn2_mse/sg2_mse)*100:.1f}%")

    # Depth efficiency: does APN-1L match SwiGLU-2L?
    print(f"\n  Depth efficiency check:")
    print(f"    APN-1L MSE={apn1_mse:.6f}  vs  SwiGLU-2L MSE={sg2_mse:.6f}")
    if apn1_mse < sg2_mse * 1.1:
        print(f"    → APN-1L APPROXIMATELY MATCHES SwiGLU-2L (within 10%)")
    elif apn1_mse < sg1_mse:
        print(f"    → APN-1L is better than SwiGLU-1L but cannot match SwiGLU-2L")
    else:
        print(f"    → APN-1L does not outperform SwiGLU-1L")

    return results


# ================================================================
# EXPERIMENT 4: APN specialization analysis
# ================================================================

def exp_specialization():
    """
    After training, which functions do APN neurons specialize in?
    This tells us if the function bank is actually used.
    """
    print("\n" + "="*70)
    print("  EXPERIMENT 4: APN neuron specialization after training")
    print("="*70)

    # Train a small model and check specialization
    torch.manual_seed(42)
    N, D = 3000, 8
    X = torch.randn(N, D).to(DEVICE)
    X = X.abs() * 0.5 + 0.1
    # Target with mixed nonlinearities
    Y = torch.zeros(N, 1).to(DEVICE)
    Y[:, 0] = X[:, 0] / (X[:, 1] + 0.05) + torch.sin(X[:, 2]) + X[:, 3] ** 2
    Y = (Y - Y.mean()) / (Y.std() + 1e-8)

    H = 32
    model = APNLayer(D, H, 1).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for step in range(1500):
        prog = step / 1500
        model.apn.anneal(prog, 3.0, 0.05)
        opt.zero_grad()
        loss = F.mse_loss(model(X[:2500]), Y[:2500])
        loss.backward()
        opt.step()

    # Analyze specialization
    with torch.no_grad():
        alpha = F.softmax(model.apn.logits / (model.apn.tau + 1e-7), dim=-1)
        dominant = alpha.argmax(dim=-1)

    counts = {}
    for name, idx in zip(APNFunction.NAMES, range(6)):
        c = (dominant == idx).sum().item()
        pct = c / H * 100
        if c > 0:
            counts[name] = pct
            print(f"    {name:<12}: {c:3d}/{H} neurons ({pct:.0f}%)")

    print(f"\n  Final tau: {model.apn.tau:.4f}")

    # Test loss
    with torch.no_grad():
        test_loss = F.mse_loss(model(X[2500:]), Y[2500:]).item()
    print(f"  Test MSE: {test_loss:.6f}")

    return counts


# ================================================================
# EXPERIMENT 5: Speed benchmark
# ================================================================

def exp_speed():
    """Measure forward+backward time per layer."""
    print("\n" + "="*70)
    print("  EXPERIMENT 5: Speed benchmark (forward+backward per layer)")
    print("="*70)

    D, H, B = 512, 2048, 8
    x = torch.randn(B, D, device=DEVICE)

    layers = {
        "APN": APNLayer(D, H, D).to(DEVICE),
        "SwiGLU": SwiGLULayer(D, H, D).to(DEVICE),
        "GELU": GELULayer(D, H, D).to(DEVICE),
    }

    # Warmup
    for name, layer in layers.items():
        for _ in range(10):
            y = layer(x)
            y.sum().backward()

    N_ITER = 1000
    for name, layer in layers.items():
        # Forward only
        torch.cuda.synchronize() if DEVICE.type == "cuda" else None
        t0 = time.time()
        with torch.no_grad():
            for _ in range(N_ITER):
                y = layer(x)
        torch.cuda.synchronize() if DEVICE.type == "cuda" else None
        fwd_time = (time.time() - t0) / N_ITER * 1000

        # Forward + backward
        torch.cuda.synchronize() if DEVICE.type == "cuda" else None
        t0 = time.time()
        for _ in range(N_ITER):
            y = layer(x)
            y.sum().backward()
        torch.cuda.synchronize() if DEVICE.type == "cuda" else None
        fwdbwd_time = (time.time() - t0) / N_ITER * 1000

        n_params = sum(p.numel() for p in layer.parameters())
        print(f"  {name:<10}  params={n_params/1e6:.2f}M  "
              f"fwd={fwd_time:.2f}ms  fwd+bwd={fwdbwd_time:.2f}ms")

    # APN with hard argmax (inference optimization)
    print("\n  APN inference optimization: softmax vs hard argmax")
    apn = APNLayer(D, H, D).to(DEVICE)
    # Train to get meaningful weights
    for _ in range(100):
        y = apn(x)
        y.sum().backward()

    # Softmax (current)
    torch.cuda.synchronize() if DEVICE.type == "cuda" else None
    t0 = time.time()
    with torch.no_grad():
        for _ in range(N_ITER):
            y = apn(x)
    torch.cuda.synchronize() if DEVICE.type == "cuda" else None
    softmax_time = (time.time() - t0) / N_ITER * 1000

    # Hard argmax (optimized)
    # Replace softmax with one-hot of argmax
    with torch.no_grad():
        alpha = F.softmax(apn.apn.logits / (apn.apn.tau + 1e-7), dim=-1)
        dominant = alpha.argmax(dim=-1)
        hard_alpha = torch.zeros_like(alpha)
        hard_alpha.scatter_(1, dominant.unsqueeze(1), 1.0)

    # Temporarily replace logits
    orig_logits = apn.apn.logits.data.clone()
    # Make logits so that softmax gives hard assignment
    apn.apn.logits.data.copy_(hard_alpha * 20.0 - 10.0)  # sharpened to near-one-hot

    torch.cuda.synchronize() if DEVICE.type == "cuda" else None
    t0 = time.time()
    with torch.no_grad():
        for _ in range(N_ITER):
            y = apn(x)
    torch.cuda.synchronize() if DEVICE.type == "cuda" else None
    hard_time = (time.time() - t0) / N_ITER * 1000

    apn.apn.logits.data.copy_(orig_logits)  # restore

    print(f"    Softmax (current):  {softmax_time:.2f}ms")
    print(f"    Hard argmax:        {hard_time:.2f}ms")
    print(f"    Speedup:            {softmax_time/max(hard_time,0.01):.2f}x")


# ================================================================
# MAIN
# ================================================================

def main():
    print("\n" + "="*70)
    print("  APN DEPTH HYPOTHESIS VALIDATION")
    print("  Testing: Can APN achieve same capability with fewer layers?")
    print("="*70)
    print(f"  Device: {DEVICE}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")

    results = {}

    results["exp1_single_layer"] = exp_single_layer_approximation()
    results["exp2_depth_scaling"] = exp_depth_scaling()
    results["exp3_per_layer"] = exp_per_layer_capacity()
    results["exp4_specialization"] = exp_specialization()
    exp_speed()  # Speed doesn't save well (timing varies)

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "validation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    # Final verdict
    print("\n" + "="*70)
    print("  VERDICT")
    print("="*70)

    e1 = results["exp1_single_layer"]
    e3 = results["exp3_per_layer"]

    # Count how many tasks APN-1L beats SwiGLU-1L
    apn_wins = 0
    sgl_wins = 0
    for task, task_results in e1.items():
        apn_mse = task_results["APN-1L"]["mse"]
        sgl_mse = task_results["SwiGLU-1L"]["mse"]
        if apn_mse < sgl_mse * 0.9:
            apn_wins += 1
        elif sgl_mse < apn_mse * 0.9:
            sgl_wins += 1

    apn1_r2 = e3.get("APN-1L", {}).get("r2", 0)
    sg1_r2 = e3.get("SwiGLU-1L", {}).get("r2", 0)
    depth_adv = e3.get("APN-1L", {}).get("mse", 1) / e3.get("SwiGLU-2L", {}).get("mse", 1)

    print(f"  Single-layer tasks where APN wins: {apn_wins}/{len(e1)}")
    print(f"  Single-layer tasks where SwiGLU wins: {sgl_wins}/{len(e1)}")
    print(f"  APN-1L R²: {apn1_r2:.4f}  vs  SwiGLU-1L R²: {sg1_r2:.4f}")
    print(f"  APN-1L MSE / SwiGLU-2L MSE: {depth_adv:.4f}")
    if depth_adv < 1.1:
        print(f"  → APN-1L APPROXIMATELY MATCHES SwiGLU-2L")
    print()
    if apn_wins > sgl_wins and depth_adv < 1.5:
        print("  HYPOTHESIS SUPPORTED: APN has higher per-layer expressivity.")
        print("  Fewer APN layers may suffice for comparable capability.")
    elif apn_wins > sgl_wins:
        print("  HYPOTHESIS PARTIALLY SUPPORTED: APN-1L outperforms SwiGLU-1L")
        print("  but does NOT match SwiGLU-2L with a single layer.")
    else:
        print("  HYPOTHESIS NOT SUPPORTED: APN does not show clear")
        print("  per-layer advantage over SwiGLU on these tasks.")


if __name__ == "__main__":
    main()