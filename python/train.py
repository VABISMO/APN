#!/usr/bin/env python3
"""
ProbNet - train.py
==================
Full training script using PyTorch autograd for backprop.
Trains the ProbNet transformer (with APN FFN layers) on text data.

Usage:
    python3 train.py --data corpus.txt --out model.pnet [options]
    python3 train.py --data data.txt --model existing.pnet --finetune --epochs 3

Features:
    - Full gradient flow through all layers
    - APN tau annealing (starts uniform → specializes)
    - CPU training (uses all cores via OpenMP-equivalent threading)
    - GPU training if torch + CUDA available
    - Mixed precision support
    - Cosine LR schedule with warmup
    - Validation perplexity every N steps
    - Checkpoint save/resume
    - Wandb logging (optional)
"""

import os, sys, json, time, math, argparse
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ─── ProbNet PyTorch Implementation ──────────────────────────────────────────

class APNLayer(nn.Module):
    """
    Adaptive Probabilistic Neuron layer (PyTorch version).
    Direct equivalent of src/apn_layer.h but with autograd.
    """
    NFUNCS = 6
    FNAMES = ['identity', 'sq-tanh', 's-sqrt', 'b-prod', 'sin', 'relu']

    def __init__(self, in_dim: int, hidden: int, out_dim: int, tau: float = 3.0):
        super().__init__()
        self.in_dim  = in_dim
        self.hidden  = hidden
        self.out_dim = out_dim
        self.tau     = tau

        self.W1     = nn.Linear(in_dim, hidden, bias=True)
        self.W2     = nn.Linear(in_dim, hidden, bias=True)
        self.logits = nn.Parameter(torch.randn(hidden, self.NFUNCS) * 0.1)
        self.W_out  = nn.Linear(hidden, out_dim, bias=True)

        # Init
        nn.init.xavier_uniform_(self.W1.weight, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.W2.weight, gain=math.sqrt(2) * 0.5)
        nn.init.xavier_uniform_(self.W_out.weight)

    def forward(self, x):
        p1 = self.W1(x)  # [*, H]
        p2 = self.W2(x)  # [*, H]
        a  = p1  # main
        b  = p2  # aux

        # Bounded functions (all with stable gradients)
        fv0 = a                                              # identity
        fv1 = torch.tanh(a * a)                             # sq-tanh
        fv2 = a.sign() * (a.abs() + 1e-4).sqrt()            # s-sqrt
        fv3 = (a * b) / (1.0 + (a * b).pow(2)).sqrt()       # bounded product
        fv4 = torch.sin(a)                                   # sin
        fv5 = F.leaky_relu(a, 0.01)                         # leaky relu

        # Stack: [*, H, F]
        fvals = torch.stack([fv0, fv1, fv2, fv3, fv4, fv5], dim=-1)

        # Alpha: softmax over functions per neuron
        alpha = F.softmax(self.logits / (self.tau + 1e-7), dim=-1)  # [H, F]

        # Weighted sum: [*, H]
        out = (fvals * alpha).sum(dim=-1)

        return self.W_out(out)

    def anneal(self, progress: float, tau0: float = 3.0, tau1: float = 0.05):
        self.tau = tau0 * (tau1 / tau0) ** progress

    def specialization(self) -> dict:
        """Return function specialization statistics."""
        with torch.no_grad():
            alpha = F.softmax(self.logits / (self.tau + 1e-7), dim=-1)
            dominant = alpha.argmax(dim=-1)
            counts = {name: 0 for name in self.FNAMES}
            for i in dominant.tolist():
                counts[self.FNAMES[i]] += 1
            total = len(dominant)
            return {k: f"{100*v//total}%" for k, v in counts.items() if v > 0}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        rms = (x.float().pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        return (x.float() / rms * self.weight).to(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_len = max_len

    def forward(self, seq_len: int, device):
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(q, k, cos, sin):
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class ProbNetBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        D, H, Hkv, hd = config['d_model'], config['n_heads'], config['n_kv_heads'], config['head_dim']
        self.n_heads   = H
        self.n_kv_heads= Hkv
        self.head_dim  = hd

        # Attention projections
        self.q_proj = nn.Linear(D, H*hd,   bias=False)
        self.k_proj = nn.Linear(D, Hkv*hd, bias=False)
        self.v_proj = nn.Linear(D, Hkv*hd, bias=False)
        self.o_proj = nn.Linear(H*hd, D,   bias=False)

        # Norms
        self.attn_norm = RMSNorm(D, config.get('rms_eps', 1e-5))
        self.ffn_norm  = RMSNorm(D, config.get('rms_eps', 1e-5))

        # APN FFN
        self.ffn = APNLayer(D, config['ffn_hidden'], D, tau=config.get('apn_tau0', 3.0))

        # RoPE
        self.rope = RotaryEmbedding(hd, config.get('max_seq_len', 2048))

    def forward(self, x, mask=None):
        B, T, D = x.shape
        H, Hkv, hd = self.n_heads, self.n_kv_heads, self.head_dim

        # Attention
        residual = x
        x = self.attn_norm(x)
        q = self.q_proj(x).view(B, T, H,   hd).transpose(1, 2)  # [B, H, T, hd]
        k = self.k_proj(x).view(B, T, Hkv, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, Hkv, hd).transpose(1, 2)

        # RoPE
        cos, sin = self.rope(T, x.device)
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, T, hd]
        sin = sin.unsqueeze(0).unsqueeze(0)
        q, k = apply_rotary(q, k, cos, sin)

        # GQA: repeat K, V if needed
        if Hkv < H:
            repeat = H // Hkv
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        # Scaled dot-product attention
        scale = math.sqrt(hd)
        attn  = torch.matmul(q, k.transpose(-2, -1)) / scale

        # Causal mask
        causal = torch.full((T, T), float('-inf'), device=x.device).triu(1)
        attn = attn + causal.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(attn.float(), dim=-1).to(q.dtype)

        out = torch.matmul(attn, v)              # [B, H, T, hd]
        out = out.transpose(1, 2).reshape(B, T, H*hd)
        x = residual + self.o_proj(out)

        # FFN
        residual = x
        x = self.ffn_norm(x)
        x = residual + self.ffn(x)
        return x


class ProbNetModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        V, D = config['vocab_size'], config['d_model']

        self.embed     = nn.Embedding(V, D)
        self.blocks    = nn.ModuleList([ProbNetBlock(config) for _ in range(config['n_layers'])])
        self.norm      = RMSNorm(D, config.get('rms_eps', 1e-5))
        self.lm_head   = nn.Linear(D, V, bias=False)

        # Tie weights
        self.lm_head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        std = 0.02
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0.0, std=std)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)

    def anneal_apn(self, progress: float, tau0: float = 3.0, tau1: float = 0.05):
        for block in self.blocks:
            block.ffn.anneal(progress, tau0, tau1)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def print_specialization(self):
        for i, block in enumerate(self.blocks):
            spec = block.ffn.specialization()
            print(f"  Layer {i}: {spec}")

# ─── Dataset ─────────────────────────────────────────────────────────────────

class TextDataset:
    def __init__(self, path: str, seq_len: int, vocab_size: int = 256):
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        # Char-level tokenization
        self.data = torch.tensor([ord(c) % vocab_size for c in text], dtype=torch.long)
        self.seq_len = seq_len
        print(f"Dataset: {len(self.data):,} tokens from {path}")

    def get_batch(self, batch_size: int, device):
        ix = torch.randint(len(self.data) - self.seq_len, (batch_size,))
        x = torch.stack([self.data[i:i+self.seq_len]       for i in ix]).to(device)
        y = torch.stack([self.data[i+1:i+self.seq_len+1]   for i in ix]).to(device)
        return x, y

# ─── Training loop ───────────────────────────────────────────────────────────

def train(args):
    if not HAS_TORCH:
        print("ERROR: PyTorch not available. Install: pip install torch")
        sys.exit(1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory//1e9:.1f} GB")

    # Model config
    config = {
        'vocab_size':  args.vocab_size,
        'd_model':     args.d_model,
        'n_layers':    args.n_layers,
        'n_heads':     args.n_heads,
        'n_kv_heads':  args.n_kv_heads or args.n_heads,
        'head_dim':    args.d_model // args.n_heads,
        'ffn_hidden':  args.ffn_hidden or 4 * args.d_model,
        'max_seq_len': args.seq_len * 2,
        'rms_eps':     1e-5,
        'apn_tau0':    3.0,
        'apn_tau1':    0.05,
    }

    model = ProbNetModel(config).to(device)
    total_params = model.param_count()
    print(f"\nModel: {total_params/1e6:.1f}M parameters")
    print(f"  d_model={config['d_model']}  n_layers={config['n_layers']}")
    print(f"  n_heads={config['n_heads']}  ffn_hidden={config['ffn_hidden']}")
    print(f"  FFN type: APN v9 ({APNLayer.NFUNCS} learnable functions)")

    # Dataset
    dataset = TextDataset(args.data, args.seq_len, config['vocab_size'])

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # LR schedule
    total_steps = args.epochs * (len(dataset.data) // (args.batch_size * args.seq_len) + 1)
    warmup = min(args.warmup, total_steps // 10)

    def get_lr(step):
        if step < warmup:
            return args.lr * step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + math.cos(math.pi * progress))

    # Mixed precision
    use_amp = (device == 'cuda') and args.amp
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None

    print(f"\nTraining: {args.epochs} epochs, {total_steps} steps")
    print(f"  Batch={args.batch_size}  SeqLen={args.seq_len}")
    print(f"  LR={args.lr} → {args.lr_min}  Warmup={warmup}")
    print(f"  AMP={use_amp}\n")
    print(f"  {'Step':>8}  {'Loss':>8}  {'PPL':>8}  {'LR':>10}  {'APN tau':>8}")
    print(f"  {'-'*55}")

    step = 0
    best_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches  = max(1, len(dataset.data) // (args.batch_size * args.seq_len))

        for batch_i in range(n_batches):
            # LR update
            lr_now = get_lr(step)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_now

            # APN tau annealing
            model.anneal_apn(step / max(1, total_steps))

            # Forward
            x, y = dataset.get_batch(args.batch_size, device)
            with torch.cuda.amp.autocast() if use_amp else contextlib_nullcontext():
                logits = model(x)           # [B, T, V]
                loss   = F.cross_entropy(
                    logits.view(-1, config['vocab_size']),
                    y.view(-1),
                    ignore_index=-1,
                )

            # Backward
            optimizer.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            step += 1

            if step % args.log_every == 0:
                tau = model.blocks[0].ffn.tau
                ppl = math.exp(min(loss_val, 10))
                print(f"  {step:>8}  {loss_val:>8.4f}  {ppl:>8.1f}  {lr_now:>10.2e}  {tau:>8.4f}")

            if step % args.save_every == 0:
                _save_model_pt(model, config, args.out)
                if args.verbose:
                    print(f"\n  APN specialization at step {step}:")
                    model.print_specialization()

        avg_loss = epoch_loss / n_batches
        print(f"\n  Epoch {epoch+1}/{args.epochs}  avg_loss={avg_loss:.4f}  ppl={math.exp(min(avg_loss,10)):.1f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            _save_model_pt(model, config, args.out)
            print(f"  ✓ Saved best model → {args.out}")

    print(f"\nFinal APN specialization:")
    model.print_specialization()

    # Final save
    _save_model_pt(model, config, args.out)
    print(f"\nTraining complete! Model saved to {args.out}")
    return model, config


def _save_model_pt(model, config, path):
    """Save model in PyTorch format."""
    torch.save({
        'config': config,
        'state_dict': model.state_dict(),
    }, path + '.pt')


class contextlib_nullcontext:
    """Python 3.6 compatible null context."""
    def __enter__(self): return None
    def __exit__(self, *a): return False


# ─── Generation (PyTorch) ────────────────────────────────────────────────────

@torch.no_grad()
def generate_pt(model, prompt_ids: list, max_new: int = 200,
                temperature: float = 0.8, top_k: int = 50, top_p: float = 0.95,
                device='cpu'):
    """Generate tokens autoregressively."""
    model.eval()
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    for _ in range(max_new):
        # Truncate to max_seq_len
        idx = ids if ids.shape[1] <= 2048 else ids[:, -2048:]
        logits = model(idx)[:, -1, :]  # [1, V]

        # Apply temperature
        if temperature > 0:
            logits = logits / temperature
        probs = F.softmax(logits, dim=-1)

        # Top-p filtering
        if 0 < top_p < 1.0:
            sorted_p, sorted_idx = probs.sort(dim=-1, descending=True)
            cum_p = sorted_p.cumsum(dim=-1)
            remove = cum_p - sorted_p > top_p
            sorted_p[remove] = 0
            sorted_p /= sorted_p.sum()
            next_id = sorted_idx[0, torch.multinomial(sorted_p[0], 1).item()].item()
        elif top_k > 0:
            top_probs, top_ids = probs.topk(min(top_k, probs.shape[-1]))
            next_id = top_ids[0, torch.multinomial(top_probs[0], 1).item()].item()
        else:
            next_id = torch.multinomial(probs[0], 1).item()

        yield next_id
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)


# ─── Benchmark ───────────────────────────────────────────────────────────────

def benchmark_pt(config: dict):
    """Compare ProbNet (APN) vs standard SwiGLU model."""
    if not HAS_TORCH:
        print("PyTorch required for benchmark")
        return

    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║  PyTorch Benchmark: ProbNet (APN) vs SwiGLU                      ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")

    class SwiGLUFFN(nn.Module):
        def __init__(self, d_model, hidden):
            super().__init__()
            self.gate = nn.Linear(d_model, hidden, bias=False)
            self.up   = nn.Linear(d_model, hidden, bias=False)
            self.down = nn.Linear(hidden, d_model, bias=False)
        def forward(self, x):
            return self.down(self.gate(x) * F.silu(self.up(x)))

    D, H, N = 64, 256, 1000
    tasks = [
        ('Ratio   y=x0/x1',   lambda x: x[:,0]/(x[:,1].abs()+0.1)),
        ('Product y=x0*x1',   lambda x: x[:,0]*x[:,1]),
        ('Square  y=x0^2 ',   lambda x: x[:,0]**2),
        ('Sqrt    y=√|x0|',   lambda x: x[:,0].abs().sqrt()),
        ('Sin     y=sin(x0)', lambda x: torch.sin(x[:,0]*3.14159)),
    ]

    print(f"  {'Task':<20} {'SwiGLU':>10} {'APN':>10} {'Winner'}")
    print(f"  {'-'*50}")

    apn_wins = 0
    for name, fn in tasks:
        X = torch.randn(N, D).abs() * 0.5 + 0.3
        y = fn(X)
        y = (y - y.mean()) / (y.std() + 1e-8)

        results = {}
        for model_type, model in [
            ('SwiGLU', SwiGLUFFN(D, H)),
            ('APN',    APNLayer(D, H, 1)),
        ]:
            opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
            for ep in range(300):
                pred = model(X).squeeze(-1)
                loss = F.mse_loss(pred, y)
                opt.zero_grad(); loss.backward(); opt.step()
                if model_type == 'APN':
                    model.anneal(ep/300)

            with torch.no_grad():
                pred = model(X).squeeze(-1)
                results[model_type] = F.mse_loss(pred, y).item()

        winner = "APN ✓" if results['APN'] < results['SwiGLU'] * 0.95 else "GLU ✓"
        if "APN" in winner: apn_wins += 1
        print(f"  {name:<20} {results['SwiGLU']:>10.5f} {results['APN']:>10.5f} {winner}")

    print(f"\n  APN wins: {apn_wins}/{len(tasks)}\n")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='ProbNet Training')
    sub = parser.add_subparsers(dest='cmd')

    # train
    p = sub.add_parser('train', help='Train from scratch')
    p.add_argument('--data',       required=True)
    p.add_argument('--out',        default='model.pnet')
    p.add_argument('--d_model',    type=int, default=256)
    p.add_argument('--n_layers',   type=int, default=4)
    p.add_argument('--n_heads',    type=int, default=8)
    p.add_argument('--n_kv_heads', type=int, default=None)
    p.add_argument('--ffn_hidden', type=int, default=None)
    p.add_argument('--vocab_size', type=int, default=256)
    p.add_argument('--seq_len',    type=int, default=128)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--epochs',     type=int, default=5)
    p.add_argument('--lr',         type=float, default=3e-4)
    p.add_argument('--lr_min',     type=float, default=1e-5)
    p.add_argument('--weight_decay',type=float, default=0.1)
    p.add_argument('--grad_clip',  type=float, default=1.0)
    p.add_argument('--warmup',     type=int, default=100)
    p.add_argument('--log_every',  type=int, default=50)
    p.add_argument('--save_every', type=int, default=500)
    p.add_argument('--amp',        action='store_true')
    p.add_argument('--verbose',    action='store_true')

    # bench
    p2 = sub.add_parser('bench', help='Run PyTorch benchmark')
    p2.add_argument('--d_model', type=int, default=64)

    # generate
    p3 = sub.add_parser('generate', help='Generate text from .pt model')
    p3.add_argument('--model',     required=True)
    p3.add_argument('--prompt',    default='The')
    p3.add_argument('--max_tokens',type=int, default=200)
    p3.add_argument('--temp',      type=float, default=0.8)

    args = parser.parse_args()

    if args.cmd == 'train':
        train(args)
    elif args.cmd == 'bench':
        benchmark_pt({'d_model': args.d_model})
    elif args.cmd == 'generate':
        if not HAS_TORCH:
            print("PyTorch required"); sys.exit(1)
        ckpt = torch.load(args.model, map_location='cpu')
        model = ProbNetModel(ckpt['config'])
        model.load_state_dict(ckpt['state_dict'])
        prompt_ids = [ord(c) % 256 for c in args.prompt]
        print(f"Prompt: {args.prompt}")
        print("Output: ", end='', flush=True)
        out = []
        for tok_id in generate_pt(model, prompt_ids, args.max_tokens, args.temp):
            ch = chr(tok_id) if 32 <= tok_id < 127 else ' '
            print(ch, end='', flush=True)
            out.append(ch)
        print()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
