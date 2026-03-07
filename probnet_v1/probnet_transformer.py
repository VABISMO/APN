"""
ProbNet Transformer (v1)
========================
A standard GPT-style transformer where every nn.Linear has been replaced
by ProbNetLayer (deterministic, ratio-based prediction, no weight matrices).

This is the ORIGINAL version from session 1.
For the APN v9 version (with learned adaptive functions), use probnet_complete.py

Usage
-----
# Train from scratch
from probnet_transformer import ProbNetTransformerConfig, ProbNetTransformerLM
cfg   = ProbNetTransformerConfig(vocab_size=50257, n_layer=6, n_head=8, n_embd=512)
model = ProbNetTransformerLM(cfg)

# Generate text
from probnet_transformer import generate
text = generate(model, tokenizer, "The Fibonacci sequence is", max_new=50)

# Convert any HuggingFace model
from probnet_transformer import convert_to_probnet
hf_model = convert_to_probnet(hf_model, window=8)

# OEIS sequence prediction (no model needed)
from probnet_transformer import predict_sequence
print(predict_sequence([1,1,2,3,5,8,13,21], n_next=4))  # [34, 55, 89, 144]
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from probnet_layer import ProbNetLayer, probnet_predict


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ProbNetTransformerConfig:
    vocab_size:  int   = 50257
    n_layer:     int   = 6
    n_head:      int   = 8
    n_embd:      int   = 512
    block_size:  int   = 1024
    dropout:     float = 0.0
    window:      int   = 8        # ProbNet window per output neuron
    bias:        bool  = True


# ---------------------------------------------------------------------------
# Building blocks — all linear layers → ProbNetLayer
# ---------------------------------------------------------------------------

def _Linear(in_f, out_f, cfg: ProbNetTransformerConfig, bias=True):
    """Factory: ProbNetLayer with config defaults."""
    return ProbNetLayer(in_f, out_f, window=cfg.window, bias=bias and cfg.bias)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ProbNetTransformerConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head  = cfg.n_head
        self.n_embd  = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head

        # Q, K, V, O projections — all ProbNet
        self.q_proj  = _Linear(cfg.n_embd, cfg.n_embd, cfg)
        self.k_proj  = _Linear(cfg.n_embd, cfg.n_embd, cfg)
        self.v_proj  = _Linear(cfg.n_embd, cfg.n_embd, cfg)
        self.out_proj = _Linear(cfg.n_embd, cfg.n_embd, cfg)
        self.drop    = nn.Dropout(cfg.dropout)

        # causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size))
            .view(1, 1, cfg.block_size, cfg.block_size)
        )

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        att   = (q @ k.transpose(-2, -1)) / scale
        att   = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att   = F.softmax(att, dim=-1)
        att   = self.drop(att)

        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class ProbNetMLP(nn.Module):
    """Feed-forward block: 2 ProbNet layers."""
    def __init__(self, cfg: ProbNetTransformerConfig):
        super().__init__()
        hidden   = 4 * cfg.n_embd
        self.fc1 = _Linear(cfg.n_embd, hidden, cfg)
        self.fc2 = _Linear(hidden, cfg.n_embd, cfg)
        self.act = nn.GELU()
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ProbNetTransformerConfig):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = nn.LayerNorm(cfg.n_embd)
        self.mlp  = ProbNetMLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Full Language Model
# ---------------------------------------------------------------------------

class ProbNetTransformerLM(nn.Module):
    """
    GPT-style language model with ProbNetLayer instead of nn.Linear.
    Only embeddings + LayerNorm + biases are trainable — no weight matrices.
    """

    def __init__(self, cfg: ProbNetTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict({
            "wte":  nn.Embedding(cfg.vocab_size, cfg.n_embd),
            "wpe":  nn.Embedding(cfg.block_size, cfg.n_embd),
            "drop": nn.Dropout(cfg.dropout),
            "h":    nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)]),
            "ln_f": nn.LayerNorm(cfg.n_embd),
        })
        self.lm_head = _Linear(cfg.n_embd, cfg.vocab_size, cfg, bias=False)

        params = sum(p.numel() for p in self.parameters())
        print(f"ProbNetTransformerLM | {params:,} params "
              f"(embeddings + LayerNorm + biases — no weight matrices)")

    def forward(self, idx: torch.Tensor,
                targets: Optional[torch.Tensor] = None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, "Sequence too long"

        pos = torch.arange(T, device=idx.device)
        x   = self.transformer["drop"](
            self.transformer["wte"](idx) + self.transformer["wpe"](pos)
        )
        for block in self.transformer["h"]:
            x = block(x)
        x      = self.transformer["ln_f"](x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss


# ---------------------------------------------------------------------------
# Convert any HuggingFace model → ProbNet
# ---------------------------------------------------------------------------

def convert_to_probnet(model: nn.Module,
                       window: int = 8,
                       bias:   bool = True,
                       verbose: bool = True) -> nn.Module:
    """
    Replace all nn.Linear layers in any model with ProbNetLayer.
    Works on Gemma, GPT-2, LLaMA, Mistral, Phi, Qwen, etc.

    The model's attention structure, embeddings and norms are preserved.
    Only the linear projection layers are replaced.

    Parameters
    ----------
    model   : any nn.Module (HuggingFace CausalLM or similar)
    window  : ProbNet window size per output neuron (default 8)
    bias    : keep bias terms (default True)
    verbose : print each replaced layer

    Returns
    -------
    Same model object, modified in-place.
    """
    replaced = 0

    def _replace(parent, name, child):
        nonlocal replaced
        new_layer = ProbNetLayer(
            child.in_features, child.out_features,
            window=window, bias=bias and (child.bias is not None)
        )
        setattr(parent, name, new_layer)
        replaced += 1
        if verbose:
            print(f"  [{replaced:3d}] {name}: "
                  f"Linear({child.in_features}→{child.out_features}) "
                  f"→ ProbNetLayer(w={window})")

    for module in list(model.modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                _replace(module, child_name, child)

    if verbose:
        after = sum(p.numel() for p in model.parameters())
        print(f"\n  Replaced {replaced} linear layers.")
        print(f"  Remaining trainable params: {after:,} (biases + norms only)")

    return model


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(model: ProbNetTransformerLM,
             tokenizer,
             prompt:       str,
             max_new:      int   = 100,
             temperature:  float = 1.0,
             top_k:        int   = 50,
             device:       str   = "cpu") -> str:
    """
    Generate text from a ProbNetTransformerLM.

    Parameters
    ----------
    model       : ProbNetTransformerLM (or any forward(idx)→(logits,loss))
    tokenizer   : HuggingFace tokenizer with encode/decode
    prompt      : input string
    max_new     : tokens to generate
    temperature : sampling temperature
    top_k       : top-k filtering
    device      : 'cpu' or 'cuda'
    """
    model.eval().to(device)
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    for _ in range(max_new):
        ids_crop = ids[:, -model.cfg.block_size:]
        logits, _ = model(ids_crop)
        logits    = logits[:, -1, :] / max(temperature, 1e-7)

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        probs   = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids     = torch.cat([ids, next_id], dim=1)

    return tokenizer.decode(ids[0].tolist(), skip_special_tokens=True)


# ---------------------------------------------------------------------------
# OEIS / sequence prediction (pure ProbNet, no model needed)
# ---------------------------------------------------------------------------

def predict_sequence(sequence: list, n_next: int = 5) -> list:
    """
    Predict the next n_next elements of any integer sequence using raw ProbNet.

    Examples
    --------
    >>> predict_sequence([1,1,2,3,5,8,13,21], n_next=4)
    [34, 55, 89, 144]

    >>> predict_sequence([2,4,8,16,32], n_next=3)
    [64, 128, 256]
    """
    dat    = torch.tensor(sequence, dtype=torch.float64).unsqueeze(0)  # (1, L)
    result = []
    for _ in range(n_next):
        nxt     = probnet_predict(dat).item()
        nxt_int = round(nxt)
        result.append(nxt_int)
        dat = torch.cat([dat, torch.tensor([[nxt]], dtype=torch.float64)], dim=-1)
    return result
