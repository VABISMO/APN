#!/usr/bin/env python3
"""
ProbNet - convert_hf.py
======================
Converts HuggingFace LLaMA/Gemma models to ProbNet format.

Usage:
    python3 convert_hf.py --model meta-llama/Llama-2-7b-hf --out llama2_probnet.pnet
    python3 convert_hf.py --model google/gemma-2b --out gemma2b_probnet.pnet
    python3 convert_hf.py --model ./local_model_dir --out model.pnet

What this script does:
    1. Loads the HF model (safetensors or pytorch_model.bin)
    2. Extracts all transformer weights
    3. Converts FFN (SwiGLU/GELU) to APN format:
       - The SwiGLU W_gate/W_up become APN W1/W2
       - APN logits initialized so identity+product dominate (matches SwiGLU)
       - W_down becomes W_out
    4. Saves vocabulary in ProbNet format
    5. Writes .pnet binary file

Requirements:
    pip install torch transformers sentencepiece

The converted model can then be fine-tuned to let APN neurons specialize
beyond what SwiGLU achieves.
"""

import os
import sys
import struct
import json
import argparse
import numpy as np

def check_deps():
    """Check and install required packages."""
    missing = []
    try:
        import torch
    except ImportError:
        missing.append("torch")
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
    if missing:
        print(f"Missing packages: {missing}")
        print(f"Install: pip install {' '.join(missing)}")
        sys.exit(1)

def load_hf_model(model_name_or_path):
    """Load a HuggingFace model and return config + state dict."""
    check_deps()
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
    import torch

    print(f"Loading {model_name_or_path}...")
    config = AutoConfig.from_pretrained(model_name_or_path)
    print(f"  Architecture: {config.architectures}")
    print(f"  Hidden size:  {config.hidden_size}")
    print(f"  Num layers:   {config.num_hidden_layers}")
    print(f"  Num heads:    {config.num_attention_heads}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()

    tok = AutoTokenizer.from_pretrained(model_name_or_path)

    return config, model, tok

def save_vocab(hf_tokenizer, out_path):
    """Save vocabulary in ProbNet format (one token per line)."""
    vocab = hf_tokenizer.get_vocab()
    # Sort by id
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    with open(out_path, 'w', encoding='utf-8') as f:
        for token, idx in sorted_vocab:
            f.write(f"{token}\t0.0\n")
    print(f"  Saved vocab ({len(sorted_vocab)} tokens) → {out_path}")
    return len(sorted_vocab)

def to_np(tensor):
    """Convert torch tensor to numpy float32."""
    import torch
    if hasattr(tensor, 'detach'):
        return tensor.detach().float().numpy()
    return np.array(tensor, dtype=np.float32)

def write_f32_array(fp, arr):
    """Write numpy float32 array to binary file."""
    a = np.asarray(arr, dtype=np.float32)
    fp.write(a.tobytes())

def init_apn_logits(hidden, n_funcs=6, identity_bias=1.0, prod_bias=0.5):
    """
    Initialize APN logits to approximate SwiGLU behavior:
    - identity function gets high weight (like linear gate)
    - product function gets medium weight (like SwiGLU gate*up)
    - others start small
    Returns logits array [hidden, n_funcs]
    """
    logits = np.zeros((hidden, n_funcs), dtype=np.float32)
    # f0=identity, f1=sq-tanh, f2=s-sqrt, f3=b-prod, f4=sin, f5=relu
    logits[:, 0] = identity_bias  # identity dominant (like SwiGLU linear component)
    logits[:, 3] = prod_bias       # product moderate (like SwiGLU gate*up)
    logits[:, 5] = 0.3             # relu (like activation gate)
    # Add small noise for diversity
    logits += np.random.randn(*logits.shape).astype(np.float32) * 0.05
    return logits

def convert_llama(config, model, out_path, vocab_size):
    """Convert LLaMA-style model (SwiGLU FFN) to ProbNet."""
    import torch
    state = model.state_dict()

    MAGIC = 0x504E4554  # "PNET"
    VERSION = 2
    ARCH = b"llama-apn\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"

    D  = config.hidden_size
    L  = config.num_hidden_layers
    H  = config.num_attention_heads
    Hkv= getattr(config, 'num_key_value_heads', H)
    hd = D // H
    Fi = D
    Fh = config.intermediate_size
    Fo = D
    max_seq = getattr(config, 'max_position_embeddings', 4096)
    rms_eps = getattr(config, 'rms_norm_eps', 1e-5)
    vocab   = vocab_size

    print(f"\nConverting to ProbNet format:")
    print(f"  D={D}  L={L}  H={H}  Hkv={Hkv}  hd={hd}")
    print(f"  FFN: {Fi}→{Fh}→{Fo} (SwiGLU → APN)")
    print(f"  Vocab: {vocab}")

    # Build TransformerConfig struct (C layout)
    # struct TransformerConfig { int[10 ints] + float[3] + char[32] }
    cfg_bytes = struct.pack('iiiiiiiiii',
        vocab, D, L, H, Hkv, Fh, max_seq, 1,  # 8 ints
        0, 0)  # padding to 10
    cfg_bytes += struct.pack('fff', rms_eps, 3.0, 0.05)  # rms_eps, tau0, tau1
    cfg_bytes += ARCH[:32]
    # Pad to sizeof(TransformerConfig) = 10*4 + 3*4 + 32 = 84 bytes
    while len(cfg_bytes) < 84:
        cfg_bytes += b'\x00'

    with open(out_path, 'wb') as fp:
        # Header
        fp.write(struct.pack('II', MAGIC, VERSION))
        fp.write(cfg_bytes[:84])

        # Token embedding [vocab, D]
        emb_key = 'model.embed_tokens.weight'
        if emb_key in state:
            print(f"  Writing token embeddings...")
            write_f32_array(fp, to_np(state[emb_key]))
        else:
            print(f"  Warning: embedding not found, using random")
            write_f32_array(fp, np.random.randn(vocab, D).astype(np.float32) * 0.02)

        # Layers
        for l in range(L):
            print(f"  Layer {l+1}/{L}...", end='\r', flush=True)
            prefix = f'model.layers.{l}'

            # ── Attention ──
            # W_q: [H*hd, D]
            Wq = to_np(state[f'{prefix}.self_attn.q_proj.weight'])  # [H*hd, D]
            bq = np.zeros(H*hd, dtype=np.float32)
            if f'{prefix}.self_attn.q_proj.bias' in state:
                bq = to_np(state[f'{prefix}.self_attn.q_proj.bias'])
            write_f32_array(fp, Wq); write_f32_array(fp, bq)

            Wk = to_np(state[f'{prefix}.self_attn.k_proj.weight'])  # [Hkv*hd, D]
            bk = np.zeros(Hkv*hd, dtype=np.float32)
            write_f32_array(fp, Wk); write_f32_array(fp, bk)

            Wv = to_np(state[f'{prefix}.self_attn.v_proj.weight'])
            bv = np.zeros(Hkv*hd, dtype=np.float32)
            write_f32_array(fp, Wv); write_f32_array(fp, bv)

            Wo = to_np(state[f'{prefix}.self_attn.o_proj.weight'])  # [D, H*hd]
            bo = np.zeros(D, dtype=np.float32)
            write_f32_array(fp, Wo); write_f32_array(fp, bo)

            # attn_norm [D]
            attn_norm = to_np(state[f'{prefix}.input_layernorm.weight'])
            write_f32_array(fp, attn_norm)

            # ── APN (from SwiGLU) ──
            # SwiGLU: gate_proj→W1, up_proj→W2, down_proj→W_out
            W1 = to_np(state[f'{prefix}.mlp.gate_proj.weight'])  # [Fh, Fi]
            b1 = np.zeros(Fh, dtype=np.float32)
            W2 = to_np(state[f'{prefix}.mlp.up_proj.weight'])    # [Fh, Fi]
            b2 = np.zeros(Fh, dtype=np.float32)
            Wo_ = to_np(state[f'{prefix}.mlp.down_proj.weight']) # [Fo, Fh]
            bo_ = np.zeros(Fo, dtype=np.float32)

            write_f32_array(fp, W1)   # APN W1 [Fh, Fi] = [hidden, in_dim]
            write_f32_array(fp, b1)
            write_f32_array(fp, W2)   # APN W2 [Fh, Fi]
            write_f32_array(fp, b2)

            # APN logits [Fh, NFUNCS=6]: init to match SwiGLU behavior
            logits = init_apn_logits(Fh, n_funcs=6)
            write_f32_array(fp, logits)

            # APN W_out [Fh, Fo] = down_proj transposed: [Fo, Fh].T
            # Our convention: W_out is [hidden, out_dim] = [Fh, Fo]
            # down_proj is [Fo, Fh], so we need [Fh, Fo] = down_proj.T
            write_f32_array(fp, Wo_.T)  # [Fh, Fo]
            write_f32_array(fp, bo_)

            # ffn_norm [D]
            ffn_norm = to_np(state[f'{prefix}.post_attention_layernorm.weight'])
            write_f32_array(fp, ffn_norm)

        print(f"  All {L} layers written.              ")

        # Final norm [D]
        write_f32_array(fp, to_np(state['model.norm.weight']))

        # LM head [vocab, D]
        if 'lm_head.weight' in state:
            write_f32_array(fp, to_np(state['lm_head.weight']))
        else:
            # Tied weights
            write_f32_array(fp, to_np(state['model.embed_tokens.weight']))

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\n  Saved {out_path} ({size_mb:.1f} MB)")

def convert_gemma(config, model, out_path, vocab_size):
    """Convert Gemma-style model to ProbNet."""
    # Gemma uses same architecture as LLaMA, just different key names
    import torch
    state = model.state_dict()

    # Gemma keys are same as LLaMA for most parts
    # except it uses `model.layers.N.self_attn.{q,k,v,o}_proj`
    # which is same as LLaMA — so we can reuse convert_llama
    # Gemma-specific: no bias on projections, slightly different norms

    print("Gemma detected — using LLaMA converter (compatible architecture)")
    convert_llama(config, model, out_path, vocab_size)

def main():
    parser = argparse.ArgumentParser(description='Convert HuggingFace model to ProbNet')
    parser.add_argument('--model',  required=True,  help='HF model name or local path')
    parser.add_argument('--out',    required=True,  help='Output .pnet file')
    parser.add_argument('--vocab',  default=None,   help='Output vocabulary file (default: <out>.vocab)')
    parser.add_argument('--format', default='auto', choices=['auto','llama','gemma'],
                        help='Model format (default: auto-detect)')
    args = parser.parse_args()

    check_deps()
    import torch

    # Set random seed for reproducible APN init
    np.random.seed(42)

    # Load model
    config, model, hf_tok = load_hf_model(args.model)

    # Save vocabulary
    vocab_out = args.vocab or args.out.replace('.pnet','') + '.vocab'
    vocab_size = save_vocab(hf_tok, vocab_out)

    # Detect architecture
    arch = args.format
    if arch == 'auto':
        archs = getattr(config, 'architectures', [])
        if any('llama' in a.lower() for a in archs):
            arch = 'llama'
        elif any('gemma' in a.lower() for a in archs):
            arch = 'gemma'
        elif any('mistral' in a.lower() for a in archs):
            arch = 'llama'  # same architecture
        else:
            print(f"Unknown architecture {archs}, defaulting to LLaMA format")
            arch = 'llama'

    print(f"\nArchitecture: {arch}")

    if arch in ('llama', 'mistral'):
        convert_llama(config, model, args.out, vocab_size)
    elif arch == 'gemma':
        convert_gemma(config, model, args.out, vocab_size)
    else:
        print(f"Unsupported: {arch}")
        sys.exit(1)

    print(f"\nConversion complete!")
    print(f"  Model: {args.out}")
    print(f"  Vocab: {vocab_out}")
    print(f"\nTo use:")
    print(f"  ./probnet generate --model {args.out} --vocab {vocab_out} --prompt 'Hello'")
    print(f"  ./probnet chat     --model {args.out} --vocab {vocab_out}")
    print(f"\nTo fine-tune the APN layers:")
    print(f"  python3 python/train.py --model {args.out} --data data.txt --epochs 3")

if __name__ == '__main__':
    main()
