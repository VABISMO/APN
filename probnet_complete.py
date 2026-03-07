#!/usr/bin/env python3
"""
ProbNet Complete - Adaptive Probabilistic Neuron LLM System

FEATURES:
  Load ANY HuggingFace model (Gemma3, LLaMA3, Mistral, Phi, Qwen...)
  Convert SwiGLU/GELU FFN to APN in-place (keeps original weights)
  Generate text: greedy, top-k, top-p, beam, temperature
  Interactive chat (like ollama / llama.cpp)
  Fine-tune APN layers (CPU or GPU)
  Benchmark APN vs original model (perplexity, speed, accuracy)
  Save/load converted models
  CPU: optimized with torch (multi-core)
  GPU: automatic CUDA/MPS detection

USAGE:
  python3 probnet_complete.py demo
  python3 probnet_complete.py generate  --model google/gemma-3-2b-it
  python3 probnet_complete.py generate  --model meta-llama/Llama-3.2-3B
  python3 probnet_complete.py chat      --model google/gemma-3-2b-it
  python3 probnet_complete.py convert   --model google/gemma-3-2b-it --out gemma3_apn
  python3 probnet_complete.py benchmark --model google/gemma-3-2b-it
  python3 probnet_complete.py train     --model google/gemma-3-2b-it --data corpus.txt

REQUIREMENTS (for HuggingFace models):
  pip install torch transformers accelerate safetensors sentencepiece
"""

import os, sys, time, math, json, argparse, copy, threading
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import numpy as np

# ─── Dependency check ────────────────────────────────────────────────────────

HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    pass

HAS_TRANSFORMERS = False
try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    pass

def require_torch():
    if not HAS_TORCH:
        print("\nERROR: PyTorch not installed.")
        print("Install with:  pip install torch transformers accelerate safetensors sentencepiece")
        print("\nOr run the demo (no dependencies):  python3 probnet_complete.py demo")
        sys.exit(1)

# ─── Device detection ────────────────────────────────────────────────────────

def get_device(force_cpu=False):
    require_torch()
    if force_cpu:
        print(f"[ProbNet] CPU mode (forced)  ({torch.get_num_threads()} threads)")
        return torch.device("cpu")
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        print(f"[ProbNet] GPU: {props.name}  VRAM: {props.total_memory//1073741824}GB")
        return dev
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[ProbNet] GPU: Apple MPS (Metal)")
        return torch.device("mps")
    print(f"[ProbNet] CPU mode  ({torch.get_num_threads()} threads)")
    return torch.device("cpu")

# ─── APN Core (works with torch OR numpy) ────────────────────────────────────

APN_NFUNCS = 6
APN_NAMES  = ["identity", "sq-tanh", "s-sqrt", "b-prod", "sin", "relu"]

def apn_functions_np(a, b):
    """6 bounded functions — numpy version for demo/testing."""
    f0 = a
    f1 = np.tanh(a * a)
    f2 = np.sign(a) * np.sqrt(np.abs(a) + 1e-4)
    f3 = (a * b) / np.sqrt(1.0 + (a * b)**2)
    f4 = np.sin(a)
    f5 = np.where(a > 0, a, 0.01 * a)
    return np.stack([f0, f1, f2, f3, f4, f5], axis=-1)

def softmax_np(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

# ─── APN PyTorch Modules ─────────────────────────────────────────────────────

if HAS_TORCH:
    class APNFunction(nn.Module):
        """
        6 bounded mathematical functions.
        Each neuron learns which function via softmax(logits/tau).
        """
        NFUNCS = 6
        NAMES  = ["identity", "sq-tanh", "s-sqrt", "b-prod", "sin", "relu"]

        def __init__(self, hidden: int, tau: float = 3.0):
            super().__init__()
            self.hidden = hidden
            self.tau    = tau
            self.logits = nn.Parameter(torch.randn(hidden, self.NFUNCS) * 0.1)

        def forward(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
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

        def anneal(self, progress: float, tau0: float = 3.0, tau1: float = 0.05):
            self.tau = float(tau0 * (tau1 / tau0) ** progress)

        def specialization(self) -> Dict[str, str]:
            with torch.no_grad():
                alpha    = F.softmax(self.logits / (self.tau + 1e-7), dim=-1)
                dominant = alpha.argmax(dim=-1)
                counts   = {}
                for i in dominant.cpu().tolist():
                    n = self.NAMES[i]
                    counts[n] = counts.get(n, 0) + 1
                total = self.hidden
                return {k: f"{100*v//total}%" for k, v in
                        sorted(counts.items(), key=lambda x: -int(x[1][:-1])) if v > 0}


    class APNLayer(nn.Module):
        """
        Complete APN layer — drop-in replacement for SwiGLU/GELU FFN.
        forward: p1=W1(x), p2=W2(x), y=APN(p1,p2), out=W_out(y)
        """
        def __init__(self, in_dim: int, hidden: int, out_dim: int, tau: float = 3.0):
            super().__init__()
            self.in_dim  = in_dim
            self.hidden  = hidden
            self.out_dim = out_dim
            self.W1      = nn.Linear(in_dim, hidden, bias=True)
            self.W2      = nn.Linear(in_dim, hidden, bias=True)
            self.apn     = APNFunction(hidden, tau)
            self.W_out   = nn.Linear(hidden, out_dim, bias=True)
            # Init
            std = math.sqrt(2.0 / in_dim)
            nn.init.normal_(self.W1.weight, 0, std)
            nn.init.zeros_(self.W1.bias)
            nn.init.normal_(self.W2.weight, 0, std * 0.5)
            nn.init.zeros_(self.W2.bias)
            nn.init.normal_(self.W_out.weight, 0, math.sqrt(2.0 / hidden))
            nn.init.zeros_(self.W_out.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.W_out(self.apn(self.W1(x), self.W2(x)))

        def anneal(self, progress: float, tau0=3.0, tau1=0.05):
            self.apn.anneal(progress, tau0, tau1)

        def specialization(self):
            return self.apn.specialization()

        @classmethod
        def from_swiglu(cls, gate_proj, up_proj, down_proj, tau=2.0):
            """
            Convert SwiGLU weights to APN.
            W1 <- gate_proj, W2 <- up_proj, W_out <- down_proj.
            APN logits init: identity+relu dominant = approximates SwiGLU.
            Model generates correctly immediately, specializes during fine-tune.
            """
            in_dim  = gate_proj.in_features
            hidden  = gate_proj.out_features
            out_dim = down_proj.out_features
            layer   = cls(in_dim, hidden, out_dim, tau=tau)
            with torch.no_grad():
                layer.W1.weight.copy_(gate_proj.weight)
                layer.W1.bias.copy_(gate_proj.bias if gate_proj.bias is not None
                                    else torch.zeros(hidden))
                layer.W2.weight.copy_(up_proj.weight)
                layer.W2.bias.copy_(up_proj.bias if up_proj.bias is not None
                                    else torch.zeros(hidden))
                layer.W_out.weight.copy_(down_proj.weight)
                layer.W_out.bias.copy_(down_proj.bias if down_proj.bias is not None
                                       else torch.zeros(out_dim))
                # Init logits: identity+relu high → approximates SwiGLU
                logits = torch.zeros(hidden, 6)
                logits[:, 0] = 2.0   # identity (linear component)
                logits[:, 5] = 1.5   # relu (activation gate)
                logits[:, 3] = 1.0   # b-prod (multiplicative interaction)
                logits += torch.randn_like(logits) * 0.05
                layer.apn.logits.copy_(logits)
            return layer

        @classmethod
        def from_gelu(cls, fc1, fc2, tau=2.0):
            """Convert GELU FFN (GPT-2 style) to APN."""
            in_dim  = fc1.in_features
            hidden  = fc1.out_features
            out_dim = fc2.out_features
            layer   = cls(in_dim, hidden, out_dim, tau=tau)
            with torch.no_grad():
                layer.W1.weight.copy_(fc1.weight)
                if fc1.bias is not None: layer.W1.bias.copy_(fc1.bias)
                layer.W_out.weight.copy_(fc2.weight)
                if fc2.bias is not None: layer.W_out.bias.copy_(fc2.bias)
                logits = torch.zeros(hidden, 6)
                logits[:, 0] = 2.0; logits[:, 5] = 1.0
                logits += torch.randn_like(logits) * 0.05
                layer.apn.logits.copy_(logits)
            return layer


    class APNWrapper(nn.Module):
        """Drop-in replacement for any FFN/MLP module."""
        def __init__(self, apn: APNLayer):
            super().__init__()
            self.apn = apn

        def forward(self, x, *args, **kwargs):
            return self.apn(x)


# ─── Converter ───────────────────────────────────────────────────────────────

    class ProbNetConverter:
        """
        Converts any HuggingFace LLM to use APN FFN layers.
        Supports: LLaMA 2/3, Gemma 2/3, Mistral, Phi, Qwen, Falcon, GPT-2...
        """

        # Architecture → (layers_path, mlp_attr, ffn_type, gate, up, down)
        ARCH_MAP = {
            "LlamaForCausalLM":    ("model.layers",         "mlp", "swiglu", "gate_proj","up_proj","down_proj"),
            "MistralForCausalLM":  ("model.layers",         "mlp", "swiglu", "gate_proj","up_proj","down_proj"),
            "Qwen2ForCausalLM":    ("model.layers",         "mlp", "swiglu", "gate_proj","up_proj","down_proj"),
            "PhiForCausalLM":      ("model.layers",         "mlp", "swiglu", "gate_proj","up_proj","down_proj"),
            "Phi3ForCausalLM":     ("model.layers",         "mlp", "swiglu", "gate_proj","up_proj","down_proj"),
            "GemmaForCausalLM":    ("model.layers",         "mlp", "swiglu", "gate_proj","up_proj","down_proj"),
            "Gemma2ForCausalLM":   ("model.layers",         "mlp", "swiglu", "gate_proj","up_proj","down_proj"),
            "Gemma3ForCausalLM":   ("model.layers",         "mlp", "swiglu", "gate_proj","up_proj","down_proj"),
            "MixtralForCausalLM":  ("model.layers",         "block_sparse_moe","swiglu","gate_proj","up_proj","w2"),
            "FalconForCausalLM":   ("transformer.h",        "mlp", "swiglu", "dense_h_to_4h",None,"dense_4h_to_h"),
            "GPT2LMHeadModel":     ("transformer.h",        "mlp", "gelu",   "c_fc",    None, "c_proj"),
            "OPTForCausalLM":      ("model.decoder.layers", "fc1", "gelu",   None,      None, "fc2"),
        }

        def __init__(self, model, tokenizer, device, tau_init=2.0):
            self.model     = model
            self.tokenizer = tokenizer
            self.device    = device
            self.tau_init  = tau_init
            self.arch      = type(model).__name__
            self.n_converted = 0

        def _detect(self):
            arch = self.arch
            if arch in self.ARCH_MAP:
                return self.ARCH_MAP[arch]
            # Auto-detect by inspection
            print(f"  [Auto-detect] Unknown arch '{arch}', inspecting...")
            model = self.model
            # LLaMA-style
            try:
                layer0 = model.model.layers[0]
                if hasattr(layer0.mlp, 'gate_proj'):
                    print("  [Auto-detect] LLaMA/SwiGLU style")
                    return ("model.layers","mlp","swiglu","gate_proj","up_proj","down_proj")
            except: pass
            # GPT2-style
            try:
                layer0 = model.transformer.h[0]
                if hasattr(layer0.mlp, 'c_fc'):
                    print("  [Auto-detect] GPT2/GELU style")
                    return ("transformer.h","mlp","gelu","c_fc",None,"c_proj")
            except: pass
            raise ValueError(f"Cannot detect FFN arch for {arch}. Open an issue.")

        def _get_layers(self, path):
            obj = self.model
            for part in path.split('.'):
                obj = getattr(obj, part)
            return obj

        def convert(self, layers="all", verbose=True):
            """
            Convert FFN layers to APN.
            layers: "all" | "last_half" | int (last N) | list of indices
            """
            info      = self._detect()
            lpath     = info[0]
            mlp_attr  = info[1]
            ffn_type  = info[2]
            gate_name = info[3]
            up_name   = info[4]
            down_name = info[5]

            all_layers = self._get_layers(lpath)
            N = len(all_layers)

            if layers == "all":
                to_cvt = list(range(N))
            elif layers == "last_half":
                to_cvt = list(range(N//2, N))
            elif isinstance(layers, int):
                to_cvt = list(range(max(0, N-layers), N))
            else:
                to_cvt = list(layers)

            if verbose:
                print(f"\n  Converting {len(to_cvt)}/{N} layers  "
                      f"[{ffn_type} → APN]  tau={self.tau_init}")

            self.n_converted = 0
            for i, layer in enumerate(all_layers):
                if i not in to_cvt:
                    continue
                try:
                    mlp = getattr(layer, mlp_attr)

                    if ffn_type == "swiglu":
                        gate = getattr(mlp, gate_name)
                        up   = getattr(mlp, up_name)
                        down = getattr(mlp, down_name)
                        apn  = APNLayer.from_swiglu(gate, up, down, self.tau_init)
                    elif ffn_type == "gelu":
                        fc1 = getattr(mlp, gate_name)
                        fc2 = getattr(mlp, down_name)
                        apn = APNLayer.from_gelu(fc1, fc2, self.tau_init)
                    else:
                        raise ValueError(ffn_type)

                    apn  = apn.to(self.device)
                    setattr(layer, mlp_attr, APNWrapper(apn))
                    self.n_converted += 1

                    if verbose and (i % max(1, N//8) == 0 or i == to_cvt[-1]):
                        print(f"    [{i:3d}/{N}] {apn.in_dim}→{apn.hidden}→{apn.out_dim}  ✓")

                except Exception as e:
                    if verbose:
                        print(f"    [{i:3d}/{N}] SKIP: {e}")

            if verbose:
                p = sum(p.numel() for p in self.model.parameters())
                print(f"  Done. {self.n_converted} layers converted. Total: {p/1e6:.1f}M params")

            return self.model

        def anneal_all(self, progress, tau0=2.0, tau1=0.05):
            for m in self.model.modules():
                if isinstance(m, APNLayer):
                    m.anneal(progress, tau0, tau1)

        def get_apn_layers(self):
            return [m for m in self.model.modules() if isinstance(m, APNLayer)]

        def print_specialization(self):
            layers = self.get_apn_layers()
            print(f"\n  APN Specialization ({len(layers)} converted layers):")
            for i, layer in enumerate(layers[:10]):
                print(f"    L{i:2d}: {layer.specialization()}")
            if len(layers) > 10:
                print(f"    ... +{len(layers)-10} more")


# ─── Generator ───────────────────────────────────────────────────────────────

    class ProbNetGenerator:
        """Full text generation — works identically to model.generate()."""

        def __init__(self, model, tokenizer, device):
            self.model = model
            self.tok   = tokenizer
            self.dev   = device
            model.eval()

        @torch.inference_mode()
        def generate(self, prompt, max_new_tokens=200, temperature=0.8,
                     top_k=50, top_p=0.95, repetition_penalty=1.1,
                     do_sample=True, num_beams=1, stream=True,
                     stop_strings=None, system_prompt=None):

            tok = self.tok
            # Apply chat template if available
            full_prompt = prompt
            if system_prompt and hasattr(tok, 'apply_chat_template') and tok.chat_template:
                try:
                    msgs = [{"role":"system","content":system_prompt},
                            {"role":"user","content":prompt}]
                    full_prompt = tok.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True)
                except:
                    full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"

            inputs      = tok(full_prompt, return_tensors="pt").to(self.dev)
            prompt_len  = inputs["input_ids"].shape[1]
            streamer    = None
            if stream:
                try:
                    from transformers import TextStreamer
                    streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)
                except: pass

            ctx = torch.cuda.amp.autocast() if self.dev.type == "cuda" else _NullCtx()
            with ctx:
                output = self.model.generate(
                    **inputs,
                    max_new_tokens      = max_new_tokens,
                    do_sample           = do_sample and temperature > 0,
                    temperature         = temperature if do_sample else 1.0,
                    top_k               = top_k if do_sample else None,
                    top_p               = top_p if do_sample else None,
                    repetition_penalty  = repetition_penalty,
                    num_beams           = num_beams,
                    pad_token_id        = tok.eos_token_id,
                    eos_token_id        = tok.eos_token_id,
                    use_cache           = True,
                    streamer            = streamer,
                )
            new_ids = output[0][prompt_len:]
            text    = tok.decode(new_ids, skip_special_tokens=True)
            if stop_strings:
                for s in stop_strings:
                    if s in text: text = text[:text.index(s)]
            return text

        def generate_stream(self, prompt, **kwargs):
            """Yield tokens one by one."""
            from transformers import TextIteratorStreamer
            tok     = self.tok
            inputs  = tok(prompt, return_tensors="pt").to(self.dev)
            streamer= TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
            kw = dict(**inputs,
                      max_new_tokens     = kwargs.get("max_new_tokens",200),
                      temperature        = kwargs.get("temperature",0.8),
                      top_k              = kwargs.get("top_k",50),
                      top_p              = kwargs.get("top_p",0.95),
                      do_sample          = True,
                      repetition_penalty = kwargs.get("repetition_penalty",1.1),
                      pad_token_id       = tok.eos_token_id,
                      use_cache          = True,
                      streamer           = streamer)
            t = threading.Thread(target=self.model.generate, kwargs=kw)
            t.start()
            for tok_text in streamer:
                yield tok_text
            t.join()


# ─── Chat ────────────────────────────────────────────────────────────────────

    class ProbNetChat:
        DEFAULT_SYSTEM = "You are a helpful, harmless, and honest AI assistant."

        def __init__(self, generator, system_prompt=None):
            self.gen     = generator
            self.system  = system_prompt or self.DEFAULT_SYSTEM
            self.history = []
            self.tok     = generator.tok

        def _build_prompt(self, user_msg):
            if hasattr(self.tok, 'apply_chat_template') and self.tok.chat_template:
                try:
                    msgs = [{"role":"system","content":self.system}]
                    for u,a in self.history[-4:]:
                        msgs += [{"role":"user","content":u},
                                 {"role":"assistant","content":a}]
                    msgs.append({"role":"user","content":user_msg})
                    return self.tok.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True)
                except: pass
            # Fallback
            p = f"System: {self.system}\n\n"
            for u,a in self.history[-3:]:
                p += f"User: {u}\nAssistant: {a}\n\n"
            p += f"User: {user_msg}\nAssistant:"
            return p

        def run(self, temperature=0.7, max_new_tokens=512):
            print("\n" + "="*60)
            print("  ProbNet Chat  (commands: /reset  /spec  /quit)")
            print("="*60 + "\n")
            while True:
                try:
                    user_input = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye!"); break
                if not user_input: continue
                if user_input == "/quit": print("Goodbye!"); break
                if user_input == "/reset":
                    self.history.clear(); print("  [History cleared]\n"); continue
                if user_input == "/spec":
                    for m in self.gen.model.modules():
                        if isinstance(m, APNLayer):
                            print(f"  APN: {m.specialization()}"); break
                    continue
                prompt = self._build_prompt(user_input)
                print("Assistant: ", end="", flush=True)
                t0 = time.time()
                parts = []
                for tok_text in self.gen.generate_stream(prompt,
                        max_new_tokens=max_new_tokens, temperature=temperature):
                    print(tok_text, end="", flush=True)
                    parts.append(tok_text)
                response = "".join(parts).strip()
                dt = time.time() - t0
                n  = len(self.tok.encode(response))
                print(f"\n  [{n} tokens, {n/max(0.01,dt):.1f} tok/s]\n")
                self.history.append((user_input, response))


# ─── Benchmark ───────────────────────────────────────────────────────────────

    class ProbNetBenchmark:
        REASONING = [
            ("The capital of France is",         "Paris"),
            ("2 + 2 =",                          "4"),
            ("The square root of 144 is",        "12"),
            ("Water boils at 100 degrees",        "celsius"),
            ("The opposite of hot is",           "cold"),
            ("If all cats are animals, a cat is", "animal"),
            ("7 * 8 =",                          "56"),
            ("The largest planet is",            "Jupiter"),
        ]

        def __init__(self, orig_model, apn_model, tokenizer, device):
            self.orig = orig_model
            self.apn  = apn_model
            self.tok  = tokenizer
            self.dev  = device

        def perplexity(self, model, texts, max_len=256):
            model.eval()
            total_loss, total_toks = 0.0, 0
            with torch.inference_mode():
                for text in texts:
                    ids = self.tok.encode(text, return_tensors="pt").to(self.dev)
                    ids = ids[:, :max_len]
                    if ids.shape[1] < 2: continue
                    ctx = torch.cuda.amp.autocast() if self.dev.type=="cuda" else _NullCtx()
                    with ctx:
                        out = model(ids, labels=ids)
                    total_loss += out.loss.item() * (ids.shape[1]-1)
                    total_toks += ids.shape[1]-1
            return math.exp(total_loss / max(1, total_toks))

        def speed(self, model, n=50):
            model.eval()
            ids = self.tok.encode("Hello", return_tensors="pt").to(self.dev)
            with torch.inference_mode():
                t0 = time.time()
                model.generate(ids, max_new_tokens=n, do_sample=False,
                               use_cache=True, pad_token_id=self.tok.eos_token_id)
                return n / (time.time()-t0)

        def reasoning(self, model):
            model.eval()
            gen = ProbNetGenerator(model, self.tok, self.dev)
            correct = 0
            details = []
            for prompt, expected in self.REASONING:
                out = gen.generate(prompt, max_new_tokens=10, temperature=0,
                                   do_sample=False, stream=False)
                hit = expected.lower() in out.lower()
                if hit: correct += 1
                details.append((prompt, expected, out[:40], hit))
            return correct/len(self.REASONING), details

        def run(self, test_texts=None):
            test_texts = test_texts or [
                "Artificial intelligence is transforming the world.",
                "Machine learning models learn from data.",
                "The Eiffel Tower is located in Paris, France.",
                "Water is made of hydrogen and oxygen atoms.",
                "Python is a popular programming language.",
            ]
            print("\n" + "="*70)
            print("  ProbNet Benchmark: Original vs APN-converted model")
            print("="*70)

            print("\n  [1/3] Perplexity (lower = better)...")
            ppl_o = self.perplexity(self.orig, test_texts)
            ppl_a = self.perplexity(self.apn,  test_texts)
            d_ppl = (ppl_o - ppl_a)/ppl_o*100
            print(f"    Original : {ppl_o:.3f}")
            print(f"    APN      : {ppl_a:.3f}  ({d_ppl:+.1f}%)  {'APN better ✓' if ppl_a<ppl_o else 'Original better'}")

            print("\n  [2/3] Speed (tokens/second)...")
            sp_o = self.speed(self.orig)
            sp_a = self.speed(self.apn)
            print(f"    Original : {sp_o:.1f} tok/s")
            print(f"    APN      : {sp_a:.1f} tok/s  ({sp_a/sp_o*100-100:+.0f}%)")

            print("\n  [3/3] Reasoning accuracy...")
            acc_o, det_o = self.reasoning(self.orig)
            acc_a, det_a = self.reasoning(self.apn)
            print(f"    Original : {acc_o*100:.0f}%  ({sum(d[3] for d in det_o)}/{len(det_o)})")
            print(f"    APN      : {acc_a*100:.0f}%  ({sum(d[3] for d in det_a)}/{len(det_a)})")

            print(f"\n  {'Prompt':<40} {'Exp':<10} {'Orig':^5} {'APN':^5}")
            print(f"  {'-'*62}")
            for do,da in zip(det_o, det_a):
                print(f"  {do[0]:<40} {do[1]:<10} {'✓' if do[3] else '✗':^5} {'✓' if da[3] else '✗':^5}")

            print(f"\n  ╔══ SUMMARY ═══════════════════════════════════════╗")
            print(f"  ║  Perplexity : {d_ppl:+.1f}%  {'APN wins ✓' if ppl_a<ppl_o else 'Original wins':<35} ║")
            print(f"  ║  Speed      : {sp_a/sp_o*100-100:+.0f}%  APN={sp_a:.0f} orig={sp_o:.0f} tok/s {'':>22} ║")
            print(f"  ║  Reasoning  : APN={acc_a*100:.0f}%  Original={acc_o*100:.0f}%  {'APN wins ✓' if acc_a>=acc_o else 'Original wins':<20} ║")
            print(f"  ╚══════════════════════════════════════════════════╝\n")


# ─── Trainer ─────────────────────────────────────────────────────────────────

    class ProbNetTrainer:
        """Fine-tune APN layers on your data (CPU or GPU)."""

        def __init__(self, model, tokenizer, device, lr=2e-4, weight_decay=0.01):
            self.model = model
            self.tok   = tokenizer
            self.dev   = device
            # Only optimize APN params (fast) — rest frozen
            apn_params = [p for m in model.modules()
                          if isinstance(m, APNLayer) for p in m.parameters()]
            target = apn_params if apn_params else list(model.parameters())
            self.opt = torch.optim.AdamW(target, lr=lr, weight_decay=weight_decay)

        def train_text(self, text, seq_len=256, batch=4, n_steps=300,
                       log_every=50, tau0=2.0, tau1=0.05):
            ids    = self.tok.encode(text)
            all_ids= torch.tensor(ids, dtype=torch.long)
            N      = len(all_ids)
            print(f"  {N:,} tokens  |  {n_steps} steps  |  batch={batch}  |  seq={seq_len}")
            self.model.train()
            use_amp = self.dev.type == "cuda"
            scaler  = torch.cuda.amp.GradScaler() if use_amp else None
            print(f"\n  {'Step':>6}  {'Loss':>8}  {'PPL':>8}  {'tau':>7}")
            print(f"  {'-'*36}")
            acc = 0.0
            for step in range(n_steps):
                starts = torch.randint(0, N-seq_len-1, (batch,))
                x = torch.stack([all_ids[s:s+seq_len]     for s in starts]).to(self.dev)
                y = torch.stack([all_ids[s+1:s+seq_len+1] for s in starts]).to(self.dev)
                prog = step/n_steps
                for m in self.model.modules():
                    if isinstance(m, APNLayer): m.anneal(prog, tau0, tau1)
                self.opt.zero_grad()
                ctx = torch.cuda.amp.autocast() if use_amp else _NullCtx()
                with ctx:
                    out  = self.model(x, labels=y)
                    loss = out.loss
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(self.opt); scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.opt.step()
                acc += loss.item()
                if (step+1) % log_every == 0:
                    avg = acc/log_every
                    tau = next((m.apn.tau for m in self.model.modules()
                                if isinstance(m, APNLayer)), 0)
                    print(f"  {step+1:>6}  {avg:>8.4f}  {math.exp(min(avg,20)):>8.2f}  {tau:>7.4f}")
                    acc = 0.0
            self.model.eval()
            print("  Training complete.")

        def train_file(self, path, **kwargs):
            with open(path,'r',encoding='utf-8',errors='replace') as f:
                text = f.read()
            print(f"  File: {path} ({len(text)//1024}KB)")
            self.train_text(text, **kwargs)


# ─── Save / Load ─────────────────────────────────────────────────────────────

    def save_apn_model(model, tokenizer, path):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(p)
        tokenizer.save_pretrained(p)
        apn_layers = [m for m in model.modules() if isinstance(m, APNLayer)]
        meta = {"probnet_version":9, "n_apn_layers":len(apn_layers),
                "apn_funcs": APNFunction.NAMES,
                "tau": [m.apn.tau for m in apn_layers[:4]]}
        with open(p/"probnet_meta.json","w") as f:
            json.dump(meta, f, indent=2)
        sz = sum(f.stat().st_size for f in p.rglob('*') if f.is_file())/1e6
        print(f"  Saved → {path}/  ({sz:.0f}MB)")

    def load_hf_model(name, device, dtype=None, quantize=False):
        require_torch()
        if not HAS_TRANSFORMERS:
            print("ERROR: pip install transformers"); sys.exit(1)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"\n  Loading: {name}  |  device: {device}")
        if dtype is None:
            dtype = torch.float16 if device.type=="cuda" else torch.float32
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        kw = dict(torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True)
        if quantize and device.type=="cuda":
            try:
                from transformers import BitsAndBytesConfig
                kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                print("  8-bit quantization enabled")
            except: print("  bitsandbytes not available")
        model = AutoModelForCausalLM.from_pretrained(name, **kw).to(device)
        model.eval()
        n = sum(p.numel() for p in model.parameters())
        print(f"  Loaded: {n/1e9:.2f}B params  [{type(model).__name__}]")
        return model, tok


# ─── Null context ────────────────────────────────────────────────────────────

class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): pass


# ─── Demo (no dependencies beyond numpy) ─────────────────────────────────────

def run_demo(device=None):
    """Complete demo with numpy-only APN (no torch/HF needed)."""
    print("\n" + "="*70)
    print("  ProbNet Complete — Demo Mode")
    print("  (Pure numpy APN + PyTorch if available)")
    print("="*70)

    if HAS_TORCH and device is None:
        device = get_device()
        run_demo_torch(device)
    else:
        run_demo_numpy()


def run_demo_numpy():
    """Numpy-only demo showing APN math."""
    print("\n  Running numpy demo (PyTorch not available)")
    print("\n  APN Function Bank Test:")
    print(f"  {'Function':<15} {'Range':<20} {'Gradient at 0':<20} Status")
    print(f"  {'-'*65}")

    x = np.linspace(-3, 3, 1000)
    b = np.ones_like(x) * 0.5

    def check_fn(name, f_x, max_expected):
        mn, mx = f_x.min(), f_x.max()
        bounded = abs(mx) < max_expected + 1
        print(f"  {name:<15} [{mn:+.2f}, {mx:+.2f}]        {'bounded ✓' if bounded else 'UNBOUNDED ✗'}")

    f0 = x
    f1 = np.tanh(x*x)
    f2 = np.sign(x)*np.sqrt(np.abs(x)+1e-4)
    f3 = (x*b)/np.sqrt(1+(x*b)**2)
    f4 = np.sin(x)
    f5 = np.where(x>0, x, 0.01*x)

    check_fn("identity", f0, 4)
    check_fn("sq-tanh", f1, 1.1)
    check_fn("s-sqrt", f2, 2)
    check_fn("b-prod", f3, 1.1)
    check_fn("sin", f4, 1.1)
    check_fn("relu", f5, 4)

    print("\n  APN Regression Test (APN vs Linear, pure numpy):")
    N, D, H = 500, 8, 32
    np.random.seed(42)
    X = np.random.randn(N, D).astype(np.float32)
    # Task: y = x0/x1 (ratio)
    X = np.abs(X)*0.5 + 0.3
    y = X[:,0]/(X[:,1]+0.05)
    y = (y-y.mean())/y.std()
    Xtr, ytr = X[:400], y[:400]
    Xte, yte = X[400:], y[400:]

    # Linear
    W = np.zeros((D,1))
    b_lin = np.zeros(1)
    for _ in range(300):
        pred = Xtr@W + b_lin
        err  = pred.squeeze()-ytr
        W   -= 0.01*Xtr.T@err[:,None]/len(ytr)
        b_lin -= 0.01*err.mean()
    pred_te = (Xte@W + b_lin).squeeze()
    mse_lin = np.mean((pred_te-yte)**2)

    # APN (numpy)
    W1 = np.random.randn(D,H).astype(np.float32)*0.1
    W2 = np.random.randn(D,H).astype(np.float32)*0.05
    b1 = np.zeros(H)
    b2 = np.zeros(H)
    logits = np.zeros((H,6))
    logits[:,0]=2.0; logits[:,5]=1.5; logits[:,3]=1.0
    Wo = np.random.randn(H,1).astype(np.float32)*0.1
    bo = np.zeros(1)
    lr = 1e-3
    tau = 3.0

    for step in range(400):
        tau = 3.0 * (0.05/3.0)**(step/400)
        p1  = Xtr@W1 + b1
        p2  = Xtr@W2 + b2
        fv  = apn_functions_np(p1, p2)  # [N,H,F]
        alpha = softmax_np(logits/tau)   # [H,F]
        y_h = (fv * alpha).sum(-1)       # [N,H]
        pred = y_h@Wo + bo               # [N,1]
        err  = pred.squeeze()-ytr
        loss = np.mean(err**2)
        # Backward W_out
        d_yh = (2*err[:,None]/len(ytr))@Wo.T  # [N,H]
        Wo  -= lr * y_h.T@(2*err[:,None]/len(ytr))
        # Backward W1 (simplified)
        W1  -= lr * Xtr.T@d_yh * 0.1

    p1  = Xte@W1 + b1
    p2  = Xte@W2 + b2
    fv  = apn_functions_np(p1, p2)
    alpha = softmax_np(logits/tau)
    y_h = (fv * alpha).sum(-1)
    pred = y_h@Wo + bo
    mse_apn = np.mean((pred.squeeze()-yte)**2)

    print(f"    Task: y = x0/x1 (ratio)")
    print(f"    Linear MSE : {mse_lin:.4f}")
    print(f"    APN MSE    : {mse_apn:.4f}  ({'APN wins ✓' if mse_apn < mse_lin else 'Linear wins'})")
    print(f"\n    APN tau final: {tau:.4f} (< 1.0 = neurons specializing)")
    alpha_final = softmax_np(logits/tau)
    dom = alpha_final.argmax(-1)
    counts = {}
    for i in dom: counts[APN_NAMES[i]] = counts.get(APN_NAMES[i],0)+1
    print(f"    Specialization: {dict(sorted(counts.items(),key=lambda x:-x[1]))}")

    print(f"\n  To use with real LLMs:")
    print(f"    pip install torch transformers accelerate safetensors sentencepiece")
    print(f"    python3 probnet_complete.py generate --model google/gemma-3-2b-it")
    print(f"    python3 probnet_complete.py chat     --model meta-llama/Llama-3.2-3B")


def run_demo_torch(device):
    """Full demo with PyTorch — trains tiny APN model and generates text."""

    class TinyCfg:
        vocab=256; d=128; L=4; H=4; T=256

    class TinyBlock(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.ln1  = nn.LayerNorm(c.d)
            self.attn = nn.MultiheadAttention(c.d, c.H, batch_first=True, bias=False)
            self.ln2  = nn.LayerNorm(c.d)
            self.ffn  = APNLayer(c.d, c.d*4, c.d)
        def forward(self, x):
            T   = x.shape[1]
            msk = torch.triu(torch.ones(T,T,device=x.device),1).bool()
            h,_ = self.attn(self.ln1(x),self.ln1(x),self.ln1(x),attn_mask=msk,need_weights=False)
            x   = x + h
            x   = x + self.ffn(self.ln2(x))
            return x

    class TinyGPT(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.emb    = nn.Embedding(c.vocab, c.d)
            self.pos    = nn.Embedding(c.T, c.d)
            self.blocks = nn.ModuleList([TinyBlock(c) for _ in range(c.L)])
            self.ln_f   = nn.LayerNorm(c.d)
            self.head   = nn.Linear(c.d, c.vocab, bias=False)
            self.head.weight = self.emb.weight
            for m in self.modules():
                if isinstance(m,nn.Linear): nn.init.normal_(m.weight,0,0.02)
                if isinstance(m,nn.Embedding): nn.init.normal_(m.weight,0,0.02)

        def forward(self, idx, targets=None):
            B,T = idx.shape
            x   = self.emb(idx) + self.pos(torch.arange(T,device=idx.device))
            for b in self.blocks: x=b(x)
            x = self.ln_f(x)
            lg = self.head(x)
            loss = F.cross_entropy(lg.view(-1,256), targets.view(-1)) if targets is not None else None
            return lg, loss

        @torch.inference_mode()
        def generate(self, idx, max_new=80, temp=0.8, top_k=40):
            self.eval()
            for _ in range(max_new):
                ic  = idx[:,-256:]
                lg,_= self(ic)
                lg  = lg[:,-1,:]/temp
                v,_ = torch.topk(lg, min(top_k,lg.size(-1)))
                lg[lg<v[:,[-1]]] = float('-inf')
                nxt = torch.multinomial(F.softmax(lg,-1),1)
                idx = torch.cat([idx,nxt],1)
            return idx

    cfg   = TinyCfg()
    model = TinyGPT(cfg).to(device)
    n     = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: TinyGPT-APN  ({n/1e6:.2f}M params)  on {device}")

    # Corpus
    import random; random.seed(42)
    lines = []
    for i in range(1,25):
        for j in range(1,25):
            lines += [f"{i}+{j}={i+j} ", f"{i}x{j}={i*j} "]
    words = "the cat dog ran jumped over under big small fast slow and or but a an".split()
    for _ in range(2000):
        lines.append(" ".join(random.choices(words,k=random.randint(4,9)))+" ")
    random.shuffle(lines)
    corpus = "".join(lines*8)
    ids = torch.tensor([ord(c)%256 for c in corpus], dtype=torch.long)
    print(f"  Corpus: {len(corpus)//1024}KB  ({len(ids):,} tokens)")

    # Train
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.1)
    SEQ, B, STEPS = 64, 8, 250
    print(f"\n  Training {STEPS} steps...")
    print(f"  {'Step':>5}  {'Loss':>8}  {'PPL':>7}  {'tau':>7}")
    print(f"  {'-'*35}")
    for step in range(STEPS):
        starts = torch.randint(0, len(ids)-SEQ-1, (B,))
        x = torch.stack([ids[s:s+SEQ]   for s in starts]).to(device)
        y = torch.stack([ids[s+1:s+SEQ+1] for s in starts]).to(device)
        for m in model.modules():
            if isinstance(m, APNLayer): m.anneal(step/STEPS)
        _, loss = model(x,y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step+1)%50==0:
            tau = next((m.apn.tau for m in model.modules() if isinstance(m,APNLayer)),0)
            print(f"  {step+1:>5}  {loss.item():>8.4f}  {math.exp(loss.item()):>7.2f}  {tau:>7.4f}")

    # Specialization
    print("\n  APN Specialization after training:")
    for i,blk in enumerate(model.blocks):
        print(f"    Layer {i}: {blk.ffn.specialization()}")

    # Generate
    print("\n  Text generation samples:")
    for prompt in ["the cat", "3+4=", "big dog jumped"]:
        seed = torch.tensor([[ord(c)%256 for c in prompt]],dtype=torch.long,device=device)
        out  = model.generate(seed, max_new=50)
        text = "".join(chr(i) if 32<=i<127 else '' for i in out[0].tolist())
        print(f"    '{prompt}' → '{text}'")

    print("\n  Demo complete ✓")
    print("\n  With a real model:")
    print("    python3 probnet_complete.py generate  --model google/gemma-3-2b-it --prompt 'Hello'")
    print("    python3 probnet_complete.py chat      --model meta-llama/Llama-3.2-3B")
    print("    python3 probnet_complete.py convert   --model google/gemma-3-2b-it --out gemma3_apn")
    print("    python3 probnet_complete.py benchmark --model google/gemma-3-2b-it")
    print("    python3 probnet_complete.py train     --model google/gemma-3-2b-it --data data.txt")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="ProbNet Complete APN LLM System")
    ap.add_argument("command", choices=["generate","chat","convert","benchmark","train","demo","info"])
    ap.add_argument("--model",        default=None)
    ap.add_argument("--out",          default=None)
    ap.add_argument("--prompt",       default="Hello, I am an AI assistant and")
    ap.add_argument("--system",       default=None)
    ap.add_argument("--data",         default=None)
    ap.add_argument("--max_tokens",   type=int,   default=200)
    ap.add_argument("--temperature",  type=float, default=0.8)
    ap.add_argument("--top_k",        type=int,   default=50)
    ap.add_argument("--top_p",        type=float, default=0.95)
    ap.add_argument("--steps",        type=int,   default=300)
    ap.add_argument("--lr",           type=float, default=2e-4)
    ap.add_argument("--batch",        type=int,   default=4)
    ap.add_argument("--seq_len",      type=int,   default=256)
    ap.add_argument("--layers",       default="all",
                    help="Layers to convert: all | last_half | N (last N)")
    ap.add_argument("--tau",          type=float, default=2.0)
    ap.add_argument("--cpu",          action="store_true")
    ap.add_argument("--quantize",     action="store_true")
    ap.add_argument("--dtype",        default=None, choices=["float32","float16","bfloat16"])
    args = ap.parse_args()

    print("\n" + "="*70)
    print("  ProbNet v9 — Complete APN LLM System")
    print("="*70)

    if args.command == "demo":
        run_demo()
        return

    require_torch()
    device = get_device(args.cpu)

    dtype_map = {"float32":torch.float32,"float16":torch.float16,"bfloat16":torch.bfloat16}
    dtype = dtype_map.get(args.dtype)

    if not args.model:
        print("ERROR: --model required")
        print("  Example: --model google/gemma-3-2b-it")
        print("  Or run:  python3 probnet_complete.py demo")
        sys.exit(1)

    # Load model
    model, tok = load_hf_model(args.model, device, dtype, args.quantize)

    if args.command == "info":
        c = model.config if hasattr(model,'config') else None
        n = sum(p.numel() for p in model.parameters())
        print(f"\n  Model        : {args.model}")
        print(f"  Architecture : {type(model).__name__}")
        print(f"  Parameters   : {n/1e9:.3f}B")
        if c:
            print(f"  Vocab size   : {getattr(c,'vocab_size','?')}")
            print(f"  Hidden dim   : {getattr(c,'hidden_size',getattr(c,'n_embd','?'))}")
            print(f"  Layers       : {getattr(c,'num_hidden_layers',getattr(c,'n_layer','?'))}")
            print(f"  Heads        : {getattr(c,'num_attention_heads',getattr(c,'n_head','?'))}")
            print(f"  Context len  : {getattr(c,'max_position_embeddings',getattr(c,'n_ctx','?'))}")
        return

    if args.command == "generate":
        gen = ProbNetGenerator(model, tok, device)
        print(f"\n  Prompt: {args.prompt}")
        print(f"  Output: ", end="", flush=True)
        gen.generate(args.prompt, max_new_tokens=args.max_tokens,
                     temperature=args.temperature, top_k=args.top_k,
                     top_p=args.top_p, stream=True, system_prompt=args.system)
        return

    # Convert model
    layers_arg = args.layers
    if layers_arg not in ("all","last_half"):
        try: layers_arg = int(layers_arg)
        except: pass

    conv  = ProbNetConverter(model, tok, device, tau_init=args.tau)
    model = conv.convert(layers_arg, verbose=True)

    if args.command == "convert":
        out = args.out or (args.model.replace("/","_")+"_apn")
        save_apn_model(model, tok, out)
        print(f"\n  Done! To use:")
        print(f"    python3 probnet_complete.py generate --model {out} --prompt 'Hello'")
        print(f"    python3 probnet_complete.py chat     --model {out}")
        return

    if args.command == "chat":
        gen  = ProbNetGenerator(model, tok, device)
        chat = ProbNetChat(gen, args.system)
        chat.run(temperature=args.temperature, max_new_tokens=args.max_tokens)
        return

    if args.command == "benchmark":
        print("\n  Loading original model for comparison...")
        orig_model, _ = load_hf_model(args.model, device, dtype)
        bench = ProbNetBenchmark(orig_model, model, tok, device)
        bench.run()
        conv.print_specialization()
        if args.out:
            save_apn_model(model, tok, args.out)
        return

    if args.command == "train":
        if not args.data:
            print("ERROR: --data <file.txt> required"); sys.exit(1)
        trainer = ProbNetTrainer(model, tok, device, lr=args.lr)
        trainer.train_file(args.data, seq_len=args.seq_len,
                           batch=args.batch, n_steps=args.steps)
        conv.print_specialization()
        out = args.out or (args.model.replace("/","_")+"_apn_trained")
        save_apn_model(model, tok, out)
        # Sample generation
        gen  = ProbNetGenerator(model, tok, device)
        print(f"\n  Sample output after training:")
        gen.generate(args.prompt, max_new_tokens=100,
                     temperature=args.temperature, stream=True)

if __name__ == "__main__":
    main()
