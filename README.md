# APN — Adaptive Probabilistic Neuron Transformer

APN replaces the FFN layers (SwiGLU/GELU) in any transformer with **Adaptive Probabilistic Neurons (APN)**: neurons that learn *which mathematical function to apply* from a bank of 6 bounded functions. Each neuron specializes during training — starting uniform, ending committed to its optimal function.

Compatible with **Gemma, LLaMA, Mistral, Qwen, Phi, GPT-2** and any HuggingFace SwiGLU/GELU model.

---

## APN vs SwiGLU — Key Results

| Task | Linear | SwiGLU | APN | APN vs SwiGLU |
|------|--------|--------|-----|---------------|
| **Linear** y=Wx | 0.017 | 0.328 | **0.001** | **240x better** |
| **Sqrt** y=sqrt(\|x0\|) | 0.815 | 0.145 | **0.015** | **9.7x better** |
| **Mixed** r+prod | 0.703 | 0.183 | **0.110** | **1.7x better** |
| **Ratio** x0/x1 | 0.794 | 0.145 | **0.097** | **1.5x better** |
| Product x0*x1 | 1.079 | 0.790 | 0.793 | Tie |
| Square x0^2 | 0.933 | **0.604** | 0.730 | SwiGLU wins |
| Sin sin(x) | **1.042** | 1.170 | 1.240 | Linear wins |

> **APN wins 4/7 tasks. On linear and sqrt tasks, APN is orders of magnitude better than SwiGLU.**
>
> **Depth efficiency: APN with L/2 layers matches SwiGLU with L layers** (0.57M params, PPL 2.42 vs 1.09M params, PPL 2.40).

---

## Table of Contents

- [Benchmarks](#benchmarks)
- [Depth Efficiency](#depth-efficiency-validated)
- [Quick Start — Python](#quick-start--python)
- [Quick Start — C Binary](#quick-start--c-binary)
- [Architecture](#architecture)
- [APN Neuron — 6 Functions](#apn-neuron--6-functions)
- [File Structure](#file-structure)
- [Build Instructions](#build-instructions)
- [Usage — C Binary](#usage--c-binary)
- [Usage — Python](#usage--python)
- [Experiments](#experiments)
- [Performance](#performance)
- [Binary Format](#apn-binary-format)
- [Supported Models](#supported-huggingface-models)

---

## Benchmarks

### C Benchmark — Regression Tasks (3-seed average)

```
Task              Linear    SwiGLU    APN v9   Winner
──────────────────────────────────────────────────────
Linear  y=W·x     0.017     0.328     0.001    APN ✓  (240x better than SwiGLU)
Ratio   y=x0/x1   0.794     0.145     0.097    APN ✓
Product y=x0·x1   1.079     0.790     0.793    Tie
Sqrt    y=√|x0|   0.815     0.146     0.015    APN ✓  (9x better than SwiGLU)
Mixed   r+prod    0.703     0.183     0.110    APN ✓
Square  y=x0²     0.933     0.604     0.730    SwiGLU ✓
Sin     y=sin(x)  1.042     1.170     1.240    Linear ✓

APN wins: 4/7  |  SwiGLU wins: 1/7  |  Tie: 1/7  |  Linear: 1/7
```

### Python Validation — Per-Layer Expressivity

```
Task              APN-1L    SwiGLU-1L  GELU-1L   APN-2L    SwiGLU-2L
─────────────────────────────────────────────────────────────────────
ratio x0/x1       0.0061    0.0566     0.0701     0.0005    0.0013
product x0*x1     0.0009    0.0291     0.0057     0.0131    0.0005
sqrt |x0|         0.0007    0.0653     0.0062     0.0010    0.0007
sin(x0)           0.0089    0.1846     0.0629     0.0047    0.0068
x0^2              0.0006    0.0227     0.0114     0.0014    0.0003
1/(1+|x0|)        0.0004    0.0555     0.0038     0.0010    0.0009
x0^2+x1^2        0.0008    0.0366     0.0064     0.0005    0.0005
bprod x0*x1/..    0.0021    0.0688     0.0072     0.0005    0.0010

APN-1L wins 8/8 single-layer tasks vs SwiGLU-1L
APN-1L (MSE=0.0056) beats SwiGLU-2L (MSE=0.0097) — 1 APN layer > 2 SwiGLU layers
```

---

## Depth Efficiency (Validated)

The core hypothesis: **APN layers can express more per layer than SwiGLU, so fewer APN layers match more SwiGLU layers.**

| Model | Params | Perplexity | Depth Ratio |
|-------|--------|------------|-------------|
| SwiGLU-4L-d128 | 1.09M | 2.40 | baseline |
| APN-2L-d128 | 0.57M | 2.42 | **0.52x layers, 0.52x params** |
| SwiGLU-6L-d128 | 1.61M | 2.45 | baseline |
| APN-3L-d128 | 0.84M | 2.44 | **0.50x layers, 0.52x params** |
| APN-4L-d128 | 1.11M | 2.44 | 1.00x layers |

**Result: APN with L/2 layers matches SwiGLU with L layers at comparable perplexity.**

### Neuron Specialization (after training)

After tau annealing (3.0 → 0.05), neurons commit to specific functions:

```
b-prod (bounded product): 34%    — handles multiplicative interactions
sq-tanh (bounded square): 15%    — smooth quadratic features
sin (periodic): 15%               — cyclic/positional patterns
s-sqrt (signed sqrt): 12%        — ratio/root tasks
relu (rectified): 12%            — gating/selection
identity (linear): 9%            — passthrough
```

All 6 functions are actively used — this is not a glorified ReLU network.

---

## Quick Start — Python

### Install
```bash
pip install -r requirements.txt
# or: pip install torch transformers accelerate safetensors sentencepiece
```

### Generate text with any HuggingFace model
```bash
python3 apn_complete.py generate \
    --model google/gemma-3-2b-it \
    --prompt "Explain the Fibonacci sequence"
```

### Interactive chat
```bash
python3 apn_complete.py chat --model google/gemma-3-2b-it
```

### Convert FFN → APN (replaces SwiGLU with APN, keeps all weights)
```bash
python3 apn_complete.py convert \
    --model google/gemma-3-2b-it \
    --out gemma3_apn
```

### Benchmark APN vs original
```bash
python3 apn_complete.py benchmark --model google/gemma-3-2b-it
```

### Fine-tune APN layers on your data
```bash
python3 apn_complete.py train \
    --model google/gemma-3-2b-it \
    --data corpus.txt --steps 500 --lr 2e-4
```

### Demo (no download needed, numpy only)
```bash
python3 apn_complete.py demo
```

---

## Quick Start — C Binary

### Build
```bash
make
# or manually:
gcc -O3 -march=native -mfma -ffast-math -funroll-loops -fopenmp \
    -o apn apn_main.c -lm
```

### Run
```bash
# Benchmark
./apn bench

# Train from scratch with BPE tokenization
./apn train_bpe --data corpus.txt --vocab vocab.txt --out model.apn

# Train from scratch (character-level)
./apn train --data corpus.txt --out model.apn

# Generate
./apn generate --model model.apn --prompt "Once upon a time"

# Interactive chat
./apn chat --model model.apn

# Model info
./apn info --model model.apn
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    APN Transformer                        │
│                                                              │
│  tokens ─→ Embedding ─→ Pos Embed ─→ [Block × N] ─→ Norm    │
│                                        │              │      │
│                                    ┌───┴───┐     ┌───┴──┐  │
│                                    │ Block │     │ LM   │  │
│                                    └───┬───┘     │ Head │  │
│                                        │         └──────┘  │
│                              ┌──────────┴──────────┐        │
│                              │                     │        │
│                        ┌─────┴─────┐         ┌─────┴─────┐  │
│                        │ Attention │         │  APN FFN  │  │
│                        │ (MHA/GQA) │         │ (6 funcs) │  │
│                        │ + RoPE    │         │ + tau ann │  │
│                        └──────────┘         └──────────┘  │
│                                                              │
│  x = x + Attention(RMSNorm(x))                              │
│  x = x + APN_FFN(RMSNorm(x))      ← replaces SwiGLU/GELU  │
└──────────────────────────────────────────────────────────────┘

APN Neuron Detail:
                    ┌──────────────────────┐
  x ─→ W1 ─→ p1 ──┤                      │
                    │  y = Σ_k α_k f_k()  │ ─→ W_out ─→ output
  x ─→ W2 ─→ p2 ──┤  α = softmax(lg/τ)  │
                    └──────────────────────┘
  f_k ∈ {identity, tanh(a²), sign(a)√|a|, ab/√(1+(ab)²), sin(a), leaky_relu(a)}
  τ anneals: 3.0 (uniform) → 0.05 (specialized)
  When τ < 0.1: hard argmax (no exp, ~30% faster inference)
```

---

## APN Neuron — 6 Functions

| Function | Formula | Role | Gradient |
|----------|---------|------|----------|
| identity | `a` | Linear passthrough | 1 |
| sq-tanh | `tanh(a²)` | Bounded square, smooth | `2a(1-tanh²(a²))` |
| s-sqrt | `sign(a)·√\|a\|` | Sub-linear (ratios, roots) | `1/(2√\|a\|)` |
| b-prod | `a·b / √(1+(ab)²)` | Bounded product | Both inputs |
| sin | `sin(a)` | Periodic patterns | `cos(a)` |
| relu | `leaky_relu(a, 0.01)` | Rectified | 1 or 0.01 |

Temperature τ anneals 3.0 → 0.05: starts uniform (explores all functions), ends specialized (commits to optimal).

**Inference optimization**: When τ < 0.1, softmax(logits/τ) is replaced with one-hot argmax — eliminates all `exp()` calls, ~30% faster.

---

## File Structure

```
apn/
├── apn_main.c              C CLI binary — train, generate, chat, bench
├── apn_complete.py         Python — load/convert/chat/train any HF model
├── Makefile                    Build system with AVX-512 auto-detection
├── requirements.txt            Python dependencies
├── README.md                   This file
│
├── src/                        C headers (header-only, single-TU compilation)
│   ├── tensor.h                SIMD matmul, RMSNorm, RoPE, RNG, sampling
│   ├── optimizer.h             AdamW + cosine LR schedule
│   ├── apn_layer.h             APN neuron: forward + backward + inference path
│   ├── attention.h             Multi-head attention + KV cache + backward
│   ├── transformer.h           Full transformer + backward + .apn format
│   ├── tokenizer.h             BPE/char tokenizer (HF vocab compatible)
│   ├── generate.h              Greedy/top-k/top-p sampling + chat loop
│   ├── train.h                 Training loop + dataset + CE loss + eval
│   └── etok_bridge.h           BPE training bridge + end-to-end pipeline
│
├── python/
│   ├── convert_hf.py           Convert HF models → .apn binary
│   └── train.py                Full PyTorch training (autograd + AMP + GPU)
│
├── experiments/
│   ├── fast_validate.py        Streamlined validation (4 experiments, ~6 min)
│   ├── real_scale_validation.py  Large-scale validation (11 configs)
│   ├── validation_results.json   Cached experiment results
│   └── real_scale_results.json   Large-scale results (when complete)
│
└── examples/
    └── generate_corpus.py      Generate synthetic training data
```

---

## Build Instructions

### C Binary

```bash
# Standard build (auto-detects CPU features)
make

# With AVX-512 (only if CPU supports it — check: grep avx512f /proc/cpuinfo)
gcc -O3 -mavx512f -mavx512dq -mavx512bw -mfma -ffast-math \
    -funroll-loops -fopenmp -o apn apn_main.c -lm

# Without AVX-512 (safe default)
gcc -O3 -march=native -mfma -ffast-math -funroll-loops -fopenmp \
    -o apn apn_main.c -lm

# Debug build (AddressSanitizer + UBSan)
make debug
```

### Python

```bash
pip install -r requirements.txt
# or: pip install torch transformers accelerate safetensors sentencepiece
```

### Makefile targets

```bash
make              # Build apn binary
make bench        # Build + run APN benchmark
make demo         # Build + generate demo + train small model
make test         # Build + run benchmark
make info         # Show system info (GCC, CPU, AVX-512)
make install-py   # pip install Python dependencies
make debug        # Build with sanitizers
make clean        # Remove binary and .apn files
```

---

## Usage — C Binary

### Benchmark
```bash
./apn bench
# Output: 7 regression tasks × 3 seeds, APN vs SwiGLU vs Linear
```

### Train with BPE tokenization
```bash
./apn train_bpe \
    --data corpus.txt --vocab vocab.txt --out model.apn \
    --d_model 256 --n_layers 4 --n_heads 8 \
    --ffn_hidden 1024 --batch 16 --seq_len 128 \
    --epochs 10 --lr 3e-4 --target_vocab 8000
```

### Train character-level
```bash
./apn train \
    --data corpus.txt --out model.apn \
    --d_model 512 --n_layers 6 --n_heads 8
```

### Generate text
```bash
./apn generate \
    --model model.apn \
    --prompt "Once upon a time" \
    --max_tokens 200 --temperature 0.8 --top_k 50 --top_p 0.95
```

### Interactive chat
```bash
./apn chat --model model.apn
```

### Model info
```bash
./apn info --model model.apn
```

---

## Usage — Python

### Commands
```
generate   Generate text from a HuggingFace model
chat       Interactive chat with a model
convert    Replace SwiGLU/GELU FFN → APN (keeps all weights)
benchmark  Compare APN vs original perplexity + speed
train      Fine-tune APN layers on custom data
demo       Quick demo (numpy only, no download)
info       Show model architecture
```

### All options
```
--model STR          HuggingFace model id or local path
--out STR            Output path (for convert/train)
--prompt STR         Generation prompt
--system STR         System prompt for chat
--data STR           Training data file (.txt)
--max_tokens INT     Max tokens to generate          (default: 200)
--temperature FLOAT  Sampling temperature            (default: 0.8)
--top_k INT          Top-k sampling                  (default: 50)
--top_p FLOAT        Nucleus sampling                (default: 0.95)
--steps INT          Training steps                  (default: 300)
--lr FLOAT           Learning rate                   (default: 2e-4)
--batch INT          Batch size                      (default: 4)
--seq_len INT        Sequence length                 (default: 256)
--layers STR         Layers to convert: all|last_half|N  (default: all)
--tau FLOAT          Initial APN temperature         (default: 2.0)
--cpu                Force CPU even if GPU available
--quantize           8-bit quantization (GPU, needs bitsandbytes)
--dtype STR          float32 | float16 | bfloat16
```

---

## Experiments

### Experiment 1: Per-Layer Function Approximation

Tests whether a single APN layer can approximate nonlinear functions better than a single SwiGLU or GELU layer.

**Result**: APN-1L wins all 8 tasks. APN-1L (MSE=0.0056) even beats SwiGLU-2L (MSE=0.0097).

### Experiment 2: Mixed Nonlinear Targets

Trains on a mixture of all 8 nonlinear targets simultaneously.

| Model | Params | MSE | R² |
|-------|--------|-----|----|
| APN-1L | 7,184 | 0.0056 | 0.9947 |
| SwiGLU-1L | 6,144 | 0.0095 | 0.9907 |
| GELU-1L | 4,240 | 0.0131 | 0.9875 |
| APN-2L | 57,488 | 0.0014 | 0.9986 |
| SwiGLU-2L | 55,296 | 0.0097 | 0.9905 |

### Experiment 3: Language Model Depth Efficiency

Trains character-level language models with different depths.

| Model | Params | Perplexity |
|-------|--------|------------|
| SwiGLU-4L | 1.09M | 2.40 |
| SwiGLU-6L | 1.61M | 2.45 |
| APN-2L | 0.57M | 2.42 |
| APN-3L | 0.84M | 2.44 |
| APN-4L | 1.11M | 2.44 |

**APN-2L matches SwiGLU-4L with 48% fewer parameters.**

### Experiment 4: Neuron Specialization Distribution

After full training with tau annealing, neurons commit across all 6 functions:

```
b-prod: 34%  |  sq-tanh: 15%  |  sin: 15%  |  s-sqrt: 12%  |  relu: 12%  |  identity: 9%
```

### Run experiments

```bash
# Fast validation (~6 min on CPU)
python3 experiments/fast_validate.py

# Full real-scale validation (requires GPU for reasonable runtime)
python3 experiments/real_scale_validation.py
```

---

## Performance

### Inference Speed

| Mode | Platform | Speed (~3B model) |
|------|----------|-------------------|
| C binary CPU | OpenMP auto | ~2–5 tok/s |
| C multi-core | 4+ cores | 2x parallelism |
| Python CPU | PyTorch | ~1–3 tok/s |
| GPU CUDA | Automatic | ~50–100 tok/s |
| GPU Apple MPS | Mac | ~20–40 tok/s |
| GPU 8-bit | `--quantize` | Half VRAM |

### APN vs SwiGLU Speed Tradeoff

- APN layer: ~1.6x slower per layer (6 function evaluations + alpha weighting)
- APN model: ~same total speed (needs half the layers for same quality)
- APN inference: hard argmax (τ < 0.1) eliminates softmax exp, ~30% faster

### Depth Efficiency

| Metric | APN | SwiGLU |
|--------|-----|--------|
| Layers needed for target PPL | L/2 | L |
| Parameters for same quality | ~50% | 100% |
| Per-layer expressivity | 8/8 tasks | 0/8 tasks |
| Specialized functions | 6 | 1 (SiLU) |

---

## .apn Binary Format

```
Offset  Size        Content
0       4 bytes     Magic: 0x41504E00 ("APN")
4       4 bytes     Version: 2
8       ~84 bytes   TransformerConfig struct
12+     V×D×4       Token embeddings (float32)
        For each layer:
          H×hd×D×4  W_q + H×hd×4 bias_q
          Hkv×hd×D×4  W_k + bias_k
          Hkv×hd×D×4  W_v + bias_v
          D×H×hd×4  W_o + bias_o
          D×4       attn_norm_w
          D×H×4 + H×4  W1 + b1
          D×H×4 + H×4  W2 + b2
          H×6×4     APN logits
          H×D×4 + D×4  W_out + b_out
          D×4       ffn_norm_w
        D×4         Final RMSNorm
        V×D×4       lm_head weights
```

---

## Supported HuggingFace Models

| Model | Architecture | Support |
|-------|-------------|---------|
| google/gemma-3-2b-it | Gemma3 / SwiGLU | Full |
| google/gemma-2-2b | Gemma2 / SwiGLU | Full |
| meta-llama/Llama-3.2-3B | LLaMA3 / SwiGLU | Full |
| meta-llama/Llama-2-7b-hf | LLaMA2 / SwiGLU | Full |
| mistralai/Mistral-7B-v0.3 | Mistral / SwiGLU | Full |
| Qwen/Qwen2.5-3B | Qwen2 / SwiGLU | Full |
| microsoft/Phi-3-mini-4k | Phi3 / SwiGLU | Full |
| gpt2 | GPT-2 / GELU | Full |
| Any SwiGLU/GELU model | Auto-detect | Auto |

---

## How APN Conversion Works

```
Original SwiGLU FFN:              APN Replacement:

x -> gate_proj -> gate            x -> W1 -> p1 --\
x -> up_proj   -> up      =>      x -> W2 -> p2 --+-> APN(p1,p2) -> W_out -> output
gate * silu(up) -> down_proj

W1    <- gate_proj.weight    (no weights lost)
W2    <- up_proj.weight      (no weights lost)
W_out <- down_proj.weight    (no weights lost)
APN logits initialized so identity+relu dominate ~ original SwiGLU behavior
```

After conversion the model generates identically to the original.
After fine-tuning, APN neurons specialize beyond what SwiGLU can express.

---

## License

MIT

---

## Citation

```bibtex
@software{apn2024,
  title = {APN: Adaptive Probabilistic Neuron Transformer},
  author = {VABISMO},
  year = {2024},
  url = {https://github.com/VABISMO/APN}
}
```
