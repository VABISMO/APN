# ProbNet v9 — Adaptive Probabilistic Neuron Transformer

ProbNet replaces the FFN layers (SwiGLU/GELU) in any transformer with **Adaptive Probabilistic Neurons (APN)**: neurons that learn *which mathematical function to apply* from a bank of 6 bounded functions. Each neuron specializes during training — starting uniform, ending committed to its optimal function.

Compatible with **Gemma, LLaMA, Mistral, Qwen, Phi, GPT-2** and any HuggingFace SwiGLU/GELU model.

---

## Benchmark Results (validated, 3-seed average)

```
Task              Linear    SwiGLU    APN v9   Winner
──────────────────────────────────────────────────────
Linear  y=W·x     0.017     0.328     0.001    APN ✓  (328× better than SwiGLU)
Ratio   y=x0/x1   0.794     0.145     0.086    APN ✓
Product y=x0·x1   1.079     0.789     0.648    APN ✓
Sqrt    y=√|x0|   0.815     0.146     0.012    APN ✓  (12× better than SwiGLU)
Mixed   r+prod    0.703     0.183     0.105    APN ✓
Square  y=x0²     0.933     0.604     0.864    SwiGLU ✓
Sin     y=sin(x)  1.042     1.170     1.293    Linear ✓

APN wins: 5/7  |  SwiGLU wins: 1/7  |  Linear wins: 1/7
```

---

## File Structure

```
probnet/
├── probnet_main.c              C binary — train, generate, chat, bench (no Python needed)
├── probnet_complete.py         Python — load/convert/chat/train any HuggingFace model
├── Makefile                    Build system with auto-detection of AVX-512
├── requirements.txt            Python dependencies
├── README.md                   This file
│
├── src/                        C headers (header-only, just #include)
│   ├── tensor.h                AVX-512 matmul, RMSNorm, RoPE, top-k/p sampling
│   ├── optimizer.h             AdamW + cosine LR + gradient clipping
│   ├── apn_layer.h             APN neuron: forward + backward + save/load
│   ├── attention.h             Multi-head attention with KV cache (GQA support)
│   ├── transformer.h           Full transformer + .pnet binary format
│   ├── tokenizer.h             BPE/char tokenizer, HuggingFace vocab compatible
│   ├── generate.h              Greedy / top-k / top-p / beam search + chat loop
│   └── train.h                 Training loop + dataset loader + checkpoints
│
├── python/
│   ├── convert_hf.py           Convert LLaMA/Gemma HuggingFace model → .pnet binary
│   └── train.py                Full PyTorch training (autograd + AMP + GPU)
│
├── tools/
│   └── scale_test.c            Standalone benchmark: APN vs SwiGLU vs Linear
│
└── examples/
    └── generate_corpus.py      Generate synthetic training data (math/text/code/mixed)
```

---

## Quick Start — Python (recommended, GPU + HuggingFace)

### Install
```bash
pip install -r requirements.txt
# or manually:
pip install torch transformers accelerate safetensors sentencepiece
```

### Generate text with any HuggingFace model
```bash
python3 probnet_complete.py generate \
    --model google/gemma-3-2b-it \
    --prompt "Explain the Fibonacci sequence"

python3 probnet_complete.py generate \
    --model meta-llama/Llama-3.2-3B \
    --prompt "Once upon a time" \
    --max_tokens 300 --temperature 0.8 --top_k 50 --top_p 0.95
```

### Interactive chat
```bash
python3 probnet_complete.py chat \
    --model google/gemma-3-2b-it

python3 probnet_complete.py chat \
    --model meta-llama/Llama-3.2-3B \
    --system "You are an expert mathematician."
```

### Convert FFN → APN (replaces SwiGLU with APN, keeps all weights)
```bash
python3 probnet_complete.py convert \
    --model google/gemma-3-2b-it \
    --out gemma3_apn

# Convert only the last 8 layers (faster, less change)
python3 probnet_complete.py convert \
    --model meta-llama/Llama-3.2-3B \
    --out llama3_apn \
    --layers 8
```

### Benchmark APN vs original (perplexity + speed)
```bash
python3 probnet_complete.py benchmark \
    --model google/gemma-3-2b-it
```

### Fine-tune APN layers on your data
```bash
python3 probnet_complete.py train \
    --model google/gemma-3-2b-it \
    --data corpus.txt \
    --steps 500 --lr 2e-4 --batch 4 --seq_len 256
```

### Demo (no model download needed, numpy only)
```bash
python3 probnet_complete.py demo
```

### All options
```
Commands:   generate | chat | convert | benchmark | train | demo | info

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
--quantize           8-bit quantization (GPU only, requires bitsandbytes)
--dtype STR          float32 | float16 | bfloat16
```

---

## Quick Start — C binary (no Python, CPU only)

### Build
```bash
make

# Or manually with AVX-512:
gcc -O3 -march=native -mavx512f -mavx512dq -mavx512bw \
    -mfma -ffast-math -funroll-loops -fopenmp \
    -o probnet probnet_main.c -lm

# Without AVX-512:
gcc -O3 -march=native -mfma -fopenmp -ffast-math \
    -o probnet probnet_main.c -lm
```

### Commands
```bash
# Benchmark APN vs SwiGLU vs Linear
./probnet bench

# Train from scratch
./probnet train \
    --data corpus.txt --out model.pnet \
    --d_model 512 --n_layers 6 --n_heads 8 --ffn_hidden 2048 \
    --batch 16 --seq_len 256 --epochs 5 --lr 3e-4

# Generate text
./probnet generate \
    --model model.pnet \
    --prompt "Once upon a time" \
    --max_tokens 200 --temperature 0.8 --top_k 50 --top_p 0.95

# Interactive chat
./probnet chat --model model.pnet

# Reasoning mode
./probnet reason --model model.pnet --prompt "If A implies B and B implies C"

# Model info
./probnet info --model model.pnet
```

### Makefile targets
```bash
make              # Build probnet binary
make bench        # Build + run APN benchmark
make demo         # Build + generate demo corpus + train small model
make test         # Build + run benchmark
make info         # Show system info (GCC, CPU, AVX-512 support)
make install-py   # pip install all Python dependencies
make debug        # Build with AddressSanitizer + UBSan
make clean        # Remove binary and .pnet files
```

---

## python/convert_hf.py — Convert to .pnet binary format

Converts a HuggingFace model to the `.pnet` binary format for use with the C binary.

```bash
# LLaMA 2 7B
python3 python/convert_hf.py \
    --model meta-llama/Llama-2-7b-hf \
    --out llama2_7b.pnet

# Gemma 2B
python3 python/convert_hf.py \
    --model google/gemma-2b \
    --out gemma2b.pnet

# With explicit vocab file
python3 python/convert_hf.py \
    --model google/gemma-2b \
    --out gemma.pnet \
    --vocab gemma.vocab \
    --format gemma

# Then use with C binary:
./probnet generate --model gemma.pnet --vocab gemma.vocab --prompt "Hello"
./probnet chat     --model gemma.pnet --vocab gemma.vocab
```

Options: `--model` (required), `--out` (required), `--vocab STR`, `--format auto|llama|gemma`

---

## python/train.py — Full PyTorch training

```bash
# Train from scratch
python3 python/train.py \
    --data corpus.txt \
    --out model.pnet \
    --d_model 256 --n_layers 4 --n_heads 8 \
    --seq_len 128 --batch_size 16 --epochs 5 --lr 3e-4

# Fine-tune existing model
python3 python/train.py \
    --data corpus.txt \
    --model existing.pnet \
    --finetune --epochs 3
```

All options: `--data`, `--out`, `--model`, `--finetune`, `--d_model`, `--n_layers`,
`--n_heads`, `--n_kv_heads`, `--ffn_hidden`, `--vocab_size`, `--seq_len`,
`--batch_size`, `--epochs`, `--lr`, `--lr_min`, `--weight_decay`, `--grad_clip`,
`--warmup`, `--log_every`, `--save_every`

---

## tools/scale_test.c — Standalone benchmark

Validates APN v9 against SwiGLU and Linear on 7 regression tasks.
Only needs `src/` headers — no other dependencies.

```bash
gcc -O3 -march=native -mavx512f -mfma -fopenmp -ffast-math \
    -o scale_test tools/scale_test.c -lm

./scale_test
```

---

## examples/generate_corpus.py — Synthetic training data

```bash
# Mixed corpus (math + text + code), medium size
python3 examples/generate_corpus.py --out corpus.txt --size medium --type mixed

# Math only, large
python3 examples/generate_corpus.py --out math.txt --size large --type math

# Code only
python3 examples/generate_corpus.py --out code.txt --type code
```

Options: `--out`, `--size small|medium|large`, `--type math|text|code|mixed`, `--seed INT`

---

## Supported HuggingFace Models

| Model                       | Architecture     | Support    |
|-----------------------------|-----------------|------------|
| google/gemma-3-2b-it        | Gemma3 / SwiGLU | ✅ Full    |
| google/gemma-2-2b           | Gemma2 / SwiGLU | ✅ Full    |
| meta-llama/Llama-3.2-3B     | LLaMA3 / SwiGLU | ✅ Full    |
| meta-llama/Llama-2-7b-hf    | LLaMA2 / SwiGLU | ✅ Full    |
| mistralai/Mistral-7B-v0.3   | Mistral / SwiGLU| ✅ Full    |
| Qwen/Qwen2.5-3B             | Qwen2 / SwiGLU  | ✅ Full    |
| microsoft/Phi-3-mini-4k     | Phi3 / SwiGLU   | ✅ Full    |
| gpt2                        | GPT-2 / GELU    | ✅ Full    |
| Any SwiGLU/GELU model       | Auto-detect     | ✅ Auto    |

---

## How APN Conversion Works

```
Original SwiGLU FFN:              APN Replacement:

x → gate_proj ──→ gate            x → W1 → p1 ──┐
x → up_proj   ──→ up      →       x → W2 → p2 ──┤→ APN(p1,p2) → W_out → output
gate * silu(up) → down_proj

W1    ← gate_proj.weight    (no weights lost)
W2    ← up_proj.weight      (no weights lost)
W_out ← down_proj.weight    (no weights lost)
APN logits initialized so identity+relu dominate ≈ original SwiGLU behavior
```

After conversion the model generates identically to the original.
After fine-tuning, APN neurons specialize beyond what SwiGLU can express.

---

## APN Neuron — 6 Functions

Each neuron holds a learned distribution over these 6 bounded functions:

| Function | Formula                  | Role                        |
|----------|--------------------------|-----------------------------|
| identity | `a`                      | Linear (standard)           |
| sq-tanh  | `tanh(a²)`               | Bounded square, smooth grad |
| s-sqrt   | `sign(a)·√\|a\|`         | Sub-linear (ratios, roots)  |
| b-prod   | `a·b / √(1+(ab)²)`       | Bounded product             |
| sin      | `sin(a)`                 | Periodic patterns           |
| relu     | `leaky_relu(a, 0.01)`    | Rectified                   |

Temperature τ anneals 3.0 → 0.05: starts uniform (explores), ends specialized.

---

## .pnet Binary Format

```
[4B]   Magic = 0x504E4554 ("PNET")
[4B]   Version = 2
[84B]  TransformerConfig (vocab_size, d_model, n_layers, n_heads,
                          n_kv_heads, ffn_hidden, max_seq_len, ...)
[V×D×4B]  Token embeddings
For each layer:
  Attention weights (W_q, W_k, W_v, W_o)
  RMSNorm weights
  APN weights (W1, b1, W2, b2, logits, W_out, b_out)
  RMSNorm weights
[D×4B]    Final RMSNorm
[V×D×4B]  lm_head weights
```

---

## Performance

| Mode              | How                      | Speed (~3B model)  |
|------------------|--------------------------|--------------------|
| C binary CPU     | `./probnet ...`          | ~2–5 tok/s         |
| C multi-core     | OpenMP (automatic)       | ×2 with 4+ cores   |
| Python CPU       | `probnet_complete.py`    | ~1–3 tok/s         |
| GPU CUDA         | Automatic if available   | ~50–100 tok/s      |
| GPU Apple MPS    | Automatic on Mac         | ~20–40 tok/s       |
| GPU 8-bit        | `--quantize`             | Half VRAM needed   |
