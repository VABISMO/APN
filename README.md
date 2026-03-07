# ProbNet v9 — Adaptive Probabilistic Neuron Transformer

Sistema completo de investigación LLM que reemplaza las capas FFN (SwiGLU/GELU) de
cualquier transformer por **Neuronas Probabilísticas Adaptativas (APN)**: neuronas
que aprenden *qué función matemática aplicar* de un banco de 6 funciones acotadas.

---

## Estructura de ficheros

```
probnet_FINAL/
│
├── probnet_main.c              CLI C completo — compilar y usar sin Python
├── probnet_complete.py         Sistema Python APN v9 — fichero principal
├── Makefile                    Build automático con auto-detección AVX-512
├── requirements.txt            Dependencias Python
├── README.md                   Este fichero
│
├── src/                        Cabeceras C (header-only, basta con incluirlas)
│   ├── tensor.h                Operaciones AVX-512: matmul, RMSNorm, RoPE, sampling
│   ├── optimizer.h             AdamW + cosine LR schedule + gradient clipping
│   ├── apn_layer.h             Neurona APN v9: forward + backward + save/load
│   ├── attention.h             Multi-head attention con KV cache (soporta GQA)
│   ├── transformer.h           Transformer completo + formato binario .pnet
│   ├── tokenizer.h             Tokenizador BPE/char compatible con HuggingFace
│   ├── generate.h              Generación: greedy / top-k / top-p / beam + chat
│   └── train.h                 Bucle de entrenamiento + dataset + checkpoints
│
├── python/
│   ├── convert_hf.py           Convierte LLaMA/Gemma HuggingFace → .pnet binario
│   └── train.py                Entrenamiento PyTorch completo (autograd + AMP GPU)
│
├── tools/
│   └── scale_test.c            Benchmark standalone: APN vs SwiGLU vs Linear (7 tareas)
│
├── examples/
│   └── generate_corpus.py      Genera corpus de entrenamiento (math/text/code/mixed)
│
└── probnet_v1/                 Sistema original v1 — ProbNetLayer determinista
    ├── probnet_layer.py         Capa ProbNetLayer: predicción por ratios sin pesos
    ├── probnet_transformer.py   GPT con ProbNetLayer + convert_to_probnet() + generate()
    ├── convert_model.py         Convierte cualquier modelo HuggingFace a ProbNet v1
    ├── train_v1.py              Entrena desde cero con ProbNetLayer
    └── README.md                Documentación específica de v1
```

---

## Qué son las neuronas APN

En lugar de que cada neurona compute `gate * silu(up)` (SwiGLU estándar),
cada neurona APN mantiene una distribución de probabilidad sobre 6 funciones acotadas:

| Función    | Fórmula                    | Rol                              |
|-----------|---------------------------|----------------------------------|
| identity  | `a`                       | Lineal (como capa estándar)      |
| sq-tanh   | `tanh(a²)`                | Cuadrado acotado, gradiente suave|
| s-sqrt    | `sign(a)·√|a|`            | Sub-lineal (buena para ratios)   |
| b-prod    | `a·b / √(1+(ab)²)`        | Producto acotado (multiplicativo)|
| sin       | `sin(a)`                  | Periódica                        |
| relu      | `leaky_relu(a, 0.01)`     | Rectificada estándar             |

La temperatura τ se anula de 3.0 → 0.05 durante el entrenamiento:
comienza **uniforme** (explora), termina **especializada** (cada neurona elige su función).

---

## Resultados del benchmark (validados)

```
Tarea               Linear    SwiGLU    APN v9   Ganador
─────────────────────────────────────────────────────────
Linear  y=W·x       0.017     0.328     0.001    APN ✓  (328× mejor)
Ratio   y=x0/x1     0.794     0.145     0.086    APN ✓  (1.7× mejor)
Product y=x0·x1     1.079     0.789     0.648    APN ✓
Sqrt    y=√|x0|     0.815     0.146     0.012    APN ✓  (12× mejor)
Mixed   r+prod      0.703     0.183     0.105    APN ✓
Square  y=x0²       0.933     0.604     0.864    SwiGLU ✓
Sin     y=sin(x)    1.042     1.170     1.293    Linear ✓

APN wins: 5/7  |  SwiGLU wins: 1/7  |  Linear wins: 1/7
```

---

## Inicio rápido — opción A: binario C (sin Python)

```bash
# 1. Compilar
make

# O manualmente:
gcc -O3 -march=native -mavx512f -mavx512dq -mavx512bw -mfma \
    -ffast-math -funroll-loops -fopenmp \
    -o probnet probnet_main.c -lm

# Sin AVX-512:
gcc -O3 -march=native -mfma -fopenmp -ffast-math \
    -o probnet probnet_main.c -lm

# 2. Benchmark APN vs SwiGLU vs Linear
./probnet bench

# 3. Entrenar desde cero
./probnet train --data corpus.txt --out model.pnet \
    --d_model 512 --n_layers 6 --n_heads 8 --ffn_hidden 2048 \
    --batch 16 --seq_len 256 --epochs 5 --lr 3e-4

# 4. Generar texto
./probnet generate --model model.pnet \
    --prompt "Once upon a time" \
    --max_tokens 200 --temperature 0.8 --top_k 50 --top_p 0.95

# 5. Chat interactivo
./probnet chat --model model.pnet

# 6. Info del modelo
./probnet info --model model.pnet

# 7. Razonamiento (modo especial)
./probnet reason --model model.pnet --prompt "Si A implica B y B implica C"
```

### Targets del Makefile

```bash
make              # Compila probnet
make bench        # Compila + ejecuta benchmark
make demo         # Compila + genera corpus demo + entrena modelo pequeño
make test         # Compila + ejecuta benchmark (alias de bench)
make info         # Info del sistema (GCC, CPU, AVX-512)
make install-py   # pip install torch transformers accelerate safetensors sentencepiece
make debug        # Build con AddressSanitizer + UBSan
make clean        # Elimina binarios y modelos
```

---

## Inicio rápido — opción B: Python APN v9 (GPU + HuggingFace)

```bash
# 1. Instalar dependencias
pip install -r requirements.txt
# o: pip install torch transformers accelerate safetensors sentencepiece

# 2. Demo sin modelo (funciona con numpy solo, sin torch ni red)
python3 probnet_complete.py demo

# 3. Generar texto con modelo HuggingFace (sin conversión, modelo original)
python3 probnet_complete.py generate \
    --model google/gemma-3-2b-it \
    --prompt "Explain quantum computing"

# 4. Convertir FFN → APN y guardar
python3 probnet_complete.py convert \
    --model google/gemma-3-2b-it \
    --out gemma3_apn \
    --layers all          # all | last_half | N (últimas N capas)

# 5. Chat interactivo (con historial, system prompt, streaming)
python3 probnet_complete.py chat \
    --model google/gemma-3-2b-it \
    --system "Eres un asistente experto en matemáticas."

# 6. Benchmark APN vs original (perplexidad, velocidad, razonamiento)
python3 probnet_complete.py benchmark \
    --model meta-llama/Llama-3.2-3B

# 7. Fine-tuning de capas APN sobre tus datos
python3 probnet_complete.py train \
    --model google/gemma-3-2b-it \
    --data corpus.txt \
    --steps 500 --lr 2e-4 --batch 4 --seq_len 256

# 8. Info del modelo
python3 probnet_complete.py info \
    --model google/gemma-3-2b-it
```

### Opciones completas de probnet_complete.py

```
Comandos:  generate | chat | convert | benchmark | train | demo | info

--model STR          Modelo HuggingFace o ruta local
--out STR            Ruta de salida (convertir/entrenar)
--prompt STR         Prompt de generación (default: "Hello, I am an AI assistant and")
--system STR         System prompt para chat
--data STR           Fichero .txt para entrenamiento
--max_tokens INT     Máximo de tokens a generar (default: 200)
--temperature FLOAT  Temperatura de muestreo (default: 0.8)
--top_k INT          Top-k sampling (default: 50)
--top_p FLOAT        Nucleus sampling (default: 0.95)
--steps INT          Pasos de entrenamiento (default: 300)
--lr FLOAT           Learning rate (default: 2e-4)
--batch INT          Batch size entrenamiento (default: 4)
--seq_len INT        Longitud de secuencia (default: 256)
--layers STR         Capas a convertir: all | last_half | N (default: all)
--tau FLOAT          Temperatura APN inicial (default: 2.0)
--cpu                Forzar CPU aunque haya GPU
--quantize           Cuantización 8-bit (solo GPU, requiere bitsandbytes)
--dtype STR          float32 | float16 | bfloat16
```

---

## Script: python/convert_hf.py

Convierte modelos HuggingFace al **formato binario .pnet** para usar con el binario C.

```bash
# LLaMA 2
python3 python/convert_hf.py \
    --model meta-llama/Llama-2-7b-hf \
    --out llama2_7b.pnet

# Gemma 2B
python3 python/convert_hf.py \
    --model google/gemma-2b \
    --out gemma2b.pnet

# Modelo local
python3 python/convert_hf.py \
    --model ./mi_modelo_local \
    --out mi_modelo.pnet

# Con vocab explícito
python3 python/convert_hf.py \
    --model google/gemma-2b \
    --out gemma.pnet \
    --vocab gemma.vocab \
    --format gemma

# Después usar con el C:
./probnet generate --model gemma.pnet --vocab gemma.vocab --prompt "Hola"
./probnet chat     --model gemma.pnet --vocab gemma.vocab
```

Opciones: `--model` (requerido), `--out` (requerido), `--vocab`, `--format auto|llama|gemma`

---

## Script: python/train.py

Entrenamiento completo con PyTorch autograd, soporta CPU y GPU.

```bash
# Entrenamiento desde cero
python3 python/train.py \
    --data corpus.txt \
    --out model.pnet \
    --d_model 256 --n_layers 4 --n_heads 8 \
    --seq_len 128 --batch_size 16 --epochs 5 --lr 3e-4

# Fine-tuning de modelo existente
python3 python/train.py \
    --data corpus.txt \
    --model existing.pnet \
    --finetune --epochs 3
```

Opciones completas: `--data`, `--out`, `--d_model`, `--n_layers`, `--n_heads`,
`--n_kv_heads`, `--ffn_hidden`, `--vocab_size`, `--seq_len`, `--batch_size`,
`--epochs`, `--lr`, `--lr_min`, `--weight_decay`, `--grad_clip`,
`--warmup`, `--log_every`, `--save_every`

---

## Herramienta: tools/scale_test.c

Benchmark standalone que compara APN v9 vs SwiGLU vs Linear en 7 tareas de regresión.
No necesita ningún fichero extra excepto `src/`.

```bash
# Compilar
gcc -O3 -march=native -mavx512f -mfma -fopenmp -ffast-math \
    -o scale_test tools/scale_test.c -lm

# Sin AVX-512
gcc -O3 -march=native -mfma -fopenmp -ffast-math \
    -o scale_test tools/scale_test.c -lm

# Ejecutar
./scale_test
```

Parámetros internos: D=16, H=64, N=800, 600 épocas, 3 seeds, promediado.
Produce tabla con MSE por tarea y ganador confirmado.

---

## Herramienta: examples/generate_corpus.py

Genera corpus de entrenamiento sintético sin necesidad de datos externos.

```bash
# Corpus mixto mediano (math + text + code)
python3 examples/generate_corpus.py --out corpus.txt --size medium --type mixed

# Solo matemáticas
python3 examples/generate_corpus.py --out math.txt --size large --type math

# Solo texto
python3 examples/generate_corpus.py --out text.txt --size small --type text

# Solo código
python3 examples/generate_corpus.py --out code.txt --type code
```

Opciones: `--out` (default: corpus.txt), `--size small|medium|large`,
`--type math|text|code|mixed`, `--seed INT`

---

## probnet_v1/ — Sistema original determinista

La implementación **v1 original** (sin pesos matriciales). Cada `nn.Linear` se
reemplaza por `ProbNetLayer`: predicción determinista por ratios entre elementos
consecutivos. No hay backprop de pesos — solo embeddings y LayerNorm entrenan.

```
probnet_v1/
├── probnet_layer.py        ProbNetLayer + funciones de ratio (predict1, prerr)
├── probnet_transformer.py  GPT completo + convert_to_probnet() + generate() + predict_sequence()
├── convert_model.py        CLI para convertir cualquier modelo HF
├── train_v1.py             Entrenamiento desde cero (solo embeddings + norms)
└── README.md               Documentación específica de v1
```

### Predicción de secuencias OEIS (sin modelo)

```python
from probnet_v1.probnet_transformer import predict_sequence

print(predict_sequence([1, 1, 2, 3, 5, 8, 13, 21], n_next=4))
# → [34, 55, 89, 144]

print(predict_sequence([2, 4, 8, 16, 32], n_next=3))
# → [64, 128, 256]
```

### Convertir GPT-2 a ProbNet v1

```bash
# Desde probnet_v1/
cd probnet_v1
python3 convert_model.py --model gpt2 --generate "The Fibonacci sequence"
python3 convert_model.py --model gpt2 --output gpt2_probnet.pt --window 8 --verbose
```

Opciones: `--model`, `--output`, `--window INT`, `--generate STR`,
`--max_new INT`, `--temperature FLOAT`, `--top_k INT`, `--verbose`

### Entrenar desde cero con ProbNet v1

```bash
cd probnet_v1
python3 train_v1.py                                # demo interno
python3 train_v1.py --dataset ../corpus.txt        # tu fichero
python3 train_v1.py --n_layer 8 --n_embd 512 --window 16 --epochs 10
```

Opciones: `--dataset`, `--n_layer`, `--n_head`, `--n_embd`, `--block_size`,
`--batch_size`, `--epochs`, `--lr`, `--dropout`, `--window`, `--output`

---

## Formato binario .pnet

```
[4 bytes]   Magic = 0x504E4554 ("PNET")
[4 bytes]   Version = 2
[84 bytes]  TransformerConfig struct
            (vocab_size, d_model, n_layers, n_heads, n_kv_heads,
             ffn_hidden, max_seq_len, use_rope, tie_weights)
[V×D×4B]   Token embeddings
[N_layers × (attention weights + RMSNorm + APN weights + RMSNorm)]
[D×4B]     Final RMSNorm
[V×D×4B]   lm_head weights
```

---

## Modelos HuggingFace compatibles

| Modelo                          | Arquitectura        | Conversión     |
|--------------------------------|---------------------|----------------|
| google/gemma-3-2b-it           | Gemma3 / SwiGLU     | ✅ Completa    |
| google/gemma-2-2b              | Gemma2 / SwiGLU     | ✅ Completa    |
| meta-llama/Llama-3.2-3B        | LLaMA3 / SwiGLU     | ✅ Completa    |
| meta-llama/Llama-2-7b-hf       | LLaMA2 / SwiGLU     | ✅ Completa    |
| mistralai/Mistral-7B-v0.3      | Mistral / SwiGLU    | ✅ Completa    |
| Qwen/Qwen2.5-3B                | Qwen2 / SwiGLU      | ✅ Completa    |
| microsoft/Phi-3-mini-4k        | Phi3 / SwiGLU       | ✅ Completa    |
| gpt2                           | GPT-2 / GELU        | ✅ Completa    |
| Cualquier modelo SwiGLU/GELU   | Auto-detección      | ✅ Auto        |

---

## CPU vs GPU

| Modo              | Cómo                     | Velocidad (~3B params) |
|------------------|--------------------------|------------------------|
| CPU (C binario)  | `./probnet ...`          | ~2–5 tok/s             |
| CPU multi-core   | OpenMP automático        | ×2 con 4+ cores        |
| CPU Python       | `probnet_complete.py`    | ~1–3 tok/s             |
| GPU CUDA         | Automático si disponible | ~50–100 tok/s          |
| GPU Apple MPS    | Automático en Mac        | ~20–40 tok/s           |
| GPU 8-bit        | `--quantize`             | Mitad de VRAM          |

---

## Cómo funciona la conversión

```
FFN SwiGLU original:          Reemplazo APN:

x → gate_proj → gate          x → W1 → p1 ─┐
x → up_proj  → up    →        x → W2 → p2 ─┤→ APN(p1,p2) → W_out → salida
gate * silu(up) → down_proj

W1   ← gate_proj.weight    (sin pérdida de pesos)
W2   ← up_proj.weight      (sin pérdida de pesos)
W_out← down_proj.weight    (sin pérdida de pesos)
logits APN init: identity+relu dominantes ≈ comportamiento SwiGLU inicial
```

Tras la conversión el modelo genera igual que el original.
Tras fine-tuning las neuronas se especializan más allá de SwiGLU.

---

## Diferencia entre v1 y APN v9

| Característica          | ProbNet v1              | APN v9                     |
|------------------------|-------------------------|---------------------------|
| Matrices de pesos      | Ninguna                 | Sí (W1, W2, W_out)        |
| Función               | Ratios deterministas    | 6 funciones con backprop  |
| Entrenamiento         | Solo embeddings + norms | Backprop completo          |
| Beneficio GPU         | Mínimo                  | Completo                   |
| Conversión HF         | Reemplaza nn.Linear     | Reemplaza FFN (SwiGLU)    |
| Mejor para            | OEIS, secuencias        | LLMs generales             |

---

## Compilar con opciones avanzadas

```bash
# Con AVX-512 completo (Intel Ice Lake+, AMD Zen4+)
gcc -O3 -march=native -mavx512f -mavx512dq -mavx512bw \
    -mfma -ffast-math -funroll-loops -fopenmp \
    -o probnet probnet_main.c -lm

# Genérico x86-64 (sin AVX-512)
gcc -O3 -march=native -mfma -ffast-math -fopenmp \
    -o probnet probnet_main.c -lm

# Debug con sanitizers
gcc -g -O0 -fsanitize=address,undefined -fopenmp \
    -o probnet probnet_main.c -lm

# scale_test (benchmark standalone)
gcc -O3 -march=native -mavx512f -mfma -fopenmp -ffast-math \
    -o scale_test tools/scale_test.c -lm
```
