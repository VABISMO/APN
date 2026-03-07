# ProbNet v1 — Deterministic Transformer

This is the **original ProbNet** implementation (session 1).

Every `nn.Linear` is replaced by a `ProbNetLayer` — deterministic,
ratio-based prediction, **no weight matrices**, no matrix multiplication.

## Files

```
probnet_layer.py       Core: ProbNetLayer + ProbNet math
probnet_transformer.py Full transformer LM + convert_to_probnet() + generate()
convert_model.py       Convert Gemma / GPT-2 / any HF model
train_v1.py            Train from scratch
```

## Quick Start

### Sequence prediction (no model needed)
```python
from probnet_transformer import predict_sequence

print(predict_sequence([1,1,2,3,5,8,13,21], n_next=4))   # Fibonacci
# → [34, 55, 89, 144]

print(predict_sequence([2,4,8,16,32], n_next=3))           # Powers of 2
# → [64, 128, 256]
```

### Convert GPT-2 → ProbNet
```bash
python convert_model.py --model gpt2 --generate "The Fibonacci sequence"
```

### Train from scratch
```bash
python train_v1.py --dataset demo --n_layer 4 --n_embd 256 --epochs 5
```

## How it works

For each output neuron `j`:
1. Select `window` inputs starting at `(j × stride) % in_features`
2. Compute ratios between consecutive elements
3. Find the element nearest to the last value
4. Apply its ratio → prediction
5. Add error-correction term → final output

No matrix multiply. No learned weights. Fully deterministic.

## Difference from APN v9

| | ProbNet v1 | APN v9 |
|---|---|---|
| Weight matrices | None | Yes (learned) |
| Function | Fixed (ratio/predict) | 6 learnable functions |
| Training | Embeddings + norms only | Full backprop |
| GPU benefit | Minimal | Full |
| Use case | OEIS, sequences | General LLM |
