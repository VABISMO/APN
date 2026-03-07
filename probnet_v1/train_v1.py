"""
train_v1.py — Train a ProbNetTransformerLM from scratch (v1)
=============================================================
Because ProbNetLayer is deterministic (no weight matrices), the only
trainable parameters are:
  - Token + positional embeddings
  - LayerNorm scales/biases
  - Optional ProbNetLayer bias terms

This dramatically reduces training cost while keeping sequence modeling.

Quick start
-----------
  python train_v1.py                           # demo (Fibonacci + primes)
  python train_v1.py --dataset my_corpus.txt  # your text file
  python train_v1.py --n_layer 8 --n_embd 512 --window 16
"""

import argparse
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from probnet_transformer import ProbNetTransformerConfig, ProbNetTransformerLM


# ---------------------------------------------------------------------------
# Character-level dataset
# ---------------------------------------------------------------------------

class CharDataset(Dataset):
    def __init__(self, text: str, block_size: int = 256):
        chars           = sorted(set(text))
        self.stoi       = {c: i for i, c in enumerate(chars)}
        self.itos       = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars)
        self.block_size = block_size
        self.data       = [self.stoi[c] for c in text]

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx: idx + self.block_size + 1]
        return (torch.tensor(chunk[:-1], dtype=torch.long),
                torch.tensor(chunk[1:],  dtype=torch.long))

    def encode(self, text):
        return torch.tensor([self.stoi.get(c, 0) for c in text],
                             dtype=torch.long).unsqueeze(0)

    def decode(self, ids):
        return "".join(self.itos.get(i, "?") for i in ids)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Dataset
    if args.dataset == "demo":
        text = (
            "The Fibonacci sequence: 1 1 2 3 5 8 13 21 34 55 89 144 233 377 610 987 "
            "The prime numbers: 2 3 5 7 11 13 17 19 23 29 31 37 41 43 47 53 59 61 67 "
            "The square numbers: 1 4 9 16 25 36 49 64 81 100 121 144 169 196 225 "
        ) * 600
    else:
        with open(args.dataset, "r", encoding="utf-8") as f:
            text = f.read()

    dataset = CharDataset(text, block_size=args.block_size)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=True, num_workers=0, drop_last=True)
    print(f"Dataset: {len(dataset):,} samples  |  vocab: {dataset.vocab_size}")

    # Model
    cfg = ProbNetTransformerConfig(
        vocab_size  = dataset.vocab_size,
        n_layer     = args.n_layer,
        n_head      = args.n_head,
        n_embd      = args.n_embd,
        block_size  = args.block_size,
        dropout     = args.dropout,
        window      = args.window,
    )
    model     = ProbNetTransformerLM(cfg).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable tensors: {len(trainable)}  |  "
          f"params: {sum(p.numel() for p in trainable):,}")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(loader)
    )

    print(f"\n{'Epoch':>6}  {'Step':>6}  {'Loss':>8}  {'PPL':>8}")
    print(f"{'─'*36}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            if step % 50 == 0:
                avg = total_loss / (step + 1)
                ppl = math.exp(min(avg, 20))
                print(f"{epoch:>6}  {step:>6}  {avg:>8.4f}  {ppl:>8.2f}")

        # Sample after epoch
        model.eval()
        prompt = "The Fibonacci"
        ids    = dataset.encode(prompt).to(device)
        with torch.no_grad():
            for _ in range(80):
                logits, _ = model(ids[:, -cfg.block_size:])
                next_id   = logits[:, -1, :].argmax(-1, keepdim=True)
                ids       = torch.cat([ids, next_id], dim=1)
        print(f"\n  [Sample] {dataset.decode(ids[0].tolist())}\n")

    torch.save({"model_state": model.state_dict(), "config": cfg}, args.output)
    print(f"Saved → {args.output}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Train ProbNetTransformerLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    p.add_argument("--dataset",    default="demo",
                   help="'demo' or path to .txt file")
    p.add_argument("--n_layer",    type=int,   default=4)
    p.add_argument("--n_head",     type=int,   default=4)
    p.add_argument("--n_embd",     type=int,   default=256)
    p.add_argument("--block_size", type=int,   default=128)
    p.add_argument("--batch_size", type=int,   default=16)
    p.add_argument("--epochs",     type=int,   default=5)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--dropout",    type=float, default=0.0)
    p.add_argument("--window",     type=int,   default=8,
                   help="ProbNet window size per output neuron")
    p.add_argument("--output",     default="probnet_v1_model.pt")
    args = p.parse_args()
    train(args)
