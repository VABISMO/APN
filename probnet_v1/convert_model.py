"""
convert_model.py — Convert any HuggingFace model to ProbNet (v1)
=================================================================
Replaces all nn.Linear layers with ProbNetLayer.
Keeps embeddings, LayerNorm, and attention structure intact.

Examples
--------
  # Convert GPT-2 (no auth needed)
  python convert_model.py --model gpt2 --generate "The Fibonacci sequence"

  # Convert Gemma 2B (needs HF token)
  python convert_model.py --model google/gemma-2b --output gemma_probnet.pt

  # Convert and save, then generate
  python convert_model.py --model gpt2 --output gpt2_probnet.pt \\
                          --generate "Once upon a time" --max_new 100
"""

import argparse
import torch
import torch.nn as nn
from probnet_transformer import convert_to_probnet, generate


def main(args):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise SystemExit("Install transformers: pip install transformers")

    print(f"\nLoading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, trust_remote_code=True
    )

    before   = sum(p.numel() for p in model.parameters())
    n_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    print(f"Parameters before: {before:,}")
    print(f"nn.Linear layers:  {n_linear}")

    print(f"\nConverting to ProbNet (window={args.window})...")
    model = convert_to_probnet(model, window=args.window, verbose=args.verbose)

    after = sum(p.numel() for p in model.parameters())
    print(f"\nParameters after:  {after:,}  (removed {before-after:,} weight params)")

    if args.output:
        torch.save(model.state_dict(), args.output)
        print(f"Saved → {args.output}")

    if args.generate:
        print(f"\nGenerating from: '{args.generate}'")
        print("-" * 50)
        out = generate(
            model, tokenizer, args.generate,
            max_new=args.max_new,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print(out)
        print("-" * 50)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Convert HuggingFace model to ProbNet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    p.add_argument("--model",       default="gpt2",
                   help="HuggingFace model id or local path")
    p.add_argument("--output",      default="",
                   help="Path to save converted model (.pt)")
    p.add_argument("--window",      type=int,   default=8,
                   help="ProbNet window size per neuron")
    p.add_argument("--generate",    default="",
                   help="Prompt for text generation after conversion")
    p.add_argument("--max_new",     type=int,   default=100)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k",       type=int,   default=50)
    p.add_argument("--verbose",     action="store_true", default=False,
                   help="Print each replaced layer")
    args = p.parse_args()
    main(args)
