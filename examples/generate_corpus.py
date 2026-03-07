#!/usr/bin/env python3
"""
generate_corpus.py - Generate example training corpus for ProbNet

Usage:
    python3 examples/generate_corpus.py --out corpus.txt --size medium
    python3 examples/generate_corpus.py --out math_corpus.txt --type math
"""
import random, math, argparse

def gen_math(n=5000):
    lines = []
    for _ in range(n):
        a, b = random.randint(1,99), random.randint(1,99)
        op = random.choice(['+','-','*'])
        if op == '+': lines.append(f"{a} + {b} = {a+b}")
        elif op == '-': lines.append(f"{a} - {b} = {a-b}")
        elif op == '*': lines.append(f"{a} * {b} = {a*b}")
    for i in range(1, 150):
        lines.append(f"The square of {i} is {i*i}")
        lines.append(f"The square root of {i*i} is {i}")
    for i in range(2, 20):
        for j in range(1, 13):
            lines.append(f"{i} times {j} equals {i*j}")
    return lines

def gen_text(n=5000):
    templates = [
        "The {adj} {noun} {verb} over the {noun2}.",
        "A {adj} {noun} can {verb} very {adv}.",
        "Scientists discovered that {noun} can {verb} {noun2}.",
        "The history of {noun} began in {year}.",
        "To {verb} a {noun}, you need {noun2} and patience.",
        "The most important {noun} is {adj} {noun2}.",
        "{name} said that {noun} is {adj}.",
        "In {year}, the world changed because of {noun}.",
    ]
    nouns   = "science music art history technology nature water fire earth air \
               intelligence knowledge language mathematics physics chemistry".split()
    verbs   = "transform explain discover create improve advance shape influence \
               generate understand support develop".split()
    adjs    = "important beautiful complex simple powerful useful ancient modern \
               fundamental extraordinary remarkable significant".split()
    adv     = "quickly slowly carefully precisely remarkably deeply".split()
    names   = "Einstein Darwin Newton Curie Turing Feynman Hawking".split()
    years   = [str(y) for y in range(1800, 2024, 10)]
    lines   = []
    for _ in range(n):
        t = random.choice(templates)
        line = t.replace("{noun}",  random.choice(nouns))
        line = line.replace("{noun2}", random.choice(nouns))
        line = line.replace("{verb}",  random.choice(verbs))
        line = line.replace("{adj}",   random.choice(adjs))
        line = line.replace("{adv}",   random.choice(adv))
        line = line.replace("{name}",  random.choice(names))
        line = line.replace("{year}",  random.choice(years))
        lines.append(line)
    return lines

def gen_code(n=2000):
    lines = []
    fns = ["add","multiply","divide","square","sqrt","sigmoid","relu","softmax"]
    for fn in fns:
        lines.append(f"def {fn}(x):")
        if fn == "add":      lines.append(f"    return x + 1")
        elif fn == "multiply":lines.append(f"    return x * 2")
        elif fn == "square":  lines.append(f"    return x * x")
        elif fn == "sqrt":    lines.append(f"    return x ** 0.5")
        elif fn == "sigmoid": lines.append(f"    return 1 / (1 + exp(-x))")
        elif fn == "relu":    lines.append(f"    return max(0, x)")
        else:                 lines.append(f"    return x")
        lines.append("")
    for i in range(n):
        var = random.choice("xyzabcnmkj")
        val = random.randint(-10, 100)
        lines.append(f"{var} = {val}")
        op  = random.choice(["+", "-", "*", "//"])
        val2= random.randint(1, 20)
        lines.append(f"{var} = {var} {op} {val2}  # result: {eval(f'{val} {op} {val2}')}")
    return lines

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",  default="corpus.txt")
    ap.add_argument("--size", default="medium", choices=["small","medium","large"])
    ap.add_argument("--type", default="mixed",  choices=["math","text","code","mixed"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    sizes = {"small": 1000, "medium": 5000, "large": 20000}
    n = sizes[args.size]

    lines = []
    if args.type == "math":
        lines = gen_math(n)
    elif args.type == "text":
        lines = gen_text(n)
    elif args.type == "code":
        lines = gen_code(n)
    else:  # mixed
        lines = gen_math(n//3) + gen_text(n//3) + gen_code(n//3)

    random.shuffle(lines)
    text = "\n".join(lines)

    with open(args.out, "w") as f:
        f.write(text)

    print(f"Generated: {args.out}")
    print(f"  Lines  : {len(lines):,}")
    print(f"  Chars  : {len(text):,}")
    print(f"  Type   : {args.type}  |  Size: {args.size}")
    print(f"\nUse with:")
    print(f"  ./probnet train --data {args.out} --out model.pnet --d_model 256 --n_layers 4")
    print(f"  python3 probnet_complete.py train --model google/gemma-3-2b-it --data {args.out}")

if __name__ == "__main__":
    main()
