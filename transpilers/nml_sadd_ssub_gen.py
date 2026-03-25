#!/usr/bin/env python3
"""
NML Stage-3 targeted generator — SADD and SSUB only.

Generates ~500 high-quality prompt/completion pairs covering:
  - SADD Rd Rs #imm  (scalar add: every element + imm)
  - SSUB Rd Rs #imm  (scalar subtract: every element - imm)
  - Mixed SADD/SSUB with other ops (RELU, SCLR, MMUL, etc.)
  - All three syntax forms: classic, symbolic (∔ ∸), verbose

Output: domain/output/training/nml_sadd_ssub_pairs.jsonl
"""

import json
import random
from pathlib import Path

random.seed(99)
OUTPUT = Path(__file__).parent.parent / "domain" / "output" / "training" / "nml_sadd_ssub_pairs.jsonl"

INPUT_NAMES  = ["input", "x", "features", "data", "hidden", "activations", "logits", "embeddings", "query", "values"]
OUTPUT_NAMES = ["output", "result", "y", "out", "scores", "predictions", "response"]


def rand_input():  return random.choice(INPUT_NAMES)
def rand_output(): return random.choice(OUTPUT_NAMES)
def rand_v(lo=0.1, hi=5.0): return round(random.uniform(lo, hi), 2)
def pair(prompt, code):
    return {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": code}]}


# ── 1. Pure SADD ─────────────────────────────────────────────────────────────
def gen_sadd():
    pairs = []
    # Classic syntax
    for _ in range(80):
        inp, out, v = rand_input(), rand_output(), rand_v()
        pairs.append(pair(
            f"Write NML to add scalar {v} to every element of {inp} and store in {out}.",
            f"LD R0 @{inp}\nSADD R1 R0 #{v}\nST R1 @{out}\nHALT"))
    # Asks using "SADD" by name
    for _ in range(60):
        inp, out, v = rand_input(), rand_output(), rand_v()
        pairs.append(pair(
            f"Write NML to add a scalar {v} to every element using SADD.",
            f"LD R0 @{inp}\nSADD R1 R0 #{v}\nST R1 @{out}\nHALT"))
    # Bias addition (common ML use case)
    for _ in range(60):
        inp, out, v = rand_input(), rand_output(), rand_v(0.01, 1.0)
        pairs.append(pair(
            f"Add bias {v} to every activation in {inp}.",
            f"LD R0 @{inp}\nSADD R1 R0 #{v}\nST R1 @{out}\nHALT"))
    # Symbolic form ∔
    for _ in range(40):
        inp, out, v = rand_input(), rand_output(), rand_v()
        pairs.append(pair(
            f"Write NML in symbolic syntax to add scalar {v} to {inp}.",
            f"↓ R0 @{inp}\n∔ R1 R0 #{v}\n↑ R1 @{out}\n◼"))
    # Verbose form
    for _ in range(30):
        inp, out, v = rand_input(), rand_output(), rand_v()
        pairs.append(pair(
            f"Write NML in verbose syntax to add scalar {v} to {inp}.",
            f"LOAD R0 @{inp}\nSCALAR_ADD R1 R0 #{v}\nSTORE R1 @{out}\nHALT"))
    return pairs


# ── 2. Pure SSUB ─────────────────────────────────────────────────────────────
def gen_ssub():
    pairs = []
    # Classic syntax
    for _ in range(80):
        inp, out, v = rand_input(), rand_output(), rand_v()
        pairs.append(pair(
            f"Write NML to subtract scalar {v} from every element of {inp}.",
            f"LD R0 @{inp}\nSSUB R1 R0 #{v}\nST R1 @{out}\nHALT"))
    # Asks using "SSUB" by name
    for _ in range(60):
        inp, out, v = rand_input(), rand_output(), rand_v()
        pairs.append(pair(
            f"Write NML to subtract scalar {v} from a tensor using SSUB.",
            f"LD R0 @{inp}\nSSUB R1 R0 #{v}\nST R1 @{out}\nHALT"))
    # Mean centering (common ML use case)
    for _ in range(60):
        inp, out, v = rand_input(), rand_output(), rand_v(0.0, 3.0)
        pairs.append(pair(
            f"Center {inp} by subtracting mean {v} from every element.",
            f"LD R0 @{inp}\nSSUB R1 R0 #{v}\nST R1 @{out}\nHALT"))
    # Symbolic form ∸
    for _ in range(40):
        inp, out, v = rand_input(), rand_output(), rand_v()
        pairs.append(pair(
            f"Write NML in symbolic syntax to subtract scalar {v} from {inp}.",
            f"↓ R0 @{inp}\n∸ R1 R0 #{v}\n↑ R1 @{out}\n◼"))
    # Verbose form
    for _ in range(30):
        inp, out, v = rand_input(), rand_output(), rand_v()
        pairs.append(pair(
            f"Write NML in verbose syntax to subtract scalar {v} from {inp}.",
            f"LOAD R0 @{inp}\nSCALAR_SUB R1 R0 #{v}\nSTORE R1 @{out}\nHALT"))
    return pairs


# ── 3. SADD/SSUB combined with other ops ─────────────────────────────────────
def gen_combined():
    pairs = []
    # SADD then RELU (bias + activation)
    for _ in range(50):
        inp, out, v = rand_input(), rand_output(), rand_v(0.01, 1.0)
        pairs.append(pair(
            f"Add bias {v} to {inp} then apply ReLU.",
            f"LD R0 @{inp}\nSADD R1 R0 #{v}\nRELU R2 R1\nST R2 @{out}\nHALT"))
    # SSUB then SCLR (normalize: subtract mean, scale by std)
    for _ in range(50):
        inp, out = rand_input(), rand_output()
        mean = rand_v(0.0, 2.0)
        std  = rand_v(0.1, 2.0)
        pairs.append(pair(
            f"Normalize {inp} by subtracting mean {mean} and scaling by {std}.",
            f"LD R0 @{inp}\nSSUB R1 R0 #{mean}\nSCLR R2 R1 #{std}\nST R2 @{out}\nHALT"))
    # MMUL then SADD (linear layer: Wx + b)
    for _ in range(50):
        inp, out, v = rand_input(), rand_output(), rand_v(0.01, 0.5)
        pairs.append(pair(
            f"Multiply {inp} by weights, then add bias {v} to every output element.",
            f"LD R0 @{inp}\nLD R1 @weights\nMMUL R2 R0 R1\nSADD R3 R2 #{v}\nST R3 @{out}\nHALT"))
    # SADD then SSUB (shift up then down)
    for _ in range(30):
        inp, out, v1, v2 = rand_input(), rand_output(), rand_v(), rand_v()
        pairs.append(pair(
            f"Add {v1} to every element of {inp}, then subtract {v2}.",
            f"LD R0 @{inp}\nSADD R1 R0 #{v1}\nSSUB R2 R1 #{v2}\nST R2 @{out}\nHALT"))
    return pairs


def main():
    categories = [
        ("SADD",     gen_sadd),
        ("SSUB",     gen_ssub),
        ("COMBINED", gen_combined),
    ]

    all_pairs = []
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    print("NML Stage-3 SADD/SSUB Generator")
    print("-" * 40)
    for name, fn in categories:
        p = fn()
        random.shuffle(p)
        all_pairs.extend(p)
        print(f"  {name:<10} {len(p):>4} pairs")

    random.shuffle(all_pairs)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print("-" * 40)
    print(f"  TOTAL      {len(all_pairs):>4} pairs")
    print(f"\nWritten to: {OUTPUT}")


if __name__ == "__main__":
    main()
