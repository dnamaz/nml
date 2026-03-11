#!/usr/bin/env python3
"""
Generate ~10K cascade ↔ tensor conversion pairs.

Teaches the model that cascade CMPF/JMPT/LEAF programs and TNET-based
tensor network programs compute the same function. Covers progressive,
flat-rate, fixed, and blended bracket structures.

Output: domain/output/training/nml_cascade_conv_pairs.jsonl
"""

import json
import random
import argparse
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent.parent / "domain" / "output" / "training" / "nml_cascade_conv_pairs.jsonl"

random.seed(2026_04)

from nml_core_training_gen import (
    _fmt, _pair, _fval, apply_syntax, pick_syntax, syntax_tag,
)


def _rand_brackets(n=3):
    thresholds = sorted(random.sample(range(5000, 500000, 1000), n))
    rates = [round(random.uniform(0.01, 0.15), 4) for _ in range(n + 1)]
    return list(zip(thresholds, rates[:-1])), rates[-1]


def gen_cascade_to_tensor(count=3500):
    """Convert cascade CMPF/JMPT/LEAF to equivalent TNET."""
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        n_brackets = random.choice([2, 3, 4, 5])
        brackets, top_rate = _rand_brackets(n_brackets)

        cascade_lines = [_fmt("LD", "R0", "@input")]
        for i, (thresh, rate) in enumerate(brackets):
            cascade_lines.append(_fmt("CMPI", "RE", "R0", f"#{thresh}.0"))
            skip = 3
            cascade_lines.append(_fmt("JMPF", f"#{skip}"))
            cascade_lines.append(_fmt("SCLR", "R1", "R0", f"#{rate}"))
            remaining = (len(brackets) - i - 1) * 4 + 2
            cascade_lines.append(_fmt("JUMP", f"#{remaining}"))
        cascade_lines.append(_fmt("SCLR", "R1", "R0", f"#{top_rate}"))
        cascade_lines.append(_fmt("ST", "R1", "@result"))
        cascade_lines.append("HALT")

        cascade_code = "\n".join(apply_syntax(cascade_lines, syntax))

        tensor_lines = [
            _fmt("LD", "R0", "@input"),
            _fmt("LD", "R9", "@targets"),
            _fmt("LD", "R1", "@w1"), _fmt("LD", "R2", "@b1"),
            _fmt("LD", "R3", "@w2"), _fmt("LD", "R4", "@b2"),
            _fmt("TNET", "#2000", "#0.01"),
            _fmt("LD", "R0", "@input"),
            _fmt("MMUL", "R5", "R0", "R1"),
            _fmt("MADD", "R5", "R5", "R2"),
            _fmt("RELU", "R5", "R5"),
            _fmt("MMUL", "R6", "R5", "R3"),
            _fmt("MADD", "R6", "R6", "R4"),
            _fmt("ST", "R6", "@result"),
            "HALT",
        ]
        tensor_code = "\n".join(apply_syntax(tensor_lines, syntax))

        q = f"Convert this cascade program to a TNET-based tensor network:\n{cascade_code}" + syntax_tag(syntax)
        pairs.append(_pair(q, tensor_code))
    return pairs


def gen_tensor_to_cascade(count=3500):
    """Convert TNET-based to equivalent cascade."""
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        n_brackets = random.choice([2, 3, 4])
        brackets, top_rate = _rand_brackets(n_brackets)

        tensor_lines = [
            _fmt("LD", "R0", "@input"),
            _fmt("LD", "R1", "@w1"), _fmt("LD", "R2", "@b1"),
            _fmt("LD", "R3", "@w2"), _fmt("LD", "R4", "@b2"),
            _fmt("MMUL", "R5", "R0", "R1"),
            _fmt("MADD", "R5", "R5", "R2"),
            _fmt("RELU", "R5", "R5"),
            _fmt("MMUL", "R6", "R5", "R3"),
            _fmt("MADD", "R6", "R6", "R4"),
            _fmt("ST", "R6", "@result"),
            "HALT",
        ]
        tensor_code = "\n".join(apply_syntax(tensor_lines, syntax))

        cascade_lines = [_fmt("LD", "R0", "@input")]
        for i, (thresh, rate) in enumerate(brackets):
            cascade_lines.append(_fmt("CMPI", "RE", "R0", f"#{thresh}.0"))
            cascade_lines.append(_fmt("JMPF", "#3"))
            cascade_lines.append(_fmt("SCLR", "R1", "R0", f"#{rate}"))
            remaining = (len(brackets) - i - 1) * 4 + 2
            cascade_lines.append(_fmt("JUMP", f"#{remaining}"))
        cascade_lines.append(_fmt("SCLR", "R1", "R0", f"#{top_rate}"))
        cascade_lines.append(_fmt("ST", "R1", "@result"))
        cascade_lines.append("HALT")
        cascade_code = "\n".join(apply_syntax(cascade_lines, syntax))

        q = f"Convert this neural network program to an equivalent cascade program:\n{tensor_code}" + syntax_tag(syntax)
        pairs.append(_pair(q, cascade_code))
    return pairs


def gen_equivalence_explanation(count=3000):
    """Explain that cascade and tensor approaches compute the same function."""
    pairs = []
    structures = [
        ("progressive bracket", "graduated rates based on thresholds"),
        ("flat rate", "single rate applied to entire amount"),
        ("blended rate", "weighted combination of rates"),
        ("tiered bracket", "different rates for different ranges"),
    ]
    for _ in range(count):
        struct_name, struct_desc = random.choice(structures)
        n = random.choice([2, 3, 4, 5])

        prompts = [
            f"Explain how a {n}-bracket {struct_name} can be expressed as both cascade NML and tensor NML",
            f"How would you convert a {struct_name} ({struct_desc}) between cascade and TNET form?",
            f"What is the relationship between CMPF/JMPT cascade programs and TNET tensor programs for {struct_name}?",
        ]
        q = random.choice(prompts)

        a = (
            f"A {n}-bracket {struct_name} ({struct_desc}) can be expressed two ways in NML:\n\n"
            f"1. **Cascade form**: Uses {n} CMPI/JMPF/SCLR chains. Each bracket tests the input against "
            f"a threshold and applies the corresponding rate. Deterministic, auditable, exact.\n\n"
            f"2. **Tensor form**: Uses TNET to train a neural network that approximates the same "
            f"function. The network learns the thresholds and rates from training data. Compact, "
            f"adaptable, but approximate.\n\n"
            f"Both produce the same output for the same input (within floating-point tolerance "
            f"for the tensor form). The cascade form is preferred when auditability matters; "
            f"the tensor form is preferred when the bracket structure may change or isn't known in advance."
        )
        pairs.append(_pair(q, a))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Generate cascade ↔ tensor conversion pairs")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    args = parser.parse_args()

    print("Generating cascade ↔ tensor conversion pairs...")
    print(f"{'─' * 60}")

    p1 = gen_cascade_to_tensor(3500)
    print(f"  Cascade → tensor:       {len(p1):>6}")
    p2 = gen_tensor_to_cascade(3500)
    print(f"  Tensor → cascade:       {len(p2):>6}")
    p3 = gen_equivalence_explanation(3000)
    print(f"  Equivalence explain:    {len(p3):>6}")

    all_pairs = p1 + p2 + p3
    random.shuffle(all_pairs)

    print(f"{'─' * 60}")
    print(f"  TOTAL:                  {len(all_pairs):>6}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\nWritten to: {out_path}")


if __name__ == "__main__":
    main()
