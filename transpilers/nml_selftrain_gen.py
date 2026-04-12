#!/usr/bin/env python3
"""
Generate ~15K self-training pairs using TRAIN+INFER/BKWD/WUPD/LOSS with diverse topologies.

Output: domain/output/training/nml_selftrain_pairs.jsonl
"""

import json
import random
import argparse
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent.parent / "domain" / "output" / "training" / "nml_selftrain_pairs.jsonl"

random.seed(2026_02)

from nml_core_training_gen import (
    _fmt, _pair, _fval, apply_syntax, pick_syntax, syntax_tag,
    to_symbolic, to_verbose,
)

TOPOLOGIES = [
    ("2-4-1", "XOR gate"), ("2-8-1", "logic gate"), ("1-16-1", "regression"),
    ("1-64-1", "function approximation"), ("3-8-1", "multi-input classifier"),
    ("4-16-1", "feature predictor"), ("2-4-2-1", "deep logic gate"),
    ("1-32-16-1", "deep regression"), ("4-8-4-1", "multi-layer classifier"),
]

ACTIVATIONS = ["RELU", "SIGM", "TANH", "GELU"]
LOSS_MODES = [("#0", "MSE"), ("#1", "MAE"), ("#2", "cross-entropy")]


def gen_tnet_programs(count=5000):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        topo_name, desc = random.choice(TOPOLOGIES)
        epochs = random.choice([50, 100, 200, 500, 1000, 2000, 5000])
        lr = random.choice([0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5])
        optimizer = random.choice([0, 1])  # 0=SGD, 1=Adam
        opt_name = "Adam" if optimizer else "SGD"

        prompts = [
            f"Write NML to train a {topo_name} {desc} using TRAIN+INFER for {epochs} epochs at lr {lr}",
            f"Self-train a {topo_name} network with TRAIN+INFER, {epochs} epochs, learning rate {lr}",
            f"Write NML TRAIN+INFER program for {desc} ({topo_name} topology)",
            f"Train a neural network ({topo_name}) using TRAIN+INFER with {opt_name} optimizer",
        ]
        q = random.choice(prompts) + syntax_tag(syntax)

        lines = [
            _fmt("LD", "R0", "@training_inputs"),
            _fmt("LD", "R9", "@training_targets"),
            _fmt("LD", "R1", "@w1"), _fmt("LD", "R2", "@b1"),
            _fmt("LD", "R3", "@w2"), _fmt("LD", "R4", "@b2"),
            _fmt("LD", "RU", "@train_config"),
            f"; @train_config shape=6 data={epochs},{lr},{optimizer},0,0,0.0001",
            _fmt("TRAIN", "RU"),
            _fmt("INFER", "R8", "R0"),
            _fmt("ST", "R8", "@predictions"),
            _fmt("ST", "R1", "@trained_w1"), _fmt("ST", "R2", "@trained_b1"),
            _fmt("ST", "R3", "@trained_w2"), _fmt("ST", "R4", "@trained_b2"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs


def gen_manual_training(count=5000):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        act = random.choice(ACTIVATIONS)
        act_name = {"RELU": "ReLU", "SIGM": "sigmoid", "TANH": "tanh", "GELU": "GELU"}[act]
        loss_code, loss_name = random.choice(LOSS_MODES)
        epochs = random.choice([10, 50, 100, 200, 500])
        lr = random.choice([0.001, 0.005, 0.01, 0.05, 0.1])

        prompts = [
            f"Write NML manual training loop with {act_name} and {loss_name} loss for {epochs} epochs",
            f"Train a network manually using BKWD+WUPD, {act_name} activation, {loss_name} loss",
            f"Write NML gradient descent loop: forward pass, {loss_name} loss, BKWD, WUPD, {epochs} epochs",
            f"Manual training in NML: {act_name} hidden layer, lr={lr}, {epochs} iterations",
        ]
        q = random.choice(prompts) + syntax_tag(syntax)

        lines = [
            _fmt("LD", "R0", "@training_inputs"),
            _fmt("LD", "R9", "@training_targets"),
            _fmt("LD", "R1", "@w1"), _fmt("LD", "R2", "@b1"),
            _fmt("LD", "R3", "@w2"), _fmt("LD", "R4", "@b2"),
            _fmt("LEAF", "RJ", f"#{lr}"),
            _fmt("LEAF", "RD", f"#{epochs}"),
            _fmt("LOOP", "RD"),
            _fmt("MMUL", "R5", "R0", "R1"),
            _fmt("MADD", "R5", "R5", "R2"),
            _fmt(act, "R5", "R5"),
            _fmt("MMUL", "R6", "R5", "R3"),
            _fmt("MADD", "R6", "R6", "R4"),
            _fmt("SIGM", "R6", "R6"),
            _fmt("LOSS", "RG", "R6", "R9", loss_code),
            _fmt("BKWD", "RH", "R6", "R9"),
            _fmt("WUPD", "R3", "R3", "RH"),
            _fmt("BKWD", "RI", "R5", "R9"),
            _fmt("WUPD", "R1", "R1", "RI"),
            "ENDP",
            _fmt("ST", "R1", "@trained_w1"),
            _fmt("ST", "R3", "@trained_w2"),
            _fmt("ST", "RG", "@final_loss"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs


def gen_loss_patterns(count=2500):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        loss_code, loss_name = random.choice(LOSS_MODES)
        q = f"Write NML to compute {loss_name} loss and check if below threshold" + syntax_tag(syntax)

        thresh = _fval(0.001, 0.1)
        lines = [
            _fmt("LD", "R0", "@predictions"),
            _fmt("LD", "R9", "@targets"),
            _fmt("LOSS", "RG", "R0", "R9", loss_code),
            _fmt("CMPI", "RE", "RG", f"#{thresh}"),
            _fmt("JMPF", "#3"),
            _fmt("ST", "RG", "@converged_loss"),
            _fmt("JUMP", "#2"),
            _fmt("ST", "RG", "@loss_above_threshold"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs


def gen_train_infer(count=2500):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        topo_name, desc = random.choice(TOPOLOGIES)
        epochs = random.choice([500, 1000, 2000])
        lr = random.choice([0.01, 0.05, 0.1])

        q = f"Write NML to train a {topo_name} network then run inference on new input" + syntax_tag(syntax)
        lines = [
            _fmt("LD", "R0", "@training_inputs"),
            _fmt("LD", "R9", "@training_targets"),
            _fmt("LD", "R1", "@w1"), _fmt("LD", "R2", "@b1"),
            _fmt("LD", "R3", "@w2"), _fmt("LD", "R4", "@b2"),
            _fmt("ALLC", "RU", f"#[6]", f"{epochs},{lr},0,0,0,0"),
            _fmt("TRAIN", "RU"),
            _fmt("LD", "R0", "@new_input"),
            _fmt("INFER", "R8", "R0"),
            _fmt("ST", "R8", "@prediction"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Generate NML self-training pairs")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    args = parser.parse_args()

    print("Generating NML self-training pairs...")
    print(f"{'─' * 60}")

    p1 = gen_tnet_programs(5000)
    print(f"  TRAIN+INFER programs:   {len(p1):>6}")
    p2 = gen_manual_training(5000)
    print(f"  Manual training loops:  {len(p2):>6}")
    p3 = gen_loss_patterns(2500)
    print(f"  Loss patterns:          {len(p3):>6}")
    p4 = gen_train_infer(2500)
    print(f"  Train + infer:          {len(p4):>6}")

    all_pairs = p1 + p2 + p3 + p4
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
