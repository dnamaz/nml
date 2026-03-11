#!/usr/bin/env python3
"""
Generate ~10K tensor table pairs: NML programs paired with .nml.data file contents.

Teaches the model to produce both the .nml program AND the .nml.data file.

Output: domain/output/training/nml_tensor_table_pairs.jsonl
"""

import json
import random
import argparse
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent.parent / "domain" / "output" / "training" / "nml_tensor_table_pairs.jsonl"

random.seed(2026_03)

from nml_core_training_gen import (
    _fmt, _pair, _fval, apply_syntax, pick_syntax, syntax_tag,
)


def _rand_data(n, lo=-2.0, hi=2.0, decimals=4):
    return ",".join(str(round(random.uniform(lo, hi), decimals)) for _ in range(n))


def gen_nn_with_data(count=4000):
    """Neural network programs with matching weight/bias data files."""
    pairs = []
    configs = [
        {"input": 2, "hidden": 4, "output": 1, "name": "2-4-1"},
        {"input": 2, "hidden": 8, "output": 1, "name": "2-8-1"},
        {"input": 4, "hidden": 8, "output": 1, "name": "4-8-1"},
        {"input": 1, "hidden": 16, "output": 1, "name": "1-16-1"},
        {"input": 3, "hidden": 4, "output": 2, "name": "3-4-2"},
        {"input": 4, "hidden": 16, "output": 4, "name": "4-16-4"},
    ]
    acts = ["RELU", "SIGM", "TANH", "GELU"]

    for _ in range(count):
        syntax = pick_syntax()
        cfg = random.choice(configs)
        act = random.choice(acts)
        act_name = {"RELU": "ReLU", "SIGM": "sigmoid", "TANH": "tanh", "GELU": "GELU"}[act]
        inp_size = cfg["input"]; hid_size = cfg["hidden"]; out_size = cfg["output"]

        w1_shape = f"{inp_size},{hid_size}"; b1_shape = f"1,{hid_size}"
        w2_shape = f"{hid_size},{out_size}"; b2_shape = f"1,{out_size}"
        inp_shape = f"1,{inp_size}"

        w1_data = _rand_data(inp_size * hid_size)
        b1_data = _rand_data(hid_size, -0.5, 0.5)
        w2_data = _rand_data(hid_size * out_size)
        b2_data = _rand_data(out_size, -0.5, 0.5)
        inp_data = _rand_data(inp_size, 0.0, 1.0)

        q = f"Write an NML program and data file for a {cfg['name']} network with {act_name}" + syntax_tag(syntax)

        program_lines = [
            _fmt("LD", "R0", "@input"),
            _fmt("LD", "R1", "@w1"), _fmt("LD", "R2", "@b1"),
            _fmt("MMUL", "R3", "R0", "R1"),
            _fmt("MADD", "R3", "R3", "R2"),
            _fmt(act, "R3", "R3"),
            _fmt("LD", "R4", "@w2"), _fmt("LD", "R5", "@b2"),
            _fmt("MMUL", "R6", "R3", "R4"),
            _fmt("MADD", "R6", "R6", "R5"),
            _fmt("SIGM", "R6", "R6"),
            _fmt("ST", "R6", "@output"),
            "HALT",
        ]
        program = "\n".join(apply_syntax(program_lines, syntax))

        data_file = "\n".join([
            f"@input shape={inp_shape} data={inp_data}",
            f"@w1 shape={w1_shape} data={w1_data}",
            f"@b1 shape={b1_shape} data={b1_data}",
            f"@w2 shape={w2_shape} data={w2_data}",
            f"@b2 shape={b2_shape} data={b2_data}",
        ])

        answer = f"Program (program.nml):\n{program}\n\nData file (program.nml.data):\n{data_file}"
        pairs.append(_pair(q, answer))
    return pairs


def gen_training_with_data(count=3000):
    """TNET training programs with matching data files."""
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        n_samples = random.choice([4, 8, 16])
        inp_size = random.choice([1, 2, 3, 4])
        hid_size = random.choice([4, 8, 16])
        epochs = random.choice([100, 500, 1000, 2000])
        lr = random.choice([0.01, 0.05, 0.1])

        inp_shape = f"{n_samples},{inp_size}"
        tgt_shape = f"{n_samples},1"
        w1_shape = f"{inp_size},{hid_size}"; b1_shape = f"1,{hid_size}"
        w2_shape = f"{hid_size},1"; b2_shape = "1,1"

        inp_data = _rand_data(n_samples * inp_size, 0.0, 1.0)
        tgt_data = _rand_data(n_samples, 0.0, 1.0)
        w1_data = _rand_data(inp_size * hid_size)
        b1_data = _rand_data(hid_size, -0.1, 0.1)
        w2_data = _rand_data(hid_size)
        b2_data = _rand_data(1, -0.1, 0.1)

        q = f"Write NML TNET training program with data file for {n_samples} samples, {inp_size} features" + syntax_tag(syntax)

        program_lines = [
            _fmt("LD", "R0", "@training_inputs"),
            _fmt("LD", "R9", "@training_targets"),
            _fmt("LD", "R1", "@w1"), _fmt("LD", "R2", "@b1"),
            _fmt("LD", "R3", "@w2"), _fmt("LD", "R4", "@b2"),
            _fmt("TNET", f"#{epochs}", f"#{lr}"),
            _fmt("ST", "RA", "@predictions"),
            _fmt("ST", "R1", "@trained_w1"),
            _fmt("ST", "R3", "@trained_w2"),
            "HALT",
        ]
        program = "\n".join(apply_syntax(program_lines, syntax))

        data_file = "\n".join([
            f"@training_inputs shape={inp_shape} data={inp_data}",
            f"@training_targets shape={tgt_shape} data={tgt_data}",
            f"@w1 shape={w1_shape} data={w1_data}",
            f"@b1 shape={b1_shape} data={b1_data}",
            f"@w2 shape={w2_shape} data={w2_data}",
            f"@b2 shape={b2_shape} data={b2_data}",
        ])

        answer = f"Program (train.nml):\n{program}\n\nData file (train.nml.data):\n{data_file}"
        pairs.append(_pair(q, answer))
    return pairs


def gen_data_file_only(count=3000):
    """Teach the model to produce valid .nml.data files from descriptions."""
    pairs = []
    for _ in range(count):
        n_features = random.choice([1, 2, 3, 4, 8])
        n_outputs = random.choice([1, 2, 4])
        hid = random.choice([4, 8, 16, 32, 64])
        dtype = random.choice(["f32", "f64", "f32", "f32"])  # weighted toward f32

        prompts = [
            f"Write an NML data file for a {n_features}-input, {hid}-hidden, {n_outputs}-output network",
            f"Generate .nml.data for a neural network with {n_features} inputs and {n_outputs} outputs",
            f"Create tensor data file with weights for a {n_features}-{hid}-{n_outputs} topology",
        ]
        q = random.choice(prompts)

        w1_data = _rand_data(n_features * hid)
        b1_data = _rand_data(hid, -0.5, 0.5)
        w2_data = _rand_data(hid * n_outputs)
        b2_data = _rand_data(n_outputs, -0.5, 0.5)
        inp_data = _rand_data(n_features, 0.0, 1.0)

        dtype_str = f" dtype={dtype}" if dtype != "f32" else ""
        lines = [
            f"@input shape=1,{n_features}{dtype_str} data={inp_data}",
            f"@w1 shape={n_features},{hid}{dtype_str} data={w1_data}",
            f"@b1 shape=1,{hid}{dtype_str} data={b1_data}",
            f"@w2 shape={hid},{n_outputs}{dtype_str} data={w2_data}",
            f"@b2 shape=1,{n_outputs}{dtype_str} data={b2_data}",
        ]

        pairs.append(_pair(q, "\n".join(lines)))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Generate NML tensor table pairs")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    args = parser.parse_args()

    print("Generating NML tensor table pairs...")
    print(f"{'─' * 60}")

    p1 = gen_nn_with_data(4000)
    print(f"  NN + data:              {len(p1):>6}")
    p2 = gen_training_with_data(3000)
    print(f"  TNET + data:            {len(p2):>6}")
    p3 = gen_data_file_only(3000)
    print(f"  Data file only:         {len(p3):>6}")

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
