#!/usr/bin/env python3
"""
Generate ~20K tensor table pairs: NML programs paired with .nml.data file contents.

Teaches the model to produce both the .nml program AND the .nml.data file
across all opcode categories: simple ops, neural nets, vision, attention,
signal, reduction, training, and data-file-only format.

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
    INPUT_NAMES, OUTPUT_NAMES,
)


def _rand_data(n, lo=-2.0, hi=2.0, decimals=4):
    return ",".join(str(round(random.uniform(lo, hi), decimals)) for _ in range(n))

def _prog_and_data(q, program_lines, data_lines, syntax, prog_name="program"):
    program = "\n".join(apply_syntax(program_lines, syntax))
    data_file = "\n".join(data_lines)
    answer = f"Program ({prog_name}.nml):\n{program}\n\nData file ({prog_name}.nml.data):\n{data_file}"
    return _pair(q, answer)


def gen_nn_with_data(count=4000):
    """Neural network programs with matching weight/bias data files."""
    pairs = []
    configs = [
        {"input": 2, "hidden": 4, "output": 1, "name": "2-4-1"},
        {"input": 2, "hidden": 8, "output": 1, "name": "2-8-1"},
        {"input": 4, "hidden": 8, "output": 1, "name": "4-8-1"},
        {"input": 1, "hidden": 8, "output": 1, "name": "1-8-1"},
        {"input": 3, "hidden": 4, "output": 2, "name": "3-4-2"},
        {"input": 4, "hidden": 8, "output": 2, "name": "4-8-2"},
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
    """TRAIN+INFER training programs with matching data files."""
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        n_samples = random.choice([4, 8])
        inp_size = random.choice([1, 2, 3])
        hid_size = random.choice([4, 8])
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

        q = f"Write NML TRAIN+INFER training program with data file for {n_samples} samples, {inp_size} features" + syntax_tag(syntax)

        program_lines = [
            _fmt("LD", "R0", "@training_inputs"),
            _fmt("LD", "R9", "@training_targets"),
            _fmt("LD", "R1", "@w1"), _fmt("LD", "R2", "@b1"),
            _fmt("LD", "R3", "@w2"), _fmt("LD", "R4", "@b2"),
            _fmt("ALLC", "RU", f"#[6]", f"{epochs},{lr},0,0,0,0"),
            _fmt("TRAIN", "RU"),
            _fmt("INFER", "R8", "R0"),
            _fmt("ST", "R8", "@predictions"),
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
        n_features = random.choice([1, 2, 3, 4])
        n_outputs = random.choice([1, 2])
        hid = random.choice([4, 8, 16])
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


def gen_simple_with_data(count=2000):
    """Simple LD/SCLR/MADD/MSUB programs with scalar or small tensor data."""
    pairs = []
    ops = [
        ("scale {name} by {v}", lambda n, v: [
            _fmt("LD", "R0", f"@{n}"), _fmt("SCLR", "R1", "R0", f"#{v}"),
            _fmt("ST", "R1", "@result"), "HALT"]),
        ("add {name} and {name2}", lambda n, v, n2=None: [
            _fmt("LD", "R0", f"@{n}"), _fmt("LD", "R1", f"@{n2}"),
            _fmt("MADD", "R2", "R0", "R1"), _fmt("ST", "R2", "@result"), "HALT"]),
        ("subtract {name2} from {name}", lambda n, v, n2=None: [
            _fmt("LD", "R0", f"@{n}"), _fmt("LD", "R1", f"@{n2}"),
            _fmt("MSUB", "R2", "R0", "R1"), _fmt("ST", "R2", "@result"), "HALT"]),
        ("element-wise multiply {name} and {name2}", lambda n, v, n2=None: [
            _fmt("LD", "R0", f"@{n}"), _fmt("LD", "R1", f"@{n2}"),
            _fmt("EMUL", "R2", "R0", "R1"), _fmt("ST", "R2", "@result"), "HALT"]),
        ("divide {name} by {v}", lambda n, v: [
            _fmt("LD", "R0", f"@{n}"), _fmt("SDIV", "R1", "R0", f"#{v}"),
            _fmt("ST", "R1", "@result"), "HALT"]),
    ]
    for _ in range(count):
        syntax = pick_syntax()
        size = random.choice([1, 2, 4, 8])
        shape = f"1,{size}" if size > 1 else "1"
        name = random.choice(["input", "value", "data", "x", "signal"])
        name2 = random.choice(["weights", "bias", "y", "factor"])
        v = round(random.uniform(0.1, 10.0), 2)
        op_idx = random.randint(0, len(ops) - 1)
        tmpl, gen_fn = ops[op_idx]

        q = "Write NML program and data file to " + tmpl.format(name=name, name2=name2, v=v) + syntax_tag(syntax)

        if op_idx in (1, 2, 3):
            lines = gen_fn(name, v, n2=name2)
            data_lines = [
                f"@{name} shape={shape} data={_rand_data(size, 0, 10)}",
                f"@{name2} shape={shape} data={_rand_data(size, 0, 10)}",
            ]
        else:
            lines = gen_fn(name, v)
            data_lines = [f"@{name} shape={shape} data={_rand_data(size, 0, 10)}"]

        pairs.append(_prog_and_data(q, lines, data_lines, syntax))
    return pairs


def gen_activation_with_data(count=1500):
    """Activation functions with proper input data."""
    pairs = []
    acts = [("RELU", "ReLU"), ("SIGM", "sigmoid"), ("TANH", "tanh"),
            ("SOFT", "softmax"), ("GELU", "GELU")]
    for _ in range(count):
        syntax = pick_syntax()
        op, name = random.choice(acts)
        size = random.choice([4, 8, 16])
        q = f"Write NML program and data file to apply {name} to a {size}-element vector" + syntax_tag(syntax)
        lines = [
            _fmt("LD", "R0", "@input"),
            _fmt(op, "R1", "R0"),
            _fmt("ST", "R1", "@result"),
            "HALT",
        ]
        data_lines = [f"@input shape=1,{size} data={_rand_data(size, -2.0, 2.0)}"]
        pairs.append(_prog_and_data(q, lines, data_lines, syntax))
    return pairs


def gen_vision_with_data(count=1500):
    """CONV, POOL, UPSC, PADZ programs with matching kernel/image data."""
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        h, w = random.choice([(4, 4), (8, 8), (3, 3), (6, 6)])
        kh, kw = random.choice([(2, 2), (3, 3)])
        variant = random.randint(0, 2)

        if variant == 0:
            q = f"Write NML program and data file for convolution on a {h}x{w} image with {kh}x{kw} kernel" + syntax_tag(syntax)
            lines = [
                _fmt("LD", "R0", "@image"), _fmt("LD", "R1", "@kernel"),
                _fmt("CONV", "R2", "R0", "R1"), _fmt("RELU", "R2", "R2"),
                _fmt("ST", "R2", "@result"), "HALT",
            ]
            data_lines = [
                f"@image shape={h},{w} data={_rand_data(h*w, 0, 1)}",
                f"@kernel shape={kh},{kw} data={_rand_data(kh*kw, -1, 1)}",
            ]
        elif variant == 1:
            q = f"Write NML program and data file for conv + max pool on a {h}x{w} image" + syntax_tag(syntax)
            lines = [
                _fmt("LD", "R0", "@image"), _fmt("LD", "R1", "@kernel"),
                _fmt("CONV", "R2", "R0", "R1"), _fmt("RELU", "R2", "R2"),
                _fmt("POOL", "R3", "R2"),
                _fmt("ST", "R3", "@result"), "HALT",
            ]
            data_lines = [
                f"@image shape={h},{w} data={_rand_data(h*w, 0, 1)}",
                f"@kernel shape={kh},{kw} data={_rand_data(kh*kw, -1, 1)}",
            ]
        else:
            q = f"Write NML program and data file to zero-pad and convolve a {h}x{w} image" + syntax_tag(syntax)
            lines = [
                _fmt("LD", "R0", "@image"),
                _fmt("PADZ", "R1", "R0"),
                _fmt("LD", "R2", "@kernel"),
                _fmt("CONV", "R3", "R1", "R2"),
                _fmt("ST", "R3", "@result"), "HALT",
            ]
            data_lines = [
                f"@image shape={h},{w} data={_rand_data(h*w, 0, 1)}",
                f"@kernel shape={kh},{kw} data={_rand_data(kh*kw, -1, 1)}",
            ]
        pairs.append(_prog_and_data(q, lines, data_lines, syntax))
    return pairs


def gen_attention_with_data(count=1500):
    """ATTN, NORM, EMBD, GELU programs with Q/K/V data of correct shapes."""
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        seq_len = random.choice([2, 4])
        d_model = random.choice([4, 8])
        variant = random.randint(0, 3)

        if variant == 0:
            q = f"Write NML program and data file for attention with seq_len={seq_len}, d_model={d_model}" + syntax_tag(syntax)
            lines = [
                _fmt("LD", "R0", "@queries"), _fmt("LD", "R1", "@keys"),
                _fmt("LD", "R2", "@values"),
                _fmt("ATTN", "R3", "R0", "R1", "R2"),
                _fmt("ST", "R3", "@result"), "HALT",
            ]
            data_lines = [
                f"@queries shape={seq_len},{d_model} data={_rand_data(seq_len*d_model, -1, 1)}",
                f"@keys shape={seq_len},{d_model} data={_rand_data(seq_len*d_model, -1, 1)}",
                f"@values shape={seq_len},{d_model} data={_rand_data(seq_len*d_model, -1, 1)}",
            ]
        elif variant == 1:
            q = f"Write NML program and data file for layer normalization on a {seq_len}x{d_model} tensor" + syntax_tag(syntax)
            lines = [
                _fmt("LD", "R0", "@input"),
                _fmt("NORM", "R1", "R0"),
                _fmt("ST", "R1", "@result"), "HALT",
            ]
            data_lines = [
                f"@input shape={seq_len},{d_model} data={_rand_data(seq_len*d_model, -2, 2)}",
            ]
        elif variant == 2:
            vocab = random.choice([8, 16, 32])
            embed_d = random.choice([4, 8])
            q = f"Write NML program and data file for embedding lookup (vocab={vocab}, d={embed_d})" + syntax_tag(syntax)
            indices = ",".join(str(random.randint(0, vocab-1)) for _ in range(seq_len))
            lines = [
                _fmt("LD", "R0", "@token_ids"), _fmt("LD", "R1", "@embed_table"),
                _fmt("EMBD", "R2", "R1", "R0"),
                _fmt("ST", "R2", "@result"), "HALT",
            ]
            data_lines = [
                f"@token_ids shape={seq_len} data={indices}",
                f"@embed_table shape={vocab},{embed_d} data={_rand_data(vocab*embed_d, -0.5, 0.5)}",
            ]
        else:
            q = f"Write NML program and data file for attention + norm + GELU FFN" + syntax_tag(syntax)
            ff_dim = d_model * 2
            lines = [
                _fmt("LD", "R0", "@input"),
                _fmt("LD", "R1", "@keys"), _fmt("LD", "R2", "@values"),
                _fmt("ATTN", "R3", "R0", "R1", "R2"),
                _fmt("MADD", "R3", "R3", "R0"),
                _fmt("NORM", "R3", "R3"),
                _fmt("LD", "R4", "@ff_w"),
                _fmt("MMUL", "R5", "R3", "R4"),
                _fmt("GELU", "R5", "R5"),
                _fmt("ST", "R5", "@result"), "HALT",
            ]
            data_lines = [
                f"@input shape={seq_len},{d_model} data={_rand_data(seq_len*d_model, -1, 1)}",
                f"@keys shape={seq_len},{d_model} data={_rand_data(seq_len*d_model, -1, 1)}",
                f"@values shape={seq_len},{d_model} data={_rand_data(seq_len*d_model, -1, 1)}",
                f"@ff_w shape={d_model},{ff_dim} data={_rand_data(d_model*ff_dim, -0.5, 0.5)}",
            ]
        pairs.append(_prog_and_data(q, lines, data_lines, syntax))
    return pairs


def gen_signal_with_data(count=1000):
    """FFT and FILT programs with 1D signal data."""
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        length = random.choice([8, 16, 32, 64])
        variant = random.randint(0, 1)

        if variant == 0:
            q = f"Write NML program and data file to compute FFT of a {length}-sample signal" + syntax_tag(syntax)
            lines = [
                _fmt("LD", "R0", "@signal"),
                _fmt("FFT", "R1", "R2", "R0"),
                _fmt("ST", "R1", "@real_part"), _fmt("ST", "R2", "@imag_part"),
                "HALT",
            ]
            data_lines = [
                f"@signal shape=1,{length} data={_rand_data(length, -1, 1)}",
            ]
        else:
            taps = random.choice([4, 8, 16])
            q = f"Write NML program and data file to filter a {length}-sample signal with {taps} FIR taps" + syntax_tag(syntax)
            lines = [
                _fmt("LD", "R0", "@signal"), _fmt("LD", "R1", "@coeffs"),
                _fmt("FILT", "R2", "R0", "R1"),
                _fmt("ST", "R2", "@result"), "HALT",
            ]
            data_lines = [
                f"@signal shape=1,{length} data={_rand_data(length, -1, 1)}",
                f"@coeffs shape=1,{taps} data={_rand_data(taps, -0.5, 0.5)}",
            ]
        pairs.append(_prog_and_data(q, lines, data_lines, syntax))
    return pairs


def gen_reduction_with_data(count=1500):
    """RDUC, WHER, CLMP, CMPR, SDOT, DIST programs with data."""
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        size = random.choice([4, 8, 16])
        variant = random.randint(0, 5)

        if variant == 0:
            op_code, op_name = random.choice([("#0", "sum"), ("#1", "mean"), ("#2", "max"), ("#3", "min")])
            q = f"Write NML program and data file to compute {op_name} of a {size}-element tensor" + syntax_tag(syntax)
            lines = [
                _fmt("LD", "R0", "@data"),
                _fmt("RDUC", "R1", "R0", op_code),
                _fmt("ST", "R1", "@result"), "HALT",
            ]
            data_lines = [f"@data shape=1,{size} data={_rand_data(size, 0, 100)}"]
        elif variant == 1:
            lo, hi = round(random.uniform(0, 30), 1), round(random.uniform(50, 100), 1)
            q = f"Write NML program and data file to clamp values to [{lo}, {hi}]" + syntax_tag(syntax)
            lines = [
                _fmt("LD", "R0", "@data"),
                _fmt("CLMP", "R1", "R0", f"#{lo}", f"#{hi}"),
                _fmt("ST", "R1", "@result"), "HALT",
            ]
            data_lines = [f"@data shape=1,{size} data={_rand_data(size, -50, 150)}"]
        elif variant == 2:
            q = f"Write NML program and data file to conditionally select between two tensors" + syntax_tag(syntax)
            lines = [
                _fmt("LD", "R0", "@condition"), _fmt("LD", "R1", "@a"), _fmt("LD", "R2", "@b"),
                _fmt("WHER", "R3", "R0", "R1", "R2"),
                _fmt("ST", "R3", "@result"), "HALT",
            ]
            cond_data = ",".join(str(random.choice([0, 1])) for _ in range(size))
            data_lines = [
                f"@condition shape=1,{size} data={cond_data}",
                f"@a shape=1,{size} data={_rand_data(size, 0, 10)}",
                f"@b shape=1,{size} data={_rand_data(size, 0, 10)}",
            ]
        elif variant == 3:
            thresh = round(random.uniform(10, 90), 1)
            q = f"Write NML program and data file to create a comparison mask against {thresh}" + syntax_tag(syntax)
            lines = [
                _fmt("LD", "R0", "@data"),
                _fmt("CMPR", "R1", "R0", f"#{thresh}", "#4"),
                _fmt("ST", "R1", "@result"), "HALT",
            ]
            data_lines = [f"@data shape=1,{size} data={_rand_data(size, 0, 100)}"]
        elif variant == 4:
            q = f"Write NML program and data file for dot product of two {size}-element vectors" + syntax_tag(syntax)
            lines = [
                _fmt("LD", "R0", "@a"), _fmt("LD", "R1", "@b"),
                _fmt("SDOT", "R2", "R0", "R1"),
                _fmt("ST", "R2", "@result"), "HALT",
            ]
            data_lines = [
                f"@a shape=1,{size} data={_rand_data(size, -1, 1)}",
                f"@b shape=1,{size} data={_rand_data(size, -1, 1)}",
            ]
        else:
            metric_code, metric_name = random.choice([("#0", "cosine"), ("#1", "euclidean")])
            q = f"Write NML program and data file for {metric_name} distance between two {size}-d vectors" + syntax_tag(syntax)
            lines = [
                _fmt("LD", "R0", "@a"), _fmt("LD", "R1", "@b"),
                _fmt("DIST", "R2", "R0", "R1", metric_code),
                _fmt("ST", "R2", "@result"), "HALT",
            ]
            data_lines = [
                f"@a shape=1,{size} data={_rand_data(size, -1, 1)}",
                f"@b shape=1,{size} data={_rand_data(size, -1, 1)}",
            ]
        pairs.append(_prog_and_data(q, lines, data_lines, syntax))
    return pairs


def gen_conditional_with_data(count=1500):
    """Branching programs (CMPI/CMPF/JMPT/JMPF) with scalar data."""
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        val = round(random.uniform(10, 500), 2)
        thresh = round(random.uniform(50, 300), 2)
        rate_lo = round(random.uniform(0.01, 0.15), 4)
        rate_hi = round(random.uniform(0.15, 0.50), 4)
        variant = random.randint(0, 1)

        if variant == 0:
            q = f"Write NML program and data file: if input < {thresh} scale by {rate_lo} else by {rate_hi}" + syntax_tag(syntax)
            lines = [
                _fmt("LD", "R0", "@input"),
                _fmt("CMPI", "RE", "R0", f"#{thresh}"),
                _fmt("JMPF", "#3"),
                _fmt("SCLR", "R1", "R0", f"#{rate_lo}"),
                _fmt("JUMP", "#2"),
                _fmt("SCLR", "R1", "R0", f"#{rate_hi}"),
                _fmt("ST", "R1", "@result"), "HALT",
            ]
            data_lines = [f"@input shape=1 data={val}"]
        else:
            tiers = sorted([round(random.uniform(50, 500), 2) for _ in range(2)])
            rates = [round(random.uniform(0.05, 0.15), 4),
                     round(random.uniform(0.15, 0.25), 4),
                     round(random.uniform(0.25, 0.40), 4)]
            q = (f"Write NML program and data file for a 3-tier rate: "
                 f"below {tiers[0]} at {rates[0]}, {tiers[0]}-{tiers[1]} at {rates[1]}, above at {rates[2]}") + syntax_tag(syntax)
            lines = [
                _fmt("LD", "R0", "@input"),
                _fmt("ALLC", "RA", "#[1]"),
                _fmt("CMPF", "RE", "R0", "#0", f"#{tiers[1]}"),
                _fmt("JMPF", "#5"),
                _fmt("CMPF", "RE", "R0", "#0", f"#{tiers[0]}"),
                _fmt("JMPF", "#5"),
                _fmt("SCLR", "RA", "R0", f"#{rates[0]}"),
                _fmt("JUMP", "#8"),
                _fmt("LEAF", "RA", f"#{round(tiers[0] * rates[0], 2)}"),
                _fmt("LEAF", "RC", f"#{tiers[0]}"),
                _fmt("MSUB", "R8", "R0", "RC"),
                _fmt("SCLR", "R8", "R8", f"#{rates[1]}"),
                _fmt("TACC", "RA", "RA", "R8"),
                _fmt("JUMP", "#4"),
                _fmt("LEAF", "RA", f"#{round(tiers[0]*rates[0] + (tiers[1]-tiers[0])*rates[1], 2)}"),
                _fmt("LEAF", "RC", f"#{tiers[1]}"),
                _fmt("MSUB", "R8", "R0", "RC"),
                _fmt("SCLR", "R8", "R8", f"#{rates[2]}"),
                _fmt("TACC", "RA", "RA", "R8"),
                _fmt("ST", "RA", "@result"), "HALT",
            ]
            data_lines = [f"@input shape=1 data={round(random.uniform(10, 800), 2)}"]
        pairs.append(_prog_and_data(q, lines, data_lines, syntax))
    return pairs


def gen_loop_with_data(count=1000):
    """Loop programs with data files."""
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        n = random.choice([5, 10, 20, 50])
        size = random.choice([1, 4, 8])
        shape = f"1,{size}" if size > 1 else "1"
        variant = random.randint(0, 1)

        if variant == 0:
            q = f"Write NML program and data file to accumulate input over {n} iterations" + syntax_tag(syntax)
            lines = [
                _fmt("LD", "R0", "@input"),
                _fmt("ALLC", "RA", f"#[{size}]" if size > 1 else "#[1]"),
                _fmt("LEAF", "RD", f"#{n}"),
                _fmt("LOOP", "RD"),
                _fmt("TACC", "RA", "RA", "R0"),
                "ENDP",
                _fmt("ST", "RA", "@result"), "HALT",
            ]
        else:
            v = round(random.uniform(1.01, 1.1), 4)
            q = f"Write NML program and data file for compound growth over {n} periods at rate {v}" + syntax_tag(syntax)
            lines = [
                _fmt("LD", "R0", "@principal"),
                _fmt("LEAF", "R1", f"#{v}"),
                _fmt("LEAF", "RD", f"#{n}"),
                _fmt("LOOP", "RD"),
                _fmt("EMUL", "R0", "R0", "R1"),
                "ENDP",
                _fmt("ST", "R0", "@result"), "HALT",
            ]
        data_lines = [f"@{random.choice(['input', 'principal'])} shape={shape} data={_rand_data(size, 1, 100)}"]
        pairs.append(_prog_and_data(q, lines, data_lines, syntax))
    return pairs


def gen_matrix_ops_with_data(count=1500):
    """MMUL, TRNS, RSHP, MERG, SPLT with correct shapes."""
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        variant = random.randint(0, 4)

        if variant == 0:
            m, k, n = random.choice([(2,3,1), (4,8,1), (1,4,2), (3,4,2), (2,4,4)])
            q = f"Write NML program and data file for matrix multiply ({m}x{k}) @ ({k}x{n})" + syntax_tag(syntax)
            lines = [
                _fmt("LD", "R0", "@a"), _fmt("LD", "R1", "@b"),
                _fmt("MMUL", "R2", "R0", "R1"),
                _fmt("ST", "R2", "@result"), "HALT",
            ]
            data_lines = [
                f"@a shape={m},{k} data={_rand_data(m*k, -1, 1)}",
                f"@b shape={k},{n} data={_rand_data(k*n, -1, 1)}",
            ]
        elif variant == 1:
            r, c = random.choice([(2,3), (3,4), (4,8), (2,2)])
            q = f"Write NML program and data file to transpose a {r}x{c} matrix" + syntax_tag(syntax)
            lines = [
                _fmt("LD", "R0", "@matrix"),
                _fmt("TRNS", "R1", "R0"),
                _fmt("ST", "R1", "@result"), "HALT",
            ]
            data_lines = [f"@matrix shape={r},{c} data={_rand_data(r*c, -2, 2)}"]
        elif variant == 2:
            total = random.choice([8, 12, 16])
            shapes = [(1, total), (2, total//2), (4, total//4)] if total % 4 == 0 else [(1, total), (2, total//2)]
            new_r, new_c = random.choice(shapes)
            q = f"Write NML program and data file to reshape a {total}-element tensor to {new_r}x{new_c}" + syntax_tag(syntax)
            lines = [
                _fmt("LD", "R0", "@data"),
                _fmt("RSHP", "R1", "R0", f"#[{new_r},{new_c}]"),
                _fmt("ST", "R1", "@result"), "HALT",
            ]
            data_lines = [f"@data shape=1,{total} data={_rand_data(total, -1, 1)}"]
        elif variant == 3:
            size = random.choice([4, 8])
            q = f"Write NML program and data file to concatenate two {size}-element vectors" + syntax_tag(syntax)
            lines = [
                _fmt("LD", "R0", "@a"), _fmt("LD", "R1", "@b"),
                _fmt("MERG", "R2", "R0", "R1", "#0"),
                _fmt("ST", "R2", "@result"), "HALT",
            ]
            data_lines = [
                f"@a shape=1,{size} data={_rand_data(size, 0, 10)}",
                f"@b shape=1,{size} data={_rand_data(size, 0, 10)}",
            ]
        else:
            size = random.choice([4, 8, 16])
            q = f"Write NML program and data file to split a {size}-element tensor along dim 0" + syntax_tag(syntax)
            lines = [
                _fmt("LD", "R0", "@data"),
                _fmt("SPLT", "R1", "R2", "R0", "#0"),
                _fmt("ST", "R1", "@lower"), _fmt("ST", "R2", "@upper"), "HALT",
            ]
            data_lines = [f"@data shape=2,{size//2} data={_rand_data(size, -5, 5)}"]
        pairs.append(_prog_and_data(q, lines, data_lines, syntax))
    return pairs


def gen_gather_scatter_with_data(count=1000):
    """GATH, SCAT/SCTR with proper index data."""
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        src_size = random.choice([8, 16, 32])
        n_indices = random.choice([4, 8])
        variant = random.randint(0, 1)

        indices = ",".join(str(random.randint(0, src_size - 1)) for _ in range(n_indices))

        if variant == 0:
            q = f"Write NML program and data file to gather {n_indices} elements from a {src_size}-element tensor" + syntax_tag(syntax)
            lines = [
                _fmt("LD", "R0", "@data"), _fmt("LD", "R1", "@indices"),
                _fmt("GATH", "R2", "R0", "R1"),
                _fmt("ST", "R2", "@result"), "HALT",
            ]
        else:
            q = f"Write NML program and data file to scatter {n_indices} values into a {src_size}-element buffer" + syntax_tag(syntax)
            lines = [
                _fmt("LD", "R0", "@values"), _fmt("LD", "R1", "@indices"),
                _fmt("ALLC", "R2", f"#[{src_size}]"),
                _fmt("SCTR", "R2", "R0", "R1"),
                _fmt("ST", "R2", "@result"), "HALT",
            ]

        data_lines = [
            f"@{'data' if variant == 0 else 'values'} shape=1,{src_size if variant == 0 else n_indices} data={_rand_data(src_size if variant == 0 else n_indices, 0, 100)}",
            f"@indices shape=1,{n_indices} data={indices}",
        ]
        pairs.append(_prog_and_data(q, lines, data_lines, syntax))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Generate NML tensor table pairs")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    args = parser.parse_args()

    print("=" * 60)
    print("  NML Tensor Table Generator — Programs + Data Files")
    print("=" * 60)

    categories = [
        ("NN + data",            gen_nn_with_data, 3000),
        ("TRAIN+INFER + data",   gen_training_with_data, 2000),
        ("Data file only",       gen_data_file_only, 2000),
        ("Simple ops + data",    gen_simple_with_data, 2000),
        ("Activations + data",   gen_activation_with_data, 1500),
        ("Vision + data",        gen_vision_with_data, 1500),
        ("Attention + data",     gen_attention_with_data, 1500),
        ("Signal + data",        gen_signal_with_data, 1000),
        ("Reduction + data",     gen_reduction_with_data, 1500),
        ("Conditional + data",   gen_conditional_with_data, 1500),
        ("Loop + data",          gen_loop_with_data, 1000),
        ("Matrix ops + data",    gen_matrix_ops_with_data, 1500),
        ("Gather/scatter + data", gen_gather_scatter_with_data, 1000),
    ]

    all_pairs = []
    for name, gen_fn, count in categories:
        pairs = gen_fn(count)
        all_pairs.extend(pairs)
        print(f"  {name:<26} {len(pairs):>6}")

    random.shuffle(all_pairs)

    print(f"{'─' * 60}")
    print(f"  TOTAL:                   {len(all_pairs):>6}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\n  Written to: {out_path}")


if __name__ == "__main__":
    main()
