#!/usr/bin/env python3
"""
Generate ~10K library pairs exercising diverse NML program patterns.

Covers varied computation types: rate calculations, classification,
signal processing, data transformations, anomaly detection, and more.
Uses the full register set (R0-RV) and all opcode categories.

Output: domain/output/training/nml_library_pairs.jsonl
"""

import json
import random
import argparse
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent.parent / "domain" / "output" / "training" / "nml_library_pairs.jsonl"

random.seed(2026_05)

from nml_core_training_gen import (
    REGS_CLASSIC, _fmt, _pair, _fval, _ival, _pick,
    INPUT_NAMES, OUTPUT_NAMES,
    apply_syntax, pick_syntax, syntax_tag, _inp, _out,
)


def gen_rate_calculations(count=2000):
    """Flat-rate, progressive, and blended rate computations."""
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        rate = round(random.uniform(0.005, 0.15), 4)
        i = _inp(); o = _out()

        pattern = random.choice(["flat", "progressive", "capped"])

        if pattern == "flat":
            q = f"Write NML for a flat {rate*100:.2f}% rate on {i}" + syntax_tag(syntax)
            lines = [
                _fmt("LD", "R0", f"@{i}"),
                _fmt("SCLR", "R1", "R0", f"#{rate}"),
                _fmt("ST", "R1", f"@{o}"),
                "HALT",
            ]
        elif pattern == "progressive":
            t1 = _ival(10000, 50000); t2 = _ival(t1 + 10000, 200000)
            r1 = round(random.uniform(0.01, 0.05), 4)
            r2 = round(random.uniform(0.05, 0.10), 4)
            r3 = round(random.uniform(0.10, 0.15), 4)
            q = f"Write NML for progressive rates: {r1*100:.1f}% up to {t1}, {r2*100:.1f}% to {t2}, {r3*100:.1f}% above" + syntax_tag(syntax)
            lines = [
                _fmt("LD", "R0", f"@{i}"),
                _fmt("CMPI", "RE", "R0", f"#{t1}.0"),
                _fmt("JMPF", "#3"),
                _fmt("SCLR", "R1", "R0", f"#{r1}"),
                _fmt("JUMP", "#6"),
                _fmt("CMPI", "RE", "R0", f"#{t2}.0"),
                _fmt("JMPF", "#3"),
                _fmt("SCLR", "R1", "R0", f"#{r2}"),
                _fmt("JUMP", "#2"),
                _fmt("SCLR", "R1", "R0", f"#{r3}"),
                _fmt("ST", "R1", f"@{o}"),
                "HALT",
            ]
        else:
            cap = _fval(100, 10000)
            q = f"Write NML for {rate*100:.2f}% rate capped at {cap}" + syntax_tag(syntax)
            lines = [
                _fmt("LD", "R0", f"@{i}"),
                _fmt("SCLR", "R1", "R0", f"#{rate}"),
                _fmt("LEAF", "R2", f"#{cap}"),
                _fmt("CMP", "R1", "R2"),
                _fmt("JMPT", "#2"),
                _fmt("MOV", "R1", "R2"),
                _fmt("ST", "R1", f"@{o}"),
                "HALT",
            ]

        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs


def gen_anomaly_detection(count=1500):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        layers = random.choice([2, 3])
        q = f"Write NML for a {layers}-layer anomaly detector" + syntax_tag(syntax)

        if layers == 2:
            lines = [
                _fmt("LD", "R0", "@sensor_data"),
                _fmt("LD", "R1", "@w1"), _fmt("LD", "R2", "@b1"),
                _fmt("MMUL", "R3", "R0", "R1"),
                _fmt("MADD", "R3", "R3", "R2"),
                _fmt("RELU", "R3", "R3"),
                _fmt("LD", "R4", "@w2"), _fmt("LD", "R5", "@b2"),
                _fmt("MMUL", "R6", "R3", "R4"),
                _fmt("MADD", "R6", "R6", "R5"),
                _fmt("SIGM", "R6", "R6"),
                _fmt("ST", "R6", "@anomaly_score"),
                "HALT",
            ]
        else:
            lines = [
                _fmt("LD", "R0", "@sensor_data"),
                _fmt("LD", "R1", "@w1"), _fmt("LD", "R2", "@b1"),
                _fmt("MMUL", "R3", "R0", "R1"), _fmt("MADD", "R3", "R3", "R2"),
                _fmt("RELU", "R3", "R3"),
                _fmt("LD", "R4", "@w2"), _fmt("LD", "R5", "@b2"),
                _fmt("MMUL", "R6", "R3", "R4"), _fmt("MADD", "R6", "R6", "R5"),
                _fmt("RELU", "R6", "R6"),
                _fmt("LD", "R7", "@w3"), _fmt("LD", "R8", "@b3"),
                _fmt("MMUL", "R9", "R6", "R7"), _fmt("MADD", "R9", "R9", "R8"),
                _fmt("SIGM", "R9", "R9"),
                _fmt("ST", "R9", "@anomaly_score"),
                "HALT",
            ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs


def gen_signal_processing(count=1500):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        pattern = random.choice(["fft_filter", "filter_reduce", "normalize_fft"])

        if pattern == "fft_filter":
            q = "Write NML to FFT a signal, filter it, and find the peak" + syntax_tag(syntax)
            lines = [
                _fmt("LD", "R0", "@signal"),
                _fmt("FFT", "R1", "R2", "R0"),
                _fmt("LD", "R3", "@filter_coeffs"),
                _fmt("FILT", "R4", "R0", "R3"),
                _fmt("RDUC", "R5", "R4", "#2"),
                _fmt("ST", "R1", "@spectrum"),
                _fmt("ST", "R4", "@filtered"),
                _fmt("ST", "R5", "@peak"),
                "HALT",
            ]
        elif pattern == "filter_reduce":
            q = "Write NML to filter a signal and compute its mean and max" + syntax_tag(syntax)
            lines = [
                _fmt("LD", "R0", "@signal"),
                _fmt("LD", "R1", "@coefficients"),
                _fmt("FILT", "R2", "R0", "R1"),
                _fmt("RDUC", "R3", "R2", "#1"),
                _fmt("RDUC", "R4", "R2", "#2"),
                _fmt("ST", "R3", "@mean_val"),
                _fmt("ST", "R4", "@max_val"),
                "HALT",
            ]
        else:
            q = "Write NML to normalize a signal and compute its frequency spectrum" + syntax_tag(syntax)
            lines = [
                _fmt("LD", "R0", "@signal"),
                _fmt("NORM", "R1", "R0"),
                _fmt("FFT", "R2", "R3", "R1"),
                _fmt("ST", "R1", "@normalized"),
                _fmt("ST", "R2", "@real_spectrum"),
                _fmt("ST", "R3", "@imag_spectrum"),
                "HALT",
            ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs


def gen_data_transform(count=2000):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        pattern = random.choice(["reshape_transpose", "split_merge", "clamp_convert",
                                  "gather_scatter", "embed_project", "extended_regs"])

        if pattern == "reshape_transpose":
            i = _inp(); o = _out()
            shape = random.choice(["#[4,4]", "#[2,8]", "#[8,2]", "#[1,16]"])
            q = f"Write NML to reshape {i} to {shape.replace('#','')} and transpose" + syntax_tag(syntax)
            lines = [
                _fmt("LD", "R0", f"@{i}"),
                _fmt("RSHP", "R1", "R0", shape),
                _fmt("TRNS", "R2", "R1"),
                _fmt("ST", "R2", f"@{o}"),
                "HALT",
            ]
        elif pattern == "split_merge":
            i = _inp(); o = _out()
            q = f"Write NML to split {i}, process halves, and merge" + syntax_tag(syntax)
            lines = [
                _fmt("LD", "R0", f"@{i}"),
                _fmt("SPLT", "R1", "R2", "R0", "#0"),
                _fmt("SCLR", "R1", "R1", "#2.0"),
                _fmt("SCLR", "R2", "R2", "#0.5"),
                _fmt("MERG", "R3", "R1", "R2", "#0"),
                _fmt("ST", "R3", f"@{o}"),
                "HALT",
            ]
        elif pattern == "clamp_convert":
            i = _inp(); o = _out()
            lo = _fval(0, 10); hi = _fval(50, 200)
            q = f"Write NML to clamp {i} to [{lo},{hi}] then convert to integer" + syntax_tag(syntax)
            lines = [
                _fmt("LD", "R0", f"@{i}"),
                _fmt("CLMP", "R1", "R0", f"#{lo}", f"#{hi}"),
                _fmt("FTOI", "R2", "R1"),
                _fmt("ST", "R2", f"@{o}"),
                "HALT",
            ]
        elif pattern == "gather_scatter":
            i = _inp(); o = _out()
            q = f"Write NML to gather elements from {i} and scatter to a new tensor" + syntax_tag(syntax)
            lines = [
                _fmt("LD", "R0", f"@{i}"),
                _fmt("LD", "R1", "@indices"),
                _fmt("GATH", "R2", "R0", "R1"),
                _fmt("ALLC", "R3", "#[8]"),
                _fmt("SCAT", "R2", "R3", "R1"),
                _fmt("ST", "R3", f"@{o}"),
                "HALT",
            ]
        elif pattern == "embed_project":
            o = _out()
            q = f"Write NML to embed tokens, project, and compute distance" + syntax_tag(syntax)
            lines = [
                _fmt("LD", "R0", "@token_ids"),
                _fmt("LD", "R1", "@embed_table"),
                _fmt("EMBD", "R2", "R1", "R0"),
                _fmt("LD", "R3", "@projection_matrix"),
                _fmt("PROJ", "R4", "R2", "R3"),
                _fmt("LD", "R5", "@reference"),
                _fmt("DIST", "R6", "R4", "R5", "#0"),
                _fmt("ST", "R6", f"@{o}"),
                "HALT",
            ]
        else:
            q = "Write NML using extended registers RG-RM for multi-stage computation" + syntax_tag(syntax)
            lines = [
                _fmt("LD", "RG", "@stage1_weights"),
                _fmt("LD", "RH", "@stage2_weights"),
                _fmt("LD", "R0", "@input"),
                _fmt("MMUL", "RK", "R0", "RG"),
                _fmt("RELU", "RK", "RK"),
                _fmt("MMUL", "RL", "RK", "RH"),
                _fmt("SIGM", "RL", "RL"),
                _fmt("ST", "RL", "@output"),
                "HALT",
            ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs


def gen_multi_opcode_programs(count=3000):
    """Programs that combine many opcodes in one program."""
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        pattern = random.choice(["full_pipeline", "conditional_nn", "subroutine_train",
                                  "vote_ensemble", "m2m_signed"])

        if pattern == "full_pipeline":
            q = "Write NML for a full data pipeline: load, normalize, dense layer, clamp, store" + syntax_tag(syntax)
            lines = [
                _fmt("LD", "R0", "@raw_data"),
                _fmt("NORM", "R0", "R0"),
                _fmt("LD", "R1", "@weights"), _fmt("LD", "R2", "@bias"),
                _fmt("MMUL", "R3", "R0", "R1"),
                _fmt("MADD", "R3", "R3", "R2"),
                _fmt("RELU", "R3", "R3"),
                _fmt("CLMP", "R4", "R3", "#0.0", "#1.0"),
                _fmt("ST", "R4", "@output"),
                "HALT",
            ]
        elif pattern == "conditional_nn":
            q = "Write NML that uses a different network path based on input magnitude" + syntax_tag(syntax)
            lines = [
                _fmt("LD", "R0", "@input"),
                _fmt("RDUC", "R1", "R0", "#1"),
                _fmt("CMPI", "RE", "R1", "#100.0"),
                _fmt("JMPF", "#5"),
                _fmt("LD", "R2", "@small_weights"),
                _fmt("MMUL", "R3", "R0", "R2"),
                _fmt("RELU", "R3", "R3"),
                _fmt("JUMP", "#4"),
                _fmt("LD", "R2", "@large_weights"),
                _fmt("MMUL", "R3", "R0", "R2"),
                _fmt("SIGM", "R3", "R3"),
                _fmt("ST", "R3", "@output"),
                "HALT",
            ]
        elif pattern == "subroutine_train":
            q = "Write NML with a training subroutine called from main" + syntax_tag(syntax)
            lines = [
                _fmt("LD", "R0", "@training_inputs"),
                _fmt("LD", "R9", "@training_targets"),
                _fmt("LD", "R1", "@w1"), _fmt("LD", "R2", "@b1"),
                _fmt("LD", "R3", "@w2"), _fmt("LD", "R4", "@b2"),
                _fmt("CALL", "#2"),
                _fmt("ST", "R1", "@trained_w1"),
                _fmt("JUMP", "#5"),
                _fmt("ALLC", "RU", "[6]", "1000,0.01,1,0,0,0"),
                _fmt("TRAIN", "RU", "@training_inputs", "@training_targets"),
                _fmt("INFER", "RA", "R0"),
                _fmt("ST", "RA", "@predictions"),
                "RET",
                "HALT",
            ]
        elif pattern == "vote_ensemble":
            q = "Write NML for an ensemble that runs 3 models and takes median vote" + syntax_tag(syntax)
            lines = [
                _fmt("LD", "R0", "@input"),
                _fmt("LD", "R1", "@model1_w"), _fmt("MMUL", "R4", "R0", "R1"), _fmt("SIGM", "R4", "R4"),
                _fmt("LD", "R2", "@model2_w"), _fmt("MMUL", "R5", "R0", "R2"), _fmt("SIGM", "R5", "R5"),
                _fmt("LD", "R3", "@model3_w"), _fmt("MMUL", "R6", "R0", "R3"), _fmt("SIGM", "R6", "R6"),
                _fmt("MERG", "R7", "R4", "R5", "#0"),
                _fmt("MERG", "R7", "R7", "R6", "#0"),
                _fmt("VOTE", "R8", "R7", "#0"),
                _fmt("ST", "R8", "@ensemble_output"),
                "HALT",
            ]
        else:
            q = "Write a signed M2M NML program with metadata and verification" + syntax_tag(syntax)
            agent = f"agent_{random.randint(1,50)}"
            lines = [
                f'SIGN  agent={agent}  key=ed25519:k{_ival(1000,9999)}  sig=s{_ival(1000,9999)}',
                f'META  @name "verified_calc"',
                f'META  @version "2.0"',
                f'META  @author "{agent}"',
                f'VRFY  @self  @{agent}',
                _fmt("LD", "R0", "@input"),
                _fmt("SCLR", "R1", "R0", f"#{_fval(0.01,1.0)}"),
                _fmt("ST", "R1", "@output"),
                "HALT",
            ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Generate NML library pairs")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    args = parser.parse_args()

    print("Generating NML library pairs...")
    print(f"{'─' * 60}")

    p1 = gen_rate_calculations(2000)
    print(f"  Rate calculations:      {len(p1):>6}")
    p2 = gen_anomaly_detection(1500)
    print(f"  Anomaly detection:      {len(p2):>6}")
    p3 = gen_signal_processing(1500)
    print(f"  Signal processing:      {len(p3):>6}")
    p4 = gen_data_transform(2000)
    print(f"  Data transforms:        {len(p4):>6}")
    p5 = gen_multi_opcode_programs(3000)
    print(f"  Multi-opcode programs:  {len(p5):>6}")

    all_pairs = p1 + p2 + p3 + p4 + p5
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
