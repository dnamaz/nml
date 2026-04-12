#!/usr/bin/env python3
"""
Generate ~85K equalization pairs to bring under-represented opcodes to 10K+ appearances.

Each under-represented opcode gets parameterized template generators with:
- Diverse prompt phrasings
- Variable names from generic pools
- Random numeric values
- Tri-syntax coverage (~60% classic, 25% symbolic, 15% verbose)
- Programs of varying complexity (3-15 instructions)
- 32-register usage (R0-RV)

Output: domain/output/training/nml_equalize_pairs.jsonl
"""

import json
import random
import argparse
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent.parent / "domain" / "output" / "training" / "nml_equalize_pairs.jsonl"

random.seed(2026)

# Import shared utilities from core generator
from nml_core_training_gen import (
    REGS_CLASSIC, REGS_GREEK, REGS_VERBOSE, SYM, VERBOSE,
    INPUT_NAMES, OUTPUT_NAMES, MEM_NAMES,
    _pick, _fval, _ival, _fmt, _pair,
    apply_syntax, pick_syntax, syntax_tag, _inp, _out, _mem,
    to_symbolic, to_verbose,
)

# Opcode targets: (opcode, pairs_needed)
ZERO_TIER = {
    "BKWD": 3500, "WUPD": 3500, "LOSS": 3500, "TRAIN": 3500,
}

CRITICAL_TIER = {
    "PTCH": 3000, "CMPR": 3000, "SYNC": 3000, "TRAP": 3000, "BNOT": 3000,
    "PADZ": 3000, "RSHP": 3000, "PROJ": 3000, "UPSC": 3000, "TRNS": 3000,
    "WHER": 2500, "SIGN": 2500, "VRFY": 2500, "ITOF": 2500, "CLMP": 2500,
    "VOTE": 2500, "EMBD": 2500, "FTOI": 2500, "SOFT": 2500, "DIST": 2500,
    "GATH": 2500, "FILT": 2500,
}

LOW_TIER = {
    "LINK": 2000, "FRAG": 2000, "ENDF": 2000,
    "TANH": 2000, "SDIV": 2000, "FFT": 2000,
    "ATTN": 2000, "POOL": 2000,
    "EDIV": 1500,
    "SCAT": 1500, "CMPF": 1500,
    "NORM": 1500, "CONV": 1500, "SIGM": 1500,
    "SDOT": 1000, "MERG": 1000, "META": 1000,
    "SPLT": 500, "CMP": 500, "GELU": 500,
}

ALL_TARGETS = {**ZERO_TIER, **CRITICAL_TIER, **LOW_TIER}

GRAD_REGS = ["RG", "RH", "RI"]
TRAIN_REGS = ["RJ", "RK", "RL", "RM"]
EXTENDED_REGS = ["RG","RH","RI","RJ","RK","RL","RM","RN","RO","RP","RQ","RR","RS","RT","RU","RV"]


def _r(exclude=()):
    pool = [r for r in REGS_CLASSIC[:16] if r not in exclude]
    return random.choice(pool)

def _rext():
    return random.choice(REGS_CLASSIC)

def _gr():
    return random.choice(GRAD_REGS)


# ═══════════════════════════════════════════════════════════════════════════════
# Per-opcode generators
# ═══════════════════════════════════════════════════════════════════════════════

def gen_BKWD(count):
    pairs = []
    prompts = [
        "Write NML to compute gradients of predictions w.r.t. targets using BKWD",
        "Use BKWD to backpropagate and store the gradient",
        "Compute the gradient of the loss for a neural network layer",
        "Write NML backpropagation using BKWD on the output layer",
        "Use BKWD to find how weights should change to reduce error",
    ]
    for _ in range(count):
        syntax = pick_syntax()
        r_pred = _r(); r_tgt = _r(exclude=[r_pred]); r_grad = _gr()
        o = _out(); i = _inp()
        q = random.choice(prompts) + syntax_tag(syntax)
        lines = [
            _fmt("LD", r_pred, f"@{i}"),
            _fmt("LD", r_tgt, "@targets"),
            _fmt("BKWD", r_grad, r_pred, r_tgt),
            _fmt("ST", r_grad, f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_WUPD(count):
    pairs = []
    prompts = [
        "Write NML to update weights using gradients with WUPD",
        "Apply gradient descent weight update in NML",
        "Use WUPD to adjust neural network weights",
        "Write NML weight update step using learning rate in RJ",
        "Update model parameters using computed gradients",
    ]
    for _ in range(count):
        syntax = pick_syntax()
        r_w = _r(); r_grad = _gr()
        lr = random.choice([0.001, 0.005, 0.01, 0.05, 0.1])
        q = random.choice(prompts) + syntax_tag(syntax)
        lines = [
            _fmt("LD", r_w, "@weights"),
            _fmt("LD", r_grad, "@gradients"),
            _fmt("LEAF", "RJ", f"#{lr}"),
            _fmt("WUPD", r_w, r_w, r_grad),
            _fmt("ST", r_w, "@updated_weights"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_LOSS(count):
    pairs = []
    modes = [("#0", "MSE"), ("#1", "cross-entropy"), ("#2", "MAE")]
    prompts = [
        "Write NML to compute {mode} loss between predictions and targets",
        "Calculate {mode} loss using the LOSS opcode",
        "Use NML LOSS instruction to measure {mode} error",
        "Write NML that computes {mode} loss and stores it",
    ]
    for _ in range(count):
        syntax = pick_syntax()
        mode_code, mode_name = random.choice(modes)
        r_pred = _r(); r_tgt = _r(exclude=[r_pred]); r_loss = _gr()
        q = random.choice(prompts).format(mode=mode_name) + syntax_tag(syntax)
        lines = [
            _fmt("LD", r_pred, "@predictions"),
            _fmt("LD", r_tgt, "@targets"),
            _fmt("LOSS", r_loss, r_pred, r_tgt, mode_code),
            _fmt("ST", r_loss, "@loss_value"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_TRAIN(count):
    pairs = []
    prompts = [
        "Write NML to train a neural network using TRAIN+INFER for {e} epochs at lr {lr}",
        "Use TRAIN+INFER to self-train a {topo} network",
        "Write a self-training NML program with TRAIN and INFER",
        "Train a network end-to-end using TRAIN with {e} epochs",
        "Write NML that uses TRAIN+INFER to learn from training data",
    ]
    topos = ["2-4-1", "2-8-1", "1-16-1", "3-8-1", "4-16-1", "1-64-1"]
    for _ in range(count):
        syntax = pick_syntax()
        epochs = random.choice([100, 200, 500, 1000, 2000, 5000])
        lr = random.choice([0.001, 0.005, 0.01, 0.05, 0.1, 0.5])
        topo = random.choice(topos)
        q = random.choice(prompts).format(e=epochs, lr=lr, topo=topo) + syntax_tag(syntax)
        lines = [
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
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_PTCH(count):
    pairs = []
    directives = ["@replace", "@insert", "@delete", "@update"]
    for _ in range(count):
        syntax = pick_syntax()
        d = random.choice(directives)
        i = _inp(); o = _out()
        q = f"Write NML with a PTCH directive to patch a program section{syntax_tag(syntax)}"
        lines = [
            f'META  @name "patched_calc"',
            f'PTCH  {d}  line=3  "SCLR R1 R0 #2.0"',
            _fmt("LD", "R0", f"@{i}"),
            _fmt("SCLR", "R1", "R0", f"#{_fval(0.1,10.0)}"),
            _fmt("ST", "R1", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_CMPR(count):
    pairs = []
    ops = [("#0", "equal"), ("#1", "not-equal"), ("#2", "less"),
           ("#3", "less-equal"), ("#4", "greater"), ("#5", "greater-equal")]
    for _ in range(count):
        syntax = pick_syntax()
        code, name = random.choice(ops)
        thresh = _fval(0, 100)
        i = _inp(); o = _out()
        q = f"Write NML to create a {name} comparison mask against {thresh}{syntax_tag(syntax)}"
        lines = [
            _fmt("LD", "R0", f"@{i}"),
            _fmt("CMPR", "R1", "R0", f"#{thresh}", code),
            _fmt("ST", "R1", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_SYNC(count):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        i = _inp(); o = _out()
        q = f"Write NML with a synchronization barrier between processing stages{syntax_tag(syntax)}"
        lines = [
            _fmt("LD", "R0", f"@{i}"),
            _fmt("SCLR", "R1", "R0", f"#{_fval(0.1,5.0)}"),
            "SYNC",
            _fmt("MADD", "R2", "R1", "R0"),
            _fmt("ST", "R2", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_TRAP(count):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        code = random.choice([1, 2, 3, 99])
        i = _inp(); o = _out()
        q = f"Write NML with input validation that TRAPs on invalid data{syntax_tag(syntax)}"
        lines = [
            _fmt("LD", "R0", f"@{i}"),
            _fmt("CMPI", "RE", "R0", "#0"),
            _fmt("JMPF", "#2"),
            _fmt("TRAP", f"#{code}"),
            _fmt("SCLR", "R1", "R0", f"#{_fval(0.01,1.0)}"),
            _fmt("ST", "R1", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_BNOT(count):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        i = _inp(); o = _out()
        q = f"Write NML to compute bitwise NOT of {i}{syntax_tag(syntax)}"
        lines = [
            _fmt("LD", "R0", f"@{i}"),
            _fmt("BNOT", "R1", "R0"),
            _fmt("ST", "R1", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_PADZ(count):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        i = _inp(); o = _out()
        q = f"Write NML to zero-pad {i}{syntax_tag(syntax)}"
        lines = [
            _fmt("LD", "R0", f"@{i}"),
            _fmt("PADZ", "R1", "R0"),
            _fmt("ST", "R1", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_RSHP(count):
    pairs = []
    shapes = ["#[4]", "#[2,2]", "#[1,8]", "#[8]", "#[3,3]", "#[1,16]", "#[4,4]"]
    for _ in range(count):
        syntax = pick_syntax()
        s = random.choice(shapes)
        i = _inp(); o = _out()
        q = f"Write NML to reshape {i} to {s.replace('#','')}{syntax_tag(syntax)}"
        lines = [
            _fmt("LD", "R0", f"@{i}"),
            _fmt("RSHP", "R1", "R0", s),
            _fmt("ST", "R1", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_PROJ(count):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        i = _inp(); o = _out()
        q = f"Write NML to project {i} through an embedding matrix{syntax_tag(syntax)}"
        lines = [
            _fmt("LD", "R0", f"@{i}"),
            _fmt("LD", "R1", "@projection_matrix"),
            _fmt("PROJ", "R2", "R0", "R1"),
            _fmt("ST", "R2", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_UPSC(count):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        i = _inp(); o = _out()
        q = f"Write NML to upscale {i}{syntax_tag(syntax)}"
        lines = [
            _fmt("LD", "R0", f"@{i}"),
            _fmt("UPSC", "R1", "R0"),
            _fmt("ST", "R1", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_TRNS(count):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        i = _inp(); o = _out()
        q = f"Write NML to transpose {i}{syntax_tag(syntax)}"
        lines = [
            _fmt("LD", "R0", f"@{i}"),
            _fmt("TRNS", "R1", "R0"),
            _fmt("ST", "R1", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_WHER(count):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        i = _inp(); i2 = _pick(INPUT_NAMES, 1, exclude=[i])[0]; o = _out()
        q = f"Write NML to conditionally select between {i} and {i2}{syntax_tag(syntax)}"
        lines = [
            _fmt("LD", "R0", "@condition"),
            _fmt("LD", "R1", f"@{i}"),
            _fmt("LD", "R2", f"@{i2}"),
            _fmt("WHER", "R3", "R0", "R1", "R2"),
            _fmt("ST", "R3", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_SIGN(count):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        i = _inp(); o = _out()
        agent = f"agent_{random.randint(1,100)}"
        q = f"Write a cryptographically signed NML program{syntax_tag(syntax)}"
        lines = [
            f'SIGN  agent={agent}  key=ed25519:k{_ival(1000,9999)}  sig=s{_ival(1000,9999)}',
            f'META  @name "signed_program"',
            _fmt("LD", "R0", f"@{i}"),
            _fmt("SCLR", "R1", "R0", f"#{_fval(0.1,10.0)}"),
            _fmt("ST", "R1", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_VRFY(count):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        i = _inp(); o = _out()
        q = f"Write NML that verifies a program signature before executing{syntax_tag(syntax)}"
        lines = [
            f'SIGN  agent=trusted_v1  key=ed25519:abc  sig=xyz',
            f'VRFY  @self  @trusted_v1',
            _fmt("LD", "R0", f"@{i}"),
            _fmt("SCLR", "R1", "R0", f"#{_fval(0.5,5.0)}"),
            _fmt("ST", "R1", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_ITOF(count):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        i = _inp(); o = _out()
        q = f"Write NML to convert {i} from integer to float{syntax_tag(syntax)}"
        lines = [
            _fmt("LD", "R0", f"@{i}"),
            _fmt("ITOF", "R1", "R0"),
            _fmt("ST", "R1", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_CLMP(count):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        lo = _fval(0, 50); hi = _fval(lo + 10, lo + 500)
        i = _inp(); o = _out()
        q = f"Write NML to clamp {i} to [{lo}, {hi}]{syntax_tag(syntax)}"
        lines = [
            _fmt("LD", "R0", f"@{i}"),
            _fmt("CLMP", "R1", "R0", f"#{lo}", f"#{hi}"),
            _fmt("ST", "R1", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_VOTE(count):
    pairs = []
    strategies = [("#0", "median"), ("#1", "mean"), ("#2", "quorum"), ("#3", "min"), ("#4", "max")]
    for _ in range(count):
        syntax = pick_syntax()
        code, name = random.choice(strategies)
        i = _inp(); o = _out()
        q = f"Write NML to compute {name} consensus of {i}{syntax_tag(syntax)}"
        lines = [
            _fmt("LD", "R0", f"@{i}"),
            _fmt("VOTE", "R1", "R0", code),
            _fmt("ST", "R1", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_EMBD(count):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        o = _out()
        q = f"Write NML to look up embeddings from a table{syntax_tag(syntax)}"
        lines = [
            _fmt("LD", "R0", "@token_ids"),
            _fmt("LD", "R1", "@embedding_table"),
            _fmt("EMBD", "R2", "R1", "R0"),
            _fmt("ST", "R2", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_FTOI(count):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        i = _inp(); o = _out()
        q = f"Write NML to convert {i} from float to integer{syntax_tag(syntax)}"
        lines = [
            _fmt("LD", "R0", f"@{i}"),
            _fmt("FTOI", "R1", "R0"),
            _fmt("ST", "R1", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_SOFT(count):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        i = _inp(); o = _out()
        q = f"Write NML to apply softmax to {i}{syntax_tag(syntax)}"
        lines = [
            _fmt("LD", "R0", f"@{i}"),
            _fmt("SOFT", "R1", "R0"),
            _fmt("ST", "R1", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_DIST(count):
    pairs = []
    metrics = [("#0", "cosine"), ("#1", "euclidean"), ("#2", "dot product")]
    for _ in range(count):
        syntax = pick_syntax()
        code, name = random.choice(metrics)
        a, b = _pick(INPUT_NAMES, 2); o = _out()
        q = f"Write NML to compute {name} distance between {a} and {b}{syntax_tag(syntax)}"
        lines = [
            _fmt("LD", "R0", f"@{a}"),
            _fmt("LD", "R1", f"@{b}"),
            _fmt("DIST", "R2", "R0", "R1", code),
            _fmt("ST", "R2", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_GATH(count):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        i = _inp(); o = _out()
        q = f"Write NML to gather elements from {i} by index{syntax_tag(syntax)}"
        lines = [
            _fmt("LD", "R0", f"@{i}"),
            _fmt("LD", "R1", "@indices"),
            _fmt("GATH", "R2", "R0", "R1"),
            _fmt("ST", "R2", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_FILT(count):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        i = _inp(); o = _out()
        q = f"Write NML to apply a FIR filter to {i}{syntax_tag(syntax)}"
        lines = [
            _fmt("LD", "R0", f"@{i}"),
            _fmt("LD", "R1", "@filter_coeffs"),
            _fmt("FILT", "R2", "R0", "R1"),
            _fmt("ST", "R2", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_LINK(count):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        i = _inp(); o = _out()
        q = f"Write NML using FRAG/LINK to compose reusable fragments{syntax_tag(syntax)}"
        lines = [
            _fmt("FRAG", "process"),
            _fmt("LD", "R0", f"@{i}"),
            _fmt("SCLR", "R1", "R0", f"#{_fval(0.1,10.0)}"),
            _fmt("ST", "R1", "@intermediate"),
            "ENDF",
            _fmt("FRAG", "main"),
            _fmt("LINK", "@process"),
            _fmt("LD", "R2", "@intermediate"),
            _fmt("ST", "R2", f"@{o}"),
            "HALT",
            "ENDF",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

# Reuse gen_LINK for FRAG and ENDF since they always appear together
gen_FRAG = gen_LINK
gen_ENDF = gen_LINK

def gen_TANH(count):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        i = _inp(); o = _out()
        q = f"Write NML to apply tanh activation to {i}{syntax_tag(syntax)}"
        lines = [
            _fmt("LD", "R0", f"@{i}"),
            _fmt("TANH", "R1", "R0"),
            _fmt("ST", "R1", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_SDIV(count):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        c = _fval(1, 100)
        i = _inp(); o = _out()
        q = f"Write NML to divide {i} by {c}{syntax_tag(syntax)}"
        lines = [
            _fmt("LD", "R0", f"@{i}"),
            _fmt("SDIV", "R1", "R0", f"#{c}"),
            _fmt("ST", "R1", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_FFT(count):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        i = _inp()
        q = f"Write NML to compute FFT of {i}{syntax_tag(syntax)}"
        lines = [
            _fmt("LD", "R0", f"@{i}"),
            _fmt("FFT", "R1", "R2", "R0"),
            _fmt("ST", "R1", "@real_part"),
            _fmt("ST", "R2", "@imag_part"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_ATTN(count):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        i = _inp(); o = _out()
        q = f"Write NML for attention on {i}{syntax_tag(syntax)}"
        lines = [
            _fmt("LD", "R0", f"@{i}"),
            _fmt("LD", "R1", "@keys"),
            _fmt("LD", "R2", "@values"),
            _fmt("ATTN", "R3", "R0", "R1", "R2"),
            _fmt("ST", "R3", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_POOL(count):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        i = _inp(); o = _out()
        q = f"Write NML to max-pool {i}{syntax_tag(syntax)}"
        lines = [
            _fmt("LD", "R0", f"@{i}"),
            _fmt("POOL", "R1", "R0"),
            _fmt("ST", "R1", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_EDIV(count):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        a, b = _pick(INPUT_NAMES, 2); o = _out()
        q = f"Write NML to element-wise divide {a} by {b}{syntax_tag(syntax)}"
        lines = [
            _fmt("LD", "R0", f"@{a}"),
            _fmt("LD", "R1", f"@{b}"),
            _fmt("EDIV", "R2", "R0", "R1"),
            _fmt("ST", "R2", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_SCAT(count):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        i = _inp(); o = _out()
        q = f"Write NML to scatter values from {i} by index{syntax_tag(syntax)}"
        lines = [
            _fmt("LD", "R0", f"@{i}"),
            _fmt("LD", "R1", "@indices"),
            _fmt("ALLC", "R2", "#[8]"),
            _fmt("SCAT", "R0", "R2", "R1"),
            _fmt("ST", "R2", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_CMPF(count):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        feat = random.randint(0, 7)
        thresh = _fval(0, 1000)
        i = _inp(); o = _out()
        q = f"Write NML tree comparison: check feature {feat} against {thresh}{syntax_tag(syntax)}"
        lines = [
            _fmt("LD", "R0", f"@{i}"),
            _fmt("CMPF", "RE", "R0", f"#{feat}", f"#{thresh}"),
            _fmt("JMPT", "#3"),
            _fmt("LEAF", "RC", f"#{_fval(1,100)}"),
            _fmt("JUMP", "#2"),
            _fmt("LEAF", "RC", f"#{_fval(1,100)}"),
            _fmt("TACC", "RA", "RA", "RC"),
            _fmt("ST", "RA", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_NORM(count):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        i = _inp(); o = _out()
        q = f"Write NML to layer-normalize {i}{syntax_tag(syntax)}"
        lines = [
            _fmt("LD", "R0", f"@{i}"),
            _fmt("NORM", "R1", "R0"),
            _fmt("ST", "R1", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_CONV(count):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        i = _inp(); o = _out()
        q = f"Write NML convolution on {i}{syntax_tag(syntax)}"
        lines = [
            _fmt("LD", "R0", f"@{i}"),
            _fmt("LD", "R1", "@kernel"),
            _fmt("CONV", "R2", "R0", "R1"),
            _fmt("ST", "R2", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_SIGM(count):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        i = _inp(); o = _out()
        q = f"Write NML to apply sigmoid to {i}{syntax_tag(syntax)}"
        lines = [
            _fmt("LD", "R0", f"@{i}"),
            _fmt("SIGM", "R1", "R0"),
            _fmt("ST", "R1", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_SDOT(count):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        a, b = _pick(INPUT_NAMES, 2); o = _out()
        q = f"Write NML dot product of {a} and {b}{syntax_tag(syntax)}"
        lines = [
            _fmt("LD", "R0", f"@{a}"),
            _fmt("LD", "R1", f"@{b}"),
            _fmt("SDOT", "R2", "R0", "R1"),
            _fmt("ST", "R2", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_MERG(count):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        a, b = _pick(INPUT_NAMES, 2); o = _out()
        dim = random.choice(["#0", "#1"])
        q = f"Write NML to concatenate {a} and {b} along dim {dim}{syntax_tag(syntax)}"
        lines = [
            _fmt("LD", "R0", f"@{a}"),
            _fmt("LD", "R1", f"@{b}"),
            _fmt("MERG", "R2", "R0", "R1", dim),
            _fmt("ST", "R2", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_META(count):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        i = _inp(); o = _out()
        desc = random.choice(["transform", "analysis", "filter", "projection", "classifier"])
        q = f"Write a self-describing NML program with META headers{syntax_tag(syntax)}"
        lines = [
            f'META  @name "{desc}"',
            f'META  @version "{random.randint(1,5)}.{random.randint(0,9)}"',
            f'META  @author "agent_{random.randint(1,50)}"',
            _fmt("LD", "R0", f"@{i}"),
            _fmt("SCLR", "R1", "R0", f"#{_fval(0.1,10.0)}"),
            _fmt("ST", "R1", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_SPLT(count):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        i = _inp()
        q = f"Write NML to split {i} along dimension 0{syntax_tag(syntax)}"
        lines = [
            _fmt("LD", "R0", f"@{i}"),
            _fmt("SPLT", "R1", "R2", "R0", "#0"),
            _fmt("ST", "R1", "@lower"),
            _fmt("ST", "R2", "@upper"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_CMP(count):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        a, b = _pick(INPUT_NAMES, 2); o = _out()
        q = f"Write NML to compare {a} and {b}{syntax_tag(syntax)}"
        lines = [
            _fmt("LD", "R0", f"@{a}"),
            _fmt("LD", "R1", f"@{b}"),
            _fmt("CMP", "R0", "R1"),
            _fmt("JMPT", "#3"),
            _fmt("MOV", "R2", "R0"),
            _fmt("JUMP", "#2"),
            _fmt("MOV", "R2", "R1"),
            _fmt("ST", "R2", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_GELU(count):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        i = _inp(); o = _out()
        q = f"Write NML to apply GELU activation to {i}{syntax_tag(syntax)}"
        lines = [
            _fmt("LD", "R0", f"@{i}"),
            _fmt("GELU", "R1", "R0"),
            _fmt("ST", "R1", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs


# ═══════════════════════════════════════════════════════════════════════════════
# Dispatch and main
# ═══════════════════════════════════════════════════════════════════════════════

GENERATORS = {
    "BKWD": gen_BKWD, "WUPD": gen_WUPD, "LOSS": gen_LOSS, "TRAIN": gen_TRAIN,
    "PTCH": gen_PTCH, "CMPR": gen_CMPR, "SYNC": gen_SYNC, "TRAP": gen_TRAP,
    "BNOT": gen_BNOT, "PADZ": gen_PADZ, "RSHP": gen_RSHP, "PROJ": gen_PROJ,
    "UPSC": gen_UPSC, "TRNS": gen_TRNS, "WHER": gen_WHER, "SIGN": gen_SIGN,
    "VRFY": gen_VRFY, "ITOF": gen_ITOF, "CLMP": gen_CLMP, "VOTE": gen_VOTE,
    "EMBD": gen_EMBD, "FTOI": gen_FTOI, "SOFT": gen_SOFT, "DIST": gen_DIST,
    "GATH": gen_GATH, "FILT": gen_FILT,
    "LINK": gen_LINK, "FRAG": gen_FRAG, "ENDF": gen_ENDF,
    "TANH": gen_TANH, "SDIV": gen_SDIV, "FFT": gen_FFT,
    "ATTN": gen_ATTN, "POOL": gen_POOL,
    "EDIV": gen_EDIV, "SCAT": gen_SCAT, "CMPF": gen_CMPF,
    "NORM": gen_NORM, "CONV": gen_CONV, "SIGM": gen_SIGM,
    "SDOT": gen_SDOT, "MERG": gen_MERG, "META": gen_META,
    "SPLT": gen_SPLT, "CMP": gen_CMP, "GELU": gen_GELU,
}

def main():
    parser = argparse.ArgumentParser(description="Generate NML equalization training pairs")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    args = parser.parse_args()

    print("Generating NML equalization pairs...")
    print(f"{'─' * 60}")

    all_pairs = []
    for opcode, target in sorted(ALL_TARGETS.items()):
        gen_fn = GENERATORS.get(opcode)
        if gen_fn is None:
            print(f"  WARNING: No generator for {opcode}, skipping")
            continue
        pairs = gen_fn(target)
        all_pairs.extend(pairs)
        print(f"  {opcode:<6}  {len(pairs):>6} pairs")

    random.shuffle(all_pairs)

    print(f"{'─' * 60}")
    print(f"  TOTAL:  {len(all_pairs):>6} pairs")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\nWritten to: {out_path}")


if __name__ == "__main__":
    main()
