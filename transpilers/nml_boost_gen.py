#!/usr/bin/env python3
"""
Generate ~25K uniformly distributed training pairs across ALL 82 NML opcodes.

Every opcode gets ~350 pairs with:
- Diverse prompt phrasings
- Random variable names
- Tri-syntax coverage (~60% classic, 25% symbolic, 15% verbose)
- Programs of varying complexity (3-20 instructions)
- Multi-opcode combo programs for realism

Output: domain/output/training/nml_boost_pairs.jsonl
"""

import json
import random
import argparse
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent.parent / "domain" / "output" / "training" / "nml_boost_pairs.jsonl"

random.seed(2027)

from nml_core_training_gen import (
    REGS_CLASSIC, REGS_GREEK, REGS_VERBOSE, SYM, VERBOSE,
    INPUT_NAMES, OUTPUT_NAMES, MEM_NAMES,
    _pick, _fval, _ival, _fmt, _pair,
    apply_syntax, pick_syntax, syntax_tag, _inp, _out, _mem,
    to_symbolic, to_verbose,
)

PAIRS_PER_OPCODE = 350

GRAD_REGS = ["RG", "RH", "RI"]
EXTENDED_REGS = ["RG","RH","RI","RJ","RK","RL","RM","RN","RO","RP","RQ","RR","RS","RT","RU","RV"]

def _r(exclude=()):
    pool = [r for r in REGS_CLASSIC[:16] if r not in exclude]
    return random.choice(pool)

def _rext():
    return random.choice(REGS_CLASSIC)

def _gr():
    return random.choice(GRAD_REGS)


# ═══════════════════════════════════════════════════════════════════════════════
# Generators for the 25 opcodes NOT in nml_equalize_gen.py
# ═══════════════════════════════════════════════════════════════════════════════

def gen_MMUL(count):
    pairs = []
    prompts = [
        "Write NML to multiply matrices {a} and {b}",
        "Compute matrix product of {a} and {b} in NML",
        "Write NML for a linear layer: output = {a} @ {b}",
        "Use MMUL to project {a} through {b}",
        "Matrix multiply {a} by {b} and store the result",
    ]
    for _ in range(count):
        syntax = pick_syntax()
        a, b = _pick(INPUT_NAMES, 2); o = _out()
        q = random.choice(prompts).format(a=a, b=b) + syntax_tag(syntax)
        lines = [
            _fmt("LD", "R0", f"@{a}"),
            _fmt("LD", "R1", f"@{b}"),
            _fmt("MMUL", "R2", "R0", "R1"),
            _fmt("ST", "R2", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_MADD(count):
    pairs = []
    prompts = [
        "Write NML to element-wise add {a} and {b}",
        "Add tensors {a} and {b} in NML",
        "Write NML to compute {a} + {b}",
        "Use MADD to sum {a} and {b}",
        "Element-wise addition of {a} and {b}",
    ]
    for _ in range(count):
        syntax = pick_syntax()
        a, b = _pick(INPUT_NAMES, 2); o = _out()
        q = random.choice(prompts).format(a=a, b=b) + syntax_tag(syntax)
        lines = [
            _fmt("LD", "R0", f"@{a}"),
            _fmt("LD", "R1", f"@{b}"),
            _fmt("MADD", "R2", "R0", "R1"),
            _fmt("ST", "R2", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_MSUB(count):
    pairs = []
    prompts = [
        "Write NML to subtract {b} from {a}",
        "Compute {a} - {b} in NML",
        "Use MSUB to find the difference between {a} and {b}",
        "Element-wise subtraction: {a} minus {b}",
    ]
    for _ in range(count):
        syntax = pick_syntax()
        a, b = _pick(INPUT_NAMES, 2); o = _out()
        q = random.choice(prompts).format(a=a, b=b) + syntax_tag(syntax)
        lines = [
            _fmt("LD", "R0", f"@{a}"),
            _fmt("LD", "R1", f"@{b}"),
            _fmt("MSUB", "R2", "R0", "R1"),
            _fmt("ST", "R2", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_EMUL(count):
    pairs = []
    prompts = [
        "Write NML to element-wise multiply {a} and {b}",
        "Hadamard product of {a} and {b} in NML",
        "Multiply {a} and {b} element-by-element",
        "Use EMUL for element-wise product of {a} and {b}",
    ]
    for _ in range(count):
        syntax = pick_syntax()
        a, b = _pick(INPUT_NAMES, 2); o = _out()
        q = random.choice(prompts).format(a=a, b=b) + syntax_tag(syntax)
        lines = [
            _fmt("LD", "R0", f"@{a}"),
            _fmt("LD", "R1", f"@{b}"),
            _fmt("EMUL", "R2", "R0", "R1"),
            _fmt("ST", "R2", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_SCLR(count):
    pairs = []
    prompts = [
        "Write NML to scale {a} by {v}",
        "Multiply every element of {a} by {v}",
        "Scalar multiplication: {a} * {v}",
        "Use SCLR to scale {a} by a factor of {v}",
    ]
    for _ in range(count):
        syntax = pick_syntax()
        a = _inp(); o = _out()
        v = _fval(0.01, 50.0)
        q = random.choice(prompts).format(a=a, v=v) + syntax_tag(syntax)
        lines = [
            _fmt("LD", "R0", f"@{a}"),
            _fmt("SCLR", "R1", "R0", f"#{v}"),
            _fmt("ST", "R1", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_LD(count):
    pairs = []
    prompts = [
        "Write NML to load {a} from memory into a register",
        "Load {a} and {b} and store their sum",
        "Read {a} from memory, scale by {v}, and save",
        "Load {a}, apply ReLU, store result",
    ]
    for _ in range(count):
        syntax = pick_syntax()
        a = _inp(); b = _inp(); o = _out()
        v = _fval(0.1, 10.0)
        variant = random.randint(0, 3)
        if variant == 0:
            q = f"Write NML to load {a} from memory into a register{syntax_tag(syntax)}"
            lines = [
                _fmt("LD", "R0", f"@{a}"),
                _fmt("ST", "R0", f"@{o}"),
                "HALT",
            ]
        elif variant == 1:
            q = f"Write NML to load {a} and {b} and store their sum{syntax_tag(syntax)}"
            lines = [
                _fmt("LD", "R0", f"@{a}"),
                _fmt("LD", "R1", f"@{b}"),
                _fmt("MADD", "R2", "R0", "R1"),
                _fmt("ST", "R2", f"@{o}"),
                "HALT",
            ]
        elif variant == 2:
            q = f"Write NML to load {a}, scale by {v}, and save{syntax_tag(syntax)}"
            lines = [
                _fmt("LD", "R0", f"@{a}"),
                _fmt("SCLR", "R1", "R0", f"#{v}"),
                _fmt("ST", "R1", f"@{o}"),
                "HALT",
            ]
        else:
            q = f"Write NML to load {a}, apply ReLU, and store{syntax_tag(syntax)}"
            lines = [
                _fmt("LD", "R0", f"@{a}"),
                _fmt("RELU", "R1", "R0"),
                _fmt("ST", "R1", f"@{o}"),
                "HALT",
            ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_ST(count):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        a = _inp(); o = _out()
        v = _fval(0.1, 50.0)
        q = f"Write NML to load {a}, transform it, and store to {o}{syntax_tag(syntax)}"
        lines = [
            _fmt("LD", "R0", f"@{a}"),
            _fmt("SCLR", "R1", "R0", f"#{v}"),
            _fmt("ST", "R1", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_MOV(count):
    pairs = []
    prompts = [
        "Write NML to copy register R0 into R1",
        "Move {a} to a different register before overwriting",
        "Use MOV to preserve {a} while processing",
        "Copy data between registers in NML",
    ]
    for _ in range(count):
        syntax = pick_syntax()
        a = _inp(); o = _out()
        r_src = _r(); r_dst = _r(exclude=[r_src])
        q = random.choice(prompts).format(a=a) + syntax_tag(syntax)
        lines = [
            _fmt("LD", r_src, f"@{a}"),
            _fmt("MOV", r_dst, r_src),
            _fmt("SCLR", r_src, r_src, f"#{_fval(0.1, 5.0)}"),
            _fmt("MADD", "RA", r_src, r_dst),
            _fmt("ST", "RA", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_ALLC(count):
    pairs = []
    shapes = ["#[1]", "#[4]", "#[8]", "#[2,2]", "#[3,3]", "#[1,8]", "#[4,4]", "#[16]"]
    prompts = [
        "Write NML to allocate a zero tensor of shape {s}",
        "Initialize an accumulator of shape {s} with ALLC",
        "Allocate a {s} buffer and accumulate values into it",
    ]
    for _ in range(count):
        syntax = pick_syntax()
        s = random.choice(shapes)
        i = _inp(); o = _out()
        q = random.choice(prompts).format(s=s.replace('#','')) + syntax_tag(syntax)
        lines = [
            _fmt("ALLC", "RA", s),
            _fmt("LD", "R0", f"@{i}"),
            _fmt("TACC", "RA", "RA", "R0"),
            _fmt("ST", "RA", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_RELU(count):
    pairs = []
    prompts = [
        "Write NML to apply ReLU activation to {a}",
        "Apply ReLU to the output of a linear layer on {a}",
        "Use ReLU to zero out negative values in {a}",
        "Write NML for a dense layer with ReLU on {a}",
    ]
    for _ in range(count):
        syntax = pick_syntax()
        a = _inp(); o = _out()
        variant = random.randint(0, 1)
        q = random.choice(prompts).format(a=a) + syntax_tag(syntax)
        if variant == 0:
            lines = [
                _fmt("LD", "R0", f"@{a}"),
                _fmt("RELU", "R1", "R0"),
                _fmt("ST", "R1", f"@{o}"),
                "HALT",
            ]
        else:
            lines = [
                _fmt("LD", "R0", f"@{a}"),
                _fmt("LD", "R1", "@weights"),
                _fmt("MMUL", "R2", "R0", "R1"),
                _fmt("LD", "R3", "@bias"),
                _fmt("MADD", "R2", "R2", "R3"),
                _fmt("RELU", "R2", "R2"),
                _fmt("ST", "R2", f"@{o}"),
                "HALT",
            ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_LEAF(count):
    pairs = []
    prompts = [
        "Write NML to load the constant {v} into a register",
        "Use LEAF to set a register to {v}",
        "Initialize a register with the immediate value {v}",
        "Write NML that loads constant {v} and stores it",
    ]
    for _ in range(count):
        syntax = pick_syntax()
        v = _fval(0.01, 10000.0)
        o = _out()
        q = random.choice(prompts).format(v=v) + syntax_tag(syntax)
        lines = [
            _fmt("LEAF", "R0", f"#{v}"),
            _fmt("ST", "R0", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_TACC(count):
    pairs = []
    prompts = [
        "Write NML to accumulate {a} into a running total",
        "Use TACC to sum values from {a}",
        "Accumulate {a} and a constant using TACC",
        "Write NML to add {a} to the accumulator register",
    ]
    for _ in range(count):
        syntax = pick_syntax()
        a = _inp(); o = _out()
        v = _fval(1.0, 500.0)
        q = random.choice(prompts).format(a=a) + syntax_tag(syntax)
        lines = [
            _fmt("ALLC", "RA", "#[1]"),
            _fmt("LD", "R0", f"@{a}"),
            _fmt("TACC", "RA", "RA", "R0"),
            _fmt("LEAF", "RC", f"#{v}"),
            _fmt("TACC", "RA", "RA", "RC"),
            _fmt("ST", "RA", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_CMPI(count):
    pairs = []
    prompts = [
        "Write NML to compare {a} against {v} and branch",
        "Use CMPI to check if {a} exceeds {v}",
        "Write NML conditional: if {a} < {v} then scale by 0.1 else scale by 0.2",
        "Compare {a} to threshold {v} and take different actions",
    ]
    for _ in range(count):
        syntax = pick_syntax()
        a = _inp(); o = _out()
        v = _fval(10.0, 5000.0)
        q = random.choice(prompts).format(a=a, v=v) + syntax_tag(syntax)
        lines = [
            _fmt("LD", "R0", f"@{a}"),
            _fmt("CMPI", "RE", "R0", f"#{v}"),
            _fmt("JMPF", "#3"),
            _fmt("SCLR", "R1", "R0", f"#{_fval(0.01, 0.5)}"),
            _fmt("JUMP", "#2"),
            _fmt("SCLR", "R1", "R0", f"#{_fval(0.01, 0.5)}"),
            _fmt("ST", "R1", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_JMPT(count):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        a = _inp(); o = _out()
        v = _fval(10.0, 1000.0)
        q = f"Write NML using JMPT to branch when {a} < {v}{syntax_tag(syntax)}"
        lines = [
            _fmt("LD", "R0", f"@{a}"),
            _fmt("CMPI", "RE", "R0", f"#{v}"),
            _fmt("JMPT", "#2"),
            _fmt("LEAF", "R1", f"#{_fval(1,100)}"),
            _fmt("JUMP", "#1"),
            _fmt("LEAF", "R1", f"#{_fval(1,100)}"),
            _fmt("ST", "R1", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_JMPF(count):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        a = _inp(); o = _out()
        v = _fval(10.0, 1000.0)
        q = f"Write NML using JMPF to skip code when condition is false{syntax_tag(syntax)}"
        lines = [
            _fmt("LD", "R0", f"@{a}"),
            _fmt("CMPI", "RE", "R0", f"#{v}"),
            _fmt("JMPF", "#2"),
            _fmt("SCLR", "R1", "R0", f"#{_fval(0.01, 1.0)}"),
            _fmt("JUMP", "#1"),
            _fmt("SCLR", "R1", "R0", f"#{_fval(0.01, 1.0)}"),
            _fmt("ST", "R1", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_JUMP(count):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        a = _inp(); o = _out()
        q = f"Write NML with unconditional JUMP to skip over a code block{syntax_tag(syntax)}"
        lines = [
            _fmt("LD", "R0", f"@{a}"),
            _fmt("JUMP", "#2"),
            _fmt("LEAF", "R0", "#0"),
            _fmt("SCLR", "R1", "R0", f"#{_fval(0.1, 5.0)}"),
            _fmt("ST", "R1", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_LOOP(count):
    pairs = []
    prompts = [
        "Write NML to loop {n} times and accumulate values from {a}",
        "Use LOOP/ENDP to iterate {n} times over {a}",
        "Write a counted loop in NML that runs {n} iterations",
        "Loop {n} times, accumulating scaled {a} each iteration",
    ]
    for _ in range(count):
        syntax = pick_syntax()
        n = random.choice([3, 5, 8, 10, 20, 50, 100])
        a = _inp(); o = _out()
        q = random.choice(prompts).format(n=n, a=a) + syntax_tag(syntax)
        lines = [
            _fmt("LD", "R0", f"@{a}"),
            _fmt("ALLC", "R1", "#[1]"),
            _fmt("LEAF", "R2", "#1"),
            _fmt("LOOP", "R0"),
            _fmt("TACC", "R1", "R1", "R2"),
            "ENDP",
            _fmt("ST", "R1", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_ENDP(count):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        a = _inp(); o = _out()
        n = random.choice([5, 10, 20])
        v = _fval(0.5, 5.0)
        q = f"Write NML with a LOOP/ENDP block that scales {a} by {v} across {n} iterations{syntax_tag(syntax)}"
        lines = [
            _fmt("LEAF", "RD", f"#{n}"),
            _fmt("LD", "R0", f"@{a}"),
            _fmt("ALLC", "RA", "#[1]"),
            _fmt("LEAF", "RC", f"#{v}"),
            _fmt("LOOP", "RD"),
            _fmt("EMUL", "R0", "R0", "RC"),
            _fmt("TACC", "RA", "RA", "R0"),
            "ENDP",
            _fmt("ST", "RA", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_CALL(count):
    pairs = []
    prompts = [
        "Write NML with a subroutine that doubles R0 using CALL/RET",
        "Use CALL to invoke a subroutine that scales by {v}",
        "Write NML with a CALL #2 subroutine pattern",
        "Create an NML program with a reusable subroutine via CALL/RET",
    ]
    for _ in range(count):
        syntax = pick_syntax()
        a = _inp(); o = _out()
        v = _fval(0.5, 10.0)
        q = random.choice(prompts).format(v=v) + syntax_tag(syntax)
        lines = [
            _fmt("LD", "R0", f"@{a}"),
            _fmt("CALL", "#2"),
            _fmt("ST", "R0", f"@{o}"),
            "HALT",
            _fmt("SCLR", "R0", "R0", f"#{v}"),
            "RET",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_RET(count):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        a = _inp(); o = _out()
        act = random.choice(["RELU", "SIGM", "TANH"])
        q = f"Write NML with a subroutine that applies {act} and returns{syntax_tag(syntax)}"
        lines = [
            _fmt("LD", "R0", f"@{a}"),
            _fmt("CALL", "#2"),
            _fmt("ST", "R0", f"@{o}"),
            "HALT",
            _fmt(act, "R0", "R0"),
            "RET",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_HALT(count):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        a = _inp(); o = _out()
        v = _fval(0.1, 10.0)
        q = f"Write a complete NML program that loads {a}, scales by {v}, stores, and halts{syntax_tag(syntax)}"
        lines = [
            _fmt("LD", "R0", f"@{a}"),
            _fmt("SCLR", "R1", "R0", f"#{v}"),
            _fmt("ST", "R1", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_RDUC(count):
    pairs = []
    ops = [("#0", "sum"), ("#1", "mean"), ("#2", "max"), ("#3", "min")]
    for _ in range(count):
        syntax = pick_syntax()
        code, name = random.choice(ops)
        a = _inp(); o = _out()
        q = f"Write NML to reduce {a} with {name}{syntax_tag(syntax)}"
        lines = [
            _fmt("LD", "R0", f"@{a}"),
            _fmt("RDUC", "R1", "R0", code),
            _fmt("ST", "R1", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_SYS(count):
    pairs = []
    sys_ops = [("#0", "print"), ("#1", "read"), ("#2", "timestamp")]
    for _ in range(count):
        syntax = pick_syntax()
        code, name = random.choice(sys_ops)
        a = _inp(); o = _out()
        q = f"Write NML to perform a {name} system call{syntax_tag(syntax)}"
        lines = [
            _fmt("LD", "R0", f"@{a}"),
            _fmt("SYS", code, "R0"),
            _fmt("ST", "R0", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_MOD(count):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        a = _inp(); o = _out()
        m = random.choice([2, 3, 5, 7, 10, 16, 100])
        q = f"Write NML to compute {a} modulo {m}{syntax_tag(syntax)}"
        lines = [
            _fmt("LD", "R0", f"@{a}"),
            _fmt("MOD", "R1", "R0", f"#{m}"),
            _fmt("ST", "R1", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs

def gen_SCTR(count):
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        a = _inp(); o = _out()
        q = f"Write NML to scatter {a} using SCTR (dest-first order){syntax_tag(syntax)}"
        lines = [
            _fmt("LD", "R0", f"@{a}"),
            _fmt("LD", "R1", "@indices"),
            _fmt("ALLC", "R2", "#[8]"),
            _fmt("SCTR", "R2", "R0", "R1"),
            _fmt("ST", "R2", f"@{o}"),
            "HALT",
        ]
        pairs.append(_pair(q, apply_syntax(lines, syntax)))
    return pairs


# ═══════════════════════════════════════════════════════════════════════════════
# Import existing generators from nml_equalize_gen for the 46 opcodes it covers
# ═══════════════════════════════════════════════════════════════════════════════

from nml_equalize_gen import (
    gen_BKWD, gen_WUPD, gen_LOSS, gen_TNET,
    gen_PTCH, gen_CMPR, gen_SYNC, gen_TRAP, gen_BNOT,
    gen_PADZ, gen_RSHP, gen_PROJ, gen_UPSC, gen_TRNS,
    gen_WHER, gen_SIGN, gen_VRFY, gen_ITOF, gen_CLMP,
    gen_VOTE, gen_EMBD, gen_FTOI, gen_SOFT, gen_DIST,
    gen_GATH, gen_FILT,
    gen_LINK, gen_FRAG, gen_ENDF,
    gen_TANH, gen_SDIV, gen_FFT,
    gen_ATTN, gen_POOL,
    gen_EDIV, gen_SCAT, gen_CMPF,
    gen_NORM, gen_CONV, gen_SIGM,
    gen_SDOT, gen_MERG, gen_META,
    gen_SPLT, gen_CMP, gen_GELU,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-opcode combo generators (realistic programs using 4-8 opcodes together)
# ═══════════════════════════════════════════════════════════════════════════════

def gen_combos(count):
    pairs = []
    combo_generators = [
        _combo_nn_layer,
        _combo_attention_block,
        _combo_training_step,
        _combo_conditional_pipeline,
        _combo_loop_accumulate,
        _combo_vision_pipeline,
        _combo_signal_pipeline,
        _combo_m2m_fragment,
    ]
    for _ in range(count):
        gen_fn = random.choice(combo_generators)
        pairs.append(gen_fn())
    return pairs

def _combo_nn_layer():
    syntax = pick_syntax()
    i = _inp(); o = _out()
    act = random.choice(["RELU", "SIGM", "TANH", "GELU"])
    q = f"Write NML for a dense neural network layer with {act} activation on {i}{syntax_tag(syntax)}"
    lines = [
        _fmt("LD", "R0", f"@{i}"),
        _fmt("LD", "R1", "@weights"),
        _fmt("LD", "R2", "@bias"),
        _fmt("MMUL", "R3", "R0", "R1"),
        _fmt("MADD", "R3", "R3", "R2"),
        _fmt(act, "R3", "R3"),
        _fmt("ST", "R3", f"@{o}"),
        "HALT",
    ]
    return _pair(q, apply_syntax(lines, syntax))

def _combo_attention_block():
    syntax = pick_syntax()
    i = _inp(); o = _out()
    q = f"Write NML for a full transformer block on {i}: attention, residual, norm, FFN{syntax_tag(syntax)}"
    lines = [
        _fmt("LD", "R0", f"@{i}"),
        _fmt("LD", "R1", "@wq"), _fmt("LD", "R2", "@wk"), _fmt("LD", "R3", "@wv"),
        _fmt("MMUL", "R4", "R0", "R1"),
        _fmt("MMUL", "R5", "R0", "R2"),
        _fmt("MMUL", "R6", "R0", "R3"),
        _fmt("ATTN", "R7", "R4", "R5", "R6"),
        _fmt("MADD", "R7", "R7", "R0"),
        _fmt("NORM", "R7", "R7"),
        _fmt("LD", "R8", "@ff_w1"),
        _fmt("MMUL", "R9", "R7", "R8"),
        _fmt("GELU", "R9", "R9"),
        _fmt("LD", "RA", "@ff_w2"),
        _fmt("MMUL", "R9", "R9", "RA"),
        _fmt("MADD", "R9", "R9", "R7"),
        _fmt("NORM", "R9", "R9"),
        _fmt("ST", "R9", f"@{o}"),
        "HALT",
    ]
    return _pair(q, apply_syntax(lines, syntax))

def _combo_training_step():
    syntax = pick_syntax()
    q = f"Write NML for a forward pass, loss computation, backprop, and weight update{syntax_tag(syntax)}"
    lines = [
        _fmt("LD", "R0", "@training_data"),
        _fmt("LD", "R1", "@weights"),
        _fmt("LD", "R2", "@bias"),
        _fmt("MMUL", "R3", "R0", "R1"),
        _fmt("MADD", "R3", "R3", "R2"),
        _fmt("RELU", "R3", "R3"),
        _fmt("LD", "R4", "@targets"),
        _fmt("LOSS", _gr(), "R3", "R4", "#0"),
        _fmt("BKWD", _gr(), "R3", "R4"),
        _fmt("LEAF", "RJ", "#0.01"),
        _fmt("WUPD", "R1", "R1", _gr()),
        _fmt("ST", "R1", "@updated_weights"),
        "HALT",
    ]
    return _pair(q, apply_syntax(lines, syntax))

def _combo_conditional_pipeline():
    syntax = pick_syntax()
    i = _inp(); o = _out()
    v = _fval(100.0, 10000.0)
    q = f"Write NML that branches on whether {i} exceeds {v} and applies different scales{syntax_tag(syntax)}"
    lines = [
        _fmt("LD", "R0", f"@{i}"),
        _fmt("ALLC", "RA", "#[1]"),
        _fmt("CMPI", "RE", "R0", f"#{v}"),
        _fmt("JMPF", "#3"),
        _fmt("SCLR", "RC", "R0", f"#{_fval(0.01, 0.2)}"),
        _fmt("JUMP", "#2"),
        _fmt("SCLR", "RC", "R0", f"#{_fval(0.2, 0.5)}"),
        _fmt("TACC", "RA", "RA", "RC"),
        _fmt("ST", "RA", f"@{o}"),
        "HALT",
    ]
    return _pair(q, apply_syntax(lines, syntax))

def _combo_loop_accumulate():
    syntax = pick_syntax()
    n = random.choice([5, 10, 20, 50])
    i = _inp(); o = _out()
    q = f"Write NML that loops {n} times accumulating scaled {i}{syntax_tag(syntax)}"
    lines = [
        _fmt("LD", "R0", f"@{i}"),
        _fmt("ALLC", "RA", "#[1]"),
        _fmt("LEAF", "RC", f"#{_fval(0.01, 1.0)}"),
        _fmt("LEAF", "RD", f"#{n}"),
        _fmt("LOOP", "RD"),
        _fmt("EMUL", "R1", "R0", "RC"),
        _fmt("TACC", "RA", "RA", "R1"),
        "ENDP",
        _fmt("ST", "RA", f"@{o}"),
        "HALT",
    ]
    return _pair(q, apply_syntax(lines, syntax))

def _combo_vision_pipeline():
    syntax = pick_syntax()
    i = _inp(); o = _out()
    q = f"Write NML for a vision pipeline: conv, relu, pool on {i}{syntax_tag(syntax)}"
    lines = [
        _fmt("LD", "R0", f"@{i}"),
        _fmt("LD", "R1", "@kernel"),
        _fmt("CONV", "R2", "R0", "R1"),
        _fmt("RELU", "R2", "R2"),
        _fmt("POOL", "R3", "R2"),
        _fmt("ST", "R3", f"@{o}"),
        "HALT",
    ]
    return _pair(q, apply_syntax(lines, syntax))

def _combo_signal_pipeline():
    syntax = pick_syntax()
    i = _inp(); o = _out()
    q = f"Write NML to FFT {i}, filter in frequency domain, and store{syntax_tag(syntax)}"
    lines = [
        _fmt("LD", "R0", f"@{i}"),
        _fmt("FFT", "R1", "R2", "R0"),
        _fmt("LD", "R3", "@filter_coeffs"),
        _fmt("FILT", "R4", "R1", "R3"),
        _fmt("ST", "R4", f"@{o}"),
        "HALT",
    ]
    return _pair(q, apply_syntax(lines, syntax))

def _combo_m2m_fragment():
    syntax = pick_syntax()
    i = _inp(); o = _out()
    q = f"Write NML with META headers and FRAG/LINK composition{syntax_tag(syntax)}"
    lines = [
        f'META  @name "composed_{random.randint(1,99)}"',
        f'META  @version "1.{random.randint(0,9)}"',
        _fmt("FRAG", "compute"),
        _fmt("LD", "R0", f"@{i}"),
        _fmt("SCLR", "R1", "R0", f"#{_fval(0.1,10.0)}"),
        _fmt("ST", "R1", "@intermediate"),
        "ENDF",
        _fmt("FRAG", "main"),
        _fmt("LINK", "@compute"),
        _fmt("LD", "R2", "@intermediate"),
        _fmt("ST", "R2", f"@{o}"),
        "HALT",
        "ENDF",
    ]
    return _pair(q, apply_syntax(lines, syntax))


# ═══════════════════════════════════════════════════════════════════════════════
# All 82 opcodes + combos
# ═══════════════════════════════════════════════════════════════════════════════

ALL_GENERATORS = {
    # Core arithmetic (8)
    "MMUL": gen_MMUL, "MADD": gen_MADD, "MSUB": gen_MSUB, "EMUL": gen_EMUL,
    "EDIV": gen_EDIV, "SDOT": gen_SDOT, "SCLR": gen_SCLR, "SDIV": gen_SDIV,
    # Activations (5)
    "RELU": gen_RELU, "SIGM": gen_SIGM, "TANH": gen_TANH, "SOFT": gen_SOFT, "GELU": gen_GELU,
    # Memory (4)
    "LD": gen_LD, "ST": gen_ST, "MOV": gen_MOV, "ALLC": gen_ALLC,
    # Data flow (4)
    "RSHP": gen_RSHP, "TRNS": gen_TRNS, "SPLT": gen_SPLT, "MERG": gen_MERG,
    # Comparison (3)
    "CMPF": gen_CMPF, "CMP": gen_CMP, "CMPI": gen_CMPI,
    # Control (5)
    "JMPT": gen_JMPT, "JMPF": gen_JMPF, "JUMP": gen_JUMP, "LOOP": gen_LOOP, "ENDP": gen_ENDP,
    # Subroutine (2)
    "CALL": gen_CALL, "RET": gen_RET,
    # Tree (2)
    "LEAF": gen_LEAF, "TACC": gen_TACC,
    # System (3)
    "SYNC": gen_SYNC, "HALT": gen_HALT, "TRAP": gen_TRAP,
    # Vision (4)
    "CONV": gen_CONV, "POOL": gen_POOL, "UPSC": gen_UPSC, "PADZ": gen_PADZ,
    # Transformer (4)
    "ATTN": gen_ATTN, "NORM": gen_NORM, "EMBD": gen_EMBD, "GELU": gen_GELU,
    # Reduction (4)
    "RDUC": gen_RDUC, "WHER": gen_WHER, "CLMP": gen_CLMP, "CMPR": gen_CMPR,
    # Signal (2)
    "FFT": gen_FFT, "FILT": gen_FILT,
    # M2M (13)
    "META": gen_META, "FRAG": gen_FRAG, "ENDF": gen_ENDF, "LINK": gen_LINK,
    "PTCH": gen_PTCH, "SIGN": gen_SIGN, "VRFY": gen_VRFY, "VOTE": gen_VOTE,
    "PROJ": gen_PROJ, "DIST": gen_DIST, "GATH": gen_GATH, "SCAT": gen_SCAT, "SCTR": gen_SCTR,
    # Training (4)
    "BKWD": gen_BKWD, "WUPD": gen_WUPD, "LOSS": gen_LOSS, "TNET": gen_TNET,
    # General (5)
    "SYS": gen_SYS, "MOD": gen_MOD, "ITOF": gen_ITOF, "FTOI": gen_FTOI, "BNOT": gen_BNOT,
}


def main():
    parser = argparse.ArgumentParser(description="Generate uniform NML training boost across all 82 opcodes")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    parser.add_argument("--per-opcode", type=int, default=PAIRS_PER_OPCODE,
                        help=f"Pairs per opcode (default: {PAIRS_PER_OPCODE})")
    parser.add_argument("--combos", type=int, default=1500,
                        help="Multi-opcode combo programs (default: 1500)")
    args = parser.parse_args()

    print("=" * 60)
    print("  NML Boost Generator — All 82 Opcodes")
    print("=" * 60)
    print(f"  Per-opcode: {args.per_opcode}")
    print(f"  Combos:     {args.combos}")
    print(f"{'─' * 60}")

    all_pairs = []
    seen_opcodes = set()

    for opcode in sorted(ALL_GENERATORS.keys()):
        if opcode in seen_opcodes:
            continue
        seen_opcodes.add(opcode)
        gen_fn = ALL_GENERATORS[opcode]
        pairs = gen_fn(args.per_opcode)
        all_pairs.extend(pairs)
        print(f"  {opcode:<6}  {len(pairs):>5} pairs")

    combo_pairs = gen_combos(args.combos)
    all_pairs.extend(combo_pairs)
    print(f"  {'COMBO':<6}  {len(combo_pairs):>5} pairs")

    random.shuffle(all_pairs)

    print(f"{'─' * 60}")
    print(f"  TOTAL:  {len(all_pairs):>6} pairs  ({len(seen_opcodes)} opcodes + combos)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\n  Written to: {out_path}")


if __name__ == "__main__":
    main()
