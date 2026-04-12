#!/usr/bin/env python3
"""
Generate 10,000 general-purpose NML training pairs.

Teaches the model that NML is a GENERAL computation language, not just
a tax calculator.  Covers arithmetic, loops, conditionals, subroutines,
neural networks, finance, data science, and M2M / agent patterns.

Output: ../output/training/general_purpose_pairs.jsonl
Format: {"messages": [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}

Usage:
    python3 general_purpose_training_gen.py
    python3 general_purpose_training_gen.py --limit 500
"""

import json
import random
import argparse
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent.parent / "output" / "training" / "general_purpose_pairs.jsonl"

random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════════
# Domain-neutral name pools
# ═══════════════════════════════════════════════════════════════════════════════

INPUT_NAMES = [
    "value", "amount", "price", "quantity", "rate", "factor", "input",
    "data", "weights", "bias", "features", "scores", "signal", "readings",
    "samples", "matrix_a", "matrix_b", "vector", "scalar", "principal",
    "interest_rate", "months", "years", "balance", "payment", "cost",
    "revenue", "profit", "loss", "temperature", "pressure", "velocity",
    "distance", "time", "mass", "energy", "force", "voltage", "current",
    "resistance", "height", "width", "depth", "radius", "angle",
    "frequency", "amplitude", "threshold", "limit", "capacity",
]

OUTPUT_NAMES = [
    "result", "output", "total", "answer", "computed", "final_value",
    "sum", "product", "quotient", "difference", "scaled", "accumulated",
    "prediction", "estimate", "score", "classification", "probability",
    "mean_val", "max_val", "min_val", "normalized", "projected",
    "filtered", "transformed", "response", "measurement",
]

# Classic numbered registers
REGS_NUMBERED = ["R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9",
                 "RA", "RB", "RC", "RD", "RE", "RF"]
# Greek registers (symbolic syntax)
REGS_GREEK = ["ι", "κ", "λ", "μ", "ν", "ξ", "ο", "π", "ρ", "ς",
              "α", "β", "γ", "δ", "φ", "ψ"]

# Classic↔Symbolic opcode mapping
SYM = {
    "LD": "↓", "ST": "↑", "MOV": "←", "ALLC": "□",
    "MMUL": "×", "MADD": "⊕", "MSUB": "⊖", "EMUL": "⊗",
    "EDIV": "⊘", "SDOT": "·", "SCLR": "∗", "SDIV": "÷",
    "LEAF": "∎", "TACC": "∑",
    "RELU": "⌐", "SIGM": "σ", "TANH": "τ", "SOFT": "Σ", "GELU": "ℊ",
    "RSHP": "⊟", "TRNS": "⊤",
    "SPLT": "⊢", "MERG": "⊣",
    "CMPF": "⋈", "CMP": "≶", "CMPI": "≺",
    "JMPT": "↗", "JMPF": "↘", "JUMP": "→",
    "LOOP": "↻", "ENDP": "↺",
    "CALL": "⇒", "RET": "⇐",
    "SYNC": "⊥", "HALT": "◼", "TRAP": "⚠",
    "CONV": "⊛", "POOL": "⊓", "UPSC": "⊔", "PADZ": "⊡",
    "ATTN": "⊙", "NORM": "‖", "EMBD": "⊏",
    "RDUC": "⊥", "WHER": "⊻", "CLMP": "⊧", "CMPR": "⊜",
    "FFT": "∿", "FILT": "⋐",
    "META": "§", "FRAG": "◆", "ENDF": "◇", "LINK": "⊿",
    "PTCH": "⊿", "SIGN": "✦", "VRFY": "✓",
    "VOTE": "⚖", "PROJ": "⟐", "DIST": "⟂",
    "GATH": "⊃", "SCAT": "⊂",
}


def _pick(pool, n=1, exclude=()):
    """Pick n unique random items from pool, excluding given names."""
    avail = [x for x in pool if x not in exclude]
    return random.sample(avail, min(n, len(avail)))


def _inp():
    return random.choice(INPUT_NAMES)


def _out():
    return random.choice(OUTPUT_NAMES)


def _reg(idx):
    return REGS_NUMBERED[idx % len(REGS_NUMBERED)]


def _greg(idx):
    return REGS_GREEK[idx % len(REGS_GREEK)]


def _fval(lo=0.01, hi=100.0, decimals=2):
    return round(random.uniform(lo, hi), decimals)


def _ival(lo=1, hi=1000):
    return random.randint(lo, hi)


def _pair(user_text, nml_lines):
    return {
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": "\n".join(nml_lines)},
        ]
    }


def _to_symbolic(classic_lines):
    """Convert a list of classic NML lines to symbolic syntax."""
    result = []
    for line in classic_lines:
        tokens = line.split()
        if not tokens:
            result.append(line)
            continue
        op = tokens[0]
        sym_op = SYM.get(op, op)
        operands = tokens[1:]
        new_operands = []
        for o in operands:
            if o in REGS_NUMBERED:
                idx = REGS_NUMBERED.index(o)
                new_operands.append(REGS_GREEK[idx])
            else:
                new_operands.append(o)
        parts = [sym_op] + new_operands
        result.append("  ".join(parts))
    return result


SYMBOLIC_INLINE_RATE = 0.12


def _maybe_symbolic(pairs):
    """Randomly convert ~12% of classic pairs to symbolic syntax in-place."""
    for p in pairs:
        if random.random() < SYMBOLIC_INLINE_RATE:
            content = p["messages"][1]["content"]
            classic_lines = content.split("\n")
            if any(tok in classic_lines[0].split() for tok in REGS_NUMBERED if tok in classic_lines[0]):
                sym_lines = _to_symbolic(classic_lines)
                p["messages"][1]["content"] = "\n".join(sym_lines)
                old_prompt = p["messages"][0]["content"]
                if "symbolic" not in old_prompt.lower():
                    p["messages"][0]["content"] = old_prompt + " (use symbolic NML syntax)"
    return pairs


def _fmt(op, *operands):
    """Format a classic NML instruction with aligned columns."""
    parts = [f"{op:<5}"] + list(operands)
    return " ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Category 1: Basic Arithmetic (target: 1,000)
# ═══════════════════════════════════════════════════════════════════════════════

def gen_basic_arithmetic(count=1000):
    pairs = []

    templates = [
        ("Add {a} and {b}",
         lambda a, b, o: [_fmt("LD", "R0", f"@{a}"), _fmt("LD", "R1", f"@{b}"),
                          _fmt("MADD", "R2", "R0", "R1"), _fmt("ST", "R2", f"@{o}"), "HALT"]),
        ("Compute {a} + {b}",
         lambda a, b, o: [_fmt("LD", "R0", f"@{a}"), _fmt("LD", "R1", f"@{b}"),
                          _fmt("MADD", "R2", "R0", "R1"), _fmt("ST", "R2", f"@{o}"), "HALT"]),
        ("Subtract {b} from {a}",
         lambda a, b, o: [_fmt("LD", "R0", f"@{a}"), _fmt("LD", "R1", f"@{b}"),
                          _fmt("MSUB", "R2", "R0", "R1"), _fmt("ST", "R2", f"@{o}"), "HALT"]),
        ("Compute {a} minus {b}",
         lambda a, b, o: [_fmt("LD", "R0", f"@{a}"), _fmt("LD", "R1", f"@{b}"),
                          _fmt("MSUB", "R2", "R0", "R1"), _fmt("ST", "R2", f"@{o}"), "HALT"]),
        ("Multiply {a} by {b}",
         lambda a, b, o: [_fmt("LD", "R0", f"@{a}"), _fmt("LD", "R1", f"@{b}"),
                          _fmt("EMUL", "R2", "R0", "R1"), _fmt("ST", "R2", f"@{o}"), "HALT"]),
        ("Compute the product of {a} and {b}",
         lambda a, b, o: [_fmt("LD", "R0", f"@{a}"), _fmt("LD", "R1", f"@{b}"),
                          _fmt("EMUL", "R2", "R0", "R1"), _fmt("ST", "R2", f"@{o}"), "HALT"]),
        ("Divide {a} by {b}",
         lambda a, b, o: [_fmt("LD", "R0", f"@{a}"), _fmt("LD", "R1", f"@{b}"),
                          _fmt("EDIV", "R2", "R0", "R1"), _fmt("ST", "R2", f"@{o}"), "HALT"]),
        ("Scale {a} by {c}",
         lambda a, b, o, c=None: [_fmt("LD", "R0", f"@{a}"),
                                  _fmt("SCLR", "R1", "R0", f"#{c}"),
                                  _fmt("ST", "R1", f"@{o}"), "HALT"]),
        ("Multiply {a} by the constant {c}",
         lambda a, b, o, c=None: [_fmt("LD", "R0", f"@{a}"),
                                  _fmt("SCLR", "R1", "R0", f"#{c}"),
                                  _fmt("ST", "R1", f"@{o}"), "HALT"]),
        ("Divide {a} by {c_int} to get the {period} portion",
         lambda a, b, o, c=None, c_int=None, period=None:
             [_fmt("LD", "R0", f"@{a}"),
              _fmt("SDIV", "R1", "R0", f"#{c_int}.0"),
              _fmt("ST", "R1", f"@{o}"), "HALT"]),
        ("Compute {a} divided by {c_int}",
         lambda a, b, o, c=None, c_int=None, period=None:
             [_fmt("LD", "R0", f"@{a}"),
              _fmt("SDIV", "R1", "R0", f"#{c_int}.0"),
              _fmt("ST", "R1", f"@{o}"), "HALT"]),
        ("Add {c} to {a}",
         lambda a, b, o, c=None: [_fmt("LD", "R0", f"@{a}"),
                                  _fmt("MADD", "R1", "R0", f"#{c}"),
                                  _fmt("ST", "R1", f"@{o}"), "HALT"]),
        ("Negate {a} (compute 0 - {a})",
         lambda a, b, o: [_fmt("LD", "R0", f"@{a}"),
                          _fmt("SCLR", "R1", "R0", "#-1.0"),
                          _fmt("ST", "R1", f"@{o}"), "HALT"]),
        ("Double the value of {a}",
         lambda a, b, o: [_fmt("LD", "R0", f"@{a}"),
                          _fmt("SCLR", "R1", "R0", "#2.0"),
                          _fmt("ST", "R1", f"@{o}"), "HALT"]),
        ("Triple {a}",
         lambda a, b, o: [_fmt("LD", "R0", f"@{a}"),
                          _fmt("SCLR", "R1", "R0", "#3.0"),
                          _fmt("ST", "R1", f"@{o}"), "HALT"]),
        ("Halve {a}",
         lambda a, b, o: [_fmt("LD", "R0", f"@{a}"),
                          _fmt("SCLR", "R1", "R0", "#0.5"),
                          _fmt("ST", "R1", f"@{o}"), "HALT"]),
        ("Compute the square of {a} (a*a)",
         lambda a, b, o: [_fmt("LD", "R0", f"@{a}"),
                          _fmt("EMUL", "R1", "R0", "R0"),
                          _fmt("ST", "R1", f"@{o}"), "HALT"]),
        ("Compute {a} + {b} - {c}",
         lambda a, b, o, c=None:
             [_fmt("LD", "R0", f"@{a}"), _fmt("LD", "R1", f"@{b}"),
              _fmt("MADD", "R2", "R0", "R1"),
              _fmt("MSUB", "R2", "R2", f"#{c}"),
              _fmt("ST", "R2", f"@{o}"), "HALT"]),
        ("Compute weighted sum: {a}*{c} + {b}",
         lambda a, b, o, c=None:
             [_fmt("LD", "R0", f"@{a}"), _fmt("LD", "R1", f"@{b}"),
              _fmt("SCLR", "R2", "R0", f"#{c}"),
              _fmt("MADD", "R3", "R2", "R1"),
              _fmt("ST", "R3", f"@{o}"), "HALT"]),
        ("Compute average of {a} and {b}",
         lambda a, b, o:
             [_fmt("LD", "R0", f"@{a}"), _fmt("LD", "R1", f"@{b}"),
              _fmt("MADD", "R2", "R0", "R1"),
              _fmt("SDIV", "R3", "R2", "#2.0"),
              _fmt("ST", "R3", f"@{o}"), "HALT"]),
    ]

    periods = ["monthly", "weekly", "daily", "quarterly", "biweekly"]
    divisors = [12, 52, 365, 4, 26]

    while len(pairs) < count:
        t_idx = random.randint(0, len(templates) - 1)
        prompt_tpl, fn = templates[t_idx]

        a, b = _pick(INPUT_NAMES, 2)
        o = random.choice(OUTPUT_NAMES)
        c = _fval(0.01, 50.0)
        pidx = random.randint(0, len(periods) - 1)
        c_int = divisors[pidx]
        period = periods[pidx]

        prompt = prompt_tpl.format(a=a, b=b, c=c, c_int=c_int, period=period)
        prompt = f"Write NML to: {prompt}"

        try:
            lines = fn(a, b, o, c=c, c_int=c_int, period=period)
        except TypeError:
            try:
                lines = fn(a, b, o, c=c)
            except TypeError:
                lines = fn(a, b, o)

        pairs.append(_pair(prompt, lines))

    return _maybe_symbolic(pairs[:count])


# ═══════════════════════════════════════════════════════════════════════════════
# Category 2: Loops and Iteration (target: 1,500)
# ═══════════════════════════════════════════════════════════════════════════════

def gen_loops(count=1500):
    pairs = []

    loop_counts = [3, 5, 10, 12, 20, 26, 30, 52, 100, 360]

    def _sum_1_to_n(n, inp, out):
        return [
            _fmt("LEAF", "R0", f"#{n}"),
            _fmt("ALLC", "R1", "#[1]"),
            _fmt("LEAF", "R2", "#1"),
            _fmt("LOOP", "R0"),
            _fmt("TACC", "R1", "R1", "R2"),
            "ENDP",
            _fmt("ST", "R1", f"@{out}"),
            "HALT",
        ]

    def _factorial(n, inp, out):
        return [
            _fmt("LEAF", "R0", f"#{n}"),
            _fmt("LEAF", "R1", "#1"),
            _fmt("LEAF", "R2", "#1"),
            _fmt("LOOP", "R0"),
            _fmt("EMUL", "R1", "R1", "R2"),
            _fmt("MADD", "R2", "R2", "#1"),
            "ENDP",
            _fmt("ST", "R1", f"@{out}"),
            "HALT",
        ]

    def _power(inp, n, out):
        return [
            _fmt("LD", "R0", f"@{inp}"),
            _fmt("LEAF", "R1", f"#{n}"),
            _fmt("MOV", "R2", "R0"),
            _fmt("LOOP", "R1"),
            _fmt("EMUL", "R2", "R2", "R0"),
            "ENDP",
            _fmt("ST", "R2", f"@{out}"),
            "HALT",
        ]

    def _compound_interest(principal, rate_name, n, out):
        return [
            _fmt("LD", "R0", f"@{principal}"),
            _fmt("LD", "R1", f"@{rate_name}"),
            _fmt("MADD", "R1", "R1", "#1.0"),
            _fmt("MOV", "R2", "R0"),
            _fmt("LEAF", "R3", f"#{n}"),
            _fmt("LOOP", "R3"),
            _fmt("EMUL", "R2", "R2", "R1"),
            "ENDP",
            _fmt("ST", "R2", f"@{out}"),
            "HALT",
        ]

    def _average(inp, out):
        return [
            _fmt("LD", "R0", f"@{inp}"),
            _fmt("RDUC", "R1", "R0", "#1"),
            _fmt("ST", "R1", f"@{out}"),
            "HALT",
        ]

    def _find_max(inp, out):
        return [
            _fmt("LD", "R0", f"@{inp}"),
            _fmt("RDUC", "R1", "R0", "#2"),
            _fmt("ST", "R1", f"@{out}"),
            "HALT",
        ]

    def _find_min(inp, out):
        return [
            _fmt("LD", "R0", f"@{inp}"),
            _fmt("RDUC", "R1", "R0", "#3"),
            _fmt("ST", "R1", f"@{out}"),
            "HALT",
        ]

    def _sum_reduce(inp, out):
        return [
            _fmt("LD", "R0", f"@{inp}"),
            _fmt("RDUC", "R1", "R0", "#0"),
            _fmt("ST", "R1", f"@{out}"),
            "HALT",
        ]

    def _repeated_scaling(inp, n, factor, out):
        return [
            _fmt("LD", "R0", f"@{inp}"),
            _fmt("LEAF", "R1", f"#{n}"),
            _fmt("LOOP", "R1"),
            _fmt("SCLR", "R0", "R0", f"#{factor}"),
            "ENDP",
            _fmt("ST", "R0", f"@{out}"),
            "HALT",
        ]

    def _accumulate_loop(inp, n, out):
        return [
            _fmt("LD", "R0", f"@{inp}"),
            _fmt("ALLC", "R1", "#[1]"),
            _fmt("LEAF", "R2", f"#{n}"),
            _fmt("LOOP", "R2"),
            _fmt("TACC", "R1", "R1", "R0"),
            "ENDP",
            _fmt("ST", "R1", f"@{out}"),
            "HALT",
        ]

    def _countdown(n, out):
        return [
            _fmt("LEAF", "R0", f"#{n}"),
            _fmt("LEAF", "R1", "#1"),
            _fmt("LOOP", "R0"),
            _fmt("MSUB", "R0", "R0", "R1"),
            "ENDP",
            _fmt("ST", "R0", f"@{out}"),
            "HALT",
        ]

    prompts_and_fns = [
        ("Sum the integers from 1 to {n}", "sum_n"),
        ("Compute the sum 1+2+...+{n}", "sum_n"),
        ("Calculate {n} factorial", "factorial"),
        ("Compute factorial of {n}", "factorial"),
        ("Raise {inp} to the power {n}", "power"),
        ("Compute {inp}^{n} (exponentiation)", "power"),
        ("Compound interest on {inp} at rate {rate_name} for {n} periods", "compound"),
        ("Grow {inp} by {rate_name} compounded {n} times", "compound"),
        ("Average of the values in {inp}", "average"),
        ("Compute the mean of {inp}", "average"),
        ("Find the maximum value in {inp}", "max"),
        ("Get the largest element of {inp}", "max"),
        ("Find the minimum value in {inp}", "min"),
        ("Get the smallest element of {inp}", "min"),
        ("Sum all elements of {inp}", "sum_reduce"),
        ("Compute the total of {inp}", "sum_reduce"),
        ("Apply a scaling factor {factor} to {inp} repeatedly {n} times", "repeated_scale"),
        ("Scale {inp} by {factor} in a loop of {n} iterations", "repeated_scale"),
        ("Accumulate {inp} over {n} iterations", "accumulate"),
        ("Add {inp} to itself {n} times", "accumulate"),
        ("Count down from {n} to zero", "countdown"),
        ("Decrement from {n} to 0 in a loop", "countdown"),
    ]

    use_cases = [
        ("monthly payments", 12), ("yearly compounding", 30),
        ("weekly iterations", 52), ("quarterly cycles", 4),
        ("daily steps", 365), ("biweekly periods", 26),
    ]

    while len(pairs) < count:
        p_idx = random.randint(0, len(prompts_and_fns) - 1)
        prompt_tpl, fn_name = prompts_and_fns[p_idx]

        n = random.choice(loop_counts)
        inp = random.choice(INPUT_NAMES)
        out = random.choice(OUTPUT_NAMES)
        rate_name = random.choice(["interest_rate", "rate", "growth_rate", "factor"])
        factor = _fval(0.5, 2.0)

        prompt = prompt_tpl.format(n=n, inp=inp, rate_name=rate_name, factor=factor)
        prompt = f"Write NML to: {prompt}"

        if fn_name == "sum_n":
            lines = _sum_1_to_n(n, inp, out)
        elif fn_name == "factorial":
            n = random.choice([3, 5, 6, 7, 8, 10, 12])
            lines = _factorial(n, inp, out)
        elif fn_name == "power":
            n = random.choice([2, 3, 4, 5, 8, 10])
            lines = _power(inp, n, out)
        elif fn_name == "compound":
            lines = _compound_interest(inp, rate_name, n, out)
        elif fn_name == "average":
            lines = _average(inp, out)
        elif fn_name == "max":
            lines = _find_max(inp, out)
        elif fn_name == "min":
            lines = _find_min(inp, out)
        elif fn_name == "sum_reduce":
            lines = _sum_reduce(inp, out)
        elif fn_name == "repeated_scale":
            lines = _repeated_scaling(inp, n, factor, out)
        elif fn_name == "accumulate":
            lines = _accumulate_loop(inp, n, out)
        elif fn_name == "countdown":
            lines = _countdown(n, out)
        else:
            continue

        pairs.append(_pair(prompt, lines))

    return _maybe_symbolic(pairs[:count])


# ═══════════════════════════════════════════════════════════════════════════════
# Category 3: Conditionals and Branching (target: 1,500)
# ═══════════════════════════════════════════════════════════════════════════════

def gen_conditionals(count=1500):
    pairs = []

    domains = [
        ("pricing", "price", "adjusted_price"),
        ("shipping", "weight", "shipping_cost"),
        ("grading", "score", "grade"),
        ("temperature", "temperature", "status"),
        ("speed", "velocity", "alert"),
        ("inventory", "quantity", "reorder"),
        ("quality", "measurement", "quality_score"),
        ("power", "voltage", "power_status"),
        ("pressure", "pressure", "safety_level"),
        ("load", "current", "load_class"),
    ]

    def _if_gt_scale(inp, out, thresh, rate_a, rate_b):
        return [
            _fmt("LD", "R0", f"@{inp}"),
            _fmt("CMPI", "R8", "R0", f"#{thresh}"),
            _fmt("JMPF", "#3"),
            _fmt("SCLR", "R1", "R0", f"#{rate_a}"),
            _fmt("JUMP", "#2"),
            _fmt("SCLR", "R1", "R0", f"#{rate_b}"),
            _fmt("ST", "R1", f"@{out}"),
            "HALT",
        ]

    def _clamp(inp, out, lo, hi):
        return [
            _fmt("LD", "R0", f"@{inp}"),
            _fmt("CLMP", "R1", "R0", f"#{lo}", f"#{hi}"),
            _fmt("ST", "R1", f"@{out}"),
            "HALT",
        ]

    def _discount_if_gt(inp, out, thresh, discount):
        return [
            _fmt("LD", "R0", f"@{inp}"),
            _fmt("CMPI", "R8", "R0", f"#{thresh}"),
            _fmt("JMPF", "#3"),
            _fmt("SCLR", "R1", "R0", f"#{discount}"),
            _fmt("JUMP", "#2"),
            _fmt("MOV", "R1", "R0"),
            _fmt("ST", "R1", f"@{out}"),
            "HALT",
        ]

    def _three_tier(inp, out, t1, t2, r1, r2, r3):
        return [
            _fmt("LD", "R0", f"@{inp}"),
            _fmt("CMPI", "R8", "R0", f"#{t2}"),
            _fmt("JMPF", "#3"),
            _fmt("SCLR", "R1", "R0", f"#{r3}"),
            _fmt("JUMP", "#6"),
            _fmt("CMPI", "R8", "R0", f"#{t1}"),
            _fmt("JMPF", "#3"),
            _fmt("SCLR", "R1", "R0", f"#{r2}"),
            _fmt("JUMP", "#2"),
            _fmt("SCLR", "R1", "R0", f"#{r1}"),
            _fmt("ST", "R1", f"@{out}"),
            "HALT",
        ]

    def _zero_check_default(inp, out, default):
        return [
            _fmt("LD", "R0", f"@{inp}"),
            _fmt("CMPI", "R8", "R0", "#0"),
            _fmt("JMPF", "#3"),
            _fmt("LEAF", "R1", f"#{default}"),
            _fmt("JUMP", "#2"),
            _fmt("MOV", "R1", "R0"),
            _fmt("ST", "R1", f"@{out}"),
            "HALT",
        ]

    def _if_lt_cap(inp, out, cap, rate):
        return [
            _fmt("LD", "R0", f"@{inp}"),
            _fmt("CMPI", "R8", "R0", f"#{cap}"),
            _fmt("JMPT", "#3"),
            _fmt("SCLR", "R1", "R0", f"#{rate}"),
            _fmt("JUMP", "#2"),
            _fmt("LEAF", "R1", f"#{round(cap * rate, 2)}"),
            _fmt("ST", "R1", f"@{out}"),
            "HALT",
        ]

    def _abs_value(inp, out):
        return [
            _fmt("LD", "R0", f"@{inp}"),
            _fmt("CMPI", "R8", "R0", "#0"),
            _fmt("JMPT", "#2"),
            _fmt("SCLR", "R0", "R0", "#-1.0"),
            _fmt("ST", "R0", f"@{out}"),
            "HALT",
        ]

    def _max_of_two(a_name, b_name, out):
        return [
            _fmt("LD", "R0", f"@{a_name}"),
            _fmt("LD", "R1", f"@{b_name}"),
            _fmt("CMP", "R8", "R0", "R1"),
            _fmt("JMPF", "#3"),
            _fmt("MOV", "R2", "R0"),
            _fmt("JUMP", "#2"),
            _fmt("MOV", "R2", "R1"),
            _fmt("ST", "R2", f"@{out}"),
            "HALT",
        ]

    def _min_of_two(a_name, b_name, out):
        return [
            _fmt("LD", "R0", f"@{a_name}"),
            _fmt("LD", "R1", f"@{b_name}"),
            _fmt("CMP", "R8", "R0", "R1"),
            _fmt("JMPT", "#3"),
            _fmt("MOV", "R2", "R0"),
            _fmt("JUMP", "#2"),
            _fmt("MOV", "R2", "R1"),
            _fmt("ST", "R2", f"@{out}"),
            "HALT",
        ]

    prompts_fns = [
        ("If {inp} > {thresh}, scale by {rateA}, else scale by {rateB}", "if_gt_scale"),
        ("Apply rate {rateA} when {inp} exceeds {thresh}, otherwise {rateB}", "if_gt_scale"),
        ("Clamp {inp} between {lo} and {hi}", "clamp"),
        ("Bound {inp} to the range [{lo}, {hi}]", "clamp"),
        ("Apply {discount}x discount if {inp} > {thresh}", "discount"),
        ("Give a discount of {discount} when {inp} is above {thresh}", "discount"),
        ("Three-tier rate: below {t1} at {r1}, {t1}-{t2} at {r2}, above {t2} at {r3}", "three_tier"),
        ("Tiered scaling on {inp}: {r1} up to {t1}, {r2} up to {t2}, {r3} above", "three_tier"),
        ("If {inp} is zero, use default value {default}, otherwise keep {inp}", "zero_default"),
        ("Return {default} when {inp} is zero", "zero_default"),
        ("Cap {inp} at {cap} and apply rate {rate}", "cap"),
        ("Apply {rate} to {inp} but cap the base at {cap}", "cap"),
        ("Compute absolute value of {inp}", "abs"),
        ("Make {inp} non-negative (absolute value)", "abs"),
        ("Return the larger of {a_name} and {b_name}", "max_two"),
        ("Compute max({a_name}, {b_name})", "max_two"),
        ("Return the smaller of {a_name} and {b_name}", "min_two"),
        ("Compute min({a_name}, {b_name})", "min_two"),
    ]

    while len(pairs) < count:
        p_idx = random.randint(0, len(prompts_fns) - 1)
        prompt_tpl, fn_name = prompts_fns[p_idx]

        domain = random.choice(domains)
        inp, out = domain[1], domain[2]
        a_name, b_name = _pick(INPUT_NAMES, 2)
        thresh = _fval(10, 5000)
        rateA = _fval(0.01, 0.99)
        rateB = _fval(0.01, 0.99)
        lo = _fval(0, 100)
        hi = _fval(lo + 10, lo + 10000)
        discount = round(random.uniform(0.5, 0.99), 2)
        t1 = _fval(10, 500)
        t2 = round(t1 + random.uniform(100, 5000), 2)
        r1 = _fval(0.01, 0.5)
        r2 = _fval(0.01, 0.5)
        r3 = _fval(0.01, 0.5)
        default = _fval(1, 1000)
        cap = _fval(100, 50000)
        rate = _fval(0.01, 0.99)

        prompt = prompt_tpl.format(
            inp=inp, thresh=thresh, rateA=rateA, rateB=rateB,
            lo=lo, hi=hi, discount=discount, t1=t1, t2=t2,
            r1=r1, r2=r2, r3=r3, default=default, cap=cap, rate=rate,
            a_name=a_name, b_name=b_name,
        )
        prompt = f"Write NML to: {prompt}"

        if fn_name == "if_gt_scale":
            lines = _if_gt_scale(inp, out, thresh, rateA, rateB)
        elif fn_name == "clamp":
            lines = _clamp(inp, out, lo, hi)
        elif fn_name == "discount":
            lines = _discount_if_gt(inp, out, thresh, discount)
        elif fn_name == "three_tier":
            lines = _three_tier(inp, out, t1, t2, r1, r2, r3)
        elif fn_name == "zero_default":
            lines = _zero_check_default(inp, out, default)
        elif fn_name == "cap":
            lines = _if_lt_cap(inp, out, cap, rate)
        elif fn_name == "abs":
            lines = _abs_value(inp, out)
        elif fn_name == "max_two":
            lines = _max_of_two(a_name, b_name, out)
        elif fn_name == "min_two":
            lines = _min_of_two(a_name, b_name, out)
        else:
            continue

        pairs.append(_pair(prompt, lines))

    return _maybe_symbolic(pairs[:count])


# ═══════════════════════════════════════════════════════════════════════════════
# Category 4: Subroutines (target: 800)
# ═══════════════════════════════════════════════════════════════════════════════

def gen_subroutines(count=800):
    pairs = []

    def _double_sub(inp, out):
        return [
            _fmt("LD", "R0", f"@{inp}"),
            _fmt("CALL", "#3"),
            _fmt("ST", "R0", f"@{out}"),
            "HALT",
            _fmt("SCLR", "R0", "R0", "#2.0"),
            "RET",
        ]

    def _triple_sub(inp, out):
        return [
            _fmt("LD", "R0", f"@{inp}"),
            _fmt("CALL", "#3"),
            _fmt("ST", "R0", f"@{out}"),
            "HALT",
            _fmt("SCLR", "R0", "R0", "#3.0"),
            "RET",
        ]

    def _negate_sub(inp, out):
        return [
            _fmt("LD", "R0", f"@{inp}"),
            _fmt("CALL", "#3"),
            _fmt("ST", "R0", f"@{out}"),
            "HALT",
            _fmt("SCLR", "R0", "R0", "#-1.0"),
            "RET",
        ]

    def _scale_sub(inp, out, factor):
        return [
            _fmt("LD", "R0", f"@{inp}"),
            _fmt("CALL", "#3"),
            _fmt("ST", "R0", f"@{out}"),
            "HALT",
            _fmt("SCLR", "R0", "R0", f"#{factor}"),
            "RET",
        ]

    def _square_sub(inp, out):
        return [
            _fmt("LD", "R0", f"@{inp}"),
            _fmt("CALL", "#3"),
            _fmt("ST", "R0", f"@{out}"),
            "HALT",
            _fmt("EMUL", "R0", "R0", "R0"),
            "RET",
        ]

    def _add_const_sub(inp, out, c):
        return [
            _fmt("LD", "R0", f"@{inp}"),
            _fmt("CALL", "#3"),
            _fmt("ST", "R0", f"@{out}"),
            "HALT",
            _fmt("MADD", "R0", "R0", f"#{c}"),
            "RET",
        ]

    def _two_input_add_sub(a, b, out):
        return [
            _fmt("LD", "R0", f"@{a}"),
            _fmt("LD", "R1", f"@{b}"),
            _fmt("CALL", "#3"),
            _fmt("ST", "R2", f"@{out}"),
            "HALT",
            _fmt("MADD", "R2", "R0", "R1"),
            "RET",
        ]

    def _two_input_mul_sub(a, b, out):
        return [
            _fmt("LD", "R0", f"@{a}"),
            _fmt("LD", "R1", f"@{b}"),
            _fmt("CALL", "#3"),
            _fmt("ST", "R2", f"@{out}"),
            "HALT",
            _fmt("EMUL", "R2", "R0", "R1"),
            "RET",
        ]

    def _scale_and_add_sub(inp, bias, out, factor):
        return [
            _fmt("LD", "R0", f"@{inp}"),
            _fmt("LD", "R1", f"@{bias}"),
            _fmt("CALL", "#3"),
            _fmt("ST", "R2", f"@{out}"),
            "HALT",
            _fmt("SCLR", "R2", "R0", f"#{factor}"),
            _fmt("MADD", "R2", "R2", "R1"),
            "RET",
        ]

    def _chain_two_subs(inp, out, f1, f2):
        return [
            _fmt("LD", "R0", f"@{inp}"),
            _fmt("CALL", "#4"),
            _fmt("CALL", "#5"),
            _fmt("ST", "R0", f"@{out}"),
            "HALT",
            _fmt("SCLR", "R0", "R0", f"#{f1}"),
            "RET",
            _fmt("MADD", "R0", "R0", f"#{f2}"),
            "RET",
        ]

    prompts_fns = [
        ("Double {inp} using a subroutine", "double"),
        ("Create a subroutine to double {inp}", "double"),
        ("Triple {inp} via a subroutine call", "triple"),
        ("Use a subroutine to triple {inp}", "triple"),
        ("Negate {inp} using a subroutine", "negate"),
        ("Subroutine that negates {inp}", "negate"),
        ("Scale {inp} by {factor} using a reusable subroutine", "scale"),
        ("Create a reusable scale function for {inp} at factor {factor}", "scale"),
        ("Square {inp} using a subroutine", "square"),
        ("Subroutine that computes {inp}*{inp}", "square"),
        ("Add constant {c} to {inp} via subroutine", "add_const"),
        ("Subroutine that adds {c} to {inp}", "add_const"),
        ("Subroutine that adds {a} and {b}", "two_add"),
        ("Create a reusable add subroutine for {a} + {b}", "two_add"),
        ("Subroutine that multiplies {a} and {b}", "two_mul"),
        ("Call a subroutine to compute {a} * {b}", "two_mul"),
        ("Scale {inp} by {factor} and add {bias} in a subroutine", "scale_add"),
        ("Subroutine: {inp} * {factor} + {bias}", "scale_add"),
        ("Chain two subroutines: scale by {f1} then add {f2}", "chain"),
        ("Call two subroutines in sequence on {inp}", "chain"),
    ]

    while len(pairs) < count:
        p_idx = random.randint(0, len(prompts_fns) - 1)
        prompt_tpl, fn_name = prompts_fns[p_idx]

        inp = random.choice(INPUT_NAMES)
        out = random.choice(OUTPUT_NAMES)
        a, b = _pick(INPUT_NAMES, 2)
        bias = random.choice(["bias", "offset", "constant", "adjustment"])
        factor = _fval(0.1, 10.0)
        c = _fval(1, 500)
        f1 = _fval(0.5, 5.0)
        f2 = _fval(1, 100)

        prompt = prompt_tpl.format(
            inp=inp, out=out, factor=factor, c=c, a=a, b=b,
            bias=bias, f1=f1, f2=f2,
        )
        prompt = f"Write NML to: {prompt}"

        if fn_name == "double":
            lines = _double_sub(inp, out)
        elif fn_name == "triple":
            lines = _triple_sub(inp, out)
        elif fn_name == "negate":
            lines = _negate_sub(inp, out)
        elif fn_name == "scale":
            lines = _scale_sub(inp, out, factor)
        elif fn_name == "square":
            lines = _square_sub(inp, out)
        elif fn_name == "add_const":
            lines = _add_const_sub(inp, out, c)
        elif fn_name == "two_add":
            lines = _two_input_add_sub(a, b, out)
        elif fn_name == "two_mul":
            lines = _two_input_mul_sub(a, b, out)
        elif fn_name == "scale_add":
            lines = _scale_and_add_sub(inp, bias, out, factor)
        elif fn_name == "chain":
            lines = _chain_two_subs(inp, out, f1, f2)
        else:
            continue

        pairs.append(_pair(prompt, lines))

    return _maybe_symbolic(pairs[:count])


# ═══════════════════════════════════════════════════════════════════════════════
# Category 5: Neural Network Programs (target: 1,200)
# ═══════════════════════════════════════════════════════════════════════════════

def gen_neural_nets(count=1200):
    pairs = []

    layer_sizes = [2, 4, 8, 16, 32, 64]
    activations_classic = ["RELU", "SIGM", "TANH", "GELU"]
    act_names = {"RELU": "ReLU", "SIGM": "sigmoid", "TANH": "tanh", "GELU": "GELU"}

    def _one_layer(inp, w, b, out, act):
        return [
            _fmt("LD", "R0", f"@{inp}"),
            _fmt("LD", "R1", f"@{w}"),
            _fmt("LD", "R2", f"@{b}"),
            _fmt("MMUL", "R3", "R0", "R1"),
            _fmt("MADD", "R3", "R3", "R2"),
            _fmt(act, "R3", "R3"),
            _fmt("ST", "R3", f"@{out}"),
            "HALT",
        ]

    def _two_layer(inp, w1, b1, w2, b2, out, act1, act2):
        return [
            _fmt("LD", "R0", f"@{inp}"),
            _fmt("LD", "R1", f"@{w1}"),
            _fmt("LD", "R2", f"@{b1}"),
            _fmt("MMUL", "R3", "R0", "R1"),
            _fmt("MADD", "R3", "R3", "R2"),
            _fmt(act1, "R3", "R3"),
            _fmt("LD", "R4", f"@{w2}"),
            _fmt("LD", "R5", f"@{b2}"),
            _fmt("MMUL", "R6", "R3", "R4"),
            _fmt("MADD", "R6", "R6", "R5"),
            _fmt(act2, "R6", "R6"),
            _fmt("ST", "R6", f"@{out}"),
            "HALT",
        ]

    def _three_layer(inp, out, act):
        return [
            _fmt("LD", "R0", f"@{inp}"),
            _fmt("LD", "R1", "@weights_1"),
            _fmt("LD", "R2", "@bias_1"),
            _fmt("MMUL", "R3", "R0", "R1"),
            _fmt("MADD", "R3", "R3", "R2"),
            _fmt(act, "R3", "R3"),
            _fmt("LD", "R4", "@weights_2"),
            _fmt("LD", "R5", "@bias_2"),
            _fmt("MMUL", "R6", "R3", "R4"),
            _fmt("MADD", "R6", "R6", "R5"),
            _fmt(act, "R6", "R6"),
            _fmt("LD", "R7", "@weights_3"),
            _fmt("LD", "R8", "@bias_3"),
            _fmt("MMUL", "R9", "R6", "R7"),
            _fmt("MADD", "R9", "R9", "R8"),
            _fmt("SIGM", "R9", "R9"),
            _fmt("ST", "R9", f"@{out}"),
            "HALT",
        ]

    def _softmax_classifier(inp, out):
        return [
            _fmt("LD", "R0", f"@{inp}"),
            _fmt("LD", "R1", "@weights"),
            _fmt("LD", "R2", "@bias"),
            _fmt("MMUL", "R3", "R0", "R1"),
            _fmt("MADD", "R3", "R3", "R2"),
            _fmt("SOFT", "R3", "R3"),
            _fmt("ST", "R3", f"@{out}"),
            "HALT",
        ]

    def _linear_regression(inp, out):
        return [
            _fmt("LD", "R0", f"@{inp}"),
            _fmt("LD", "R1", "@weights"),
            _fmt("LD", "R2", "@bias"),
            _fmt("MMUL", "R3", "R0", "R1"),
            _fmt("MADD", "R3", "R3", "R2"),
            _fmt("ST", "R3", f"@{out}"),
            "HALT",
        ]

    def _autoencoder(inp, out, act):
        return [
            _fmt("LD", "R0", f"@{inp}"),
            _fmt("LD", "R1", "@encoder_w"),
            _fmt("LD", "R2", "@encoder_b"),
            _fmt("MMUL", "R3", "R0", "R1"),
            _fmt("MADD", "R3", "R3", "R2"),
            _fmt(act, "R3", "R3"),
            _fmt("LD", "R4", "@decoder_w"),
            _fmt("LD", "R5", "@decoder_b"),
            _fmt("MMUL", "R6", "R3", "R4"),
            _fmt("MADD", "R6", "R6", "R5"),
            _fmt("SIGM", "R6", "R6"),
            _fmt("ST", "R6", f"@{out}"),
            "HALT",
        ]

    def _with_norm(inp, out, act):
        return [
            _fmt("LD", "R0", f"@{inp}"),
            _fmt("LD", "R1", "@weights"),
            _fmt("LD", "R2", "@bias"),
            _fmt("MMUL", "R3", "R0", "R1"),
            _fmt("MADD", "R3", "R3", "R2"),
            _fmt("NORM", "R3", "R3"),
            _fmt(act, "R3", "R3"),
            _fmt("ST", "R3", f"@{out}"),
            "HALT",
        ]

    nn_prompts = [
        ("1-layer neural network with {n_in} inputs, {act_name} activation", "one_layer"),
        ("Single hidden layer network: {n_in} → {n_hid} with {act_name}", "one_layer"),
        ("2-layer classifier: {n_in} → {n_hid} → output with {act_name}", "two_layer"),
        ("Two-layer neural network with {act_name} activations", "two_layer"),
        ("3-layer anomaly detector with {act_name} hidden layers", "three_layer"),
        ("Deep network: 3 layers with {act_name} and sigmoid output", "three_layer"),
        ("Softmax classifier with {n_out} classes", "softmax"),
        ("{n_out}-class softmax classifier from {n_in} features", "softmax"),
        ("Linear regression: y = Wx + b", "linear"),
        ("Simple linear model for {n_in} features", "linear"),
        ("Autoencoder: {n_in} → {n_hid} → {n_in} with {act_name}", "autoencoder"),
        ("Encoder-decoder network with {act_name} in the bottleneck", "autoencoder"),
        ("Neural network layer with layer normalization and {act_name}", "with_norm"),
        ("Normalized {act_name} layer for {n_in} inputs", "with_norm"),
    ]

    while len(pairs) < count:
        p_idx = random.randint(0, len(nn_prompts) - 1)
        prompt_tpl, fn_name = nn_prompts[p_idx]

        n_in = random.choice(layer_sizes)
        n_hid = random.choice(layer_sizes)
        n_out = random.choice([2, 3, 5, 10])
        act = random.choice(activations_classic)
        act_name = act_names[act]
        act2 = random.choice(activations_classic)
        inp = random.choice(["input", "features", "data", "samples", "signal", "readings"])
        out = random.choice(["prediction", "output", "classification", "probability", "score", "result"])

        prompt = prompt_tpl.format(
            n_in=n_in, n_hid=n_hid, n_out=n_out, act_name=act_name
        )
        prompt = f"Write NML for a {prompt}"

        if fn_name == "one_layer":
            lines = _one_layer(inp, "weights", "bias", out, act)
        elif fn_name == "two_layer":
            lines = _two_layer(inp, "weights_1", "bias_1", "weights_2", "bias_2", out, act, act2)
        elif fn_name == "three_layer":
            lines = _three_layer(inp, out, act)
        elif fn_name == "softmax":
            lines = _softmax_classifier(inp, out)
        elif fn_name == "linear":
            lines = _linear_regression(inp, out)
        elif fn_name == "autoencoder":
            lines = _autoencoder(inp, out, act)
        elif fn_name == "with_norm":
            lines = _with_norm(inp, out, act)
        else:
            continue

        pairs.append(_pair(prompt, lines))

    return _maybe_symbolic(pairs[:count])


# ═══════════════════════════════════════════════════════════════════════════════
# Category 6: Financial Calculations — non-tax (target: 1,000)
# ═══════════════════════════════════════════════════════════════════════════════

def gen_financial(count=1000):
    pairs = []

    def _simple_interest(p, r, t, out):
        return [
            _fmt("LD", "R0", f"@{p}"),
            _fmt("LD", "R1", f"@{r}"),
            _fmt("LD", "R2", f"@{t}"),
            _fmt("EMUL", "R3", "R0", "R1"),
            _fmt("EMUL", "R3", "R3", "R2"),
            _fmt("ST", "R3", f"@{out}"),
            "HALT",
        ]

    def _compound_interest(p, r, n_periods, out):
        return [
            _fmt("LD", "R0", f"@{p}"),
            _fmt("LD", "R1", f"@{r}"),
            _fmt("MADD", "R1", "R1", "#1.0"),
            _fmt("MOV", "R2", "R0"),
            _fmt("LEAF", "R3", f"#{n_periods}"),
            _fmt("LOOP", "R3"),
            _fmt("EMUL", "R2", "R2", "R1"),
            "ENDP",
            _fmt("ST", "R2", f"@{out}"),
            "HALT",
        ]

    def _depreciation(val, rate, years, out):
        return [
            _fmt("LD", "R0", f"@{val}"),
            _fmt("LD", "R1", f"@{rate}"),
            _fmt("LEAF", "R9", "#1.0"),
            _fmt("MSUB", "R1", "R9", "R1"),
            _fmt("LEAF", "R2", f"#{years}"),
            _fmt("LOOP", "R2"),
            _fmt("EMUL", "R0", "R0", "R1"),
            "ENDP",
            _fmt("ST", "R0", f"@{out}"),
            "HALT",
        ]

    def _roi(gain, cost_name, out):
        return [
            _fmt("LD", "R0", f"@{gain}"),
            _fmt("LD", "R1", f"@{cost_name}"),
            _fmt("MSUB", "R2", "R0", "R1"),
            _fmt("EDIV", "R3", "R2", "R1"),
            _fmt("ST", "R3", f"@{out}"),
            "HALT",
        ]

    def _tip_calculator(amount, rate_val, out):
        return [
            _fmt("LD", "R0", f"@{amount}"),
            _fmt("SCLR", "R1", "R0", f"#{rate_val}"),
            _fmt("MADD", "R2", "R0", "R1"),
            _fmt("ST", "R2", f"@{out}"),
            "HALT",
        ]

    def _currency_conversion(amount, rate_name, out):
        return [
            _fmt("LD", "R0", f"@{amount}"),
            _fmt("LD", "R1", f"@{rate_name}"),
            _fmt("EMUL", "R2", "R0", "R1"),
            _fmt("ST", "R2", f"@{out}"),
            "HALT",
        ]

    def _mortgage_payment(principal, monthly_rate, n_months, out):
        return [
            _fmt("LD", "R0", f"@{principal}"),
            _fmt("LD", "R1", f"@{monthly_rate}"),
            _fmt("MADD", "R2", "R1", "#1.0"),
            _fmt("MOV", "R3", "R2"),
            _fmt("LEAF", "R4", f"#{n_months}"),
            _fmt("LOOP", "R4"),
            _fmt("EMUL", "R3", "R3", "R2"),
            "ENDP",
            _fmt("EMUL", "R5", "R1", "R3"),
            _fmt("MSUB", "R6", "R3", "#1.0"),
            _fmt("EDIV", "R7", "R5", "R6"),
            _fmt("EMUL", "R8", "R0", "R7"),
            _fmt("ST", "R8", f"@{out}"),
            "HALT",
        ]

    def _profit_margin(revenue, cost_name, out):
        return [
            _fmt("LD", "R0", f"@{revenue}"),
            _fmt("LD", "R1", f"@{cost_name}"),
            _fmt("MSUB", "R2", "R0", "R1"),
            _fmt("EDIV", "R3", "R2", "R0"),
            _fmt("ST", "R3", f"@{out}"),
            "HALT",
        ]

    def _break_even(fixed_cost, price_name, var_cost, out):
        return [
            _fmt("LD", "R0", f"@{fixed_cost}"),
            _fmt("LD", "R1", f"@{price_name}"),
            _fmt("LD", "R2", f"@{var_cost}"),
            _fmt("MSUB", "R3", "R1", "R2"),
            _fmt("EDIV", "R4", "R0", "R3"),
            _fmt("ST", "R4", f"@{out}"),
            "HALT",
        ]

    def _present_value(future_val, rate, n_periods, out):
        return [
            _fmt("LD", "R0", f"@{future_val}"),
            _fmt("LD", "R1", f"@{rate}"),
            _fmt("MADD", "R1", "R1", "#1.0"),
            _fmt("LEAF", "R2", "#1.0"),
            _fmt("LEAF", "R3", f"#{n_periods}"),
            _fmt("LOOP", "R3"),
            _fmt("EMUL", "R2", "R2", "R1"),
            "ENDP",
            _fmt("EDIV", "R4", "R0", "R2"),
            _fmt("ST", "R4", f"@{out}"),
            "HALT",
        ]

    fin_prompts = [
        ("Simple interest: I = P * r * t for {p}, {r}, {t}", "simple"),
        ("Compute simple interest on {p} at rate {r} for {t} periods", "simple"),
        ("Compound interest: grow {p} at rate {r} for {n} periods", "compound"),
        ("Compute A = P*(1+r)^n for {p}", "compound"),
        ("Depreciation: {val} depreciating at rate {rate} for {years} years", "depreciation"),
        ("Compute asset value after {years} years of depreciation", "depreciation"),
        ("ROI: ({gain} - {cost}) / {cost}", "roi"),
        ("Return on investment for {gain} over {cost}", "roi"),
        ("Tip calculator: {amount} with tip rate {rate_val}", "tip"),
        ("Add a {rate_val}x tip to {amount}", "tip"),
        ("Convert {amount} at exchange rate {rate_name}", "currency"),
        ("Currency conversion: {amount} * {rate_name}", "currency"),
        ("Monthly mortgage payment: {principal} at {monthly_rate} for {n} months", "mortgage"),
        ("Loan payment for {principal} over {n} months", "mortgage"),
        ("Profit margin: ({revenue} - {cost}) / {revenue}", "profit_margin"),
        ("Compute profit margin on {revenue}", "profit_margin"),
        ("Break-even point: {fixed_cost} / ({price} - {var_cost})", "break_even"),
        ("How many units to break even given {fixed_cost} fixed costs", "break_even"),
        ("Present value of {future_val} at {rate} over {n} periods", "present_value"),
        ("Discount {future_val} to present value", "present_value"),
    ]

    while len(pairs) < count:
        p_idx = random.randint(0, len(fin_prompts) - 1)
        prompt_tpl, fn_name = fin_prompts[p_idx]

        p = random.choice(["principal", "balance", "amount", "investment"])
        r = random.choice(["interest_rate", "rate", "growth_rate"])
        t = random.choice(["years", "months", "time", "periods"])
        val = random.choice(["value", "asset_value", "amount"])
        rate = random.choice(["rate", "depreciation_rate"])
        years = random.choice([3, 5, 10, 15, 20, 30])
        n = random.choice([12, 24, 36, 60, 120, 180, 360])
        gain = random.choice(["revenue", "gain", "return_amount"])
        cost_name = random.choice(["cost", "investment", "expense"])
        amount = random.choice(["amount", "price", "total", "subtotal"])
        rate_val = round(random.uniform(0.05, 0.30), 2)
        rate_name = random.choice(["exchange_rate", "rate", "conversion_factor"])
        principal = random.choice(["principal", "loan_amount", "balance"])
        monthly_rate = random.choice(["monthly_rate", "rate"])
        revenue = random.choice(["revenue", "sales", "income"])
        fixed_cost = random.choice(["fixed_cost", "overhead", "fixed_expense"])
        price_name = random.choice(["price", "unit_price", "selling_price"])
        var_cost = random.choice(["variable_cost", "unit_cost", "cost_per_unit"])
        future_val = random.choice(["future_value", "amount", "value"])
        out = random.choice(OUTPUT_NAMES)

        prompt = prompt_tpl.format(
            p=p, r=r, t=t, val=val, rate=rate, years=years, n=n,
            gain=gain, cost=cost_name, amount=amount, rate_val=rate_val,
            rate_name=rate_name, principal=principal, monthly_rate=monthly_rate,
            revenue=revenue, fixed_cost=fixed_cost, price=price_name,
            var_cost=var_cost, future_val=future_val,
        )
        prompt = f"Write NML for: {prompt}"

        if fn_name == "simple":
            lines = _simple_interest(p, r, t, out)
        elif fn_name == "compound":
            lines = _compound_interest(p, r, n, out)
        elif fn_name == "depreciation":
            lines = _depreciation(val, rate, years, out)
        elif fn_name == "roi":
            lines = _roi(gain, cost_name, out)
        elif fn_name == "tip":
            lines = _tip_calculator(amount, rate_val, out)
        elif fn_name == "currency":
            lines = _currency_conversion(amount, rate_name, out)
        elif fn_name == "mortgage":
            lines = _mortgage_payment(principal, monthly_rate, n, out)
        elif fn_name == "profit_margin":
            lines = _profit_margin(revenue, cost_name, out)
        elif fn_name == "break_even":
            lines = _break_even(fixed_cost, price_name, var_cost, out)
        elif fn_name == "present_value":
            lines = _present_value(future_val, rate, n, out)
        else:
            continue

        pairs.append(_pair(prompt, lines))

    return _maybe_symbolic(pairs[:count])


# ═══════════════════════════════════════════════════════════════════════════════
# Category 7: Data Processing / Science (target: 1,000)
# ═══════════════════════════════════════════════════════════════════════════════

def gen_data_science(count=1000):
    pairs = []

    def _dot_product(a, b, out):
        return [
            _fmt("LD", "R0", f"@{a}"),
            _fmt("LD", "R1", f"@{b}"),
            _fmt("SDOT", "R2", "R0", "R1"),
            _fmt("ST", "R2", f"@{out}"),
            "HALT",
        ]

    def _matrix_multiply(a, b, out):
        return [
            _fmt("LD", "R0", f"@{a}"),
            _fmt("LD", "R1", f"@{b}"),
            _fmt("MMUL", "R2", "R0", "R1"),
            _fmt("ST", "R2", f"@{out}"),
            "HALT",
        ]

    def _cosine_similarity(a, b, out):
        return [
            _fmt("LD", "R0", f"@{a}"),
            _fmt("LD", "R1", f"@{b}"),
            _fmt("DIST", "R2", "R0", "R1", "#1"),
            _fmt("ST", "R2", f"@{out}"),
            "HALT",
        ]

    def _euclidean_distance(a, b, out):
        return [
            _fmt("LD", "R0", f"@{a}"),
            _fmt("LD", "R1", f"@{b}"),
            _fmt("DIST", "R2", "R0", "R1", "#0"),
            _fmt("ST", "R2", f"@{out}"),
            "HALT",
        ]

    def _threshold_mask(data, thresh, out):
        return [
            _fmt("LD", "R0", f"@{data}"),
            _fmt("CMPR", "R1", "R0", f"#{thresh}"),
            _fmt("EMUL", "R2", "R0", "R1"),
            _fmt("ST", "R2", f"@{out}"),
            "HALT",
        ]

    def _running_average(data, out):
        return [
            _fmt("LD", "R0", f"@{data}"),
            _fmt("RDUC", "R1", "R0", "#1"),
            _fmt("ST", "R1", f"@{out}"),
            "HALT",
        ]

    def _standard_score(data, out):
        return [
            _fmt("LD", "R0", f"@{data}"),
            _fmt("RDUC", "R1", "R0", "#1"),
            _fmt("MSUB", "R2", "R0", "R1"),
            _fmt("LD", "R3", f"@std_dev"),
            _fmt("EDIV", "R4", "R2", "R3"),
            _fmt("ST", "R4", f"@{out}"),
            "HALT",
        ]

    def _projection(data, proj_matrix, out):
        return [
            _fmt("LD", "R0", f"@{data}"),
            _fmt("LD", "R1", f"@{proj_matrix}"),
            _fmt("PROJ", "R2", "R0", "R1"),
            _fmt("ST", "R2", f"@{out}"),
            "HALT",
        ]

    def _transpose(data, out):
        return [
            _fmt("LD", "R0", f"@{data}"),
            _fmt("TRNS", "R1", "R0"),
            _fmt("ST", "R1", f"@{out}"),
            "HALT",
        ]

    def _sum_and_mean(data, out_sum, out_mean):
        return [
            _fmt("LD", "R0", f"@{data}"),
            _fmt("RDUC", "R1", "R0", "#0"),
            _fmt("RDUC", "R2", "R0", "#1"),
            _fmt("ST", "R1", f"@{out_sum}"),
            _fmt("ST", "R2", f"@{out_mean}"),
            "HALT",
        ]

    def _min_max(data, out_min, out_max):
        return [
            _fmt("LD", "R0", f"@{data}"),
            _fmt("RDUC", "R1", "R0", "#2"),
            _fmt("RDUC", "R2", "R0", "#3"),
            _fmt("ST", "R1", f"@{out_max}"),
            _fmt("ST", "R2", f"@{out_min}"),
            "HALT",
        ]

    def _weighted_sum(data, weights_name, out):
        return [
            _fmt("LD", "R0", f"@{data}"),
            _fmt("LD", "R1", f"@{weights_name}"),
            _fmt("EMUL", "R2", "R0", "R1"),
            _fmt("RDUC", "R3", "R2", "#0"),
            _fmt("ST", "R3", f"@{out}"),
            "HALT",
        ]

    def _normalize(data, out):
        return [
            _fmt("LD", "R0", f"@{data}"),
            _fmt("NORM", "R1", "R0"),
            _fmt("ST", "R1", f"@{out}"),
            "HALT",
        ]

    def _clamp_data(data, lo, hi, out):
        return [
            _fmt("LD", "R0", f"@{data}"),
            _fmt("CLMP", "R1", "R0", f"#{lo}", f"#{hi}"),
            _fmt("ST", "R1", f"@{out}"),
            "HALT",
        ]

    def _reshape(data, dim, out):
        return [
            _fmt("LD", "R0", f"@{data}"),
            _fmt("RSHP", "R1", "R0", f"#{dim}"),
            _fmt("ST", "R1", f"@{out}"),
            "HALT",
        ]

    ds_prompts = [
        ("Dot product of {a} and {b}", "dot"),
        ("Compute the inner product of {a} and {b}", "dot"),
        ("Matrix multiply {a} by {b}", "matmul"),
        ("Compute {a} @ {b}", "matmul"),
        ("Cosine similarity between {a} and {b}", "cosine"),
        ("Measure similarity of {a} and {b} using cosine distance", "cosine"),
        ("Euclidean distance between {a} and {b}", "euclidean"),
        ("L2 distance from {a} to {b}", "euclidean"),
        ("Apply threshold mask: zero out {data} values below {thresh}", "threshold"),
        ("Mask {data} where values are less than {thresh}", "threshold"),
        ("Average of {data}", "running_avg"),
        ("Compute the mean of all elements in {data}", "running_avg"),
        ("Standard score (z-score) of {data}", "zscore"),
        ("Normalize {data} to zero mean using z-score", "zscore"),
        ("Project {data} into embedding space using {proj}", "project"),
        ("Reduce dimensionality of {data} via projection with {proj}", "project"),
        ("Transpose {data}", "transpose"),
        ("Compute the transpose of matrix {data}", "transpose"),
        ("Sum and mean of {data}", "sum_mean"),
        ("Compute both total and average of {data}", "sum_mean"),
        ("Find min and max of {data}", "min_max"),
        ("Get the range (min, max) of {data}", "min_max"),
        ("Weighted sum of {data} with {weights}", "weighted_sum"),
        ("Compute a weighted combination of {data}", "weighted_sum"),
        ("Layer-normalize {data}", "normalize"),
        ("Apply normalization to {data}", "normalize"),
        ("Clamp {data} to range [{lo}, {hi}]", "clamp_data"),
        ("Bound all values in {data} between {lo} and {hi}", "clamp_data"),
        ("Reshape {data} to dimension {dim}", "reshape"),
        ("Flatten {data} into a {dim}-element vector", "reshape"),
    ]

    while len(pairs) < count:
        p_idx = random.randint(0, len(ds_prompts) - 1)
        prompt_tpl, fn_name = ds_prompts[p_idx]

        a = random.choice(["vector", "features", "signal", "readings", "samples", "data"])
        b = random.choice(["vector_b", "weights", "kernel", "matrix_b", "reference"])
        data = random.choice(["data", "signal", "readings", "features", "samples", "scores"])
        proj = random.choice(["projection_matrix", "basis", "transform"])
        weights_name = random.choice(["weights", "coefficients", "importance"])
        out = random.choice(OUTPUT_NAMES)
        thresh = _fval(0.0, 10.0)
        lo = _fval(0, 5)
        hi = _fval(lo + 1, 100)
        dim = random.choice([4, 8, 16, 32, 64, 128])

        prompt = prompt_tpl.format(
            a=a, b=b, data=data, thresh=thresh, proj=proj,
            weights=weights_name, lo=lo, hi=hi, dim=dim,
        )
        prompt = f"Write NML to: {prompt}"

        if fn_name == "dot":
            lines = _dot_product(a, b, out)
        elif fn_name == "matmul":
            lines = _matrix_multiply(a, b, out)
        elif fn_name == "cosine":
            lines = _cosine_similarity(a, b, out)
        elif fn_name == "euclidean":
            lines = _euclidean_distance(a, b, out)
        elif fn_name == "threshold":
            lines = _threshold_mask(data, thresh, out)
        elif fn_name == "running_avg":
            lines = _running_average(data, out)
        elif fn_name == "zscore":
            lines = _standard_score(data, out)
        elif fn_name == "project":
            lines = _projection(data, proj, out)
        elif fn_name == "transpose":
            lines = _transpose(data, out)
        elif fn_name == "sum_mean":
            lines = _sum_and_mean(data, "total", "mean_val")
        elif fn_name == "min_max":
            lines = _min_max(data, "min_val", "max_val")
        elif fn_name == "weighted_sum":
            lines = _weighted_sum(data, weights_name, out)
        elif fn_name == "normalize":
            lines = _normalize(data, out)
        elif fn_name == "clamp_data":
            lines = _clamp_data(data, lo, hi, out)
        elif fn_name == "reshape":
            lines = _reshape(data, dim, out)
        else:
            continue

        pairs.append(_pair(prompt, lines))

    return _maybe_symbolic(pairs[:count])


# ═══════════════════════════════════════════════════════════════════════════════
# Category 8: M2M and Agent Patterns (target: 1,000)
# ═══════════════════════════════════════════════════════════════════════════════

def gen_m2m(count=1000):
    pairs = []

    def _signed_program(inp, out, op_desc, body_lines):
        return [
            "SIGN ed25519 @author_key",
            f'META #version "1.0"',
            f'META #description "{op_desc}"',
        ] + body_lines + [
            _fmt("ST", "R2", f"@{out}"),
            "HALT",
        ]

    def _fragment(name, body_lines):
        return [_fmt("FRAG", name)] + body_lines + ["ENDF"]

    def _compose_frags(name_a, body_a, name_b, body_b):
        return (
            [_fmt("FRAG", name_a)] + body_a + ["ENDF"]
            + [_fmt("FRAG", name_b), _fmt("LINK", f"@{name_a}")] + body_b + ["ENDF"]
        )

    def _vote(inp, out, n):
        return [
            _fmt("LD", "R0", f"@{inp}"),
            _fmt("VOTE", "R1", "R0", "#0", f"#{n}"),
            _fmt("ST", "R1", f"@{out}"),
            "HALT",
        ]

    def _project_embed(inp, proj_name, out):
        return [
            _fmt("LD", "R0", f"@{inp}"),
            _fmt("LD", "R1", f"@{proj_name}"),
            _fmt("PROJ", "R2", "R0", "R1"),
            _fmt("ST", "R2", f"@{out}"),
            "HALT",
        ]

    def _distance(a, b, out, metric=0):
        return [
            _fmt("LD", "R0", f"@{a}"),
            _fmt("LD", "R1", f"@{b}"),
            _fmt("DIST", "R2", "R0", "R1", f"#{metric}"),
            _fmt("ST", "R2", f"@{out}"),
            "HALT",
        ]

    def _gather(data, indices, out):
        return [
            _fmt("LD", "R0", f"@{data}"),
            _fmt("LD", "R1", f"@{indices}"),
            _fmt("GATH", "R2", "R0", "R1"),
            _fmt("ST", "R2", f"@{out}"),
            "HALT",
        ]

    def _scatter(data, indices, out):
        return [
            _fmt("LD", "R0", f"@{data}"),
            _fmt("LD", "R1", f"@{indices}"),
            _fmt("SCAT", "R2", "R0", "R1"),
            _fmt("ST", "R2", f"@{out}"),
            "HALT",
        ]

    def _meta_program(inp, out, desc):
        return [
            f'META #version "1.0"',
            f'META #author "agent"',
            f'META #description "{desc}"',
            _fmt("LD", "R0", f"@{inp}"),
            _fmt("SCLR", "R1", "R0", "#2.0"),
            _fmt("ST", "R1", f"@{out}"),
            "HALT",
        ]

    def _patch(base, set_val, end_val):
        return [
            _fmt("PTCH", f"@{base}", f"@{set_val}", f"@{end_val}"),
        ]

    m2m_prompts = [
        ("Create a signed program that doubles {inp}", "signed"),
        ("Write a signed NML program to scale {inp}", "signed"),
        ("Write a fragment that computes {inp} + offset", "fragment"),
        ("Create a reusable NML fragment for adding to {inp}", "fragment"),
        ("Compose two fragments: scale then add", "compose"),
        ("Link two NML fragments together", "compose"),
        ("Find consensus among values in {inp} (n={n})", "vote"),
        ("Use VOTE to find agreement across {n} values in {inp}", "vote"),
        ("Project {inp} into embedding space using {proj}", "project"),
        ("Embed {inp} via projection matrix {proj}", "project"),
        ("Measure distance between {a} and {b}", "distance"),
        ("Compute L2 distance from {a} to {b}", "distance"),
        ("Look up values at indices from {data}", "gather"),
        ("Gather elements from {data} at {indices}", "gather"),
        ("Write values at indices into {data}", "scatter"),
        ("Scatter {data} to positions given by {indices}", "scatter"),
        ("Self-describing program with META headers for {desc}", "meta"),
        ("Write a META-annotated NML program: {desc}", "meta"),
        ("Patch a program: change base to new value", "patch"),
        ("Apply a PTCH to update program constants", "patch"),
    ]

    descriptions = [
        "data normalization", "feature extraction", "signal processing",
        "vector projection", "anomaly scoring", "embedding lookup",
        "distance computation", "consensus voting", "matrix transform",
        "batch scaling",
    ]

    while len(pairs) < count:
        p_idx = random.randint(0, len(m2m_prompts) - 1)
        prompt_tpl, fn_name = m2m_prompts[p_idx]

        inp = random.choice(INPUT_NAMES)
        out = random.choice(OUTPUT_NAMES)
        a, b = _pick(INPUT_NAMES, 2)
        proj = random.choice(["projection_matrix", "embed_weights", "basis"])
        data = random.choice(["data", "features", "values", "signal"])
        indices = random.choice(["indices", "positions", "lookup_keys"])
        n = random.choice([3, 5, 7, 9, 11])
        desc = random.choice(descriptions)

        prompt = prompt_tpl.format(
            inp=inp, out=out, a=a, b=b, proj=proj,
            data=data, indices=indices, n=n, desc=desc,
        )
        prompt = f"Write NML to: {prompt}"

        if fn_name == "signed":
            body = [
                _fmt("LD", "R0", f"@{inp}"),
                _fmt("SCLR", "R2", "R0", "#2.0"),
            ]
            lines = _signed_program(inp, out, f"double {inp}", body)
        elif fn_name == "fragment":
            body = [
                _fmt("LD", "R0", f"@{inp}"),
                _fmt("LD", "R1", "@offset"),
                _fmt("MADD", "R2", "R0", "R1"),
                _fmt("ST", "R2", f"@{out}"),
            ]
            lines = _fragment("add_offset", body)
        elif fn_name == "compose":
            body_a = [
                _fmt("LD", "R0", f"@{inp}"),
                _fmt("SCLR", "R1", "R0", "#2.0"),
                _fmt("ST", "R1", "@intermediate"),
            ]
            body_b = [
                _fmt("LD", "R2", "@intermediate"),
                _fmt("MADD", "R3", "R2", "#10.0"),
                _fmt("ST", "R3", f"@{out}"),
            ]
            lines = _compose_frags("scale_frag", body_a, "add_frag", body_b)
        elif fn_name == "vote":
            lines = _vote(inp, out, n)
        elif fn_name == "project":
            lines = _project_embed(inp, proj, out)
        elif fn_name == "distance":
            metric = random.choice([0, 1])
            lines = _distance(a, b, out, metric)
        elif fn_name == "gather":
            lines = _gather(data, indices, out)
        elif fn_name == "scatter":
            lines = _scatter(data, indices, out)
        elif fn_name == "meta":
            lines = _meta_program(inp, out, desc)
        elif fn_name == "patch":
            lines = _patch("program_base", "new_constants", "patch_end")
        else:
            continue

        pairs.append(_pair(prompt, lines))

    return _maybe_symbolic(pairs[:count])


# ═══════════════════════════════════════════════════════════════════════════════
# Category 9: Symbolic Syntax Variants (target: 1,000)
# ═══════════════════════════════════════════════════════════════════════════════

def gen_symbolic_variants(source_pairs, count=1000):
    """Convert a random selection of classic-syntax pairs to symbolic."""
    pairs = []
    eligible = [p for p in source_pairs
                if p["messages"][1]["content"].startswith(("LD", "LEAF", "SIGN", "META", "FRAG"))]

    if not eligible:
        eligible = source_pairs

    sampled = random.sample(eligible, min(count * 2, len(eligible)))

    for p in sampled:
        if len(pairs) >= count:
            break

        original_prompt = p["messages"][0]["content"]
        classic_code = p["messages"][1]["content"]
        classic_lines = classic_code.split("\n")

        symbolic_lines = _to_symbolic(classic_lines)
        symbolic_code = "\n".join(symbolic_lines)

        if symbolic_code == classic_code:
            continue

        prefixes = [
            "Write in symbolic NML: ",
            "Symbolic NML for: ",
            "Using symbolic syntax, write NML to: ",
            "In symbolic NML notation: ",
            "Express in symbolic NML: ",
        ]
        prefix = random.choice(prefixes)

        core = original_prompt
        for strip_prefix in ["Write NML to: ", "Write NML for: ", "Write NML for a "]:
            if core.startswith(strip_prefix):
                core = core[len(strip_prefix):]
                break

        new_prompt = prefix + core

        pairs.append(_pair(new_prompt, symbolic_lines))

    return pairs[:count]


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate general-purpose NML training pairs")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit total pairs (scales categories proportionally)")
    args = parser.parse_args()

    targets = {
        "basic_arithmetic": 1000,
        "loops_iteration": 1500,
        "conditionals":    1500,
        "subroutines":     800,
        "neural_nets":     1200,
        "financial":       1000,
        "data_science":    1000,
        "m2m_agent":       1000,
        "symbolic":        1000,
    }

    if args.limit:
        total_default = sum(targets.values())
        scale = args.limit / total_default
        for k in targets:
            targets[k] = max(1, int(targets[k] * scale))

    print("Generating general-purpose NML training pairs...")
    print(f"{'─' * 50}")

    cat1 = gen_basic_arithmetic(targets["basic_arithmetic"])
    print(f"  Category 1  Basic Arithmetic:        {len(cat1):>5}")

    cat2 = gen_loops(targets["loops_iteration"])
    print(f"  Category 2  Loops & Iteration:        {len(cat2):>5}")

    cat3 = gen_conditionals(targets["conditionals"])
    print(f"  Category 3  Conditionals & Branching: {len(cat3):>5}")

    cat4 = gen_subroutines(targets["subroutines"])
    print(f"  Category 4  Subroutines:              {len(cat4):>5}")

    cat5 = gen_neural_nets(targets["neural_nets"])
    print(f"  Category 5  Neural Networks:          {len(cat5):>5}")

    cat6 = gen_financial(targets["financial"])
    print(f"  Category 6  Financial (non-tax):      {len(cat6):>5}")

    cat7 = gen_data_science(targets["data_science"])
    print(f"  Category 7  Data Science:             {len(cat7):>5}")

    cat8 = gen_m2m(targets["m2m_agent"])
    print(f"  Category 8  M2M & Agent Patterns:     {len(cat8):>5}")

    classic_pairs = cat1 + cat2 + cat3 + cat4 + cat5 + cat6 + cat7 + cat8
    cat9 = gen_symbolic_variants(classic_pairs, targets["symbolic"])
    print(f"  Category 9  Symbolic Variants:        {len(cat9):>5}")

    all_pairs = classic_pairs + cat9
    random.shuffle(all_pairs)

    print(f"{'─' * 50}")
    print(f"  TOTAL:                                {len(all_pairs):>5}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\nWritten to: {out_path}")


if __name__ == "__main__":
    main()
