#!/usr/bin/env python3
"""
Generate ~100K NML core training pairs for language-level fine-tuning.

Covers all 67 opcodes across 15 categories with NO domain vocabulary.
Produces pairs in tri-syntax (classic ~60%, symbolic ~25%, verbose ~15%).

Output: domain/output/training/nml_core_pairs.jsonl
Format: {"messages": [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}

Usage:
    python3 nml_core_training_gen.py
    python3 nml_core_training_gen.py --limit 5000
    python3 nml_core_training_gen.py --output /path/to/output.jsonl
"""

import json
import random
import argparse
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent.parent / "domain" / "output" / "training" / "nml_core_pairs.jsonl"

random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════════
# Shared: register, name, and opcode pools
# ═══════════════════════════════════════════════════════════════════════════════

REGS_CLASSIC = ["R0","R1","R2","R3","R4","R5","R6","R7","R8","R9",
                "RA","RB","RC","RD","RE","RF",
                "RG","RH","RI","RJ","RK","RL","RM","RN",
                "RO","RP","RQ","RR","RS","RT","RU","RV"]
REGS_GREEK   = ["ι","κ","λ","μ","ν","ξ","ο","π","ρ","ς","α","β","γ","δ","φ","ψ",
                "η","θ","ζ","ω","χ","υ","ε","RN","RO","RP","RQ","RR","RS","RT","RU","RV"]
REGS_VERBOSE = {
    "RA": "ACCUMULATOR", "RB": "GENERAL", "RC": "SCRATCH",
    "RD": "COUNTER", "RE": "FLAG", "RF": "STACK",
    "RG": "GRADIENT1", "RH": "GRADIENT2", "RI": "GRADIENT3",
    "RJ": "LEARNING_RATE",
}

SYM = {
    "LD":"↓","ST":"↑","MOV":"←","ALLC":"□",
    "MMUL":"×","MADD":"⊕","MSUB":"⊖","EMUL":"⊗",
    "EDIV":"⊘","SDOT":"·","SCLR":"∗","SDIV":"÷",
    "LEAF":"∎","TACC":"∑",
    "RELU":"⌐","SIGM":"σ","TANH":"τ","SOFT":"Σ","GELU":"ℊ",
    "RSHP":"⊞","TRNS":"⊤","SPLT":"⊢","MERG":"⊣",
    "CMPF":"⋈","CMP":"≶","CMPI":"≺",
    "JMPT":"↗","JMPF":"↘","JUMP":"→",
    "LOOP":"↻","ENDP":"↺",
    "CALL":"⇒","RET":"⇐",
    "SYNC":"⏸","HALT":"◼","TRAP":"⚠",
    "CONV":"⊛","POOL":"⊓","UPSC":"⊔","PADZ":"⊡",
    "ATTN":"⊙","NORM":"‖","EMBD":"⊏",
    "RDUC":"⊥","WHER":"⊻","CLMP":"⊧","CMPR":"⊜",
    "FFT":"∿","FILT":"⋐",
    "META":"§","FRAG":"◆","ENDF":"◇","LINK":"⊿",
    "PTCH":"⊿","SIGN":"✦","VRFY":"✓",
    "VOTE":"⚖","PROJ":"⟐","DIST":"⟂",
    "GATH":"⊃","SCAT":"⊂",
    "SYS":"⚙","MOD":"%","ITOF":"⊶","FTOI":"⊷","BNOT":"¬",
    "BKWD":"∇","WUPD":"⟳","LOSS":"△","TNET":"⥁",
}

VERBOSE = {
    "MMUL":"MATRIX_MULTIPLY","MADD":"ADD","MSUB":"SUBTRACT",
    "EMUL":"ELEMENT_MULTIPLY","EDIV":"ELEMENT_DIVIDE",
    "SDOT":"DOT_PRODUCT","SCLR":"SCALE","SDIV":"DIVIDE",
    "SIGM":"SIGMOID","SOFT":"SOFTMAX",
    "LD":"LOAD","ST":"STORE","MOV":"COPY","ALLC":"ALLOCATE",
    "RSHP":"RESHAPE","TRNS":"TRANSPOSE","SPLT":"SPLIT","MERG":"MERGE",
    "LOOP":"REPEAT","ENDP":"END_REPEAT",
    "CMPF":"COMPARE_FEATURE","CMP":"COMPARE","CMPI":"COMPARE_VALUE",
    "LEAF":"SET_VALUE","TACC":"ACCUMULATE",
    "JMPT":"BRANCH_TRUE","JMPF":"BRANCH_FALSE",
    "HALT":"STOP","RET":"RETURN","TRAP":"FAULT","SYNC":"BARRIER",
    "SYS":"SYSTEM","MOD":"MODULO","ITOF":"INT_TO_FLOAT",
    "FTOI":"FLOAT_TO_INT","BNOT":"BITWISE_NOT",
    "BKWD":"BACKPROPAGATE","WUPD":"WEIGHT_UPDATE",
    "LOSS":"COMPUTE_LOSS","TNET":"TRAIN_NETWORK",
}

INPUT_NAMES = [
    "input","value","data","vector","matrix","signal","readings",
    "samples","features","weights","bias","coefficients","kernel",
    "params","tensor","array","sequence","buffer","source","operand",
    "x","y","a","b","factor","scalar","threshold","limit","base",
    "initial","primary","secondary","left","right","top","bottom",
]
OUTPUT_NAMES = [
    "result","output","answer","computed","final","total","sum",
    "product","quotient","difference","response","transformed",
    "processed","filtered","normalized","projected","reduced",
    "accumulated","predicted","classified","score","measurement",
]
MEM_NAMES = INPUT_NAMES + OUTPUT_NAMES + [
    "temp","intermediate","scratch_val","working","buffer_a","buffer_b",
    "layer_out","hidden","state","context","embedding","logits",
]


def _pick(pool, n=1, exclude=()):
    avail = [x for x in pool if x not in exclude]
    return random.sample(avail, min(n, len(avail)))

def _fval(lo=0.01, hi=100.0, decimals=2):
    return round(random.uniform(lo, hi), decimals)

def _ival(lo=1, hi=1000):
    return random.randint(lo, hi)

def _fmt(op, *operands):
    parts = [f"{op:<5}"] + list(operands)
    return " ".join(parts)

def _pair(user_text, nml_lines):
    code = "\n".join(nml_lines) if isinstance(nml_lines, list) else nml_lines
    return {
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": code},
        ]
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Syntax conversion utilities
# ═══════════════════════════════════════════════════════════════════════════════

def to_symbolic(lines):
    result = []
    for line in lines:
        tokens = line.split()
        if not tokens:
            result.append(line)
            continue
        op = tokens[0]
        sym_op = SYM.get(op, op)
        new_operands = []
        for o in tokens[1:]:
            base = o.split(":")[0]
            suffix = ":" + o.split(":",1)[1] if ":" in o else ""
            if base in REGS_CLASSIC:
                idx = REGS_CLASSIC.index(base)
                new_operands.append(REGS_GREEK[idx] + suffix)
            else:
                new_operands.append(o)
        result.append("  ".join([sym_op] + new_operands))
    return result

def to_verbose(lines):
    result = []
    for line in lines:
        tokens = line.split()
        if not tokens:
            result.append(line)
            continue
        op = tokens[0]
        v_op = VERBOSE.get(op, op)
        new_operands = []
        for o in tokens[1:]:
            base = o.split(":")[0]
            suffix = ":" + o.split(":",1)[1] if ":" in o else ""
            if base in REGS_VERBOSE:
                new_operands.append(REGS_VERBOSE[base] + suffix)
            else:
                new_operands.append(o)
        result.append("  ".join([v_op] + new_operands))
    return result

def apply_syntax(lines, syntax):
    if syntax == "symbolic":
        return to_symbolic(lines)
    elif syntax == "verbose":
        return to_verbose(lines)
    return lines

def pick_syntax():
    r = random.random()
    if r < 0.60:
        return "classic"
    elif r < 0.85:
        return "symbolic"
    else:
        return "verbose"

def syntax_tag(syntax):
    if syntax == "symbolic":
        return random.choice([
            " (use symbolic NML syntax)",
            " (symbolic NML)",
            " in symbolic NML",
        ])
    elif syntax == "verbose":
        return random.choice([
            " (use verbose NML syntax)",
            " (verbose NML)",
            " in verbose/human-readable NML",
        ])
    return ""

def _inp():
    return random.choice(INPUT_NAMES)

def _out():
    return random.choice(OUTPUT_NAMES)

def _mem():
    return random.choice(MEM_NAMES)


# ═══════════════════════════════════════════════════════════════════════════════
# Category 1: Opcode Reference (target: 5,000)
# ═══════════════════════════════════════════════════════════════════════════════

OPCODE_INFO = {
    "MMUL": ("Matrix multiply: Rd = Rs1 @ Rs2", "Arithmetic", "Rd Rs1 Rs2"),
    "MADD": ("Element-wise add: Rd = Rs1 + Rs2", "Arithmetic", "Rd Rs1 Rs2"),
    "MSUB": ("Element-wise subtract: Rd = Rs1 - Rs2", "Arithmetic", "Rd Rs1 Rs2"),
    "EMUL": ("Element-wise multiply: Rd = Rs1 * Rs2", "Arithmetic", "Rd Rs1 Rs2"),
    "EDIV": ("Element-wise divide: Rd = Rs1 / Rs2", "Arithmetic", "Rd Rs1 Rs2"),
    "SDOT": ("Scalar dot product: Rd = dot(Rs1, Rs2)", "Arithmetic", "Rd Rs1 Rs2"),
    "SCLR": ("Scalar multiply: Rd = Rs1 * imm", "Arithmetic", "Rd Rs1 #imm"),
    "SDIV": ("Scalar divide: Rd = Rs1 / imm", "Arithmetic", "Rd Rs1 #imm"),
    "RELU": ("ReLU activation: Rd = max(0, Rs)", "Activation", "Rd Rs"),
    "SIGM": ("Sigmoid: Rd = 1/(1+exp(-Rs))", "Activation", "Rd Rs"),
    "TANH": ("Tanh activation: Rd = tanh(Rs)", "Activation", "Rd Rs"),
    "SOFT": ("Softmax across last dimension", "Activation", "Rd Rs"),
    "LD":   ("Load tensor from memory address", "Memory", "Rd @addr"),
    "ST":   ("Store tensor to memory address", "Memory", "Rs @addr"),
    "MOV":  ("Copy register: Rd = Rs", "Memory", "Rd Rs"),
    "ALLC": ("Allocate zero tensor with shape", "Memory", "Rd #shape"),
    "RSHP": ("Reshape tensor (preserves data)", "Data Flow", "Rd Rs #shape"),
    "TRNS": ("Transpose (swap last two dims)", "Data Flow", "Rd Rs"),
    "SPLT": ("Split tensor along dim", "Data Flow", "Rd Re Rs #dim"),
    "MERG": ("Concatenate along dim", "Data Flow", "Rd Rs1 Rs2 #dim"),
    "CMPF": ("Tree feature compare: flag = (Rs[feat] < thresh)", "Comparison", "Rd Rs #feat #thresh"),
    "CMP":  ("General compare: flag = (Rs1[0] < Rs2[0])", "Comparison", "Rs1 Rs2"),
    "CMPI": ("Immediate compare: flag = (Rs[0] < imm)", "Comparison", "Rd Rs #imm"),
    "JMPT": ("Jump by offset if flag is true", "Control", "#offset"),
    "JMPF": ("Jump by offset if flag is false", "Control", "#offset"),
    "JUMP": ("Unconditional jump by offset", "Control", "#offset"),
    "LOOP": ("Begin counted loop", "Control", "#count"),
    "ENDP": ("End loop (branch back if iterations remain)", "Control", ""),
    "CALL": ("Push return address, jump by offset", "Subroutine", "#offset"),
    "RET":  ("Pop return address, resume after CALL", "Subroutine", ""),
    "LEAF": ("Store scalar leaf value: Rd = value", "Tree", "Rd #value"),
    "TACC": ("Tree accumulate: Rd = Rs1 + Rs2 (scalar add)", "Tree", "Rd Rs1 Rs2"),
    "SYNC": ("Barrier / synchronization point", "System", ""),
    "HALT": ("Stop execution", "System", ""),
    "TRAP": ("Trigger program fault with error code", "System", "#code"),
    "CONV": ("2D convolution: Rd = Rs1 conv Rs2", "NML-V", "Rd Rs1 Rs2"),
    "POOL": ("Max pooling", "NML-V", "Rd Rs"),
    "UPSC": ("Nearest-neighbor upscale", "NML-V", "Rd Rs"),
    "PADZ": ("Zero-padding", "NML-V", "Rd Rs"),
    "ATTN": ("Scaled dot-product attention", "NML-T", "Rd Rq Rk [Rv]"),
    "NORM": ("Layer normalization", "NML-T", "Rd Rs"),
    "EMBD": ("Embedding table lookup", "NML-T", "Rd Rtable Rindices"),
    "GELU": ("GELU activation", "NML-T", "Rd Rs"),
    "RDUC": ("Reduce: 0=sum, 1=mean, 2=max, 3=min", "NML-R", "Rd Rs #op"),
    "WHER": ("Conditional select: Rd = cond ? a : b", "NML-R", "Rd Rcond Ra [Rb]"),
    "CLMP": ("Clamp values to [lo, hi]", "NML-R", "Rd Rs #lo #hi"),
    "CMPR": ("Element-wise comparison -> 0/1 mask", "NML-R", "Rd Rs #threshold #op"),
    "FFT":  ("Discrete Fourier Transform", "NML-S", "Rd_real Rd_imag Rs"),
    "FILT": ("FIR filter (1D convolution)", "NML-S", "Rd Rs Rs_coeffs"),
    "META": ("Program metadata (no-op at runtime)", "M2M", "@key value"),
    "FRAG": ("Open named fragment scope", "M2M", "name"),
    "ENDF": ("Close fragment scope", "M2M", ""),
    "LINK": ("Import named fragment inline", "M2M", "@name"),
    "PTCH": ("Differential program patch", "M2M", "@directive args"),
    "SIGN": ("Cryptographic signature", "M2M", "agent=N key=K sig=S"),
    "VRFY": ("Verify signature (TRAP on mismatch)", "M2M", "@hash @signer"),
    "VOTE": ("Multi-agent consensus", "M2M", "Rd Rs #strategy"),
    "PROJ": ("Embedding projection with L2 norm", "M2M", "Rd Rs Rmatrix"),
    "DIST": ("Embedding distance", "M2M", "Rd Rs1 Rs2 #metric"),
    "GATH": ("Gather: tensor index lookup", "M2M", "Rd Rs Ridx"),
    "SCAT": ("Scatter: tensor index write", "M2M", "Rs Rd Ridx"),
    "SYS":  ("System call: print, read, time, rand, exit", "NML-G", "Rd #code"),
    "MOD":  ("Integer modulo: Rd = Rs1 % Rs2", "NML-G", "Rd Rs1 Rs2"),
    "ITOF": ("Integer to float conversion", "NML-G", "Rd Rs"),
    "FTOI": ("Float to integer (truncates)", "NML-G", "Rd Rs"),
    "BNOT": ("Bitwise NOT", "NML-G", "Rd Rs"),
    "BKWD": ("Backpropagation: compute gradient of loss w.r.t. Rs into Rd", "NML-TR", "Rd Rs Rtarget"),
    "WUPD": ("Weight update: Rd = Rs - lr * Rgrad", "NML-TR", "Rd Rs Rgrad"),
    "LOSS": ("Loss computation: mode 0=MSE, 1=cross-entropy, 2=MAE", "NML-TR", "Rd Rs Rtarget"),
    "TNET": ("Self-training loop: train network for N epochs at learning rate", "NML-TR", "#epochs #lr"),
}

def gen_opcode_reference(count=5000):
    pairs = []
    opcodes = list(OPCODE_INFO.keys())

    question_templates = [
        "What does {op} do in NML?",
        "Explain the NML {op} instruction.",
        "What is the purpose of {op}?",
        "How does {op} work in NML?",
        "Describe the {op} opcode.",
    ]

    symbolic_q = [
        "What is the symbolic opcode for {op}?",
        "What Unicode symbol represents {op} in NML?",
        "How do you write {op} in symbolic NML syntax?",
    ]

    example_q = [
        "Write a simple NML example using {op}.",
        "Show me {op} in a short NML program.",
        "Give an example of {op} with context.",
    ]

    while len(pairs) < count:
        op = random.choice(opcodes)
        desc, category, operands = OPCODE_INFO[op]
        sym = SYM.get(op, op)
        verbose = VERBOSE.get(op, op)

        choice = random.random()
        if choice < 0.35:
            q = random.choice(question_templates).format(op=op)
            a = f"{op} {operands} — {desc}. Category: {category}."
            if sym != op:
                a += f" Symbolic: {sym}. Verbose: {verbose}."
        elif choice < 0.55:
            q = random.choice(symbolic_q).format(op=op)
            a = f"The symbolic opcode for {op} ({desc.split(':')[0].lower()}) is {sym}."
            if verbose != op:
                a += f" The verbose form is {verbose}."
        elif choice < 0.80:
            q = random.choice(example_q).format(op=op)
            syntax = pick_syntax()
            lines = _make_opcode_example(op)
            if lines:
                lines = apply_syntax(lines, syntax)
                a = "\n".join(lines)
                q += syntax_tag(syntax)
            else:
                continue
        else:
            q = f"What NML opcode would you use to {_opcode_task(op)}?"
            a = f"{op} — {desc}."

        pairs.append(_pair(q, a))

    return pairs[:count]


def _make_opcode_example(op):
    i, o = _inp(), _out()
    i2 = _pick(INPUT_NAMES, 1, exclude=[i])[0]
    if op in ("MMUL","MADD","MSUB","EMUL","EDIV"):
        return [_fmt("LD","R0",f"@{i}"), _fmt("LD","R1",f"@{i2}"),
                _fmt(op,"R2","R0","R1"), _fmt("ST","R2",f"@{o}"), "HALT"]
    if op in ("SDOT",):
        return [_fmt("LD","R0",f"@{i}"), _fmt("LD","R1",f"@{i2}"),
                _fmt("SDOT","R2","R0","R1"), _fmt("ST","R2",f"@{o}"), "HALT"]
    if op in ("SCLR","SDIV"):
        v = f"#{_fval(0.1,10.0)}"
        return [_fmt("LD","R0",f"@{i}"), _fmt(op,"R1","R0",v),
                _fmt("ST","R1",f"@{o}"), "HALT"]
    if op in ("RELU","SIGM","TANH","GELU"):
        return [_fmt("LD","R0",f"@{i}"), _fmt(op,"R1","R0"),
                _fmt("ST","R1",f"@{o}"), "HALT"]
    if op == "SOFT":
        return [_fmt("LD","R0",f"@{i}"), _fmt("SOFT","R1","R0"),
                _fmt("ST","R1",f"@{o}"), "HALT"]
    if op == "LD":
        return [_fmt("LD","R0",f"@{i}"), _fmt("ST","R0",f"@{o}"), "HALT"]
    if op == "ST":
        return [_fmt("LD","R0",f"@{i}"), _fmt("ST","R0",f"@{o}"), "HALT"]
    if op == "MOV":
        return [_fmt("LD","R0",f"@{i}"), _fmt("MOV","R1","R0"),
                _fmt("ST","R1",f"@{o}"), "HALT"]
    if op == "ALLC":
        s = random.choice(["#[1]","#[4]","#[3,3]","#[8]"])
        return [_fmt("ALLC","R0",s), _fmt("ST","R0",f"@{o}"), "HALT"]
    if op == "RSHP":
        s = random.choice(["#[4]","#[2,2]","#[1,8]"])
        return [_fmt("LD","R0",f"@{i}"), _fmt("RSHP","R1","R0",s),
                _fmt("ST","R1",f"@{o}"), "HALT"]
    if op == "TRNS":
        return [_fmt("LD","R0",f"@{i}"), _fmt("TRNS","R1","R0"),
                _fmt("ST","R1",f"@{o}"), "HALT"]
    if op in ("LEAF","TACC"):
        v = _fval(1,500)
        return [_fmt("LEAF","RA",f"#{v}"), _fmt("LEAF","RC","#1.0"),
                _fmt("TACC","RA","RA","RC"), _fmt("ST","RA",f"@{o}"), "HALT"]
    if op in ("CMPF","CMPI"):
        v = _fval(1,100)
        return [_fmt("LD","R0",f"@{i}"), _fmt("CMPI","RE","R0",f"#{v}"),
                _fmt("JMPT","#2"), _fmt("ST","R0",f"@{o}"), "HALT"]
    if op == "CMP":
        return [_fmt("LD","R0",f"@{i}"), _fmt("LD","R1",f"@{i2}"),
                _fmt("CMP","R0","R1"), _fmt("JMPT","#2"),
                _fmt("ST","R1",f"@{o}"), "HALT",
                _fmt("ST","R0",f"@{o}"), "HALT"]
    if op in ("JMPT","JMPF","JUMP"):
        return [_fmt("LD","R0",f"@{i}"), _fmt("CMPI","RE","R0","#50.0"),
                _fmt(op,"#2"), _fmt("SCLR","R0","R0","#2.0"),
                _fmt("ST","R0",f"@{o}"), "HALT"]
    if op == "LOOP":
        return [_fmt("ALLC","R0","#[1]"), _fmt("LEAF","R1","#1.0"),
                _fmt("LEAF","R2","#5"), _fmt("LOOP","R2"),
                _fmt("TACC","R0","R0","R1"), "ENDP",
                _fmt("ST","R0",f"@{o}"), "HALT"]
    if op == "ENDP":
        return [_fmt("LEAF","R0","#3"), _fmt("LOOP","R0"),
                _fmt("LEAF","R1","#1.0"), "ENDP",
                _fmt("ST","R1",f"@{o}"), "HALT"]
    if op == "CALL":
        return [_fmt("LD","R0",f"@{i}"), _fmt("CALL","#2"),
                _fmt("ST","R0",f"@{o}"), "HALT",
                _fmt("SCLR","R0","R0","#2.0"), "RET"]
    if op == "RET":
        return [_fmt("LEAF","R0","#42.0"), _fmt("CALL","#2"),
                _fmt("ST","R0",f"@{o}"), "HALT",
                _fmt("SCLR","R0","R0","#2.0"), "RET"]
    if op == "HALT":
        return [_fmt("LD","R0",f"@{i}"), _fmt("ST","R0",f"@{o}"), "HALT"]
    if op == "TRAP":
        return [_fmt("LD","R0",f"@{i}"), _fmt("CMPI","RE","R0","#0"),
                _fmt("JMPT","#2"), _fmt("TRAP","#1"),
                _fmt("ST","R0",f"@{o}"), "HALT"]
    if op == "SYNC":
        return [_fmt("LD","R0",f"@{i}"), "SYNC",
                _fmt("ST","R0",f"@{o}"), "HALT"]
    if op in ("CONV","POOL","UPSC","PADZ"):
        if op == "CONV":
            return [_fmt("LD","R0",f"@{i}"), _fmt("LD","R1",f"@{i2}"),
                    _fmt("CONV","R2","R0","R1"), _fmt("ST","R2",f"@{o}"), "HALT"]
        return [_fmt("LD","R0",f"@{i}"), _fmt(op,"R1","R0"),
                _fmt("ST","R1",f"@{o}"), "HALT"]
    if op == "ATTN":
        return [_fmt("LD","R0",f"@{i}"), _fmt("LD","R1","@keys"),
                _fmt("LD","R2","@values"),
                _fmt("ATTN","R3","R0","R1","R2"),
                _fmt("ST","R3",f"@{o}"), "HALT"]
    if op == "NORM":
        return [_fmt("LD","R0",f"@{i}"), _fmt("NORM","R1","R0"),
                _fmt("ST","R1",f"@{o}"), "HALT"]
    if op == "EMBD":
        return [_fmt("LD","R0","@indices"), _fmt("LD","R1","@table"),
                _fmt("EMBD","R2","R1","R0"), _fmt("ST","R2",f"@{o}"), "HALT"]
    if op == "RDUC":
        r_op = random.choice(["#0","#1","#2","#3"])
        return [_fmt("LD","R0",f"@{i}"), _fmt("RDUC","R1","R0",r_op),
                _fmt("ST","R1",f"@{o}"), "HALT"]
    if op == "WHER":
        return [_fmt("LD","R0","@condition"), _fmt("LD","R1",f"@{i}"),
                _fmt("LD","R2",f"@{i2}"),
                _fmt("WHER","R3","R0","R1","R2"),
                _fmt("ST","R3",f"@{o}"), "HALT"]
    if op == "CLMP":
        lo, hi = _fval(0,10), _fval(50,200)
        return [_fmt("LD","R0",f"@{i}"), _fmt("CLMP","R1","R0",f"#{lo}",f"#{hi}"),
                _fmt("ST","R1",f"@{o}"), "HALT"]
    if op == "CMPR":
        return [_fmt("LD","R0",f"@{i}"),
                _fmt("CMPR","R1","R0",f"#{_fval(0,100)}","#5"),
                _fmt("ST","R1",f"@{o}"), "HALT"]
    if op == "FFT":
        return [_fmt("LD","R0",f"@{i}"), _fmt("FFT","R1","R2","R0"),
                _fmt("ST","R1","@real_part"), _fmt("ST","R2","@imag_part"), "HALT"]
    if op == "FILT":
        return [_fmt("LD","R0",f"@{i}"), _fmt("LD","R1","@coefficients"),
                _fmt("FILT","R2","R0","R1"), _fmt("ST","R2",f"@{o}"), "HALT"]
    if op in ("META","FRAG","ENDF","LINK","PTCH","SIGN","VRFY"):
        return [f'META  @name "example"', f'META  @version "1.0"',
                _fmt("LD","R0",f"@{i}"), _fmt("ST","R0",f"@{o}"), "HALT"]
    if op == "VOTE":
        return [_fmt("LD","R0",f"@{i}"), _fmt("VOTE","R1","R0","#0"),
                _fmt("ST","R1",f"@{o}"), "HALT"]
    if op in ("PROJ","DIST"):
        return [_fmt("LD","R0",f"@{i}"), _fmt("LD","R1",f"@{i2}"),
                _fmt(op,"R2","R0","R1"), _fmt("ST","R2",f"@{o}"), "HALT"]
    if op in ("GATH","SCAT"):
        return [_fmt("LD","R0",f"@{i}"), _fmt("LD","R1","@indices"),
                _fmt(op,"R2","R0","R1"), _fmt("ST","R2",f"@{o}"), "HALT"]
    if op == "SYS":
        code = random.choice(["#0","#1","#5"])
        return [_fmt("LEAF","R0",f"#{_fval(1,100)}"),
                _fmt("SYS","R0",code), "HALT"]
    if op == "MOD":
        return [_fmt("LD","R0",f"@{i}"), _fmt("LD","R1",f"@{i2}"),
                _fmt("MOD","R2","R0","R1"), _fmt("ST","R2",f"@{o}"), "HALT"]
    if op in ("ITOF","FTOI","BNOT"):
        return [_fmt("LD","R0",f"@{i}"), _fmt(op,"R1","R0"),
                _fmt("ST","R1",f"@{o}"), "HALT"]
    if op == "BKWD":
        return [_fmt("LD","R0",f"@{i}"), _fmt("LD","R1","@weights"),
                _fmt("LD","R9","@targets"), _fmt("MMUL","R2","R0","R1"),
                _fmt("BKWD","RG","R2","R9"),
                _fmt("ST","RG",f"@{o}"), "HALT"]
    if op == "WUPD":
        return [_fmt("LD","R1","@weights"), _fmt("LD","RG","@gradients"),
                _fmt("LEAF","RJ","#0.01"),
                _fmt("WUPD","R1","R1","RG"),
                _fmt("ST","R1",f"@{o}"), "HALT"]
    if op == "LOSS":
        mode = random.choice(["#0","#1","#2"])
        return [_fmt("LD","R0","@predictions"), _fmt("LD","R9","@targets"),
                _fmt("LOSS","RG","R0","R9",mode),
                _fmt("ST","RG",f"@{o}"), "HALT"]
    if op == "TNET":
        epochs = random.choice([100,500,1000,2000])
        lr = random.choice(["0.01","0.001","0.1","0.005"])
        return [_fmt("LD","R0",f"@{i}"), _fmt("LD","R9","@targets"),
                _fmt("LD","R1","@w1"), _fmt("LD","R2","@b1"),
                _fmt("LD","R3","@w2"), _fmt("LD","R4","@b2"),
                _fmt("TNET",f"#{epochs}",f"#{lr}"),
                _fmt("ST","RA",f"@{o}"), "HALT"]
    if op == "SPLT":
        return [_fmt("LD","R0",f"@{i}"),
                _fmt("SPLT","R1","R2","R0","#0"),
                _fmt("ST","R1","@lower"), _fmt("ST","R2","@upper"), "HALT"]
    if op == "MERG":
        return [_fmt("LD","R0",f"@{i}"), _fmt("LD","R1",f"@{i2}"),
                _fmt("MERG","R2","R0","R1","#0"),
                _fmt("ST","R2",f"@{o}"), "HALT"]
    return None


def _opcode_task(op):
    tasks = {
        "MMUL": "multiply two matrices",
        "MADD": "add two tensors element-wise",
        "MSUB": "subtract one tensor from another",
        "EMUL": "multiply tensors element-wise",
        "EDIV": "divide tensors element-wise",
        "SDOT": "compute a dot product",
        "SCLR": "multiply a tensor by a constant",
        "SDIV": "divide a tensor by a constant",
        "RELU": "apply ReLU activation",
        "SIGM": "apply sigmoid activation",
        "TANH": "apply tanh activation",
        "SOFT": "apply softmax",
        "GELU": "apply GELU activation",
        "LD":   "load data from memory",
        "ST":   "store a result to memory",
        "MOV":  "copy one register to another",
        "ALLC": "allocate a zero tensor",
        "RSHP": "reshape a tensor",
        "TRNS": "transpose a matrix",
        "SPLT": "split a tensor along a dimension",
        "MERG": "concatenate two tensors",
        "CMPF": "compare a feature against a threshold",
        "CMP":  "compare two register values",
        "CMPI": "compare a register against a constant",
        "JMPT": "branch when a condition is true",
        "JMPF": "branch when a condition is false",
        "JUMP": "jump unconditionally",
        "LOOP": "start a counted loop",
        "ENDP": "end a counted loop",
        "CALL": "call a subroutine",
        "RET":  "return from a subroutine",
        "LEAF": "set a register to a constant value",
        "TACC": "accumulate (add) two scalar values",
        "HALT": "stop program execution",
        "TRAP": "trigger an error",
        "SYNC": "insert a synchronization barrier",
        "CONV": "apply 2D convolution",
        "POOL": "downsample with max pooling",
        "UPSC": "upscale a tensor",
        "PADZ": "zero-pad a tensor",
        "ATTN": "compute attention",
        "NORM": "normalize a tensor",
        "EMBD": "look up embeddings",
        "RDUC": "reduce a tensor (sum/mean/max/min)",
        "WHER": "conditionally select between two values",
        "CLMP": "clamp values to a range",
        "CMPR": "create a comparison mask",
        "FFT":  "compute a Fourier transform",
        "FILT": "apply a FIR filter",
        "META": "add program metadata",
        "FRAG": "define a reusable code fragment",
        "ENDF": "close a fragment",
        "LINK": "import a fragment",
        "PTCH": "patch a program",
        "SIGN": "sign a program cryptographically",
        "VRFY": "verify a program signature",
        "VOTE": "find consensus among values",
        "PROJ": "project into embedding space",
        "DIST": "compute distance between embeddings",
        "GATH": "gather elements by index",
        "SCAT": "scatter elements by index",
        "SYS":  "perform system I/O",
        "MOD":  "compute integer modulo",
        "ITOF": "convert integer to float",
        "FTOI": "convert float to integer",
        "BNOT": "bitwise NOT",
        "BKWD": "compute gradients via backpropagation",
        "WUPD": "update weights using gradients",
        "LOSS": "compute the loss between predictions and targets",
        "TNET": "train a neural network end-to-end",
    }
    return tasks.get(op, f"use {op}")


# ═══════════════════════════════════════════════════════════════════════════════
# Category 2: Basic Arithmetic (target: 5,000)
# ═══════════════════════════════════════════════════════════════════════════════

def gen_basic_arithmetic(count=5000):
    pairs = []

    def _binary_op(op_name, prompt_verb):
        a, b = _pick(INPUT_NAMES, 2)
        o = _out()
        syntax = pick_syntax()
        q = f"Write NML to {prompt_verb} {a} and {b}{syntax_tag(syntax)}"
        lines = [_fmt("LD","R0",f"@{a}"), _fmt("LD","R1",f"@{b}"),
                 _fmt(op_name,"R2","R0","R1"), _fmt("ST","R2",f"@{o}"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    def _scale():
        a = _inp(); o = _out(); c = _fval(0.01, 50.0)
        syntax = pick_syntax()
        prompts = [f"Scale {a} by {c}", f"Multiply {a} by the constant {c}",
                   f"Compute {a} * {c}"]
        q = f"Write NML to: {random.choice(prompts)}{syntax_tag(syntax)}"
        lines = [_fmt("LD","R0",f"@{a}"), _fmt("SCLR","R1","R0",f"#{c}"),
                 _fmt("ST","R1",f"@{o}"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    def _divide():
        a = _inp(); o = _out(); c = _fval(1, 100)
        syntax = pick_syntax()
        q = f"Write NML to divide {a} by {c}{syntax_tag(syntax)}"
        lines = [_fmt("LD","R0",f"@{a}"), _fmt("SDIV","R1","R0",f"#{c}"),
                 _fmt("ST","R1",f"@{o}"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    def _chained():
        a, b = _pick(INPUT_NAMES, 2); o = _out(); c = _fval(0.1, 10.0)
        syntax = pick_syntax()
        patterns = [
            (f"Compute ({a} + {b}) * {c}",
             [_fmt("LD","R0",f"@{a}"), _fmt("LD","R1",f"@{b}"),
              _fmt("MADD","R2","R0","R1"), _fmt("SCLR","R3","R2",f"#{c}"),
              _fmt("ST","R3",f"@{o}"), "HALT"]),
            (f"Compute {a} * {b} - {a}",
             [_fmt("LD","R0",f"@{a}"), _fmt("LD","R1",f"@{b}"),
              _fmt("EMUL","R2","R0","R1"), _fmt("MSUB","R3","R2","R0"),
              _fmt("ST","R3",f"@{o}"), "HALT"]),
            (f"Compute ({a} - {b}) / {c}",
             [_fmt("LD","R0",f"@{a}"), _fmt("LD","R1",f"@{b}"),
              _fmt("MSUB","R2","R0","R1"), _fmt("SDIV","R3","R2",f"#{c}"),
              _fmt("ST","R3",f"@{o}"), "HALT"]),
            (f"Compute dot product of {a} and {b}, then scale by {c}",
             [_fmt("LD","R0",f"@{a}"), _fmt("LD","R1",f"@{b}"),
              _fmt("SDOT","R2","R0","R1"), _fmt("SCLR","R3","R2",f"#{c}"),
              _fmt("ST","R3",f"@{o}"), "HALT"]),
            (f"Average of {a} and {b}",
             [_fmt("LD","R0",f"@{a}"), _fmt("LD","R1",f"@{b}"),
              _fmt("MADD","R2","R0","R1"), _fmt("SDIV","R3","R2","#2.0"),
              _fmt("ST","R3",f"@{o}"), "HALT"]),
        ]
        prompt, lines = random.choice(patterns)
        q = f"Write NML to: {prompt}{syntax_tag(syntax)}"
        return _pair(q, apply_syntax(lines, syntax))

    generators = [
        (0.15, lambda: _binary_op("MADD", random.choice(["add","compute the sum of"]))),
        (0.10, lambda: _binary_op("MSUB", random.choice(["subtract","compute the difference of"]))),
        (0.10, lambda: _binary_op("EMUL", random.choice(["multiply element-wise","element-multiply"]))),
        (0.10, lambda: _binary_op("EDIV", random.choice(["divide element-wise","element-divide"]))),
        (0.08, lambda: _binary_op("MMUL", random.choice(["matrix multiply","compute the matrix product of"]))),
        (0.08, lambda: _binary_op("SDOT", random.choice(["compute the dot product of","dot-product"]))),
        (0.12, _scale),
        (0.07, _divide),
        (0.20, _chained),
    ]

    weights = [g[0] for g in generators]
    fns = [g[1] for g in generators]

    while len(pairs) < count:
        fn = random.choices(fns, weights=weights, k=1)[0]
        pairs.append(fn())

    return pairs[:count]


# ═══════════════════════════════════════════════════════════════════════════════
# Category 3: Loops and Iteration (target: 8,000)
# ═══════════════════════════════════════════════════════════════════════════════

def gen_loops(count=8000):
    pairs = []
    loop_ns = [3, 5, 7, 10, 12, 15, 20, 25, 30, 50, 100]

    def _sum_1_to_n():
        n = random.choice(loop_ns); o = _out(); syntax = pick_syntax()
        q = f"Write NML to sum the integers from 1 to {n}{syntax_tag(syntax)}"
        lines = [_fmt("LEAF","R0",f"#{n}"), _fmt("ALLC","R1","#[1]"),
                 _fmt("LEAF","R2","#1"), _fmt("LOOP","R0"),
                 _fmt("TACC","R1","R1","R2"), "ENDP",
                 _fmt("ST","R1",f"@{o}"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    def _factorial():
        n = random.choice([3,4,5,6,7,8,10]); o = _out(); syntax = pick_syntax()
        q = f"Write NML to compute {n} factorial{syntax_tag(syntax)}"
        lines = [_fmt("LEAF","R0",f"#{n}"), _fmt("LEAF","R1","#1"),
                 _fmt("LEAF","R2","#1"), _fmt("LOOP","R0"),
                 _fmt("EMUL","R1","R1","R2"), _fmt("LEAF","RC","#1.0"),
                 _fmt("TACC","R2","R2","RC"), "ENDP",
                 _fmt("ST","R1",f"@{o}"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    def _power():
        i = _inp(); n = random.choice([2,3,4,5,8]); o = _out(); syntax = pick_syntax()
        q = f"Write NML to raise {i} to the power {n}{syntax_tag(syntax)}"
        lines = [_fmt("LD","R0",f"@{i}"), _fmt("MOV","R1","R0"),
                 _fmt("LEAF","R2",f"#{n-1}"), _fmt("LOOP","R2"),
                 _fmt("EMUL","R1","R1","R0"), "ENDP",
                 _fmt("ST","R1",f"@{o}"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    def _compound():
        i = _inp(); r = _pick(INPUT_NAMES,1,exclude=[i])[0]
        n = random.choice([5,10,12,20,30]); o = _out(); syntax = pick_syntax()
        q = f"Write NML to compound {i} at rate {r} for {n} periods{syntax_tag(syntax)}"
        lines = [_fmt("LD","R0",f"@{i}"), _fmt("LD","R1",f"@{r}"),
                 _fmt("MADD","R1","R1","#1.0"), _fmt("MOV","R2","R0"),
                 _fmt("LEAF","R3",f"#{n}"), _fmt("LOOP","R3"),
                 _fmt("EMUL","R2","R2","R1"), "ENDP",
                 _fmt("ST","R2",f"@{o}"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    def _reduce_op():
        i = _inp(); o = _out(); syntax = pick_syntax()
        ops = [("#0","sum","sum"),("#1","mean","average"),
               ("#2","max","maximum"),("#3","min","minimum")]
        code, name, desc = random.choice(ops)
        q = f"Write NML to compute the {desc} of {i}{syntax_tag(syntax)}"
        lines = [_fmt("LD","R0",f"@{i}"), _fmt("RDUC","R1","R0",code),
                 _fmt("ST","R1",f"@{o}"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    def _accumulate():
        i = _inp(); n = random.choice(loop_ns); o = _out(); syntax = pick_syntax()
        q = f"Write NML to accumulate {i} over {n} iterations{syntax_tag(syntax)}"
        lines = [_fmt("LD","R0",f"@{i}"), _fmt("ALLC","R1","#[1]"),
                 _fmt("LEAF","R2",f"#{n}"), _fmt("LOOP","R2"),
                 _fmt("TACC","R1","R1","R0"), "ENDP",
                 _fmt("ST","R1",f"@{o}"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    def _repeated_scale():
        i = _inp(); n = random.choice([3,5,10,20]); o = _out(); syntax = pick_syntax()
        f_val = _fval(0.5, 2.0)
        q = f"Write NML to scale {i} by {f_val} repeatedly {n} times{syntax_tag(syntax)}"
        lines = [_fmt("LD","R0",f"@{i}"), _fmt("LEAF","R1",f"#{n}"),
                 _fmt("LOOP","R1"), _fmt("SCLR","R0","R0",f"#{f_val}"),
                 "ENDP", _fmt("ST","R0",f"@{o}"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    def _backward_jump():
        n = random.choice([5,10,20]); o = _out(); syntax = pick_syntax()
        q = f"Write NML to sum 1 to {n} using backward jumps (while loop){syntax_tag(syntax)}"
        lines = [_fmt("LEAF","RA","#0.0"), _fmt("LEAF","RD","#1.0"),
                 _fmt("TACC","RA","RA","RD"), _fmt("LEAF","RC","#1.0"),
                 _fmt("TACC","RD","RD","RC"),
                 _fmt("CMPI","RE","RD",f"#{n+1}.0"),
                 _fmt("JMPT","#-5"),
                 _fmt("ST","RA",f"@{o}"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    generators = [
        (0.15, _sum_1_to_n), (0.10, _factorial), (0.12, _power),
        (0.12, _compound), (0.15, _reduce_op), (0.10, _accumulate),
        (0.12, _repeated_scale), (0.14, _backward_jump),
    ]
    weights = [g[0] for g in generators]; fns = [g[1] for g in generators]
    while len(pairs) < count:
        pairs.append(random.choices(fns, weights=weights, k=1)[0]())
    return pairs[:count]


# ═══════════════════════════════════════════════════════════════════════════════
# Category 4: Conditionals and Branching (target: 8,000)
# ═══════════════════════════════════════════════════════════════════════════════

def gen_conditionals(count=8000):
    pairs = []

    def _if_gt_scale():
        i = _inp(); o = _out(); syntax = pick_syntax()
        thresh = _fval(10, 5000); rA = _fval(0.01, 0.99); rB = _fval(0.01, 0.99)
        q = f"Write NML to: if {i} > {thresh}, scale by {rA}, else scale by {rB}{syntax_tag(syntax)}"
        lines = [_fmt("LD","R0",f"@{i}"), _fmt("CMPI","RE","R0",f"#{thresh}"),
                 _fmt("JMPF","#3"), _fmt("SCLR","R1","R0",f"#{rA}"),
                 _fmt("JUMP","#2"), _fmt("SCLR","R1","R0",f"#{rB}"),
                 _fmt("ST","R1",f"@{o}"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    def _clamp():
        i = _inp(); o = _out(); syntax = pick_syntax()
        lo = _fval(0, 100); hi = _fval(lo + 10, lo + 10000)
        q = f"Write NML to clamp {i} between {lo} and {hi}{syntax_tag(syntax)}"
        lines = [_fmt("LD","R0",f"@{i}"), _fmt("CLMP","R1","R0",f"#{lo}",f"#{hi}"),
                 _fmt("ST","R1",f"@{o}"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    def _three_tier():
        i = _inp(); o = _out(); syntax = pick_syntax()
        t1 = _fval(10, 500); t2 = round(t1 + random.uniform(100, 5000), 2)
        r1 = _fval(0.01, 0.5); r2 = _fval(0.01, 0.5); r3 = _fval(0.01, 0.5)
        q = f"Write NML for three-tier: below {t1} at {r1}, {t1}-{t2} at {r2}, above {t2} at {r3}{syntax_tag(syntax)}"
        lines = [_fmt("LD","R0",f"@{i}"),
                 _fmt("CMPI","RE","R0",f"#{t2}"), _fmt("JMPF","#3"),
                 _fmt("SCLR","R1","R0",f"#{r3}"), _fmt("JUMP","#6"),
                 _fmt("CMPI","RE","R0",f"#{t1}"), _fmt("JMPF","#3"),
                 _fmt("SCLR","R1","R0",f"#{r2}"), _fmt("JUMP","#2"),
                 _fmt("SCLR","R1","R0",f"#{r1}"),
                 _fmt("ST","R1",f"@{o}"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    def _abs_value():
        i = _inp(); o = _out(); syntax = pick_syntax()
        q = f"Write NML to compute absolute value of {i}{syntax_tag(syntax)}"
        lines = [_fmt("LD","R0",f"@{i}"), _fmt("CMPI","RE","R0","#0"),
                 _fmt("JMPT","#2"), _fmt("SCLR","R0","R0","#-1.0"),
                 _fmt("ST","R0",f"@{o}"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    def _max_of_two():
        a, b = _pick(INPUT_NAMES, 2); o = _out(); syntax = pick_syntax()
        q = f"Write NML to return the larger of {a} and {b}{syntax_tag(syntax)}"
        lines = [_fmt("LD","R0",f"@{a}"), _fmt("LD","R1",f"@{b}"),
                 _fmt("CMP","R0","R1"), _fmt("JMPF","#3"),
                 _fmt("MOV","R2","R0"), _fmt("JUMP","#2"),
                 _fmt("MOV","R2","R1"), _fmt("ST","R2",f"@{o}"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    def _min_of_two():
        a, b = _pick(INPUT_NAMES, 2); o = _out(); syntax = pick_syntax()
        q = f"Write NML to return the smaller of {a} and {b}{syntax_tag(syntax)}"
        lines = [_fmt("LD","R0",f"@{a}"), _fmt("LD","R1",f"@{b}"),
                 _fmt("CMP","R0","R1"), _fmt("JMPT","#3"),
                 _fmt("MOV","R2","R0"), _fmt("JUMP","#2"),
                 _fmt("MOV","R2","R1"), _fmt("ST","R2",f"@{o}"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    def _zero_guard():
        i = _inp(); o = _out(); syntax = pick_syntax()
        default = _fval(1, 1000)
        q = f"Write NML to: if {i} is zero use default {default}, else keep {i}{syntax_tag(syntax)}"
        lines = [_fmt("LD","R0",f"@{i}"), _fmt("CMPI","RE","R0","#0"),
                 _fmt("JMPF","#3"), _fmt("LEAF","R1",f"#{default}"),
                 _fmt("JUMP","#2"), _fmt("MOV","R1","R0"),
                 _fmt("ST","R1",f"@{o}"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    def _wher_select():
        i = _inp(); i2 = _pick(INPUT_NAMES, 1, exclude=[i])[0]
        o = _out(); syntax = pick_syntax()
        q = f"Write NML to conditionally select between {i} and {i2} based on a condition mask{syntax_tag(syntax)}"
        lines = [_fmt("LD","R0","@condition"), _fmt("LD","R1",f"@{i}"),
                 _fmt("LD","R2",f"@{i2}"),
                 _fmt("WHER","R3","R0","R1","R2"),
                 _fmt("ST","R3",f"@{o}"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    generators = [
        (0.18, _if_gt_scale), (0.12, _clamp), (0.15, _three_tier),
        (0.10, _abs_value), (0.12, _max_of_two), (0.10, _min_of_two),
        (0.10, _zero_guard), (0.13, _wher_select),
    ]
    weights = [g[0] for g in generators]; fns = [g[1] for g in generators]
    while len(pairs) < count:
        pairs.append(random.choices(fns, weights=weights, k=1)[0]())
    return pairs[:count]


# ═══════════════════════════════════════════════════════════════════════════════
# Category 5: Subroutines (target: 5,000)
# ═══════════════════════════════════════════════════════════════════════════════

def gen_subroutines(count=5000):
    pairs = []

    def _single_sub(body_op, body_arg, desc):
        i = _inp(); o = _out(); syntax = pick_syntax()
        q = f"Write NML to {desc} {i} using a subroutine{syntax_tag(syntax)}"
        lines = [_fmt("LD","R0",f"@{i}"), _fmt("CALL","#2"),
                 _fmt("ST","R0",f"@{o}"), "HALT",
                 _fmt(body_op,"R0","R0",body_arg), "RET"]
        return _pair(q, apply_syntax(lines, syntax))

    def _two_input_sub():
        a, b = _pick(INPUT_NAMES, 2); o = _out(); syntax = pick_syntax()
        op = random.choice([("MADD","add"),("EMUL","multiply"),("MSUB","subtract")])
        q = f"Write NML subroutine to {op[1]} {a} and {b}{syntax_tag(syntax)}"
        lines = [_fmt("LD","R0",f"@{a}"), _fmt("LD","R1",f"@{b}"),
                 _fmt("CALL","#2"), _fmt("ST","R2",f"@{o}"), "HALT",
                 _fmt(op[0],"R2","R0","R1"), "RET"]
        return _pair(q, apply_syntax(lines, syntax))

    def _chained_sub():
        i = _inp(); o = _out(); syntax = pick_syntax()
        f1 = _fval(0.5, 5.0); f2 = _fval(1, 100)
        q = f"Write NML to chain two subroutines: scale by {f1} then add {f2}{syntax_tag(syntax)}"
        lines = [_fmt("LD","R0",f"@{i}"), _fmt("CALL","#3"),
                 _fmt("CALL","#4"), _fmt("ST","R0",f"@{o}"), "HALT",
                 _fmt("SCLR","R0","R0",f"#{f1}"), "RET",
                 _fmt("LEAF","R1",f"#{f2}"),
                 _fmt("TACC","R0","R0","R1"), "RET"]
        return _pair(q, apply_syntax(lines, syntax))

    while len(pairs) < count:
        r = random.random()
        if r < 0.25:
            pairs.append(_single_sub("SCLR", "#2.0", "double"))
        elif r < 0.40:
            pairs.append(_single_sub("SCLR", "#3.0", "triple"))
        elif r < 0.50:
            pairs.append(_single_sub("SCLR", "#-1.0", "negate"))
        elif r < 0.60:
            pairs.append(_single_sub("SCLR", f"#{_fval(0.1,10.0)}", f"scale"))
        elif r < 0.70:
            pairs.append(_single_sub("EMUL", "R0", "square"))
        elif r < 0.85:
            pairs.append(_two_input_sub())
        else:
            pairs.append(_chained_sub())

    return pairs[:count]


# ═══════════════════════════════════════════════════════════════════════════════
# Category 6: Neural Networks (target: 10,000)
# ═══════════════════════════════════════════════════════════════════════════════

def gen_neural_nets(count=10000):
    pairs = []
    acts = ["RELU","SIGM","TANH","GELU"]
    act_names = {"RELU":"ReLU","SIGM":"sigmoid","TANH":"tanh","GELU":"GELU"}

    def _one_layer():
        act = random.choice(acts); syntax = pick_syntax()
        i = _inp(); o = _out()
        q = f"Write NML for a single dense layer with {act_names[act]} activation{syntax_tag(syntax)}"
        lines = [_fmt("LD","R0",f"@{i}"), _fmt("LD","R1","@weights"),
                 _fmt("LD","R2","@bias"), _fmt("MMUL","R3","R0","R1"),
                 _fmt("MADD","R3","R3","R2"), _fmt(act,"R3","R3"),
                 _fmt("ST","R3",f"@{o}"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    def _two_layer():
        a1 = random.choice(acts); a2 = random.choice(acts); syntax = pick_syntax()
        i = _inp(); o = _out()
        q = f"Write NML for a 2-layer network with {act_names[a1]} and {act_names[a2]}{syntax_tag(syntax)}"
        lines = [_fmt("LD","R0",f"@{i}"),
                 _fmt("LD","R1","@w1"), _fmt("LD","R2","@b1"),
                 _fmt("MMUL","R3","R0","R1"), _fmt("MADD","R3","R3","R2"),
                 _fmt(a1,"R3","R3"),
                 _fmt("LD","R4","@w2"), _fmt("LD","R5","@b2"),
                 _fmt("MMUL","R6","R3","R4"), _fmt("MADD","R6","R6","R5"),
                 _fmt(a2,"R6","R6"),
                 _fmt("ST","R6",f"@{o}"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    def _three_layer():
        act = random.choice(acts); syntax = pick_syntax()
        i = _inp(); o = _out()
        q = f"Write NML for a 3-layer network with {act_names[act]} hidden and sigmoid output{syntax_tag(syntax)}"
        lines = [_fmt("LD","R0",f"@{i}"),
                 _fmt("LD","R1","@w1"), _fmt("LD","R2","@b1"),
                 _fmt("MMUL","R3","R0","R1"), _fmt("MADD","R3","R3","R2"),
                 _fmt(act,"R3","R3"),
                 _fmt("LD","R4","@w2"), _fmt("LD","R5","@b2"),
                 _fmt("MMUL","R6","R3","R4"), _fmt("MADD","R6","R6","R5"),
                 _fmt(act,"R6","R6"),
                 _fmt("LD","R7","@w3"), _fmt("LD","R8","@b3"),
                 _fmt("MMUL","R9","R6","R7"), _fmt("MADD","R9","R9","R8"),
                 _fmt("SIGM","R9","R9"),
                 _fmt("ST","R9",f"@{o}"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    def _softmax_classifier():
        syntax = pick_syntax(); i = _inp(); o = _out()
        q = f"Write NML for a softmax classifier{syntax_tag(syntax)}"
        lines = [_fmt("LD","R0",f"@{i}"), _fmt("LD","R1","@weights"),
                 _fmt("LD","R2","@bias"), _fmt("MMUL","R3","R0","R1"),
                 _fmt("MADD","R3","R3","R2"), _fmt("SOFT","R3","R3"),
                 _fmt("ST","R3",f"@{o}"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    def _autoencoder():
        act = random.choice(acts); syntax = pick_syntax()
        i = _inp(); o = _out()
        q = f"Write NML for an autoencoder with {act_names[act]}{syntax_tag(syntax)}"
        lines = [_fmt("LD","R0",f"@{i}"),
                 _fmt("LD","R1","@enc_w"), _fmt("LD","R2","@enc_b"),
                 _fmt("MMUL","R3","R0","R1"), _fmt("MADD","R3","R3","R2"),
                 _fmt(act,"R3","R3"),
                 _fmt("LD","R4","@dec_w"), _fmt("LD","R5","@dec_b"),
                 _fmt("MMUL","R6","R3","R4"), _fmt("MADD","R6","R6","R5"),
                 _fmt("SIGM","R6","R6"),
                 _fmt("ST","R6",f"@{o}"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    def _with_norm():
        act = random.choice(acts); syntax = pick_syntax()
        i = _inp(); o = _out()
        q = f"Write NML for a dense layer with layer normalization and {act_names[act]}{syntax_tag(syntax)}"
        lines = [_fmt("LD","R0",f"@{i}"), _fmt("LD","R1","@weights"),
                 _fmt("LD","R2","@bias"), _fmt("MMUL","R3","R0","R1"),
                 _fmt("MADD","R3","R3","R2"), _fmt("NORM","R3","R3"),
                 _fmt(act,"R3","R3"), _fmt("ST","R3",f"@{o}"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    def _linear():
        syntax = pick_syntax(); i = _inp(); o = _out()
        q = f"Write NML for linear regression: y = Wx + b{syntax_tag(syntax)}"
        lines = [_fmt("LD","R0",f"@{i}"), _fmt("LD","R1","@weights"),
                 _fmt("LD","R2","@bias"), _fmt("MMUL","R3","R0","R1"),
                 _fmt("MADD","R3","R3","R2"),
                 _fmt("ST","R3",f"@{o}"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    generators = [
        (0.18, _one_layer), (0.18, _two_layer), (0.14, _three_layer),
        (0.10, _softmax_classifier), (0.12, _autoencoder),
        (0.14, _with_norm), (0.14, _linear),
    ]
    weights = [g[0] for g in generators]; fns = [g[1] for g in generators]
    while len(pairs) < count:
        pairs.append(random.choices(fns, weights=weights, k=1)[0]())
    return pairs[:count]


# ═══════════════════════════════════════════════════════════════════════════════
# Category 7: Data Science / Linear Algebra (target: 8,000)
# ═══════════════════════════════════════════════════════════════════════════════

def gen_data_science(count=8000):
    pairs = []

    templates = [
        ("Dot product of {a} and {b}",
         lambda a,b,o: [_fmt("LD","R0",f"@{a}"), _fmt("LD","R1",f"@{b}"),
                        _fmt("SDOT","R2","R0","R1"), _fmt("ST","R2",f"@{o}"), "HALT"]),
        ("Matrix multiply {a} by {b}",
         lambda a,b,o: [_fmt("LD","R0",f"@{a}"), _fmt("LD","R1",f"@{b}"),
                        _fmt("MMUL","R2","R0","R1"), _fmt("ST","R2",f"@{o}"), "HALT"]),
        ("Cosine distance between {a} and {b}",
         lambda a,b,o: [_fmt("LD","R0",f"@{a}"), _fmt("LD","R1",f"@{b}"),
                        _fmt("DIST","R2","R0","R1","#0"), _fmt("ST","R2",f"@{o}"), "HALT"]),
        ("Euclidean distance from {a} to {b}",
         lambda a,b,o: [_fmt("LD","R0",f"@{a}"), _fmt("LD","R1",f"@{b}"),
                        _fmt("DIST","R2","R0","R1","#1"), _fmt("ST","R2",f"@{o}"), "HALT"]),
        ("Transpose {a}",
         lambda a,b,o: [_fmt("LD","R0",f"@{a}"), _fmt("TRNS","R1","R0"),
                        _fmt("ST","R1",f"@{o}"), "HALT"]),
        ("Normalize {a}",
         lambda a,b,o: [_fmt("LD","R0",f"@{a}"), _fmt("NORM","R1","R0"),
                        _fmt("ST","R1",f"@{o}"), "HALT"]),
        ("Project {a} through {b}",
         lambda a,b,o: [_fmt("LD","R0",f"@{a}"), _fmt("LD","R1",f"@{b}"),
                        _fmt("PROJ","R2","R0","R1"), _fmt("ST","R2",f"@{o}"), "HALT"]),
        ("Weighted sum of {a} with weights {b}",
         lambda a,b,o: [_fmt("LD","R0",f"@{a}"), _fmt("LD","R1",f"@{b}"),
                        _fmt("EMUL","R2","R0","R1"), _fmt("RDUC","R3","R2","#0"),
                        _fmt("ST","R3",f"@{o}"), "HALT"]),
        ("Sum and mean of {a}",
         lambda a,b,o: [_fmt("LD","R0",f"@{a}"), _fmt("RDUC","R1","R0","#0"),
                        _fmt("RDUC","R2","R0","#1"),
                        _fmt("ST","R1","@total"), _fmt("ST","R2","@mean_val"), "HALT"]),
        ("Min and max of {a}",
         lambda a,b,o: [_fmt("LD","R0",f"@{a}"), _fmt("RDUC","R1","R0","#2"),
                        _fmt("RDUC","R2","R0","#3"),
                        _fmt("ST","R1","@max_val"), _fmt("ST","R2","@min_val"), "HALT"]),
        ("Reshape {a} to dimension {dim}",
         lambda a,b,o,dim=None: [_fmt("LD","R0",f"@{a}"),
                                  _fmt("RSHP","R1","R0",f"#[{dim}]"),
                                  _fmt("ST","R1",f"@{o}"), "HALT"]),
    ]

    while len(pairs) < count:
        t_idx = random.randint(0, len(templates) - 1)
        prompt_tpl, fn = templates[t_idx]
        a, b = _pick(INPUT_NAMES, 2); o = _out(); syntax = pick_syntax()
        dim = random.choice([4,8,16,32,64])
        prompt = prompt_tpl.format(a=a, b=b, dim=dim)
        try:
            lines = fn(a, b, o, dim=dim)
        except TypeError:
            lines = fn(a, b, o)
        q = f"Write NML to: {prompt}{syntax_tag(syntax)}"
        pairs.append(_pair(q, apply_syntax(lines, syntax)))

    return pairs[:count]


# ═══════════════════════════════════════════════════════════════════════════════
# Category 8: Vision Pipelines NML-V (target: 5,000)
# ═══════════════════════════════════════════════════════════════════════════════

def gen_vision(count=5000):
    pairs = []

    def _conv_relu():
        i = _inp(); o = _out(); syntax = pick_syntax()
        q = f"Write NML for a convolutional layer with ReLU on {i}{syntax_tag(syntax)}"
        lines = [_fmt("LD","R0",f"@{i}"), _fmt("LD","R1","@kernel"),
                 _fmt("LD","R2","@bias"), _fmt("CONV","R3","R0","R1"),
                 _fmt("MADD","R3","R3","R2"), _fmt("RELU","R3","R3"),
                 _fmt("ST","R3",f"@{o}"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    def _conv_pool():
        i = _inp(); o = _out(); syntax = pick_syntax()
        q = f"Write NML for convolution followed by max pooling{syntax_tag(syntax)}"
        lines = [_fmt("LD","R0",f"@{i}"), _fmt("LD","R1","@kernel"),
                 _fmt("CONV","R2","R0","R1"), _fmt("RELU","R2","R2"),
                 _fmt("POOL","R3","R2"), _fmt("ST","R3",f"@{o}"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    def _two_layer_cnn():
        i = _inp(); o = _out(); syntax = pick_syntax()
        q = f"Write NML for a 2-layer CNN{syntax_tag(syntax)}"
        lines = [_fmt("LD","R0",f"@{i}"),
                 _fmt("LD","R1","@k1"), _fmt("LD","R2","@b1"),
                 _fmt("CONV","R3","R0","R1"), _fmt("MADD","R3","R3","R2"),
                 _fmt("RELU","R3","R3"), _fmt("POOL","R3","R3"),
                 _fmt("LD","R4","@k2"), _fmt("LD","R5","@b2"),
                 _fmt("CONV","R6","R3","R4"), _fmt("MADD","R6","R6","R5"),
                 _fmt("RELU","R6","R6"), _fmt("POOL","R6","R6"),
                 _fmt("ST","R6",f"@{o}"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    def _upscale():
        i = _inp(); o = _out(); syntax = pick_syntax()
        q = f"Write NML to upscale {i} by 2x{syntax_tag(syntax)}"
        lines = [_fmt("LD","R0",f"@{i}"), _fmt("UPSC","R1","R0"),
                 _fmt("ST","R1",f"@{o}"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    def _pad_conv():
        i = _inp(); o = _out(); syntax = pick_syntax()
        q = f"Write NML to zero-pad {i} then convolve{syntax_tag(syntax)}"
        lines = [_fmt("LD","R0",f"@{i}"), _fmt("PADZ","R1","R0"),
                 _fmt("LD","R2","@kernel"), _fmt("CONV","R3","R1","R2"),
                 _fmt("ST","R3",f"@{o}"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    generators = [
        (0.25, _conv_relu), (0.25, _conv_pool), (0.20, _two_layer_cnn),
        (0.15, _upscale), (0.15, _pad_conv),
    ]
    weights = [g[0] for g in generators]; fns = [g[1] for g in generators]
    while len(pairs) < count:
        pairs.append(random.choices(fns, weights=weights, k=1)[0]())
    return pairs[:count]


# ═══════════════════════════════════════════════════════════════════════════════
# Category 9: Transformer Blocks NML-T (target: 5,000)
# ═══════════════════════════════════════════════════════════════════════════════

def gen_transformer(count=5000):
    pairs = []

    def _self_attention():
        i = _inp(); o = _out(); syntax = pick_syntax()
        q = f"Write NML for self-attention on {i}{syntax_tag(syntax)}"
        lines = [_fmt("LD","R0",f"@{i}"),
                 _fmt("LD","R1","@wq"), _fmt("LD","R2","@wk"), _fmt("LD","R3","@wv"),
                 _fmt("MMUL","R4","R0","R1"), _fmt("MMUL","R5","R0","R2"),
                 _fmt("MMUL","R6","R0","R3"),
                 _fmt("ATTN","R7","R4","R5","R6"),
                 _fmt("ST","R7",f"@{o}"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    def _transformer_block():
        i = _inp(); o = _out(); syntax = pick_syntax()
        q = f"Write NML for a full transformer block (attention + FFN + residual + norm){syntax_tag(syntax)}"
        lines = [_fmt("LD","R0",f"@{i}"),
                 _fmt("LD","R1","@wq"), _fmt("LD","R2","@wk"), _fmt("LD","R3","@wv"),
                 _fmt("MMUL","R4","R0","R1"), _fmt("MMUL","R5","R0","R2"),
                 _fmt("MMUL","R6","R0","R3"),
                 _fmt("ATTN","R7","R4","R5","R6"),
                 _fmt("MADD","R7","R7","R0"), _fmt("NORM","R7","R7"),
                 _fmt("LD","R8","@ff_w1"), _fmt("LD","R9","@ff_w2"),
                 _fmt("MMUL","RA","R7","R8"), _fmt("GELU","RA","RA"),
                 _fmt("MMUL","RA","RA","R9"),
                 _fmt("MADD","RA","RA","R7"), _fmt("NORM","RA","RA"),
                 _fmt("ST","RA",f"@{o}"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    def _embedding():
        o = _out(); syntax = pick_syntax()
        q = f"Write NML to embed token IDs into dense vectors{syntax_tag(syntax)}"
        lines = [_fmt("LD","R0","@token_ids"), _fmt("LD","R1","@embed_table"),
                 _fmt("EMBD","R2","R1","R0"), _fmt("ST","R2",f"@{o}"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    def _ffn_gelu():
        i = _inp(); o = _out(); syntax = pick_syntax()
        q = f"Write NML for a feedforward layer with GELU activation{syntax_tag(syntax)}"
        lines = [_fmt("LD","R0",f"@{i}"), _fmt("LD","R1","@ff_weights"),
                 _fmt("LD","R2","@ff_bias"), _fmt("MMUL","R3","R0","R1"),
                 _fmt("MADD","R3","R3","R2"), _fmt("GELU","R3","R3"),
                 _fmt("ST","R3",f"@{o}"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    generators = [
        (0.30, _self_attention), (0.30, _transformer_block),
        (0.20, _embedding), (0.20, _ffn_gelu),
    ]
    weights = [g[0] for g in generators]; fns = [g[1] for g in generators]
    while len(pairs) < count:
        pairs.append(random.choices(fns, weights=weights, k=1)[0]())
    return pairs[:count]


# ═══════════════════════════════════════════════════════════════════════════════
# Category 10: Signal Processing NML-S (target: 3,000)
# ═══════════════════════════════════════════════════════════════════════════════

def gen_signal(count=3000):
    pairs = []

    def _fft():
        i = _inp(); syntax = pick_syntax()
        q = f"Write NML to compute the frequency spectrum of {i}{syntax_tag(syntax)}"
        lines = [_fmt("LD","R0",f"@{i}"), _fmt("FFT","R1","R2","R0"),
                 _fmt("ST","R1","@real_part"), _fmt("ST","R2","@imag_part"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    def _filter():
        i = _inp(); o = _out(); syntax = pick_syntax()
        q = f"Write NML to apply a FIR filter to {i}{syntax_tag(syntax)}"
        lines = [_fmt("LD","R0",f"@{i}"), _fmt("LD","R1","@coefficients"),
                 _fmt("FILT","R2","R0","R1"), _fmt("ST","R2",f"@{o}"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    def _fft_filter_chain():
        i = _inp(); syntax = pick_syntax()
        q = f"Write NML for signal processing: FFT, filter, and find peak{syntax_tag(syntax)}"
        lines = [_fmt("LD","R0",f"@{i}"), _fmt("FFT","R1","R2","R0"),
                 _fmt("LD","R3","@filter_coeffs"), _fmt("FILT","R4","R0","R3"),
                 _fmt("RDUC","R5","R4","#2"),
                 _fmt("ST","R1","@spectrum"), _fmt("ST","R4","@filtered"),
                 _fmt("ST","R5","@peak"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    generators = [(0.35, _fft), (0.35, _filter), (0.30, _fft_filter_chain)]
    weights = [g[0] for g in generators]; fns = [g[1] for g in generators]
    while len(pairs) < count:
        pairs.append(random.choices(fns, weights=weights, k=1)[0]())
    return pairs[:count]


# ═══════════════════════════════════════════════════════════════════════════════
# Category 11: M2M and Agent Patterns (target: 5,000)
# ═══════════════════════════════════════════════════════════════════════════════

def gen_m2m(count=5000):
    pairs = []

    def _meta_program():
        i = _inp(); o = _out(); syntax = pick_syntax()
        desc = random.choice(["data transform","scaling","projection","normalization"])
        q = f"Write a self-describing NML program with META headers for {desc}{syntax_tag(syntax)}"
        lines = [f'META  @name "{desc.replace(" ","_")}"',
                 f'META  @version "1.0"', f'META  @author "agent"',
                 _fmt("LD","R0",f"@{i}"), _fmt("SCLR","R1","R0","#2.0"),
                 _fmt("ST","R1",f"@{o}"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    def _fragment():
        i = _inp(); o = _out(); syntax = pick_syntax()
        q = f"Write NML with two composable fragments{syntax_tag(syntax)}"
        lines = [_fmt("FRAG","step_one"),
                 _fmt("LD","R0",f"@{i}"), _fmt("SCLR","R1","R0","#2.0"),
                 _fmt("ST","R1","@intermediate"), "ENDF",
                 _fmt("FRAG","step_two"),
                 _fmt("LINK","@step_one"),
                 _fmt("LD","R2","@intermediate"),
                 _fmt("MADD","R3","R2","#10.0"),
                 _fmt("ST","R3",f"@{o}"), "ENDF",
                 _fmt("FRAG","main"),
                 _fmt("LINK","@step_two"), "HALT", "ENDF"]
        return _pair(q, apply_syntax(lines, syntax))

    def _vote():
        i = _inp(); o = _out(); syntax = pick_syntax()
        strat = random.choice([("#0","median"),("#1","mean"),("#2","quorum")])
        q = f"Write NML to find {strat[1]} consensus of {i}{syntax_tag(syntax)}"
        lines = [_fmt("LD","R0",f"@{i}"), _fmt("VOTE","R1","R0",strat[0]),
                 _fmt("ST","R1",f"@{o}"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    def _signed():
        i = _inp(); o = _out(); syntax = pick_syntax()
        q = f"Write a signed NML program{syntax_tag(syntax)}"
        lines = ["SIGN  agent=core_v1  key=ed25519:abc123  sig=def456",
                 f'META  @name "signed_calc"', f'META  @version "1.0"',
                 "VRFY  @self  @core_v1",
                 _fmt("LD","R0",f"@{i}"), _fmt("SCLR","RA","R0","#2.0"),
                 _fmt("ST","RA",f"@{o}"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    def _gather_scatter():
        i = _inp(); o = _out(); syntax = pick_syntax()
        op = random.choice([("GATH","gather","look up"),("SCAT","scatter","write")])
        q = f"Write NML to {op[2]} elements by index from {i}{syntax_tag(syntax)}"
        lines = [_fmt("LD","R0",f"@{i}"), _fmt("LD","R1","@indices"),
                 _fmt(op[0],"R2","R0","R1"), _fmt("ST","R2",f"@{o}"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    generators = [
        (0.20, _meta_program), (0.25, _fragment), (0.20, _vote),
        (0.20, _signed), (0.15, _gather_scatter),
    ]
    weights = [g[0] for g in generators]; fns = [g[1] for g in generators]
    while len(pairs) < count:
        pairs.append(random.choices(fns, weights=weights, k=1)[0]())
    return pairs[:count]


# ═══════════════════════════════════════════════════════════════════════════════
# Category 12: General Purpose NML-G (target: 5,000)
# ═══════════════════════════════════════════════════════════════════════════════

def gen_general_purpose(count=5000):
    pairs = []

    def _print_number():
        v = _fval(1,1000); syntax = pick_syntax()
        q = f"Write NML to print the number {v}{syntax_tag(syntax)}"
        lines = [_fmt("LEAF","R0",f"#{v}"), _fmt("SYS","R0","#0"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    def _fibonacci():
        n = random.choice([10,15,20,30]); syntax = pick_syntax()
        q = f"Write NML for Fibonacci: print the first {n} numbers{syntax_tag(syntax)}"
        lines = [_fmt("LEAF","R0","#0.0"), _fmt("LEAF","R1","#1.0"),
                 _fmt("LEAF","RD","#0.0"), _fmt("LEAF","R5",f"#{n}.0"),
                 _fmt("SYS","R0","#0"),
                 _fmt("TACC","R2","R0","R1"), _fmt("MOV","R0","R1"),
                 _fmt("MOV","R1","R2"), _fmt("LEAF","RC","#1.0"),
                 _fmt("TACC","RD","RD","RC"),
                 _fmt("CMP","RD","R5"), _fmt("JMPT","#-8"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    def _fizzbuzz():
        syntax = pick_syntax()
        q = f"Write NML for FizzBuzz 1 to 30 using MOD{syntax_tag(syntax)}"
        lines = [_fmt("LEAF","RD","#1.0"), _fmt("LEAF","R5","#31.0"),
                 _fmt("LEAF","R3","#15.0"), _fmt("LEAF","R4","#3.0"),
                 _fmt("LEAF","R6","#5.0"),
                 _fmt("MOD","R0","RD","R3"), _fmt("CMPI","RE","R0","#0.5"),
                 _fmt("JMPF","#3"),
                 _fmt("SYS","RD","#0"), _fmt("JUMP","#8"),
                 _fmt("MOD","R0","RD","R4"), _fmt("CMPI","RE","R0","#0.5"),
                 _fmt("JMPF","#3"),
                 _fmt("SYS","RD","#0"), _fmt("JUMP","#4"),
                 _fmt("MOD","R0","RD","R6"), _fmt("CMPI","RE","R0","#0.5"),
                 _fmt("JMPF","#2"),
                 _fmt("SYS","RD","#0"),
                 _fmt("LEAF","RC","#1.0"), _fmt("TACC","RD","RD","RC"),
                 _fmt("CMP","RD","R5"), _fmt("JMPT","#-18"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    def _modulo():
        a, b = _pick(INPUT_NAMES, 2); o = _out(); syntax = pick_syntax()
        q = f"Write NML to compute {a} modulo {b}{syntax_tag(syntax)}"
        lines = [_fmt("LD","R0",f"@{a}"), _fmt("LD","R1",f"@{b}"),
                 _fmt("MOD","R2","R0","R1"), _fmt("ST","R2",f"@{o}"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    def _type_convert():
        i = _inp(); o = _out(); syntax = pick_syntax()
        op = random.choice([("ITOF","convert integer to float"),
                            ("FTOI","convert float to integer")])
        q = f"Write NML to {op[1]} for {i}{syntax_tag(syntax)}"
        lines = [_fmt("LD","R0",f"@{i}"), _fmt(op[0],"R1","R0"),
                 _fmt("ST","R1",f"@{o}"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    def _random_val():
        o = _out(); syntax = pick_syntax()
        q = f"Write NML to generate a random value{syntax_tag(syntax)}"
        lines = [_fmt("ALLC","R0","#[1]"), _fmt("SYS","R0","#5"),
                 _fmt("ST","R0",f"@{o}"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    generators = [
        (0.15, _print_number), (0.20, _fibonacci), (0.15, _fizzbuzz),
        (0.20, _modulo), (0.15, _type_convert), (0.15, _random_val),
    ]
    weights = [g[0] for g in generators]; fns = [g[1] for g in generators]
    while len(pairs) < count:
        pairs.append(random.choices(fns, weights=weights, k=1)[0]())
    return pairs[:count]


# ═══════════════════════════════════════════════════════════════════════════════
# Category 13: Debugging and Repair (target: 8,000)
# ═══════════════════════════════════════════════════════════════════════════════

def gen_debugging(count=8000):
    pairs = []

    bug_templates = [
        {
            "desc": "Missing HALT",
            "broken": [_fmt("LD","R0","@input"), _fmt("SCLR","R1","R0","#2.0"),
                       _fmt("ST","R1","@result")],
            "fixed":  [_fmt("LD","R0","@input"), _fmt("SCLR","R1","R0","#2.0"),
                       _fmt("ST","R1","@result"), "HALT"],
            "explain": "The program is missing a HALT instruction. Without HALT, execution continues past the last instruction, causing undefined behavior.",
        },
        {
            "desc": "Unbalanced LOOP/ENDP",
            "broken": [_fmt("LEAF","R0","#5"), _fmt("LOOP","R0"),
                       _fmt("LEAF","R1","#1.0"), _fmt("ST","R1","@result"), "HALT"],
            "fixed":  [_fmt("LEAF","R0","#5"), _fmt("LOOP","R0"),
                       _fmt("LEAF","R1","#1.0"), "ENDP",
                       _fmt("ST","R1","@result"), "HALT"],
            "explain": "LOOP without matching ENDP. Every LOOP must have a corresponding ENDP to mark where the loop body ends.",
        },
        {
            "desc": "Register aliasing in MSUB",
            "broken": [_fmt("LD","R0","@a"), _fmt("LD","R1","@b"),
                       _fmt("MSUB","R0","R0","R1"), _fmt("ST","R0","@result"), "HALT"],
            "fixed":  [_fmt("LD","R0","@a"), _fmt("LD","R1","@b"),
                       _fmt("MSUB","R2","R0","R1"), _fmt("ST","R2","@result"), "HALT"],
            "explain": "MSUB R0 R0 R1 has register aliasing: Rd==Rs1. The runtime copies Rs1 into Rd before reading Rs2, which can cause incorrect results. Use a separate destination register.",
        },
        {
            "desc": "Wrong jump offset",
            "broken": [_fmt("LD","R0","@input"), _fmt("CMPI","RE","R0","#50.0"),
                       _fmt("JMPT","#3"), _fmt("SCLR","R1","R0","#2.0"),
                       _fmt("JUMP","#1"), _fmt("SCLR","R1","R0","#0.5"),
                       _fmt("ST","R1","@result"), "HALT"],
            "fixed":  [_fmt("LD","R0","@input"), _fmt("CMPI","RE","R0","#50.0"),
                       _fmt("JMPT","#3"), _fmt("SCLR","R1","R0","#2.0"),
                       _fmt("JUMP","#2"), _fmt("SCLR","R1","R0","#0.5"),
                       _fmt("ST","R1","@result"), "HALT"],
            "explain": "JUMP #1 skips only 1 instruction, but needs to skip the else-branch SCLR. After JUMP, PC advances by offset+1, so JUMP #2 is needed to land on ST.",
        },
        {
            "desc": "Using uninitialized register",
            "broken": [_fmt("MADD","R2","R0","R1"), _fmt("ST","R2","@result"), "HALT"],
            "fixed":  [_fmt("LD","R0","@a"), _fmt("LD","R1","@b"),
                       _fmt("MADD","R2","R0","R1"), _fmt("ST","R2","@result"), "HALT"],
            "explain": "R0 and R1 are used without being loaded first. All registers must be initialized (via LD, LEAF, ALLC, or MOV) before use.",
        },
        {
            "desc": "CALL with no RET",
            "broken": [_fmt("LD","R0","@input"), _fmt("CALL","#2"),
                       _fmt("ST","R0","@result"), "HALT",
                       _fmt("SCLR","R0","R0","#2.0")],
            "fixed":  [_fmt("LD","R0","@input"), _fmt("CALL","#2"),
                       _fmt("ST","R0","@result"), "HALT",
                       _fmt("SCLR","R0","R0","#2.0"), "RET"],
            "explain": "The subroutine is missing RET. Without RET, execution falls through past the subroutine, which is undefined. Every CALL target must end with RET.",
        },
    ]

    prompt_templates = [
        "Fix this NML program:\n{code}",
        "This NML has a bug. Find and fix it:\n{code}",
        "Debug this NML:\n{code}",
        "What is wrong with this NML program and how do you fix it?\n{code}",
    ]

    while len(pairs) < count:
        bug = random.choice(bug_templates)
        syntax = pick_syntax()
        broken_lines = apply_syntax(bug["broken"], syntax)
        fixed_lines = apply_syntax(bug["fixed"], syntax)
        q = random.choice(prompt_templates).format(code="\n".join(broken_lines))
        a = f"{bug['explain']}\n\nFixed program:\n" + "\n".join(fixed_lines)
        pairs.append(_pair(q, a))

    return pairs[:count]


# ═══════════════════════════════════════════════════════════════════════════════
# Category 14: Explanation (target: 8,000)
# ═══════════════════════════════════════════════════════════════════════════════

def gen_explanation(count=8000):
    pairs = []

    program_templates = [
        {
            "lines": lambda i,o: [_fmt("LD","R0",f"@{i}"), _fmt("SCLR","R1","R0","#2.0"),
                                   _fmt("ST","R1",f"@{o}"), "HALT"],
            "explain": lambda i,o: f"This program loads {i}, doubles it (scales by 2.0), stores the result in {o}, and halts.",
        },
        {
            "lines": lambda i,o: [_fmt("LD","R0",f"@{i}"), _fmt("LD","R1","@weights"),
                                   _fmt("LD","R2","@bias"), _fmt("MMUL","R3","R0","R1"),
                                   _fmt("MADD","R3","R3","R2"), _fmt("RELU","R3","R3"),
                                   _fmt("ST","R3",f"@{o}"), "HALT"],
            "explain": lambda i,o: f"This is a single dense neural network layer. It loads {i}, multiplies by a weight matrix, adds bias, applies ReLU activation, and stores the output in {o}.",
        },
        {
            "lines": lambda i,o: [_fmt("LD","R0",f"@{i}"), _fmt("CMPI","RE","R0","#100.0"),
                                   _fmt("JMPF","#3"), _fmt("SCLR","R1","R0","#0.5"),
                                   _fmt("JUMP","#2"), _fmt("MOV","R1","R0"),
                                   _fmt("ST","R1",f"@{o}"), "HALT"],
            "explain": lambda i,o: f"This program loads {i} and checks if it's less than 100. If yes (flag true), it halves the value. Otherwise, it keeps the original value. The result is stored in {o}.",
        },
        {
            "lines": lambda i,o: [_fmt("LEAF","RA","#0.0"), _fmt("LEAF","RD","#1.0"),
                                   _fmt("TACC","RA","RA","RD"), _fmt("LEAF","RC","#1.0"),
                                   _fmt("TACC","RD","RD","RC"), _fmt("CMPI","RE","RD","#11.0"),
                                   _fmt("JMPT","#-5"), _fmt("ST","RA",f"@{o}"), "HALT"],
            "explain": lambda i,o: f"This program sums the integers 1 through 10 using a backward-jump while loop. RA accumulates the sum, RD is the counter. It increments RD each iteration and loops back while RD < 11. The final sum (55) is stored in {o}.",
        },
        {
            "lines": lambda i,o: [_fmt("LD","R0",f"@{i}"), _fmt("RDUC","R1","R0","#1"),
                                   _fmt("RDUC","R2","R0","#2"), _fmt("RDUC","R3","R0","#3"),
                                   _fmt("ST","R1","@mean_val"), _fmt("ST","R2","@max_val"),
                                   _fmt("ST","R3","@min_val"), "HALT"],
            "explain": lambda i,o: f"This program computes three statistics of {i}: the mean (RDUC #1), maximum (RDUC #2), and minimum (RDUC #3), storing each to a separate memory address.",
        },
        {
            "lines": lambda i,o: [_fmt("LD","R0",f"@{i}"), _fmt("CALL","#2"),
                                   _fmt("ST","R0",f"@{o}"), "HALT",
                                   _fmt("SCLR","R0","R0","#3.0"), "RET"],
            "explain": lambda i,o: f"This program uses a subroutine to triple {i}. It loads {i}, calls a subroutine (offset +2) that scales R0 by 3.0 and returns, then stores the result in {o}.",
        },
    ]

    q_templates = [
        "Explain what this NML program does:\n{code}",
        "What does this NML program do?\n{code}",
        "Describe the behavior of this NML:\n{code}",
        "Walk through this NML program step by step:\n{code}",
    ]

    while len(pairs) < count:
        tmpl = random.choice(program_templates)
        i = _inp(); o = _out(); syntax = pick_syntax()
        lines = apply_syntax(tmpl["lines"](i, o), syntax)
        code = "\n".join(lines)
        q = random.choice(q_templates).format(code=code)
        a = tmpl["explain"](i, o)
        pairs.append(_pair(q, a))

    return pairs[:count]


# ═══════════════════════════════════════════════════════════════════════════════
# Category 15: Symbolic + Verbose Variants (target: 12,000)
# Generated by converting a random selection from other categories.
# ═══════════════════════════════════════════════════════════════════════════════

def gen_syntax_variants(source_pairs, count=12000):
    pairs = []
    sampled = random.sample(source_pairs, min(count * 3, len(source_pairs)))

    sym_prefixes = [
        "Write in symbolic NML: ", "Symbolic NML for: ",
        "Using symbolic syntax, write NML to: ", "In symbolic NML: ",
    ]
    verbose_prefixes = [
        "Write in verbose NML: ", "Verbose NML for: ",
        "Using verbose/human-readable syntax: ", "In verbose NML: ",
    ]

    for p in sampled:
        if len(pairs) >= count:
            break

        original_prompt = p["messages"][0]["content"]
        classic_code = p["messages"][1]["content"]
        classic_lines = classic_code.split("\n")

        is_symbolic = random.random() < 0.65
        if is_symbolic:
            new_lines = to_symbolic(classic_lines)
            prefix = random.choice(sym_prefixes)
        else:
            new_lines = to_verbose(classic_lines)
            prefix = random.choice(verbose_prefixes)

        new_code = "\n".join(new_lines)
        if new_code == classic_code:
            continue

        core = original_prompt
        for strip in ["Write NML to: ","Write NML to ","Write NML for ","Write NML for a "]:
            if core.startswith(strip):
                core = core[len(strip):]
                break

        pairs.append(_pair(prefix + core, new_code))

    return pairs[:count]


# ═══════════════════════════════════════════════════════════════════════════════
# Category 16: Training Programs NML-TR (target: 5,000)
# ═══════════════════════════════════════════════════════════════════════════════

def gen_training(count=5000):
    pairs = []
    topologies = [
        ("2-4-1", [2,4], [4,1]),
        ("2-8-1", [2,8], [8,1]),
        ("1-16-1", [1,16], [16,1]),
        ("1-64-1", [1,64], [64,1]),
        ("4-16-8-1", [4,16], [16,1]),
        ("3-8-1", [3,8], [8,1]),
    ]

    def _tnet_basic():
        syntax = pick_syntax(); o = _out()
        epochs = random.choice([100,200,500,1000,2000,5000])
        lr = random.choice([0.001, 0.005, 0.01, 0.05, 0.1, 0.5])
        topo_name, _, _ = random.choice(topologies)
        q = f"Write NML to train a {topo_name} network using TNET for {epochs} epochs at lr {lr}{syntax_tag(syntax)}"
        lines = [_fmt("LD","R0","@training_inputs"), _fmt("LD","R9","@training_targets"),
                 _fmt("LD","R1","@w1"), _fmt("LD","R2","@b1"),
                 _fmt("LD","R3","@w2"), _fmt("LD","R4","@b2"),
                 _fmt("TNET",f"#{epochs}",f"#{lr}"),
                 _fmt("ST","RA",f"@{o}"),
                 _fmt("ST","R1","@trained_w1"), _fmt("ST","R2","@trained_b1"),
                 _fmt("ST","R3","@trained_w2"), _fmt("ST","R4","@trained_b2"),
                 "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    def _manual_training_loop():
        syntax = pick_syntax(); o = _out()
        epochs = random.choice([10,50,100,500])
        lr = random.choice([0.001, 0.01, 0.1])
        q = f"Write NML to manually train a network for {epochs} epochs using BKWD and WUPD{syntax_tag(syntax)}"
        lines = [_fmt("LD","R0","@training_inputs"), _fmt("LD","R9","@training_targets"),
                 _fmt("LD","R1","@w1"), _fmt("LD","R2","@b1"),
                 _fmt("LEAF","RJ",f"#{lr}"),
                 _fmt("LEAF","RD",f"#{epochs}"), _fmt("LOOP","RD"),
                 _fmt("MMUL","R5","R0","R1"), _fmt("MADD","R5","R5","R2"),
                 _fmt("RELU","R5","R5"),
                 _fmt("LOSS","RG","R5","R9","#0"),
                 _fmt("BKWD","RH","R5","R9"),
                 _fmt("WUPD","R1","R1","RH"),
                 "ENDP",
                 _fmt("ST","R1","@trained_w1"), _fmt("ST","RG",f"@{o}"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    def _loss_comparison():
        syntax = pick_syntax(); o = _out()
        q = f"Write NML to compute MSE, cross-entropy, and MAE loss{syntax_tag(syntax)}"
        lines = [_fmt("LD","R0","@predictions"), _fmt("LD","R9","@targets"),
                 _fmt("LOSS","RG","R0","R9","#0"),
                 _fmt("LOSS","RH","R0","R9","#1"),
                 _fmt("LOSS","RI","R0","R9","#2"),
                 _fmt("ST","RG","@mse_loss"), _fmt("ST","RH","@ce_loss"),
                 _fmt("ST","RI","@mae_loss"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    def _train_and_infer():
        syntax = pick_syntax(); o = _out()
        epochs = random.choice([500,1000,2000])
        lr = random.choice([0.01, 0.05, 0.1])
        q = f"Write NML to train a network with TNET then run inference on new data{syntax_tag(syntax)}"
        lines = [_fmt("LD","R0","@training_inputs"), _fmt("LD","R9","@training_targets"),
                 _fmt("LD","R1","@w1"), _fmt("LD","R2","@b1"),
                 _fmt("LD","R3","@w2"), _fmt("LD","R4","@b2"),
                 _fmt("TNET",f"#{epochs}",f"#{lr}"),
                 _fmt("LD","R0","@new_input"),
                 _fmt("MMUL","R5","R0","R1"), _fmt("MADD","R5","R5","R2"),
                 _fmt("RELU","R5","R5"),
                 _fmt("MMUL","R6","R5","R3"), _fmt("MADD","R6","R6","R4"),
                 _fmt("SIGM","R6","R6"),
                 _fmt("ST","R6",f"@{o}"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    def _gradient_registers():
        syntax = pick_syntax(); o = _out()
        q = f"Write NML that uses gradient registers RG-RI and learning rate RJ{syntax_tag(syntax)}"
        lines = [_fmt("LD","R0","@data"), _fmt("LD","R1","@weights"),
                 _fmt("LD","R9","@targets"),
                 _fmt("LEAF","RJ","#0.01"),
                 _fmt("MMUL","R5","R0","R1"),
                 _fmt("LOSS","RG","R5","R9","#0"),
                 _fmt("BKWD","RH","R5","R9"),
                 _fmt("WUPD","R1","R1","RH"),
                 _fmt("ST","R1","@updated_weights"),
                 _fmt("ST","RG",f"@{o}"), "HALT"]
        return _pair(q, apply_syntax(lines, syntax))

    generators = [
        (0.25, _tnet_basic), (0.25, _manual_training_loop),
        (0.15, _loss_comparison), (0.20, _train_and_infer),
        (0.15, _gradient_registers),
    ]
    weights = [g[0] for g in generators]; fns = [g[1] for g in generators]
    while len(pairs) < count:
        pairs.append(random.choices(fns, weights=weights, k=1)[0]())
    return pairs[:count]


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

TARGETS = {
    "opcode_reference":   5000,
    "basic_arithmetic":   5000,
    "loops":              8000,
    "conditionals":       8000,
    "subroutines":        5000,
    "neural_nets":       10000,
    "data_science":       8000,
    "vision":             5000,
    "transformer":        5000,
    "signal":             3000,
    "m2m":                5000,
    "general_purpose":    5000,
    "debugging":          8000,
    "explanation":        8000,
    "training":           5000,
    "syntax_variants":   12000,
}

def main():
    parser = argparse.ArgumentParser(description="Generate NML core training pairs")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit total pairs (scales categories proportionally)")
    args = parser.parse_args()

    targets = dict(TARGETS)
    if args.limit:
        total_default = sum(targets.values())
        scale = args.limit / total_default
        for k in targets:
            targets[k] = max(1, int(targets[k] * scale))

    print("Generating NML core training pairs...")
    print(f"{'─' * 60}")

    cat1  = gen_opcode_reference(targets["opcode_reference"])
    print(f"  Cat  1  Opcode Reference:          {len(cat1):>6}")

    cat2  = gen_basic_arithmetic(targets["basic_arithmetic"])
    print(f"  Cat  2  Basic Arithmetic:           {len(cat2):>6}")

    cat3  = gen_loops(targets["loops"])
    print(f"  Cat  3  Loops & Iteration:          {len(cat3):>6}")

    cat4  = gen_conditionals(targets["conditionals"])
    print(f"  Cat  4  Conditionals & Branching:   {len(cat4):>6}")

    cat5  = gen_subroutines(targets["subroutines"])
    print(f"  Cat  5  Subroutines:                {len(cat5):>6}")

    cat6  = gen_neural_nets(targets["neural_nets"])
    print(f"  Cat  6  Neural Networks:            {len(cat6):>6}")

    cat7  = gen_data_science(targets["data_science"])
    print(f"  Cat  7  Data Science:               {len(cat7):>6}")

    cat8  = gen_vision(targets["vision"])
    print(f"  Cat  8  Vision (NML-V):             {len(cat8):>6}")

    cat9  = gen_transformer(targets["transformer"])
    print(f"  Cat  9  Transformer (NML-T):        {len(cat9):>6}")

    cat10 = gen_signal(targets["signal"])
    print(f"  Cat 10  Signal (NML-S):             {len(cat10):>6}")

    cat11 = gen_m2m(targets["m2m"])
    print(f"  Cat 11  M2M & Agent:                {len(cat11):>6}")

    cat12 = gen_general_purpose(targets["general_purpose"])
    print(f"  Cat 12  General Purpose (NML-G):    {len(cat12):>6}")

    cat13 = gen_debugging(targets["debugging"])
    print(f"  Cat 13  Debugging & Repair:         {len(cat13):>6}")

    cat14 = gen_explanation(targets["explanation"])
    print(f"  Cat 14  Explanation:                {len(cat14):>6}")

    cat16 = gen_training(targets["training"])
    print(f"  Cat 16  Training (NML-TR):          {len(cat16):>6}")

    classic_pairs = cat1+cat2+cat3+cat4+cat5+cat6+cat7+cat8+cat9+cat10+cat11+cat12+cat13+cat14+cat16
    cat15 = gen_syntax_variants(classic_pairs, targets["syntax_variants"])
    print(f"  Cat 15  Syntax Variants:            {len(cat15):>6}")

    all_pairs = classic_pairs + cat15
    random.shuffle(all_pairs)

    print(f"{'─' * 60}")
    print(f"  TOTAL:                              {len(all_pairs):>6}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\nWritten to: {out_path}")


if __name__ == "__main__":
    main()
