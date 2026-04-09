#!/usr/bin/env python3
"""
NML Grammar Validator — formal grammar checker for NML programs.

Parses every instruction and validates structure without executing.
Supports all three NML syntax variants: symbolic, classic, and verbose.
"""

import re
import sys
import os
import json
from dataclasses import dataclass, field
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GrammarError:
    line: int
    error_type: str
    message: str
    source: str = ""       # the offending source line
    fix: str = ""          # actionable fix hint (schema, example, or suggestion)


@dataclass
class GrammarWarning:
    line: int
    warning_type: str
    message: str
    source: str = ""
    fix: str = ""


@dataclass
class GrammarReport:
    valid: bool
    errors: list[GrammarError] = field(default_factory=list)
    warnings: list[GrammarWarning] = field(default_factory=list)
    instruction_count: int = 0
    registers_used: set = field(default_factory=set)
    memory_inputs: list = field(default_factory=list)
    memory_outputs: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "errors": [
                {"line": e.line, "type": e.error_type, "message": e.message,
                 **({"source": e.source} if e.source else {}),
                 **({"fix": e.fix} if e.fix else {})}
                for e in self.errors
            ],
            "warnings": [
                {"line": w.line, "type": w.warning_type, "message": w.message,
                 **({"source": w.source} if w.source else {}),
                 **({"fix": w.fix} if w.fix else {})}
                for w in self.warnings
            ],
            "instruction_count": self.instruction_count,
            "registers_used": sorted(self.registers_used),
            "memory_inputs": self.memory_inputs,
            "memory_outputs": self.memory_outputs,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Register definitions
# ═══════════════════════════════════════════════════════════════════════════

_NUMBERED_LIST = [
    "R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9",
    "RA", "RB", "RC", "RD", "RE", "RF",
    "RG", "RH", "RI", "RJ", "RK", "RL", "RM", "RN",
    "RO", "RP", "RQ", "RR", "RS", "RT", "RU", "RV",
]
_GREEK_LIST = [
    "ι", "κ", "λ", "μ", "ν", "ξ", "ο", "π", "ρ", "ς",
    "α", "β", "γ", "δ", "φ", "ψ",
    "η", "θ", "ζ", "ω", "χ", "υ", "ε", "RN",
    "RO", "RP", "RQ", "RR", "RS", "RT", "RU", "RV",
]
_VERBOSE_REG_LIST = [
    "INPUT", "KERNEL", "LAYER", "MOMENTUM", "NORM", "XI",
    "OMICRON", "PI", "RHO", "SIGMA",
    "ACCUMULATOR", "GENERAL", "SCRATCH", "COUNTER", "FLAG", "STACK",
    "GRAD1", "GRAD2", "GRAD3", "LRATE", "RK", "RL", "RM", "RN",
    "RO", "RP", "RQ", "RR", "RS", "RT", "RU", "RV",
]

NUMBERED_REGISTERS = set(_NUMBERED_LIST)
GREEK_REGISTERS = set(_GREEK_LIST)
VERBOSE_REGISTERS = set(_VERBOSE_REG_LIST)
ALL_REGISTERS = NUMBERED_REGISTERS | GREEK_REGISTERS | VERBOSE_REGISTERS

REGISTER_CANONICAL: dict[str, str] = {}
for _i, _nr in enumerate(_NUMBERED_LIST):
    REGISTER_CANONICAL[_nr] = _nr
    REGISTER_CANONICAL[_GREEK_LIST[_i]] = _nr
    REGISTER_CANONICAL[_VERBOSE_REG_LIST[_i]] = _nr


# ═══════════════════════════════════════════════════════════════════════════
# Opcode mapping — every recognized opcode string → canonical classic opcode
# ═══════════════════════════════════════════════════════════════════════════

_OPCODE_TO_CANONICAL: dict[str, str] = {}

_CLASSIC_OPCODES = [
    "LD", "ST", "MOV", "ALLC", "MMUL", "MADD", "MSUB", "EMUL", "SDOT",
    "SCLR", "SDIV", "SADD", "SSUB", "EDIV", "RELU", "SIGM", "TANH", "SOFT", "RSHP",
    "TRNS", "SPLT", "MERG", "CMPF", "CMP", "CMPI", "JMPT", "JMPF",
    "JUMP", "LOOP", "ENDP", "LEAF", "TACC", "CALL", "RET", "SYNC",
    "HALT", "TRAP",
    "CONV", "POOL", "UPSC", "PADZ", "ATTN", "NORM", "EMBD", "GELU",
    "RDUC", "WHER", "CLMP", "CMPR", "FFT", "FILT",
    "META", "FRAG", "ENDF", "LINK", "PTCH", "SIGN", "VRFY", "VOTE",
    "PROJ", "DIST", "GATH", "SCAT",
    "SYS", "MOD", "ITOF", "FTOI", "BNOT",
    "BKWD", "WUPD", "LOSS", "TNET",
    "RELUBK", "SIGMBK", "TANHBK", "GELUBK", "SOFTBK",
    "MMULBK", "CONVBK", "POOLBK", "NORMBK", "ATTNBK", "TNDEEP",
    "TLOG", "TRAIN", "INFER", "WDECAY",
    "BN", "DROP",
]
for _op in _CLASSIC_OPCODES:
    _OPCODE_TO_CANONICAL[_op] = _op

_SYMBOLIC_TO_CANONICAL = {
    "↓": "LD", "↑": "ST", "←": "MOV", "□": "ALLC",
    "×": "MMUL",
    "⊕": "MADD",
    "⊖": "MSUB",
    "⊗": "EMUL",
    "⊘": "EDIV",
    "·": "SDOT",
    "∗": "SCLR",
    "÷": "SDIV",
    "∔": "SADD",
    "∸": "SSUB",
    "∎": "LEAF",
    "∑": "TACC",
    "⨁": "TACC",
    "⌐": "RELU", "σ": "SIGM", "τ": "TANH", "Σ": "SOFT",
    "⊟": "RSHP", "⊞": "BN", "⊤": "TRNS",
    "⊢": "SPLT", "⊣": "MERG",
    "⋈": "CMPF", "≶": "CMP", "≺": "CMPI",
    "↗": "JMPT", "↘": "JMPF", "→": "JUMP",
    "↻": "LOOP", "↺": "ENDP",
    "⏸": "SYNC",
    "◼": "HALT",
    "⇒": "CALL", "⇐": "RET",
    "◎": "CALL", "◉": "RET",
    "⚠": "TRAP",
    "⊛": "CONV", "⊓": "POOL", "⊔": "UPSC", "⊡": "PADZ",
    "⊙": "ATTN", "‖": "NORM", "⊏": "EMBD", "ℊ": "GELU",
    "⊥": "RDUC", "⊻": "WHER", "⊧": "CLMP", "⊜": "CMPR",
    "∿": "FFT", "⋐": "FILT",
    "§": "META", "◆": "FRAG", "◇": "ENDF",
    "⊿": "PTCH", "✦": "SIGN", "✓": "VRFY",
    "⚖": "VOTE", "⟐": "PROJ", "⟂": "DIST", "⊃": "GATH", "⊂": "SCAT",
    "⚙": "SYS", "%": "MOD", "⊶": "ITOF", "⊷": "FTOI", "¬": "BNOT",
    "∇": "BKWD", "⟳": "WUPD", "△": "LOSS", "⥁": "TNET",
    "⌐ˈ": "RELUBK", "σˈ": "SIGMBK", "τˈ": "TANHBK", "ℊˈ": "GELUBK", "Σˈ": "SOFTBK",
    "×ˈ": "MMULBK", "⊛ˈ": "CONVBK", "⊓ˈ": "POOLBK", "‖ˈ": "NORMBK", "⊙ˈ": "ATTNBK",
    "⥁ˈ": "TNDEEP",
    "⧖": "TLOG", "⟴": "TRAIN", "⟶": "INFER", "ω": "WDECAY",
    "≋": "DROP",
}
_OPCODE_TO_CANONICAL.update(_SYMBOLIC_TO_CANONICAL)

_VERBOSE_TO_CANONICAL = {
    "LOAD": "LD", "STORE": "ST", "COPY": "MOV", "ALLOCATE": "ALLC",
    "MATRIX_MULTIPLY": "MMUL",
    "ACCUMULATE": "MADD", "ADD": "MADD",
    "SUBTRACT": "MSUB",
    "SCALE": "EMUL", "ELEMENT_MULTIPLY": "EMUL",
    "DOT_PRODUCT": "SDOT",
    "SET": "SCLR", "SET_VALUE": "LEAF",
    "SCALAR_ADD": "SADD", "SCALAR_SUB": "SSUB",
    "SCALAR_DIVIDE": "SDIV", "DIVIDE": "SDIV",
    "ELEMENT_DIVIDE": "EDIV",
    "RECTIFY": "RELU",
    "SIGMOID": "SIGM",
    "HYPERBOLIC_TANGENT": "TANH",
    "SOFTMAX": "SOFT",
    "RESHAPE": "RSHP", "TRANSPOSE": "TRNS",
    "SPLIT": "SPLT", "MERGE": "MERG",
    "COMPARE_FEATURE": "CMPF",
    "COMPARE": "CMP",
    "COMPARE_IMMEDIATE": "CMPI", "COMPARE_VALUE": "CMPI",
    "BRANCH_TRUE": "JMPT", "BRANCH_FALSE": "JMPF", "BRANCH": "JUMP",
    "BEGIN_LOOP": "LOOP", "END_LOOP": "ENDP",
    "REPEAT": "LOOP", "END_REPEAT": "ENDP",
    "SET_LEAF": "LEAF",
    "TREE_ACCUMULATE": "TACC",
    "CALL_SUB": "CALL", "RETURN_SUB": "RET", "RETURN": "RET",
    "BARRIER": "SYNC", "STOP": "HALT", "FAULT": "TRAP",
    "CONVOLVE": "CONV", "MAX_POOL": "POOL", "UPSCALE": "UPSC", "ZERO_PAD": "PADZ",
    "ATTENTION": "ATTN", "LAYER_NORM": "NORM", "EMBED": "EMBD",
    "REDUCE": "RDUC", "WHERE": "WHER", "CLAMP": "CLMP", "MASK_COMPARE": "CMPR",
    "FOURIER": "FFT", "FILTER": "FILT",
    "METADATA": "META", "FRAGMENT": "FRAG", "END_FRAGMENT": "ENDF",
    "IMPORT": "LINK", "PATCH": "PTCH",
    "SIGN_PROGRAM": "SIGN", "VERIFY_SIGNATURE": "VRFY",
    "CONSENSUS": "VOTE", "PROJECT": "PROJ", "DISTANCE": "DIST", "GATHER": "GATH", "SCATTER": "SCAT",
    "SYSTEM": "SYS", "MODULO": "MOD", "INT_TO_FLOAT": "ITOF", "FLOAT_TO_INT": "FTOI", "BITWISE_NOT": "BNOT",
    "BACKWARD": "BKWD", "WEIGHT_UPDATE": "WUPD", "COMPUTE_LOSS": "LOSS", "TRAIN_NETWORK": "TNET",
    "RELU_BACKWARD": "RELUBK", "SIGMOID_BACKWARD": "SIGMBK", "TANH_BACKWARD": "TANHBK",
    "GELU_BACKWARD": "GELUBK", "SOFTMAX_BACKWARD": "SOFTBK", "MATMUL_BACKWARD": "MMULBK",
    "CONV_BACKWARD": "CONVBK", "POOL_BACKWARD": "POOLBK", "NORM_BACKWARD": "NORMBK",
    "ATTN_BACKWARD": "ATTNBK", "TRAIN_DEEP": "TNDEEP",
    # Short underscore aliases (LLM-natural spelling)
    "RELU_BK": "RELUBK", "SIGM_BK": "SIGMBK", "TANH_BK": "TANHBK",
    "GELU_BK": "GELUBK", "SOFT_BK": "SOFTBK", "LOSS_BK": "SOFTBK", "MMUL_BK": "MMULBK",
    "CONV_BK": "CONVBK", "POOL_BK": "POOLBK", "NORM_BK": "NORMBK",
    "ATTN_BK": "ATTNBK",
    # v0.9 config-driven training
    "TRAIN_LOG": "TLOG", "TRAIN_CONFIG": "TRAIN", "FORWARD_PASS": "INFER", "WEIGHT_DECAY": "WDECAY",
    # Phase 3
    "BATCH_NORM": "BN", "DROPOUT": "DROP",
}
_OPCODE_TO_CANONICAL.update(_VERBOSE_TO_CANONICAL)

_ALIAS_TO_CANONICAL = {
    "ϟ": "CMPI",       # Greek koppa — alternative symbolic for CMPI
    "ϛ": "RDUC",       # Greek stigma — alternative symbolic for RDUC
    "DOT": "SDOT",     # Classic alias for SDOT
    "SCTR": "SCAT",    # Rd-first alias for SCAT
    "JMP":   "JUMP",   # x86-style alias for JUMP
    "JMPNZ": "JMPT",  # jump-if-not-zero: alias for JMPT
    "JMPZ":  "JMPF",  # jump-if-zero:     alias for JMPF
}
_OPCODE_TO_CANONICAL.update(_ALIAS_TO_CANONICAL)


# ═══════════════════════════════════════════════════════════════════════════
# Operand count rules per canonical opcode: (min, max)
# ═══════════════════════════════════════════════════════════════════════════

_OPERAND_COUNTS: dict[str, tuple[int, int]] = {
    "LD":   (2, 2),
    "ST":   (2, 2),
    "MOV":  (2, 2),
    "ALLC": (2, 2),
    "MMUL": (3, 3),
    "MADD": (3, 3),
    "MSUB": (3, 3),
    "EMUL": (3, 3),
    "SDOT": (3, 3),
    "SCLR": (2, 3),
    "SDIV": (3, 3),
    "SADD": (3, 3),
    "SSUB": (3, 3),
    "EDIV": (3, 3),
    "LEAF": (2, 2),
    "TACC": (2, 3),
    "RELU": (2, 2),
    "SIGM": (2, 2),
    "TANH": (2, 2),
    "SOFT": (2, 2),
    "RSHP": (2, 3),
    "TRNS": (1, 2),
    "SPLT": (2, 4),
    "MERG": (2, 4),
    "CMPF": (3, 4),
    "CMP":  (2, 2),
    "CMPI": (2, 3),
    "JMPT": (1, 1),
    "JMPF": (1, 1),
    "JUMP": (1, 1),
    "LOOP": (0, 2),
    "ENDP": (0, 0),
    "CALL": (1, 2),
    "RET":  (0, 0),
    "SYNC": (0, 0),
    "HALT": (0, 0),
    "TRAP": (0, 1),
    "CONV": (3, 5),
    "POOL": (2, 4),
    "UPSC": (2, 3),
    "PADZ": (2, 3),
    "ATTN": (3, 4),
    "NORM": (2, 4),
    "EMBD": (2, 3),
    "GELU": (2, 2),
    "RDUC": (2, 4),
    "WHER": (3, 4),
    "CLMP": (3, 4),
    "CMPR": (3, 4),
    "FFT":  (2, 3),
    "FILT": (3, 4),
    "FRAG": (1, 1),
    "ENDF": (0, 0),
    "LINK": (1, 1),
    "VRFY": (2, 2),
    "VOTE": (2, 4),
    "PROJ": (3, 3),
    "DIST": (3, 4),
    "GATH": (3, 3),
    "SCAT": (3, 3),
    "SYS":  (2, 2),
    "MOD":  (3, 3),
    "ITOF": (2, 2),
    "FTOI": (2, 2),
    "BNOT": (2, 2),
    "BKWD": (3, 4),
    "WUPD": (3, 4),
    "LOSS": (3, 4),
    "TNET": (2, 9),
    "RELUBK": (3, 3),
    "SIGMBK": (3, 3),
    "TANHBK": (3, 3),
    "GELUBK": (3, 3),
    "SOFTBK": (3, 3),
    "MMULBK": (5, 5),
    "CONVBK": (5, 5),
    "POOLBK": (3, 5),
    "NORMBK": (3, 3),
    "ATTNBK": (4, 5),
    "TNDEEP": (2, 9),
    "BN":   (2, 4),
    "DROP": (2, 3),
}

_STRUCTURAL_OPCODES = {"META", "PTCH", "SIGN"}


# ═══════════════════════════════════════════════════════════════════════════
# Operand schemas for fix hints
# ═══════════════════════════════════════════════════════════════════════════

_OPERAND_SCHEMAS: dict[str, str] = {
    "LD": "Rd @memory|Rs|#imm", "ST": "Rs @memory", "MOV": "Rd Rs|#imm|@memory", "ALLC": "Rd #[N]",
    "MMUL": "Rd Rs1 Rs2", "MADD": "Rd Rs1 Rs2", "MSUB": "Rd Rs1 Rs2",
    "EMUL": "Rd Rs1 Rs2", "EDIV": "Rd Rs1 Rs2",
    "SDOT": "Rd Rs1 Rs2", "SCLR": "Rd Rs #imm", "SDIV": "Rd Rs1 Rs2|#imm",
    "SADD": "Rd Rs Rs2|#imm", "SSUB": "Rd Rs Rs2|#imm",
    "LEAF": "Rd #imm|@memory", "TACC": "Rd Rs [Rs2]",
    "RELU": "Rd Rs", "SIGM": "Rd Rs", "TANH": "Rd Rs", "SOFT": "Rd Rs", "GELU": "Rd Rs",
    "RSHP": "Rd Rs [#shape]", "TRNS": "Rd [Rs]", "SPLT": "Rd Rs [#idx] [#count]", "MERG": "Rd Rs1 [Rs2] [Rs3]",
    "CMPF": "Rflag Rs [#threshold] [Rs2]", "CMP": "Rflag Rs", "CMPI": "Rflag Rs|#imm [#imm]",
    "JMPT": "#offset", "JMPF": "#offset", "JUMP": "#offset",
    "LOOP": "[Rs] [#count]", "ENDP": "",
    "CALL": "#offset [Rs]", "RET": "", "SYNC": "", "HALT": "", "TRAP": "[#code]",
    "CONV": "Rd Rs Rkernel [#stride] [#pad]", "POOL": "Rd Rs [#size] [#stride]",
    "UPSC": "Rd Rs [#factor]", "PADZ": "Rd Rs [#amount]",
    "ATTN": "Rd Rq Rk [Rv]", "NORM": "Rd Rs [Rgamma] [Rbeta]",
    "EMBD": "Rd Rs [Rtable]", "RDUC": "Rd Rs [#op] [#axis]",
    "WHER": "Rd Rmask Rs1 [Rs2]", "CLMP": "Rd Rs #min [#max]",
    "CMPR": "Rd Rs1 Rs2 [#op]", "FFT": "Rd Rs [#inverse]", "FILT": "Rd Rs Rkernel [#mode]",
    "FRAG": "name", "ENDF": "", "LINK": "@module",
    "VRFY": "@prog @sig", "VOTE": "Rd Rs [#strategy] [#threshold]",
    "PROJ": "Rd Rs Rweight", "DIST": "Rd Rs1 Rs2 [#metric]",
    "GATH": "Rd Rs Ridx", "SCAT": "Rd Rs Ridx",
    "SYS": "Rd #code", "MOD": "Rd Rs1 Rs2",
    "ITOF": "Rd Rs", "FTOI": "Rd Rs", "BNOT": "Rd Rs",
    "BKWD": "Rd Rs1 Rs2 [Rs3]", "WUPD": "Rd Rgrad Rlr [Rmomentum]",
    "LOSS": "Rd Rpred Rtarget [#type]", "TNET": "Rconfig #epochs [#lr] ...",
    "RELUBK": "Rd Rs Rgrad", "SIGMBK": "Rd Rs Rgrad", "TANHBK": "Rd Rs Rgrad",
    "GELUBK": "Rd Rs Rgrad", "SOFTBK": "Rd Rs Rgrad",
    "MMULBK": "RdA RdB Rs1 Rs2 Rgrad", "CONVBK": "RdI RdK Rs Rkernel Rgrad",
    "POOLBK": "Rd Rs Rgrad [#size] [#stride]",
    "NORMBK": "Rd Rs Rgrad", "ATTNBK": "RdQ RdK Rs Rgrad [Rv]",
    "TNDEEP": "Rconfig #epochs [#lr] ...",
    "TLOG": "[#level]", "TRAIN": "Rconfig [#epochs] [#lr]", "INFER": "[Rs] [Rd]", "WDECAY": "Rweights #lambda",
    "BN": "Rd Rs [Rgamma] [Rbeta]", "DROP": "Rd Rs [#rate]",
}


def _levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(dp[j], dp[j - 1], prev)
            prev = temp
    return dp[n]


def _suggest_opcode(unknown: str) -> str | None:
    """Find the closest known opcode to an unknown string."""
    upper = unknown.upper()
    best, best_dist = None, float("inf")
    for op in _CLASSIC_OPCODES:
        d = _levenshtein(upper, op)
        if d < best_dist:
            best, best_dist = op, d
    for v in _VERBOSE_TO_CANONICAL:
        d = _levenshtein(upper, v)
        if d < best_dist:
            best, best_dist = v, d
    max_dist = 2 if len(upper) <= 4 else 3
    return best if best_dist <= max_dist else None


def _get_schema(canonical: str) -> str | None:
    """Get the operand schema string for a canonical opcode."""
    return _OPERAND_SCHEMAS.get(canonical)


def _schema_fix(canonical: str) -> str:
    """Build a fix hint from the operand schema."""
    schema = _get_schema(canonical)
    return f"Correct usage: {canonical} {schema}" if schema else ""


# ═══════════════════════════════════════════════════════════════════════════
# Operand classifiers
# ═══════════════════════════════════════════════════════════════════════════

_IMM_RE = re.compile(r"^#-?\d+\.?\d*$")
_ARR_RE = re.compile(r"^#\[\d+\]$")
_MEM_RE = re.compile(r"^@\w+$")

_VALID_TYPE_ANNOTATIONS = {
    "float", "currency", "ratio", "category",
    "count", "bool", "embedding", "probability",
}


def _strip_type_annotation(token: str) -> str:
    """Strip a :type suffix from a register token if present and valid."""
    if ":" in token:
        base, annotation = token.rsplit(":", 1)
        if annotation in _VALID_TYPE_ANNOTATIONS:
            return base
    return token


def _is_register(token: str) -> bool:
    return _strip_type_annotation(token) in ALL_REGISTERS


def _is_immediate(token: str) -> bool:
    return bool(_IMM_RE.match(token))


def _is_array_literal(token: str) -> bool:
    return bool(_ARR_RE.match(token))


def _is_memory_ref(token: str) -> bool:
    return bool(_MEM_RE.match(token))


def _canonical_register(token: str) -> str | None:
    return REGISTER_CANONICAL.get(_strip_type_annotation(token))


def _resolve_opcode(raw: str) -> str | None:
    """Resolve any opcode string to its canonical classic form."""
    c = _OPCODE_TO_CANONICAL.get(raw)
    if c is not None:
        return c
    return _OPCODE_TO_CANONICAL.get(raw.upper())


# ═══════════════════════════════════════════════════════════════════════════
# Operand-type validation per opcode category
# ═══════════════════════════════════════════════════════════════════════════

def _validate_operands(canonical: str, opcode_raw: str, operands: list[str], line_no: int, source_line: str = "") -> list[GrammarError]:
    """Type-check operands for a resolved canonical opcode."""
    errs: list[GrammarError] = []

    if canonical in ("LD", "LEAF", "MOV"):
        if not _is_register(operands[0]):
            errs.append(GrammarError(line_no, "invalid_register",
                f"First operand of {opcode_raw} must be a register, got '{operands[0]}'"))
        op1 = operands[1]
        if not (_is_register(op1) or _is_immediate(op1) or _is_memory_ref(op1)):
            errs.append(GrammarError(line_no, "invalid_operand",
                f"Second operand of {opcode_raw} must be register, #value, or @memory, got '{op1}'"))

    elif canonical == "ST":
        if not _is_register(operands[0]):
            errs.append(GrammarError(line_no, "invalid_register",
                f"First operand of {opcode_raw} must be a register, got '{operands[0]}'"))
        if not _is_memory_ref(operands[1]):
            errs.append(GrammarError(line_no, "invalid_operand",
                f"Second operand of {opcode_raw} must be @memory, got '{operands[1]}'"))

    elif canonical == "ALLC":
        if not _is_register(operands[0]):
            errs.append(GrammarError(line_no, "invalid_register",
                f"First operand of {opcode_raw} must be a register, got '{operands[0]}'"))
        if not _is_array_literal(operands[1]):
            errs.append(GrammarError(line_no, "invalid_operand",
                f"Second operand of {opcode_raw} must be #[N], got '{operands[1]}'"))

    elif canonical in ("RELU", "SIGM", "TANH", "SOFT", "TRNS", "GELU"):
        for i, op in enumerate(operands):
            if not _is_register(op) and not _is_immediate(op):
                errs.append(GrammarError(line_no, "invalid_operand",
                    f"Operand {i+1} of {opcode_raw} must be a register or #immediate, got '{op}'"))

    elif canonical in ("MMUL", "MADD", "MSUB", "EMUL", "SDOT", "SCLR",
                        "SDIV", "SADD", "SSUB", "EDIV", "TACC"):
        if not _is_register(operands[0]):
            errs.append(GrammarError(line_no, "invalid_register",
                f"Destination of {opcode_raw} must be a register, got '{operands[0]}'"))
        for i, op in enumerate(operands[1:], start=2):
            if not _is_register(op) and not _is_immediate(op):
                errs.append(GrammarError(line_no, "invalid_operand",
                    f"Operand {i} of {opcode_raw} must be register or #immediate, got '{op}'"))

    elif canonical == "CMPF":
        if not _is_register(operands[0]):
            errs.append(GrammarError(line_no, "invalid_register",
                f"Flag register of {opcode_raw} must be a register, got '{operands[0]}'"))
        if not _is_register(operands[1]):
            errs.append(GrammarError(line_no, "invalid_register",
                f"Source of {opcode_raw} must be a register, got '{operands[1]}'"))
        for idx in range(2, len(operands)):
            op = operands[idx]
            if not _is_immediate(op) and not _is_register(op):
                errs.append(GrammarError(line_no, "invalid_operand",
                    f"Operand {idx+1} of {opcode_raw} must be register or #immediate, got '{op}'"))

    elif canonical in ("CMP", "CMPI"):
        if not _is_register(operands[0]):
            errs.append(GrammarError(line_no, "invalid_register",
                f"Flag register of {opcode_raw} must be a register, got '{operands[0]}'"))
        for i, op in enumerate(operands[1:], start=2):
            if not _is_register(op) and not _is_immediate(op):
                errs.append(GrammarError(line_no, "invalid_operand",
                    f"Operand {i} of {opcode_raw} must be register or #immediate, got '{op}'"))

    elif canonical in ("JMPT", "JMPF", "JUMP", "CALL"):
        op = operands[0]
        raw = op[1:] if op.startswith("#") else op
        try:
            int(float(raw))
        except ValueError:
            errs.append(GrammarError(line_no, "invalid_immediate",
                f"Jump/call offset must be a number, got '{op}'"))

    elif canonical == "FRAG":
        if operands[0].startswith("@") or operands[0].startswith("#"):
            errs.append(GrammarError(line_no, "invalid_operand",
                f"FRAG name must be a plain identifier, got '{operands[0]}'"))

    elif canonical == "LINK":
        if not _is_memory_ref(operands[0]):
            errs.append(GrammarError(line_no, "invalid_operand",
                f"LINK operand must be @name, got '{operands[0]}'"))

    elif canonical == "VRFY":
        for i, op in enumerate(operands):
            if not _is_memory_ref(op):
                errs.append(GrammarError(line_no, "invalid_operand",
                    f"Operand {i+1} of {opcode_raw} must be @name, got '{op}'"))

    elif canonical == "VOTE":
        if not _is_register(operands[0]):
            errs.append(GrammarError(line_no, "invalid_register",
                f"Destination of {opcode_raw} must be a register, got '{operands[0]}'"))
        if len(operands) >= 2 and not _is_register(operands[1]):
            errs.append(GrammarError(line_no, "invalid_register",
                f"Source of {opcode_raw} must be a register, got '{operands[1]}'"))
        if len(operands) >= 3 and not _is_immediate(operands[2]):
            errs.append(GrammarError(line_no, "invalid_immediate",
                f"Strategy of {opcode_raw} must be #immediate, got '{operands[2]}'"))
        if len(operands) >= 4 and not _is_immediate(operands[3]):
            errs.append(GrammarError(line_no, "invalid_immediate",
                f"Threshold of {opcode_raw} must be #immediate, got '{operands[3]}'"))

    elif canonical == "PROJ":
        for i, op in enumerate(operands):
            if not _is_register(op):
                errs.append(GrammarError(line_no, "invalid_register",
                    f"Operand {i+1} of {opcode_raw} must be a register, got '{op}'"))

    elif canonical == "DIST":
        for i, op in enumerate(operands[:3]):
            if not _is_register(op):
                errs.append(GrammarError(line_no, "invalid_register",
                    f"Operand {i+1} of {opcode_raw} must be a register, got '{op}'"))
        if len(operands) >= 4 and not _is_immediate(operands[3]):
            errs.append(GrammarError(line_no, "invalid_immediate",
                f"Metric of {opcode_raw} must be #immediate, got '{operands[3]}'"))

    elif canonical in ("GATH", "SCAT"):
        for i, op in enumerate(operands):
            if not _is_register(op):
                errs.append(GrammarError(line_no, "invalid_register",
                    f"Operand {i+1} of {opcode_raw} must be a register, got '{op}'"))

    elif canonical == "SYS":
        if not _is_register(operands[0]):
            errs.append(GrammarError(line_no, "invalid_register",
                f"First operand of {opcode_raw} must be a register, got '{operands[0]}'"))
        if not _is_immediate(operands[1]):
            errs.append(GrammarError(line_no, "invalid_immediate",
                f"Second operand of {opcode_raw} must be #code, got '{operands[1]}'"))

    elif canonical == "MOD":
        for i, op in enumerate(operands):
            if not _is_register(op):
                errs.append(GrammarError(line_no, "invalid_register",
                    f"Operand {i+1} of {opcode_raw} must be a register, got '{op}'"))

    elif canonical in ("ITOF", "FTOI", "BNOT"):
        for i, op in enumerate(operands):
            if not _is_register(op):
                errs.append(GrammarError(line_no, "invalid_register",
                    f"Operand {i+1} of {opcode_raw} must be a register, got '{op}'"))

    # Attach source line and fix hint to all operand errors
    fix = _schema_fix(canonical)
    for err in errs:
        if not err.source:
            err.source = source_line
        if not err.fix:
            err.fix = fix

    return errs


# ═══════════════════════════════════════════════════════════════════════════
# Main validator
# ═══════════════════════════════════════════════════════════════════════════

def validate_grammar(nml_program: str) -> GrammarReport:
    """Validate the grammar of an NML program string."""
    errors: list[GrammarError] = []
    warnings: list[GrammarWarning] = []
    registers_used: set[str] = set()
    memory_inputs: list[str] = []
    memory_outputs: list[str] = []
    fragment_stack: list[tuple[int, str]] = []

    lines = nml_program.splitlines()
    instruction_lines: list[tuple[int, list[str]]] = []

    for line_no, raw in enumerate(lines, start=1):
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith(";"):
            continue
        if ";" in stripped:
            stripped = stripped[: stripped.index(";")].strip()
        if not stripped:
            continue
        tokens = stripped.split()
        instruction_lines.append((line_no, tokens))

    total_instructions = len(instruction_lines)

    for idx, (line_no, tokens) in enumerate(instruction_lines):
        opcode_raw = tokens[0]
        operands = tokens[1:]

        source_line = lines[line_no - 1].rstrip() if line_no <= len(lines) else ""

        canonical = _resolve_opcode(opcode_raw)
        if canonical is None:
            suggestion = _suggest_opcode(opcode_raw)
            if suggestion:
                resolved = _OPCODE_TO_CANONICAL.get(suggestion, suggestion)
                schema = _get_schema(resolved)
                fix = f"Did you mean '{suggestion}'?"
                if schema:
                    fix += f" Usage: {suggestion} {schema}"
            else:
                fix = "See NML spec for valid opcodes"
            errors.append(GrammarError(
                line=line_no,
                error_type="invalid_opcode",
                message=f"Unknown opcode '{opcode_raw}'",
                source=source_line,
                fix=fix,
            ))
            continue

        # --- structural opcodes: accept without operand validation ---
        if canonical in _STRUCTURAL_OPCODES:
            continue

        # --- fragment scope tracking ---
        if canonical == "FRAG":
            frag_name = operands[0] if operands else "<unnamed>"
            fragment_stack.append((line_no, frag_name))
        elif canonical == "ENDF":
            if fragment_stack:
                fragment_stack.pop()
            else:
                errors.append(GrammarError(
                    line=line_no,
                    error_type="unmatched_endf",
                    message="ENDF without matching FRAG",
                    source=source_line,
                    fix="Add a FRAG <name> instruction before this ENDF, or remove this ENDF",
                ))

        # --- operand count ---
        min_ops, max_ops = _OPERAND_COUNTS.get(canonical, (0, 10))
        if len(operands) < min_ops or len(operands) > max_ops:
            expected = str(min_ops) if min_ops == max_ops else f"{min_ops}-{max_ops}"
            errors.append(GrammarError(
                line=line_no,
                error_type="wrong_operand_count",
                message=f"{opcode_raw} expects {expected} operands, got {len(operands)}",
                source=source_line,
                fix=_schema_fix(canonical),
            ))
            continue

        # --- collect registers ---
        for tok in operands:
            cr = _canonical_register(tok)
            if cr is not None:
                registers_used.add(cr)

        # --- type-check operands ---
        if operands:
            errs = _validate_operands(canonical, opcode_raw, operands, line_no, source_line)
            errors.extend(errs)

        # --- track memory inputs / outputs ---
        if canonical == "LD" and len(operands) >= 2 and _is_memory_ref(operands[1]):
            name = operands[1][1:]
            if name not in memory_inputs:
                memory_inputs.append(name)

        if canonical == "ST" and len(operands) >= 2 and _is_memory_ref(operands[1]):
            name = operands[1][1:]
            if name not in memory_outputs:
                memory_outputs.append(name)

        # --- validate jump targets ---
        if canonical in ("JMPT", "JMPF", "JUMP") and operands:
            op = operands[0]
            if op.startswith("#"):
                try:
                    offset = int(op[1:])
                    target = idx + offset
                    if target < 0 or target >= total_instructions:
                        errors.append(GrammarError(
                            line=line_no,
                            error_type="invalid_jump",
                            message=(
                                f"Jump offset {offset} from instruction {idx+1} "
                                f"targets instruction {target+1}, "
                                f"out of range 1..{total_instructions}"
                            ),
                            source=source_line,
                            fix=f"Use an offset between {-idx} and {total_instructions - 1 - idx} (relative to current instruction {idx+1})",
                        ))
                except ValueError:
                    pass  # already reported by _validate_operands

    # --- unclosed fragments ---
    for frag_line, frag_name in fragment_stack:
        frag_src = lines[frag_line - 1].rstrip() if frag_line <= len(lines) else ""
        warnings.append(GrammarWarning(
            line=frag_line,
            warning_type="unclosed_fragment",
            message=f"Fragment '{frag_name}' opened but never closed with ENDF",
            source=frag_src,
            fix="Add ENDF after the fragment body to close this block",
        ))

    # --- check program contains HALT ---
    if instruction_lines:
        has_halt = any(
            _resolve_opcode(tokens[0]) == "HALT"
            for _, tokens in instruction_lines
        )
        if not has_halt:
            last_line_no = instruction_lines[-1][0]
            last_opcode = instruction_lines[-1][1][0]
            last_src = lines[last_line_no - 1].rstrip() if last_line_no <= len(lines) else ""
            errors.append(GrammarError(
                line=last_line_no,
                error_type="missing_halt",
                message=f"Program does not contain HALT (last opcode: '{last_opcode}')",
                source=last_src,
                fix="Add HALT as the last instruction of the program",
            ))
    else:
        errors.append(GrammarError(
            line=0,
            error_type="missing_halt",
            message="Empty program — no instructions found",
            fix="Write at least one instruction and end with HALT",
        ))

    # --- warnings ---
    if total_instructions > 0 and not memory_outputs:
        warnings.append(GrammarWarning(
            line=0,
            warning_type="no_output",
            message="Program has no store instructions — no outputs written",
            fix="Add ST <register> @<name> to write results to named memory",
        ))

    return GrammarReport(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        instruction_count=total_instructions,
        registers_used=registers_used,
        memory_inputs=memory_inputs,
        memory_outputs=memory_outputs,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Batch validation
# ═══════════════════════════════════════════════════════════════════════════

def validate_directory(directory: str) -> dict:
    """Validate all .nml files in a directory tree and return summary stats."""
    results = {
        "total_files": 0,
        "valid_files": 0,
        "invalid_files": 0,
        "total_errors": 0,
        "total_warnings": 0,
        "total_instructions": 0,
        "files": {},
    }

    root = Path(directory)
    for nml_path in sorted(root.rglob("*.nml")):
        results["total_files"] += 1
        with open(nml_path) as f:
            source = f.read()
        report = validate_grammar(source)
        rel = str(nml_path.relative_to(root))
        results["files"][rel] = report.to_dict()
        results["total_instructions"] += report.instruction_count
        results["total_errors"] += len(report.errors)
        results["total_warnings"] += len(report.warnings)
        if report.valid:
            results["valid_files"] += 1
        else:
            results["invalid_files"] += 1

    return results


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 nml_grammar.py <path.nml | directory>")
        sys.exit(1)

    target = sys.argv[1]

    if os.path.isdir(target):
        results = validate_directory(target)
        print(f"\n{'═' * 60}")
        print(f"  NML Grammar Validation — {target}")
        print(f"{'═' * 60}")
        print(f"  Files scanned:  {results['total_files']}")
        print(f"    Valid:        {results['valid_files']}")
        print(f"    Invalid:      {results['invalid_files']}")
        print(f"  Instructions:   {results['total_instructions']}")
        print(f"  Errors:         {results['total_errors']}")
        print(f"  Warnings:       {results['total_warnings']}")

        if results["invalid_files"] > 0:
            print(f"\n{'─' * 60}")
            print("  Invalid files:")
            for path, report in results["files"].items():
                if not report["valid"]:
                    print(f"\n    {path}")
                    for err in report["errors"]:
                        print(f"      L{err['line']:>4}  [{err['type']}] {err['message']}")

        print()
        sys.exit(0 if results["invalid_files"] == 0 else 1)

    else:
        if not os.path.exists(target):
            print(f"Error: file not found: {target}", file=sys.stderr)
            sys.exit(1)

        with open(target) as f:
            source = f.read()

        report = validate_grammar(source)

        print(f"\n{'═' * 60}")
        print(f"  NML Grammar Validation — {os.path.basename(target)}")
        print(f"{'═' * 60}")
        print(f"  Valid:          {'yes' if report.valid else 'NO'}")
        print(f"  Instructions:   {report.instruction_count}")
        print(f"  Registers used: {sorted(report.registers_used)}")
        print(f"  Memory inputs:  {report.memory_inputs}")
        print(f"  Memory outputs: {report.memory_outputs}")

        if report.errors:
            print(f"\n  Errors ({len(report.errors)}):")
            for err in report.errors:
                print(f"    L{err.line:>4}  [{err.error_type}] {err.message}")
                if err.source:
                    print(f"           Source: {err.source}")
                if err.fix:
                    print(f"           Fix:    {err.fix}")

        if report.warnings:
            print(f"\n  Warnings ({len(report.warnings)}):")
            for warn in report.warnings:
                print(f"    L{warn.line:>4}  [{warn.warning_type}] {warn.message}")
                if warn.fix:
                    print(f"           Fix:    {warn.fix}")

        print()

        if "--json" in sys.argv:
            print(json.dumps(report.to_dict(), indent=2))

        sys.exit(0 if report.valid else 1)


if __name__ == "__main__":
    main()
