"""
NML Program Builder — shared label-based instruction emitter.

Supports three output syntaxes:
  - "classic"  : LEAF RA #236.80    (default, backward-compatible)
  - "symbolic" : ∎ α #236.80        (Unicode math symbols + Greek registers)
  - "verbose"  : SET_VALUE ACCUMULATOR #236.80  (human-readable)
"""

OPCODE_MAP = {
    "symbolic": {
        "MMUL": "×", "MADD": "⊕", "MSUB": "⊖", "EMUL": "⊗", "EDIV": "⊘",
        "SDOT": "·", "SCLR": "∗", "SDIV": "÷",
        "RELU": "⌐", "SIGM": "σ", "TANH": "τ", "SOFT": "Σ",
        "LD": "↓", "ST": "↑", "MOV": "←", "ALLC": "□",
        "RSHP": "⊟", "TRNS": "⊤", "SPLT": "⊢", "MERG": "⊣",
        "LOOP": "↻", "ENDP": "↺",
        "SYNC": "⏸", "HALT": "◼",
        "CMPF": "⋈", "CMP": "≶", "CMPI": "≺",
        "LEAF": "∎", "TACC": "∑",
        "JMPT": "↗", "JMPF": "↘", "JUMP": "→",
        "CALL": "⇒", "RET": "⇐", "TRAP": "⚠",
        "CONV": "⊛", "POOL": "⊓", "UPSC": "⊔", "PADZ": "⊡",
        "ATTN": "⊙", "NORM": "‖", "EMBD": "⊏", "GELU": "ℊ",
        "RDUC": "⊥", "WHER": "⊻", "CLMP": "⊧", "CMPR": "⊜",
        "FFT": "∿", "FILT": "⋐",
        "META": "§", "FRAG": "◆", "ENDF": "◇", "LINK": "⊚",
        "PTCH": "⊿", "SIGN": "✦", "VRFY": "✓",
        "VOTE": "⚖", "PROJ": "⟐", "DIST": "⟂", "GATH": "⊃", "SCAT": "⊂",
    },
    "verbose": {
        "MMUL": "MATRIX_MULTIPLY", "MADD": "ADD", "MSUB": "SUBTRACT",
        "EMUL": "ELEMENT_MULTIPLY", "EDIV": "ELEMENT_DIVIDE",
        "SDOT": "DOT_PRODUCT", "SCLR": "SCALE", "SDIV": "DIVIDE",
        "RELU": "RELU", "SIGM": "SIGMOID", "TANH": "TANH", "SOFT": "SOFTMAX",
        "LD": "LOAD", "ST": "STORE", "MOV": "COPY", "ALLC": "ALLOCATE",
        "RSHP": "RESHAPE", "TRNS": "TRANSPOSE", "SPLT": "SPLIT", "MERG": "MERGE",
        "LOOP": "REPEAT", "ENDP": "END_REPEAT",
        "SYNC": "BARRIER", "HALT": "STOP",
        "CMPF": "COMPARE_FEATURE", "CMP": "COMPARE", "CMPI": "COMPARE_VALUE",
        "LEAF": "SET_VALUE", "TACC": "ACCUMULATE",
        "JMPT": "BRANCH_TRUE", "JMPF": "BRANCH_FALSE", "JUMP": "JUMP",
        "CALL": "CALL", "RET": "RETURN", "TRAP": "FAULT",
        "CONV": "CONVOLVE", "POOL": "MAX_POOL", "UPSC": "UPSCALE", "PADZ": "ZERO_PAD",
        "ATTN": "ATTENTION", "NORM": "LAYER_NORM", "EMBD": "EMBED", "GELU": "GELU",
        "RDUC": "REDUCE", "WHER": "WHERE", "CLMP": "CLAMP", "CMPR": "MASK_COMPARE",
        "FFT": "FOURIER", "FILT": "FILTER",
        "META": "METADATA", "FRAG": "FRAGMENT", "ENDF": "END_FRAGMENT", "LINK": "IMPORT",
        "PTCH": "PATCH", "SIGN": "SIGN_PROGRAM", "VRFY": "VERIFY_SIGNATURE",
        "VOTE": "CONSENSUS", "PROJ": "PROJECT", "DIST": "DISTANCE", "GATH": "GATHER", "SCAT": "SCATTER",
    },
}

REGISTER_MAP = {
    "symbolic": {
        "R0": "ι", "R1": "κ", "R2": "λ", "R3": "μ", "R4": "ν",
        "R5": "ξ", "R6": "ο", "R7": "π", "R8": "ρ", "R9": "ς",
        "RA": "α", "RB": "β", "RC": "γ", "RD": "δ", "RE": "φ", "RF": "ψ",
    },
    "verbose": {
        "R0": "R0", "R1": "R1", "R2": "R2", "R3": "R3", "R4": "R4",
        "R5": "R5", "R6": "R6", "R7": "R7", "R8": "R8", "R9": "R9",
        "RA": "ACCUMULATOR", "RB": "GENERAL", "RC": "SCRATCH",
        "RD": "COUNTER", "RE": "FLAG", "RF": "STACK",
    },
}


def translate_line(line: str, syntax: str) -> str:
    """Translate one NML instruction line from classic to target syntax."""
    if syntax == "classic" or not line.strip() or line.strip().startswith(";"):
        return line

    stripped = line.split(";")[0].strip()
    comment_part = ""
    if ";" in line:
        idx = line.index(";")
        comment_part = " " + line[idx:]

    tokens = stripped.split()
    if not tokens:
        return line

    opmap = OPCODE_MAP.get(syntax, {})
    regmap = REGISTER_MAP.get(syntax, {})

    opcode = tokens[0]
    translated_op = opmap.get(opcode, opcode)
    translated_args = []
    for tok in tokens[1:]:
        upper = tok.upper()
        if upper in regmap:
            translated_args.append(regmap[upper])
        else:
            translated_args.append(tok)

    if syntax == "symbolic":
        parts = [translated_op] + translated_args
        return "  ".join(parts) + comment_part
    else:
        parts = [translated_op.ljust(18)] + translated_args
        return " ".join(parts) + comment_part


class NMLProgram:
    """Builds an NML program with automatic jump offset resolution."""

    def __init__(self, syntax="classic", comments=True):
        self.entries = []
        self.labels = {}
        self._instr_count = 0
        self.syntax = syntax
        self.comments = comments

    def comment(self, text):
        self.entries.append({"type": "comment", "text": f"; {text}"})

    def blank(self):
        self.entries.append({"type": "comment", "text": ""})

    def instr(self, text):
        self.entries.append({"type": "instr", "text": text, "idx": self._instr_count})
        self._instr_count += 1

    def jump(self, opcode, label):
        self.entries.append({
            "type": "instr", "text": opcode, "idx": self._instr_count,
            "fixup": label,
        })
        self._instr_count += 1

    def label(self, name):
        if name in self.labels:
            raise ValueError(f"Duplicate label: {name}")
        self.labels[name] = self._instr_count

    def build(self):
        lines = []
        for e in self.entries:
            if e["type"] == "comment":
                if self.comments:
                    lines.append(e["text"])
            elif "fixup" in e:
                target = self.labels[e["fixup"]]
                offset = target - e["idx"] - 1
                classic_line = f"{e['text']}  #{offset}"
                lines.append(translate_line(classic_line, self.syntax))
            else:
                lines.append(translate_line(e["text"], self.syntax))
        if not self.comments:
            lines = [l for l in lines if l.strip()]
        return "\n".join(lines)

    def meta(self, key, value):
        """Emit a META instruction (not counted as executable)."""
        if isinstance(value, str) and " " in value:
            self.entries.append({"type": "comment", "text": translate_line(f'META  @{key}  "{value}"', self.syntax)})
        else:
            self.entries.append({"type": "comment", "text": translate_line(f'META  @{key}  {value}', self.syntax)})

    def fragment(self, name):
        """Open a named fragment scope."""
        self.entries.append({"type": "comment", "text": translate_line(f'FRAG  {name}', self.syntax)})

    def end_fragment(self):
        """Close the current fragment scope."""
        self.entries.append({"type": "comment", "text": translate_line(f'ENDF', self.syntax)})

    def link(self, name):
        """Import a named fragment."""
        self.instr(f'LINK  @{name}')

    @property
    def instruction_count(self):
        return self._instr_count
