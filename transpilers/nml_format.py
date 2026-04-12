#!/usr/bin/env python3
"""
NML Compact/Format — convert between multi-line and single-line NML.

Compact form uses │ (U+2502) as the instruction delimiter, stripping
comments and collapsing whitespace. The C runtime natively parses both
newline-delimited and │-delimited programs.

Usage:
    python3 nml_format.py compact  program.nml          # → stdout
    python3 nml_format.py format   "∎ α #42│↑ α @r│◼"  # → stdout
    python3 nml_format.py compact  program.nml -o out.nml
"""

import re
import sys
from pathlib import Path

COMPACT_DELIM = "\u00b6"  # ¶

SYMBOLIC_OPCODES = frozenset(
    "× ⊕ ⊖ ⊗ ⊘ · ∗ ÷ ⌐ σ τ Σ ↓ ↑ ← □ ⊟ ⊤ ⊢ ⊣ "
    "⋈ ≶ ≺ ↗ ↘ → ↻ ↺ ∎ ∑ ⇒ ⇐ ⏸ ◼ ⚠ "
    "⊛ ⊓ ⊔ ⊡ ⊙ ‖ ⊏ ℊ ⊥ ⊻ ⊧ ⊜ ∿ ⋐ ⚙ ⊚ ⊞".split()
)

CLASSIC_OPCODES = frozenset(
    "MMUL MADD MSUB EMUL EDIV SDOT SCLR SDIV "
    "RELU SIGM TANH SOFT "
    "LD ST MOV ALLC RSHP TRNS SPLT MERG "
    "CMPF CMP CMPI JMPT JMPF JUMP LOOP ENDP "
    "LEAF TACC CALL RET SYNC HALT TRAP "
    "CONV POOL UPSC PADZ ATTN NORM EMBD GELU "
    "RDUC WHER CLMP CMPR FFT FILT "
    "META FRAG ENDF LINK PTCH SIGN VRFY VOTE PROJ DIST GATH SCAT "
    "SYS MOD".split()
)


def compact(nml_source: str) -> str:
    """Convert multi-line NML to single-line │-delimited compact form."""
    instructions = []
    for line in nml_source.split("\n"):
        comment_pos = line.find(";")
        if comment_pos >= 0:
            line = line[:comment_pos]
        stripped = line.strip()
        if not stripped:
            continue
        normalized = re.sub(r"[ \t]+", " ", stripped)
        instructions.append(normalized)

    return COMPACT_DELIM.join(instructions)


def format_nml(compact_source: str, align: bool = True) -> str:
    """Convert compact (or multi-line) NML to formatted multi-line output.

    If align is True, pads opcode and operand columns for readability.
    """
    if COMPACT_DELIM in compact_source:
        parts = compact_source.split(COMPACT_DELIM)
    else:
        parts = compact_source.split("\n")

    instructions = []
    for part in parts:
        stripped = part.strip()
        if not stripped:
            continue
        instructions.append(stripped)

    if not align:
        return "\n".join(instructions)

    lines = []
    for instr in instructions:
        tokens = instr.split()
        if not tokens:
            continue

        opcode = tokens[0]

        is_meta = opcode.upper() == "META" or opcode == "§"
        if is_meta:
            lines.append("  ".join(tokens))
            continue

        is_symbolic = opcode in SYMBOLIC_OPCODES
        if is_symbolic:
            if len(tokens) == 1:
                lines.append(opcode)
            else:
                operands = "  ".join(tokens[1:])
                lines.append(f"{opcode}  {operands}")
        else:
            if len(tokens) == 1:
                lines.append(opcode)
            else:
                operands = " ".join(tokens[1:])
                lines.append(f"{opcode:<6}{operands}")

    return "\n".join(lines)


def is_compact(nml_source: str) -> bool:
    """Check if an NML program is in compact (single-line) form."""
    return COMPACT_DELIM in nml_source and "\n" not in nml_source.strip()


def detect_syntax(nml_source: str) -> str:
    """Detect whether NML source uses symbolic, classic, or verbose syntax."""
    sample = nml_source[:500]
    for sym in SYMBOLIC_OPCODES:
        if sym in sample:
            return "symbolic"
    for op in ("LEAF", "TACC", "HALT", "SCLR", "CMPF", "JMPF"):
        if op in sample:
            return "classic"
    for op in ("SET_VALUE", "ACCUMULATE", "STOP", "LOAD", "STORE"):
        if op in sample:
            return "verbose"
    return "unknown"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="NML Compact/Format converter")
    parser.add_argument("mode", choices=["compact", "format", "detect"],
                        help="compact: multi-line → single-line; format: single-line → multi-line; detect: show syntax type")
    parser.add_argument("input", help="NML file path or quoted NML string")
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    parser.add_argument("--no-align", action="store_true", help="Skip column alignment in format mode")

    args = parser.parse_args()

    if Path(args.input).is_file():
        source = Path(args.input).read_text()
    else:
        source = args.input

    if args.mode == "compact":
        result = compact(source)
    elif args.mode == "format":
        result = format_nml(source, align=not args.no_align)
    elif args.mode == "detect":
        syntax = detect_syntax(source)
        is_comp = is_compact(source)
        lines = len(source.split(COMPACT_DELIM)) if is_comp else len(source.strip().split("\n"))
        result = f"Syntax: {syntax}\nCompact: {is_comp}\nInstructions: {lines}"

    if args.output:
        Path(args.output).write_text(result + "\n")
        print(f"Written to {args.output}")
    else:
        print(result)


if __name__ == "__main__":
    main()
