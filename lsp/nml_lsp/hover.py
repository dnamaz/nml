"""
Hover information for NML tokens.

- Opcodes: description, operand pattern, category, all aliases
- Registers: index, canonical name, Greek, purpose
- Memory refs: LD/ST usage lines in the document
- Jump immediates: resolved target line
"""

from __future__ import annotations

import re
import sys
import os

from lsprotocol.types import (
    Hover,
    MarkupContent,
    MarkupKind,
    Position,
    Range,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "transpilers"))
from nml_grammar import _resolve_opcode, ALL_REGISTERS  # type: ignore[import-untyped]

from .opcode_db import lookup, lookup_register, OpcodeInfo, RegisterInfo

_IMM_RE = re.compile(r"^#?-?\d+\.?\d*$")
_MEM_RE = re.compile(r"^@[a-z_][a-z0-9_]*$")


def _word_at(line: str, character: int) -> tuple[str, int, int] | None:
    """Extract the token at the given character offset, returning (word, start, end)."""
    if character > len(line):
        return None

    start = character
    while start > 0 and not line[start - 1].isspace():
        start -= 1
    end = character
    while end < len(line) and not line[end].isspace():
        end += 1

    word = line[start:end]
    if not word:
        return None
    return word, start, end


def _opcode_hover(info: OpcodeInfo) -> str:
    """Build markdown hover content for an opcode."""
    lines = [
        f"**{info.canonical}** — {info.category}",
        "",
        info.description,
        "",
        f"**Pattern:** `{info.canonical} {info.operand_schema}`",
        "",
        f"**Operands:** {info.min_ops}" + (f"–{info.max_ops}" if info.min_ops != info.max_ops else ""),
    ]
    aliases: list[str] = []
    if info.symbolic:
        aliases.append(f"`{info.symbolic}` (symbolic)")
    if info.verbose and info.verbose != info.canonical:
        aliases.append(f"`{info.verbose}` (verbose)")
    for a in info.aliases:
        aliases.append(f"`{a}`")
    if aliases:
        lines.append("")
        lines.append("**Aliases:** " + ", ".join(aliases))
    return "\n".join(lines)


def _register_hover(info: RegisterInfo) -> str:
    """Build markdown hover content for a register."""
    parts = [f"**{info.canonical}**"]
    if info.greek:
        parts.append(f"/ {info.greek}")
    if info.verbose:
        parts.append(f"/ {info.verbose}")
    parts.append(f"— index {info.index}")
    line1 = " ".join(parts)
    return f"{line1}\n\n{info.purpose}"


def _memory_hover(name: str, source: str) -> str:
    """Build hover showing where @name is loaded / stored."""
    loads: list[int] = []
    stores: list[int] = []
    for i, line in enumerate(source.splitlines(), start=1):
        stripped = line.split(";")[0].strip()
        tokens = stripped.split()
        if len(tokens) < 2:
            continue
        canonical = _resolve_opcode(tokens[0])
        if canonical == "LD" and len(tokens) >= 3 and tokens[2] == f"@{name}":
            loads.append(i)
        elif canonical == "ST" and len(tokens) >= 3 and tokens[2] == f"@{name}":
            stores.append(i)
        elif canonical == "LD" and len(tokens) >= 3 and tokens[1] == f"@{name}":
            loads.append(i)

    lines = [f"**@{name}** — memory slot"]
    if loads:
        lines.append(f"\nLoaded (LD) on line(s): {', '.join(str(l) for l in loads)}")
    if stores:
        lines.append(f"\nStored (ST) on line(s): {', '.join(str(l) for l in stores)}")
    if not loads and not stores:
        lines.append("\n*No LD/ST references found in this file.*")
    return "\n".join(lines)


def _jump_hover(token: str, line_idx: int, source: str) -> str | None:
    """For a jump/call immediate, show the resolved target line."""
    raw = token.lstrip("#")
    try:
        offset = int(float(raw))
    except ValueError:
        return None

    instruction_lines: list[int] = []
    for i, line in enumerate(source.splitlines()):
        stripped = line.split(";")[0].strip()
        if stripped and not stripped.startswith(";"):
            instruction_lines.append(i)

    current_instr_idx = None
    for idx, il in enumerate(instruction_lines):
        if il == line_idx:
            current_instr_idx = idx
            break

    if current_instr_idx is None:
        return None

    target_instr = current_instr_idx + offset
    if 0 <= target_instr < len(instruction_lines):
        target_line = instruction_lines[target_instr] + 1
        target_text = source.splitlines()[instruction_lines[target_instr]].strip()
        return (
            f"**Jump offset** `{token}`\n\n"
            f"Resolves to **line {target_line}**: `{target_text}`\n\n"
            f"*(PC {current_instr_idx} + {offset} + 1 = instruction {target_instr + 1})*"
        )
    else:
        return f"**Jump offset** `{token}` — **out of bounds** (target instruction {target_instr + 1})"


def get_hover(source: str, position: Position) -> Hover | None:
    """Return hover information for the token at the given position."""
    lines = source.splitlines()
    if position.line >= len(lines):
        return None

    line = lines[position.line]
    result = _word_at(line, position.character)
    if result is None:
        return None

    word, start, end = result
    token_range = Range(
        start=Position(line=position.line, character=start),
        end=Position(line=position.line, character=end),
    )

    stripped = line.split(";")[0].strip()
    tokens = stripped.split()
    token_idx = 0
    for i, t in enumerate(tokens):
        if t == word:
            token_idx = i
            break

    # Opcode
    opinfo = lookup(word)
    if opinfo:
        return Hover(
            contents=MarkupContent(kind=MarkupKind.Markdown, value=_opcode_hover(opinfo)),
            range=token_range,
        )

    # Register
    clean = word.split(":")[0]  # strip type annotation
    reginfo = lookup_register(clean)
    if reginfo:
        return Hover(
            contents=MarkupContent(kind=MarkupKind.Markdown, value=_register_hover(reginfo)),
            range=token_range,
        )

    # Memory ref
    if _MEM_RE.match(word):
        name = word[1:]
        return Hover(
            contents=MarkupContent(kind=MarkupKind.Markdown, value=_memory_hover(name, source)),
            range=token_range,
        )

    # Jump/call immediate — only when the opcode is a jump/call
    if _IMM_RE.match(word) and token_idx > 0:
        opcode_word = tokens[0] if tokens else ""
        canonical = _resolve_opcode(opcode_word)
        if canonical in ("JMPT", "JMPF", "JUMP", "CALL"):
            hover_text = _jump_hover(word, position.line, source)
            if hover_text:
                return Hover(
                    contents=MarkupContent(kind=MarkupKind.Markdown, value=hover_text),
                    range=token_range,
                )

    return None
