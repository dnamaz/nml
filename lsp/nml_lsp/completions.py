"""
Context-aware NML completions.

- Start of line → opcode completions with snippet tabstops
- After opcode → register / memory / immediate completions
- After @ → memory ref names from file context
"""

from __future__ import annotations

import re

from lsprotocol.types import (
    CompletionItem,
    CompletionItemKind,
    CompletionList,
    InsertTextFormat,
    Position,
)

from .opcode_db import OPCODES, OpcodeInfo, REGISTERS, RegisterInfo

_MEM_RE = re.compile(r"@([a-z_][a-z0-9_]*)")


def _collect_memory_refs(source: str) -> list[str]:
    """Gather all @name references from the document."""
    seen: set[str] = set()
    result: list[str] = []
    for m in _MEM_RE.finditer(source):
        name = m.group(1)
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


def _opcode_completions() -> list[CompletionItem]:
    """Build completion items for all opcodes (classic syntax)."""
    items: list[CompletionItem] = []
    for info in OPCODES.values():
        detail_parts = [info.category]
        if info.symbolic:
            detail_parts.append(f"sym: {info.symbolic}")
        items.append(CompletionItem(
            label=info.canonical,
            kind=CompletionItemKind.Keyword,
            detail=" | ".join(detail_parts),
            documentation=f"{info.description}\n\nPattern: `{info.canonical} {info.operand_schema}`",
            insert_text=info.snippet,
            insert_text_format=InsertTextFormat.Snippet,
            sort_text=f"0_{info.canonical}",
        ))
        if info.symbolic:
            items.append(CompletionItem(
                label=info.symbolic,
                kind=CompletionItemKind.Operator,
                detail=f"{info.canonical} ({info.category})",
                documentation=info.description,
                insert_text=info.snippet.replace(info.canonical, info.symbolic, 1),
                insert_text_format=InsertTextFormat.Snippet,
                sort_text=f"1_{info.canonical}",
            ))
    return items


def _register_completions() -> list[CompletionItem]:
    """Build completion items for all registers."""
    items: list[CompletionItem] = []
    for info in REGISTERS.values():
        label_parts = [info.canonical]
        if info.greek:
            label_parts.append(info.greek)
        detail = info.purpose
        items.append(CompletionItem(
            label=info.canonical,
            kind=CompletionItemKind.Variable,
            detail=detail,
            documentation=f"Index {info.index} — {info.purpose}",
            sort_text=f"2_{info.index:02d}",
        ))
        if info.greek:
            items.append(CompletionItem(
                label=info.greek,
                kind=CompletionItemKind.Variable,
                detail=f"{info.canonical} — {info.purpose}",
                sort_text=f"3_{info.index:02d}",
            ))
    return items


def _memory_completions(source: str) -> list[CompletionItem]:
    """Build completion items for @memory refs found in the document."""
    items: list[CompletionItem] = []
    for name in _collect_memory_refs(source):
        items.append(CompletionItem(
            label=f"@{name}",
            kind=CompletionItemKind.Field,
            detail="Memory slot",
            sort_text=f"4_{name}",
        ))
    return items


def _get_line_context(source: str, position: Position) -> tuple[str, int]:
    """Return the current line text and the number of tokens before cursor."""
    lines = source.splitlines()
    if position.line >= len(lines):
        return "", 0
    line = lines[position.line]
    prefix = line[:position.character]
    stripped = prefix.split(";")[0]
    tokens = stripped.split()
    return prefix, len(tokens)


def get_completions(source: str, position: Position) -> CompletionList:
    """Return context-aware completions for the given cursor position."""
    prefix, token_count = _get_line_context(source, position)

    items: list[CompletionItem] = []

    if prefix.rstrip().endswith("@"):
        items.extend(_memory_completions(source))
    elif token_count == 0:
        items.extend(_opcode_completions())
    else:
        items.extend(_register_completions())
        items.extend(_memory_completions(source))

    return CompletionList(is_incomplete=False, items=items)
