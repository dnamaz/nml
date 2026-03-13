"""
Document symbol provider for NML.

Reports:
- META keys as Property
- FRAG/ENDF blocks as Function (with range)
- @memory slots as Variable
- CALL targets as Method
"""

from __future__ import annotations

import re
import sys
import os

from lsprotocol.types import (
    DocumentSymbol,
    Position,
    Range,
    SymbolKind,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "transpilers"))
from nml_grammar import _resolve_opcode  # type: ignore[import-untyped]

_MEM_RE = re.compile(r"@([a-z_][a-z0-9_]*)")


def get_document_symbols(source: str) -> list[DocumentSymbol]:
    """Extract document symbols from NML source."""
    symbols: list[DocumentSymbol] = []
    lines = source.splitlines()
    frag_stack: list[tuple[str, int]] = []
    memory_seen: set[str] = set()

    for line_idx, line in enumerate(lines):
        stripped = line.split(";")[0].strip()
        if not stripped:
            continue

        tokens = stripped.split()
        canonical = _resolve_opcode(tokens[0])

        if canonical == "META" and len(tokens) >= 3:
            key = tokens[1]
            value = " ".join(tokens[2:])
            symbols.append(DocumentSymbol(
                name=f"{key} = {value}",
                kind=SymbolKind.Property,
                range=Range(
                    start=Position(line=line_idx, character=0),
                    end=Position(line=line_idx, character=len(line)),
                ),
                selection_range=Range(
                    start=Position(line=line_idx, character=0),
                    end=Position(line=line_idx, character=len(line)),
                ),
            ))

        elif canonical == "FRAG" and len(tokens) >= 2:
            frag_name = tokens[1]
            frag_stack.append((frag_name, line_idx))

        elif canonical == "ENDF" and frag_stack:
            frag_name, start_line = frag_stack.pop()
            symbols.append(DocumentSymbol(
                name=f"FRAG {frag_name}",
                kind=SymbolKind.Function,
                range=Range(
                    start=Position(line=start_line, character=0),
                    end=Position(line=line_idx, character=len(line)),
                ),
                selection_range=Range(
                    start=Position(line=start_line, character=0),
                    end=Position(line=start_line, character=len(lines[start_line])),
                ),
            ))

        if canonical in ("LD", "ST"):
            for m in _MEM_RE.finditer(stripped):
                name = m.group(1)
                if name not in memory_seen:
                    memory_seen.add(name)
                    symbols.append(DocumentSymbol(
                        name=f"@{name}",
                        kind=SymbolKind.Variable,
                        range=Range(
                            start=Position(line=line_idx, character=0),
                            end=Position(line=line_idx, character=len(line)),
                        ),
                        selection_range=Range(
                            start=Position(line=line_idx, character=line.find(f"@{name}")),
                            end=Position(line=line_idx, character=line.find(f"@{name}") + len(name) + 1),
                        ),
                    ))

    for frag_name, start_line in frag_stack:
        symbols.append(DocumentSymbol(
            name=f"FRAG {frag_name} (unclosed)",
            kind=SymbolKind.Function,
            range=Range(
                start=Position(line=start_line, character=0),
                end=Position(line=len(lines) - 1, character=len(lines[-1]) if lines else 0),
            ),
            selection_range=Range(
                start=Position(line=start_line, character=0),
                end=Position(line=start_line, character=len(lines[start_line])),
            ),
        ))

    return symbols
