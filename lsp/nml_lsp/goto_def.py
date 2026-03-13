"""
Go-to-definition for NML.

- @name → first LD/ST of that name in the file, or definition in .nml.data
- LINK @fragment → FRAG declaration
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from urllib.parse import urlparse, unquote

from lsprotocol.types import Location, Position, Range

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "transpilers"))
from nml_grammar import _resolve_opcode  # type: ignore[import-untyped]

_MEM_RE = re.compile(r"^@([a-z_][a-z0-9_]*)$")


def _word_at(line: str, character: int) -> str | None:
    """Extract the token at the given character offset."""
    start = character
    while start > 0 and not line[start - 1].isspace():
        start -= 1
    end = character
    while end < len(line) and not line[end].isspace():
        end += 1
    word = line[start:end]
    return word if word else None


def _uri_to_path(uri: str) -> str:
    """Convert a file URI to a filesystem path."""
    parsed = urlparse(uri)
    return unquote(parsed.path)


def _path_to_uri(path: str) -> str:
    """Convert a filesystem path to a file URI."""
    return f"file://{path}"


def _find_in_nml_data(name: str, nml_uri: str) -> Location | None:
    """Search the companion .nml.data file for @name definition."""
    nml_path = _uri_to_path(nml_uri)
    data_path = nml_path + ".data"
    if not os.path.isfile(data_path):
        return None

    try:
        with open(data_path) as f:
            data_lines = f.readlines()
    except OSError:
        return None

    pattern = f"@{name}"
    for i, line in enumerate(data_lines):
        if line.strip().startswith(pattern):
            col = line.find(pattern)
            return Location(
                uri=_path_to_uri(data_path),
                range=Range(
                    start=Position(line=i, character=col),
                    end=Position(line=i, character=col + len(pattern)),
                ),
            )
    return None


def _find_memory_in_source(name: str, source: str, uri: str) -> list[Location]:
    """Find LD/ST lines referencing @name in the current file."""
    locations: list[Location] = []
    target = f"@{name}"
    for i, line in enumerate(source.splitlines()):
        stripped = line.split(";")[0].strip()
        tokens = stripped.split()
        if len(tokens) < 2:
            continue
        canonical = _resolve_opcode(tokens[0])
        if canonical in ("LD", "ST") and target in tokens:
            col = line.find(target)
            locations.append(Location(
                uri=uri,
                range=Range(
                    start=Position(line=i, character=col),
                    end=Position(line=i, character=col + len(target)),
                ),
            ))
    return locations


def _find_frag_declaration(frag_name: str, source: str, uri: str) -> Location | None:
    """Find the FRAG declaration for a given fragment name."""
    for i, line in enumerate(source.splitlines()):
        stripped = line.split(";")[0].strip()
        tokens = stripped.split()
        if len(tokens) >= 2:
            canonical = _resolve_opcode(tokens[0])
            if canonical == "FRAG" and tokens[1] == frag_name:
                col = line.find(frag_name, len(tokens[0]))
                return Location(
                    uri=uri,
                    range=Range(
                        start=Position(line=i, character=col),
                        end=Position(line=i, character=col + len(frag_name)),
                    ),
                )
    return None


def get_definition(source: str, position: Position, uri: str) -> Location | list[Location] | None:
    """Return definition location(s) for the token at the given position."""
    lines = source.splitlines()
    if position.line >= len(lines):
        return None

    line = lines[position.line]
    word = _word_at(line, position.character)
    if not word:
        return None

    m = _MEM_RE.match(word)
    if m:
        name = m.group(1)

        stripped = line.split(";")[0].strip()
        tokens = stripped.split()
        canonical = _resolve_opcode(tokens[0]) if tokens else None
        if canonical == "LINK":
            loc = _find_frag_declaration(name, source, uri)
            if loc:
                return loc

        data_loc = _find_in_nml_data(name, uri)
        if data_loc:
            return data_loc

        in_source = _find_memory_in_source(name, source, uri)
        current = [l for l in in_source if l.range.start.line != position.line]
        if current:
            return current
        if in_source:
            return in_source

    return None
