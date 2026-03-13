"""
Semantic token provider for NML.

Classifies every token in a line as opcode, register, immediate,
memory ref, shape literal, comment, fragment name, or decorator (META).
"""

from __future__ import annotations

import re
import sys
import os

from lsprotocol.types import SemanticTokens

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "transpilers"))
from nml_grammar import _resolve_opcode, ALL_REGISTERS, _strip_type_annotation  # type: ignore[import-untyped]

TOKEN_TYPES = [
    "keyword",      # 0  opcode
    "variable",     # 1  register
    "number",       # 2  immediate
    "string",       # 3  memory ref @name
    "comment",      # 4  ; comment
    "type",         # 5  shape literal #[N]
    "namespace",    # 6  FRAG name
    "decorator",    # 7  META keyword
]
TOKEN_MODIFIERS: list[str] = []

_TYPE_INDEX = {t: i for i, t in enumerate(TOKEN_TYPES)}

_IMM_RE = re.compile(r"^#-?\d+\.?\d*$")
_BARE_NUM_RE = re.compile(r"^-?\d+\.?\d*$")
_MEM_RE = re.compile(r"^@[a-z_][a-z0-9_]*$")
_SHAPE_RE = re.compile(r"^#?\[\d+(,\d+)*\]$")


def _classify_token(token: str, token_idx: int, canonical_opcode: str | None) -> int | None:
    """Return the TOKEN_TYPES index for a token, or None to skip."""
    if _resolve_opcode(token) is not None:
        if canonical_opcode in ("META",) and token_idx == 0:
            return _TYPE_INDEX["decorator"]
        return _TYPE_INDEX["keyword"]

    clean = _strip_type_annotation(token)
    if clean in ALL_REGISTERS:
        return _TYPE_INDEX["variable"]

    if _SHAPE_RE.match(token):
        return _TYPE_INDEX["type"]

    if _IMM_RE.match(token):
        return _TYPE_INDEX["number"]

    if _MEM_RE.match(token):
        return _TYPE_INDEX["string"]

    if _BARE_NUM_RE.match(token) and canonical_opcode in ("JMPT", "JMPF", "JUMP", "CALL"):
        return _TYPE_INDEX["number"]

    if canonical_opcode == "FRAG" and token_idx == 1:
        return _TYPE_INDEX["namespace"]

    return None


def get_semantic_tokens(source: str) -> SemanticTokens:
    """Compute semantic tokens for the full document."""
    data: list[int] = []
    prev_line = 0
    prev_start = 0

    for line_idx, line in enumerate(source.splitlines()):
        comment_pos = line.find(";")
        if comment_pos >= 0:
            comment_len = len(line) - comment_pos
            delta_line = line_idx - prev_line
            delta_start = comment_pos if delta_line > 0 else comment_pos - prev_start
            data.extend([delta_line, delta_start, comment_len, _TYPE_INDEX["comment"], 0])
            prev_line = line_idx
            prev_start = comment_pos

        code_part = line[:comment_pos] if comment_pos >= 0 else line
        stripped = code_part.strip()
        if not stripped:
            continue

        tokens = stripped.split()
        canonical = _resolve_opcode(tokens[0]) if tokens else None

        col = 0
        for t_idx, token in enumerate(tokens):
            tok_start = code_part.find(token, col)
            if tok_start < 0:
                continue
            tok_len = len(token)
            col = tok_start + tok_len

            tok_type = _classify_token(token, t_idx, canonical)
            if tok_type is None:
                continue

            delta_line = line_idx - prev_line
            delta_start = tok_start if delta_line > 0 else tok_start - prev_start
            data.extend([delta_line, delta_start, tok_len, tok_type, 0])
            prev_line = line_idx
            prev_start = tok_start

    return SemanticTokens(data=data)
