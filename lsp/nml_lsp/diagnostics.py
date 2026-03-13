"""
Bridge between nml_grammar.validate_grammar() and LSP Diagnostic objects.
"""

from __future__ import annotations

import sys
import os

from lsprotocol.types import (
    Diagnostic,
    DiagnosticSeverity,
    Position,
    Range,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "transpilers"))
from nml_grammar import validate_grammar, GrammarReport  # type: ignore[import-untyped]

_SEVERITY_MAP: dict[str, DiagnosticSeverity] = {
    "invalid_opcode": DiagnosticSeverity.Error,
    "wrong_operand_count": DiagnosticSeverity.Error,
    "invalid_register": DiagnosticSeverity.Error,
    "invalid_operand": DiagnosticSeverity.Error,
    "invalid_immediate": DiagnosticSeverity.Error,
    "invalid_jump": DiagnosticSeverity.Warning,
    "unmatched_endf": DiagnosticSeverity.Error,
    "missing_halt": DiagnosticSeverity.Error,
    "unclosed_fragment": DiagnosticSeverity.Warning,
    "no_output": DiagnosticSeverity.Hint,
}


def _line_length(source: str, line_0based: int) -> int:
    """Return the length of a specific line in the source."""
    lines = source.splitlines()
    if 0 <= line_0based < len(lines):
        return len(lines[line_0based])
    return 0


def get_diagnostics(source: str) -> list[Diagnostic]:
    """Run the NML grammar validator and return LSP diagnostics."""
    report: GrammarReport = validate_grammar(source)
    diagnostics: list[Diagnostic] = []

    for err in report.errors:
        line_0 = max(0, err.line - 1)
        end_col = _line_length(source, line_0)
        diagnostics.append(Diagnostic(
            range=Range(
                start=Position(line=line_0, character=0),
                end=Position(line=line_0, character=end_col),
            ),
            message=err.message,
            severity=_SEVERITY_MAP.get(err.error_type, DiagnosticSeverity.Error),
            source="nml",
            code=err.error_type,
        ))

    for warn in report.warnings:
        line_0 = max(0, warn.line - 1)
        end_col = _line_length(source, line_0)
        diagnostics.append(Diagnostic(
            range=Range(
                start=Position(line=line_0, character=0),
                end=Position(line=line_0, character=end_col),
            ),
            message=warn.message,
            severity=_SEVERITY_MAP.get(warn.warning_type, DiagnosticSeverity.Warning),
            source="nml",
            code=warn.warning_type,
        ))

    return diagnostics
