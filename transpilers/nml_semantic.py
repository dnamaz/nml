#!/usr/bin/env python3
"""
NML Semantic Analyzer — domain-specific validation for tax NML programs.

Goes beyond grammar validation to check tax-domain invariants:
bracket monotonicity, rate bounds, filing status coverage,
input/output completeness, and standard deduction extraction.

Operates on SYMBOLIC syntax NML programs.
"""

import json
import sys
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


FILING_STATUS_MAP = {
    1.5: "Single",
    2.5: "MFJ",
    3.5: "MFS",
    4.5: "HoH",
}

INCOME_TAX_TYPES = {"FIT", "SIT", "COUNTY"}

BRACKET_GATE_MIN_JUMP = 5

META_OPCODES = {"META", "§", "METADATA"}
FRAG_OPCODES = {"FRAG", "◆", "FRAGMENT"}
ENDF_OPCODES = {"ENDF", "◇", "END_FRAGMENT"}
NON_COMPUTATIONAL_OPCODES = META_OPCODES | FRAG_OPCODES | ENDF_OPCODES

PASSTHROUGH_OPCODES = {"VOTE", "PROJ", "DIST"}

ARITHMETIC_MUL = {"∗", "MUL", "×"}
ARITHMETIC_ADD = {"+", "ADD", "⊕", "−", "SUB", "⊖"}
ARITHMETIC_ALL = ARITHMETIC_MUL | ARITHMETIC_ADD

TYPE_COMPAT = {
    ("currency", "ratio", "mul"): "ok",
    ("ratio", "currency", "mul"): "ok",
    ("currency", "currency", "add"): "ok",
    ("currency", "currency", "mul"): "warn",
    ("category", "currency", "add"): "warn",
    ("currency", "category", "add"): "warn",
}


# ── Data classes ────────────────────────────────────────────────────────


@dataclass
class SemanticError:
    line: int
    error_type: str
    message: str


@dataclass
class SemanticWarning:
    line: int
    warning_type: str
    message: str


@dataclass
class BracketInfo:
    threshold: float
    rate: float
    cumulative_tax: float
    line: int


@dataclass
class SemanticReport:
    valid: bool
    errors: list[SemanticError] = field(default_factory=list)
    warnings: list[SemanticWarning] = field(default_factory=list)
    bracket_structure: dict = field(default_factory=dict)  # {filing_status: [BracketInfo]}
    rate_schedule: dict = field(default_factory=dict)  # {filing_status: [rate_floats]}
    standard_deductions: dict = field(default_factory=dict)  # {filing_status: deduction_amount}
    instruction_count: int = 0
    register_types: dict[str, str] = field(default_factory=dict)
    fragments: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "errors": [{"line": e.line, "type": e.error_type, "message": e.message} for e in self.errors],
            "warnings": [{"line": w.line, "type": w.warning_type, "message": w.message} for w in self.warnings],
            "bracket_structure": {
                k: [{"threshold": b.threshold, "rate": b.rate, "cumulative_tax": b.cumulative_tax, "line": b.line} for b in v]
                for k, v in self.bracket_structure.items()
            },
            "rate_schedule": self.rate_schedule,
            "standard_deductions": self.standard_deductions,
            "instruction_count": self.instruction_count,
            "register_types": self.register_types,
            "fragments": self.fragments,
        }


@dataclass
class _Instruction:
    line: int
    opcode: str
    operands: list[str]


# ── Parsing helpers ─────────────────────────────────────────────────────


def _parse_program(nml_program: str) -> list[_Instruction]:
    instructions = []
    for line_no, raw in enumerate(nml_program.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped:
            continue
        tokens = stripped.split()
        if tokens:
            instructions.append(_Instruction(line_no, tokens[0], tokens[1:]))
    return instructions


def _parse_immediate(token: str) -> Optional[float]:
    """Extract numeric value from #123.45.  Returns None for array refs like #[1]."""
    if not token.startswith("#"):
        return None
    inner = token[1:]
    if inner.startswith("[") and inner.endswith("]"):
        return None
    try:
        return float(inner)
    except ValueError:
        return None


def _extract_type_annotations(instructions: list[_Instruction]) -> dict[str, str]:
    """Strip :type suffixes from register operands and return a register->type mapping."""
    register_types: dict[str, str] = {}
    for inst in instructions:
        if inst.opcode in NON_COMPUTATIONAL_OPCODES:
            continue
        cleaned = []
        for op in inst.operands:
            if ":" in op and not op.startswith("#") and not op.startswith("@"):
                reg, typ = op.rsplit(":", 1)
                register_types[reg] = typ
                cleaned.append(reg)
            else:
                cleaned.append(op)
        inst.operands = cleaned
    return register_types


def _extract_fragments(instructions: list[_Instruction]) -> dict:
    """Extract FRAG/ENDF blocks with their META @input/@output declarations."""
    fragments: dict = {}
    i = 0
    while i < len(instructions):
        inst = instructions[i]
        if inst.opcode in FRAG_OPCODES and inst.operands:
            frag_name = inst.operands[0]
            inputs: list[str] = []
            outputs: list[str] = []
            j = i + 1
            while j < len(instructions):
                inner = instructions[j]
                if inner.opcode in ENDF_OPCODES:
                    break
                if inner.opcode in META_OPCODES:
                    ops = inner.operands
                    for k, op in enumerate(ops):
                        if op == "@input" and k + 1 < len(ops):
                            inputs.append(ops[k + 1])
                        elif op.startswith("@input:"):
                            inputs.append(op[len("@input:"):])
                        elif op == "@output" and k + 1 < len(ops):
                            outputs.append(ops[k + 1])
                        elif op.startswith("@output:"):
                            outputs.append(op[len("@output:"):])
                j += 1
            fragments[frag_name] = {
                "inputs": inputs,
                "outputs": outputs,
                "start_line": inst.line,
            }
            i = j + 1
        else:
            i += 1
    return fragments


# ── Structure identification ────────────────────────────────────────────


def _identify_gates(instructions: list[_Instruction]) -> list[dict]:
    """Find ⋈ φ κ comparisons followed by ↘ — filing-status branch points."""
    gates = []
    for i, inst in enumerate(instructions):
        if inst.opcode != "⋈" or len(inst.operands) < 4:
            continue
        if inst.operands[0] != "φ" or inst.operands[1] != "κ":
            continue
        status_val = _parse_immediate(inst.operands[3])
        if status_val is None or i + 1 >= len(instructions):
            continue
        nxt = instructions[i + 1]
        if nxt.opcode != "↘" or not nxt.operands:
            continue
        jump_size = _parse_immediate(nxt.operands[0])
        if jump_size is None:
            continue
        gates.append({
            "line": inst.line,
            "index": i,
            "status_value": status_val,
            "status_name": FILING_STATUS_MAP.get(status_val, f"status_{status_val}"),
            "jump_size": int(jump_size),
            "branch_start": i + 2,
        })
    return gates


def _segment_branches(instructions: list[_Instruction], gates: list[dict]) -> list[tuple]:
    """Segment program into filing-status bracket branches.

    Returns [(status_name, start_idx, end_idx_exclusive, gate_or_None), ...]
    """
    bracket_gates = sorted(
        [g for g in gates if g["jump_size"] >= BRACKET_GATE_MIN_JUMP],
        key=lambda g: g["index"],
    )
    if not bracket_gates:
        return []

    branches = []
    for i, gate in enumerate(bracket_gates):
        start = gate["branch_start"]
        if i + 1 < len(bracket_gates):
            end = bracket_gates[i + 1]["index"]
        else:
            end = gate["branch_start"] + gate["jump_size"]
        branches.append((gate["status_name"], start, min(end, len(instructions)), gate))

    last = bracket_gates[-1]
    default_start = last["branch_start"] + last["jump_size"]
    default_end = len(instructions)
    for idx in range(default_start, len(instructions)):
        if instructions[idx].opcode in ("↑", "◼"):
            default_end = idx
            break
    if default_start < default_end:
        branches.append(("default", default_start, default_end, None))

    return branches


# ── Extraction helpers ──────────────────────────────────────────────────


def _extract_comparison_thresholds(instructions: list[_Instruction], start: int, end: int) -> list[dict]:
    """Extract ⋈ φ π #0 #THRESHOLD comparisons (the bracket cascade)."""
    results = []
    for i in range(start, min(end, len(instructions))):
        inst = instructions[i]
        if inst.opcode == "⋈" and len(inst.operands) >= 4:
            if inst.operands[0] == "φ" and inst.operands[1] == "π":
                val = _parse_immediate(inst.operands[3])
                if val is not None:
                    results.append({"value": val, "line": inst.line})
    return results


def _extract_computation_blocks(instructions: list[_Instruction], start: int, end: int) -> list[BracketInfo]:
    """Extract bracket computation blocks: ∎ α #cumulative / ∎ γ #threshold / ∗ _ _ #rate."""
    brackets = []
    bound = min(end, len(instructions))
    i = start
    while i < bound - 1:
        inst = instructions[i]
        if inst.opcode == "∎" and len(inst.operands) >= 2 and inst.operands[0] == "α":
            cumulative = _parse_immediate(inst.operands[1])
            if cumulative is not None and i + 1 < bound:
                nxt = instructions[i + 1]
                if nxt.opcode == "∎" and len(nxt.operands) >= 2 and nxt.operands[0] == "γ":
                    threshold = _parse_immediate(nxt.operands[1])
                    if threshold is not None:
                        rate = 0.0
                        for j in range(i + 2, min(i + 5, bound)):
                            if instructions[j].opcode == "∗" and len(instructions[j].operands) >= 3:
                                r = _parse_immediate(instructions[j].operands[2])
                                if r is not None:
                                    rate = r
                                    break
                        brackets.append(BracketInfo(
                            threshold=threshold, rate=rate,
                            cumulative_tax=cumulative, line=inst.line,
                        ))
        i += 1
    return brackets


def _extract_rates(instructions: list[_Instruction], start: int, end: int) -> list[float]:
    """Collect rate immediates from all ∗ instructions in a range."""
    rates = []
    for i in range(start, min(end, len(instructions))):
        inst = instructions[i]
        if inst.opcode == "∗" and len(inst.operands) >= 3:
            val = _parse_immediate(inst.operands[2])
            if val is not None:
                rates.append(val)
    return rates


def _extract_deductions(instructions: list[_Instruction], gates: list[dict]) -> dict:
    """Extract standard deduction values (∎ γ #amount) from the deduction-selection
    region, which uses small-jump filing-status gates."""
    deduction_gates = [g for g in gates if g["jump_size"] < BRACKET_GATE_MIN_JUMP]
    bracket_gates = [g for g in gates if g["jump_size"] >= BRACKET_GATE_MIN_JUMP]
    if not deduction_gates:
        return {}

    region_start = deduction_gates[0]["index"]
    region_end = bracket_gates[0]["index"] if bracket_gates else min(region_start + 30, len(instructions))

    deductions: dict[str, float] = {}
    current_status: Optional[str] = None
    for i in range(region_start, region_end):
        inst = instructions[i]
        if inst.opcode == "⋈" and len(inst.operands) >= 4 and inst.operands[1] == "κ":
            sv = _parse_immediate(inst.operands[3])
            if sv is not None:
                current_status = FILING_STATUS_MAP.get(sv, f"status_{sv}")
        elif inst.opcode == "→":
            current_status = None
        elif inst.opcode == "∎" and len(inst.operands) >= 2 and inst.operands[0] == "γ":
            val = _parse_immediate(inst.operands[1])
            if val is not None and val > 0:
                if current_status and current_status not in deductions:
                    deductions[current_status] = val
                elif current_status is None and "default" not in deductions:
                    deductions["default"] = val
    return deductions


# ── Semantic checks ─────────────────────────────────────────────────────


def _check_bracket_monotonicity(report: SemanticReport, instructions: list[_Instruction], gates: list[dict]):
    """Verify bracket thresholds are monotonically decreasing within each
    filing-status comparison cascade, and build bracket_structure / rate_schedule."""
    for status_name, start, end, _gate in _segment_branches(instructions, gates):
        thresholds = _extract_comparison_thresholds(instructions, start, end)
        if thresholds:
            values = [t["value"] for t in thresholds]
            for j in range(1, len(values)):
                if values[j] >= values[j - 1]:
                    report.errors.append(SemanticError(
                        line=thresholds[j]["line"],
                        error_type="bracket_monotonicity",
                        message=(
                            f"Bracket threshold {values[j]:g} >= previous {values[j-1]:g} "
                            f"in {status_name} branch (expected descending order)"
                        ),
                    ))

        blocks = _extract_computation_blocks(instructions, start, end)
        if blocks:
            report.bracket_structure[status_name] = blocks

        rates = _extract_rates(instructions, start, end)
        if rates:
            report.rate_schedule[status_name] = rates


def _check_rate_bounds(report: SemanticReport, instructions: list[_Instruction]):
    """Warn if any ∗ rate immediate is outside [0.0, 1.0]."""
    for inst in instructions:
        if inst.opcode != "∗" or len(inst.operands) < 3:
            continue
        val = _parse_immediate(inst.operands[2])
        if val is not None and (val < 0.0 or val > 1.0):
            report.warnings.append(SemanticWarning(
                line=inst.line,
                warning_type="rate_bounds",
                message=f"Rate {val:g} is outside [0.0, 1.0]",
            ))


def _check_filing_status_coverage(report: SemanticReport, gates: list[dict], tax_type: str):
    """For income-tax programs, verify Single and MFJ branches exist."""
    if tax_type.upper() not in INCOME_TAX_TYPES:
        return
    found = {g["status_value"] for g in gates if g["jump_size"] >= BRACKET_GATE_MIN_JUMP}
    if 1.5 not in found:
        report.warnings.append(SemanticWarning(
            line=0, warning_type="filing_status_coverage",
            message="No Single filing status branch (⋈ φ κ #0 #1.5) found",
        ))
    if 2.5 not in found:
        report.warnings.append(SemanticWarning(
            line=0, warning_type="filing_status_coverage",
            message="No MFJ filing status branch (⋈ φ κ #0 #2.5) found",
        ))


def _check_input_output(report: SemanticReport, instructions: list[_Instruction]):
    """Check that all inputs are referenced and at least one output exists."""
    inputs: dict[str, tuple[str, int]] = {}
    output_count = 0
    used_registers: set[str] = set()

    for inst in instructions:
        if inst.opcode in NON_COMPUTATIONAL_OPCODES:
            continue
        if inst.opcode == "↓" and len(inst.operands) >= 2:
            inputs[inst.operands[0]] = (inst.operands[1], inst.line)
        elif inst.opcode == "↑":
            output_count += 1
        else:
            for op in inst.operands:
                if not op.startswith("#") and not op.startswith("@"):
                    used_registers.add(op)

    if output_count == 0:
        report.errors.append(SemanticError(
            line=0, error_type="missing_output",
            message="Program has no output (↑) instruction",
        ))

    for reg, (binding, line) in inputs.items():
        if reg not in used_registers:
            report.warnings.append(SemanticWarning(
                line=line, warning_type="unused_input",
                message=f"Input {binding} loaded into {reg} but never referenced",
            ))


def _check_exempt_zero(report: SemanticReport, instructions: list[_Instruction]):
    """If @is_exempt is loaded, verify there is a branch that skips computation
    (jumps near the program end, producing a zero result)."""
    exempt_reg = None
    for inst in instructions:
        if inst.opcode == "↓" and len(inst.operands) >= 2 and "@is_exempt" in inst.operands[1]:
            exempt_reg = inst.operands[0]
            break
    if exempt_reg is None:
        return

    n = len(instructions)
    for i, inst in enumerate(instructions):
        if inst.opcode != "⋈" or len(inst.operands) < 4:
            continue
        if inst.operands[1] != exempt_reg:
            continue
        if i + 1 < n and instructions[i + 1].opcode == "↘":
            jump_val = _parse_immediate(instructions[i + 1].operands[0])
            if jump_val is not None:
                target = (i + 1) + int(jump_val)
                if target >= n - 5:
                    return

    report.warnings.append(SemanticWarning(
        line=0, warning_type="exempt_handling",
        message="@is_exempt is loaded but no branch to skip computation was detected",
    ))


def _check_type_compatibility(report: SemanticReport, instructions: list[_Instruction], register_types: dict[str, str]):
    """Flag obviously wrong type combinations in arithmetic instructions."""
    for inst in instructions:
        if inst.opcode in ARITHMETIC_MUL:
            op_kind = "mul"
        elif inst.opcode in ARITHMETIC_ADD:
            op_kind = "add"
        else:
            continue

        if len(inst.operands) < 3:
            continue
        src0, src1 = inst.operands[1], inst.operands[2]
        if src0.startswith("#") or src0.startswith("@") or src1.startswith("#") or src1.startswith("@"):
            continue

        t0 = register_types.get(src0)
        t1 = register_types.get(src1)
        if t0 is None or t1 is None:
            continue

        key = (t0, t1, op_kind)
        result = TYPE_COMPAT.get(key)
        if result == "warn":
            report.warnings.append(SemanticWarning(
                line=inst.line,
                warning_type="type_compatibility",
                message=f"{t0} {inst.opcode} {t1} is suspicious ({src0} {inst.opcode} {src1})",
            ))
        elif result is None:
            report.warnings.append(SemanticWarning(
                line=inst.line,
                warning_type="type_compatibility",
                message=f"Unknown type combination: {t0} {inst.opcode} {t1} ({src0} {inst.opcode} {src1})",
            ))


# ── Public API ──────────────────────────────────────────────────────────


def validate_semantics(nml_program: str, tax_type: str = "", jurisdiction_key: str = "") -> SemanticReport:
    """Run domain-specific semantic validation on a symbolic NML tax program.

    Args:
        nml_program: The symbolic NML source code.
        tax_type: Tax type hint (e.g. "FIT", "SIT", "EIC").
        jurisdiction_key: Optional key like "00-000-0000-FIT-000".

    Returns:
        SemanticReport with validation results and extracted tax structure.
    """
    if not tax_type and jurisdiction_key:
        parts = jurisdiction_key.split("-")
        if len(parts) >= 4:
            tax_type = parts[3]

    instructions = _parse_program(nml_program)
    register_types = _extract_type_annotations(instructions)
    report = SemanticReport(valid=True, instruction_count=len(instructions))
    report.register_types = register_types
    report.fragments = _extract_fragments(instructions)

    if not instructions:
        report.errors.append(SemanticError(line=0, error_type="empty_program", message="Program is empty"))
        report.valid = False
        return report

    gates = _identify_gates(instructions)

    _check_bracket_monotonicity(report, instructions, gates)
    _check_rate_bounds(report, instructions)
    _check_filing_status_coverage(report, gates, tax_type)
    _check_input_output(report, instructions)
    _check_exempt_zero(report, instructions)
    _check_type_compatibility(report, instructions, register_types)

    report.standard_deductions = _extract_deductions(instructions, gates)

    if report.errors:
        report.valid = False

    return report


def validate_directory(directory: str, tax_type: str = "") -> dict:
    """Validate all .nml files in a directory tree.

    Returns dict mapping file paths to SemanticReport.to_dict() results.
    """
    results = {}
    for nml_path in sorted(Path(directory).rglob("*.nml")):
        source = nml_path.read_text(encoding="utf-8")
        key = nml_path.stem
        inferred = tax_type
        if not inferred:
            parts = key.split("-")
            if len(parts) >= 4:
                inferred = parts[3]
        report = validate_semantics(source, tax_type=inferred, jurisdiction_key=key)
        results[str(nml_path)] = report.to_dict()
    return results


# ── CLI ─────────────────────────────────────────────────────────────────


def _print_report(report: SemanticReport):
    tag = "VALID" if report.valid else "INVALID"
    print(f"Semantic analysis: {tag}")
    print(f"Instructions: {report.instruction_count}")

    if report.standard_deductions:
        print("\nStandard deductions:")
        for name, amount in report.standard_deductions.items():
            print(f"  {name}: ${amount:,.2f}")

    if report.bracket_structure:
        print("\nBracket structure:")
        for name, brackets in report.bracket_structure.items():
            print(f"  {name}:")
            for b in brackets:
                print(f"    ${b.threshold:>14,.2f}  rate={b.rate:.4f}  cumulative=${b.cumulative_tax:>12,.2f}")

    if report.errors:
        print(f"\nErrors ({len(report.errors)}):")
        for e in report.errors:
            print(f"  line {e.line}: [{e.error_type}] {e.message}")

    if report.warnings:
        print(f"\nWarnings ({len(report.warnings)}):")
        for w in report.warnings:
            print(f"  line {w.line}: [{w.warning_type}] {w.message}")


def _print_batch(results: dict):
    total = len(results)
    errors = sum(1 for r in results.values() if not r["valid"])
    warnings = sum(len(r["warnings"]) for r in results.values())
    print(f"Validated {total} programs: {errors} with errors, {warnings} total warnings")
    for path, r in results.items():
        status = "FAIL" if not r["valid"] else ("WARN" if r["warnings"] else " OK ")
        name = Path(path).stem
        detail = f"{r['instruction_count']} instructions"
        if r["errors"]:
            detail += f", {len(r['errors'])} error(s)"
        if r["warnings"]:
            detail += f", {len(r['warnings'])} warning(s)"
        print(f"  [{status}] {name} — {detail}")
        for e in r["errors"]:
            print(f"         ERROR line {e['line']}: {e['message']}")
        for w in r["warnings"]:
            print(f"         WARN  line {w['line']}: {w['message']}")


def main():
    parser = argparse.ArgumentParser(description="Semantic analyzer for symbolic NML tax programs")
    parser.add_argument("path", help="Path to .nml file or directory")
    parser.add_argument("--tax-type", default="", help="Tax type (FIT, SIT, EIC, ...)")
    parser.add_argument("--key", default="", help="Jurisdiction key (e.g. 00-000-0000-FIT-000)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--batch", action="store_true", help="Validate all .nml in directory")
    args = parser.parse_args()

    target = Path(args.path)

    if args.batch or target.is_dir():
        results = validate_directory(str(target), tax_type=args.tax_type)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            _print_batch(results)
        return

    if not target.exists():
        print(f"Error: {target} not found", file=sys.stderr)
        sys.exit(1)

    source = target.read_text(encoding="utf-8")
    report = validate_semantics(source, tax_type=args.tax_type, jurisdiction_key=args.key or target.stem)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        _print_report(report)


if __name__ == "__main__":
    main()
