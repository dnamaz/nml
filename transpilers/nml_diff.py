#!/usr/bin/env python3
"""
NML Semantic Diff Engine — structural comparison of NML tax programs.

Compares two NML programs (e.g., FIT rules from 2025 vs 2026) at the
tax-structure level rather than line-by-line text diff.  Reports changes
in bracket thresholds, tax rates, standard deductions, cumulative tax
breakpoints, and added/removed brackets.

Operates on SYMBOLIC syntax NML programs.
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Constants ────────────────────────────────────────────────────────────


FILING_STATUS_MAP = {
    1.5: "single",
    2.5: "mfj",
    3.5: "mfs",
    4.5: "hoh",
}

BRACKET_GATE_MIN_JUMP = 5


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class DiffEntry:
    category: str       # "threshold", "rate", "deduction", "cumulative", "bracket_added", "bracket_removed"
    filing_status: str  # "single", "mfj", "hoh", "unknown"
    old_value: float | None
    new_value: float | None
    line_old: int | None
    line_new: int | None
    description: str


@dataclass
class DiffReport:
    jurisdiction_key: str
    old_label: str
    new_label: str
    changes: list[DiffEntry] = field(default_factory=list)
    total_changes: int = 0
    thresholds_changed: int = 0
    rates_changed: int = 0
    deductions_changed: int = 0

    def to_dict(self) -> dict:
        return {
            "jurisdiction_key": self.jurisdiction_key,
            "old_label": self.old_label,
            "new_label": self.new_label,
            "total_changes": self.total_changes,
            "thresholds_changed": self.thresholds_changed,
            "rates_changed": self.rates_changed,
            "deductions_changed": self.deductions_changed,
            "changes": [
                {
                    "category": c.category,
                    "filing_status": c.filing_status,
                    "old_value": c.old_value,
                    "new_value": c.new_value,
                    "line_old": c.line_old,
                    "line_new": c.line_new,
                    "description": c.description,
                }
                for c in self.changes
            ],
        }

    def summary(self) -> str:
        lines = []
        header = f"Diff: {self.jurisdiction_key}" if self.jurisdiction_key else "Diff"
        lines.append(f"{header}  ({self.old_label} → {self.new_label})")
        lines.append(
            f"  {self.total_changes} change(s): "
            f"{self.thresholds_changed} threshold(s), "
            f"{self.rates_changed} rate(s), "
            f"{self.deductions_changed} deduction(s)"
        )
        if not self.changes:
            lines.append("  No structural changes detected.")
        for c in self.changes:
            status = f"[{c.filing_status}] " if c.filing_status != "unknown" else ""
            if c.old_value is not None and c.new_value is not None:
                if c.category == "rate":
                    lines.append(f"  {status}{c.category}: {c.old_value:.6f} → {c.new_value:.6f}")
                else:
                    lines.append(f"  {status}{c.category}: ${c.old_value:,.2f} → ${c.new_value:,.2f}")
            else:
                lines.append(f"  {status}{c.description}")
        return "\n".join(lines)


# ── Parsing helpers ──────────────────────────────────────────────────────


@dataclass
class _Instruction:
    line: int
    opcode: str
    operands: list[str]


def _parse_program(nml_text: str) -> list[_Instruction]:
    instructions: list[_Instruction] = []
    for line_no, raw in enumerate(nml_text.splitlines(), start=1):
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


# ── Gate identification ──────────────────────────────────────────────────


def _identify_gates(instructions: list[_Instruction]) -> list[dict]:
    """Find ⋈ φ κ comparisons followed by ↘ — filing-status branch points."""
    gates: list[dict] = []
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
            "status_name": FILING_STATUS_MAP.get(status_val, "unknown"),
            "jump_size": int(jump_size),
            "branch_start": i + 2,
        })
    return gates


def _segment_branches(
    instructions: list[_Instruction], gates: list[dict]
) -> list[tuple[str, int, int, dict | None]]:
    """Segment program into filing-status bracket branches.

    Returns [(status_name, start_idx, end_idx_exclusive, gate_or_None), ...]
    When no bracket gates exist, returns the entire program as a single "unknown" segment.
    """
    bracket_gates = sorted(
        [g for g in gates if g["jump_size"] >= BRACKET_GATE_MIN_JUMP],
        key=lambda g: g["index"],
    )
    if not bracket_gates:
        return [("unknown", 0, len(instructions), None)]

    branches: list[tuple[str, int, int, dict | None]] = []
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
        default_name = "hoh" if any(g["status_name"] == "mfj" for g in bracket_gates) else "default"
        branches.append((default_name, default_start, default_end, None))

    return branches


# ── Extraction helpers ───────────────────────────────────────────────────


def _extract_brackets_from_range(
    instructions: list[_Instruction], start: int, end: int
) -> list[dict]:
    """Extract bracket computation blocks from an instruction range.

    Captures two patterns:
      - Base bracket:  ∗ α π #RATE  (lowest bracket, taxed from zero)
      - Higher bracket: ∎ α #CUMULATIVE / ∎ γ #THRESHOLD / ∗ ρ ρ #RATE
    """
    brackets: list[dict] = []
    bound = min(end, len(instructions))

    for i in range(start, bound):
        inst = instructions[i]
        if inst.opcode == "∗" and len(inst.operands) >= 3:
            if inst.operands[0] == "α" and inst.operands[1] == "π":
                rate = _parse_immediate(inst.operands[2])
                if rate is not None:
                    brackets.append({
                        "threshold": 0.0,
                        "rate": rate,
                        "cumulative": 0.0,
                        "line": inst.line,
                    })
                    break

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
                        brackets.append({
                            "threshold": threshold,
                            "rate": rate,
                            "cumulative": cumulative,
                            "line": inst.line,
                        })
        i += 1

    brackets.sort(key=lambda b: b["threshold"])
    return brackets


def _extract_flat_rate(
    instructions: list[_Instruction], start: int, end: int
) -> dict | None:
    """Extract flat rate from ∗ γ ι #RATE pattern (non-bracket programs)."""
    for i in range(start, min(end, len(instructions))):
        inst = instructions[i]
        if inst.opcode == "∗" and len(inst.operands) >= 3:
            if inst.operands[0] == "γ" and inst.operands[1] == "ι":
                rate = _parse_immediate(inst.operands[2])
                if rate is not None:
                    return {"value": rate, "line": inst.line}
    return None


def _extract_deductions(
    instructions: list[_Instruction], gates: list[dict]
) -> dict[str, dict]:
    """Extract standard deductions (∎ γ #amount) organized by filing status.

    These appear in the deduction-selection region, identified by small-jump
    filing-status gates that precede the bracket gates.
    """
    deduction_gates = [g for g in gates if g["jump_size"] < BRACKET_GATE_MIN_JUMP]
    bracket_gates = [g for g in gates if g["jump_size"] >= BRACKET_GATE_MIN_JUMP]
    if not deduction_gates:
        return {}

    region_start = deduction_gates[0]["index"]
    region_end = (
        bracket_gates[0]["index"]
        if bracket_gates
        else min(region_start + 30, len(instructions))
    )

    deductions: dict[str, dict] = {}
    current_status: Optional[str] = None
    for i in range(region_start, region_end):
        inst = instructions[i]
        if inst.opcode == "⋈" and len(inst.operands) >= 4 and inst.operands[1] == "κ":
            sv = _parse_immediate(inst.operands[3])
            if sv is not None:
                current_status = FILING_STATUS_MAP.get(sv, "unknown")
        elif inst.opcode == "→":
            current_status = None
        elif inst.opcode == "∎" and len(inst.operands) >= 2 and inst.operands[0] == "γ":
            val = _parse_immediate(inst.operands[1])
            if val is not None and val > 0:
                key = current_status or "default"
                if key not in deductions:
                    deductions[key] = {"value": val, "line": inst.line}
    return deductions


# ── Public structure extraction ──────────────────────────────────────────


def extract_structure(nml_program: str) -> dict:
    """Extract tax bracket structure from a symbolic NML program.

    Returns::

        {
            "filing_statuses": {
                "<status>": {
                    "deduction": {"value": float, "line": int},   # optional
                    "brackets": [{"threshold", "rate", "cumulative", "line"}, ...],
                    "flat_rate": {"value": float, "line": int},   # optional
                },
                ...
            },
            "instruction_count": int,
        }
    """
    instructions = _parse_program(nml_program)
    if not instructions:
        return {"filing_statuses": {}, "instruction_count": 0}

    gates = _identify_gates(instructions)
    deductions = _extract_deductions(instructions, gates)
    branches = _segment_branches(instructions, gates)

    filing_statuses: dict[str, dict] = {}
    for status_name, start, end, _gate in branches:
        brackets = _extract_brackets_from_range(instructions, start, end)
        entry: dict = {"brackets": brackets}
        if not brackets:
            flat = _extract_flat_rate(instructions, start, end)
            if flat:
                entry["flat_rate"] = flat
        if status_name in deductions:
            entry["deduction"] = deductions[status_name]
        elif "default" in deductions and status_name in ("unknown", "default"):
            entry["deduction"] = deductions["default"]
        filing_statuses[status_name] = entry

    for status, ded in deductions.items():
        if status not in filing_statuses and status != "default":
            filing_statuses[status] = {"brackets": [], "deduction": ded}

    return {
        "filing_statuses": filing_statuses,
        "instruction_count": len(instructions),
    }


# ── Diff logic ───────────────────────────────────────────────────────────


def _compare_brackets(
    old_brackets: list[dict], new_brackets: list[dict], filing_status: str
) -> list[DiffEntry]:
    """Compare bracket lists pairwise between old and new programs."""
    changes: list[DiffEntry] = []
    max_len = max(len(old_brackets), len(new_brackets))

    for i in range(max_len):
        old_b = old_brackets[i] if i < len(old_brackets) else None
        new_b = new_brackets[i] if i < len(new_brackets) else None

        if old_b is None and new_b is not None:
            changes.append(DiffEntry(
                category="bracket_added",
                filing_status=filing_status,
                old_value=None,
                new_value=new_b["threshold"],
                line_old=None,
                line_new=new_b["line"],
                description=f"Bracket added: ${new_b['threshold']:,.2f} @ {new_b['rate']:.6f}",
            ))
            continue

        if old_b is not None and new_b is None:
            changes.append(DiffEntry(
                category="bracket_removed",
                filing_status=filing_status,
                old_value=old_b["threshold"],
                new_value=None,
                line_old=old_b["line"],
                line_new=None,
                description=f"Bracket removed: ${old_b['threshold']:,.2f} @ {old_b['rate']:.6f}",
            ))
            continue

        assert old_b is not None and new_b is not None
        if abs(old_b["threshold"] - new_b["threshold"]) > 0.005:
            changes.append(DiffEntry(
                category="threshold",
                filing_status=filing_status,
                old_value=old_b["threshold"],
                new_value=new_b["threshold"],
                line_old=old_b["line"],
                line_new=new_b["line"],
                description=(
                    f"Bracket {i + 1} threshold: "
                    f"${old_b['threshold']:,.2f} → ${new_b['threshold']:,.2f}"
                ),
            ))

        if abs(old_b["rate"] - new_b["rate"]) > 1e-8:
            changes.append(DiffEntry(
                category="rate",
                filing_status=filing_status,
                old_value=old_b["rate"],
                new_value=new_b["rate"],
                line_old=old_b["line"],
                line_new=new_b["line"],
                description=(
                    f"Bracket {i + 1} rate: "
                    f"{old_b['rate']:.6f} → {new_b['rate']:.6f}"
                ),
            ))

        if abs(old_b["cumulative"] - new_b["cumulative"]) > 0.005:
            changes.append(DiffEntry(
                category="cumulative",
                filing_status=filing_status,
                old_value=old_b["cumulative"],
                new_value=new_b["cumulative"],
                line_old=old_b["line"],
                line_new=new_b["line"],
                description=(
                    f"Bracket {i + 1} cumulative tax: "
                    f"${old_b['cumulative']:,.2f} → ${new_b['cumulative']:,.2f}"
                ),
            ))

    return changes


def _compare_flat_rates(
    old_fs: dict, new_fs: dict, filing_status: str
) -> list[DiffEntry]:
    """Compare flat rates between old and new filing-status sections."""
    changes: list[DiffEntry] = []
    old_flat = old_fs.get("flat_rate")
    new_flat = new_fs.get("flat_rate")
    if old_flat and new_flat:
        if abs(old_flat["value"] - new_flat["value"]) > 1e-8:
            changes.append(DiffEntry(
                category="rate",
                filing_status=filing_status,
                old_value=old_flat["value"],
                new_value=new_flat["value"],
                line_old=old_flat["line"],
                line_new=new_flat["line"],
                description=f"Flat rate: {old_flat['value']:.6f} → {new_flat['value']:.6f}",
            ))
    elif old_flat and not new_flat:
        changes.append(DiffEntry(
            category="rate",
            filing_status=filing_status,
            old_value=old_flat["value"],
            new_value=None,
            line_old=old_flat["line"],
            line_new=None,
            description=f"Flat rate removed: {old_flat['value']:.6f}",
        ))
    elif new_flat and not old_flat:
        changes.append(DiffEntry(
            category="rate",
            filing_status=filing_status,
            old_value=None,
            new_value=new_flat["value"],
            line_old=None,
            line_new=new_flat["line"],
            description=f"Flat rate added: {new_flat['value']:.6f}",
        ))
    return changes


def _compare_deductions(
    old_fs: dict, new_fs: dict, filing_status: str
) -> list[DiffEntry]:
    """Compare standard deductions between old and new filing-status sections."""
    changes: list[DiffEntry] = []
    old_ded = old_fs.get("deduction")
    new_ded = new_fs.get("deduction")
    if old_ded and new_ded:
        if abs(old_ded["value"] - new_ded["value"]) > 0.005:
            changes.append(DiffEntry(
                category="deduction",
                filing_status=filing_status,
                old_value=old_ded["value"],
                new_value=new_ded["value"],
                line_old=old_ded["line"],
                line_new=new_ded["line"],
                description=(
                    f"Standard deduction: "
                    f"${old_ded['value']:,.2f} → ${new_ded['value']:,.2f}"
                ),
            ))
    elif old_ded and not new_ded:
        changes.append(DiffEntry(
            category="deduction",
            filing_status=filing_status,
            old_value=old_ded["value"],
            new_value=None,
            line_old=old_ded["line"],
            line_new=None,
            description=f"Standard deduction removed: ${old_ded['value']:,.2f}",
        ))
    elif new_ded and not old_ded:
        changes.append(DiffEntry(
            category="deduction",
            filing_status=filing_status,
            old_value=None,
            new_value=new_ded["value"],
            line_old=None,
            line_new=new_ded["line"],
            description=f"Standard deduction added: ${new_ded['value']:,.2f}",
        ))
    return changes


# ── Public diff API ──────────────────────────────────────────────────────


def diff_nml(
    old_program: str,
    new_program: str,
    jurisdiction_key: str = "",
    old_label: str = "old",
    new_label: str = "new",
) -> DiffReport:
    """Compute structural diff between two NML programs.

    Args:
        old_program: Source text of the baseline NML program.
        new_program: Source text of the updated NML program.
        jurisdiction_key: Optional identifier (e.g. "00-000-0000-FIT-000").
        old_label: Human label for old version (e.g. "2025").
        new_label: Human label for new version (e.g. "2026").

    Returns:
        DiffReport with all structural changes.
    """
    old_struct = extract_structure(old_program)
    new_struct = extract_structure(new_program)

    changes: list[DiffEntry] = []
    all_statuses = sorted(
        set(old_struct["filing_statuses"]) | set(new_struct["filing_statuses"])
    )

    for status in all_statuses:
        old_fs = old_struct["filing_statuses"].get(status, {})
        new_fs = new_struct["filing_statuses"].get(status, {})

        changes.extend(_compare_deductions(old_fs, new_fs, status))
        changes.extend(_compare_brackets(
            old_fs.get("brackets", []),
            new_fs.get("brackets", []),
            status,
        ))
        changes.extend(_compare_flat_rates(old_fs, new_fs, status))

    thresholds_changed = sum(1 for c in changes if c.category == "threshold")
    rates_changed = sum(1 for c in changes if c.category == "rate")
    deductions_changed = sum(1 for c in changes if c.category == "deduction")

    return DiffReport(
        jurisdiction_key=jurisdiction_key,
        old_label=old_label,
        new_label=new_label,
        changes=changes,
        total_changes=len(changes),
        thresholds_changed=thresholds_changed,
        rates_changed=rates_changed,
        deductions_changed=deductions_changed,
    )


# ── CLI helpers ──────────────────────────────────────────────────────────


def _diff_files(
    path_old: str, path_new: str, key: str = "",
    old_label: str = "old", new_label: str = "new",
    as_json: bool = False,
):
    old_text = Path(path_old).read_text(encoding="utf-8")
    new_text = Path(path_new).read_text(encoding="utf-8")
    key = key or Path(path_old).stem
    report = diff_nml(
        old_text, new_text,
        jurisdiction_key=key, old_label=old_label, new_label=new_label,
    )
    if as_json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.summary())


def _diff_directories(
    dir1: str, dir2: str, tax_type: str = "", as_json: bool = False,
):
    """Diff matching NML files across two library directories."""
    p1, p2 = Path(dir1), Path(dir2)
    files1 = {f.name: f for f in sorted(p1.rglob("*.nml"))}
    files2 = {f.name: f for f in sorted(p2.rglob("*.nml"))}

    if tax_type:
        files1 = {k: v for k, v in files1.items() if f"-{tax_type}-" in k}
        files2 = {k: v for k, v in files2.items() if f"-{tax_type}-" in k}

    common = sorted(set(files1) & set(files2))
    only1 = sorted(set(files1) - set(files2))
    only2 = sorted(set(files2) - set(files1))

    results: dict[str, DiffReport] = {}
    changed = 0
    for name in common:
        old_text = files1[name].read_text(encoding="utf-8")
        new_text = files2[name].read_text(encoding="utf-8")
        report = diff_nml(
            old_text, new_text,
            jurisdiction_key=Path(name).stem,
            old_label=str(p1), new_label=str(p2),
        )
        if report.total_changes > 0:
            changed += 1
        results[name] = report

    if as_json:
        out = {
            "compared": len(common),
            "changed": changed,
            "only_in_old": only1,
            "only_in_new": only2,
            "diffs": {
                k: v.to_dict()
                for k, v in results.items()
                if v.total_changes > 0
            },
        }
        print(json.dumps(out, indent=2))
    else:
        print(f"Compared {len(common)} programs: {changed} changed")
        if only1:
            print(f"Only in {dir1}: {len(only1)} programs")
            for name in only1:
                print(f"  - {name}")
        if only2:
            print(f"Only in {dir2}: {len(only2)} programs")
            for name in only2:
                print(f"  + {name}")
        print()
        for name, report in results.items():
            if report.total_changes > 0:
                print(report.summary())
                print()


# ── CLI ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Semantic diff engine for symbolic NML tax programs",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--old", help="Path to old .nml file")
    group.add_argument("--dir1", help="Path to old library directory")
    parser.add_argument("--new", help="Path to new .nml file")
    parser.add_argument("--dir2", help="Path to new library directory")
    parser.add_argument("--key", default="", help="Jurisdiction key")
    parser.add_argument("--type", default="", help="Filter by tax type (e.g. FIT)")
    parser.add_argument("--old-label", default="old", help="Label for old version")
    parser.add_argument("--new-label", default="new", help="Label for new version")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if args.old:
        if not args.new:
            parser.error("--new is required when using --old")
        _diff_files(
            args.old, args.new, key=args.key,
            old_label=args.old_label, new_label=args.new_label,
            as_json=args.json,
        )
    else:
        if not args.dir2:
            parser.error("--dir2 is required when using --dir1")
        _diff_directories(args.dir1, args.dir2, tax_type=args.type, as_json=args.json)


if __name__ == "__main__":
    main()
