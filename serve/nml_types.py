"""NML v0.6 – Semantic tensor type system.

Provides type annotations, compatibility checking, and type inference
for NML instructions operating on typed tensors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


class SemanticType(IntEnum):
    FLOAT = 0
    CURRENCY = 1
    RATIO = 2
    CATEGORY = 3
    COUNT = 4
    BOOL = 5
    EMBEDDING = 6
    PROBABILITY = 7


TYPE_NAMES: dict[str, SemanticType] = {t.name.lower(): t for t in SemanticType}

# (left_type, right_type, operation) -> result_type
COMPATIBILITY: dict[tuple[SemanticType, SemanticType, str], SemanticType] = {
    (SemanticType.CURRENCY, SemanticType.RATIO, "multiply"): SemanticType.CURRENCY,
    (SemanticType.RATIO, SemanticType.CURRENCY, "multiply"): SemanticType.CURRENCY,
    (SemanticType.CURRENCY, SemanticType.CURRENCY, "add"): SemanticType.CURRENCY,
    (SemanticType.CURRENCY, SemanticType.CURRENCY, "subtract"): SemanticType.CURRENCY,
    (SemanticType.CURRENCY, SemanticType.COUNT, "divide"): SemanticType.CURRENCY,
    (SemanticType.RATIO, SemanticType.RATIO, "add"): SemanticType.RATIO,
    (SemanticType.RATIO, SemanticType.RATIO, "subtract"): SemanticType.RATIO,
    (SemanticType.COUNT, SemanticType.COUNT, "add"): SemanticType.COUNT,
    (SemanticType.COUNT, SemanticType.COUNT, "subtract"): SemanticType.COUNT,
    (SemanticType.COUNT, SemanticType.COUNT, "multiply"): SemanticType.COUNT,
}

FORCED_TYPES: dict[str, SemanticType] = {
    "CMPF": SemanticType.BOOL,
    "CMP": SemanticType.BOOL,
    "CMPI": SemanticType.BOOL,
    "SOFT": SemanticType.PROBABILITY,
    "PROJ": SemanticType.EMBEDDING,
}

OPCODE_TO_OP: dict[str, str] = {
    "SCLR": "multiply", "∗": "multiply", "EMUL": "multiply", "⊗": "multiply",
    "MADD": "add", "⊕": "add", "∑": "add", "TACC": "add",
    "MSUB": "subtract", "⊖": "subtract",
    "SDIV": "divide", "÷": "divide", "EDIV": "divide", "⊘": "divide",
}

# FLOAT is compatible with anything via implicit promotion
_FLOAT_PROMOTABLE = frozenset(SemanticType)


def parse_type_annotation(token: str) -> tuple[str, Optional[SemanticType]]:
    """Split a typed token like ``ι:currency`` into (name, SemanticType).

    Returns (token, None) when no annotation is present.
    """
    if ":" not in token:
        return token, None
    name, type_str = token.split(":", 1)
    type_str = type_str.strip().lower()
    sem = TYPE_NAMES.get(type_str)
    if sem is None:
        raise ValueError(f"Unknown semantic type '{type_str}' in token '{token}'")
    return name, sem


def check_compatibility(
    left: SemanticType, right: SemanticType, op: str
) -> tuple[bool, SemanticType]:
    """Check whether *left op right* is type-safe.

    Returns ``(compatible, result_type)``.  When either operand is FLOAT
    the result adopts the other operand's type (implicit promotion).
    """
    if left == right == SemanticType.FLOAT:
        return True, SemanticType.FLOAT

    if left == SemanticType.FLOAT:
        return True, right
    if right == SemanticType.FLOAT:
        return True, left

    key = (left, right, op)
    if key in COMPATIBILITY:
        return True, COMPATIBILITY[key]

    return False, SemanticType.FLOAT


def infer_type(opcode: str, input_types: list[SemanticType]) -> SemanticType:
    """Infer the output semantic type for *opcode* applied to *input_types*."""
    upper = opcode.upper().strip()

    if upper in FORCED_TYPES:
        return FORCED_TYPES[upper]

    op = OPCODE_TO_OP.get(opcode) or OPCODE_TO_OP.get(upper)

    if op is None:
        if input_types:
            return input_types[0]
        return SemanticType.FLOAT

    if len(input_types) == 0:
        return SemanticType.FLOAT

    if len(input_types) == 1:
        return input_types[0]

    result = input_types[0]
    for t in input_types[1:]:
        _, result = check_compatibility(result, t, op)
    return result


# ---------------------------------------------------------------------------
# Typed tensor descriptor (useful for downstream tooling)
# ---------------------------------------------------------------------------

@dataclass
class TypedTensor:
    name: str
    semantic_type: SemanticType = SemanticType.FLOAT
    shape: list[int] = field(default_factory=list)
    unit: str = ""

    @classmethod
    def from_annotation(cls, token: str, shape: list[int] | None = None) -> TypedTensor:
        name, sem = parse_type_annotation(token)
        return cls(name=name, semantic_type=sem or SemanticType.FLOAT, shape=shape or [])

    @property
    def annotation(self) -> str:
        return f"{self.name}:{self.semantic_type.name.lower()}"


# ---------------------------------------------------------------------------
# Bulk annotation helpers
# ---------------------------------------------------------------------------

def annotate_line(line: str, type_map: dict[str, SemanticType]) -> str:
    """Rewrite an NML instruction line, adding type annotations to known names."""
    tokens = line.split()
    out: list[str] = []
    for tok in tokens:
        bare, existing = parse_type_annotation(tok) if ":" in tok else (tok, None)
        if existing is not None:
            out.append(tok)
        elif bare in type_map:
            out.append(f"{bare}:{type_map[bare].name.lower()}")
        else:
            out.append(tok)
    return "  ".join(out)


def extract_type_map(nml_program: str) -> dict[str, SemanticType]:
    """Scan an NML program for META @type declarations and inline annotations."""
    result: dict[str, SemanticType] = {}
    for line in nml_program.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("META") and "@type" in stripped.lower():
            parts = stripped.split()
            for part in parts[1:]:
                if ":" in part and not part.startswith("@"):
                    name, sem = parse_type_annotation(part)
                    if sem is not None:
                        result[name] = sem
        else:
            for tok in stripped.split():
                if ":" in tok and not tok.startswith("@") and not tok.startswith("#"):
                    try:
                        name, sem = parse_type_annotation(tok)
                        if sem is not None:
                            result[name] = sem
                    except ValueError:
                        pass
    return result


def validate_program_types(nml_program: str) -> list[str]:
    """Run type-compatibility checks across all instructions in a program.

    Returns a list of diagnostic strings (empty == all OK).
    """
    type_map = extract_type_map(nml_program)
    diagnostics: list[str] = []

    for lineno, line in enumerate(nml_program.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("META"):
            continue

        tokens = stripped.split()
        if not tokens:
            continue
        opcode = tokens[0]
        op = OPCODE_TO_OP.get(opcode) or OPCODE_TO_OP.get(opcode.upper())
        if op is None:
            continue

        operand_types: list[SemanticType] = []
        for tok in tokens[1:]:
            bare, _ = (parse_type_annotation(tok) if ":" in tok else (tok, None))
            if bare in type_map:
                operand_types.append(type_map[bare])

        if len(operand_types) >= 2:
            ok, _ = check_compatibility(operand_types[0], operand_types[1], op)
            if not ok:
                diagnostics.append(
                    f"line {lineno}: incompatible types "
                    f"{operand_types[0].name} {op} {operand_types[1].name}"
                )

    return diagnostics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="NML semantic type utilities")
    sub = parser.add_subparsers(dest="command")

    p_check = sub.add_parser("check", help="Type-check an NML program")
    p_check.add_argument("program", help="Path to .nml file")

    p_map = sub.add_parser("map", help="Extract type map from an NML program")
    p_map.add_argument("program", help="Path to .nml file")

    p_annotate = sub.add_parser("annotate", help="Add type annotations to an NML program")
    p_annotate.add_argument("program", help="Path to .nml file")
    p_annotate.add_argument("-o", "--output", help="Output file (default: stdout)")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    text = open(args.program).read()

    if args.command == "check":
        diags = validate_program_types(text)
        if diags:
            for d in diags:
                print(f"  ⚠  {d}", file=sys.stderr)
            sys.exit(1)
        else:
            print("✓ all types OK")

    elif args.command == "map":
        tm = extract_type_map(text)
        for name, sem in sorted(tm.items()):
            print(f"  {name}: {sem.name.lower()}")

    elif args.command == "annotate":
        tm = extract_type_map(text)
        result = "\n".join(annotate_line(l, tm) for l in text.splitlines()) + "\n"
        if args.output:
            open(args.output, "w").write(result)
        else:
            print(result, end="")


if __name__ == "__main__":
    _cli()
