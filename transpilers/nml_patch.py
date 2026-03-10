"""NML v0.6 – Differential program generation and application.

Computes minimal patches between NML program versions and applies
them with hash verification.
"""

from __future__ import annotations

import difflib
import hashlib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class PatchEntry:
    operation: str  # "set", "del", "ins"
    line: int
    instruction: str = ""


@dataclass
class Patch:
    base_hash: str
    entries: list[PatchEntry] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def compute_hash(nml_program: str) -> str:
    return hashlib.sha256(nml_program.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------

def generate_patch(old_nml: str, new_nml: str) -> Patch:
    """Compare two NML programs and produce a minimal :class:`Patch`."""
    old_lines = old_nml.splitlines()
    new_lines = new_nml.splitlines()
    base_hash = compute_hash(old_nml)

    matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
    entries: list[PatchEntry] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        elif tag == "replace":
            old_span = i2 - i1
            new_span = j2 - j1
            common = min(old_span, new_span)
            for k in range(common):
                entries.append(PatchEntry("set", i1 + k + 1, new_lines[j1 + k]))
            if new_span > old_span:
                for k in range(common, new_span):
                    entries.append(PatchEntry("ins", i1 + common + 1, new_lines[j1 + k]))
            elif old_span > new_span:
                for k in range(common, old_span):
                    entries.append(PatchEntry("del", i1 + k + 1))
        elif tag == "insert":
            for k in range(j1, j2):
                entries.append(PatchEntry("ins", i1 + 1, new_lines[k]))
        elif tag == "delete":
            for k in range(i1, i2):
                entries.append(PatchEntry("del", k + 1))

    return Patch(base_hash=base_hash, entries=entries)


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------

class PatchError(Exception):
    pass


def apply_patch(base_nml: str, patch: Patch) -> str:
    """Apply *patch* to *base_nml*, returning the patched program.

    Raises :class:`PatchError` if the base hash doesn't match.
    Operations are applied in reverse line order to preserve indices.
    """
    actual_hash = compute_hash(base_nml)
    if actual_hash != patch.base_hash:
        raise PatchError(
            f"Hash mismatch: expected {patch.base_hash[:16]}… "
            f"got {actual_hash[:16]}…"
        )

    lines = base_nml.splitlines()

    deletes = sorted(
        [e for e in patch.entries if e.operation == "del"],
        key=lambda e: e.line, reverse=True,
    )
    sets = sorted(
        [e for e in patch.entries if e.operation == "set"],
        key=lambda e: e.line, reverse=True,
    )
    inserts = sorted(
        [e for e in patch.entries if e.operation == "ins"],
        key=lambda e: e.line, reverse=True,
    )

    for e in deletes:
        idx = e.line - 1
        if 0 <= idx < len(lines):
            lines.pop(idx)

    for e in sets:
        idx = e.line - 1
        if 0 <= idx < len(lines):
            lines[idx] = e.instruction

    for e in inserts:
        idx = e.line - 1
        idx = max(0, min(idx, len(lines)))
        lines.insert(idx, e.instruction)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Format / Parse
# ---------------------------------------------------------------------------

def format_patch(patch: Patch) -> str:
    """Serialise *patch* to the NML PTCH text format."""
    parts: list[str] = []
    parts.append(f"⊿  @base  sha256:{patch.base_hash}")
    for e in patch.entries:
        if e.operation == "del":
            parts.append(f'⊿  @del   {e.line}')
        elif e.operation == "set":
            parts.append(f'⊿  @set   {e.line}  "{e.instruction}"')
        elif e.operation == "ins":
            parts.append(f'⊿  @ins   {e.line}  "{e.instruction}"')
    parts.append("⊿  @end")
    return "\n".join(parts)


_PATCH_BASE = re.compile(r'^\s*⊿\s+@base\s+sha256:(\S+)')
_PATCH_SET = re.compile(r'^\s*⊿\s+@set\s+(\d+)\s+"(.*)"')
_PATCH_DEL = re.compile(r'^\s*⊿\s+@del\s+(\d+)')
_PATCH_INS = re.compile(r'^\s*⊿\s+@ins\s+(\d+)\s+"(.*)"')
_PATCH_END = re.compile(r'^\s*⊿\s+@end')


def parse_patch(patch_text: str) -> Patch:
    """Parse NML PTCH text into a :class:`Patch`."""
    base_hash = ""
    entries: list[PatchEntry] = []

    for line in patch_text.splitlines():
        m = _PATCH_BASE.match(line)
        if m:
            base_hash = m.group(1)
            continue
        m = _PATCH_SET.match(line)
        if m:
            entries.append(PatchEntry("set", int(m.group(1)), m.group(2)))
            continue
        m = _PATCH_DEL.match(line)
        if m:
            entries.append(PatchEntry("del", int(m.group(1))))
            continue
        m = _PATCH_INS.match(line)
        if m:
            entries.append(PatchEntry("ins", int(m.group(1)), m.group(2)))
            continue
        if _PATCH_END.match(line):
            break

    if not base_hash:
        raise ValueError("No @base hash found in patch text")

    return Patch(base_hash=base_hash, entries=entries)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="NML differential patch tool")
    sub = parser.add_subparsers(dest="command")

    p_gen = sub.add_parser("generate", help="Generate patch between two NML files")
    p_gen.add_argument("--old", required=True, help="Old NML file")
    p_gen.add_argument("--new", required=True, help="New NML file")
    p_gen.add_argument("-o", "--output", help="Output file (default: stdout)")

    p_apply = sub.add_parser("apply", help="Apply a patch to a base NML file")
    p_apply.add_argument("--base", required=True, help="Base NML file")
    p_apply.add_argument("--patch", required=True, help="Patch file")
    p_apply.add_argument("-o", "--output", help="Output file (default: stdout)")

    p_verify = sub.add_parser("verify", help="Verify patch applicability")
    p_verify.add_argument("--base", required=True, help="Base NML file")
    p_verify.add_argument("--patch", required=True, help="Patch file")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "generate":
        old_text = Path(args.old).read_text()
        new_text = Path(args.new).read_text()
        patch = generate_patch(old_text, new_text)
        result = format_patch(patch)
        _write_output(result, args.output)

    elif args.command == "apply":
        base_text = Path(args.base).read_text()
        patch_text = Path(args.patch).read_text()
        patch = parse_patch(patch_text)
        try:
            result = apply_patch(base_text, patch)
        except PatchError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        _write_output(result, getattr(args, "output", None))

    elif args.command == "verify":
        base_text = Path(args.base).read_text()
        patch_text = Path(args.patch).read_text()
        patch = parse_patch(patch_text)
        actual = compute_hash(base_text)
        if actual == patch.base_hash:
            print(f"✓ hash match ({actual[:16]}…) – {len(patch.entries)} operations")
        else:
            print(
                f"✗ hash mismatch: base={actual[:16]}… patch expects={patch.base_hash[:16]}…",
                file=sys.stderr,
            )
            sys.exit(1)


def _write_output(text: str, path: Optional[str]) -> None:
    if path:
        Path(path).write_text(text + "\n")
        print(f"Written to {path}")
    else:
        print(text)


if __name__ == "__main__":
    _cli()
