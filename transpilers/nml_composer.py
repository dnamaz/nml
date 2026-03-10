"""NML v0.6 – Fragment composition engine.

Extracts, links, and composes NML program fragments for modular
program construction and M2M pipelines.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Fragment model
# ---------------------------------------------------------------------------

@dataclass
class Fragment:
    name: str
    lines: list[str]
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    meta: dict[str, str] = field(default_factory=dict)

    @property
    def body(self) -> str:
        return "\n".join(self.lines)

    def __repr__(self) -> str:
        return (
            f"Fragment({self.name!r}, {len(self.lines)} lines, "
            f"in={self.inputs}, out={self.outputs})"
        )


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_FRAG_START = re.compile(
    r"^\s*(?:◆|FRAG(?:MENT)?)\s+(\S+)", re.IGNORECASE
)
_FRAG_END = re.compile(
    r"^\s*(?:◇|ENDF|END_FRAGMENT)\b", re.IGNORECASE
)
_LINK_LINE = re.compile(
    r"^\s*(?:⊕|LINK|IMPORT)\s+@(\S+)", re.IGNORECASE
)
_META_LINE = re.compile(
    r"^\s*META\s+@(\S+)\s*(.*)", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------

def extract_fragments(nml_program: str) -> dict[str, Fragment]:
    """Parse *nml_program* and return all named fragments."""
    fragments: dict[str, Fragment] = {}
    current_name: Optional[str] = None
    current_lines: list[str] = []
    current_meta: dict[str, str] = {}
    current_inputs: list[str] = []
    current_outputs: list[str] = []

    for raw_line in nml_program.splitlines():
        m_start = _FRAG_START.match(raw_line)
        m_end = _FRAG_END.match(raw_line)

        if m_start:
            current_name = m_start.group(1)
            current_lines = []
            current_meta = {}
            current_inputs = []
            current_outputs = []
            continue

        if m_end and current_name is not None:
            fragments[current_name] = Fragment(
                name=current_name,
                lines=current_lines,
                inputs=current_inputs,
                outputs=current_outputs,
                meta=current_meta,
            )
            current_name = None
            continue

        if current_name is not None:
            m_meta = _META_LINE.match(raw_line)
            if m_meta:
                key, value = m_meta.group(1).lower(), m_meta.group(2).strip()
                current_meta[key] = value
                if key == "input":
                    current_inputs.extend(_split_names(value))
                elif key == "output":
                    current_outputs.extend(_split_names(value))
            current_lines.append(raw_line)

    return fragments


def _split_names(value: str) -> list[str]:
    return [n.strip() for n in re.split(r"[,\s]+", value) if n.strip()]


# ---------------------------------------------------------------------------
# Resolve links
# ---------------------------------------------------------------------------

def resolve_links(
    nml_program: str,
    fragment_library: Optional[dict[str, Fragment]] = None,
) -> str:
    """Replace all LINK / ⊕ @name references with the fragment body inline."""
    if fragment_library is None:
        fragment_library = extract_fragments(nml_program)

    resolved: list[str] = []
    seen: set[str] = set()

    for line in nml_program.splitlines():
        m = _LINK_LINE.match(line)
        if m:
            name = m.group(1)
            if name in seen:
                resolved.append(f"#  [already inlined: {name}]")
                continue
            frag = fragment_library.get(name)
            if frag is None:
                resolved.append(f"#  [unresolved link: {name}]")
                continue
            seen.add(name)
            resolved.append(f"#  ── begin {name} ──")
            resolved.extend(frag.lines)
            resolved.append(f"#  ── end {name} ──")
        else:
            resolved.append(line)

    return "\n".join(resolved)


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------

def compose_fragments(
    fragments: list[Fragment],
    entry_name: Optional[str] = None,
) -> str:
    """Combine *fragments* into a single NML program.

    If *entry_name* is given, that fragment is placed last (as the entry
    point); otherwise fragments are emitted in dependency order.
    """
    by_name = {f.name: f for f in fragments}
    order = _topo_sort(fragments, by_name)

    if entry_name and entry_name in by_name:
        order = [n for n in order if n != entry_name] + [entry_name]

    parts: list[str] = []
    for name in order:
        frag = by_name[name]
        parts.append(f"#  ══ fragment: {name} ══")
        parts.extend(frag.lines)
        parts.append("")

    combined = "\n".join(parts)
    lib = {f.name: f for f in fragments}
    return resolve_links(combined, fragment_library=lib)


def _topo_sort(
    fragments: list[Fragment], by_name: dict[str, Fragment]
) -> list[str]:
    """Simple topological sort based on input/output name overlap."""
    provides: dict[str, str] = {}
    for f in fragments:
        for o in f.outputs:
            provides[o] = f.name

    deps: dict[str, set[str]] = {f.name: set() for f in fragments}
    for f in fragments:
        for inp in f.inputs:
            provider = provides.get(inp)
            if provider and provider != f.name:
                deps[f.name].add(provider)

    order: list[str] = []
    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in visiting:
            order.append(name)
            visited.add(name)
            return
        visiting.add(name)
        for dep in sorted(deps.get(name, set())):
            visit(dep)
        visiting.discard(name)
        visited.add(name)
        order.append(name)

    for f in fragments:
        visit(f.name)

    return order


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

def validate_fragment_compatibility(
    producer: Fragment, consumer: Fragment
) -> list[str]:
    """Check that *producer* outputs satisfy *consumer* inputs."""
    errors: list[str] = []
    producer_outputs = set(producer.outputs)

    for inp in consumer.inputs:
        bare_inp = inp.split(":")[0]
        matched = False
        for out in producer_outputs:
            bare_out = out.split(":")[0]
            if bare_inp == bare_out:
                matched = True
                inp_type = inp.split(":")[1] if ":" in inp else None
                out_type = out.split(":")[1] if ":" in out else None
                if inp_type and out_type and inp_type != out_type:
                    errors.append(
                        f"type mismatch: {producer.name}.{out} -> "
                        f"{consumer.name}.{inp}"
                    )
                break
        if not matched:
            errors.append(
                f"missing input: {consumer.name} requires '{bare_inp}' "
                f"not provided by {producer.name}"
            )

    return errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="NML fragment composer")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--extract", metavar="FILE", help="List fragments in an NML program")
    group.add_argument("--compose", nargs="+", metavar="FILE", help="Compose fragment files")
    group.add_argument("--resolve", metavar="FILE", help="Resolve all LINKs inline")

    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    parser.add_argument("--entry", help="Entry-point fragment name (for --compose)")

    args = parser.parse_args()

    if args.extract:
        text = Path(args.extract).read_text()
        frags = extract_fragments(text)
        if not frags:
            print("No fragments found.")
            return
        for name, frag in frags.items():
            print(f"  ◆ {name}  ({len(frag.lines)} lines)")
            if frag.inputs:
                print(f"      inputs:  {', '.join(frag.inputs)}")
            if frag.outputs:
                print(f"      outputs: {', '.join(frag.outputs)}")
            for k, v in frag.meta.items():
                if k not in ("input", "output"):
                    print(f"      @{k}: {v}")

    elif args.compose:
        all_frags: list[Fragment] = []
        for path in args.compose:
            text = Path(path).read_text()
            frags = extract_fragments(text)
            all_frags.extend(frags.values())
        if not all_frags:
            print("No fragments found in provided files.", file=sys.stderr)
            sys.exit(1)
        result = compose_fragments(all_frags, entry_name=args.entry)
        _write_output(result, args.output)

    elif args.resolve:
        text = Path(args.resolve).read_text()
        result = resolve_links(text)
        _write_output(result, args.output)


def _write_output(text: str, path: Optional[str]) -> None:
    if path:
        Path(path).write_text(text + "\n")
        print(f"Written to {path}")
    else:
        print(text)


if __name__ == "__main__":
    _cli()
