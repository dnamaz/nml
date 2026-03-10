"""
NML Provenance Tracker — Phase 4

Tracks the origin of every NML instruction back to source tax data.
Each jurisdiction gets a JSONL sidecar at output/provenance/{key}.provenance.json
containing a full ProvenanceRecord with per-instruction lineage.
"""

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ProvenanceEntry:
    line: int
    instruction: str
    source_field: str
    source_file: str
    source_path: str  # JSONPath, e.g. "$.brackets[0].upperBound"
    description: str

    def to_dict(self) -> dict:
        return {
            "line": self.line,
            "instruction": self.instruction,
            "source_field": self.source_field,
            "source_file": self.source_file,
            "source_path": self.source_path,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProvenanceEntry":
        return cls(
            line=data["line"],
            instruction=data["instruction"],
            source_field=data["source_field"],
            source_file=data["source_file"],
            source_path=data["source_path"],
            description=data["description"],
        )


@dataclass
class ProvenanceRecord:
    jurisdiction_key: str
    entries: List[ProvenanceEntry] = field(default_factory=list)
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    nml_hash: str = ""

    def to_dict(self) -> dict:
        return {
            "jurisdiction_key": self.jurisdiction_key,
            "entries": [e.to_dict() for e in self.entries],
            "generated_at": self.generated_at,
            "nml_hash": self.nml_hash,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProvenanceRecord":
        return cls(
            jurisdiction_key=data["jurisdiction_key"],
            entries=[ProvenanceEntry.from_dict(e) for e in data.get("entries", [])],
            generated_at=data.get("generated_at", ""),
            nml_hash=data.get("nml_hash", ""),
        )


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

_DEFAULT_PROVENANCE_DIR = Path("output/provenance")


class ProvenanceTracker:
    """Manages provenance sidecar files for NML programs."""

    def __init__(self, provenance_dir: Optional[Path] = None) -> None:
        self._dir = provenance_dir or _DEFAULT_PROVENANCE_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    def _sidecar_path(self, jurisdiction_key: str) -> Path:
        return self._dir / f"{jurisdiction_key}.provenance.json"

    # -- write -----------------------------------------------------------

    def store(self, record: ProvenanceRecord) -> None:
        """Persist a ProvenanceRecord as a JSON sidecar file."""
        path = self._sidecar_path(record.jurisdiction_key)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(record.to_dict(), fh, indent=2)
            fh.write("\n")

    # -- read ------------------------------------------------------------

    def load(self, jurisdiction_key: str) -> Optional[ProvenanceRecord]:
        """Load the ProvenanceRecord for *jurisdiction_key*, or None."""
        path = self._sidecar_path(jurisdiction_key)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as fh:
            return ProvenanceRecord.from_dict(json.load(fh))

    # -- queries ---------------------------------------------------------

    def query_instruction(
        self, jurisdiction_key: str, line: int
    ) -> Optional[ProvenanceEntry]:
        """Return the ProvenanceEntry for a specific line number."""
        record = self.load(jurisdiction_key)
        if record is None:
            return None
        for entry in record.entries:
            if entry.line == line:
                return entry
        return None

    def query_value(
        self, jurisdiction_key: str, value: float
    ) -> List[ProvenanceEntry]:
        """Find all instructions whose text contains *value* as a numeric literal."""
        record = self.load(jurisdiction_key)
        if record is None:
            return []

        str_value = _format_numeric(value)
        matches: List[ProvenanceEntry] = []
        for entry in record.entries:
            if str_value in entry.instruction:
                matches.append(entry)
        return matches


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_numeric(value: float) -> str:
    """Return a canonical string representation (drop trailing .0)."""
    if value == int(value):
        return str(int(value))
    return str(value)


def compute_nml_hash(nml_text: str) -> str:
    return hashlib.sha256(nml_text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="provenance_tracker",
        description="Query NML provenance sidecar files.",
    )
    sub = parser.add_subparsers(dest="command")

    q = sub.add_parser("query", help="Query provenance for a jurisdiction.")
    q.add_argument("--key", required=True, help="Jurisdiction key")
    q.add_argument("--line", type=int, default=None, help="NML line number")
    q.add_argument("--value", type=float, default=None, help="Numeric value to search")

    d = sub.add_parser("dump", help="Dump the full provenance record.")
    d.add_argument("--key", required=True, help="Jurisdiction key")

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    tracker = ProvenanceTracker()

    if args.command == "dump":
        record = tracker.load(args.key)
        if record is None:
            print(f"No provenance found for {args.key}", file=sys.stderr)
            sys.exit(1)
        print(json.dumps(record.to_dict(), indent=2))
        return

    if args.command == "query":
        if args.line is not None:
            entry = tracker.query_instruction(args.key, args.line)
            if entry is None:
                print(f"No entry for line {args.line} in {args.key}", file=sys.stderr)
                sys.exit(1)
            print(json.dumps(entry.to_dict(), indent=2))
            return

        if args.value is not None:
            entries = tracker.query_value(args.key, args.value)
            if not entries:
                print(
                    f"No entries matching value {args.value} in {args.key}",
                    file=sys.stderr,
                )
                sys.exit(1)
            print(json.dumps([e.to_dict() for e in entries], indent=2))
            return

        record = tracker.load(args.key)
        if record is None:
            print(f"No provenance found for {args.key}", file=sys.stderr)
            sys.exit(1)
        print(json.dumps(record.to_dict(), indent=2))


if __name__ == "__main__":
    main()
