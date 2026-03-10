"""
NML Audit Log — Phase 4

Append-only audit log for agent-to-agent messages.
Storage: output/audit/agent_log.jsonl  (one JSON object per line)
"""

import argparse
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Ensure sibling modules are importable when run as a script.
_SERVE_DIR = Path(__file__).resolve().parent
if str(_SERVE_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVE_DIR))

from nml_protocol import AgentMessage  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_LOG_DIR = Path("output/audit")
_LOG_FILENAME = "agent_log.jsonl"


# ---------------------------------------------------------------------------
# AuditLog
# ---------------------------------------------------------------------------


class AuditLog:
    """Append-only audit log for NML agent messages."""

    def __init__(self, log_dir: Optional[Path] = None) -> None:
        self._dir = log_dir or _DEFAULT_LOG_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self._dir / _LOG_FILENAME

    # -- write -----------------------------------------------------------

    def log(self, entry: dict) -> None:
        """Append a raw dict entry as a single JSON line."""
        entry.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        entry.setdefault("message_id", str(uuid.uuid4()))
        with open(self._log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")

    def log_message(
        self,
        message: AgentMessage,
        status: str,
        latency_ms: float,
    ) -> None:
        """Extract fields from an AgentMessage and append to the log."""
        payload_bytes = message.payload.program.encode("utf-8")
        payload_hash = "sha256:" + hashlib.sha256(payload_bytes).hexdigest()

        self.log(
            {
                "timestamp": message.header.timestamp,
                "message_id": message.header.message_id,
                "source_agent": message.header.source_agent,
                "target_agent": message.header.target_agent,
                "message_type": message.header.message_type,
                "jurisdiction_key": message.context.jurisdiction_key,
                "payload_hash": payload_hash,
                "payload_size_bytes": len(payload_bytes),
                "status": status,
                "latency_ms": latency_ms,
            }
        )

    # -- read / query ----------------------------------------------------

    def _read_all(self) -> List[dict]:
        """Read every entry from the JSONL log."""
        if not self._log_path.exists():
            return []
        entries: List[dict] = []
        with open(self._log_path, "r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, 1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    entries.append(json.loads(raw))
                except json.JSONDecodeError:
                    print(
                        f"warning: skipping malformed line {lineno}",
                        file=sys.stderr,
                    )
        return entries

    def query(
        self,
        jurisdiction_key: Optional[str] = None,
        agent: Optional[str] = None,
        since: Optional[str] = None,
        message_type: Optional[str] = None,
    ) -> List[dict]:
        """Filter log entries by optional criteria."""
        results: List[dict] = []
        for entry in self._read_all():
            if jurisdiction_key and entry.get("jurisdiction_key") != jurisdiction_key:
                continue
            if agent and entry.get("source_agent") != agent:
                continue
            if message_type and entry.get("message_type") != message_type:
                continue
            if since:
                ts = entry.get("timestamp", "")
                if ts < since:
                    continue
            results.append(entry)
        return results

    def trace(self, message_id: str) -> List[dict]:
        """Find the message with *message_id* and all messages sharing its
        jurisdiction_key, ordered by timestamp."""
        all_entries = self._read_all()

        target_jk: Optional[str] = None
        for entry in all_entries:
            if entry.get("message_id") == message_id:
                target_jk = entry.get("jurisdiction_key")
                break

        if target_jk is None:
            return []

        chain = [e for e in all_entries if e.get("jurisdiction_key") == target_jk]
        chain.sort(key=lambda e: e.get("timestamp", ""))
        return chain

    def stats(self) -> Dict:
        """Aggregate counts by agent and message type, plus average latency."""
        entries = self._read_all()
        by_agent: Dict[str, int] = {}
        by_type: Dict[str, int] = {}
        latencies: List[float] = []

        for entry in entries:
            src = entry.get("source_agent", "unknown")
            by_agent[src] = by_agent.get(src, 0) + 1

            mt = entry.get("message_type", "unknown")
            by_type[mt] = by_type.get(mt, 0) + 1

            lat = entry.get("latency_ms")
            if lat is not None:
                latencies.append(float(lat))

        return {
            "total_messages": len(entries),
            "by_agent": by_agent,
            "by_message_type": by_type,
            "average_latency_ms": (
                round(sum(latencies) / len(latencies), 2) if latencies else 0.0
            ),
            "min_latency_ms": round(min(latencies), 2) if latencies else 0.0,
            "max_latency_ms": round(max(latencies), 2) if latencies else 0.0,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audit_log",
        description="Query the NML agent audit log.",
    )
    sub = parser.add_subparsers(dest="command")

    q = sub.add_parser("query", help="Filter audit log entries.")
    q.add_argument("--jurisdiction", default=None, help="Jurisdiction key filter")
    q.add_argument("--agent", default=None, help="Source agent filter")
    q.add_argument("--since", default=None, help="ISO-8601 timestamp lower bound")
    q.add_argument("--type", dest="message_type", default=None, help="Message type")

    t = sub.add_parser("trace", help="Trace a message chain.")
    t.add_argument("--message-id", required=True, help="UUID of the message to trace")

    sub.add_parser("stats", help="Show aggregate statistics.")

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    audit = AuditLog()

    if args.command == "query":
        results = audit.query(
            jurisdiction_key=args.jurisdiction,
            agent=args.agent,
            since=args.since,
            message_type=args.message_type,
        )
        print(json.dumps(results, indent=2))

    elif args.command == "trace":
        chain = audit.trace(args.message_id)
        if not chain:
            print(f"No messages found for {args.message_id}", file=sys.stderr)
            sys.exit(1)
        print(json.dumps(chain, indent=2))

    elif args.command == "stats":
        print(json.dumps(audit.stats(), indent=2))


if __name__ == "__main__":
    main()
