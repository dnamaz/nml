#!/usr/bin/env python3
"""
NML Cross-Jurisdiction Anomaly Detection — post-batch-update health checks.

Scans an NML library for suspicious patterns: statistical outliers in rates
or thresholds, missing jurisdictions, empty programs, and duplicate programs.

Designed to run after batch updates to catch data errors before deployment.
"""

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from nml_diff import extract_structure


# ── Constants ────────────────────────────────────────────────────────────


BRACKET_TAX_TYPES = {"FIT", "SIT", "COUNTY", "CITY", "SCHL", "PIT", "EIT"}

OUTLIER_STDDEV_THRESHOLD = 2.0

MIN_INSTRUCTION_COUNT = 5

SENTINEL_THRESHOLD = 1e9


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class Anomaly:
    jurisdiction_key: str
    anomaly_type: str   # "rate_outlier", "threshold_outlier", "missing", "empty_program", "duplicate"
    severity: str       # "warning", "error"
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class AnomalyReport:
    scan_date: str
    total_programs: int
    anomalies: list[Anomaly] = field(default_factory=list)
    by_type: dict = field(default_factory=dict)
    by_severity: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "scan_date": self.scan_date,
            "total_programs": self.total_programs,
            "anomaly_count": len(self.anomalies),
            "by_type": self.by_type,
            "by_severity": self.by_severity,
            "anomalies": [
                {
                    "jurisdiction_key": a.jurisdiction_key,
                    "anomaly_type": a.anomaly_type,
                    "severity": a.severity,
                    "message": a.message,
                    "details": a.details,
                }
                for a in self.anomalies
            ],
        }

    def summary(self) -> str:
        lines = [
            f"Anomaly scan: {self.scan_date}",
            f"  Scanned {self.total_programs} programs, "
            f"found {len(self.anomalies)} anomaly(ies)",
        ]
        if self.by_type:
            lines.append(
                "  By type: "
                + ", ".join(f"{k}={v}" for k, v in sorted(self.by_type.items()))
            )
        if self.by_severity:
            lines.append(
                "  By severity: "
                + ", ".join(f"{k}={v}" for k, v in sorted(self.by_severity.items()))
            )
        if self.anomalies:
            lines.append("")
            for a in self.anomalies:
                tag = "ERROR" if a.severity == "error" else " WARN"
                lines.append(f"  [{tag}] {a.jurisdiction_key}: {a.message}")
        return "\n".join(lines)


# ── Helpers ──────────────────────────────────────────────────────────────


def _infer_tax_type(jurisdiction_key: str) -> str:
    parts = jurisdiction_key.split("-")
    return parts[3] if len(parts) >= 4 else ""


def _fips_prefix(jurisdiction_key: str) -> str:
    """Extract state-county-locality FIPS from a jurisdiction key."""
    parts = jurisdiction_key.split("-")
    if len(parts) >= 3:
        return f"{parts[0]}-{parts[1]}-{parts[2]}"
    return jurisdiction_key


def _mean_stddev(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        return (values[0] if values else 0.0, 0.0)
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


# ── Check functions ──────────────────────────────────────────────────────


def _check_rate_outliers(structures: dict[str, dict]) -> list[Anomaly]:
    """Flag jurisdictions whose tax rates are > 2 stddev from the mean
    for their tax type."""
    rates_by_type: dict[str, list[tuple[str, float]]] = defaultdict(list)

    for key, struct in structures.items():
        tax_type = _infer_tax_type(key)
        for _status, fs in struct["filing_statuses"].items():
            for bracket in fs.get("brackets", []):
                if bracket["rate"] > 0:
                    rates_by_type[tax_type].append((key, bracket["rate"]))
            flat = fs.get("flat_rate")
            if flat and flat["value"] > 0:
                rates_by_type[tax_type].append((key, flat["value"]))

    anomalies: list[Anomaly] = []
    seen: set[tuple[str, str]] = set()

    for tax_type, entries in rates_by_type.items():
        if len(entries) < 3:
            continue
        values = [v for _, v in entries]
        mean, stddev = _mean_stddev(values)
        if stddev < 1e-9:
            continue
        for key, rate in entries:
            z = abs(rate - mean) / stddev
            dedup = (key, f"{rate:.8f}")
            if z > OUTLIER_STDDEV_THRESHOLD and dedup not in seen:
                seen.add(dedup)
                anomalies.append(Anomaly(
                    jurisdiction_key=key,
                    anomaly_type="rate_outlier",
                    severity="warning",
                    message=(
                        f"Rate {rate:.6f} is {z:.1f} stddev from "
                        f"{tax_type} mean {mean:.6f}"
                    ),
                    details={
                        "rate": rate,
                        "mean": round(mean, 6),
                        "stddev": round(stddev, 6),
                        "z_score": round(z, 2),
                        "tax_type": tax_type,
                    },
                ))

    return anomalies


def _check_threshold_outliers(structures: dict[str, dict]) -> list[Anomaly]:
    """Flag jurisdictions with bracket thresholds > 2 stddev from the mean
    for their tax type."""
    thresholds_by_type: dict[str, list[tuple[str, float]]] = defaultdict(list)

    for key, struct in structures.items():
        tax_type = _infer_tax_type(key)
        for _status, fs in struct["filing_statuses"].items():
            for bracket in fs.get("brackets", []):
                t = bracket["threshold"]
                if 0 < t < SENTINEL_THRESHOLD:
                    thresholds_by_type[tax_type].append((key, t))

    anomalies: list[Anomaly] = []
    seen: set[tuple[str, str]] = set()

    for tax_type, entries in thresholds_by_type.items():
        if len(entries) < 3:
            continue
        values = [v for _, v in entries]
        mean, stddev = _mean_stddev(values)
        if stddev < 1e-9:
            continue
        for key, threshold in entries:
            z = abs(threshold - mean) / stddev
            dedup = (key, f"{threshold:.2f}")
            if z > OUTLIER_STDDEV_THRESHOLD and dedup not in seen:
                seen.add(dedup)
                anomalies.append(Anomaly(
                    jurisdiction_key=key,
                    anomaly_type="threshold_outlier",
                    severity="warning",
                    message=(
                        f"Threshold ${threshold:,.2f} is {z:.1f} stddev from "
                        f"{tax_type} mean ${mean:,.2f}"
                    ),
                    details={
                        "threshold": threshold,
                        "mean": round(mean, 2),
                        "stddev": round(stddev, 2),
                        "z_score": round(z, 2),
                        "tax_type": tax_type,
                    },
                ))

    return anomalies


def _check_missing_jurisdictions(
    found_keys: set[str], library_dir: str
) -> list[Anomaly]:
    """Check for jurisdictions listed in manifest.json but absent from the library."""
    manifest_path = Path(library_dir) / "manifest.json"
    if not manifest_path.exists():
        return []

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    expected: set[str] = set()
    if isinstance(manifest, dict):
        for key in manifest.get("jurisdictions", manifest.get("keys", [])):
            if isinstance(key, str):
                expected.add(key)
    elif isinstance(manifest, list):
        expected = {k for k in manifest if isinstance(k, str)}

    missing = expected - found_keys
    return [
        Anomaly(
            jurisdiction_key=key,
            anomaly_type="missing",
            severity="error",
            message="Expected jurisdiction missing from library",
            details={"expected_key": key},
        )
        for key in sorted(missing)
    ]


def _check_empty_programs(
    file_map: dict[str, tuple[str, int]], tax_type_filter: str
) -> list[Anomaly]:
    """Flag programs that are suspiciously short for bracket tax types."""
    anomalies: list[Anomaly] = []
    for key, (_content, instruction_count) in file_map.items():
        inferred_type = tax_type_filter or _infer_tax_type(key)
        if (
            inferred_type.upper() in BRACKET_TAX_TYPES
            and instruction_count < MIN_INSTRUCTION_COUNT
        ):
            anomalies.append(Anomaly(
                jurisdiction_key=key,
                anomaly_type="empty_program",
                severity="error",
                message=(
                    f"Program has only {instruction_count} instruction(s) "
                    f"(expected ≥{MIN_INSTRUCTION_COUNT} for {inferred_type})"
                ),
                details={
                    "instruction_count": instruction_count,
                    "tax_type": inferred_type,
                },
            ))
    return anomalies


def _check_duplicates(file_map: dict[str, tuple[str, int]]) -> list[Anomaly]:
    """Flag programs with identical content across different FIPS codes."""
    hash_to_keys: dict[str, list[str]] = defaultdict(list)
    for key, (content, _count) in file_map.items():
        normalized = content.strip()
        if not normalized:
            continue
        h = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
        hash_to_keys[h].append(key)

    anomalies: list[Anomaly] = []
    for _h, keys in hash_to_keys.items():
        if len(keys) < 2:
            continue

        fips_codes = {_fips_prefix(k) for k in keys}
        if len(fips_codes) < 2:
            continue

        for key in keys:
            others = [k for k in keys if k != key]
            anomalies.append(Anomaly(
                jurisdiction_key=key,
                anomaly_type="duplicate",
                severity="warning",
                message=(
                    f"Identical program shared with {len(others)} other "
                    f"jurisdiction(s): {', '.join(others)}"
                ),
                details={"duplicate_keys": others},
            ))

    return anomalies


# ── Public API ───────────────────────────────────────────────────────────


def scan_anomalies(library_dir: str, tax_type: str = "") -> AnomalyReport:
    """Scan an NML library directory for cross-jurisdiction anomalies.

    Args:
        library_dir: Path to the NML library root (e.g. output/nml-library-symbolic/).
        tax_type: Optional filter (e.g. "FIT") to restrict scan to one tax type.

    Returns:
        AnomalyReport with all detected anomalies.
    """
    root = Path(library_dir)
    nml_files = sorted(root.rglob("*.nml"))
    if tax_type:
        nml_files = [
            f for f in nml_files
            if f"-{tax_type}-" in f.name or f.parent.name == tax_type
        ]

    file_map: dict[str, tuple[str, int]] = {}
    structures: dict[str, dict] = {}

    for nml_path in nml_files:
        key = nml_path.stem
        content = nml_path.read_text(encoding="utf-8")
        instruction_count = sum(1 for line in content.splitlines() if line.strip())
        file_map[key] = (content, instruction_count)
        try:
            structures[key] = extract_structure(content)
        except Exception:
            structures[key] = {"filing_statuses": {}, "instruction_count": 0}

    anomalies: list[Anomaly] = []
    anomalies.extend(_check_rate_outliers(structures))
    anomalies.extend(_check_threshold_outliers(structures))
    anomalies.extend(_check_missing_jurisdictions(set(file_map), library_dir))
    anomalies.extend(_check_empty_programs(file_map, tax_type))
    anomalies.extend(_check_duplicates(file_map))

    by_type: dict[str, int] = defaultdict(int)
    by_severity: dict[str, int] = defaultdict(int)
    for a in anomalies:
        by_type[a.anomaly_type] += 1
        by_severity[a.severity] += 1

    return AnomalyReport(
        scan_date=date.today().isoformat(),
        total_programs=len(file_map),
        anomalies=anomalies,
        by_type=dict(by_type),
        by_severity=dict(by_severity),
    )


# ── CLI ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Cross-jurisdiction anomaly detection for NML library",
    )
    parser.add_argument("library_dir", help="Path to NML library directory")
    parser.add_argument("--type", default="", help="Filter by tax type (e.g. FIT)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if not Path(args.library_dir).is_dir():
        print(f"Error: {args.library_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    report = scan_anomalies(args.library_dir, tax_type=args.type)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.summary())


if __name__ == "__main__":
    main()
