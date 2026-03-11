#!/usr/bin/env python3
"""
NML Golden Test Regression Suite

Maintains known-correct input/output pairs for every jurisdiction and verifies
that NML programs still produce correct results after updates.

Usage:
    python3 nml_regression.py generate [--limit N]
    python3 nml_regression.py run [--key KEY]
    python3 nml_regression.py diff --key KEY
"""

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "domain" / "transpilers"))
sys.path.insert(0, str(Path(__file__).parent))

from domain_validate import run_nml
from domain_transpiler import scan_tax_data, TAX_DATA_DIR, generate_employee_data

NML_BINARY = _PROJECT_ROOT / "nml"
NML_LIBRARY_DIR = _PROJECT_ROOT / "domain" / "output" / "nml-library-symbolic"
GOLDEN_PATH = _PROJECT_ROOT / "domain" / "output" / "validated" / "golden_tests.jsonl"

BRACKET_TYPES = {"FIT", "SIT", "PIT"}

INCOME_LEVELS = [2000.0, 5000.0, 10000.0, 50000.0]


def _extract_inputs(nml_source: str) -> list[str]:
    """Parse ↓ directives to discover which @-named inputs the program expects."""
    inputs = []
    for line in nml_source.splitlines():
        m = re.match(r"↓\s+\S+\s+@(\S+)", line)
        if m:
            inputs.append(m.group(1))
    return inputs


def _build_data_string(input_names: list[str], values: dict[str, float]) -> str:
    """Build an NML .data file from a dict of named values."""
    lines = []
    for name in input_names:
        val = values.get(name, 0.0)
        lines.append(f"@{name} shape=1 data={val:.2f}")
    return "\n".join(lines)


def _extract_outputs(nml_source: str) -> list[str]:
    """Parse ↑ directives to discover which @-named outputs the program writes."""
    outputs = []
    for line in nml_source.splitlines():
        m = re.match(r"↑\s+\S+\s+@(\S+)", line)
        if m:
            outputs.append(m.group(1))
    return outputs


def _sha256(content: str) -> str:
    return "sha256:" + hashlib.sha256(content.encode()).hexdigest()[:16]


def _make_test_vectors(tax_type: str, input_names: list[str]) -> list[dict]:
    """Generate test vectors appropriate for the jurisdiction's tax type."""
    vectors = []
    has_filing_status = "filing_status" in input_names
    use_bracket_variants = has_filing_status and tax_type in BRACKET_TYPES
    filing_statuses = [1.0, 2.0] if use_bracket_variants else [1.0]

    for gross in INCOME_LEVELS:
        for fs in filing_statuses:
            values = {"gross_pay": gross, "is_exempt": 0.0}
            if has_filing_status:
                values["filing_status"] = fs
            if "is_resident" in input_names:
                values["is_resident"] = 1.0
            if "pay_periods_inv" in input_names:
                values["pay_periods_inv"] = 1.0 / 26
            vectors.append(values)

    return vectors


class GoldenTestSuite:
    """Generate, run, and diff golden regression tests for NML programs."""

    def generate(self, nml_library_dir: Path = NML_LIBRARY_DIR,
                 output_path: Path = GOLDEN_PATH, limit: int = 0) -> int:
        """Walk the NML library, run every program with test vectors, store golden baselines."""
        nml_files = sorted(nml_library_dir.rglob("*.nml"))
        if limit > 0:
            nml_files = nml_files[:limit]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0

        with open(output_path, "w") as out:
            for i, nml_path in enumerate(nml_files):
                jurisdiction_key = nml_path.stem
                tax_type = nml_path.parent.name

                nml_source = nml_path.read_text()
                input_names = _extract_inputs(nml_source)
                output_names = _extract_outputs(nml_source)

                if not input_names or "gross_pay" not in input_names:
                    continue

                vectors = _make_test_vectors(tax_type, input_names)
                test_vectors = []

                for values in vectors:
                    data_str = _build_data_string(input_names, values)
                    result = run_nml(nml_source, data_str)

                    if "_error" in result:
                        continue

                    expected = {}
                    for name in output_names:
                        if name in result:
                            expected[name] = result[name]
                    if not expected:
                        for k, v in result.items():
                            if not k.startswith("_"):
                                expected[k] = v

                    if expected:
                        test_vectors.append({
                            "inputs": values,
                            "expected": expected,
                        })

                if not test_vectors:
                    continue

                entry = {
                    "jurisdiction_key": jurisdiction_key,
                    "tax_type": tax_type,
                    "test_vectors": test_vectors,
                    "nml_hash": _sha256(nml_source),
                    "generated_date": date.today().isoformat(),
                }
                out.write(json.dumps(entry) + "\n")
                count += 1

                if (i + 1) % 100 == 0:
                    print(f"  [{i+1}/{len(nml_files)}] generated {count} golden entries...")

        print(f"Generated {count} golden test entries → {output_path}")
        return count

    def run(self, nml_library_dir: Path = NML_LIBRARY_DIR,
            golden_path: Path = GOLDEN_PATH, key: str | None = None) -> dict:
        """Run regression: compare current NML output against golden baselines."""
        entries = _load_golden(golden_path)
        if key:
            entries = [e for e in entries if e["jurisdiction_key"] == key]

        total = len(entries)
        passed = 0
        failed = 0
        errors = 0
        failures = []

        for i, entry in enumerate(entries):
            jk = entry["jurisdiction_key"]
            tt = entry["tax_type"]
            nml_path = nml_library_dir / tt / f"{jk}.nml"

            if not nml_path.exists():
                errors += 1
                failures.append({"key": jk, "reason": "nml_file_missing"})
                continue

            nml_source = nml_path.read_text()
            input_names = _extract_inputs(nml_source)
            entry_passed = True

            for vi, vec in enumerate(entry["test_vectors"]):
                data_str = _build_data_string(input_names, vec["inputs"])
                result = run_nml(nml_source, data_str)

                if "_error" in result:
                    entry_passed = False
                    failures.append({
                        "key": jk, "vector": vi,
                        "reason": "runtime_error",
                        "error": result["_error"],
                    })
                    break

                for out_name, exp_val in vec["expected"].items():
                    actual = result.get(out_name)
                    if actual is None:
                        entry_passed = False
                        failures.append({
                            "key": jk, "vector": vi,
                            "reason": "missing_output",
                            "output": out_name,
                        })
                        break
                    if isinstance(exp_val, (int, float)) and isinstance(actual, (int, float)):
                        if abs(actual - exp_val) > 0.02:
                            entry_passed = False
                            failures.append({
                                "key": jk, "vector": vi,
                                "reason": "value_mismatch",
                                "output": out_name,
                                "expected": exp_val,
                                "actual": actual,
                            })
                            break
                    elif actual != exp_val:
                        entry_passed = False
                        failures.append({
                            "key": jk, "vector": vi,
                            "reason": "value_mismatch",
                            "output": out_name,
                            "expected": exp_val,
                            "actual": actual,
                        })
                        break

                if not entry_passed:
                    break

            if entry_passed:
                passed += 1
            else:
                failed += 1

            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{total}] passed={passed} failed={failed} errors={errors}")

        report = {
            "total": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "failures": failures,
        }

        status = "PASS" if failed == 0 and errors == 0 else "FAIL"
        print(f"\nRegression {status}: {passed}/{total} passed, "
              f"{failed} failed, {errors} errors")

        return report

    def diff(self, golden_path: Path = GOLDEN_PATH, key: str = "") -> dict:
        """Show expected vs actual for every test vector of one jurisdiction."""
        if not key:
            print("Error: --key is required for diff")
            return {}

        entries = _load_golden(golden_path)
        entry = next((e for e in entries if e["jurisdiction_key"] == key), None)
        if not entry:
            print(f"No golden entry for key: {key}")
            return {}

        tt = entry["tax_type"]
        nml_path = NML_LIBRARY_DIR / tt / f"{key}.nml"
        if not nml_path.exists():
            print(f"NML file not found: {nml_path}")
            return {}

        nml_source = nml_path.read_text()
        input_names = _extract_inputs(nml_source)
        current_hash = _sha256(nml_source)
        hash_changed = current_hash != entry.get("nml_hash", "")

        print(f"Jurisdiction: {key}  (type={tt})")
        print(f"NML hash: {current_hash}" +
              (" [CHANGED]" if hash_changed else " [unchanged]"))
        print(f"Golden date: {entry.get('generated_date', '?')}")
        print()

        diffs = []
        for vi, vec in enumerate(entry["test_vectors"]):
            data_str = _build_data_string(input_names, vec["inputs"])
            result = run_nml(nml_source, data_str)

            input_summary = ", ".join(f"{k}={v}" for k, v in vec["inputs"].items())
            print(f"  Vector {vi}: {input_summary}")

            if "_error" in result:
                print(f"    ERROR: {result['_error']}")
                diffs.append({"vector": vi, "error": result["_error"]})
                continue

            vec_diffs = {}
            all_match = True
            for out_name, exp_val in vec["expected"].items():
                actual = result.get(out_name)
                match = False
                if isinstance(exp_val, (int, float)) and isinstance(actual, (int, float)):
                    match = abs(actual - exp_val) <= 0.02
                else:
                    match = actual == exp_val

                status = "OK" if match else "MISMATCH"
                if not match:
                    all_match = False
                print(f"    {out_name}: expected={exp_val}  actual={actual}  [{status}]")
                if not match:
                    vec_diffs[out_name] = {"expected": exp_val, "actual": actual}

            if all_match:
                print(f"    ✓ all outputs match")
            else:
                diffs.append({"vector": vi, "inputs": vec["inputs"], "diffs": vec_diffs})

        print(f"\n{'No differences' if not diffs else f'{len(diffs)} vector(s) differ'}")
        return {"key": key, "hash_changed": hash_changed, "diffs": diffs}


def _load_golden(path: Path) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def main():
    parser = argparse.ArgumentParser(description="NML Golden Test Regression Suite")
    sub = parser.add_subparsers(dest="command")

    gen_p = sub.add_parser("generate", help="Generate golden test baselines")
    gen_p.add_argument("--limit", type=int, default=0,
                       help="Max jurisdictions to process (0=all)")

    run_p = sub.add_parser("run", help="Run regression tests against golden baselines")
    run_p.add_argument("--key", type=str, default=None,
                       help="Test only this jurisdiction key")

    diff_p = sub.add_parser("diff", help="Show diffs for one jurisdiction")
    diff_p.add_argument("--key", type=str, required=True,
                        help="Jurisdiction key to diff")

    args = parser.parse_args()
    suite = GoldenTestSuite()

    if args.command == "generate":
        suite.generate(limit=args.limit)
    elif args.command == "run":
        report = suite.run(key=args.key)
        if report.get("failed", 0) > 0 or report.get("errors", 0) > 0:
            sys.exit(1)
    elif args.command == "diff":
        suite.diff(key=args.key)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
