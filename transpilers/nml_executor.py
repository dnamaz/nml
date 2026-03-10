#!/usr/bin/env python3
"""
NML Executor — retrieves programs from the library, builds .nml.data,
runs the NML runtime, and returns structured results.

This is the execution engine for the NML-as-protocol architecture.
The LLM orchestrates; this module computes.
"""

import os
import re
import json
import subprocess
import tempfile
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

_PROJECT_ROOT = Path(__file__).parent.parent
NML_BINARY = _PROJECT_ROOT / "nml"
LIBRARY_CLASSIC = _PROJECT_ROOT / "domain" / "output" / "nml-library-classic"
LIBRARY_SYMBOLIC = _PROJECT_ROOT / "domain" / "output" / "nml-library-symbolic"


class NMLExecutor:
    """Retrieves, executes, and traces NML programs from the library."""

    def __init__(self):
        self.classic_manifest = self._load_manifest(LIBRARY_CLASSIC)
        self.symbolic_manifest = self._load_manifest(LIBRARY_SYMBOLIC)
        self.program_count = len(self.classic_manifest.get("taxes", {}))

    @staticmethod
    def _load_manifest(lib_dir: Path) -> dict:
        manifest_path = lib_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                return json.load(f)
        return {}

    def find_program(self, tax_id: str) -> Optional[dict]:
        """Find a program in the library by tax ID."""
        taxes = self.classic_manifest.get("taxes", {})
        if tax_id in taxes:
            entry = taxes[tax_id]
            return {
                "tax_id": tax_id,
                "classic_path": str(LIBRARY_CLASSIC / entry["path"]),
                "symbolic_path": str(LIBRARY_SYMBOLIC / entry["path"]),
                "type": entry.get("type", ""),
                "pattern": entry.get("pattern", ""),
                "state": entry.get("state", ""),
                "instructions": entry.get("instructions", 0),
            }
        return None

    def find_by_lookup(self, state_fips: str, tax_type: str) -> list[dict]:
        """Find programs matching state + tax type."""
        results = []
        taxes = self.classic_manifest.get("taxes", {})
        for tax_id, entry in taxes.items():
            parts = tax_id.split("-")
            if len(parts) >= 4 and parts[0] == state_fips and parts[3] == tax_type:
                results.append(self.find_program(tax_id))
        return [r for r in results if r]

    def get_program_source(self, tax_id: str, syntax: str = "classic") -> Optional[str]:
        """Return the NML source code for a program."""
        prog = self.find_program(tax_id)
        if not prog:
            return None
        path = prog["classic_path"] if syntax == "classic" else prog["symbolic_path"]
        if os.path.exists(path):
            with open(path) as f:
                return f.read()
        return None

    @staticmethod
    def build_nml_data(inputs: dict, metadata: dict = None) -> str:
        """
        Build a .nml.data file from inputs and optional metadata.

        inputs: {"gross_pay": 100000.0, "filing_status": 1.0, ...}
        metadata: {"_action": 1.0, "_source_agent": 1.0, ...}
        """
        lines = []

        if metadata:
            lines.append("# ═══ NML Agent Message ═══")
            for key, value in sorted(metadata.items()):
                if isinstance(value, str):
                    lines.append(f"# _tag:{key} = {value}")
                    lines.append(f"@{key} shape=1 data=0.0")
                elif isinstance(value, float) and value != int(value):
                    lines.append(f"@{key} shape=1 dtype=f64 data={value}")
                else:
                    lines.append(f"@{key} shape=1 data={float(value)}")
            lines.append("")

        lines.append("# Computation inputs")
        for key, value in sorted(inputs.items()):
            if isinstance(value, float) and (abs(value) > 1000 or "." in f"{value}"):
                lines.append(f"@{key} shape=1 dtype=f64 data={value}")
            else:
                lines.append(f"@{key} shape=1 data={float(value)}")

        return "\n".join(lines) + "\n"

    @staticmethod
    def build_tax_data(gross_pay: float, filing_status: float = 1.0,
                       is_exempt: float = 0.0, is_resident: float = 1.0,
                       pay_periods: int = 52) -> str:
        """Build .nml.data for a standard tax calculation."""
        pay_periods_inv = 1.0 / pay_periods if pay_periods > 0 else 0.0
        return NMLExecutor.build_nml_data({
            "gross_pay": gross_pay,
            "filing_status": filing_status,
            "is_exempt": is_exempt,
            "is_resident": is_resident,
            "pay_periods_inv": pay_periods_inv,
        })

    def execute(self, tax_id: str, inputs: dict,
                trace: bool = False, syntax: str = "classic") -> dict:
        """
        Execute an NML program from the library with the given inputs.
        Returns structured result in .nml.data-compatible format.
        """
        prog = self.find_program(tax_id)
        if not prog:
            return self._error_response(f"Program not found: {tax_id}")

        nml_path = prog["classic_path"]
        if not os.path.exists(nml_path):
            return self._error_response(f"NML file missing: {nml_path}")

        data_content = self.build_nml_data(inputs, metadata={
            "_action": 1.0,
            "_program_id": tax_id,
            "_timestamp": float(int(datetime.now().timestamp())),
        })

        return self._run_nml(nml_path, data_content, trace, tax_id, syntax)

    def execute_tax(self, tax_id: str, gross_pay: float,
                    filing_status: float = 1.0,
                    is_exempt: float = 0.0, is_resident: float = 1.0,
                    pay_periods: int = 52,
                    trace: bool = False) -> dict:
        """Convenience method for standard tax calculations."""
        pay_periods_inv = 1.0 / pay_periods if pay_periods > 0 else 0.0
        return self.execute(tax_id, {
            "gross_pay": gross_pay,
            "filing_status": filing_status,
            "is_exempt": is_exempt,
            "is_resident": is_resident,
            "pay_periods_inv": pay_periods_inv,
        }, trace=trace)

    def _run_nml(self, nml_path: str, data_content: str,
                 trace: bool, tax_id: str, syntax: str = "classic") -> dict:
        """Run the NML binary and parse output."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".nml.data", delete=False) as df:
            df.write(data_content)
            data_path = df.name

        try:
            cmd = [str(NML_BINARY), nml_path, data_path]
            if trace:
                cmd.append("--trace")

            t0 = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            elapsed = time.time() - t0

            response = {
                "_status": 0.0 if result.returncode == 0 else 1.0,
                "_cycles": 0.0,
                "_elapsed_ms": round(elapsed * 1000, 2),
                "_program_id": tax_id,
                "_timestamp": float(int(datetime.now().timestamp())),
            }

            if result.returncode != 0:
                response["_status"] = 1.0
                response["_error"] = result.stderr.strip()[:500]
                return response

            trace_lines = []
            for line in result.stdout.split("\n"):
                line = line.strip()
                if not line:
                    continue

                if "HALTED" in line:
                    m = re.search(r"(\d+) cycles", line)
                    if m:
                        response["_cycles"] = float(m.group(1))

                if "tax_amount" in line and "data=[" in line:
                    m = re.search(r"data=\[([^\]]+)\]", line)
                    if m:
                        response["tax_amount"] = float(m.group(1).split(",")[0].strip())

                if trace and ("PC:" in line or line.startswith("  ")):
                    trace_lines.append(line)

            if trace and result.stderr:
                for line in result.stderr.split("\n"):
                    line = line.strip()
                    if line:
                        trace_lines.append(line)

            if trace_lines:
                response["_trace"] = trace_lines

            # Load the program source for inclusion
            prog = self.find_program(tax_id)
            if prog:
                src_path = prog["symbolic_path"] if syntax == "symbolic" else prog["classic_path"]
                if os.path.exists(src_path):
                    with open(src_path) as f:
                        response["_program_source"] = f.read().strip()

            return response

        except subprocess.TimeoutExpired:
            return self._error_response(f"Execution timeout: {tax_id}")
        except Exception as e:
            return self._error_response(str(e))
        finally:
            os.unlink(data_path)

    @staticmethod
    def _error_response(msg: str) -> dict:
        return {
            "_status": 1.0,
            "_error": msg,
            "_timestamp": float(int(datetime.now().timestamp())),
        }

    def format_as_nml_data(self, response: dict) -> str:
        """Format a response dict as .nml.data (the NML-native protocol)."""
        lines = ["# ═══ NML Agent Response ═══"]
        for key, value in sorted(response.items()):
            if key == "_trace":
                for i, tline in enumerate(value):
                    lines.append(f"# _tag:trace_{i:03d} = {tline}")
                continue
            if key == "_program_source":
                for i, pline in enumerate(value.split("\n")):
                    lines.append(f"# _tag:program_{i:03d} = {pline}")
                continue
            if key == "_error":
                lines.append(f"# _tag:error = {value}")
                continue
            if isinstance(value, float):
                if abs(value) > 1000 or (value != 0 and abs(value) < 0.01):
                    lines.append(f"@{key} shape=1 dtype=f64 data={value}")
                else:
                    lines.append(f"@{key} shape=1 data={value}")
        return "\n".join(lines) + "\n"
