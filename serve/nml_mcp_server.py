#!/usr/bin/env python3
"""
NML MCP Server — exposes NML toolchain as MCP tools for frontier LLMs.

Tools:
  nml_spec_lookup     — Retrieve NML spec sections (opcodes, syntax, registers, etc.)
  nml_transpile       — Transpile a tax jurisdiction to NML via the transpiler service
  nml_validate        — Validate an NML program (grammar + semantic + execution)
  nml_execute         — Execute an NML program against input data
  nml_library_lookup  — Retrieve golden NML examples from the pre-built library
  nml_scan            — Scan available tax jurisdictions and types
  nml_intent          — Classify a natural-language message into a structured intent
  nml_compact         — Convert multi-line NML to single-line ¶-delimited compact form
  nml_format          — Format compact or messy NML into readable multi-line output

Architecture:
  - Tools that hit HTTP services (transpile, validate, execute) require the agent
    services to be running (start_agents.sh). They degrade gracefully if offline.
  - Tools that read local files (spec_lookup, library_lookup, scan) work standalone.

Usage:
  python3 nml_mcp_server.py              # stdio mode (for Cursor / Claude Desktop)
  python3 nml_mcp_server.py --transport sse --port 9100  # SSE mode (for web clients)
"""

import json
import sys
from pathlib import Path

import aiohttp
from mcp.server.fastmcp import FastMCP

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOMAIN_ROOT = PROJECT_ROOT / "domain"
DOCS_DIR = PROJECT_ROOT / "docs"
LIBRARY_DIR = DOMAIN_ROOT / "output" / "nml-library-symbolic"
LIBRARY_CLASSIC_DIR = DOMAIN_ROOT / "output" / "nml-library-classic"
TAX_DATA_DIR = DOMAIN_ROOT / "tax-data"

TRANSPILERS_DIR = PROJECT_ROOT / "transpilers"
SERVE_DIR = PROJECT_ROOT / "serve"
DOMAIN_TRANSPILERS_DIR = DOMAIN_ROOT / "transpilers"
DOMAIN_SERVE_DIR = DOMAIN_ROOT / "serve"
for p in [str(TRANSPILERS_DIR), str(SERVE_DIR),
          str(DOMAIN_TRANSPILERS_DIR), str(DOMAIN_SERVE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

TRANSPILER_URL = "http://localhost:8083"
VALIDATOR_URL = "http://localhost:8084"
ENGINE_URL = "http://localhost:8085"

mcp = FastMCP(
    "nml-toolchain",
    instructions="NML (Neural Machine Language) toolchain — transpile, validate, execute, and inspect NML programs. Use nml_spec_lookup to learn the language before generating NML. Use nml_library_lookup for reference examples.",
)

# ── Spec sections (pre-indexed for fast lookup) ──────────────────────────────

_SPEC_CACHE: dict[str, str] = {}


def _load_spec_sections() -> dict[str, str]:
    """Parse NML_SPEC.md into named sections keyed by heading."""
    if _SPEC_CACHE:
        return _SPEC_CACHE

    spec_path = DOCS_DIR / "NML_SPEC.md"
    if not spec_path.exists():
        return {"error": f"Spec not found at {spec_path}"}

    text = spec_path.read_text()
    sections: dict[str, str] = {"full": text}
    current_heading = "overview"
    current_lines: list[str] = []

    for line in text.split("\n"):
        if line.startswith("## "):
            if current_lines:
                sections[current_heading] = "\n".join(current_lines)
            current_heading = line.lstrip("# ").strip().lower().replace(" ", "_")
            current_lines = [line]
        elif line.startswith("### ") and current_heading:
            sub_heading = line.lstrip("# ").strip().lower().replace(" ", "_")
            sections[f"{current_heading}/{sub_heading}"] = ""
            current_lines.append(line)
        else:
            current_lines.append(line)

    if current_lines:
        sections[current_heading] = "\n".join(current_lines)

    for key in list(sections):
        if "/" in key and not sections[key]:
            parent = key.split("/")[0]
            if parent in sections:
                parent_text = sections[parent]
                sub_name = key.split("/")[1].replace("_", " ")
                idx = parent_text.lower().find(sub_name)
                if idx >= 0:
                    next_h3 = parent_text.find("\n### ", idx + 1)
                    next_h2 = parent_text.find("\n## ", idx + 1)
                    end = min(
                        next_h3 if next_h3 > 0 else len(parent_text),
                        next_h2 if next_h2 > 0 else len(parent_text),
                    )
                    sections[key] = parent_text[idx:end].strip()

    _SPEC_CACHE.update(sections)
    return _SPEC_CACHE


# ── HTTP helper ──────────────────────────────────────────────────────────────

async def _http_post(url: str, body: dict, timeout: int = 30) -> dict:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=body,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                data = await resp.json()
                if resp.status >= 400:
                    return {"_error": data.get("error", f"HTTP {resp.status}"), **data}
                return data
    except aiohttp.ClientError as e:
        return {"_error": f"Service unavailable: {e}"}
    except Exception as e:
        return {"_error": str(e)}


async def _http_get(url: str, timeout: int = 10) -> dict:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                data = await resp.json()
                if resp.status >= 400:
                    return {"_error": data.get("error", f"HTTP {resp.status}"), **data}
                return data
    except aiohttp.ClientError as e:
        return {"_error": f"Service unavailable: {e}"}
    except Exception as e:
        return {"_error": str(e)}


# ═════════════════════════════════════════════════════════════════════════════
# TOOL: nml_spec_lookup
# ═════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def nml_spec_lookup(
    section: str = "overview",
    query: str = "",
) -> str:
    """Look up NML language specification sections.

    Returns authoritative NML spec content for generating correct NML code.

    Args:
        section: Section to retrieve. Options:
            "overview" — Design principles and register table
            "instruction_set" — All 62 opcodes grouped by category
            "arithmetic" — MMUL, MADD, MSUB, EMUL, EDIV, SDOT, SCLR, SDIV
            "activation" — RELU, SIGM, TANH, SOFT
            "memory" — LD, ST, MOV, ALLC
            "comparison" — CMPF, CMP, CMPI
            "control_flow" — JMPT, JMPF, JUMP, LOOP, ENDP
            "tree_model" — LEAF, TACC (bracket tax patterns)
            "symbolic_syntax" — Unicode opcode + Greek register tables
            "verbose_syntax" — Human-readable opcode aliases
            "data_types" — f32, f64, i32 type system
            "data_file_format" — .nml.data format
            "encoding_format" — 32-bit instruction encoding
            "error_handling" — Error codes
            "examples" — Example NML programs
            "m2m" — META, FRAG, LINK, VOTE, SIGN, etc.
            "registers" — Register table with purposes
            "sections" — List all available section names
            "full" — Complete spec (large)
        query: Optional keyword filter within the section.
    """
    specs = _load_spec_sections()

    if section == "sections":
        return "Available sections:\n" + "\n".join(f"  - {k}" for k in sorted(specs) if k != "full")

    section_map = {
        "arithmetic": "instruction_set_(35_core_+_14_extensions_+_11_m2m_=_60_total)",
        "activation": "instruction_set_(35_core_+_14_extensions_+_11_m2m_=_60_total)",
        "memory": "instruction_set_(35_core_+_14_extensions_+_11_m2m_=_60_total)",
        "comparison": "instruction_set_(35_core_+_14_extensions_+_11_m2m_=_60_total)",
        "control_flow": "instruction_set_(35_core_+_14_extensions_+_11_m2m_=_60_total)",
        "tree_model": "instruction_set_(35_core_+_14_extensions_+_11_m2m_=_60_total)",
        "instruction_set": "instruction_set_(35_core_+_14_extensions_+_11_m2m_=_60_total)",
        "registers": "registers",
        "symbolic_syntax": "symbolic_syntax_(dual-syntax_mode)",
        "verbose_syntax": "symbolic_syntax_(dual-syntax_mode)",
        "data_types": "data_types_(v0.5)",
        "data_file_format": "data_file_format_(.nml.data)",
        "encoding_format": "encoding_format",
        "error_handling": "error_handling",
        "examples": "example_programs",
        "m2m": "extension:_nml-m2m_—_machine-to-machine_(11_instructions)",
    }

    key = section_map.get(section, section)
    content = specs.get(key, "")

    if not content:
        matches = [k for k in specs if section.lower() in k.lower()]
        if matches:
            content = specs[matches[0]]
        else:
            return f"Section '{section}' not found. Use section='sections' to list available sections."

    if query:
        lines = content.split("\n")
        filtered = []
        for i, line in enumerate(lines):
            if query.lower() in line.lower():
                start = max(0, i - 2)
                end = min(len(lines), i + 3)
                filtered.extend(lines[start:end])
                filtered.append("---")
        if filtered:
            content = "\n".join(filtered)
        else:
            content += f"\n\n(No lines matched query '{query}')"

    if len(content) > 12000:
        content = content[:12000] + "\n\n... [truncated — use a more specific section or query]"

    return content


# ═════════════════════════════════════════════════════════════════════════════
# TOOL: nml_transpile
# ═════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def nml_transpile(
    jurisdiction_key: str,
    pay_date: str = "2025-01-01",
    syntax: str = "classic",
) -> str:
    """Transpile a tax jurisdiction from JSON rules to an NML program.

    Requires the transpiler service on port 8083 (start with start_agents.sh).

    Args:
        jurisdiction_key: Tax jurisdiction ID, e.g. "00-000-0000-FIT-000" (federal income tax),
            "06-000-0000-SIT-000" (California state income tax). Format: SS-CCC-FFFFFFF-TYPE-VVV
            where SS=state FIPS, CCC=county, FFFFFFF=feature, TYPE=tax type, VVV=variant.
        pay_date: Effective date for tax brackets (YYYY-MM-DD).
        syntax: Output syntax — "classic" (LEAF RA #100), "symbolic" (∎ α #100), or "verbose" (SET_VALUE ACCUMULATOR #100).
    """
    result = await _http_post(f"{TRANSPILER_URL}/transpile", {
        "jurisdiction_key": jurisdiction_key,
        "pay_date": pay_date,
        "syntax": syntax,
    })

    if "_error" in result:
        return f"Error: {result['_error']}"

    nml = result.get("nml_program", "")
    meta = (
        f"Jurisdiction: {result.get('jurisdiction_key', jurisdiction_key)}\n"
        f"Tax type: {result.get('tax_type', 'unknown')}\n"
        f"Instructions: {result.get('instruction_count', '?')}\n"
        f"Syntax: {syntax}\n"
        f"Pay date: {pay_date}\n"
        f"---\n"
    )
    return meta + nml


# ═════════════════════════════════════════════════════════════════════════════
# TOOL: nml_validate
# ═════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def nml_validate(
    nml_program: str,
    jurisdiction_key: str = "",
    tax_type: str = "",
    mode: str = "full",
) -> str:
    """Validate an NML program for correctness.

    Runs grammar checking, semantic analysis, and optional execution testing.
    Requires the validation service on port 8084.

    Args:
        nml_program: The NML source code to validate.
        jurisdiction_key: Optional jurisdiction for execution testing (e.g. "00-000-0000-FIT-000").
        tax_type: Optional tax type hint for semantic validation (e.g. "FIT", "SIT", "FICA").
        mode: Validation mode — "full" (grammar+semantic+execution), "quick" (grammar only).
    """
    if mode == "quick":
        try:
            from nml_grammar import validate_grammar
            report = validate_grammar(nml_program)
            result = report.to_dict() if hasattr(report, "to_dict") else report
            return json.dumps(result, indent=2)
        except ImportError:
            return "Grammar validator not available locally. Use mode='full' with the validation service."

    body: dict = {"nml_program": nml_program}
    if jurisdiction_key:
        body["jurisdiction_key"] = jurisdiction_key
    if tax_type:
        body["tax_type"] = tax_type

    result = await _http_post(f"{VALIDATOR_URL}/validate/full", body)

    if "_error" in result:
        return f"Error: {result['_error']}"

    parts = []
    overall = result.get("overall_status", "unknown")
    parts.append(f"Overall: {overall.upper()}")

    grammar = result.get("grammar", {})
    if grammar:
        valid = grammar.get("valid", False)
        errors = grammar.get("errors", [])
        warnings = grammar.get("warnings", [])
        parts.append(f"\nGrammar: {'PASS' if valid else 'FAIL'}")
        for e in errors:
            parts.append(f"  ERROR line {e.get('line', '?')}: {e.get('message', e.get('type', ''))}")
        for w in warnings:
            parts.append(f"  WARN: {w}")

    semantic = result.get("semantic", {})
    if semantic:
        valid = semantic.get("valid", False)
        errors = semantic.get("errors", [])
        parts.append(f"\nSemantic: {'PASS' if valid else 'FAIL'}")
        for e in errors:
            parts.append(f"  ERROR: {e.get('message', e.get('type', ''))}")

    execution = result.get("execution", {})
    if execution:
        status = execution.get("status", "skipped")
        parts.append(f"\nExecution: {status.upper()}")
        if status == "pass":
            outputs = execution.get("outputs", {})
            for k, v in outputs.items():
                parts.append(f"  {k} = {v}")
            if execution.get("cycles"):
                parts.append(f"  Cycles: {execution['cycles']}")
            if execution.get("time_us"):
                parts.append(f"  Time: {execution['time_us']}µs")
        elif status == "fail":
            parts.append(f"  Error: {execution.get('error', 'unknown')}")

    return "\n".join(parts)


# ═════════════════════════════════════════════════════════════════════════════
# TOOL: nml_execute
# ═════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def nml_execute(
    nml_program: str,
    data: str = "",
    trace: bool = False,
    max_cycles: int = 0,
) -> str:
    """Execute an NML program and return outputs.

    Runs the NML C runtime against the given program and input data.
    Requires the execution service on port 8085.

    Args:
        nml_program: The NML source code to execute.
        data: Input data in NML .nml.data format. Each line: "@name shape=S data=V1,V2,...".
            For tax programs, simplified format also works: "gross_pay 3846.15\\nfiling_status 1".
        trace: If true, return execution trace (each instruction as it executes).
        max_cycles: Override cycle limit (0 = use default 1M).
    """
    body: dict = {
        "nml_program": nml_program,
        "data": data,
        "trace": trace,
    }
    if max_cycles > 0:
        body["max_cycles"] = max_cycles

    endpoint = "/execute/trace" if trace else "/execute"
    result = await _http_post(f"{ENGINE_URL}{endpoint}", body)

    if "_error" in result:
        return f"Error: {result['_error']}"

    parts = []
    status = result.get("status", "unknown")
    parts.append(f"Status: {status}")

    outputs = result.get("outputs", {})
    if outputs:
        parts.append("Outputs:")
        for k, v in outputs.items():
            parts.append(f"  {k} = {v}")

    if result.get("cycles"):
        parts.append(f"Cycles: {result['cycles']}")
    if result.get("time_us"):
        parts.append(f"Time: {result['time_us']}µs")

    if trace and result.get("trace"):
        trace_lines = result["trace"]
        parts.append(f"\nTrace ({len(trace_lines)} lines):")
        for line in trace_lines[-50:]:
            parts.append(f"  {line}")
        if len(trace_lines) > 50:
            parts.insert(-50, f"  ... ({len(trace_lines) - 50} earlier lines omitted)")

    if result.get("stderr"):
        parts.append(f"\nStderr: {result['stderr']}")

    return "\n".join(parts)


# ═════════════════════════════════════════════════════════════════════════════
# TOOL: nml_library_lookup
# ═════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def nml_library_lookup(
    tax_type: str = "",
    jurisdiction_key: str = "",
    syntax: str = "symbolic",
    list_types: bool = False,
) -> str:
    """Retrieve golden NML examples from the pre-built library.

    These are pre-transpiled, validated NML programs that serve as reference
    implementations. Use these as few-shot examples when generating new NML.

    Args:
        tax_type: Tax type directory to browse (e.g. "FIT", "SIT", "FICA", "CITY", "EIC").
        jurisdiction_key: Specific jurisdiction to retrieve (e.g. "00-000-0000-FIT-000").
        syntax: Library syntax — "symbolic" (default) or "classic".
        list_types: If true, list all available tax type directories and file counts.
    """
    lib_dir = LIBRARY_DIR if syntax == "symbolic" else LIBRARY_CLASSIC_DIR

    if not lib_dir.exists():
        return f"Library not found at {lib_dir}. Run 'make transpile-library' to build."

    if list_types:
        types = []
        for d in sorted(lib_dir.iterdir()):
            if d.is_dir() and not d.name.startswith("."):
                count = sum(1 for f in d.iterdir() if f.suffix == ".nml")
                types.append(f"  {d.name}: {count} files")
        return f"Available tax types ({len(types)} total):\n" + "\n".join(types)

    if jurisdiction_key:
        for d in lib_dir.iterdir():
            if d.is_dir():
                nml_file = d / f"{jurisdiction_key}.nml"
                if nml_file.exists():
                    content = nml_file.read_text()
                    return (
                        f"Library: {jurisdiction_key} ({syntax} syntax)\n"
                        f"Lines: {len(content.splitlines())}\n"
                        f"---\n{content}"
                    )
        return f"Jurisdiction '{jurisdiction_key}' not found in {syntax} library."

    if tax_type:
        type_dir = lib_dir / tax_type.upper()
        if not type_dir.exists():
            available = [d.name for d in lib_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
            return f"Tax type '{tax_type}' not found. Available: {', '.join(sorted(available))}"

        files = sorted(f for f in type_dir.iterdir() if f.suffix == ".nml")
        if not files:
            return f"No NML files in {tax_type}."

        parts = [f"Tax type: {tax_type.upper()} ({len(files)} files in {syntax} library)"]

        for f in files[:3]:
            content = f.read_text()
            lines = content.splitlines()
            parts.append(f"\n--- {f.stem} ({len(lines)} lines) ---")
            parts.append(content)

        if len(files) > 3:
            parts.append(f"\n... and {len(files) - 3} more files")
            parts.append("Files: " + ", ".join(f.stem for f in files[3:10]))
            if len(files) > 10:
                parts.append(f"  ... ({len(files) - 10} more)")

        return "\n".join(parts)

    return (
        "Specify tax_type (e.g. 'FIT'), jurisdiction_key (e.g. '00-000-0000-FIT-000'), "
        "or list_types=true to browse the library."
    )


# ═════════════════════════════════════════════════════════════════════════════
# TOOL: nml_scan
# ═════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def nml_scan() -> str:
    """Scan the tax data directory and return jurisdiction counts by type and pattern.

    Lists how many tax jurisdictions are available and their types.
    Tries the transpiler service first, falls back to local scan.
    """
    result = await _http_get(f"{TRANSPILER_URL}/transpile/scan")

    if "_error" not in result:
        parts = [f"Total jurisdictions: {result.get('jurisdictions', '?')}"]
        tax_types = result.get("tax_types", {})
        if tax_types:
            parts.append("\nBy tax type:")
            for tt, count in sorted(tax_types.items(), key=lambda x: -x[1]):
                parts.append(f"  {tt}: {count}")
        patterns = result.get("by_pattern", {})
        if patterns:
            parts.append("\nBy pattern:")
            for pat, count in sorted(patterns.items(), key=lambda x: -x[1]):
                parts.append(f"  {pat}: {count}")
        return "\n".join(parts)

    try:
        from ste_transpiler import scan_tax_data
        from collections import Counter

        files = scan_tax_data(TAX_DATA_DIR)
        tax_types = Counter(f.tax_type for f in files)
        parts = [f"Total jurisdictions: {len(files)} (local scan)"]
        parts.append("\nBy tax type:")
        for tt, count in tax_types.most_common():
            parts.append(f"  {tt}: {count}")
        return "\n".join(parts)
    except Exception as e:
        return f"Scan failed: {e}"


# ═════════════════════════════════════════════════════════════════════════════
# TOOL: nml_intent
# ═════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def nml_intent(message: str) -> str:
    """Classify a natural-language message into a structured NML intent.

    Extracts jurisdiction keys, tax types, dollar amounts, filing status,
    and categorizes the request (run_calculation, explain, validate, update, etc.).

    Args:
        message: The natural-language request to classify.
    """
    try:
        from intent_router import classify_intent, resolve_jurisdiction_key

        intent = classify_intent(message)
        result: dict = {
            "category": intent.category,
            "confidence": intent.confidence,
        }

        if intent.jurisdiction_key:
            result["jurisdiction_key"] = intent.jurisdiction_key
        elif intent.tax_type or intent.state:
            resolved = resolve_jurisdiction_key(intent.tax_type, intent.state)
            if resolved:
                result["jurisdiction_key"] = resolved
                result["resolved_from"] = f"tax_type={intent.tax_type}, state={intent.state}"

        if intent.tax_type:
            result["tax_type"] = intent.tax_type
        if intent.state:
            result["state"] = intent.state
        if intent.tax_year:
            result["tax_year"] = intent.tax_year
        if intent.parameters:
            result["parameters"] = intent.parameters

        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Intent classification failed: {e}"


# ═════════════════════════════════════════════════════════════════════════════
# RESOURCES
# ═════════════════════════════════════════════════════════════════════════════

@mcp.resource("nml://spec")
def get_spec() -> str:
    """The complete NML specification document."""
    spec_path = DOCS_DIR / "NML_SPEC.md"
    if spec_path.exists():
        return spec_path.read_text()
    return "Spec not found."


@mcp.resource("nml://spec/m2m")
def get_m2m_spec() -> str:
    """The NML M2M (Machine-to-Machine) specification."""
    m2m_path = DOCS_DIR / "NML_M2M_Spec.md"
    if m2m_path.exists():
        return m2m_path.read_text()
    return "M2M spec not found."


@mcp.resource("nml://opcodes/symbolic")
def get_symbolic_opcodes() -> str:
    """Quick-reference table of all symbolic opcodes and Greek register aliases."""
    return """NML Symbolic Opcode Quick Reference
═══════════════════════════════════

Arithmetic:  ×(MMUL) ⊕(MADD) ⊖(MSUB) ⊗(EMUL) ⊘(EDIV) ·(SDOT) ∗(SCLR) ÷(SDIV)
Activation:  ⌐(RELU) σ(SIGM) τ(TANH) Σ(SOFT)
Memory:      ↓(LD) ↑(ST) ←(MOV) □(ALLC)
Data Flow:   ⊞(RSHP) ⊤(TRNS) ⊢(SPLT) ⊣(MERG)
Comparison:  ⋈(CMPF) ≶(CMP) ≺(CMPI)
Control:     ↗(JMPT) ↘(JMPF) →(JUMP) ↻(LOOP) ↺(ENDP)
Tree:        ∎(LEAF) ∑(TACC)
Subroutine:  ⇒(CALL) ⇐(RET)
System:      ⏸(SYNC) ◼(HALT) ⚠(TRAP)
Vision:      ⊛(CONV) ⊓(POOL) ⊔(UPSC) ⊡(PADZ)
Transformer: ⊙(ATTN) ‖(NORM) ⊏(EMBD) ℊ(GELU)
Reduction:   ⊥(RDUC) ⊻(WHER) ⊧(CLMP) ⊜(CMPR)
Signal:      ∿(FFT) ⋐(FILT)
M2M:         META FRAG ENDF LINK PTCH SIGN VRFY VOTE PROJ DIST GATH SCAT

Greek Registers:
  ι(R0) κ(R1) λ(R2) μ(R3) ν(R4) ξ(R5) ο(R6) π(R7) ρ(R8) ς(R9)
  α(RA/Accumulator) β(RB) γ(RC/Scratch) δ(RD/Counter) φ(RE/Flag) ψ(RF/Stack)

Tax Bracket Pattern (symbolic):
  ⋈ φ π #0 #THRESHOLD     ; compare income to threshold
  ↘ #2                     ; skip if above
  ∎ α #BASE_TAX            ; set base tax
  → #N                     ; jump to store
  ∎ γ #THRESHOLD           ; load threshold
  ⊖ ρ π γ                  ; marginal = income - threshold
  ∗ ρ ρ #RATE              ; marginal * rate
  ∑ α α ρ                  ; accumulate

Flat-Rate Pattern (symbolic):
  □ α #[1]                 ; allocate accumulator
  ∗ γ ι #RATE              ; wages * rate
  ∑ α α γ                  ; accumulate
  ↑ α @tax_amount          ; store result
  ◼                        ; halt
"""


# ═════════════════════════════════════════════════════════════════════════════
# TOOL: nml_compact
# ═════════════════════════════════════════════════════════════════════════════

COMPACT_DELIM = "\u00b6"  # ¶


@mcp.tool()
def nml_compact(nml_program: str) -> str:
    """Convert a multi-line NML program to single-line compact form.

    Replaces newlines with ¶ (U+00B6, pilcrow) delimiters, strips comments,
    normalizes whitespace. The compact form is directly executable
    by the NML runtime.

    Args:
        nml_program: Multi-line NML source code.
    """
    instructions = []
    for line in nml_program.split("\n"):
        comment_pos = line.find(";")
        if comment_pos >= 0:
            line = line[:comment_pos]
        stripped = line.strip()
        if not stripped:
            continue
        import re
        normalized = re.sub(r"[ \t]+", " ", stripped)
        instructions.append(normalized)

    return COMPACT_DELIM.join(instructions)


# ═════════════════════════════════════════════════════════════════════════════
# TOOL: nml_format
# ═════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def nml_format(nml_program: str) -> str:
    """Format an NML program for human readability.

    Accepts compact (¶-delimited) or multi-line input. Outputs formatted
    multi-line NML with aligned columns.

    Args:
        nml_program: NML source in compact or multi-line form.
    """
    if COMPACT_DELIM in nml_program:
        parts = nml_program.split(COMPACT_DELIM)
    else:
        parts = nml_program.split("\n")

    instructions = [p.strip() for p in parts if p.strip()]

    SYMBOLIC = frozenset(
        "\u00d7 \u2295 \u2296 \u2297 \u2298 \u00b7 \u2217 \u00f7 "
        "\u2310 \u03c3 \u03c4 \u03a3 \u2193 \u2191 \u2190 \u25a1 "
        "\u229e \u22a4 \u22a2 \u22a3 \u22c8 \u2276 \u227a "
        "\u2197 \u2198 \u2192 \u21bb \u21ba \u220e \u2211 "
        "\u21d2 \u21d0 \u23f8 \u25fc \u26a0 "
        "\u229b \u2293 \u2294 \u22a1 \u2299 \u2016 \u228f \u210a "
        "\u22a5 \u22bb \u22a7 \u229c \u223f \u22d0 \u2699".split()
    )

    lines = []
    for instr in instructions:
        tokens = instr.split()
        if not tokens:
            continue
        opcode = tokens[0]
        if opcode.upper() == "META":
            lines.append("  ".join(tokens))
        elif opcode in SYMBOLIC:
            lines.append("  ".join(tokens) if len(tokens) > 1 else opcode)
        else:
            if len(tokens) > 1:
                lines.append(f"{opcode:<6}" + " ".join(tokens[1:]))
            else:
                lines.append(opcode)

    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NML MCP Server")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio")
    parser.add_argument("--port", type=int, default=9100)
    args = parser.parse_args()

    if args.transport == "sse":
        mcp.run(transport="sse", port=args.port)
    else:
        mcp.run(transport="stdio")
