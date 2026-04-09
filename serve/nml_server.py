#!/usr/bin/env python3
"""
NML Server — generic MCP tools + optional LLM chat for NML programs.

MCP Tools:
  nml_spec       — Look up NML spec sections (opcodes, registers, syntax)
  nml_validate   — Validate an NML program (grammar check)
  nml_execute    — Execute an NML program with the C runtime
  nml_format     — Convert between compact and multi-line NML

Optional LLM Chat:
  /v1/chat/completions  — OpenAI-compatible chat endpoint using local MLX model
  /v1/models            — List available models

Usage:
    # MCP mode (stdio, for Cursor/Claude)
    python3 nml_server.py --transport stdio

    # HTTP mode (for terminal UI, chat apps)
    python3 nml_server.py --http --port 8082

    # With LLM chat (requires mlx-lm)
    python3 nml_server.py --http --port 8082 --model path/to/model
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SERVE_DIR = Path(__file__).parent
NML_BINARY = PROJECT_ROOT / ("nml.exe" if sys.platform == "win32" else "nml")
SPEC_PATH = PROJECT_ROOT / "docs" / "NML_SPEC.md"
GETTING_STARTED = PROJECT_ROOT / "docs" / "GETTING_STARTED.md"
ML_ADVISOR_KB_PATH = SERVE_DIR / "ml_advisor_kb.json"

# ═══════════════════════════════════════════
# MCP Tool: Spec Lookup
# ═══════════════════════════════════════════

SPEC_SECTIONS = {}

def _load_spec():
    if SPEC_SECTIONS:
        return
    if not SPEC_PATH.exists():
        return
    current_section = "overview"
    current_lines = []
    with open(SPEC_PATH) as f:
        for line in f:
            if line.startswith("## "):
                if current_lines:
                    SPEC_SECTIONS[current_section] = "".join(current_lines).strip()
                current_section = line.strip("# \n").lower().replace(" ", "_")
                current_lines = [line]
            else:
                current_lines.append(line)
    if current_lines:
        SPEC_SECTIONS[current_section] = "".join(current_lines).strip()


def spec_lookup(query: str) -> str:
    """Look up NML spec sections by keyword.

    Args:
        query: Section name or keyword (e.g. 'registers', 'arithmetic',
               'symbolic', 'encoding', 'data_types', 'version_history')

    Returns:
        Matching spec section content.
    """
    _load_spec()
    q = query.lower().replace(" ", "_")

    if q in SPEC_SECTIONS:
        return SPEC_SECTIONS[q]

    matches = []
    for key, content in SPEC_SECTIONS.items():
        if q in key or q in content.lower():
            matches.append(f"### {key}\n{content[:500]}")

    if matches:
        return "\n\n".join(matches[:3])

    return f"No spec section matching '{query}'. Available: {', '.join(sorted(SPEC_SECTIONS.keys()))}"


# ═══════════════════════════════════════════
# MCP Tool: Validate
# ═══════════════════════════════════════════

def validate_program(nml_program: str) -> dict:
    """Validate an NML program using the grammar validator.

    Args:
        nml_program: NML source code to validate.

    Returns:
        Dict with 'valid', 'errors', 'warnings', and 'instruction_count'.
    """
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "transpilers"))
        import nml_grammar
        report = nml_grammar.validate_grammar(nml_program)
        return report.to_dict()
    except ImportError:
        return {"valid": True, "errors": [], "warnings": [],
                "note": "Grammar validator not available"}
    except Exception as e:
        return {"valid": False, "errors": [{"message": str(e)}], "warnings": []}


# ═══════════════════════════════════════════
# MCP Tool: Execute
# ═══════════════════════════════════════════

def execute_program(nml_program: str, data: str = "",
                    trace: bool = False, max_cycles: int = 100000) -> dict:
    """Execute an NML program using the C runtime.

    Args:
        nml_program: NML source code.
        data: Optional .nml.data content (memory definitions).
        trace: If True, include instruction-level trace.
        max_cycles: Maximum execution cycles (default 100000).

    Returns:
        Dict with 'status', 'outputs', 'cycles', 'time_us', and optionally 'trace'.
    """
    if not NML_BINARY.exists():
        return {"status": "error", "message": f"NML binary not found at {NML_BINARY}. Run 'make' first."}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".nml", delete=False) as f:
        f.write(nml_program)
        prog_path = f.name

    data_path = None
    if data.strip():
        with tempfile.NamedTemporaryFile(mode="w", suffix=".nml.data", delete=False) as f:
            f.write(data)
            data_path = f.name

    try:
        cmd = [str(NML_BINARY), prog_path]
        if data_path:
            cmd.append(data_path)
        if trace:
            cmd.append("--trace")
        cmd.extend(["--max-cycles", str(max_cycles)])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        outputs = {}
        cycles = None
        time_us = None
        trace_lines = []
        status = "error" if result.returncode != 0 else "ok"

        for line in result.stdout.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            m = re.match(r"(\w+):\s+shape=.*data=\[([^\]]+)\]", stripped)
            if m:
                name, vals_str = m.group(1), m.group(2).split(",")
                if len(vals_str) == 1:
                    outputs[name] = float(vals_str[0].strip())
                else:
                    outputs[name] = [float(v.strip()) for v in vals_str if v.strip() not in ("...", "")]
            if "HALTED" in stripped:
                status = "HALTED"
                cm = re.search(r"(\d+) cycles", stripped)
                if cm:
                    cycles = int(cm.group(1))
                tm = re.search(r"([\d.]+)\s*µs", stripped)
                if tm:
                    time_us = float(tm.group(1))
            if trace:
                trace_lines.append(stripped)

        out = {"status": status, "outputs": outputs}
        if cycles is not None:
            out["cycles"] = cycles
        if time_us is not None:
            out["time_us"] = time_us
        if trace:
            out["trace"] = trace_lines
        if result.returncode != 0:
            out["stderr"] = result.stderr.strip()[:500]
        return out

    except subprocess.TimeoutExpired:
        return {"status": "timeout", "outputs": {}}
    finally:
        os.unlink(prog_path)
        if data_path:
            os.unlink(data_path)


# ═══════════════════════════════════════════
# MCP Tool: Format
# ═══════════════════════════════════════════

def format_program(nml_program: str, mode: str = "compact") -> str:
    """Convert NML between compact (single-line) and multi-line format.

    Args:
        nml_program: NML source code.
        mode: 'compact' (multi-line to single-line with pilcrow)
              or 'expand' (compact to multi-line).

    Returns:
        Reformatted NML program.
    """
    if mode == "compact":
        lines = [l.split(";")[0].strip() for l in nml_program.split("\n")]
        lines = [l for l in lines if l]
        return "\u00b6".join(lines)
    else:
        parts = nml_program.split("\u00b6")
        return "\n".join(p.strip() for p in parts)


# ═══════════════════════════════════════════
# MCP Server (stdio transport)
# ═══════════════════════════════════════════

TOOLS = [
    {
        "name": "nml_spec",
        "description": "Look up NML specification sections by keyword (opcodes, registers, syntax, encoding, etc.)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Section name or keyword to search for"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "nml_validate",
        "description": "Validate an NML program for grammar correctness",
        "inputSchema": {
            "type": "object",
            "properties": {
                "nml_program": {"type": "string", "description": "NML source code to validate"}
            },
            "required": ["nml_program"],
        },
    },
    {
        "name": "nml_execute",
        "description": "Execute an NML program using the C runtime and return outputs",
        "inputSchema": {
            "type": "object",
            "properties": {
                "nml_program": {"type": "string", "description": "NML source code"},
                "data": {"type": "string", "description": "Optional .nml.data content", "default": ""},
                "trace": {"type": "boolean", "description": "Include instruction trace", "default": False},
            },
            "required": ["nml_program"],
        },
    },
    {
        "name": "nml_format",
        "description": "Convert NML between compact (pilcrow-delimited) and multi-line format",
        "inputSchema": {
            "type": "object",
            "properties": {
                "nml_program": {"type": "string", "description": "NML source code"},
                "mode": {"type": "string", "enum": ["compact", "expand"], "default": "compact"},
            },
            "required": ["nml_program"],
        },
    },
]


async def handle_mcp_stdio():
    """Run as MCP server over stdio."""
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)

    async def write(data):
        sys.stdout.write(json.dumps(data) + "\n")
        sys.stdout.flush()

    while True:
        line = await reader.readline()
        if not line:
            break
        try:
            msg = json.loads(line.decode())
        except json.JSONDecodeError:
            continue

        method = msg.get("method", "")
        msg_id = msg.get("id")
        params = msg.get("params", {})

        if method == "initialize":
            await write({"jsonrpc": "2.0", "id": msg_id, "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "nml-server", "version": "0.6.4"},
            }})
        elif method == "tools/list":
            await write({"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}})
        elif method == "tools/call":
            tool_name = params.get("name", "")
            args = params.get("arguments", {})
            result = _dispatch_tool(tool_name, args)
            await write({"jsonrpc": "2.0", "id": msg_id, "result": {
                "content": [{"type": "text", "text": result if isinstance(result, str) else json.dumps(result, indent=2)}]
            }})
        elif method == "notifications/initialized":
            pass
        else:
            if msg_id:
                await write({"jsonrpc": "2.0", "id": msg_id, "error": {
                    "code": -32601, "message": f"Unknown method: {method}"
                }})


def _dispatch_tool(name: str, args: dict):
    if name == "nml_spec":
        return spec_lookup(args.get("query", ""))
    elif name == "nml_validate":
        return validate_program(args.get("nml_program", ""))
    elif name == "nml_execute":
        return execute_program(
            args.get("nml_program", ""),
            data=args.get("data", ""),
            trace=args.get("trace", False),
        )
    elif name == "nml_format":
        return format_program(args.get("nml_program", ""), mode=args.get("mode", "compact"))
    return {"error": f"Unknown tool: {name}"}


# ═══════════════════════════════════════════
# ML Advisor — algorithm selection via KB + high-reasoning LLM
# ═══════════════════════════════════════════

_advisor_kb = None

def _load_advisor_kb():
    global _advisor_kb
    if _advisor_kb is not None:
        return _advisor_kb
    if ML_ADVISOR_KB_PATH.exists():
        with open(ML_ADVISOR_KB_PATH) as f:
            _advisor_kb = json.load(f)
    else:
        _advisor_kb = {}
    return _advisor_kb


def advisor_match_problem(description: str) -> list:
    """Score each problem type against the user description using keyword signals."""
    kb = _load_advisor_kb()
    desc_lower = description.lower()
    scored = []
    for pt in kb.get("problem_types", []):
        hits = sum(1 for s in pt.get("signals", []) if s in desc_lower)
        if hits > 0:
            scored.append((hits, pt))
    scored.sort(key=lambda x: -x[0])
    return [pt for _, pt in scored[:3]]


def advisor_build_context(description: str) -> str:
    """Build a rich system prompt grounding the advisor in the KB."""
    kb = _load_advisor_kb()
    matches = advisor_match_problem(description)

    parts = [
        "You are an ML Advisor for the NML (Neural Machine Language) platform.",
        "NML is an 89-opcode tensor register machine. The runtime is a small C binary;",
        "programs and data are separate files with no inherent size limit.",
        "Your job: given a real-world problem description, recommend the best ML algorithm,",
        "explain WHY it fits, describe the data pipeline, and show how to implement it in NML.",
        "",
        "Your response MUST include these sections in order:",
        "",
        "1. PROBLEM TYPE — classify the problem (regression, binary classification, etc.)",
        "",
        "2. WHAT YOU'RE DOING AND WHY",
        "   Explain in plain language: what is the goal, what goes in, what comes out.",
        "   Example: 'You are training a model to learn what normal credit card transactions",
        "   look like, so that when a new transaction arrives, the model can score how",
        "   different it is from normal — a high score means likely fraud.'",
        "",
        "3. DATA REQUIREMENTS",
        "   For each tensor the model needs, explain:",
        "   - What real-world data it represents (not just '@w1 shape=4,8')",
        "   - Why this data is needed for the algorithm",
        "   - Shape and what each dimension means (rows = samples, cols = features, etc.)",
        "   - Example values with units where applicable",
        "   Group into: INPUT DATA (what the user provides), MODEL WEIGHTS (initialized by",
        "   the system), and LABELS/TARGETS (for supervised training).",
        "   Show the complete .nml.data file with inline comments explaining every tensor.",
        "",
        "4. RECOMMENDED ALGORITHM — name and 1-sentence why",
        "",
        "5. WHY THIS FITS — 2-3 sentences on why this algorithm matches the constraints",
        "",
        "6. ALTERNATIVES — 1-2 other options with trade-offs",
        "",
        "7. NML IMPLEMENTATION — specific opcodes, register layout, tensor shapes.",
        "   Show the data flow: which tensor loads into which register, what operations",
        "   transform it, and where the output lands.",
        "",
        "8. LESSON REFERENCE — which ML Journey lesson covers this",
        "",
        "The output feeds into a Think model that plans the NML architecture, and a user",
        "who needs to prepare their .nml.data file. Be concrete about dimensions and data.",
    ]

    if matches:
        parts.append("")
        parts.append("=== RELEVANT KNOWLEDGE BASE ENTRIES ===")
        for pt in matches:
            parts.append(f"\n## {pt['name']}: {pt['description']}")
            for alg in pt.get("algorithms", []):
                parts.append(f"  - {alg['name']}: {alg['when']}")
                parts.append(f"    NML: {alg['nml_pattern']}")
                parts.append(f"    Opcodes: {', '.join(alg['nml_opcodes'])}")
                parts.append(f"    Lesson: {alg['lesson_ref']}")
                if alg.get("sample"):
                    parts.append(f"    Sample: {alg['sample']}")

    heuristics = kb.get("decision_heuristics", [])
    if heuristics:
        parts.append("")
        parts.append("=== DECISION HEURISTICS ===")
        for h in heuristics:
            parts.append(f"  IF {h['condition']}:")
            parts.append(f"    → {h['recommendation']}")
            parts.append(f"    AVOID: {h['avoid']}")

    cap = kb.get("nml_capability_map", {})
    if cap:
        parts.append("")
        parts.append("=== NML CAPABILITIES ===")
        train = cap.get("supervised_training", {})
        if train:
            parts.append(f"  Config-driven training: {train.get('config_driven', '')}")
            parts.append(f"  Manual loop: {train.get('manual_loop', '')}")
            lt = train.get("loss_types", {})
            parts.append(f"  Loss types: {', '.join(f'{k}={v}' for k,v in lt.items())}")
        inf = cap.get("inference_only", {})
        if inf:
            parts.append(f"  Inference: {inf.get('infer_opcode', '')}")
        reg = cap.get("regularization", {})
        if reg:
            for k, v in reg.items():
                parts.append(f"  {k}: {v}")
        ds = cap.get("data_shapes", {})
        if ds:
            parts.append(f"  Data: {ds.get('training_data', '')}; Labels: {ds.get('labels', '')}")

    dg = cap.get("data_guide", {})
    if dg:
        parts.append("")
        parts.append("=== DATA GUIDE ===")
        parts.append(f"  Format: {dg.get('format', '')}")
        parts.append(f"  TNET shortcut: {dg.get('tnet_shortcut', '')}")
        cats = dg.get("tensor_categories", {})
        for cat_name, cat_info in cats.items():
            parts.append(f"  {cat_name}: {cat_info.get('purpose', '')}")
            for ex in cat_info.get("examples", []):
                parts.append(f"    Example: {ex}")
            if cat_info.get("notes"):
                parts.append(f"    Note: {cat_info['notes']}")
            if cat_info.get("naming"):
                parts.append(f"    Naming: {cat_info['naming']}")
        example_file = dg.get("data_file_example", "")
        if example_file:
            parts.append(f"  Example .nml.data file:\n    {example_file}")

    return "\n".join(parts)


def advisor_local_only(description: str) -> dict:
    """Return KB-grounded advice without calling an external LLM."""
    matches = advisor_match_problem(description)
    if not matches:
        return {
            "problem_type": "unknown",
            "recommendation": "Could not match problem type from description. Try being more specific about what you want to predict or detect.",
            "algorithms": [],
            "source": "kb_only",
        }

    primary = matches[0]
    algs = primary.get("algorithms", [])
    best = algs[0] if algs else None

    return {
        "problem_type": primary["name"],
        "description": primary["description"],
        "recommendation": best["name"] if best else "unknown",
        "why": best["when"] if best else "",
        "nml_pattern": best["nml_pattern"] if best else "",
        "nml_opcodes": best["nml_opcodes"] if best else [],
        "lesson_ref": best["lesson_ref"] if best else "",
        "sample": best.get("sample"),
        "alternatives": [
            {"name": a["name"], "when": a["when"], "complexity": a["complexity"]}
            for a in algs[1:3]
        ],
        "source": "kb_only",
    }


# ═══════════════════════════════════════════
# HTTP Server (for terminal UI + LLM chat)
# ═══════════════════════════════════════════

def create_http_app(model_path: str = None, advisor_llm: str = None, advisor_model: str = None, advisor_max_tokens: int = 4096):
    """Create an aiohttp app with MCP tool endpoints + optional LLM chat."""
    from aiohttp import web

    def _cors():
        return {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        }

    def _json(data, status=200):
        return web.Response(text=json.dumps(data), content_type="application/json",
                            status=status, headers=_cors())

    model = None
    tokenizer = None
    outlines_model = None
    nml_cfg = None
    llm_backend_url = None  # URL for external LLM server (e.g. llama-server)

    if model_path:
        # Check if model_path is a URL (external LLM server like llama-server)
        if model_path.startswith("http://") or model_path.startswith("https://"):
            llm_backend_url = model_path.rstrip("/")
            print(f"LLM backend (proxy): {llm_backend_url}")
        else:
            try:
                from mlx_lm import load
                print(f"Loading model: {model_path}")
                model, tokenizer = load(model_path)
                print(f"Model loaded: {model_path}")
                try:
                    import outlines
                    from outlines.types import CFG
                    grammar_path = PROJECT_ROOT / "transpilers" / "nml_lark_grammar.py"
                    if grammar_path.exists():
                        sys.path.insert(0, str(grammar_path.parent))
                        from nml_lark_grammar import NML_GRAMMAR
                        outlines_model = outlines.from_mlxlm(model, tokenizer)
                        nml_cfg = CFG(NML_GRAMMAR)
                        print(f"Constrained decoding: enabled (Outlines + NML CFG)")
                except ImportError:
                    print(f"Constrained decoding: disabled (install outlines[mlxlm] llguidance)")
                except Exception as e:
                    print(f"Constrained decoding: disabled ({e})")
            except ImportError:
                print(f"WARNING: mlx_lm not available (not macOS?). Use --model http://host:port to proxy to an external LLM server.")
            except Exception as e:
                print(f"WARNING: Could not load model: {e}")

    ADVISOR_SHORTHANDS = {
        "anthropic": "https://api.anthropic.com",
        "openai": "https://api.openai.com",
        "openrouter": "https://openrouter.ai/api",
    }
    advisor_llm_url = ADVISOR_SHORTHANDS.get(advisor_llm, advisor_llm) if advisor_llm else None
    advisor_default_model = (
        advisor_model
        or os.environ.get("NML_ADVISOR_MODEL")
        or None  # will pick per-provider default below
    )
    if advisor_llm_url:
        is_anthropic_default = "anthropic" in advisor_llm_url
        if not advisor_default_model:
            advisor_default_model = "claude-opus-4-20250514" if is_anthropic_default else "openai/gpt-4o"
        print(f"ML Advisor LLM: {advisor_llm_url}  model={advisor_default_model}")
    else:
        print(f"ML Advisor: KB-only mode (no --advisor-llm set)")

    async def handle_advise(request):
        """ML Advisor endpoint — recommend algorithm for a problem description.

        POST /advise
        Body: {"description": "I have customer data and want to predict churn"}
        Optional: {"description": "...", "llm": true}  to force LLM call

        Returns structured advice: problem_type, algorithm, why, NML pattern, lesson refs.
        """
        body = await request.json()
        description = body.get("description", "").strip()
        if not description:
            return _json({"error": "description is required"}, 400)

        use_llm = body.get("llm", advisor_llm_url is not None)

        if use_llm and advisor_llm_url:
            system_prompt = advisor_build_context(description)
            user_msg = (
                f"Problem: {description}\n\n"
                "Analyze this problem and recommend the best ML approach for NML. "
                "Be specific about NML opcodes, tensor shapes, and register layout."
            )

            from aiohttp import ClientSession, ClientTimeout
            try:
                # Detect provider from URL
                is_anthropic = "anthropic" in advisor_llm_url

                req_model = body.get("model", advisor_default_model)
                req_max_tokens = body.get("max_tokens", advisor_max_tokens)

                if is_anthropic:
                    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
                    payload = {
                        "model": req_model,
                        "max_tokens": req_max_tokens,
                        "system": system_prompt,
                        "messages": [{"role": "user", "content": user_msg}],
                    }
                    headers = {
                        "Content-Type": "application/json",
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                    }
                    endpoint = advisor_llm_url.rstrip("/")
                    if not endpoint.endswith("/v1/messages"):
                        endpoint += "/v1/messages"
                else:
                    api_key = os.environ.get("OPENAI_API_KEY", os.environ.get("OPENROUTER_API_KEY", ""))
                    payload = {
                        "model": req_model,
                        "max_tokens": req_max_tokens,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_msg},
                        ],
                    }
                    headers = {"Content-Type": "application/json"}
                    if api_key:
                        headers["Authorization"] = f"Bearer {api_key}"
                    endpoint = advisor_llm_url.rstrip("/")
                    if "/chat/completions" not in endpoint:
                        endpoint += "/v1/chat/completions"

                async with ClientSession(timeout=ClientTimeout(total=120)) as session:
                    async with session.post(endpoint, json=payload, headers=headers) as resp:
                        data = await resp.json()

                if is_anthropic:
                    content_blocks = data.get("content", [])
                    advice_text = "\n".join(
                        b.get("text", "") for b in content_blocks if b.get("type") == "text"
                    )
                else:
                    advice_text = data.get("choices", [{}])[0].get(
                        "message", {}).get("content", "")

                kb_matches = advisor_match_problem(description)
                return _json({
                    "advice": advice_text,
                    "problem_types_matched": [m["name"] for m in kb_matches],
                    "source": "llm",
                    "model": payload.get("model", "unknown"),
                })

            except Exception as e:
                print(f"  [advisor] LLM call failed: {e}, falling back to KB")
                result = advisor_local_only(description)
                result["llm_error"] = str(e)
                return _json(result)
        else:
            result = advisor_local_only(description)
            return _json(result)

    async def handle_options(request):
        return web.Response(status=204, headers=_cors())

    async def handle_health(request):
        return _json({
            "status": "healthy",
            "service": "nml-server",
            "tools": [t["name"] for t in TOOLS],
            "model": model_path if (model or llm_backend_url) else None,
            "constrained_decoding": outlines_model is not None,
            "advisor": {
                "available": True,
                "mode": "llm" if advisor_llm_url else "kb_only",
                "llm_url": advisor_llm_url,
            },
        })

    async def handle_models(request):
        if llm_backend_url:
            from aiohttp import ClientSession, ClientTimeout
            try:
                async with ClientSession(timeout=ClientTimeout(total=5)) as session:
                    async with session.get(f"{llm_backend_url}/v1/models") as resp:
                        data = await resp.json()
                        return _json(data)
            except Exception as e:
                print(f"  [proxy] /v1/models failed: {e}")
                return _json({"object": "list", "data": []})
        models = []
        if model:
            name = Path(model_path).name if model_path else "unknown"
            models.append({"id": name, "object": "model", "owned_by": "local"})
        return _json({"object": "list", "data": models})

    async def handle_validate(request):
        body = await request.json()
        result = validate_program(body.get("nml_program", ""))
        return _json(result)

    async def handle_execute(request):
        body = await request.json()
        result = execute_program(
            body.get("nml_program", ""),
            data=body.get("data", ""),
            trace=body.get("trace", False),
        )
        return _json(result)

    async def handle_format(request):
        body = await request.json()
        result = format_program(body.get("nml_program", ""), mode=body.get("mode", "compact"))
        return _json({"result": result})

    async def handle_spec(request):
        body = await request.json()
        result = spec_lookup(body.get("query", ""))
        return _json({"result": result})

    async def handle_chat(request):
        # Proxy mode: forward to external LLM server (e.g. llama-server)
        if llm_backend_url:
            from aiohttp import ClientSession
            body = await request.json()
            stream = body.get("stream", False)
            try:
                async with ClientSession() as session:
                    async with session.post(
                        f"{llm_backend_url}/v1/chat/completions",
                        json=body,
                    ) as upstream:
                        if stream:
                            resp = web.StreamResponse(headers={
                                **_cors(),
                                "Content-Type": "text/event-stream",
                                "Cache-Control": "no-cache",
                            })
                            await resp.prepare(request)
                            async for chunk in upstream.content.iter_any():
                                await resp.write(chunk)
                            return resp
                        else:
                            data = await upstream.json()
                            return _json(data)
            except Exception as e:
                return _json({"error": f"LLM backend error: {e}"}, 502)

        if not model or not tokenizer:
            return _json({"error": "No model loaded. Start with --model path/to/model or --model http://host:port"}, 503)

        body = await request.json()
        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens", 1024)
        stream = body.get("stream", False)
        mode = body.get("mode", "auto")
        constrained = body.get("constrained", None)

        if constrained is None:
            if mode == "nml":
                constrained = True
            elif mode == "chat":
                constrained = False
            elif mode == "auto":
                last_msg = messages[-1].get("content", "") if messages else ""
                nml_keywords = ["write nml", "nml program", "nml code", "symbolic nml",
                                "verbose nml", "generate nml", "nml to ", "nml for ",
                                "using nml", "in nml", "tnet", "mmul", "leaf r",
                                "include the .nml.data"]
                constrained = any(kw in last_msg.lower() for kw in nml_keywords)
            else:
                constrained = False

        if constrained and not outlines_model:
            constrained = False

        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        prompt_parts.append("<|im_start|>assistant\n")
        prompt = "\n".join(prompt_parts)

        loop = asyncio.get_event_loop()
        if constrained and outlines_model and nml_cfg:
            response_text = await loop.run_in_executor(
                None, lambda: outlines_model(prompt, output_type=nml_cfg, max_tokens=max_tokens))
        else:
            from mlx_lm import generate as mlx_generate
            response_text = await loop.run_in_executor(
                None, lambda: mlx_generate(model, tokenizer, prompt=prompt,
                                           max_tokens=max_tokens, verbose=False))

        if stream:
            async def stream_response(response):
                chunk = {
                    "id": "chatcmpl-nml",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"content": response_text}, "finish_reason": "stop"}],
                }
                await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
                await response.write(b"data: [DONE]\n\n")

            resp = web.StreamResponse(headers={**_cors(), "Content-Type": "text/event-stream"})
            await resp.prepare(request)
            await stream_response(resp)
            return resp

        return _json({
            "id": "chatcmpl-nml",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": response_text.strip()},
                "finish_reason": "stop",
            }],
            "constrained": constrained,
        })

    # ═══════════════════════════════════════════
    # M2M: Agent Fleet Distribution
    # ═══════════════════════════════════════════

    registered_agents = {}

    async def handle_agent_register(request):
        body = await request.json()
        name = body.get("name")
        url = body.get("url")
        capabilities = body.get("capabilities", [])
        if not name or not url:
            return _json({"error": "name and url required"}, 400)
        registered_agents[name] = {"url": url, "capabilities": capabilities}
        print(f"  [M2M] Agent registered: {name} at {url} ({', '.join(capabilities)})")
        return _json({"status": "registered", "agent": name})

    async def handle_agent_list(request):
        return _json({"agents": registered_agents})

    async def handle_distribute(request):
        """Distribute a signed program to all registered agents."""
        from aiohttp import ClientSession
        body = await request.json()
        program = body.get("program", "")
        data = body.get("data", None)

        if not program:
            return _json({"error": "No program provided"}, 400)
        if not registered_agents:
            return _json({"error": "No agents registered"}, 400)

        results = {}
        async with ClientSession() as session:
            tasks = []
            for name, info in registered_agents.items():
                payload = {"program": program}
                if data:
                    payload["data"] = data
                tasks.append((name, session.post(f"{info['url']}/execute", json=payload)))

            for name, coro in tasks:
                try:
                    resp = await coro
                    results[name] = await resp.json()
                except Exception as e:
                    results[name] = {"success": False, "error": str(e)}

        return _json({"distributed_to": list(registered_agents.keys()), "results": results})

    async def handle_vote(request):
        """Collect a numeric value from agent results and apply VOTE consensus."""
        body = await request.json()
        values = body.get("values", [])
        strategy = body.get("strategy", "median")

        if not values:
            return _json({"error": "No values provided"}, 400)

        nums = [float(v) for v in values if v is not None]
        if not nums:
            return _json({"error": "No numeric values"}, 400)

        if strategy == "median":
            nums.sort()
            mid = len(nums) // 2
            result = nums[mid] if len(nums) % 2 == 1 else (nums[mid-1] + nums[mid]) / 2
        elif strategy == "mean":
            result = sum(nums) / len(nums)
        elif strategy == "min":
            result = min(nums)
        elif strategy == "max":
            result = max(nums)
        else:
            result = sum(nums) / len(nums)

        return _json({"strategy": strategy, "values": nums, "consensus": result})

    async def handle_generate_validated(request):
        """Generate NML with validation loop: generate → validate → retry.

        POST /generate_validated
        Body: {"prompt": "...", "max_retries": 3, "max_tokens": 512}

        Uses the connected LLM backend (--model) for generation, the grammar
        validator for syntax checking, and the C runtime for execution checking.
        On validation failure, feeds errors + opcode schemas back to the LLM
        for self-correction (up to max_retries attempts).
        """
        body = await request.json()
        prompt = body.get("prompt", "")
        max_retries = body.get("max_retries", 3)
        max_tokens = body.get("max_tokens", 512)
        data = body.get("data", "")

        if not prompt:
            return _json({"error": "prompt is required"}, 400)

        # Build system prompt — include available data slots if provided
        data_slots_hint = ""
        if data.strip():
            slots = re.findall(r'@(\w+)\s+shape=(\S+)', data)
            if slots:
                data_slots_hint = (
                    "\nAvailable data slots (use LD Rn @name to load):\n"
                    + "\n".join(f"  @{name} shape={shape}" for name, shape in slots))
        else:
            data_slots_hint = (
                "\nNo data file provided. Use FILL to create tensors inline. "
                "Example: FILL R0 #2 #3 #0.5 creates a 2x3 tensor filled with 0.5. "
                "Do NOT use LD with @named slots unless data slots are listed above.")

        system_msg = (
            "You are an NML (Neural Machine Language) assembler. "
            "Output only valid NML assembly code. "
            "Do not include explanations, markdown, or commentary.\n\n"
            "NML opcode reference (Rd=dest, Rs=source, #imm=immediate):\n"
            "Memory:    LD Rd @name | ST Rs @name | ALLC Rd #shape | MOV Rd Rs\n"
            "Arithmetic: MADD Rd Rs1 Rs2 | MSUB Rd Rs1 Rs2 | EMUL Rd Rs1 Rs2 | EDIV Rd Rs1 Rs2\n"
            "            SADD Rd Rs #imm | SSUB Rd Rs #imm | SDIV Rd Rs #imm | SCLR Rd Rs #imm\n"
            "Matrix:    MMUL Rd Rs1 Rs2 | SDOT Rd Rs1 Rs2 | TRNS Rd Rs | RSHP Rd Rs #shape\n"
            "Activation: RELU Rd Rs | SIGM Rd Rs | TANH Rd Rs | GELU Rd Rs | SOFT Rd Rs\n"
            "Vision:    CONV Rd Rs Rkernel #stride #pad | POOL Rd Rs #size #stride | UPSC Rd Rs #factor\n"
            "Transformer: ATTN Rd Rq Rk Rv | NORM Rd Rs Rgamma Rbeta | EMBD Rd Rtable Rindex\n"
            "Reduction: RDUC Rd Rs #dim #mode | CLMP Rd Rs #min #max | WHER Rd Rcond Rs1 Rs2\n"
            "Training:  TNET Rconfig #epochs | LOSS Rd Rpred Rlabel #type | BKWD Rgrad Ract Rloss\n"
            "           WUPD Rw Rgrad Rlr | BN Rd Rs Rgamma Rbeta | DROP Rd Rs #rate\n"
            "Backward:  RELUBK/SIGMBK/TANHBK/GELUBK/SOFTBK Rd Rgrad Rin (3 ops)\n"
            "           MMULBK Rd_di Rd_dw Rgrad Rin Rw | CONVBK Rd_di Rd_dk Rgrad Rin Rk (5 ops)\n"
            "Control:   HALT | JUMP #off | JMPT #off | JMPF #off | LOOP Rs|#n | ENDP | CALL #off | RET\n"
            "General:   SYS Rd #code | CMP Rs1 Rs2 | CMPI Rd Rs #imm\n"
            "Rules: MMUL needs [M,K]×[K,N]. Always end with HALT."
            + data_slots_hint
        )

        # Load opcode schemas for error correction
        opcode_schemas = ""
        try:
            sys.path.insert(0, str(PROJECT_ROOT / "lsp" / "nml_lsp"))
            from opcode_db import OPCODES
            def _get_help(errors_text):
                """Extract opcode names from errors and return schemas."""
                ops = set()
                for word in errors_text.split():
                    # Strip punctuation and quotes
                    cleaned = word.strip("'\",:;()[]")
                    canonical = cleaned.replace("_", "").upper()
                    if canonical in OPCODES:
                        ops.add(canonical)
                # Also scan for @slot errors — add memory/load hints
                if "@" in errors_text and "not found" in errors_text:
                    ops.update({"LD", "ST"})
                if not ops:
                    return ""
                lines = ["\nCorrect operand signatures:"]
                for op in sorted(ops):
                    info = OPCODES[op]
                    lines.append(f"  {op} {info.operand_schema} — {info.description[:60]}")
                # If any backward op, show all backward ops
                if any(op.endswith("BK") for op in ops):
                    lines.append("\nAll backward ops:")
                    for op in ["RELUBK","SIGMBK","TANHBK","GELUBK","SOFTBK",
                               "MMULBK","CONVBK","POOLBK","NORMBK","ATTNBK"]:
                        if op in OPCODES:
                            info = OPCODES[op]
                            lines.append(f"  {op} {info.operand_schema}")
                # TNET/TNDEEP usage example
                if "TNET" in ops or "TNDEEP" in ops or "n_layers" in errors_text:
                    lines.append(
                        "\nTNET usage (config-driven N-layer MLP):"
                        "\n  R0 = training data [samples, features]"
                        "\n  R9 = labels [samples, outputs]"
                        "\n  R1 = arch config [n_layers, 3] where each row = [in_size, out_size, activation]"
                        "\n       activation: 0=ReLU, 1=sigmoid, 2=tanh, 3=GELU, 4=linear"
                        "\n  TNET R1 #epochs"
                        "\n"
                        "\nExample (2-layer: 2->4->1, 500 epochs):"
                        "\n  FILL  R0 #4 #2 #0.0    ; training data placeholder [4,2]"
                        "\n  FILL  R9 #4 #1 #0.0    ; labels placeholder [4,1]"
                        "\n  CONST R1 #2 #3 0 2,4,0,4,1,0  ; config [2,3]: layer1=[2,4,relu] layer2=[4,1,relu]"
                        "\n  TNET  R1 #500           ; train 500 epochs"
                        "\n  HALT")
                return "\n".join(lines)
        except ImportError:
            _get_help = lambda _: ""

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]

        attempts = []
        for attempt in range(1, max_retries + 1):
            # Generate via LLM backend
            gen_text = ""
            if llm_backend_url:
                from aiohttp import ClientSession
                try:
                    async with ClientSession() as session:
                        async with session.post(
                            f"{llm_backend_url}/v1/chat/completions",
                            json={"messages": messages, "max_tokens": max_tokens,
                                  "temperature": 0.1, "stream": False},
                        ) as resp:
                            resp_data = await resp.json()
                            gen_text = resp_data.get("choices", [{}])[0].get(
                                "message", {}).get("content", "")
                except Exception as e:
                    attempts.append({"attempt": attempt, "error": str(e)})
                    continue
            elif model and tokenizer:
                from mlx_lm import generate as mlx_generate
                prompt_text = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False)
                loop = asyncio.get_event_loop()
                gen_text = await loop.run_in_executor(
                    None, lambda pt=prompt_text, mt=max_tokens: mlx_generate(
                        model, tokenizer, prompt=pt, max_tokens=mt, verbose=False))
            else:
                return _json({"error": "No model available"}, 503)

            # Strip markdown/think blocks
            code = re.sub(r'<think>.*?</think>', '', gen_text, flags=re.DOTALL)
            code = re.sub(r'```[a-z]*\n?|```', '', code).strip()

            # Validate grammar
            vresult = validate_program(code)
            if not vresult.get("valid", False):
                errors_text = "; ".join(
                    e.get("message", str(e)) for e in vresult.get("errors", []))
                help_text = _get_help(errors_text)
                attempts.append({
                    "attempt": attempt, "stage": "grammar",
                    "code": code, "errors": errors_text,
                })
                # Feed error back to LLM
                messages.append({"role": "assistant", "content": gen_text})
                messages.append({"role": "user", "content":
                    f"Your NML code had errors. Fix them.\n\n"
                    f"Previous code:\n{code}\n\n"
                    f"Errors:\n{errors_text}\n{help_text}\n\n"
                    f"Output only the corrected NML code."
                })
                continue

            # Runtime execution — only when data is provided
            if data.strip():
                exec_result = execute_program(code, data=data)
                if exec_result.get("status") not in ("ok", "HALTED"):
                    runtime_err = (exec_result.get("stderr")
                                   or exec_result.get("error")
                                   or exec_result.get("message")
                                   or "unknown runtime error")
                    help_text = _get_help(runtime_err)
                    attempts.append({
                        "attempt": attempt, "stage": "runtime",
                        "code": code, "errors": runtime_err,
                    })
                    messages.append({"role": "assistant", "content": gen_text})
                    data_slots = re.findall(r'@(\w+)', data)
                    data_hint = ""
                    if data_slots and ("not found" in runtime_err or "@" in runtime_err):
                        data_hint = f"\nAvailable data slots: {', '.join('@' + s for s in data_slots)}"
                    messages.append({"role": "user", "content":
                        f"Your NML code had a runtime error. Fix it.\n\n"
                        f"Previous code:\n{code}\n\n"
                        f"Error:\n{runtime_err}\n{help_text}{data_hint}\n\n"
                        f"Output only the corrected NML code."
                    })
                    continue

            # Success
            return _json({
                "valid": True,
                "code": code,
                "attempts": attempt,
                "stage": "complete",
                "grammar_errors": [],
                "runtime_errors": [],
                "history": attempts,
            })

        # All retries exhausted
        last = attempts[-1] if attempts else {}
        return _json({
            "valid": False,
            "code": last.get("code", ""),
            "attempts": max_retries,
            "stage": last.get("stage", "exhausted"),
            "grammar_errors": [last.get("errors", "")] if last.get("stage") == "grammar" else [],
            "runtime_errors": [last.get("errors", "")] if last.get("stage") == "runtime" else [],
            "history": attempts,
        })

    app = web.Application()
    app.router.add_get("/health", handle_health)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_post("/v1/chat/completions", handle_chat)
    app.router.add_post("/validate", handle_validate)
    app.router.add_post("/execute", handle_execute)
    app.router.add_post("/generate_validated", handle_generate_validated)
    app.router.add_post("/format", handle_format)
    app.router.add_post("/spec", handle_spec)
    app.router.add_post("/advise", handle_advise)
    app.router.add_post("/agent/register", handle_agent_register)
    app.router.add_get("/agent/list", handle_agent_list)
    app.router.add_post("/distribute", handle_distribute)
    app.router.add_post("/vote", handle_vote)
    app.router.add_route("OPTIONS", "/{path:.*}", handle_options)

    return app


# ═══════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="NML Server — MCP tools + optional LLM chat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # MCP mode (for Cursor IDE)
  python3 nml_server.py --transport stdio

  # HTTP mode (for terminal UI)
  python3 nml_server.py --http --port 8082

  # HTTP + LLM chat
  python3 nml_server.py --http --port 8082 --model ../domain/output/model/nml-equalized-merged
        """,
    )
    parser.add_argument("--transport", choices=["stdio"], default=None,
                        help="Run as MCP server over stdio")
    parser.add_argument("--http", action="store_true",
                        help="Run as HTTP server")
    parser.add_argument("--port", type=int, default=8082,
                        help="HTTP port (default: 8082)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to MLX model for chat completions")
    parser.add_argument("--advisor-llm", type=str, default=None,
                        help="URL for ML Advisor high-reasoning model "
                             "(e.g. https://api.anthropic.com or https://openrouter.ai/api)")
    parser.add_argument("--advisor-model", type=str, default=None,
                        help="Model name for advisor LLM (default: auto per provider, "
                             "or NML_ADVISOR_MODEL env var). "
                             "Examples: claude-opus-4-20250514, claude-sonnet-4-20250514, openai/gpt-4o")
    parser.add_argument("--advisor-max-tokens", type=int, default=4096,
                        help="Max tokens for advisor LLM responses (default: 4096, "
                             "overridable per-request via max_tokens in POST body)")
    args = parser.parse_args()

    if args.transport == "stdio":
        asyncio.run(handle_mcp_stdio())
    elif args.http:
        from aiohttp import web
        app = create_http_app(model_path=args.model, advisor_llm=args.advisor_llm,
                              advisor_model=args.advisor_model,
                              advisor_max_tokens=args.advisor_max_tokens)
        print(f"NML Server on :{args.port}")
        print(f"  Tools: {', '.join(t['name'] for t in TOOLS)}")
        if args.model:
            print(f"  Model: {args.model}")
        if args.advisor_llm:
            print(f"  Advisor: {args.advisor_llm}")
        print(f"  Chat:    http://localhost:{args.port}/v1/chat/completions")
        print(f"  Advise:  http://localhost:{args.port}/advise")
        web.run_app(app, port=args.port, print=None)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
