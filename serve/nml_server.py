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
NML_BINARY = PROJECT_ROOT / ("nml.exe" if sys.platform == "win32" else "nml")
SPEC_PATH = PROJECT_ROOT / "docs" / "NML_SPEC.md"
GETTING_STARTED = PROJECT_ROOT / "docs" / "GETTING_STARTED.md"

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
                    outputs[name] = [float(v.strip()) for v in vals_str]
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
# HTTP Server (for terminal UI + LLM chat)
# ═══════════════════════════════════════════

def create_http_app(model_path: str = None):
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

    async def handle_options(request):
        return web.Response(status=204, headers=_cors())

    async def handle_health(request):
        return _json({
            "status": "healthy",
            "service": "nml-server",
            "tools": [t["name"] for t in TOOLS],
            "model": model_path if (model or llm_backend_url) else None,
            "constrained_decoding": outlines_model is not None,
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

        if constrained and outlines_model and nml_cfg:
            response_text = outlines_model(prompt, output_type=nml_cfg, max_tokens=max_tokens)
        else:
            from mlx_lm import generate as mlx_generate
            response_text = mlx_generate(model, tokenizer, prompt=prompt,
                                         max_tokens=max_tokens, verbose=False)

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

    app = web.Application()
    app.router.add_get("/health", handle_health)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_post("/v1/chat/completions", handle_chat)
    app.router.add_post("/validate", handle_validate)
    app.router.add_post("/execute", handle_execute)
    app.router.add_post("/format", handle_format)
    app.router.add_post("/spec", handle_spec)
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
    args = parser.parse_args()

    if args.transport == "stdio":
        asyncio.run(handle_mcp_stdio())
    elif args.http:
        from aiohttp import web
        app = create_http_app(model_path=args.model)
        print(f"NML Server on :{args.port}")
        print(f"  Tools: {', '.join(t['name'] for t in TOOLS)}")
        if args.model:
            print(f"  Model: {args.model}")
        print(f"  Chat:  http://localhost:{args.port}/v1/chat/completions")
        web.run_app(app, port=args.port, print=None)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
