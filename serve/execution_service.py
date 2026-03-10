#!/usr/bin/env python3
"""NML Execution Service — aiohttp wrapper around the NML C runtime."""

import argparse
import asyncio
import json
import os
import re
import tempfile
from pathlib import Path

from aiohttp import web

DEFAULT_BINARY = Path(__file__).parent.parent / "nml"
TIMEOUT_SECONDS = 5


def _cors_headers():
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }


def _json_response(data: dict, status: int = 200) -> web.Response:
    return web.Response(
        text=json.dumps(data),
        content_type="application/json",
        status=status,
        headers=_cors_headers(),
    )


async def run_nml(binary: Path, nml_program: str, data_content: str,
                  trace: bool = False, max_cycles: int | None = None) -> dict:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".nml", delete=False) as nf:
        nf.write(nml_program)
        nml_path = nf.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".nml.data", delete=False) as df:
        df.write(data_content)
        data_path = df.name

    try:
        cmd = [str(binary), nml_path, data_path]
        if trace:
            cmd.append("--trace")
        if max_cycles is not None:
            cmd.extend(["--max-cycles", str(max_cycles)])

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return {"status": "timeout", "outputs": {}}

        stdout = stdout_bytes.decode()
        stderr = stderr_bytes.decode()

        if proc.returncode != 0:
            return {"status": "error", "outputs": {}, "stderr": stderr.strip()}

        outputs = {}
        cycles = None
        time_us = None
        trace_lines = []
        status = "ok"

        for line in stdout.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue

            if ":" in stripped and "data=[" in stripped:
                m = re.match(r"(\w+):\s+shape=.*data=\[([^\]]+)\]", stripped)
                if m:
                    name = m.group(1)
                    vals = m.group(2).split(",")
                    if len(vals) == 1:
                        outputs[name] = float(vals[0].strip())
                    else:
                        outputs[name] = [float(v.strip()) for v in vals]

            if "HALTED" in stripped:
                status = "HALTED"
                m = re.search(r"(\d+) cycles", stripped)
                if m:
                    cycles = int(m.group(1))
                m = re.search(r"([\d.]+)\s*µs", stripped)
                if m:
                    time_us = float(m.group(1))

            if trace:
                trace_lines.append(stripped)

        result = {"status": status, "outputs": outputs}
        if cycles is not None:
            result["cycles"] = cycles
        if time_us is not None:
            result["time_us"] = time_us
        if trace:
            result["trace"] = trace_lines
        return result
    finally:
        os.unlink(nml_path)
        os.unlink(data_path)


async def handle_execute(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return _json_response({"error": "invalid JSON"}, 400)

    nml_program = body.get("nml_program")
    data = body.get("data", "")
    trace = body.get("trace", False)
    max_cycles = body.get("max_cycles")

    if not nml_program:
        return _json_response({"error": "nml_program is required"}, 400)

    binary = request.app["nml_binary"]
    result = await run_nml(binary, nml_program, data, trace=trace, max_cycles=max_cycles)
    return _json_response(result)


async def handle_execute_trace(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return _json_response({"error": "invalid JSON"}, 400)

    nml_program = body.get("nml_program")
    data = body.get("data", "")
    max_cycles = body.get("max_cycles")

    if not nml_program:
        return _json_response({"error": "nml_program is required"}, 400)

    binary = request.app["nml_binary"]
    result = await run_nml(binary, nml_program, data, trace=True, max_cycles=max_cycles)
    return _json_response(result)


async def handle_health(request: web.Request) -> web.Response:
    binary = request.app["nml_binary"]
    return _json_response({
        "status": "healthy",
        "service": "execution",
        "binary": str(binary),
        "binary_exists": binary.exists(),
    })


async def handle_options(request: web.Request) -> web.Response:
    return web.Response(status=204, headers=_cors_headers())


def create_app(binary: Path) -> web.Application:
    app = web.Application()
    app["nml_binary"] = binary

    app.router.add_post("/execute", handle_execute)
    app.router.add_post("/execute/trace", handle_execute_trace)
    app.router.add_get("/health", handle_health)
    app.router.add_route("OPTIONS", "/{path:.*}", handle_options)

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NML Execution Service")
    parser.add_argument("--port", type=int, default=8085)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    args = parser.parse_args()

    app = create_app(args.binary)
    print(f"NML Execution Service on :{args.port}  binary={args.binary}")
    web.run_app(app, port=args.port, print=None)
