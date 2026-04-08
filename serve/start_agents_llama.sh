#!/bin/bash
# NML Agent Pipeline — macOS Startup Script (llama-server + GGUF)
#
# Starts the think model, code model, and NML server for the
# validated code generation pipeline using llama-server with Metal.
#
# Models:
#   Think: nml-think-v2 Q8_0 (architecture planning)
#   Code:  nml-1.5b-instruct-v0.10.0 F16 (NML assembly generation)
#
# Usage:
#   ./start_agents_llama.sh                       # defaults (Metal GPU)
#   ./start_agents_llama.sh --llama /path/to/llama-server
#   ./start_agents_llama.sh --no-think            # code model only
#   ./start_agents_llama.sh --code-model /path/to/model.gguf

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/serve/logs"

PORT_SERVER=8082
PORT_THINK=8084
PORT_CODE=8085

# ── Model paths ─────────────────────────────────────────────────────────
THINK_MODEL="$ROOT/nml-model-training/models/nml-think-v2-merged/nml-think-v2-Q8_0.gguf"
CODE_MODEL="$ROOT/nml/domain/output/model/nml-1.5b-instruct-v0.10.0-f16.gguf"

# Fallback code model location
if [ ! -f "$CODE_MODEL" ]; then
    CODE_MODEL="$ROOT/nml-model-training/output/nml-1.5b-v0.11.0/nml-1.5b-instruct-v0.10.0-20260406-q4_k_m.gguf"
fi

# ── Defaults ────────────────────────────────────────────────────────────
LLAMA_SERVER="llama-server"
NO_THINK=false

# ── Parse arguments ─────────────────────────────────────────────────────
while [ $# -gt 0 ]; do
    case "$1" in
        --llama)        LLAMA_SERVER="$2"; shift 2 ;;
        --think-model)  THINK_MODEL="$2"; shift 2 ;;
        --code-model)   CODE_MODEL="$2"; shift 2 ;;
        --think-port)   PORT_THINK="$2"; shift 2 ;;
        --code-port)    PORT_CODE="$2"; shift 2 ;;
        --server-port)  PORT_SERVER="$2"; shift 2 ;;
        --no-think)     NO_THINK=true; shift ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "  --llama PATH         Path to llama-server (default: llama-server on PATH)"
            echo "  --think-model PATH   Think model GGUF (default: nml-think-v2-Q8_0.gguf)"
            echo "  --code-model PATH    Code model GGUF (default: nml-1.5b-instruct-v0.10.0-f16.gguf)"
            echo "  --think-port PORT    Think model port (default: 8084)"
            echo "  --code-port PORT     Code model port (default: 8085)"
            echo "  --server-port PORT   NML server port (default: 8082)"
            echo "  --no-think           Don't start the think model"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ── Validation ──────────────────────────────────────────────────────────
if ! command -v "$LLAMA_SERVER" &>/dev/null; then
    echo "ERROR: llama-server not found at '$LLAMA_SERVER'"
    echo "  Install: brew install llama.cpp"
    echo "  Or set: --llama /path/to/llama-server"
    exit 1
fi

PIDS=()

cleanup() {
    echo ""
    echo "  Shutting down agent services..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null
    echo "  All services stopped."
}
trap cleanup EXIT

mkdir -p "$LOG_DIR"

# ── Status ──────────────────────────────────────────────────────────────
echo ""
echo "  NML Agent Pipeline (llama-server + Metal)"
echo "  =========================================="
echo "  Think model:  $(basename "$THINK_MODEL")"
echo "  Code model:   $(basename "$CODE_MODEL")"
echo "  Think port:   $PORT_THINK"
echo "  Code port:    $PORT_CODE"
echo "  Server port:  $PORT_SERVER"
echo ""

# ── Start Think Model ──────────────────────────────────────────────────
if ! $NO_THINK; then
    if [ -f "$THINK_MODEL" ]; then
        echo "  Starting Think Model (4B Q8_0) on :$PORT_THINK..."
        "$LLAMA_SERVER" \
            -m "$THINK_MODEL" \
            --chat-template chatml \
            -c 4096 \
            --port "$PORT_THINK" \
            --host 127.0.0.1 \
            -ngl 99 \
            > "$LOG_DIR/think_llama.log" 2>&1 &
        PIDS+=($!)
        echo "  Think model PID: ${PIDS[-1]}"
    else
        echo "  WARN: Think model not found: $THINK_MODEL"
        echo "         Continuing without think model."
    fi
fi

sleep 2

# ── Start Code Model ──────────────────────────────────────────────────
if [ -f "$CODE_MODEL" ]; then
    echo "  Starting Code Model (1.5B F16) on :$PORT_CODE..."
    "$LLAMA_SERVER" \
        -m "$CODE_MODEL" \
        --chat-template chatml \
        -c 8192 \
        --port "$PORT_CODE" \
        --host 127.0.0.1 \
        -ngl 99 \
        > "$LOG_DIR/code_llama.log" 2>&1 &
    PIDS+=($!)
    echo "  Code model PID: ${PIDS[-1]}"
else
    echo "  ERROR: Code model not found: $CODE_MODEL"
    exit 1
fi

sleep 3

# ── Start NML Server ──────────────────────────────────────────────────
echo "  Starting NML Server on :$PORT_SERVER..."
python3 "$PROJECT_ROOT/serve/nml_server.py" --http --port "$PORT_SERVER" \
    --model "http://127.0.0.1:${PORT_CODE}" \
    > "$LOG_DIR/server.log" 2>&1 &
PIDS+=($!)
echo "  Server PID: ${PIDS[-1]}"

# ── Wait for services ────────────────────────────────────────────────
echo "  Waiting for services..."

wait_for() {
    local url=$1 name=$2
    for i in $(seq 1 30); do
        if curl -sf "$url" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    echo "  WARNING: $name did not become ready within 30s"
    return 1
}

wait_for "http://localhost:$PORT_CODE/health" "Code Model"
if ! $NO_THINK && [ -f "$THINK_MODEL" ]; then
    wait_for "http://localhost:$PORT_THINK/health" "Think Model"
fi
wait_for "http://localhost:$PORT_SERVER/health" "NML Server"

echo ""
echo "  ═══════════════════════════════════════════"
echo "  NML Agent Services — Ready"
echo "  ═══════════════════════════════════════════"
echo ""
echo "  Pipeline UI:    http://localhost:$PORT_SERVER"
if ! $NO_THINK && [ -f "$THINK_MODEL" ]; then
    echo "  Think model:    http://localhost:$PORT_THINK  ($(basename "$THINK_MODEL"))"
fi
echo "  Code model:     http://localhost:$PORT_CODE  ($(basename "$CODE_MODEL"))"
echo "  Validated gen:  POST http://localhost:$PORT_SERVER/generate_validated"
echo "  Logs:           $LOG_DIR/"
echo ""
echo "  Press Ctrl+C to stop all services."
echo ""

wait
