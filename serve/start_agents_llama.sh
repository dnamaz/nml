#!/bin/bash
# NML Agent Pipeline — macOS Startup Script
#
# Starts the think model and code model (both llama-server + GGUF) and
# the NML server for the validated code generation pipeline.
#
# Models:
#   Think: nml-think-v2 Q8_0 via llama-server (architecture planning)
#   Code:  nml-1.5b-instruct-v0.10.0-f16 via llama-server (NML assembly generation)
#
# Usage:
#   ./start_agents_llama.sh                       # defaults
#   ./start_agents_llama.sh --llama /path/to/llama-server
#   ./start_agents_llama.sh --no-think            # code model only
#   ./start_agents_llama.sh --code-model /path/to/model

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/serve/logs"

PORT_SERVER=8082
PORT_THINK=8084
PORT_CODE=8086

# ── Model paths ─────────────────────────────────────────────────────────
THINK_MODEL="$ROOT/nml-model-training/models/nml-think-v2-merged/nml-think-v2-Q8_0.gguf"
CODE_MODEL="$ROOT/nml-model-training/output/nml-1.5b-v0.11.0/nml-1.5b-instruct-v0.10.0-20260406-f16.gguf"

# ── Defaults ────────────────────────────────────────────────────────────
LLAMA_SERVER="llama-server"
NO_THINK=false
ADVISOR_LLM=""
ADVISOR_MODEL=""

# ── Parse arguments ─────────────────────────────────────────────────────
while [ $# -gt 0 ]; do
    case "$1" in
        --llama)          LLAMA_SERVER="$2"; shift 2 ;;
        --think-model)    THINK_MODEL="$2"; shift 2 ;;
        --code-model)     CODE_MODEL="$2"; shift 2 ;;
        --think-port)     PORT_THINK="$2"; shift 2 ;;
        --code-port)      PORT_CODE="$2"; shift 2 ;;
        --server-port)    PORT_SERVER="$2"; shift 2 ;;
        --no-think)       NO_THINK=true; shift ;;
        --advisor-llm)    ADVISOR_LLM="$2"; shift 2 ;;
        --advisor-model)  ADVISOR_MODEL="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "  --llama PATH         Path to llama-server (default: llama-server on PATH)"
            echo "  --think-model PATH   Think model GGUF (default: nml-think-v2-Q8_0.gguf)"
            echo "  --code-model PATH    Code model GGUF (default: nml-1.5b-instruct-v0.10.0-f16.gguf)"
            echo "  --think-port PORT    Think model port (default: 8084)"
            echo "  --code-port PORT     Code model port (default: 8086)"
            echo "  --server-port PORT   NML server port (default: 8082)"
            echo "  --no-think           Don't start the think model"
            echo "  --advisor-llm URL    High-reasoning LLM for ML Advisor"
            echo "                       (e.g. https://api.anthropic.com)"
            echo "  --advisor-model NAME Model for advisor (e.g. claude-sonnet-4-20250514)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ── Validation ──────────────────────────────────────────────────────────
if ! $NO_THINK; then
    if ! command -v "$LLAMA_SERVER" &>/dev/null; then
        echo "ERROR: llama-server not found at '$LLAMA_SERVER'"
        echo "  Install: brew install llama.cpp"
        echo "  Or set: --llama /path/to/llama-server"
        exit 1
    fi
fi

if [ ! -f "$CODE_MODEL" ]; then
    echo "  ERROR: Code model not found: $CODE_MODEL"
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
echo "  NML Agent Pipeline"
echo "  =========================================="
echo "  Think model:  $(basename "$THINK_MODEL") (llama-server)"
echo "  Code model:   $(basename "$CODE_MODEL") (llama-server)"
echo "  Think port:   $PORT_THINK"
echo "  Code port:    $PORT_CODE"
echo "  Server port:  $PORT_SERVER"
echo ""

# ── Start Think Model (llama-server + GGUF) ──────────────────────────
if ! $NO_THINK; then
    if [ -f "$THINK_MODEL" ]; then
        echo "  Starting Think Model on :$PORT_THINK..."
        "$LLAMA_SERVER" \
            -m "$THINK_MODEL" \
            --chat-template chatml \
            -c 4096 \
            --port "$PORT_THINK" \
            --host 127.0.0.1 \
            -ngl 99 \
            > "$LOG_DIR/think_llama.log" 2>&1 &
        PIDS+=($!)
        echo "  Think model PID: $!"
        sleep 2
    else
        echo "  WARN: Think model not found: $THINK_MODEL"
        echo "         Continuing without think model."
    fi
fi

# ── Start Code Model (llama-server + GGUF) ───────────────────────────
if [ -f "$CODE_MODEL" ]; then
    echo "  Starting Code Model (1.5B f16) on :$PORT_CODE..."
    "$LLAMA_SERVER" \
        -m "$CODE_MODEL" \
        --chat-template chatml \
        -c 4096 \
        --port "$PORT_CODE" \
        --host 127.0.0.1 \
        -ngl 99 \
        > "$LOG_DIR/code_llama.log" 2>&1 &
    PIDS+=($!)
    echo "  Code model PID: $!"
    sleep 2
else
    echo "  WARN: Code model not found: $CODE_MODEL"
    echo "         Continuing without code model."
fi

# ── Start NML Server (proxying to code model llama-server) ───────────
ADVISOR_FLAG=""
if [ -n "$ADVISOR_LLM" ]; then
    ADVISOR_FLAG="--advisor-llm $ADVISOR_LLM"
fi
if [ -n "$ADVISOR_MODEL" ]; then
    ADVISOR_FLAG="$ADVISOR_FLAG --advisor-model $ADVISOR_MODEL"
fi
echo "  Starting NML Server on :$PORT_SERVER (proxy to code model on :$PORT_CODE)..."
python3 -u "$PROJECT_ROOT/serve/nml_server.py" --http --port "$PORT_SERVER" \
    --model "http://127.0.0.1:$PORT_CODE" $ADVISOR_FLAG \
    > "$LOG_DIR/server.log" 2>&1 &
PIDS+=($!)
echo "  Server PID: $!"

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

if ! $NO_THINK && [ -f "$THINK_MODEL" ]; then
    wait_for "http://localhost:$PORT_THINK/health" "Think Model"
fi
if [ -f "$CODE_MODEL" ]; then
    wait_for "http://localhost:$PORT_CODE/health" "Code Model"
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
if [ -f "$CODE_MODEL" ]; then
    echo "  Code model:     http://localhost:$PORT_CODE  ($(basename "$CODE_MODEL"))"
fi
if [ -n "$ADVISOR_LLM" ]; then
    echo "  ML Advisor:     $ADVISOR_LLM"
else
    echo "  ML Advisor:     KB-only (use --advisor-llm URL for cloud LLM)"
fi
echo "  Advise API:     POST http://localhost:$PORT_SERVER/advise"
echo "  Validated gen:  POST http://localhost:$PORT_SERVER/generate_validated"
echo "  Logs:           $LOG_DIR/"
echo ""
echo "  Press Ctrl+C to stop all services."
echo ""

wait
