#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/serve/logs"
MODEL_DIR="$PROJECT_ROOT/domain/output/model"

PORT_SERVER=8082
PORT_GATEWAY=8083

PIDS=()
MODEL=""
NO_UI=false
NO_GATEWAY=false

for arg in "$@"; do
    case "$arg" in
        --no-ui)      NO_UI=true ;;
        --no-gateway) NO_GATEWAY=true ;;
        --help|-h)
            echo "Usage: $0 [model-name] [--no-ui] [--no-gateway]"
            echo ""
            echo "  model-name     Model directory name (default: nml-equalized-merged)"
            echo "  --no-ui        Don't launch the chat UI"
            echo "  --no-gateway   Don't start the domain RAG gateway"
            exit 0
            ;;
        --*)
            echo "  Unknown flag: $arg"
            exit 1
            ;;
        *)
            MODEL="$arg"
            ;;
    esac
done

MODEL="${MODEL:-nml-equalized-merged}"
MODEL_PATH="$MODEL_DIR/$MODEL"

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

if [ ! -f "$PROJECT_ROOT/nml" ]; then
    echo "  Building NML runtime..."
    make -C "$PROJECT_ROOT" nml
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "  Model not found: $MODEL_PATH"
    echo ""
    echo "  Available models:"
    ls -1 "$MODEL_DIR" 2>/dev/null | grep -v adapter || echo "    (none found in $MODEL_DIR)"
    exit 1
fi

echo ""
echo "  Starting NML agent services..."

echo "  NML Server on :$PORT_SERVER (model: $MODEL)"
python3 "$PROJECT_ROOT/serve/nml_server.py" --http --port "$PORT_SERVER" \
    --model "$MODEL_PATH" \
    > "$LOG_DIR/server.log" 2>&1 &
PIDS+=($!)

if ! $NO_GATEWAY; then
    echo "  Domain RAG gateway on :$PORT_GATEWAY"
    python3 "$PROJECT_ROOT/domain/transpilers/domain_rag_server.py" \
        --domains tax --port "$PORT_GATEWAY" \
        > "$LOG_DIR/gateway.log" 2>&1 &
    PIDS+=($!)
fi

echo "  Waiting for services to become ready..."

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

wait_for "http://localhost:$PORT_SERVER/health" "NML Server"
if ! $NO_GATEWAY; then
    wait_for "http://localhost:$PORT_GATEWAY/v1/rag/status" "Gateway"
fi

echo ""
echo "  ═══════════════════════════════════════════"
echo "  NML Agent Services — Ready"
echo "  ═══════════════════════════════════════════"
echo ""
echo "  NML Server  http://localhost:$PORT_SERVER"
echo "    Chat      http://localhost:$PORT_SERVER/v1/chat/completions"
echo "    Execute   http://localhost:$PORT_SERVER/execute"
echo "    Validate  http://localhost:$PORT_SERVER/validate"
if ! $NO_GATEWAY; then
    echo "  Gateway     http://localhost:$PORT_GATEWAY"
fi
echo ""
echo "  Model: $MODEL"
echo "  Logs:  $LOG_DIR/"
echo ""

if ! $NO_UI; then
    echo "  Launching NML chat UI..."
    cd "$PROJECT_ROOT" && bun terminal/jxs.ts terminal/nml_chat.jsx &
    PIDS+=($!)
fi

echo "  Press Ctrl+C to stop all services."
echo ""

wait
