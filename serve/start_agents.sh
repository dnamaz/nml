#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/serve/logs"
MODEL_DIR="$PROJECT_ROOT/../nml-model-training/models"

PORT_SERVER=8082
PORT_GATEWAY=8083
PORT_THINK=8084

THINK_MODEL="$PROJECT_ROOT/../nml-model-training/models/nml-think-v2-6bit"

PIDS=()
MODEL=""
CHAT_UI=false
PIPELINE_UI=false
NO_GATEWAY=false
NO_THINK=false
ADVISOR_LLM=""
ADVISOR_MODEL=""

for arg in "$@"; do
    case "$arg" in
        --chat-ui)     CHAT_UI=true ;;
        --pipeline-ui) PIPELINE_UI=true ;;
        --no-gateway)  NO_GATEWAY=true ;;
        --no-think)    NO_THINK=true ;;
        --advisor-llm=*)   ADVISOR_LLM="${arg#*=}" ;;
        --advisor-model=*) ADVISOR_MODEL="${arg#*=}" ;;
        --help|-h)
            echo "Usage: $0 [model-name] [options]"
            echo ""
            echo "  model-name               Code model directory name (default: nml-v09-merged-6bit)"
            echo "  --chat-ui                Launch the chat UI"
            echo "  --pipeline-ui            Launch the pipeline UI"
            echo "  --no-think               Don't start the think model server"
            echo "  --no-gateway             Don't start the domain RAG gateway"
            echo "  --advisor-llm=URL        High-reasoning LLM for ML Advisor"
            echo "                           (e.g. https://api.anthropic.com)"
            echo "  --advisor-model=MODEL    Model name (e.g. claude-sonnet-4-20250514)"
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

MODEL="${MODEL:-nml-v09-merged-6bit}"
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

# Code model server (MLX, context includes opcode reference from nml_server.py)
ADVISOR_FLAG=""
if [ -n "$ADVISOR_LLM" ]; then
    ADVISOR_FLAG="--advisor-llm $ADVISOR_LLM"
fi
if [ -n "$ADVISOR_MODEL" ]; then
    ADVISOR_FLAG="$ADVISOR_FLAG --advisor-model $ADVISOR_MODEL"
fi
echo "  Code Server  on :$PORT_SERVER (model: $MODEL)"
python3 "$PROJECT_ROOT/serve/nml_server.py" --http --port "$PORT_SERVER" \
    --model "$MODEL_PATH" $ADVISOR_FLAG \
    > "$LOG_DIR/server.log" 2>&1 &
PIDS+=($!)

# Think model server (Qwen3.5-4B reasoning)
if ! $NO_THINK; then
    echo "  Think Server on :$PORT_THINK (model: $(basename $THINK_MODEL))"
    python3 "$PROJECT_ROOT/serve/nml_server.py" --http --port "$PORT_THINK" \
        --model "$THINK_MODEL" \
        > "$LOG_DIR/think_server.log" 2>&1 &
    PIDS+=($!)
fi

# Domain RAG gateway (optional)
if ! $NO_GATEWAY; then
    if [ -f "$PROJECT_ROOT/domain/transpilers/domain_rag_server.py" ]; then
        echo "  Gateway      on :$PORT_GATEWAY"
        python3 "$PROJECT_ROOT/domain/transpilers/domain_rag_server.py" \
            --domains tax --port "$PORT_GATEWAY" \
            > "$LOG_DIR/gateway.log" 2>&1 &
        PIDS+=($!)
    else
        NO_GATEWAY=true
    fi
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

wait_for "http://localhost:$PORT_SERVER/health" "Code Server"
if ! $NO_THINK; then
    wait_for "http://localhost:$PORT_THINK/health" "Think Server"
fi
if ! $NO_GATEWAY; then
    wait_for "http://localhost:$PORT_GATEWAY/v1/rag/status" "Gateway"
fi

echo ""
echo "  ═══════════════════════════════════════════"
echo "  NML Agent Services — Ready"
echo "  ═══════════════════════════════════════════"
echo ""
echo "  Code Server   http://localhost:$PORT_SERVER  ($MODEL)"
if ! $NO_THINK; then
    echo "  Think Server  http://localhost:$PORT_THINK  ($(basename $THINK_MODEL))"
fi
if ! $NO_GATEWAY; then
    echo "  Gateway       http://localhost:$PORT_GATEWAY"
fi
if [ -n "$ADVISOR_LLM" ]; then
    echo "  ML Advisor    $ADVISOR_LLM"
else
    echo "  ML Advisor    KB-only (use --advisor-llm=URL for cloud LLM)"
fi
echo "  Advise API    http://localhost:$PORT_SERVER/advise"
echo "  Logs:         $LOG_DIR/"
echo ""

if $CHAT_UI; then
    echo "  Launching NML chat UI..."
    cd "$PROJECT_ROOT" && bun ~/.cursor/skills/jxs-runner/scripts/jsx.ts terminal/nml_chat.jsx &
    PIDS+=($!)
fi

if $PIPELINE_UI; then
    echo "  Launching NML pipeline UI..."
    cd "$PROJECT_ROOT" && bun ~/.cursor/skills/jxs-runner/scripts/jsx.ts terminal/nml_pipeline.jsx &
    PIDS+=($!)
fi

echo "  Press Ctrl+C to stop all services."
echo ""

wait
