#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/serve/logs"

PORT_MLX=8081
PORT_GATEWAY=8082
PORT_TRANSPILER=8083
PORT_VALIDATOR=8084
PORT_ENGINE=8085

PIDS=()
WITH_MLX=false
WITH_GATEWAY=false

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

for arg in "$@"; do
    case "$arg" in
        --with-mlx)     WITH_MLX=true ;;
        --with-gateway) WITH_GATEWAY=true ;;
    esac
done

mkdir -p "$LOG_DIR"

if [ ! -f "$PROJECT_ROOT/nml" ]; then
    echo "  Building NML runtime..."
    make -C "$PROJECT_ROOT" nml
fi

echo "  Starting agent services..."

if $WITH_MLX; then
    echo "  Starting MLX LM server on :$PORT_MLX"
    python3 -m mlx_lm.server --port "$PORT_MLX" \
        > "$LOG_DIR/mlx.log" 2>&1 &
    PIDS+=($!)
fi

if $WITH_GATEWAY; then
    echo "  Starting domain RAG gateway on :$PORT_GATEWAY"
    python3 "$PROJECT_ROOT/domain/transpilers/domain_rag_server.py" --domains tax \
        > "$LOG_DIR/gateway.log" 2>&1 &
    PIDS+=($!)
fi

echo "  Starting transpiler service on :$PORT_TRANSPILER"
python3 "$PROJECT_ROOT/domain/serve/transpiler_service.py" \
    > "$LOG_DIR/transpiler.log" 2>&1 &
PIDS+=($!)

echo "  Starting validation service on :$PORT_VALIDATOR"
python3 "$PROJECT_ROOT/domain/serve/validation_service.py" \
    > "$LOG_DIR/validator.log" 2>&1 &
PIDS+=($!)

echo "  Starting execution service on :$PORT_ENGINE"
python3 "$PROJECT_ROOT/serve/execution_service.py" \
    > "$LOG_DIR/engine.log" 2>&1 &
PIDS+=($!)

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

wait_for "http://localhost:$PORT_TRANSPILER/health" "Transpiler"
wait_for "http://localhost:$PORT_VALIDATOR/health"  "Validator"
wait_for "http://localhost:$PORT_ENGINE/health"     "Engine"

if $WITH_MLX; then
    wait_for "http://localhost:$PORT_MLX/health" "MLX LM"
fi
if $WITH_GATEWAY; then
    wait_for "http://localhost:$PORT_GATEWAY/health" "Gateway"
fi

echo ""
echo "  ═══════════════════════════════════════════"
echo "  NML Agent Services — Ready"
echo "  ═══════════════════════════════════════════"
echo ""
echo "  Transpiler  http://localhost:$PORT_TRANSPILER"
echo "  Validator   http://localhost:$PORT_VALIDATOR"
echo "  Engine      http://localhost:$PORT_ENGINE"
if $WITH_MLX; then
    echo "  MLX LM      http://localhost:$PORT_MLX"
fi
if $WITH_GATEWAY; then
    echo "  Gateway     http://localhost:$PORT_GATEWAY"
fi
echo ""
echo "  Logs in: $LOG_DIR/"
echo "  Press Ctrl+C to stop all services."
echo ""

wait
