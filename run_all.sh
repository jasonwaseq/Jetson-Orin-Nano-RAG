#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_SERVER="/home/group7/llama.cpp/build/bin/llama-server"
MODEL="$PROJECT_DIR/models/llm/qwen2.5-7b-instruct-q4_k_m.gguf"

HOST="127.0.0.1"
PORT="8080"
CTX="4096"
THREADS="6"
GPU_LAYERS="999"

# -----------------------------
# Pre-flight Checks
# -----------------------------
echo "[*] Checking dependencies..."

if [[ ! -x "$LLAMA_SERVER" ]]; then
    echo "Error: llama-server not found or not executable at: $LLAMA_SERVER"
    echo "Please build llama.cpp first (see README)."
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    echo "Error: Model file not found at: $MODEL"
    echo "Please download the GGUF model to: $MODEL"
    exit 1
fi

if [[ ! -d "$PROJECT_DIR/.venv" ]]; then
    echo "Warning: .venv directory not found at $PROJECT_DIR/.venv"
    echo "Ideally, create a venv and install dependencies via 'pip install -r requirements.txt'"
    # We won't exit here, just warn, in case user is using system python or another env
else
    echo "[*] Activating venv..."
    source "$PROJECT_DIR/.venv/bin/activate"
fi

# Check for critical python libs
if ! python3 -c "import streamlit, faiss, pypdf" > /dev/null 2>&1; then
    echo "Error: Missing Python dependencies (streamlit, faiss, pypdf, etc.)."
    echo "Run: pip install -r requirements.txt"
    exit 1
fi

# -----------------------------
# Cleanup Old Processes
# -----------------------------
echo "[*] Cleaning up old llama-server instances..."
pkill -f llama-server || true
sleep 2

# Check if port 8080 is still in use
if lsof -i :$PORT >/dev/null 2>&1; then
    echo "Warning: Port $PORT is still in use. Attempting to force kill..."
    lsof -t -i :$PORT | xargs kill -9 || true
    sleep 1
fi

cd "$PROJECT_DIR"

# -----------------------------
# Start Server
# -----------------------------
echo "[+] Starting llama-server..."
# Note: Reduced GPU layers to 10 to avoid OOM on Orin Nano (8GB shared RAM).
# If you have more free RAM, you can increase this (max ~28 for 7B model).
"$LLAMA_SERVER" \
  -m "$MODEL" \
  --host "$HOST" --port "$PORT" \
  -c "$CTX" -t "$THREADS" -ngl "10" \
  > "$PROJECT_DIR/llama_server.log" 2>&1 &

LLAMA_PID=$!
echo "[+] llama-server PID: $LLAMA_PID"

# Wait for server health endpoint
echo "[+] Waiting for llama-server to become ready (timeout 60s)..."
SERVER_READY=false
for i in {1..60}; do
  if curl -s "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    echo "[+] llama-server is ready."
    SERVER_READY=true
    break
  fi
  sleep 1
done

# If still not ready, show last log lines and exit
if [ "$SERVER_READY" = false ]; then
  echo "[!] llama-server did not become ready. Showing last 50 lines of log:"
  tail -n 50 "$PROJECT_DIR/llama_server.log" || true
  kill "$LLAMA_PID" || true
  exit 1
fi

# Ensure llama-server is cleaned up when streamlit exits
cleanup() {
  echo
  echo "[+] Stopping llama-server (PID $LLAMA_PID)..."
  kill "$LLAMA_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[+] Starting Streamlit..."
exec streamlit run app/app.py --server.address 0.0.0.0
