#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/group7/orin-rag"
LLAMA_SERVER="/home/group7/llama.cpp/build/bin/llama-server"
MODEL="$PROJECT_DIR/models/llm/qwen2.5-7b-instruct-q4_k_m.gguf"

HOST="127.0.0.1"
PORT="8080"
CTX="4096"
THREADS="6"
GPU_LAYERS="999"

cd "$PROJECT_DIR"

# Activate venv for streamlit
source "$PROJECT_DIR/.venv/bin/activate"

# Start llama-server in background
echo "[+] Starting llama-server..."
"$LLAMA_SERVER" \
  -m "$MODEL" \
  --host "$HOST" --port "$PORT" \
  -c "$CTX" -t "$THREADS" -ngl "$GPU_LAYERS" \
  > "$PROJECT_DIR/llama_server.log" 2>&1 &

LLAMA_PID=$!
echo "[+] llama-server PID: $LLAMA_PID"

# Wait for server health endpoint
echo "[+] Waiting for llama-server to become ready..."
for i in {1..60}; do
  if curl -s "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    echo "[+] llama-server is ready."
    break
  fi
  sleep 1
done

# If still not ready, show last log lines and exit
if ! curl -s "http://$HOST:$PORT/health" >/dev/null 2>&1; then
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
exec streamlit run app/app.py
