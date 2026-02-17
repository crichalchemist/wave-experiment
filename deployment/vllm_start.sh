#!/usr/bin/env bash
# vllm_start.sh — Start Ollama inference server
#
# On L-series: Ollama serves DeepSeek-R1 via llama.cpp on CPU.
# Exposes OpenAI-compatible API at http://localhost:11434/v1
# VLLMProvider connects with VLLM_BASE_URL=http://localhost:11434/v1
#
# Usage:
#   ./vllm_start.sh              # foreground (logs to stdout)
#   ./vllm_start.sh --daemon     # background with PID file

set -euo pipefail

MODEL="${OLLAMA_MODEL:-deepseek-r1:7b}"
OLLAMA_HOST="${OLLAMA_HOST:-0.0.0.0:11434}"
PID_FILE="/tmp/ollama-detective.pid"

daemon_mode=false
for arg in "$@"; do
  [[ "${arg}" == "--daemon" ]] && daemon_mode=true
done

# Verify model is pulled
if ! ollama list 2>/dev/null | grep -q "${MODEL}"; then
  echo "Model '${MODEL}' not found. Run: ollama pull ${MODEL}"
  exit 1
fi

echo "Starting Ollama inference server"
echo "  Model:  ${MODEL}"
echo "  Listen: ${OLLAMA_HOST}"
echo "  API:    http://localhost:11434/v1  (OpenAI-compatible)"
echo ""

export OLLAMA_HOST
export OLLAMA_NUM_PARALLEL=2          # concurrent requests (tune to vCPU count)
export OLLAMA_MAX_LOADED_MODELS=1     # keep single model in RAM (64 GB budget)

if $daemon_mode; then
  nohup ollama serve > /tmp/ollama.log 2>&1 &
  echo $! > "${PID_FILE}"
  echo "Ollama PID: $(cat ${PID_FILE})"
  echo "Logs: tail -f /tmp/ollama.log"
  echo "Stop: kill \$(cat ${PID_FILE})"
else
  exec ollama serve
fi
