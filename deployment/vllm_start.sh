#!/usr/bin/env bash
# vllm_start.sh — Start vLLM inference server with optional LoRA adapter
#
# Serves DeepSeek-R1-Distill-Qwen-7B via vLLM's OpenAI-compatible API.
# When --enable-lora is active, the DPO adapter is addressable as model="detective"
# in API calls (set VLLM_MODEL=detective in .env.local).
#
# Usage:
#   ./vllm_start.sh              # base model only
#   ./vllm_start.sh --lora       # base model + detective LoRA adapter
#   ./vllm_start.sh --daemon     # background with PID file

set -euo pipefail

BASE_MODEL="${VLLM_BASE_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
LORA_ADAPTER="${VLLM_LORA_ADAPTER:-crichalchemist/detective-llm-dpo-adapter}"
LORA_NAME="${VLLM_LORA_NAME:-detective}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
DTYPE="${VLLM_DTYPE:-bfloat16}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
PID_FILE="/tmp/vllm-detective.pid"

enable_lora=false
daemon_mode=false
for arg in "$@"; do
  [[ "${arg}" == "--lora" ]] && enable_lora=true
  [[ "${arg}" == "--daemon" ]] && daemon_mode=true
done

echo "Starting vLLM inference server"
echo "  Base model:     ${BASE_MODEL}"
echo "  Host:           ${HOST}:${PORT}"
echo "  dtype:          ${DTYPE}"
echo "  max-model-len:  ${MAX_MODEL_LEN}"

VLLM_ARGS=(
  "${BASE_MODEL}"
  --host "${HOST}"
  --port "${PORT}"
  --dtype "${DTYPE}"
  --max-model-len "${MAX_MODEL_LEN}"
)

if $enable_lora; then
  echo "  LoRA adapter:   ${LORA_NAME} → ${LORA_ADAPTER}"
  VLLM_ARGS+=(
    --enable-lora
    --lora-modules "${LORA_NAME}=${LORA_ADAPTER}"
  )
fi

echo ""
echo "  API endpoint:   http://localhost:${PORT}/v1"
echo "  Model name:     ${LORA_NAME} (with --lora) or ${BASE_MODEL}"
echo ""

if $daemon_mode; then
  nohup vllm serve "${VLLM_ARGS[@]}" > /tmp/vllm.log 2>&1 &
  echo $! > "${PID_FILE}"
  echo "vLLM PID: $(cat ${PID_FILE})"
  echo "Logs: tail -f /tmp/vllm.log"
  echo "Stop: kill \$(cat ${PID_FILE})"
else
  exec vllm serve "${VLLM_ARGS[@]}"
fi
