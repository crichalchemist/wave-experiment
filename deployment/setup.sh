#!/usr/bin/env bash
# setup.sh — Detective LLM deployment bootstrap
#
# Target: Azure Lsv3-series (8 vCPU, 64 GB RAM, NVMe local storage)
# Inference: Ollama (llama.cpp-backed) with DeepSeek-R1-Distill-Qwen-7B Q4_K_M
# API compatibility: Ollama's OpenAI-compatible endpoint (/v1) — works with VLLMProvider
#
# Usage:
#   chmod +x setup.sh && ./setup.sh

set -euo pipefail

# Resolve repo root.  Honour an explicit env-var override first, then try to
# detect from the script's own location, then walk up from CWD.
if [ -z "${REPO_ROOT:-}" ]; then
  _script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [ -f "${_script_dir}/../pyproject.toml" ]; then
    REPO_ROOT="$(cd "${_script_dir}/.." && pwd)"
  else
    # Script was run from outside the repo (e.g. copied to ~/setup-detective.sh);
    # walk up from CWD to find a directory that contains pyproject.toml.
    _candidate="${PWD}"
    while [ "${_candidate}" != "/" ]; do
      [ -f "${_candidate}/pyproject.toml" ] && break
      _candidate="$(dirname "${_candidate}")"
    done
    if [ -f "${_candidate}/pyproject.toml" ]; then
      REPO_ROOT="${_candidate}"
    else
      echo "ERROR: Could not locate repo root (no pyproject.toml found)."
      echo "  Run this script from inside the cloned repository, or override:"
      echo "    REPO_ROOT=/path/to/wave-experiment bash setup.sh"
      exit 1
    fi
  fi
fi
VENV_DIR="${REPO_ROOT}/.venv-deploy"

echo "=== Detective LLM — L-series CPU deployment setup ==="
echo "Repo:  ${REPO_ROOT}"
echo "Venv:  ${VENV_DIR}"
echo ""

# ------------------------------------------------------------------
# 1. System check
# ------------------------------------------------------------------
echo "--- System ---"
lscpu | grep -E "^(CPU\(s\)|Model name|Thread)" || true
free -h | grep Mem
echo ""

# ------------------------------------------------------------------
# 2. Ollama (OpenAI-compatible CPU inference)
# ------------------------------------------------------------------
if ! command -v ollama &>/dev/null; then
  echo "--- Installing Ollama ---"
  curl -fsSL https://ollama.com/install.sh | sh
else
  echo "Ollama already installed: $(ollama --version)"
fi

# Start Ollama daemon in background (if not already running)
if ! pgrep -x ollama &>/dev/null; then
  echo "Starting Ollama daemon..."
  nohup ollama serve > /tmp/ollama.log 2>&1 &
  sleep 3
fi

# Pull the model (cached to NVMe after first pull)
echo ""
echo "--- Pulling DeepSeek-R1-Distill-Qwen-7B (Q4_K_M, ~4.4 GB) ---"
echo "  Model will be cached at: ~/.ollama/models"
echo "  On NVMe: subsequent starts are near-instant"
ollama pull deepseek-r1:7b

echo ""
echo "--- Verifying inference ---"
# Pipe to head then discard the SIGPIPE signal; the model responding at all is the pass signal.
ollama run deepseek-r1:7b "Reply with only: OK" --nowordwrap 2>/dev/null | head -3 || true

# ------------------------------------------------------------------
# 3. Python environment (3.11/3.12 — torch requires < 3.14)
# ------------------------------------------------------------------
PYTHON_BIN=$(command -v python3.12 2>/dev/null \
  || command -v python3.11 2>/dev/null \
  || { echo "ERROR: Python 3.11 or 3.12 required (torch is not compatible with 3.14)"; exit 1; })

PY_VER=$("${PYTHON_BIN}" --version | awk '{print $2}')
echo ""
echo "--- Python ${PY_VER} at ${PYTHON_BIN} ---"

if [ ! -d "${VENV_DIR}" ]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip --quiet

# ------------------------------------------------------------------
# 4. Install Python deps (CPU-only torch — no CUDA needed)
# ------------------------------------------------------------------
echo "--- Installing PyTorch (CPU) ---"
pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet

echo "--- Installing project deps ---"
pip install -r "${REPO_ROOT}/deployment/requirements-deploy.txt" --quiet

pip install -e "${REPO_ROOT}[dev]" --no-deps --quiet

echo ""
echo "=== Setup complete ==="
echo ""
echo "Environment variables needed (copy to .env.local):"
echo "  VLLM_BASE_URL=http://localhost:11434/v1"
echo "  VLLM_MODEL=deepseek-r1:7b"
echo "  DETECTIVE_PROVIDER=vllm"
echo ""
echo "Start inference:  ollama serve  (or: systemctl start ollama)"
echo "Check model:      ollama list"
echo "Activate venv:    source ${VENV_DIR}/bin/activate"
echo "Run API:          uvicorn src.api.main:app --host 0.0.0.0 --port 8080"