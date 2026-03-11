#!/usr/bin/env bash
# setup.sh — Detective LLM deployment bootstrap
#
# Target: Azure GPU VM (A10/V100+) or CPU-only (slower, no LoRA hot-swap)
# Inference: vLLM with DeepSeek-R1-Distill-Qwen-7B (OpenAI-compatible /v1)
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

echo "=== Detective LLM — vLLM deployment setup ==="
echo "Repo:  ${REPO_ROOT}"
echo "Venv:  ${VENV_DIR}"
echo ""

# ------------------------------------------------------------------
# 1. System check
# ------------------------------------------------------------------
echo "--- System ---"
lscpu | grep -E "^(CPU\(s\)|Model name|Thread)" || true
free -h | grep Mem
# Check for GPU
if command -v nvidia-smi &>/dev/null; then
  echo ""
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi present but no GPU detected)"
else
  echo ""
  echo "  No nvidia-smi found — CPU-only mode (slower inference, no LoRA hot-swap)"
fi
echo ""

# ------------------------------------------------------------------
# 2. Python environment (3.12+ required)
# ------------------------------------------------------------------
PYTHON_BIN=$(command -v python3.13 2>/dev/null \
  || command -v python3.12 2>/dev/null \
  || { echo "ERROR: Python 3.12+ required"; exit 1; })

PY_VER=$("${PYTHON_BIN}" --version | awk '{print $2}')
echo "--- Python ${PY_VER} at ${PYTHON_BIN} ---"

if [ ! -d "${VENV_DIR}" ]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip --quiet

# ------------------------------------------------------------------
# 3. Install PyTorch (detect GPU vs CPU)
# ------------------------------------------------------------------
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
  echo "--- Installing PyTorch (CUDA) ---"
  pip install torch torchvision torchaudio --quiet
else
  echo "--- Installing PyTorch (CPU) ---"
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
fi

# ------------------------------------------------------------------
# 4. Install vLLM
# ------------------------------------------------------------------
echo "--- Installing vLLM ---"
pip install vllm --quiet

# ------------------------------------------------------------------
# 5. Install project deps
# ------------------------------------------------------------------
echo "--- Installing project deps ---"
pip install -r "${REPO_ROOT}/deployment/requirements-deploy.txt" --quiet

pip install -e "${REPO_ROOT}[dev]" --no-deps --quiet

echo ""
echo "=== Setup complete ==="
echo ""
echo "Create .env.local with:"
echo "  VLLM_BASE_URL=http://localhost:8000/v1"
echo "  VLLM_MODEL=detective"
echo "  DETECTIVE_PROVIDER=vllm"
echo ""
echo "Start inference:"
echo "  ./deployment/vllm_start.sh --lora          # base + DPO adapter"
echo "  ./deployment/vllm_start.sh                  # base model only"
echo ""
echo "Verify:"
echo "  curl http://localhost:8000/v1/models"
echo "  curl http://localhost:8000/v1/chat/completions \\"
echo "    -d '{\"model\": \"detective\", \"messages\": [{\"role\": \"user\", \"content\": \"Reply with only: OK\"}]}'"
echo ""
echo "Activate venv:    source ${VENV_DIR}/bin/activate"
echo "Run API:          uvicorn src.api.main:app --host 0.0.0.0 --port 8080"
