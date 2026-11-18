#!/usr/bin/env bash
# One-command setup (Raspberry Pi 5 with 64-bit Raspberry Pi OS Bookworm, any arm64/x86-64 Linux, or macOS).
# Installs the same stack as the arm64 twin (twin/Dockerfile): pinned deps, torch 2.15.0+cpu, ExecuTorch 1.6.0.
# Usage: bash scripts/setup.sh            (add --no-llm to skip torch/ExecuTorch, e.g. on a low-disk machine)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
WITH_LLM=1
[ "${1:-}" = "--no-llm" ] && WITH_LLM=0

if command -v apt-get >/dev/null 2>&1; then
  # python3-venv for the venv; libatomic1 is needed by ExecuTorch's tokenizer bindings
  sudo apt-get install -y python3-venv libatomic1 || true
fi

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
if [ "$WITH_LLM" = 1 ]; then
  # CPU-only torch: the default PyPI wheel pulls >1 GB of CUDA libraries a Pi cannot use
  pip install "torch==2.15.0+cpu" --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
  pip install "executorch==1.6.0"
fi
pip install -r requirements.txt
pip install -e .

echo
echo "Setup done. Try:"
echo "  source .venv/bin/activate"
echo "  bastion status        # environment tag, models, Llama runtime"
echo "  bastion demo          # handwritten demo samples"
echo "  bastion serve         # web app on http://127.0.0.1:8080  (--lan to use from a phone)"
echo "  pytest -q"
echo "Llama: copy models/llama/llama3_2_1b_8da4w.pte + tokenizer.model (see docs/EXECUTORCH_NOTES.md)."
