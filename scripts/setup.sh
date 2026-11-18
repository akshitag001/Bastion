#!/usr/bin/env bash
# One-command setup for the Python app (works on a dev laptop AND on Raspberry Pi 5, 64-bit OS).
# Usage: bash scripts/setup.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

echo
echo "Setup done. Try:"
echo "  source .venv/bin/activate"
echo "  bastion status"
echo "  bastion demo"
echo "  pytest -q"
