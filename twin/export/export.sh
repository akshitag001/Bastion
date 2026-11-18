#!/bin/sh
# Runs inside the export container (repo mounted at /work).
#   1. download the original Meta checkpoint (needs HF_TOKEN + accepted Llama 3.2 licence)
#   2. export to .pte with the versioned config twin/export/llama3_2_1b_8da4w.yaml
set -eu
CKPT=/work/models/llama/hf
OUT=/work/models/llama
mkdir -p "$OUT"

python -c "import executorch, torch; print('executorch', executorch.__version__ if hasattr(executorch,'__version__') else '?', '| torch', torch.__version__)"
pip show executorch | grep -i '^version'

if [ ! -f "$CKPT/original/consolidated.00.pth" ]; then
  if [ -z "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN is not set. Accept the licence at https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct,"
    echo "create a read token, then re-run with the token in your environment (it is never written to the repo)."
    exit 2
  fi
  hf download meta-llama/Llama-3.2-1B-Instruct --include "original/*" --local-dir "$CKPT"
fi
ls -l "$CKPT/original"

start=$(date +%s)
python -m executorch.extension.llm.export.export_llm --config /work/twin/export/llama3_2_1b_8da4w.yaml
echo "export took $(( $(date +%s) - start )) s"

cp "$CKPT/original/tokenizer.model" "$OUT/tokenizer.model"
ls -l "$OUT"/*.pte "$OUT/tokenizer.model"
sha256sum "$OUT"/*.pte
