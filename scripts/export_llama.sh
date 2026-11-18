#!/usr/bin/env bash
# Export Llama 3.2 1B Instruct to an ExecuTorch .pte (XNNPACK, 4-bit weights / 8-bit dyn. activations)
# and build the C++ llama runner. Run on your DEV MACHINE (Linux/macOS, >=16 GB RAM recommended).
#
# !!! UNTESTED TEMPLATE - VERIFY EVERY COMMAND AGAINST THE CURRENT EXECUTORCH DOCS !!!
#   ExecuTorch Llama guide: https://github.com/pytorch/executorch/tree/main/examples/models/llama
#   The export module path, flag names and build steps change between releases.
#   Record the exact ExecuTorch version + commands you used in docs/EXECUTORCH_NOTES.md.
#
# Prerequisites (manual):
#   1. Accept the Llama 3.2 licence on Hugging Face (meta-llama/Llama-3.2-1B-Instruct) or llama.com.
#   2. Download the ORIGINAL Meta checkpoint: consolidated.00.pth, params.json, tokenizer.model
#      e.g.  huggingface-cli download meta-llama/Llama-3.2-1B-Instruct --include "original/*" --local-dir models/llama/hf
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_DIR="$ROOT/models/llama"
CKPT_DIR="${CKPT_DIR:-$LLAMA_DIR/hf/original}"
ET_DIR="${ET_DIR:-$ROOT/third_party/executorch}"
OUT_NAME="llama3_2_1b_8da4w.pte"

mkdir -p "$LLAMA_DIR" "$ROOT/third_party"

# --- 1. Get ExecuTorch (pin a release tag you have verified) -------------------
if [ ! -d "$ET_DIR" ]; then
  git clone --recursive https://github.com/pytorch/executorch.git "$ET_DIR"
  # cd "$ET_DIR" && git checkout <verified-release-tag> && git submodule update --init --recursive
fi
cd "$ET_DIR"
# Installs ExecuTorch python package + deps (check docs: script name may differ)
./install_executorch.sh
# Llama example requirements
./examples/models/llama/install_requirements.sh

# --- 2. Export to .pte ----------------------------------------------------------
# -kv: KV cache, --use_sdpa_with_kv_cache: fused attention, -X: XNNPACK delegate,
# -qmode 8da4w: int8 dynamic activations + int4 grouped weights, -d fp32: dtype for non-quantised parts
python -m examples.models.llama.export_llama \
  --model llama3_2 \
  --checkpoint "$CKPT_DIR/consolidated.00.pth" \
  --params "$CKPT_DIR/params.json" \
  -kv --use_sdpa_with_kv_cache -X \
  -qmode 8da4w --group_size 128 -d fp32 \
  --max_seq_length 512 \
  --metadata '{"get_bos_id":128000, "get_eos_ids":[128009, 128001]}' \
  --output_name "$LLAMA_DIR/$OUT_NAME"

cp "$CKPT_DIR/tokenizer.model" "$LLAMA_DIR/tokenizer.model"
ls -lh "$LLAMA_DIR/$OUT_NAME"
echo "Exported -> $LLAMA_DIR/$OUT_NAME"
echo "Next: build llama_main for aarch64 ON THE PI (scripts/build_runner_pi.sh), then copy the .pte + tokenizer to the Pi."
