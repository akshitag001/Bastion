#!/usr/bin/env bash
# Build ExecuTorch's C++ Llama runner (llama_main) natively on a Raspberry Pi 5 (aarch64).
#
# !!! UNTESTED TEMPLATE - VERIFY AGAINST CURRENT EXECUTORCH DOCS !!!
#   https://github.com/pytorch/executorch/tree/main/examples/models/llama  ("Build and run on ... Linux")
#   CMake option names and presets change between releases. Use the SAME ExecuTorch
#   version/tag here as the one you exported the .pte with.
#
# Native build on the Pi is slow (expect a long build; use a fan + a good power supply).
# Alternative: cross-compile on a PC with an aarch64 toolchain.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ET_DIR="${ET_DIR:-$ROOT/third_party/executorch}"
JOBS="${JOBS:-4}"

sudo apt-get update
sudo apt-get install -y build-essential cmake git python3-venv python3-dev

if [ ! -d "$ET_DIR" ]; then
  git clone --recursive https://github.com/pytorch/executorch.git "$ET_DIR"
fi
cd "$ET_DIR"

# 1. Core ExecuTorch libs with XNNPACK + LLM extensions
cmake -DCMAKE_INSTALL_PREFIX=cmake-out \
      -DCMAKE_BUILD_TYPE=Release \
      -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
      -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
      -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
      -DEXECUTORCH_BUILD_EXTENSION_LLM=ON \
      -DEXECUTORCH_BUILD_XNNPACK=ON \
      -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
      -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
      -DEXECUTORCH_BUILD_KERNELS_LLM=ON \
      -Bcmake-out .
cmake --build cmake-out -j"$JOBS" --target install --config Release

# 2. Llama runner
cmake -DCMAKE_INSTALL_PREFIX=cmake-out \
      -DCMAKE_BUILD_TYPE=Release \
      -DEXECUTORCH_BUILD_XNNPACK=ON \
      -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
      -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
      -Bcmake-out/examples/models/llama examples/models/llama
cmake --build cmake-out/examples/models/llama -j"$JOBS" --config Release

mkdir -p "$ROOT/models/llama"
cp cmake-out/examples/models/llama/llama_main "$ROOT/models/llama/llama_main"
echo "Runner -> $ROOT/models/llama/llama_main"
echo "Check:  cd $ROOT && PYTHONPATH=src python3 -m edgeguard.cli status"
