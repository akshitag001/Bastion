# ExecuTorch + Llama notes

Status legend: **VERIFIED** = checked against the pinned source/PyPI or run in this repo. **PENDING** = not run yet.

## Pinned versions (VERIFIED on PyPI, 18 Nov 2026)
| Package | Version | Why |
|---|---|---|
| executorch | **1.6.0** (released 2026-11-04) | Ships `manylinux_2_28` wheels for **aarch64** and x86_64 (cp311-cp315), so the twin and the Pi can `pip install` it instead of compiling ExecuTorch. Declares `torch==2.15.0` properly. (1.5.1 omitted torch from its dependencies and pulled test-only packages, so we skipped it.) |
| torch | **2.15.0+cpu** from `https://download.pytorch.org/whl/cpu` | PyPI's default torch 2.15 wheels pull CUDA libraries on **both** x86_64 and aarch64 (observed: `nvidia-cublas` 513 MB for aarch64). The `+cpu` builds exist for both architectures. |
| torchao | 0.18.0 (pulled by executorch) | Warnings about `_C_mxfp8` / `_C_cutlass_90a` failing to load are CUDA kernels; harmless on CPU. |

The **same ExecuTorch version** is used for export (x86 container) and runtime (arm64 twin / Pi), since a `.pte` is tied to its runtime version.

## Export (VERIFIED commands; first full run PENDING the HF token)
Docs read at tag v1.6.0: `examples/models/llama/README.md`, `examples/models/llama/config/llama_q8da4w.yaml`,
`extension/llm/export/config/llm_config.py` (every key we set exists in `LlmConfig`).

- The old `python -m examples.models.llama.export_llama -kv -X -qmode ...` flags from the v0.1 template are **stale**.
  Current entry point: `python -m executorch.extension.llm.export.export_llm --config <yaml>` (Hydra).
  Verified working (`--help`) in `bastion-llama-export:dev` (torch 2.15.0+cpu, executorch 1.6.0).
- All settings live in a versioned file: [`twin/export/llama3_2_1b_8da4w.yaml`](../twin/export/llama3_2_1b_8da4w.yaml)
  (llama3_2, KV cache + SDPA, fp32 dtype, `8da4w` group 128, 4-bit embeddings (group 32), XNNPACK with extended ops, max seq 768).
- Run: `python twin/twinctl.py export` (builds the x86 image, downloads the original Meta checkpoint with `hf download`,
  exports, copies `tokenizer.model`, prints size and sha256).

## Runtime
**Python runner (preferred, persistent):** `executorch.extension.llm.runner.TextLLMRunner(model_path, tokenizer_path)`,
`.generate(prompt, GenerationConfig(echo=False, max_new_tokens=..., seq_len=..., temperature=0.0), token_callback, stats_callback)`,
`.reset()`. API read from `extension/llm/runner/_llm_runner.pyi` @ v1.6.0. Whether the **PyPI aarch64 wheel** includes the
compiled `_llm_runner` module: PENDING (being checked in the twin build).

**C++ runner (fallback):** `llama_main`, built with `cmake --workflow llm-release` then
`cmake --workflow --preset llama-release` in `examples/models/llama`. Flags in `main.cpp` @ v1.6.0: `--model_path`,
`--data_paths`, `--tokenizer_path`, `--prompt`, `--prompt_file`, `--temperature`, `--seq_len`, `--max_new_tokens`,
`--cpu_threads`, `--num_bos`, `--num_eos`, `--warmup`, `--ignore_eos`, `--etdump_path`, `--method_name`.

## Pre-exported fallback (only if our export is blocked)
The v1.6.0 README links official pre-exported Llama 3.2 1B Instruct `.pte` files on Hugging Face
(`executorch-community/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8-ET`, `...-QLORA_INT4_EO8-ET`). Still under the Llama licence.
Must be checked for runtime-version compatibility before use; if used, the report must say so.

## Measurements (PENDING, record with environment tag)
- [ ] `.pte` size, sha256
- [ ] load time, time-to-first-token, tokens/s, peak RAM, explanation latency: per environment tag
      (`arm64-emulated-qemu` numbers are functional proof only, never Pi 5 performance)
- [ ] persistent (python) vs per-call (subprocess) latency: before/after
