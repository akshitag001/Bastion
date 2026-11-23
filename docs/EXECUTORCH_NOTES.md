# ExecuTorch + Llama notes

Status legend: **VERIFIED** = checked against the pinned source/PyPI or run in this repo. **PENDING** = not run yet.

## Pinned versions (VERIFIED on PyPI, 18 Nov 2026)
| Package | Version | Why |
|---|---|---|
| executorch | **1.6.0** (released 2026-11-04) | Ships `manylinux_2_28` wheels for **aarch64** and x86_64 (cp311-cp315), so the twin and the Pi can `pip install` it instead of compiling ExecuTorch. Declares `torch==2.15.0` properly. (1.5.1 omitted torch from its dependencies and pulled test-only packages, so we skipped it.) |
| torch | **2.15.0+cpu** from `https://download.pytorch.org/whl/cpu` | PyPI's default torch 2.15 wheels pull CUDA libraries on **both** x86_64 and aarch64 (observed: `nvidia-cublas` 513 MB for aarch64). The `+cpu` builds exist for both architectures. |
| torchao | 0.18.0 (pulled by executorch) | Warnings about `_C_mxfp8` / `_C_cutlass_90a` failing to load are CUDA kernels; harmless on CPU. |

The **same ExecuTorch version** is used for export (x86 container) and runtime (arm64 twin / Pi), since a `.pte` is tied to its runtime version.

## Export: DONE (23 Nov 2026) with ExecuTorch 1.6.0
- Config: [`twin/export/llama3_2_1b_8da4w.yaml`](../twin/export/llama3_2_1b_8da4w.yaml) (`llama3_2`, KV cache + SDPA, fp32,
  `8da4w` group 128, 4-bit embeddings (group 32), XNNPACK with extended ops, max seq 768). Keys verified against v1.6.0 `LlmConfig`.
- Entry point: `python -m executorch.extension.llm.export.export_llm --config <yaml>` (the v0.1 `export_llama -kv -X -qmode` flags are stale).
  `export_llm` writes `output_name` relative to the **working directory**, so `twin/export/export_local.py` runs it inside `models/llama/`.
- **Weights:** Meta's gated repo (`meta-llama/Llama-3.2-1B-Instruct`, manual approval) had not approved our request by the
  deadline (HTTP 403). We used the same official Llama 3.2 1B Instruct weights redistributed under the Llama 3.2 Community
  License by `unsloth/Llama-3.2-1B-Instruct` (HF safetensors, sha256 `1ff795ff…`), converted to Meta format with the
  torchtune `hf_to_tune` + `tune_to_meta` mapping (RoPE un-permutation of q/k, tied output embedding) reimplemented in
  `export_local.py`, because torchtune 0.6.1 cannot be imported next to the torchao 0.18 that executorch 1.6.0 requires.
  The Llama 3 `tokenizer.model` is the standard one (from the ExecuTorch-community repo, sha256 `82e9d319…`).
- Run (Windows, no Docker): `python twin/export/export_local.py --hf-dir models/llama/hf-unsloth` in a venv with
  torch 2.15.0+cpu and executorch 1.6.0. Result (`models/llama/export_info.json`, not committed: weights are not redistributed here):

| Item | Value |
|---|---|
| `.pte` | `llama3_2_1b_8da4w.pte`, 802,167,040 bytes, sha256 `15fc1ef7dff74b07d8b02f7edb43841e486a03a99a54ccd51cc5f961bd8d4dda` |
| export time | ~230 s on the x86 dev machine (log timestamps 18:17:21 to 18:21:08) |
| activation memory planned | 132,210,496 bytes |
| sanity check | greedy "The capital of France is" -> " Paris. Paris is the largest city in" |

## Finding: the pre-exported community `.pte` does NOT work with the 1.6.0 runtime
`executorch-community/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8-ET` (linked from the v1.6.0 llama README, exported in 2024)
loads after registering the LLM kernels, but produces gibberish for any prompt (identical with bulk and token-by-token
prefill), while our own 1.6.0 export of the same model is correct. ExecuTorch guarantees compatibility for core operators,
not for the custom LLM kernels (`custom_sdpa`, `update_cache`, `fast_hadamard_transform`). **Always export with the same
ExecuTorch version as the runtime.**

## Runtime (VERIFIED with the real model)
- The PyPI wheels (aarch64 and x86) do **not** include the Python `TextLLMRunner` ("LLM runner is not installed"), but do
  include `executorch.runtime`, the XNNPACK backend and `pytorch_tokenizers`. Bastion's `runtime` backend
  (`src/bastion/explain/llama_executorch.py`) loads the program once, prefills the prompt in one call (dynamic shape),
  greedy-decodes token by token with streaming, and restarts the KV cache at position 0 for each explanation.
- **Kernel registration is required:** without `import executorch.extension.llm.custom_ops.custom_ops` and
  `import executorch.kernels.quantized`, `load_method("forward")` fails with "Missing operator: llama::custom_sdpa.out /
  llama::update_cache.out / quantized_decomposed::embedding_byte.dtype_out".
- **Tokenizer:** `get_tokenizer(tokenizer.model)` returns the Python `TiktokenTokenizer`; it must be called with
  `allowed_special="all"` so the Llama 3 chat-template markers become special tokens (safe: `prompts.sanitize()` strips
  `<|...|>` from untrusted text first). Decoding is incremental so multi-byte characters split across tokens are not garbled.
- XNNPACK thread pool set to 4 (`_unsafe_reset_threadpool`), matching the Pi 5 profile.

## Measurements (`benchmarks/results/edge_x86-native_20261123_182544.json`, environment `x86-native`)
| Metric | Value |
|---|---|
| model load (once, then resident) | 3,054 ms |
| explanation wall time p50 (5 runs, demo SMS) | 4,293 ms |
| prefill (≈ 380-405 prompt tokens) mean | 2,361 ms |
| decode throughput mean | 35.25 tokens/s |
| guard fallbacks | 1 of 5 (the prompt-injection SMS: the model drifted towards "safe", the guard rejected it, template shown) |

**Not measured yet:** Llama inside the arm64 twin (Docker Desktop's engine was unavailable on 23 Nov) and on a real
Raspberry Pi 5. x86 numbers are not Pi 5 numbers.

## Prompt notes (measured on real outputs)
- Rules-only prompt (2 sentences, < 50 words, no lists) gives short, on-topic but sometimes generic explanations.
- A one-shot example made the 1B model copy the example verbatim for unrelated scams, so it was removed.
