# ExecuTorch + Llama notes

> Status: **NOT YET VERIFIED.** Fill this in as you get each step working. Record exact versions.

## Plan
- Model: Llama 3.2 1B Instruct (smallest that gives readable explanations; 3B optional comparison).
- Export: `examples.models.llama.export_llama`, XNNPACK delegate, KV cache, SDPA, quant `8da4w` (int8 dynamic
  activations, int4 grouped weights, group size 128), max seq length 512.
- Runtime on Pi 5: C++ `llama_main` runner built natively for aarch64; app calls it via subprocess
  (`src/bastion/explain/llama_executorch.py`).

## Things to verify (and write the answers here)
- [ ] ExecuTorch version/tag used: ...
- [ ] Exact export command that worked: ...
- [ ] `.pte` size: ... MB
- [ ] Runner binary name and CLI flags (the wrapper assumes `--model_path --tokenizer_path --prompt --temperature --seq_len --cpu_threads`)
- [ ] Runner stdout format (the wrapper strips the echoed prompt and `PyTorchObserver` stats lines)
- [ ] Tokenizer file expected by the runner for Llama 3.2 (`tokenizer.model`)
- [ ] Measured on Pi: load time, time-to-first-token, tokens/s, peak RAM, explanation latency

## Reference
- https://github.com/pytorch/executorch/tree/main/examples/models/llama
- ExecuTorch docs: https://pytorch.org/executorch/
- The challenge page links an "ExecuTorch on Arm" course and Getting Started resources.
