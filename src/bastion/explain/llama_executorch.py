"""Llama 3.2 1B explainer running on-device through ExecuTorch (XNNPACK, Arm CPU).

Two backends, same interface (`available()`, `generate()` -> LlamaResult):

  python      (preferred) ExecuTorch's Python LLM runner, `executorch.extension.llm.runner.TextLLMRunner`
              (ExecuTorch 1.6.0). The .pte is loaded ONCE and stays in memory, so each explanation only
              pays prefill + decode, and tokens can be streamed to the UI through a callback.
  runtime     (twin / Pi default) generic `executorch.runtime` + `pytorch_tokenizers` from the PyPI wheel, with a
              greedy decode loop in Python (RuntimeLLM). Also persistent and streaming; no source build needed.
  subprocess  ExecuTorch's C++ runner `llama_main` (examples/models/llama/main.cpp @ v1.6.0;
              flags --model_path --tokenizer_path --prompt --temperature --seq_len --max_new_tokens
              --cpu_threads). Restarts the runner per call (pays model load every time). Fallback.

BASTION_LLAMA_BACKEND = auto (default: python > runtime > subprocess) | python | runtime | subprocess.
If neither backend is usable, `available()` is False and the pipeline uses the template explanation.
The LLM never sees or changes the verdict logic: see explain/guard.py and pipeline.py.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from bastion.paths import ROOT  # noqa: E402

_LLAMA = ROOT / "models" / "llama"


def _env(name: str, default: str) -> str:
    return os.environ.get(f"BASTION_LLAMA_{name}", default)


# Our own export (twin/export/llama3_2_1b_8da4w.yaml) is preferred; the official ExecuTorch-community pre-export
# (Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8, linked from the ExecuTorch v1.6.0 llama README) is the fallback.
PTE_CANDIDATES = ("llama3_2_1b_8da4w.pte", "llama3_2_1b_spinquant.pte")


def _default_pte() -> str:
    for name in PTE_CANDIDATES:
        if (_LLAMA / name).is_file():
            return str(_LLAMA / name)
    return str(_LLAMA / PTE_CANDIDATES[0])


@dataclass
class LlamaConfig:
    backend: str = field(default_factory=lambda: _env("BACKEND", "auto"))
    runner: str = field(default_factory=lambda: _env("RUNNER", str(_LLAMA / "llama_main")))
    model_path: str = field(default_factory=lambda: _env("PTE", _default_pte()))
    tokenizer_path: str = field(default_factory=lambda: _env("TOKENIZER", str(_LLAMA / "tokenizer.model")))
    seq_len: int = field(default_factory=lambda: int(_env("SEQ_LEN", "768")))       # must be <= export max_seq_length
    max_new_tokens: int = field(default_factory=lambda: int(_env("MAX_NEW_TOKENS", "160")))
    temperature: float = 0.0          # greedy: stable, reproducible explanations
    cpu_threads: int = field(default_factory=lambda: int(_env("THREADS", "4")))   # Pi 5: 4x Cortex-A76
    # Under QEMU emulation an explanation can take minutes; raise via BASTION_LLAMA_TIMEOUT_S.
    timeout_s: int = field(default_factory=lambda: int(_env("TIMEOUT_S", "120")))


@dataclass
class LlamaResult:
    text: str
    wall_ms: float
    raw_stdout: str
    stats: dict
    backend: str = ""


def python_runner_importable() -> bool:
    """TextLLMRunner needs a source build with EXECUTORCH_BUILD_PYBIND=ON. VERIFIED 18 Nov 2026: the PyPI aarch64
    wheel (1.6.0) raises 'LLM runner is not installed' on import, so on the twin/Pi `runtime` is used instead."""
    try:
        from executorch.extension.llm.runner import TextLLMRunner  # noqa: F401
        return True
    except Exception:
        return False


def runtime_importable() -> bool:
    """executorch.runtime (+ XNNPACK backend) and pytorch_tokenizers ship in the PyPI wheels (verified in the twin)."""
    try:
        from executorch.runtime import Runtime  # noqa: F401
        from pytorch_tokenizers import get_tokenizer  # noqa: F401
        return True
    except Exception:
        return False


class RuntimeLLM:
    """Greedy decode loop over an exported Llama .pte using the generic ExecuTorch runtime.

    The export (twin/export/llama3_2_1b_8da4w.yaml) uses a KV cache, so `forward(tokens[1, n], input_pos[1])`
    returns logits for the last position and updates the cache in place. Re-starting at input_pos 0 overwrites
    the cache, so every explanation starts from a clean context.
    """

    DEFAULT_EOS = (128009, 128001)   # <|eot_id|>, <|end_of_text|> (same as the export metadata)

    def __init__(self, model_path: str, tokenizer_path: str, threads: int):
        import torch
        from executorch.runtime import Runtime
        from pytorch_tokenizers import get_tokenizer
        try:   # size XNNPACK's pool to the device (QEMU/cgroups report the host's 32 cores)
            from executorch.extension.pybindings import portable_lib
            portable_lib._unsafe_reset_threadpool(threads)
        except Exception:
            pass
        # Register the LLM kernels exported Llama programs use (custom_sdpa, update_cache, fast_hadamard_transform for
        # SpinQuant, quantized embedding_byte). Without these, load_method("forward") fails with "Missing operator".
        # Same imports as ExecuTorch's own Python Llama runner. VERIFIED 23 Nov 2026 with executorch 1.6.0 wheels.
        import importlib
        for mod in ("executorch.extension.llm.custom_ops.custom_ops", "executorch.kernels.quantized"):
            try:
                importlib.import_module(mod)
            except Exception:
                pass
        self.torch = torch
        self.program = Runtime.get().load_program(model_path)
        self.forward = self.program.load_method("forward")
        self.tok = get_tokenizer(tokenizer_path)
        names = set(self.program.method_names)

        def const(name, default):
            if name not in names:
                return default
            try:
                out = self.program.load_method(name).execute([])
                return out[0] if len(out) == 1 else list(out)
            except Exception:
                return default

        self.max_seq_len = int(const("get_max_seq_len", 768))
        eos = const("get_eos_ids", list(self.DEFAULT_EOS))
        self.eos_ids = {int(e) for e in (eos if isinstance(eos, (list, tuple)) else [eos])}
        self.bos_id = int(const("get_bos_id", 128000))
        self.dynamic = bool(const("enable_dynamic_shape", True))

    def _encode(self, text: str) -> list[int]:
        """pytorch_tokenizers returns either the C++ Tiktoken (encode(text, bos, eos)) or the Python
        TiktokenTokenizer (encode(text, *, bos, eos, allowed_special)). allowed_special='all' turns the chat
        template's <|...|> markers into special tokens; this is safe because prompts.sanitize() strips every
        <|...|> sequence from the untrusted scanned text before it is placed in the prompt."""
        try:
            return list(self.tok.encode(text, bos=False, eos=False, allowed_special="all"))
        except TypeError:
            return list(self.tok.encode(text, 0, 0))

    def _decode(self, ids: list[int]) -> str:
        try:
            return self.tok.decode(ids)
        except TypeError:                                   # C++ tokenizer decodes one token at a time
            return "".join(self.tok.decode(i) for i in ids)

    def _step(self, ids: list[int], pos: int):
        t = self.torch
        out = self.forward.execute([t.tensor([ids], dtype=t.long), t.tensor([pos], dtype=t.long)])[0]
        return out.reshape(-1, out.shape[-1])[-1]          # logits of the last position

    def generate(self, prompt: str, max_new_tokens: int, seq_len: int,
                 on_token: Callable[[str], None] | None) -> tuple[str, dict]:
        t0 = time.perf_counter()
        ids = self._encode(prompt)                          # prompt already carries <|begin_of_text|>
        if not ids or ids[0] != self.bos_id:
            ids = [self.bos_id] + ids
        limit = min(seq_len, self.max_seq_len)
        if len(ids) >= limit:
            raise RuntimeError(f"prompt too long: {len(ids)} tokens >= seq_len {limit}")
        if self.dynamic:
            logits = self._step(ids, 0)
        else:
            for i, tid in enumerate(ids):
                logits = self._step([tid], i)
        t_first = time.perf_counter()
        pos, gen, emitted, n_gen = len(ids), [], "", 0
        while n_gen < max_new_tokens and pos < limit:
            nxt = int(self.torch.argmax(logits).item())     # temperature 0: greedy
            if nxt in self.eos_ids:
                break
            gen.append(nxt)
            n_gen += 1
            text = self._decode(gen)                        # incremental: multi-byte chars can span tokens
            if not text.endswith("�") and len(text) > len(emitted):
                piece, emitted = text[len(emitted):], text
                if on_token:
                    on_token(piece)
            logits = self._step([nxt], pos)
            pos += 1
        pieces = [self._decode(gen)] if gen else []
        t_end = time.perf_counter()
        decode_s = max(t_end - t_first, 1e-9)
        return "".join(pieces), {
            "prompt_tokens": len(ids), "generated_tokens": n_gen,
            "prefill_ms": round((t_first - t0) * 1000, 1),
            "decode_ms": round(decode_s * 1000, 1),
            "decode_tokens_per_s": round(n_gen / decode_s, 2) if n_gen else 0.0,
        }


class LlamaExplainer:
    def __init__(self, cfg: LlamaConfig | None = None):
        self.cfg = cfg or LlamaConfig()
        self._runner = None               # persistent TextLLMRunner (python backend)
        self._lock = threading.Lock()     # one model instance -> one generation at a time
        self.load_ms: float | None = None

    # ------------------------------------------------------------------ selection
    def _files_ok(self) -> tuple[bool, str]:
        if not Path(self.cfg.model_path).is_file():
            return False, f"model .pte not found: {self.cfg.model_path}"
        if not Path(self.cfg.tokenizer_path).is_file():
            return False, f"tokenizer not found: {self.cfg.tokenizer_path}"
        return True, "ok"

    def _runner_binary(self) -> str | None:
        r = self.cfg.runner
        return shutil.which(r) or (r if Path(r).is_file() else None)

    def backend(self) -> str | None:
        b = self.cfg.backend
        if b in ("python", "auto") and python_runner_importable():
            return "python"
        if b in ("runtime", "auto") and runtime_importable():
            return "runtime"
        if b in ("subprocess", "auto") and self._runner_binary():
            return "subprocess"
        return None

    def available(self) -> tuple[bool, str]:
        ok, why = self._files_ok()
        if not ok:
            return False, why
        b = self.backend()
        if b is None:
            return False, (f"no ExecuTorch runtime (backend={self.cfg.backend}): python/runtime not importable "
                           f"and runner binary not found: {self.cfg.runner}")
        return True, f"ok ({b} backend)"

    # ------------------------------------------------------------------ python backend
    def load(self) -> float:
        """Load the .pte once (python / runtime backends). Returns load time in ms."""
        if self._runner is None:
            t0 = time.perf_counter()
            if self.backend() == "runtime":
                self._runner = RuntimeLLM(self.cfg.model_path, self.cfg.tokenizer_path, self.cfg.cpu_threads)
            else:
                from executorch.extension.llm.runner import TextLLMRunner
                self._runner = TextLLMRunner(self.cfg.model_path, self.cfg.tokenizer_path)
            self.load_ms = round((time.perf_counter() - t0) * 1000, 1)
        return self.load_ms or 0.0

    @property
    def loaded(self) -> bool:
        return self._runner is not None

    def _generate_python(self, prompt: str, on_token: Callable[[str], None] | None) -> LlamaResult:
        from executorch.extension.llm.runner import GenerationConfig
        self.load()
        pieces: list[str] = []
        stats: dict = {}

        def tok(s: str) -> None:
            pieces.append(s)
            if on_token:
                on_token(s)

        def st(s) -> None:
            try:
                stats.update(json.loads(s.to_json_string()))
            except Exception:
                pass

        cfg = GenerationConfig(echo=False, max_new_tokens=self.cfg.max_new_tokens,
                               seq_len=self.cfg.seq_len, temperature=self.cfg.temperature)
        t0 = time.perf_counter()
        with self._lock:
            self._runner.reset()          # fresh KV cache per explanation (no cross-scan leakage)
            self._runner.generate(prompt, cfg, tok, st)
        wall = (time.perf_counter() - t0) * 1000
        raw = "".join(pieces)
        return LlamaResult(self._clean(raw), wall, raw, stats, "python")

    def _generate_runtime(self, prompt: str, on_token: Callable[[str], None] | None) -> LlamaResult:
        self.load()
        t0 = time.perf_counter()
        with self._lock:
            raw, stats = self._runner.generate(prompt, self.cfg.max_new_tokens, self.cfg.seq_len, on_token)
        wall = (time.perf_counter() - t0) * 1000
        return LlamaResult(self._clean(raw), wall, raw, stats, "runtime")

    # ------------------------------------------------------------------ subprocess backend
    def _generate_subprocess(self, prompt: str) -> LlamaResult:
        c = self.cfg
        cmd = [self._runner_binary() or c.runner,       # list args, never shell=True
               f"--model_path={c.model_path}", f"--tokenizer_path={c.tokenizer_path}",
               f"--prompt={prompt}", f"--temperature={c.temperature}", f"--seq_len={c.seq_len}",
               f"--max_new_tokens={c.max_new_tokens}", f"--cpu_threads={c.cpu_threads}"]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=c.timeout_s)
        wall = (time.perf_counter() - t0) * 1000
        if proc.returncode != 0:
            raise RuntimeError(f"llama runner failed ({proc.returncode}): {proc.stderr[-500:]}")
        return LlamaResult(self._extract(proc.stdout, prompt), wall, proc.stdout,
                           self._stats(proc.stdout + proc.stderr), "subprocess")

    # ------------------------------------------------------------------ public
    def generate(self, prompt: str, on_token: Callable[[str], None] | None = None) -> LlamaResult:
        b = self.backend()
        if b == "python":
            return self._generate_python(prompt, on_token)
        if b == "runtime":
            return self._generate_runtime(prompt, on_token)
        if b == "subprocess":
            return self._generate_subprocess(prompt)
        raise RuntimeError("no ExecuTorch runtime available")

    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r"<\|[^|>]*\|>", "", text)
        return text.strip()

    @staticmethod
    def _extract(stdout: str, prompt: str) -> str:
        """llama_main echoes the prompt then streams the completion; keep only the completion."""
        out = stdout
        marker = "<|start_header_id|>assistant<|end_header_id|>"
        if marker in out:
            out = out.rsplit(marker, 1)[1]
        elif prompt in out:
            out = out.split(prompt, 1)[1]
        out = re.sub(r"<\|[^|>]*\|>", "", out)
        lines = [ln for ln in out.splitlines()
                 if not ln.startswith(("PyTorchObserver", "I ", "W ", "E "))]
        return "\n".join(lines).strip()

    @staticmethod
    def _stats(text: str) -> dict:
        """Best-effort parse of the runner's PyTorchObserver stats line."""
        m = re.search(r"PyTorchObserver\s*(\{.*\})", text)
        if not m:
            return {}
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            return {}
