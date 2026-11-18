"""Llama 3.2 1B explainer running on-device through ExecuTorch (XNNPACK, Arm CPU).

Two backends, same interface (`available()`, `generate()` -> LlamaResult):

  python      (preferred) ExecuTorch's Python LLM runner, `executorch.extension.llm.runner.TextLLMRunner`
              (ExecuTorch 1.6.0). The .pte is loaded ONCE and stays in memory, so each explanation only
              pays prefill + decode, and tokens can be streamed to the UI through a callback.
  subprocess  ExecuTorch's C++ runner `llama_main` (examples/models/llama/main.cpp @ v1.6.0;
              flags --model_path --tokenizer_path --prompt --temperature --seq_len --max_new_tokens
              --cpu_threads). Restarts the runner per call (pays model load every time). Fallback.

BASTION_LLAMA_BACKEND = auto (default) | python | subprocess.
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


@dataclass
class LlamaConfig:
    backend: str = field(default_factory=lambda: _env("BACKEND", "auto"))
    runner: str = field(default_factory=lambda: _env("RUNNER", str(_LLAMA / "llama_main")))
    model_path: str = field(default_factory=lambda: _env("PTE", str(_LLAMA / "llama3_2_1b_8da4w.pte")))
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
    try:
        from executorch.extension.llm.runner import TextLLMRunner  # noqa: F401
        return True
    except Exception:
        return False


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
        if b in ("subprocess", "auto") and self._runner_binary():
            return "subprocess"
        return None

    def available(self) -> tuple[bool, str]:
        ok, why = self._files_ok()
        if not ok:
            return False, why
        b = self.backend()
        if b is None:
            return False, (f"no ExecuTorch runtime (backend={self.cfg.backend}): python runner not importable "
                           f"and runner binary not found: {self.cfg.runner}")
        return True, f"ok ({b} backend)"

    # ------------------------------------------------------------------ python backend
    def load(self) -> float:
        """Load the .pte once (python backend). Returns load time in ms."""
        if self._runner is None:
            from executorch.extension.llm.runner import TextLLMRunner
            t0 = time.perf_counter()
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
