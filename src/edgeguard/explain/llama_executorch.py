"""Llama explainer running on-device through ExecuTorch.

v1 integration path: call ExecuTorch's C++ Llama runner binary (`llama_main`,
built from executorch/examples/models/llama) as a subprocess with the exported
`.pte` model + tokenizer. This is the most portable route on a Raspberry Pi 5
(aarch64) and keeps the Python app free of heavy dependencies.

!! VERIFY against the current ExecuTorch docs before relying on it:
   the runner binary name, its CLI flag names, and its stdout format can change
   between ExecuTorch releases. Flags used below: --model_path, --tokenizer_path,
   --prompt, --temperature, --seq_len, --cpu_threads.  See docs/EXECUTORCH_NOTES.md.

If the runner or model is missing, `available()` is False and the pipeline uses
the deterministic template explanation, so the app always works.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


@dataclass
class LlamaConfig:
    runner: str = os.environ.get("EDGEGUARD_LLAMA_RUNNER", str(ROOT / "models" / "llama" / "llama_main"))
    model_path: str = os.environ.get("EDGEGUARD_LLAMA_PTE", str(ROOT / "models" / "llama" / "llama3_2_1b_8da4w.pte"))
    tokenizer_path: str = os.environ.get("EDGEGUARD_LLAMA_TOKENIZER", str(ROOT / "models" / "llama" / "tokenizer.model"))
    seq_len: int = int(os.environ.get("EDGEGUARD_LLAMA_SEQ_LEN", "512"))   # prompt + generated tokens
    temperature: float = 0.0          # greedy: stable, reproducible explanations
    cpu_threads: int = int(os.environ.get("EDGEGUARD_LLAMA_THREADS", "4"))  # Pi 5 has 4 Cortex-A76 cores
    timeout_s: int = 120


@dataclass
class LlamaResult:
    text: str
    wall_ms: float
    raw_stdout: str
    stats: dict


class LlamaExplainer:
    def __init__(self, cfg: LlamaConfig | None = None):
        self.cfg = cfg or LlamaConfig()

    def available(self) -> tuple[bool, str]:
        c = self.cfg
        runner = shutil.which(c.runner) or (c.runner if Path(c.runner).is_file() else None)
        if not runner:
            return False, f"runner not found: {c.runner}"
        if not Path(c.model_path).is_file():
            return False, f"model .pte not found: {c.model_path}"
        if not Path(c.tokenizer_path).is_file():
            return False, f"tokenizer not found: {c.tokenizer_path}"
        return True, "ok"

    def generate(self, prompt: str) -> LlamaResult:
        c = self.cfg
        cmd = [
            c.runner,
            f"--model_path={c.model_path}",
            f"--tokenizer_path={c.tokenizer_path}",
            f"--prompt={prompt}",
            f"--temperature={c.temperature}",
            f"--seq_len={c.seq_len}",
            f"--cpu_threads={c.cpu_threads}",
        ]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=c.timeout_s)
        wall = (time.perf_counter() - t0) * 1000
        if proc.returncode != 0:
            raise RuntimeError(f"llama runner failed ({proc.returncode}): {proc.stderr[-500:]}")
        return LlamaResult(self._extract(proc.stdout, prompt), wall, proc.stdout, self._stats(proc.stdout + proc.stderr))

    @staticmethod
    def _extract(stdout: str, prompt: str) -> str:
        """The runner echoes the prompt then streams the completion; keep only the completion."""
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
        """Best-effort parse of the runner's stats line (format may vary by version)."""
        m = re.search(r"PyTorchObserver\s*(\{.*\})", text)
        if not m:
            return {}
        import json
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            return {}
