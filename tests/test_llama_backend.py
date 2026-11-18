"""Llama backend selection + persistent python runner, tested with a fake ExecuTorch module
(the real model is exercised in the twin; see docs/EXECUTORCH_NOTES.md)."""

import sys
import types

import pytest

from bastion.explain import llama_executorch as lx
from bastion.pipeline import Bastion


class _FakeStats:
    def to_json_string(self):
        return '{"num_generated_tokens": 3}'


class _FakeRunner:
    instances = 0

    def __init__(self, model_path, tokenizer_path, data_path=None):
        _FakeRunner.instances += 1
        self.resets = 0

    def reset(self):
        self.resets += 1

    def generate(self, prompt, config, token_callback=None, stats_callback=None):
        assert config.echo is False and config.temperature == 0.0
        for t in ["EXPLANATION: This link imitates a bank ", "and asks you to log in. ",
                  "<|eot_id|>"]:
            token_callback(t)
        stats_callback(_FakeStats())


class _FakeGenConfig:
    def __init__(self, **kw):
        self.__dict__.update(kw)


@pytest.fixture
def fake_executorch(monkeypatch, tmp_path):
    mod = types.ModuleType("executorch.extension.llm.runner")
    mod.TextLLMRunner, mod.GenerationConfig = _FakeRunner, _FakeGenConfig
    for name in ("executorch", "executorch.extension", "executorch.extension.llm"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "executorch.extension.llm.runner", mod)
    pte, tok = tmp_path / "m.pte", tmp_path / "tokenizer.model"
    pte.write_bytes(b"x"), tok.write_bytes(b"x")
    _FakeRunner.instances = 0
    return lx.LlamaConfig(backend="auto", model_path=str(pte), tokenizer_path=str(tok),
                          runner=str(tmp_path / "missing_llama_main"))


def test_python_backend_selected_and_model_loaded_once(fake_executorch):
    ex = lx.LlamaExplainer(fake_executorch)
    assert ex.available() == (True, "ok (python backend)")
    streamed = []
    r1 = ex.generate("p1", on_token=streamed.append)
    r2 = ex.generate("p2")
    assert _FakeRunner.instances == 1                 # persistent: loaded once
    assert ex._runner.resets == 2                      # fresh KV cache per explanation
    assert r1.backend == "python" and "<|" not in r1.text and r1.text.startswith("EXPLANATION:")
    assert len(streamed) == 3 and r1.stats["num_generated_tokens"] == 3
    assert r2.text == r1.text


def test_missing_files_not_available(fake_executorch):
    fake_executorch.model_path = "/nope.pte"
    assert lx.LlamaExplainer(fake_executorch).available()[0] is False


def test_subprocess_backend_forced_without_binary_is_unavailable(fake_executorch):
    fake_executorch.backend = "subprocess"
    ok, why = lx.LlamaExplainer(fake_executorch).available()
    assert not ok and "runner binary not found" in why


def test_pipeline_uses_persistent_llm_and_keeps_verdict(fake_executorch):
    eg = Bastion(use_llm=False, url_model=None, sms_model=None)
    eg.llm = lx.LlamaExplainer(fake_executorch)
    r = eg.scan_url("http://sbi-kyc-update.xyz/login")
    assert r.explainer == "llama-executorch"
    assert r.evidence.label in ("SUSPICIOUS", "MALICIOUS")   # detector-owned, unchanged by the LLM


# ---------------------------------------------------------------- runtime backend (decode loop)
import numpy as np  # noqa: E402

VOCAB, BOS, EOT = 130000, 128000, 128009
SCRIPT = [11, 12, 13, EOT]          # tokens the fake model "wants" to emit after the prompt


class _FakeMethod:
    def __init__(self, name, calls):
        self.name, self.calls = name, calls

    def execute(self, inputs):
        if self.name == "get_max_seq_len":
            return [64]
        if self.name == "get_eos_ids":
            return [EOT, 128001]
        if self.name == "get_bos_id":
            return [BOS]
        ids, pos = inputs[0].tolist()[0], int(inputs[1][0])
        self.calls.append((ids, pos))
        step = len(self.calls) - 1                 # 0 = prefill, then one call per generated token
        logits = np.zeros((1, VOCAB), dtype=np.float32)
        logits[0, SCRIPT[min(step, len(SCRIPT) - 1)]] = 1.0
        return [logits]


class _FakeProgram:
    method_names = ["forward", "get_max_seq_len", "get_eos_ids", "get_bos_id"]

    def __init__(self):
        self.calls = []

    def load_method(self, name):
        return _FakeMethod(name, self.calls)


class _FakeTok:
    def encode(self, text, bos, eos):
        return [BOS, 1, 2, 3]

    def decode(self, tid):
        return f"w{tid} "


@pytest.fixture
def fake_runtime(monkeypatch, tmp_path):
    prog = _FakeProgram()
    rt_mod = types.ModuleType("executorch.runtime")
    rt_mod.Runtime = type("Runtime", (), {"get": staticmethod(
        lambda: types.SimpleNamespace(load_program=lambda p: prog))})
    tok_mod = types.ModuleType("pytorch_tokenizers")
    tok_mod.get_tokenizer = lambda path: _FakeTok()
    pl = types.ModuleType("executorch.extension.pybindings.portable_lib")
    pl.threads = None
    pl._unsafe_reset_threadpool = lambda n: setattr(pl, "threads", n)
    torch_mod = types.ModuleType("torch")
    torch_mod.long = "int64"
    torch_mod.tensor = lambda x, dtype=None: np.array(x)
    torch_mod.argmax = lambda a: np.argmax(a)
    for name, mod in {"executorch": types.ModuleType("executorch"), "executorch.runtime": rt_mod,
                      "executorch.extension": types.ModuleType("executorch.extension"),
                      "executorch.extension.pybindings": types.ModuleType("executorch.extension.pybindings"),
                      "executorch.extension.pybindings.portable_lib": pl,
                      "pytorch_tokenizers": tok_mod, "torch": torch_mod}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delitem(sys.modules, "executorch.extension.llm.runner", raising=False)
    pte, tok = tmp_path / "m.pte", tmp_path / "tokenizer.model"
    pte.write_bytes(b"x"), tok.write_bytes(b"x")
    cfg = lx.LlamaConfig(backend="auto", model_path=str(pte), tokenizer_path=str(tok),
                         runner=str(tmp_path / "missing"), cpu_threads=4)
    return cfg, prog, pl


def test_runtime_backend_decode_loop(fake_runtime):
    cfg, prog, pl = fake_runtime
    ex = lx.LlamaExplainer(cfg)
    assert ex.backend() == "runtime"
    streamed = []
    r = ex.generate("prompt", on_token=streamed.append)
    assert r.backend == "runtime" and r.text == "w11 w12 w13"     # stops at <|eot_id|>, EOT not emitted
    assert streamed == ["w11 ", "w12 ", "w13 "]
    assert prog.calls[0] == ([BOS, 1, 2, 3], 0)                    # one prefill call at position 0
    assert [c[1] for c in prog.calls[1:]] == [4, 5, 6]             # KV positions advance one per token
    assert r.stats["generated_tokens"] == 3 and r.stats["prompt_tokens"] == 4
    assert pl.threads == 4                                         # XNNPACK pool sized to the Pi 5 profile
    ex.generate("again")
    assert prog.calls[4] == ([BOS, 1, 2, 3], 0)                    # next explanation restarts at pos 0


def test_runtime_backend_respects_max_new_tokens(fake_runtime):
    cfg, prog, _ = fake_runtime
    cfg.max_new_tokens = 2
    assert lx.LlamaExplainer(cfg).generate("p").stats["generated_tokens"] == 2
