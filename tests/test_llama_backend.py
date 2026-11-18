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
