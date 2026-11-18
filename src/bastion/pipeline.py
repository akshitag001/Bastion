"""End-to-end pipeline: input -> detector -> fusion -> (Llama) explanation -> advice.

The verdict in the final report is ALWAYS the detector's verdict. The explainer
contributes text only, and only after passing the guard.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from bastion.advice.remediation import advice_for
from bastion.detectors.sms_detector import SmsDetector
from bastion.detectors.url_detector import UrlDetector
from bastion.explain import fallback_templates, guard, prompts
from bastion.explain.llama_executorch import LlamaExplainer
from bastion.schema import Evidence


@dataclass
class Report:
    evidence: Evidence
    explanation: str
    actions: list[str]
    explainer: str                     # "llama-executorch" | "template"
    explainer_note: str = ""
    timings_ms: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "verdict": self.evidence.label,
            "risk": self.evidence.risk,
            "confidence": self.evidence.confidence,
            "explanation": self.explanation,
            "actions": self.actions,
            "explainer": self.explainer,
            "explainer_note": self.explainer_note,
            "evidence": self.evidence.to_dict(),
            "timings_ms": self.timings_ms,
        }


class Bastion:
    def __init__(self, use_llm: bool = True, url_model=..., sms_model=...):
        url_kwargs = {} if url_model is ... else {"model_path": url_model}
        sms_kwargs = {} if sms_model is ... else {"model_path": sms_model}
        self.url = UrlDetector(**url_kwargs)
        self.sms = SmsDetector(url_detector=self.url, **sms_kwargs)
        self.llm = LlamaExplainer() if use_llm else None
        self.stats = {"llm_calls": 0, "llm_fallbacks": 0}

    def scan_url(self, url: str, explain: bool = True) -> Report:
        return self._finish(self.url.analyze(url), explain)

    def scan_sms(self, text: str, explain: bool = True) -> Report:
        return self._finish(self.sms.analyze(text), explain)

    def detect(self, channel: str, text: str) -> Report:
        """Verdict only (instant). The explanation is the deterministic template until explain() runs."""
        ev = self.url.analyze(text) if channel == "url" else self.sms.analyze(text)
        return self._finish(ev, explain=False)

    def _finish(self, ev: Evidence, explain: bool) -> Report:
        advice = advice_for(ev)
        timings = {"detect": ev.latency_ms}
        if not explain:
            return Report(ev, fallback_templates.explain(ev), advice, "template", "explanation disabled", timings)
        return self.explain(ev, advice, timings)

    def explain(self, ev: Evidence, advice: list[str] | None = None, timings: dict | None = None,
                on_token: Callable[[str], None] | None = None) -> Report:
        """Llama explanation for fixed evidence. With on_token, guarded tokens stream as they are generated:
        the partial guard aborts generation (GuardStop) before unsafe text is ever passed on."""
        advice = advice if advice is not None else advice_for(ev)
        timings = dict(timings or {"detect": ev.latency_ms})
        note = "LLM disabled"
        if self.llm is not None:
            ok, why = self.llm.available()
            if not ok:
                note = f"Llama unavailable ({why})"
            else:
                self.stats["llm_calls"] += 1
                t0 = time.perf_counter()
                try:
                    prompt = prompts.llama3_chat(ev, advice)
                    if on_token is None:
                        res = self.llm.generate(prompt)
                    else:
                        buf: list[str] = []

                        def guarded(piece: str) -> None:
                            buf.append(piece)
                            reason = guard.partial_violation("".join(buf), ev)
                            if reason:
                                raise guard.GuardStop(reason)
                            on_token(piece)

                        res = self.llm.generate(prompt, on_token=guarded)
                    timings["explain"] = round((time.perf_counter() - t0) * 1000, 1)
                    if getattr(res, "stats", None):
                        timings["llm_stats"] = res.stats
                    g = guard.validate(res.text, ev)
                    if g.ok:
                        # Safety: approved deterministic advice always shown; LLM actions are extra.
                        actions = advice + [a for a in g.actions if a not in advice][:2]
                        return Report(ev, g.explanation, actions, "llama-executorch",
                                      "passed guard", timings)
                    note = f"LLM output rejected by guard: {g.reason}"
                except guard.GuardStop as e:   # streaming guard stopped generation early
                    timings["explain"] = round((time.perf_counter() - t0) * 1000, 1)
                    note = f"LLM output rejected by guard (while streaming): {e}"
                except Exception as e:  # runner crash / timeout -> never block the verdict
                    timings["explain"] = round((time.perf_counter() - t0) * 1000, 1)
                    note = f"LLM error: {e}"
                self.stats["llm_fallbacks"] += 1

        return Report(ev, fallback_templates.explain(ev), advice, "template", note, timings)
