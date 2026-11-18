"""Scam / phishing SMS detector = indicator rules + optional ML text model.
Any URL found inside the message is also scored by the URL detector."""

from __future__ import annotations

import time
from pathlib import Path

from edgeguard.detectors.url_detector import UrlDetector
from edgeguard.features import sms_features as sf
from edgeguard.fusion import risk as fusion
from edgeguard.schema import Evidence

DEFAULT_MODEL = Path(__file__).resolve().parents[3] / "models" / "sms_model.joblib"

WEIGHTS = {
    "urgency": 0.15, "threat": 0.20, "credential_bait": 0.30, "reward_bait": 0.20,
    "financial": 0.10, "delivery_bait": 0.10, "call_to_action": 0.10,
    "remote_access": 0.35, "contains_url": 0.15, "contains_phone": 0.05,
    "mentions_money": 0.10, "shouting": 0.05,
}


class SmsDetector:
    def __init__(self, model_path: str | Path | None = DEFAULT_MODEL,
                 url_detector: UrlDetector | None = None,
                 thresholds: fusion.Thresholds = fusion.DEFAULT):
        self.t = thresholds
        self.url = url_detector or UrlDetector(thresholds=thresholds)
        self.model = None
        self.model_name = "rules-only"
        if model_path and Path(model_path).exists():
            import joblib
            bundle = joblib.load(model_path)  # trusted: produced by training/train_sms.py
            self.model = bundle["model"]
            self.model_name = bundle.get("name", "ml")

    def rule_score(self, hits: dict) -> float:
        score = sum(WEIGHTS.get(k, 0.0) for k in hits)
        # Combination bonus: classic scam = (pressure or bait) + action channel
        pressure = any(k in hits for k in ("urgency", "threat", "credential_bait", "reward_bait"))
        channel = any(k in hits for k in ("contains_url", "contains_phone", "remote_access"))
        if pressure and channel:
            score += 0.15
        return min(1.0, score)

    def analyze(self, text: str) -> Evidence:
        t0 = time.perf_counter()
        msg = sf.clean(text)
        hits = sf.indicators(msg)
        rule = self.rule_score(hits)

        ml = None
        if self.model is not None:
            ml = float(self.model.predict_proba([msg])[0][1])

        sub = [self.url.analyze(u) for u in hits.get("contains_url", [])[:5]]
        fused = fusion.combine(rule, ml, self.t)
        if sub:  # a malicious link makes the whole message at least that risky
            fused = max(fused, max(e.risk for e in sub) / 100 * 0.95)

        risk = int(round(fused * 100))
        reasons = []
        for k, v in hits.items():
            lbl = sf.INDICATOR_LABELS.get(k, k)
            if k in ("contains_url", "contains_phone", "mentions_money"):
                reasons.append(f"{lbl}: {', '.join(v[:2])}")
            elif k != "shouting":
                reasons.append(f"{lbl} (\"{v[0]}\")")
            else:
                reasons.append(lbl)
        for e in sub:
            if e.label != "SAFE":
                reasons.append(f"Embedded link rated {e.label} (risk {e.risk})")
        if ml is not None and ml >= 0.5:
            reasons.append(f"ML text model scores this as scam-like ({ml:.2f})")

        return Evidence(
            channel="sms", input=msg[:300], label=fusion.label_for(risk, self.t), risk=risk,
            confidence=fusion.confidence_for(fused, self.t), indicators=reasons,
            rule_score=round(rule, 3), ml_score=None if ml is None else round(ml, 3),
            model=self.model_name if self.model is not None else "rules-only",
            sub_evidence=sub, latency_ms=round((time.perf_counter() - t0) * 1000, 3),
        )
