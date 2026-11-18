"""Structured evidence passed from detectors -> fusion -> explainer.

This object is the ONLY thing the Llama explainer sees as the decision.
The verdict (label / risk) is fixed here and can never be changed by the LLM.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

LABELS = ("SAFE", "SUSPICIOUS", "MALICIOUS")


@dataclass
class Evidence:
    channel: str                      # "url" | "sms"
    input: str                        # what was scanned (truncated for display)
    label: str = "SAFE"               # SAFE / SUSPICIOUS / MALICIOUS  (detector-owned)
    risk: int = 0                     # 0-100                          (detector-owned)
    confidence: float = 0.0           # 0-1, distance from decision boundary
    indicators: list[str] = field(default_factory=list)   # human-readable reasons
    rule_score: float = 0.0           # 0-1 from the rule engine
    ml_score: float | None = None     # 0-1 from the ML model, None if no model loaded
    model: str = "rules-only"
    sub_evidence: list["Evidence"] = field(default_factory=list)  # e.g. URLs found inside an SMS
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sub_evidence"] = [s.to_dict() if isinstance(s, Evidence) else s for s in self.sub_evidence]
        return d
