"""Risk fusion: combine rule score + ML score into a 0-100 risk and a label.

Thresholds deliberately favour recall (missing a phishing attack is the
dangerous error). Tune them on a validation set with training/evaluate.py
and record the chosen values in the report.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Thresholds:
    suspicious: int = 35   # risk >= this -> SUSPICIOUS
    malicious: int = 70    # risk >= this -> MALICIOUS
    ml_weight: float = 0.7 # weight of ML score when a model is loaded


DEFAULT = Thresholds()


def combine(rule_score: float, ml_score: float | None, t: Thresholds = DEFAULT) -> float:
    """Return a 0-1 fused score."""
    if ml_score is None:
        return rule_score
    fused = t.ml_weight * ml_score + (1 - t.ml_weight) * rule_score
    # A very strong rule hit is never fully overruled by the ML model
    # (protects against ML blind spots on attack patterns not in the training data).
    return max(fused, rule_score * 0.9 if rule_score >= 0.8 else 0.0)


def label_for(risk: int, t: Thresholds = DEFAULT) -> str:
    if risk >= t.malicious:
        return "MALICIOUS"
    if risk >= t.suspicious:
        return "SUSPICIOUS"
    return "SAFE"


def confidence_for(score: float, t: Thresholds = DEFAULT) -> float:
    """Crude confidence: how far the score is from the nearest decision boundary,
    scaled to 0.5-1.0. Replace with calibrated probabilities later (M4)."""
    boundaries = [t.suspicious / 100, t.malicious / 100]
    dist = min(abs(score - b) for b in boundaries)
    return round(min(1.0, 0.5 + dist * 1.5), 2)
