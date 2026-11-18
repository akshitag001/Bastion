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


def align_ml(p: float, threshold: float | None) -> float:
    """Map a calibrated ML probability so that the model's recall-targeted operating threshold
    (chosen on the validation split, stored in the model bundle) lands exactly on 0.5.

    With the default weights, ML alone at 0.5 gives fused = 0.7 * 0.5 = 0.35 = the SUSPICIOUS boundary,
    so every input the ML model flags at its operating point is at least SUSPICIOUS. Rules can only add,
    which is why the end-to-end recall is never below the validated ML recall.
    """
    if threshold is None or not 0.0 < threshold < 1.0:
        return p
    if p < threshold:
        return 0.5 * p / threshold
    return 0.5 + 0.5 * (p - threshold) / (1.0 - threshold)


def confidence_for(score: float, t: Thresholds = DEFAULT, ml_prob: float | None = None,
                   label: str | None = None) -> float:
    """With a calibrated ML model: the calibrated probability that the verdict is right
    (P(scam) for SUSPICIOUS/MALICIOUS, 1 - P(scam) for SAFE). A low value means the ML model disagrees
    with a rule-driven verdict, which is useful to show. Rules-only: distance from the nearest boundary."""
    if ml_prob is not None and label is not None:
        return round(ml_prob if label != "SAFE" else 1.0 - ml_prob, 2)
    boundaries = [t.suspicious / 100, t.malicious / 100]
    dist = min(abs(score - b) for b in boundaries)
    return round(min(1.0, 0.5 + dist * 1.5), 2)
