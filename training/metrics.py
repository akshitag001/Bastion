"""Detection metrics + edge-cost measurements shared by the trainers."""

from __future__ import annotations

import io
import statistics
import time

import joblib
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def detection_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, proba), 4) if len(set(y_true)) > 1 else None,
        "fpr": round(fp / max(1, fp + tn), 4),
        "fnr": round(fn / max(1, fn + tp), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def threshold_for_recall(y_true, proba, target_recall: float = 0.95) -> float:
    """Highest threshold that still reaches the target recall (favours catching attacks)."""
    y_true = np.asarray(y_true).astype(int)
    pos = np.sort(np.asarray(proba)[y_true == 1])
    if len(pos) == 0:
        return 0.5
    k = int(np.floor((1 - target_recall) * len(pos)))
    return float(pos[min(k, len(pos) - 1)])


def model_size_bytes(obj) -> int:
    buf = io.BytesIO()
    joblib.dump(obj, buf, compress=3)
    return buf.tell()


def single_sample_latency(predict_fn, samples, runs: int = 300, warmup: int = 20) -> dict:
    """Latency of one-at-a-time inference (how the app actually calls it)."""
    for i in range(warmup):
        predict_fn([samples[i % len(samples)]])
    times = []
    for i in range(runs):
        t0 = time.perf_counter()
        predict_fn([samples[i % len(samples)]])
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {
        "runs": runs,
        "mean_ms": round(statistics.mean(times), 4),
        "p50_ms": round(times[len(times) // 2], 4),
        "p95_ms": round(times[int(len(times) * 0.95) - 1], 4),
        "p99_ms": round(times[int(len(times) * 0.99) - 1], 4),
    }
