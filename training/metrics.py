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
    """Latency of one-at-a-time inference, measured the way the app runs it: OpenMP limited to 1 thread
    (see bastion.detectors.url_detector._limit_openmp)."""
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1, user_api="openmp"):
        return _latency(predict_fn, samples, runs, warmup)


def _latency(predict_fn, samples, runs, warmup) -> dict:
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


def calibrate_and_threshold(model, X_cal, y_cal, X_thr, y_thr, target_recall: float, method: str = "isotonic"):
    """Calibrate a fitted model on one validation half and choose the recall-targeted operating threshold
    on the OTHER half (so the threshold is not fit on the calibration data). Returns (calibrated_model, threshold)."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator
    from sklearn.model_selection import train_test_split
    if len(set(np.asarray(y_cal).tolist())) < 2 or len(set(np.asarray(y_thr).tolist())) < 2:
        # tiny / grouped splits can leave one class in a half (seen only on synthetic smoke data): re-split stratified
        X_all = (list(X_cal) + list(X_thr)) if isinstance(X_cal, list) else np.concatenate([X_cal, X_thr])
        y_all = np.concatenate([y_cal, y_thr])
        if len(set(y_all.tolist())) < 2 or min(np.bincount(y_all.astype(int))) < 4:
            print("WARNING: validation split has too few examples of one class; calibration SKIPPED")
            return model, 0.5
        X_cal, X_thr, y_cal, y_thr = train_test_split(X_all, y_all, test_size=0.5, stratify=y_all, random_state=42)
    cal = CalibratedClassifierCV(FrozenEstimator(model), method=method).fit(X_cal, y_cal)
    thr = threshold_for_recall(y_thr, cal.predict_proba(X_thr)[:, 1], target_recall)
    return cal, thr


def end_to_end(analyze_fn, inputs, y_true) -> dict:
    """Evaluate the REAL detector (rules + ML + fusion) as the app runs it, one input at a time.
    Positive prediction = verdict SUSPICIOUS or MALICIOUS (the user is warned); also reported for MALICIOUS only."""
    from threadpoolctl import threadpool_limits
    y_true = np.asarray(y_true).astype(int)
    labels, risks, times = [], [], []
    with threadpool_limits(1, user_api="openmp"):
        for x in inputs:
            t0 = time.perf_counter()
            ev = analyze_fn(x)
            times.append((time.perf_counter() - t0) * 1000)
            labels.append(ev.label)
            risks.append(ev.risk / 100)
    warn = np.array([lbl != "SAFE" for lbl in labels], dtype=float)
    mal = np.array([lbl == "MALICIOUS" for lbl in labels], dtype=float)
    out_warn = detection_metrics(y_true, warn, 0.5)
    out_warn["roc_auc"] = round(roc_auc_score(y_true, risks), 4) if len(set(y_true)) > 1 else None
    times.sort()
    return {
        "warned_suspicious_or_malicious": out_warn,
        "malicious_only": detection_metrics(y_true, mal, 0.5),
        "verdict_counts": {k: labels.count(k) for k in ("SAFE", "SUSPICIOUS", "MALICIOUS")},
        "latency_ms": {"p50": round(times[len(times) // 2], 4), "p95": round(times[int(len(times) * 0.95) - 1], 4),
                       "mean": round(statistics.mean(times), 4)},
    }
