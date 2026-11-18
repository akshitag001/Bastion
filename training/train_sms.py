"""Train + compare SMS scam/spam text classifiers.

Usage (UCI SMS Spam Collection, tab-separated, no header: label<TAB>text):
  python training/train_sms.py --uci data/raw/SMSSpamCollection
Or any CSV:
  python training/train_sms.py --csv my.csv --text-col text --label-col label --scam-value spam

Leakage control: the UCI set contains many exact / near-exact duplicates.
They are removed BEFORE splitting (after lower-casing + whitespace/digit
normalisation), otherwise copies land in both train and test.
Outputs: models/sms_model.joblib, benchmarks/results/sms_train_<timestamp>.json
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from metrics import (calibrate_and_threshold, detection_metrics, end_to_end,  # noqa: E402
                     model_size_bytes, single_sample_latency, threshold_for_recall)

SEED = 42


def norm_key(t: str) -> str:
    return re.sub(r"\d+", "0", re.sub(r"\s+", " ", t.lower())).strip()


def load(a) -> pd.DataFrame:
    if a.uci:
        df = pd.read_csv(a.uci, sep="\t", header=None, names=["raw_label", "text"],
                         quoting=3, dtype=str, keep_default_na=False)
        scam_value = "spam"
    else:
        df = pd.read_csv(a.csv, usecols=[a.text_col, a.label_col], dtype=str, keep_default_na=False)
        df = df.rename(columns={a.text_col: "text", a.label_col: "raw_label"})
        scam_value = a.scam_value
    df["label"] = (df["raw_label"].str.strip().str.lower() == scam_value.lower()).astype(int)
    before = len(df)
    df["key"] = df["text"].map(norm_key)
    df = df[df["text"].str.strip().str.len() > 0].drop_duplicates("key")
    print(f"loaded {before} rows -> {len(df)} after removing duplicates/near-duplicates")
    return df.reset_index(drop=True)


def tfidf():
    return make_union(
        TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True),
        TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                        max_features=30000, sublinear_tf=True),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--uci", help="path to UCI SMSSpamCollection file")
    g.add_argument("--csv")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--scam-value", default="spam")
    ap.add_argument("--target-recall", type=float, default=0.95)
    ap.add_argument("--out", default=str(ROOT / "models" / "sms_model.joblib"))
    ap.add_argument("--results-dir", default=str(ROOT / "benchmarks" / "results"))
    a = ap.parse_args()

    df = load(a)
    print(f"class balance: scam={df.label.sum()}  ham={(1 - df.label).sum()}")
    X, y = df["text"].tolist(), df["label"].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    X_tr2, X_va, y_tr2, y_va = train_test_split(X_tr, y_tr, test_size=0.15, stratify=y_tr, random_state=SEED)

    candidates = {
        "tfidf_logreg": make_pipeline(tfidf(), LogisticRegression(max_iter=3000, C=10, class_weight="balanced")),
        "tfidf_linearsvm": make_pipeline(tfidf(), CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3)),
        "tfidf_complement_nb": make_pipeline(tfidf(), ComplementNB(alpha=0.3)),
    }

    results, fitted = {}, {}
    for name, model in candidates.items():
        t0 = time.time()
        model.fit(X_tr2, y_tr2)
        train_s = time.time() - t0
        p_va = model.predict_proba(X_va)[:, 1]
        thr = threshold_for_recall(y_va, p_va, a.target_recall)
        proba = model.predict_proba(X_te)[:, 1]
        results[name] = {
            "train_seconds": round(train_s, 2),
            "val_roc_auc": detection_metrics(y_va, p_va)["roc_auc"],
            "test_at_0.5": detection_metrics(y_te, proba, 0.5),
            "test_at_recall_threshold": detection_metrics(y_te, proba, thr),
            "size_bytes": model_size_bytes(model),
            "latency_single": single_sample_latency(model.predict_proba, X_te[:500]),
        }
        fitted[name] = model
        m = results[name]["test_at_0.5"]
        print(f"{name:20s} F1={m['f1']:.4f} recall={m['recall']:.4f} FPR={m['fpr']:.4f} AUC={m['roc_auc']} "
              f"size={results[name]['size_bytes'] / 1024:.0f}KB p50={results[name]['latency_single']['p50_ms']}ms")

    # Model selection on VALIDATION ROC-AUC only; ties -> smaller model.
    best = max(results, key=lambda k: (results[k]["val_roc_auc"], -results[k]["size_bytes"]))
    print(f"selected (by validation ROC-AUC): {best}")
    # No refit on train+val: the operating threshold below is chosen for THIS fitted model.
    # Validation is small (~75 scams), so sigmoid (2 parameters) rather than isotonic calibration.
    va_a, va_b = train_test_split(list(range(len(X_va))), test_size=0.5, stratify=y_va, random_state=SEED)
    cal, op_thr = calibrate_and_threshold(fitted[best], [X_va[i] for i in va_a], y_va[va_a],
                                          [X_va[i] for i in va_b], y_va[va_b], a.target_recall, method="sigmoid")
    p_te_cal = cal.predict_proba(X_te)[:, 1]
    calibrated = {"method": "sigmoid", "calibration_rows": len(va_a), "threshold_rows": len(va_b),
                  "threshold_positives": int(y_va[va_b].sum()), "operating_threshold": round(op_thr, 6),
                  "test_at_operating_threshold": detection_metrics(y_te, p_te_cal, op_thr),
                  "brier_test_uncalibrated": round(float(np.mean((fitted[best].predict_proba(X_te)[:, 1] - y_te) ** 2)), 5),
                  "brier_test_calibrated": round(float(np.mean((p_te_cal - y_te) ** 2)), 5)}
    m = calibrated["test_at_operating_threshold"]
    print(f"calibrated {best}: threshold={op_thr:.4f} -> test recall={m['recall']} precision={m['precision']} "
          f"FPR={m['fpr']}  Brier {calibrated['brier_test_uncalibrated']} -> {calibrated['brier_test_calibrated']}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": cal, "name": f"sms-{best}-cal", "threshold": float(op_thr), "calibrated": True,
                 "target_recall": a.target_recall, "trained_on": Path(a.uci or a.csv).name}, a.out, compress=3)

    # End-to-end: the real SmsDetector (indicator rules + calibrated ML + embedded-URL detector + fusion).
    from bastion.detectors.sms_detector import SmsDetector
    det = SmsDetector(model_path=a.out)
    e2e = end_to_end(det.analyze, X_te, y_te)
    w = e2e["warned_suspicious_or_malicious"]
    print(f"END-TO-END detector on test (warned = SUSPICIOUS|MALICIOUS): recall={w['recall']} precision={w['precision']} "
          f"FPR={w['fpr']} F1={w['f1']} AUC(risk)={w['roc_auc']}  verdicts={e2e['verdict_counts']}  "
          f"p50={e2e['latency_ms']['p50']}ms")

    report = {
        "task": "sms", "dataset": Path(a.uci or a.csv).name, "seed": SEED,
        "rows_after_dedup": len(df), "scam": int(df.label.sum()),
        "split": {"train": len(X_tr2), "val": len(X_va), "test": len(X_te), "stratified": True},
        "note": "all models fit on the train split only; validation used for selection, calibration, threshold",
        "target_recall": a.target_recall, "models": results, "selected": best,
        "selection": "validation ROC-AUC", "calibration": calibrated, "end_to_end_test": e2e,
        "end_to_end_url_model": det.url.model_name,
        "machine": {"platform": platform.platform(), "machine": platform.machine(),
                    "python": platform.python_version()},
    }
    out_json = Path(a.results_dir) / f"sms_train_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))
    print(f"saved model -> {a.out}\nsaved report -> {out_json}")


if __name__ == "__main__":
    main()
