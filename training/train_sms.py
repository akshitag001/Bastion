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
from metrics import (detection_metrics, model_size_bytes,  # noqa: E402
                     single_sample_latency, threshold_for_recall)

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
        thr = threshold_for_recall(y_va, model.predict_proba(X_va)[:, 1], a.target_recall)
        proba = model.predict_proba(X_te)[:, 1]
        results[name] = {
            "train_seconds": round(train_s, 2),
            "test_at_0.5": detection_metrics(y_te, proba, 0.5),
            "test_at_recall_threshold": detection_metrics(y_te, proba, thr),
            "size_bytes": model_size_bytes(model),
            "latency_single": single_sample_latency(model.predict_proba, X_te[:500]),
        }
        fitted[name] = model
        m = results[name]["test_at_0.5"]
        print(f"{name:20s} F1={m['f1']:.4f} recall={m['recall']:.4f} FPR={m['fpr']:.4f} AUC={m['roc_auc']} "
              f"size={results[name]['size_bytes'] / 1024:.0f}KB p50={results[name]['latency_single']['p50_ms']}ms")

    best = max(results, key=lambda k: (results[k]["test_at_0.5"]["f1"], -results[k]["size_bytes"]))
    print(f"selected: {best}")
    # Refit the selected model on train+val (same hyper-params) before saving.
    final = fitted[best].fit(X_tr, y_tr)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "name": f"sms-{best}", "trained_on": Path(a.uci or a.csv).name}, a.out, compress=3)

    report = {
        "task": "sms", "dataset": Path(a.uci or a.csv).name, "seed": SEED,
        "rows_after_dedup": len(df), "scam": int(df.label.sum()),
        "split": {"train": len(X_tr2), "val": len(X_va), "test": len(X_te), "stratified": True},
        "note": "test metrics are from models fit on train only (before the train+val refit)",
        "target_recall": a.target_recall, "models": results, "selected": best,
        "machine": {"platform": platform.platform(), "machine": platform.machine(),
                    "python": platform.python_version()},
    }
    out_json = Path(a.results_dir) / f"sms_train_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))
    print(f"saved model -> {a.out}\nsaved report -> {out_json}")


if __name__ == "__main__":
    main()
