"""Train + compare URL phishing classifiers on lexical features.

Usage:
  python training/train_url.py --csv data/raw/urls.csv --url-col URL --label-col label --phish-value 0
  (--phish-value = the value in label-col that means PHISHING; CHECK YOUR DATASET'S README.)

Leakage controls:
  * exact-duplicate URLs removed
  * train/test split GROUPED BY registered domain (same site never in both sets)
  * "trivial feature" probe: a model using only url_len + path_len + n_dots_host.
    If it scores near-perfect, the dataset's two classes differ in collection
    artefacts (e.g. benign = bare homepages) and headline metrics are inflated.
Outputs: models/url_model.joblib, benchmarks/results/url_train_<timestamp>.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training"))

from bastion.features import url_features as uf  # noqa: E402
from metrics import (calibrate_and_threshold, detection_metrics, end_to_end,  # noqa: E402
                     model_size_bytes, single_sample_latency, threshold_for_recall)

SEED = 42


def load(csv: str, url_col: str, label_col: str, phish_value: str, max_rows: int | None) -> pd.DataFrame:
    df = pd.read_csv(csv, usecols=[url_col, label_col], dtype=str, keep_default_na=False)
    df = df.rename(columns={url_col: "url", label_col: "raw_label"})
    df["label"] = (df["raw_label"].str.strip().str.lower() == str(phish_value).lower()).astype(int)
    before = len(df)
    df = df[df["url"].str.len() > 0].drop_duplicates("url")
    print(f"loaded {before} rows -> {len(df)} after dropping empty/duplicate URLs")
    if max_rows and len(df) > max_rows:
        # (groupby.apply drops the grouping column in pandas >= 2.2, so sample per group explicitly)
        df = pd.concat([g.sample(min(len(g), max_rows // 2), random_state=SEED)
                        for _, g in df.groupby("label")])
        print(f"subsampled to {len(df)} rows (balanced cap) for speed")
    df["domain"] = df["url"].map(lambda u: uf.parse(u)["registered_domain"] or u)
    return df.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--url-col", default="url")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--phish-value", default="1", help="label value meaning PHISHING")
    ap.add_argument("--view", choices=sorted(uf.VIEWS), default="host",
                    help="URL view the ML model sees. host = scheme://host/ (default; avoids the PhiUSIIL bare-homepage leak)")
    ap.add_argument("--max-rows", type=int, default=200_000)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--target-recall", type=float, default=0.95)
    ap.add_argument("--out", default=str(ROOT / "models" / "url_model.joblib"))
    ap.add_argument("--results-dir", default=str(ROOT / "benchmarks" / "results"))
    a = ap.parse_args()

    df = load(a.csv, a.url_col, a.label_col, a.phish_value, a.max_rows)
    print(f"class balance: phishing={df.label.sum()}  benign={(1 - df.label).sum()}")

    t0 = time.time()
    view = uf.VIEWS[a.view]
    X = np.array([uf.vector(view(u)) for u in df["url"]], dtype=np.float32)
    X_full = X if a.view == "full" else np.array([uf.vector(u) for u in df["url"]], dtype=np.float32)
    print(f"feature extraction: {len(df)} URLs in {time.time() - t0:.1f}s")
    y = df["label"].to_numpy()

    gss = GroupShuffleSplit(n_splits=1, test_size=a.test_size, random_state=SEED)
    tr, te = next(gss.split(X, y, groups=df["domain"]))
    # carve a validation set out of train (also domain-grouped) for threshold selection
    gss_v = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    tr2, va = next(gss_v.split(X[tr], y[tr], groups=df["domain"].iloc[tr]))
    tr_idx, va_idx = tr[tr2], tr[va]
    overlap = set(df.domain.iloc[tr]) & set(df.domain.iloc[te])
    print(f"train={len(tr_idx)} val={len(va_idx)} test={len(te)}  domain overlap train/test={len(overlap)}")

    candidates = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=16, n_jobs=-1,
                                                class_weight="balanced", random_state=SEED),
        "hist_gbdt": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                                    class_weight="balanced", random_state=SEED),
    }

    results, fitted = {}, {}
    for name, model in candidates.items():
        t0 = time.time()
        model.fit(X[tr_idx], y[tr_idx])
        train_s = time.time() - t0
        if "n_jobs" in model.get_params():
            # train in parallel, but predict single samples on 1 thread: on the device every scan is one URL,
            # and n_jobs=-1 would spin up a thread pool per prediction (very slow on 4-core Arm / under QEMU).
            model.set_params(n_jobs=1)
        p_va = model.predict_proba(X[va_idx])[:, 1]
        thr = threshold_for_recall(y[va_idx], p_va, a.target_recall)
        proba = model.predict_proba(X[te])[:, 1]
        results[name] = {
            "train_seconds": round(train_s, 2),
            "val_roc_auc": detection_metrics(y[va_idx], p_va)["roc_auc"],
            "test_at_0.5": detection_metrics(y[te], proba, 0.5),
            f"test_at_recall_threshold": detection_metrics(y[te], proba, thr),
            "size_bytes": model_size_bytes(model),
            "latency_single": single_sample_latency(model.predict_proba, X[te][:500].tolist()),
        }
        fitted[name] = model
        m = results[name]["test_at_0.5"]
        print(f"{name:14s} F1={m['f1']:.4f} recall={m['recall']:.4f} FPR={m['fpr']:.4f} "
              f"AUC={m['roc_auc']}  size={results[name]['size_bytes'] / 1024:.0f}KB  "
              f"p50={results[name]['latency_single']['p50_ms']}ms")

    # Leakage probe
    trivial_cols = [uf.FEATURE_NAMES.index(c) for c in ("url_len", "path_len", "n_dots_host")]
    probe = HistGradientBoostingClassifier(max_iter=100, random_state=SEED).fit(X[tr_idx][:, trivial_cols], y[tr_idx])
    probe_m = detection_metrics(y[te], probe.predict_proba(X[te][:, trivial_cols])[:, 1])
    print(f"LEAK PROBE [{a.view} view] (3 trivial features only): F1={probe_m['f1']}  AUC={probe_m['roc_auc']}"
          + ("   <-- WARNING: dataset likely has collection artefacts" if (probe_m["roc_auc"] or 0) > 0.95 else ""))
    probe_f = HistGradientBoostingClassifier(max_iter=100, random_state=SEED).fit(X_full[tr_idx][:, trivial_cols], y[tr_idx])
    probe_full = detection_metrics(y[te], probe_f.predict_proba(X_full[te][:, trivial_cols])[:, 1])
    path_rate = {lbl: round(float((df[df.label == lbl]["url"].map(lambda u: uf.extract(u)["path_len"]) > 1).mean()), 4)
                 for lbl in (0, 1)}
    print(f"LEAK PROBE [full view, for reference]: F1={probe_full['f1']}  AUC={probe_full['roc_auc']}  "
          f"share with a path: benign={path_rate[0]} phishing={path_rate[1]}")

    # Model selection on VALIDATION ROC-AUC only (the test split is never used to choose); ties -> smaller model.
    best = max(results, key=lambda k: (results[k]["val_roc_auc"], -results[k]["size_bytes"]))
    print(f"selected (by validation ROC-AUC): {best}")

    # Calibrate on one half of validation, choose the recall-targeted threshold on the other half.
    gss_c = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    va_a, va_b = (va_idx[i] for i in next(gss_c.split(X[va_idx], y[va_idx], groups=df["domain"].iloc[va_idx])))
    cal, op_thr = calibrate_and_threshold(fitted[best], X[va_a], y[va_a], X[va_b], y[va_b], a.target_recall)
    p_te_cal = cal.predict_proba(X[te])[:, 1]
    calibrated = {"method": "isotonic", "calibration_rows": len(va_a), "threshold_rows": len(va_b),
                  "operating_threshold": round(op_thr, 6),
                  "test_at_operating_threshold": detection_metrics(y[te], p_te_cal, op_thr),
                  "brier_test_uncalibrated": round(float(np.mean((fitted[best].predict_proba(X[te])[:, 1] - y[te]) ** 2)), 5),
                  "brier_test_calibrated": round(float(np.mean((p_te_cal - y[te]) ** 2)), 5)}
    m = calibrated["test_at_operating_threshold"]
    print(f"calibrated {best}: threshold={op_thr:.4f} (val target recall {a.target_recall}) -> test recall={m['recall']} "
          f"precision={m['precision']} FPR={m['fpr']}  Brier {calibrated['brier_test_uncalibrated']} -> "
          f"{calibrated['brier_test_calibrated']}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": cal, "name": f"url-{best}-cal", "feature_names": uf.FEATURE_NAMES, "view": a.view,
                 "threshold": float(op_thr), "calibrated": True, "target_recall": a.target_recall,
                 "trained_on": Path(a.csv).name}, a.out, compress=3)

    # End-to-end: the real UrlDetector (rules + calibrated ML + fusion) on the held-out test URLs.
    from bastion.detectors.url_detector import UrlDetector
    det = UrlDetector(model_path=a.out)
    e2e = end_to_end(det.analyze, df["url"].iloc[te].tolist(), y[te])
    w = e2e["warned_suspicious_or_malicious"]
    print(f"END-TO-END detector on test (warned = SUSPICIOUS|MALICIOUS): recall={w['recall']} precision={w['precision']} "
          f"FPR={w['fpr']} F1={w['f1']} AUC(risk)={w['roc_auc']}  verdicts={e2e['verdict_counts']}  "
          f"p50={e2e['latency_ms']['p50']}ms")

    report = {
        "task": "url", "dataset": Path(a.csv).name, "seed": SEED,
        "rows": len(df), "phishing": int(df.label.sum()),
        "split": {"train": len(tr_idx), "val": len(va_idx), "test": len(te), "grouped_by": "registered_domain",
                  "domain_overlap_train_test": len(overlap)},
        "target_recall": a.target_recall, "models": results, "selected": best,
        "selection": "validation ROC-AUC", "calibration": calibrated, "end_to_end_test": e2e,
        "fusion_thresholds": {"suspicious": 35, "malicious": 70, "ml_weight": 0.7},
        "view": a.view,
        "leak_probe_trivial_features": probe_m,
        "leak_probe_full_view_reference": probe_full,
        "share_of_urls_with_path": {"benign": path_rate[0], "phishing": path_rate[1]},
        "machine": {"platform": platform.platform(), "machine": platform.machine(),
                    "python": platform.python_version()},
    }
    out_json = Path(a.results_dir) / f"url_train_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))
    print(f"saved model -> {a.out}\nsaved report -> {out_json}")


if __name__ == "__main__":
    main()
