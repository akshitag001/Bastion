# Progress

## v0.1 (16 Nov 2026): done
- [x] Repo scaffold, packaging (`pip install -e .`, `bastion` CLI), setup script
- [x] URL lexical feature extractor (27 features, offline, no DNS/WHOIS)
- [x] SMS cleaning + social-engineering indicators + URL extraction (embedded URLs scored by URL detector)
- [x] Rule engines for URL and SMS (work with no trained model)
- [x] ML training scripts: URL (logreg / random forest / hist-GBDT), SMS (TF-IDF + logreg / linear SVM / ComplementNB)
      with domain-grouped split, dedup, leak probe, recall-targeted threshold, size + latency measurement
- [x] Risk fusion (0-100, SAFE / SUSPICIOUS / MALICIOUS)
- [x] Llama-on-ExecuTorch explainer wrapper (subprocess to `llama_main`), Llama 3 chat prompt,
      prompt-injection sanitising, verdict-lock guard, template fallback, deterministic remediation advice
- [x] Edge benchmark script (latency p50/p95/p99, RAM, CPU, throughput, CPU temp on Pi)
- [x] 26 tests (features, detectors, guard, injection, LLM failure fallback, no-network, training smoke)
- [ ] NOT done: no model trained on real data yet; Llama export/runner scripts are untested templates;
      nothing has run on a Raspberry Pi yet. **No metrics exist yet.**

## Next (in priority order)
1. **M1: Llama 3.2 1B -> ExecuTorch .pte, running on the Pi** (`scripts/export_llama.sh`, `scripts/build_runner_pi.sh`).
   Verify commands vs current docs. Then `bastion status` = READY, and `bench_edge.py --llm`.
2. Download datasets, train SMS + URL models, review the leak probe, commit JSON reports.
3. Tune thresholds on validation data (recall-first), calibrate confidence.
4. Benchmarks on Pi vs laptop; fill docs/HARDWARE_MAPPING.md.
5. Optional upgrade: keep Llama loaded in a persistent process (v1 restarts the runner per call, so
   per-call latency includes model load). Options: ExecuTorch Python LLM runner bindings, or a small
   C++ server around the runner. Measure before/after.
6. Stretch: network anomaly detector (NSL-KDD / UNSW-NB15).
7. Demo video script, engineering report (docs/REPORT_OUTLINE.md).

## Open questions (ask organisers: tech-queries@armbharatchallenge.com, subject "PS3")
- Are all three channels (network, URL, SMS) mandatory for Phase 1?
- Which deliverables are mandatory on 23 Nov (is the app binary/APK required)?
- Report / demo length or format limits?
