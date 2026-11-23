# Progress

## Session 3 (23 Nov 2026, submission day)
- [x] **UI redesign**: light editorial dashboard (Space Grotesk / Inter / JetBrains Mono bundled locally), hero with live
      stats from the result JSONs, 3D Pi 5 board in the hero, presets, recent scans; explanation badge reflects reality.
- [x] **Llama 3.2 1B on ExecuTorch 1.6.0 working**: own export (`twin/export/export_local.py`, weights from the
      Llama-3.2-licensed unsloth redistribution because Meta's gated repo approval was pending); runtime backend fixed
      (LLM kernel registration, Python tiktoken API, incremental decoding). x86-native: load 3.05 s, explanation p50 4.29 s,
      35.25 tok/s, 1/5 guard fallback (prompt injection). The 2024 community `.pte` is incompatible with the 1.6.0 runtime.
- [x] `docs/REPORT.md` written; README/notes updated; screenshots show a real streamed Llama explanation.
- [x] Repo prepared for GitHub: local agent notes and the master prompt excluded and removed from history.
- [ ] Llama inside the arm64 twin (Docker Desktop engine down today) and on a real Pi 5.
- [ ] Demo video.

## Sessions 1-2 (18-19 Nov 2026)

### Done
- [x] **B0** rename EdgeGuard -> Bastion; repo flattened into `Bastion/`, git initialised, LF line endings enforced.
- [x] **B1 Layer A** arm64 twin: `twin/Dockerfile` (Debian Bookworm arm64 + Debian Python 3.11.2, like Pi OS), `twin/pi5_profile.yaml`,
      `twin/twinctl.py` (setup/build/up/shell/test/bench/offline-demo/serve/export/run), `docker-compose.yml` (offline twin
      on an `internal` network + loopback-only nginx proxy so the UI works while Bastion has no route out).
      **Full pytest suite passes inside the aarch64 twin** (log: `benchmarks/results/twin_pytest_<date>.log`).
- [x] **B1 Layer B** `src/bastion/env.py` environment tags; QEMU detected from cpuinfo implementer 0x00 (recorded in the twin).
- [x] **B1 Layer C** Device Twin page: three.js board from primitives (vendored three.js 0.170.0, MIT), live CPU/RAM/stage/
      network/model state, pulse along the pipeline, verdict LED; renders only while the page is visible.
- [x] **B2** FastAPI server: instant detector verdict + guarded SSE explanation stream (partial-output guard aborts generation
      before unsafe text reaches the UI), SQLite history with hash-only privacy mode + CSV export + clear, input limits,
      per-client rate limit, localhost by default (`--lan` warns). Web UI: Scan / History / Device Twin / Benchmarks / About;
      untrusted text rendered with textContent only; no external URLs (tested).
- [x] **B3** ExecuTorch **1.6.0** + torch 2.15.0+cpu pinned and verified; export config `twin/export/llama3_2_1b_8da4w.yaml`
      (keys verified against v1.6.0 `LlmConfig`); x86 export image built. Persistent Llama backends: `python` (TextLLMRunner,
      needs a source build) and **`runtime`** (decode loop over `executorch.runtime` + `pytorch_tokenizers`, which DO ship in the
      PyPI aarch64 wheel; XNNPACK backend registered; thread pool sized to 4) with `llama_main` fallback. Tested with fakes.
- [x] **B4** detectors on real data, with two dataset problems found and handled (docs/DATASETS.md):
      - PhiUSIIL benign URLs are 100 % `https://www.` homepages. The `host` model looked perfect on PhiUSIIL (FPR 0.025) but flagged
        **99.5 %** of popular Tranco domains. Deployed: `canon` view at a 2 % validation-FPR threshold -> end-to-end recall 0.737,
        precision 0.983, FPR 0.016; Tranco popular domains flagged 2.6 %.
      - SMS threshold from a 37-scam validation half gave FPR 0.163; now from 422 out-of-fold scams -> end-to-end recall 0.944,
        precision 0.951, FPR 0.0066.
      - Validation-only model selection, calibration kept only when it improves validation Brier, end-to-end evaluation of the
        real detector (rules + ML + fusion), single-thread scoring (HistGBDT p50 6.2 -> 1.4 ms on x86; SMS 13.2 -> 2.4 ms).
- [x] Tests: 26 -> 53 (x86) passing.

### Next
1. **Llama export + run in the twin** (blocked on HF token): `python twin/twinctl.py export`, then a real explanation through the
   `runtime` backend; verify tokenizer handling of the Llama 3 chat special tokens; measure load / prefill / tok/s per env tag.
2. Benchmarks: `bastion bench --llm` (persistent `runtime` vs per-call `subprocess` if `llama_main` gets built).
3. B5: CI (GitHub Actions: x86 tests + arm64 QEMU test job), prompt-injection tests through the real Llama path.
4. B6: README (quick start, screenshots), `docs/REPORT.md`, `docs/DEMO_SCRIPT.md`, wheel, record demo video. Submit 22 Nov.

### Blockers (need the user)
- **Hugging Face:** accept the Llama 3.2 licence for `meta-llama/Llama-3.2-1B-Instruct`; set `HF_TOKEN`.
- **Real Arm hardware** for honest performance numbers (Pi 5, or a Graviton / Ampere VM).
- GitHub repo to push to.

### Open questions (organisers: tech-queries@armbharatchallenge.com, subject "PS3")
- Are all three channels (network, URL, SMS) mandatory for Phase 1?
- Which deliverables are mandatory on 23 Nov? Is a working Llama/ExecuTorch part required in Phase 1?
- Report / demo length or format limits?
