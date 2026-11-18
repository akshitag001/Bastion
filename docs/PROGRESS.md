# Progress

## Session 1 (18 Nov 2026): Bastion B0 + B1 Layer A/B + B3 setup + B4 training

### Done
- [x] **B0** rename EdgeGuard -> Bastion (package `bastion`, CLI `bastion`, env vars `BASTION_*`, docs). Repo flattened into `Bastion/`,
      git initialised, LF line endings enforced (`.gitattributes`) so shell scripts work in Linux containers.
- [x] **B1 Layer B** `src/bastion/env.py`: runtime environment tag (`raspberry-pi-5` / `arm64-native` / `arm64-emulated-qemu` /
      `x86-native`). QEMU detected from `/proc/cpuinfo` "CPU implementer 0x00" (recorded in the twin) + `BASTION_HOST_ARCH` hint.
      Offline check via empty route table / unroutable UDP connect (no packets sent). Tag is in `bastion status` and every bench JSON.
      Honesty rule added to the project rules.
- [x] **B1 Layer A** `twin/Dockerfile` (Debian Bookworm arm64 + Debian's Python 3.11, same as Pi OS), `twin/pi5_profile.yaml`,
      `twin/twinctl.py` (setup/build/up/shell/test/bench/offline-demo/export/run; works on Windows without `make`).
- [x] `src/bastion/paths.py`: `BASTION_ROOT` so models/data resolve correctly when installed non-editable (twin bug found and fixed).
- [x] **B3** ExecuTorch pinned to **1.6.0** (+ torch 2.15.0+cpu) after checking PyPI and the v1.6.0 sources; export entry point,
      config keys, runner flags and Python runner API verified (docs/EXECUTORCH_NOTES.md). Versioned export config
      `twin/export/llama3_2_1b_8da4w.yaml`; x86 export image built and `export_llm --help` verified.
- [x] Persistent Llama: `LlamaExplainer` now has a `python` backend (ExecuTorch `TextLLMRunner`, loaded once, token streaming,
      KV reset per scan) with `llama_main` subprocess fallback. Tested with a fake runner.
- [x] **B4** datasets downloaded, licences (CC BY 4.0) and PhiUSIIL label convention (0 = phishing) verified empirically.
      Found PhiUSIIL artefact: 0 % of benign URLs have a path. URL ML model now uses a host-only view; leak probe AUC 0.899 -> 0.782.
      Trained: SMS LinearSVM (F1 0.951, AUC 0.992), URL HistGBDT host view (F1 0.963, AUC 0.991). JSONs in benchmarks/results/.
- [x] Dependencies pinned exactly (requirements.txt) so pickled models load identically on x86 and arm64.
- [x] Tests: 26 -> 38, all passing on x86.

### Next
1. Twin image build -> `twinctl test` (pytest on aarch64), `twinctl offline-demo` (log in benchmarks/results/).
2. Verify the PyPI aarch64 executorch wheel ships the Python LLM runner (else build `llama_main` / cross-compile).
3. **Llama export** as soon as HF_TOKEN is available (`python twin/twinctl.py export`), then run Llama in the twin.
4. B4: recall-targeted thresholds into fusion + calibrated confidence (CalibratedClassifierCV); bench on Arm.
5. B2 FastAPI backend + scan UI (SSE streaming of the explanation).

### Blockers (need the user)
- **Hugging Face:** accept the Llama 3.2 licence for `meta-llama/Llama-3.2-1B-Instruct` and provide a read token (`$env:HF_TOKEN`).
- **Real Arm hardware** for honest performance numbers (Pi 5, or Graviton / Ampere VM).
- GitHub repo to push to (`gh` CLI not installed).

### Open questions (ask organisers: tech-queries@armbharatchallenge.com, subject "PS3")
- Are all three channels (network, URL, SMS) mandatory for Phase 1?
- Which deliverables are mandatory on 23 Nov (is the app binary/APK required)? Is a working Llama/ExecuTorch part required in Phase 1?
- Report / demo length or format limits?
