# Bastion (PS3 edge AI scam guard): engineering report

**Bharat AI-SoC Challenge 2026-27 (Arm), Problem Statement 3: On-Device Threat, Phishing & Scam Detection. Phase 1.**
*Bastion: the offline AI guard on your edge device.* (Unrelated to Microsoft's "Azure Bastion".)

Every number in this report is produced by a script in this repository and stored as JSON or a log in
`benchmarks/results/`; the file is named next to each number. Timings always carry the environment they were measured
in (`x86-native`, `arm64-emulated-qemu`, `arm64-native`, `raspberry-pi-5`).

## 1. Problem and users
SMS and link scams in India follow a few recognisable scripts: KYC/account-block threats, fake parcel fees, lottery
prizes, power-cut threats that end in a remote-access app, and look-alike banking links. Victims are often first-time
smartphone users who need a clear answer ("is this a scam?"), a reason they can understand, and the next safe step.
Cloud checkers require sending private messages to a server. **Bastion runs entirely on an Arm edge device**
(target: Raspberry Pi 5, CPU only): nothing leaves the board, it works without a connection, and it explains its
verdict in plain English using **Llama 3.2 1B running through ExecuTorch on the Arm CPU**.

## 2. Architecture
```
input (URL / SMS, untrusted)
  -> features   lexical URL features (27, offline) · SMS social-engineering indicators · links inside SMS
  -> detectors  rule engine + ML model (calibrated, operating point chosen on validation)
  -> fusion     0-100 risk, SAFE / SUSPICIOUS / MALICIOUS  ← the verdict is fixed here
  -> advice     deterministic remediation steps (block, report 1930 / cybercrime.gov.in / 1909, verify, freeze)
  -> Llama 3.2 1B on ExecuTorch (XNNPACK, Arm CPU) writes a short explanation from the evidence only
       -> guard: checks every streamed token; aborts on a contradiction of the verdict or an invented link/number
       -> deterministic template if the model is missing, fails, or is rejected
```
**Design rule (organisers' reference architecture): the detector decides, Llama only explains.** The LLM receives the
verdict as a fixed fact, the scanned text is sanitised (special tokens and injection phrases removed) and fenced as
`<data>`, and the guard validates the output while it streams, so unsafe text never reaches the screen.

| Component | Runs on | Runtime | Size |
|---|---|---|---|
| URL detector (HistGBDT, canon view) | Arm CPU | scikit-learn 1.9.1 | 447 KB |
| SMS detector (TF-IDF + logistic regression, sigmoid-calibrated) | Arm CPU | scikit-learn 1.9.1 | 527 KB |
| Llama 3.2 1B Instruct explainer | Arm CPU (4 × Cortex-A76 on Pi 5) | ExecuTorch 1.6.0 + XNNPACK, 8da4w | 802 MB `.pte` |
| Web app | Arm CPU | FastAPI + vanilla JS, SQLite | – |

## 3. Data and leakage checks (`docs/DATASETS.md`)
- **SMS:** UCI SMS Spam Collection (CC BY 4.0). Exact and near-duplicates removed: 5,574 → 5,137 messages (620 scams).
- **URL:** PhiUSIIL Phishing URL Dataset (CC BY 4.0); label 0 = phishing (verified from the data). 200,000-row balanced
  sample, split **grouped by registered domain** (train/test domain overlap = 0).
- **Finding 1: PhiUSIIL's benign URLs are a collection artefact.** 100 % of benign URLs are `https://www.` homepages
  (0 % have a path), against 41 % `www.`, 49 % https and 28 % with a path for phishing. A model trained on the host
  scored FPR 0.025 on the PhiUSIIL test split but **flagged 99.5 % of popular legitimate domains** (Tranco) in our
  outside check. The deployed URL model therefore sees the host **without scheme and `www.`** and runs at a threshold
  chosen on validation for ≤ 2 % FPR. The two independent benign sources then agree (1.6 % vs 2.6 %).
  (`benchmarks/results/archive/url_train_20261119_000332_view-host.json`, `url_train_20261119_000943.json`)
- **Finding 2: an operating threshold chosen from 37 positives is unreliable.** The first SMS threshold, chosen on a
  validation half with 37 scams, produced FPR 0.163; taking it from 5-fold out-of-fold predictions over 422 scams gives
  FPR 0.0066 at the same recall. (`archive/sms_train_20261118_195416.json`, `sms_train_20261119_001008.json`)
- Model selection uses validation ROC-AUC only; calibration is kept only if it improves validation Brier score.

## 4. Detection results: the deployed detector, end-to-end
Evaluated as the app runs it (rules + ML + fusion, one input at a time) on held-out test data. "Warned" =
SUSPICIOUS or MALICIOUS.

| Channel | n (test) | Recall | Precision | FPR | F1 | ROC-AUC (risk) | Extra | Source |
|---|---|---|---|---|---|---|---|---|
| SMS | 1,028 | **0.944** | 0.951 | **0.0066** | 0.947 | 0.993 | – | `sms_train_20261119_001008.json` |
| URL | 45,857 | **0.737** | 0.983 | **0.016** | 0.842 | 0.923 | Tranco popular domains flagged: **2.6 %** (10,000) | `url_train_20261119_000943.json` |

**Against the recall ≥ 0.95 target:** SMS reaches 0.944 on the test split (the threshold targets 0.95 on out-of-fold
predictions; 117 of 124 test scams caught). URL recall is 0.737: with lexical, offline-only features, PhiUSIIL cannot
support 0.95 recall without flagging most legitimate hosts (77-85 % at a 0.95-recall threshold, see `docs/DATASETS.md`).
We chose a usable false-alarm rate and report the recall honestly.

## 5. Llama 3.2 1B on ExecuTorch
- **Versions (verified):** ExecuTorch 1.6.0, torch 2.15.0+cpu (PyTorch CPU index: PyPI's torch 2.15 pulls CUDA
  libraries even on aarch64), torchao 0.18.0. Same ExecuTorch version for export and runtime.
- **Export:** `python -m executorch.extension.llm.export.export_llm --config twin/export/llama3_2_1b_8da4w.yaml`:
  `llama3_2`, KV cache + SDPA, `8da4w` (int8 dynamic activations, int4 weights, group 128), 4-bit embeddings,
  XNNPACK, max sequence 768. Every key was checked against `LlmConfig` at v1.6.0. (`docs/EXECUTORCH_NOTES.md`)
- **Runtime:** the PyPI aarch64 wheel does not include ExecuTorch's Python LLM runner, so Bastion drives the exported
  program through the generic `executorch.runtime` API with its own greedy decode loop: the `.pte` is loaded **once**
  and kept resident, the prompt is prefilled in one call, tokens stream to the UI, and the KV cache restarts at
  position 0 for every explanation. The LLM kernels (`custom_sdpa`, `update_cache`, quantized embeddings) are
  registered by importing ExecuTorch's kernel libraries.
- **Weights:** Meta's gated repository had not approved our access request by the deadline, so the export used the
  same official Llama 3.2 1B Instruct weights redistributed under the Llama 3.2 Community License
  (`unsloth/Llama-3.2-1B-Instruct`), converted to Meta format in `twin/export/export_local.py`. The resulting `.pte`
  is 802 MB (sha256 `15fc1ef7…`); greedy "The capital of France is" → " Paris. Paris is the largest city in".
- **Finding:** the official 2024 pre-exported community `.pte` loads in the 1.6.0 runtime but produces gibberish
  (custom LLM kernels are not covered by ExecuTorch's compatibility guarantee). Export and runtime must share a version.

### 5.1 Measured (`benchmarks/results/edge_x86-native_20261123_182544.json`, `x86-native`)
| Metric | Value |
|---|---|
| model load (once, then resident) | 3.05 s |
| explanation wall time, p50 of 5 | 4.29 s |
| prefill (≈ 380-405 prompt tokens), mean | 2.36 s |
| decode throughput, mean | 35.25 tokens/s |
| guard fallbacks | 1 of 5: the prompt-injection SMS. The model drifted towards "safe", the guard rejected it mid-stream, and the template was shown; the verdict stayed MALICIOUS |

The verdict is returned in milliseconds; the explanation streams afterwards, so the user is never kept waiting for a
decision. **Llama has not yet been run inside the arm64 twin** (Docker Desktop's engine failed on the deadline day) or
on a Raspberry Pi 5; these x86 numbers must not be read as Arm numbers.

## 6. Hardware mapping and the Raspberry Pi 5 digital twin (`docs/TWIN.md`)
No Raspberry Pi 5 was available during Phase 1. We built an **Arm64 execution twin**: a `linux/arm64` Debian Bookworm
image with Debian's Python 3.11.2 (as on Raspberry Pi OS), run with the Pi 5's limits (`--cpus=4 --memory=8g`,
4 GB selectable) and, for the offline proof, `--network none`.
- **The full test suite passes inside the twin** (53 tests, `benchmarks/results/twin_pytest_2026-11-19.log`).
- **Offline proof:** `uname -m` = `aarch64`, empty route table, real verdicts (`offline_proof_2026-11-19.log`).
- **Environment tagging** (`src/bastion/env.py`): QEMU is detected from `/proc/cpuinfo` ("CPU implementer 0x00", a
  synthetic CPU), a Pi 5 from `/proc/device-tree/model`. Every benchmark and UI metric carries the tag.

| Environment | URL detect p50 | SMS detect p50 | Source |
|---|---|---|---|
| `x86-native` | 1.52 ms | 2.39 ms | `edge_x86-native_20261119_002938.json` |
| `arm64-emulated-qemu` (twin) | 26.76 ms | 68.24 ms | `edge_arm64-emulated-qemu_20261118_185542.json` |
| `raspberry-pi-5` | not measured (no hardware) | | |

Emulated timings are 18-29× slower than native x86 here: **they prove the software runs on aarch64, not Pi 5 speed.**

The twin also surfaced deployment bugs before any hardware existed: CUDA-carrying torch wheels on aarch64; a missing
Python LLM runner in the ExecuTorch wheel; `libatomic.so.1` needed by the tokenizer; thread pools sized to the host's
32 cores (fixed with `n_jobs=1`, `OMP_NUM_THREADS=4` and single-thread scoring: HistGBDT p50 6.2 → 1.4 ms on x86);
model paths that broke on non-editable installs.

## 7. Product
`bastion serve` starts a local web app (127.0.0.1 by default; `--lan` for a phone on the same network, with a warning).
The scan page returns the detector verdict instantly and streams the guarded explanation over SSE. Also: local SQLite
history with a hash-only privacy mode and CSV export, a live 3D Device Twin page (CPU, memory, pipeline stage, model
state, network status read from the route table), a Benchmarks page grouped by environment tag, and a Methodology page.
All assets, including three.js and fonts, are bundled locally. `docker compose up` serves the UI from the offline twin
(internal network with no gateway plus a loopback-only proxy).

## 8. Safety, privacy and security
- Verdict lock: the LLM cannot change label or risk; any contradiction or invented URL/phone number is rejected mid-stream.
- Prompt injection: special tokens and instruction phrases are stripped from scanned text, which is fenced as data;
  tested through the API (the verdict holds and the template is shown).
- Inputs limited (URL 2,048 / SMS 2,000 characters) and rate-limited; untrusted text rendered with `textContent` only;
  no `shell=True`; the app loads models only from its own `models/` folder; no telemetry or external calls at runtime
  (tested).

## 9. Limitations
1. **No Raspberry Pi 5 measurements yet.** The twin proves aarch64 correctness and offline operation only; Llama was
   measured on x86 only (the twin run is pending).
2. **Explanations from a 1B model are short and sometimes generic** ("this is a scam with a high risk"); the
   detector's "Why" list carries the specific evidence, and the guard blocks anything that contradicts the verdict.
3. **URL recall 0.737** at a 2 % false-alarm budget (lexical offline features on PhiUSIIL).
4. **SMS data is dated and mostly UK English**; Indian UPI/KYC/courier scams are under-represented. The handwritten
   demo samples are not a test set.
5. The Tranco check treats popular domains as legitimate; a few host abuse, so it is an approximate false-positive rate.

## 10. Phase 2 plan
Measure on a real Raspberry Pi 5 (same image / `scripts/setup.sh`) and report under the `raspberry-pi-5` tag; an
on-device phishing blocklist snapshot to lift URL recall; an Indian-context SMS test set reported separately;
Llama latency tuning on real Arm cores (threads, sequence length, prompt size); optional Hailo NPU study.

## 11. Reproducing
```bash
python scripts/download_data.py sms url tranco
python training/train_sms.py --uci data/raw/SMSSpamCollection
python training/train_url.py --csv data/raw/PhiUSIIL_Phishing_URL_Dataset.csv --url-col URL --label-col label \
       --phish-value 0 --fp-check data/raw/top-1m.csv
bastion bench                                   # tagged latency JSON
python twin/twinctl.py build && python twin/twinctl.py test && python twin/twinctl.py offline-demo
pytest -q
```
