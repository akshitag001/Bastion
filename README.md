# Bastion (PS3 edge AI scam guard)

*Bastion: the offline AI guard on your edge device.*

On-device phishing & scam detection for the Bharat AI-SoC Challenge 2026-27 (Arm), PS3. (Not related to the Microsoft product "Azure Bastion".)

Offline AI security assistant for Arm edge devices (target: **Raspberry Pi 5, CPU-only**).
It scans **URLs** and **SMS messages**, gives a verdict + risk score, explains *why* in plain
language using a **Llama model running on-device via ExecuTorch**, and recommends what to do.
No cloud. No network calls at inference time.

```
Input (URL / SMS)
  -> features (lexical URL features, SMS social-engineering indicators)    src/bastion/features/
  -> detectors (rules + ML model)                                          src/bastion/detectors/
  -> risk fusion (0-100 score, SAFE / SUSPICIOUS / MALICIOUS)              src/bastion/fusion/
  -> Llama 3.2 1B on ExecuTorch writes a short explanation                 src/bastion/explain/
       -> guard: rejects output that contradicts the verdict / invents links
       -> fallback template if Llama is missing, crashes or is rejected
  -> remediation advice (deterministic, always shown)                      src/bastion/advice/
```

**Design rule:** the detector decides; Llama only explains. The verdict can never be changed by the LLM
(enforced in `explain/guard.py` and `pipeline.py`, covered by tests).

## Quick start (laptop or Raspberry Pi 5, 64-bit OS)

```bash
bash scripts/setup.sh
source .venv/bin/activate
bastion status            # what is loaded: rules / ML models / Llama runtime
bastion demo              # runs handwritten demo URLs + SMS
bastion scan-url "http://paypal.com.account-verify-secure.xyz/signin"
bastion scan-sms "Your SBI account will be blocked today. Update KYC at http://sbi-kyc.xyz"
bastion --json scan-sms "..."   # machine-readable
pytest -q                   # 26 tests
```

The app works immediately in **rules-only** mode. Train ML models and add Llama to reach the full system.

## Train the detectors (real public data)

```bash
python scripts/download_data.py sms url        # needs internet once; verify licences -> docs/DATASETS.md
python training/train_sms.py --uci data/raw/SMSSpamCollection
python training/train_url.py --csv data/raw/<PhiUSIIL csv> --url-col URL --label-col label --phish-value <value meaning phishing>
```
Each trainer compares 3 models, prints precision / recall / F1 / FPR / FNR / ROC-AUC, model size and
single-sample latency, saves the best model to `models/`, and writes a full JSON report to `benchmarks/results/`.
The URL trainer splits **by domain** and runs a **leak probe** (see docs/DATASETS.md).

## Add Llama via ExecuTorch (mandatory for the competition)

1. Accept the Llama 3.2 licence and download `Llama-3.2-1B-Instruct` (original Meta checkpoint).
2. `bash scripts/export_llama.sh` on your dev machine -> `models/llama/llama3_2_1b_8da4w.pte` + `tokenizer.model`.
3. `bash scripts/build_runner_pi.sh` on the Pi -> `models/llama/llama_main`.
4. Copy the `.pte` and tokenizer to the Pi's `models/llama/`, then `bastion status` should say **READY**.

Paths can be overridden with `BASTION_LLAMA_RUNNER`, `BASTION_LLAMA_PTE`, `BASTION_LLAMA_TOKENIZER`,
`BASTION_LLAMA_THREADS`. **Both scripts are untested templates. Verify them against the current ExecuTorch docs**
(see docs/EXECUTORCH_NOTES.md).

## Benchmark on the device

```bash
python benchmarks/bench_edge.py                  # detector latency p50/p95/p99, RAM, CPU, throughput, model size
python benchmarks/bench_edge.py --llm --llm-runs 5
```
Run the same commands on laptop and Pi; results are saved as JSON per machine.

## Status
v0.1 (scaffold + working rules/ML pipeline). See `docs/PROGRESS.md` for what is done, what is next, and open questions.
