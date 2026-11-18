# The Raspberry Pi 5 digital twin

**What it is:** the real Bastion software stack (Debian Bookworm arm64, Python 3.11, scikit-learn, ExecuTorch 1.6.0 with
XNNPACK, Llama 3.2 1B `.pte`) running as aarch64 code, with the Pi 5's resource limits (4 CPUs, 8 GB or 4 GB RAM) and
optionally no network.

**What it is not:** a performance model of the Raspberry Pi 5. On an x86 laptop every aarch64 instruction is translated
by QEMU, which is many times slower than real Arm silicon and has a different cache and memory system.

| Layer | Purpose | Where |
|---|---|---|
| A: execution twin | prove the Arm64 software stack runs correctly, offline | `twin/Dockerfile`, `twin/twinctl.py`, `docker-compose.yml` |
| B: environment tags | stop emulated / non-Pi timings being reported as Pi 5 numbers | `src/bastion/env.py` |
| C: 3D dashboard | demo aid: live CPU / RAM / stage / network / model state | web UI, Device Twin page |

## Commands
```bash
python twin/twinctl.py setup          # checks Docker, buildx and arm64 emulation (Linux: --install sets up binfmt)
python twin/twinctl.py build          # builds bastion-pi5-twin:dev (linux/arm64)
python twin/twinctl.py up             # startup check + `bastion status`
python twin/twinctl.py test           # pytest on aarch64
python twin/twinctl.py offline-demo   # --network none; log -> benchmarks/results/offline_proof_<date>.log
python twin/twinctl.py bench          # edge benchmark JSON (tagged arm64-emulated-qemu on x86)
python twin/twinctl.py serve          # web UI from the twin at http://127.0.0.1:8080 (network on)
docker compose up                     # web UI from the twin with the twin OFFLINE (internal network + proxy)
python twin/twinctl.py export         # x86 container: download Llama 3.2 1B + export .pte (needs HF_TOKEN)
```
Every run passes `--platform linux/arm64 --cpus=4 --memory=8g` (`--ram 4` for the 4 GB board),
`OMP_NUM_THREADS=4` and `BASTION_HOST_ARCH=<host arch>`.

## Environment detection (verified)
Inside the twin on an x86 host, `/proc/cpuinfo` reports `CPU implementer : 0x00` (QEMU's synthetic CPU; 0x00 is reserved
for software use). Real Arm silicon reports a vendor ID: 0x41 (Arm Ltd: the Pi 5's Cortex-A76 is part 0xd0b, Graviton2
Neoverse-N1 is 0xd0c), 0x61 (Apple), 0xc0 (Ampere). A real Pi 5 is identified from `/proc/device-tree/model`.

Recorded startup check (x86 host, 18 Nov 2026):
```
 uname -m      : aarch64
 python        : 3.11.2
 cpu limit     : 4.0
 memory limit  : 8.0 GiB
 environment   : arm64-emulated-qemu  (cpuinfo CPU implementer 0x00 (QEMU synthetic CPU))
```

## Offline proof
- `--network none` leaves only the loopback interface: `/proc/net/route` has no default route, and a UDP `connect()` to a
  TEST-NET address fails with "network unreachable" (no packet is ever sent). `bastion status` and the UI show OFFLINE.
- `docker compose up` puts Bastion on an `internal: true` network (no gateway). Only the nginx proxy can reach it, and the
  proxy publishes the UI on the host's loopback interface.

## Things the twin surfaced (fixed)
| Finding | Fix |
|---|---|
| Model paths were computed from the source tree; a non-editable install looked inside site-packages | `BASTION_ROOT` + `bastion/paths.py` |
| PyPI's aarch64 torch 2.15 wheel pulls >1 GB of CUDA libraries | install `torch==2.15.0+cpu` from the PyTorch CPU index |
| ExecuTorch's PyPI wheel has no Python LLM runner (`TextLLMRunner`) | `runtime` backend: decode loop over `executorch.runtime` |
| `pytorch_tokenizers` needs `libatomic.so.1` (absent in slim Debian, present in Pi OS) | `libatomic1` in the image |
| Thread pools sized to the host's 32 cores (random forest `n_jobs=-1`, OpenMP) | `n_jobs=1` after training, `OMP_NUM_THREADS=4`, OpenMP limited to 1 thread for single-sample scoring |

## Running on a real Pi 5 / Arm machine
The same image runs natively on an arm64 host (`python twin/twinctl.py build` then `up`); the tag becomes `arm64-native`,
or `raspberry-pi-5` on a Pi. Without Docker on a Pi: `bash scripts/setup.sh` (see README).
