"""Bastion Pi 5 digital twin controller (works on Windows, Linux, macOS; no `make` needed).

  python twin/twinctl.py setup           check Docker/buildx/arm64 support (installs binfmt on Linux with --install)
  python twin/twinctl.py build           build the arm64 image (bastion-pi5-twin:dev)
  python twin/twinctl.py up [--ram 4]    run `bastion status` inside the twin with Pi 5 limits
  python twin/twinctl.py shell           interactive shell in the twin
  python twin/twinctl.py test            pytest inside the twin
  python twin/twinctl.py bench           edge benchmark inside the twin (JSON -> benchmarks/results/)
  python twin/twinctl.py offline-demo    full demo with --network none, log -> benchmarks/results/offline_proof_<date>.log
  python twin/twinctl.py run -- <cmd>    any command inside the twin

Every run uses the Pi 5 profile: --platform linux/arm64 --cpus=4 --memory=8g (or 4g),
and passes BASTION_HOST_ARCH so the environment tag can never claim native/Pi timing under emulation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "bastion-pi5-twin:dev"
PROFILE = {"cpus": 4, "ram_gb": 8, "threads": 4}   # mirrors twin/pi5_profile.yaml


def host_arch() -> str:
    m = platform.machine().lower()
    return {"amd64": "x86_64", "arm64": "aarch64"}.get(m, m)


def sh(cmd: list[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check, text=True, capture_output=capture)


def docker_run_args(ram_gb: int, offline: bool, interactive: bool = False) -> list[str]:
    args = ["docker", "run", "--rm", "--platform", "linux/arm64",
            f"--cpus={PROFILE['cpus']}", f"--memory={ram_gb}g",
            "-e", f"BASTION_HOST_ARCH={host_arch()}",
            "-e", f"BASTION_TWIN_PROFILE=pi5-{ram_gb}gb",
            "-e", f"BASTION_LLAMA_THREADS={PROFILE['threads']}",
            "-v", f"{ROOT / 'models'}:/app/models:ro",
            "-v", f"{ROOT / 'benchmarks' / 'results'}:/app/benchmarks/results"]
    if offline:
        args += ["--network", "none"]
    if interactive:
        args += ["-it"]
    return args + [IMAGE]


def cmd_setup(a) -> int:
    if not shutil.which("docker"):
        print("Docker not found. Install Docker Desktop (Windows/macOS) or docker-ce + buildx (Linux).")
        return 1
    sh(["docker", "version", "--format", "{{.Server.Version}} {{.Server.Os}}/{{.Server.Arch}}"])
    sh(["docker", "buildx", "version"])
    print(f"host arch: {host_arch()}")
    if host_arch() == "aarch64":
        print("Arm host: the twin runs NATIVELY on real Arm cores (environment tag: arm64-native).")
        return 0
    ls = sh(["docker", "buildx", "ls"], capture=True).stdout
    if "linux/arm64" in ls:
        print("arm64 emulation (QEMU binfmt) is available. The twin will run emulated "
              "(environment tag: arm64-emulated-qemu; timings are NOT Pi 5 numbers).")
        return 0
    if platform.system() == "Linux" and a.install:
        sh(["docker", "run", "--privileged", "--rm", "tonistiigi/binfmt", "--install", "arm64"])
        return 0
    print("linux/arm64 not available. On Linux run: python twin/twinctl.py setup --install "
          "(docker run --privileged --rm tonistiigi/binfmt --install arm64). "
          "On Docker Desktop it is built in: update Docker Desktop.")
    return 1


def cmd_build(a) -> int:
    cmd = ["docker", "buildx", "build", "--platform", "linux/arm64", "-f", str(ROOT / "twin" / "Dockerfile"),
           "-t", IMAGE, "--load"]
    if a.progress:
        cmd += ["--progress", a.progress]
    return sh(cmd + [str(ROOT)], check=False).returncode


def _run(a, inner: list[str], offline: bool = False, interactive: bool = False) -> int:
    (ROOT / "benchmarks" / "results").mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)
    return sh(docker_run_args(a.ram, offline or a.offline, interactive) + inner, check=False).returncode


def cmd_offline_demo(a) -> int:
    results = ROOT / "benchmarks" / "results"
    results.mkdir(parents=True, exist_ok=True)
    log = results / f"offline_proof_{dt.date.today().isoformat()}.log"
    inner = ["sh", "-c",
             "echo '--- route table (empty = no network) ---'; cat /proc/net/route; "
             "echo '--- bastion status ---'; bastion status; "
             "echo '--- bastion demo ---'; bastion demo"]
    proc = subprocess.run(docker_run_args(a.ram, offline=True) + inner, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    header = (f"# Bastion offline proof, {dt.datetime.now().isoformat(timespec='seconds')}\n"
              f"# host arch: {host_arch()}  image: {IMAGE}  docker: --network none --cpus={PROFILE['cpus']} "
              f"--memory={a.ram}g\n")
    log.write_text(header + proc.stdout, encoding="utf-8")
    print(proc.stdout)
    print(f"log -> {log}")
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="twinctl", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ram", type=int, choices=(4, 8), default=PROFILE["ram_gb"], help="Pi 5 RAM variant (GB)")
    ap.add_argument("--offline", action="store_true", help="add --network none")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("setup"); p.add_argument("--install", action="store_true")
    p = sub.add_parser("build"); p.add_argument("--progress", default=None)
    sub.add_parser("up"); sub.add_parser("shell"); sub.add_parser("test"); sub.add_parser("bench")
    sub.add_parser("offline-demo")
    p = sub.add_parser("run"); p.add_argument("inner", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)

    if a.cmd == "setup":
        return cmd_setup(a)
    if a.cmd == "build":
        return cmd_build(a)
    if a.cmd == "up":
        return _run(a, ["bastion", "status"])
    if a.cmd == "shell":
        return _run(a, ["bash"], interactive=True)
    if a.cmd == "test":
        return _run(a, ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"])
    if a.cmd == "bench":
        return _run(a, ["python", "benchmarks/bench_edge.py"])
    if a.cmd == "offline-demo":
        return cmd_offline_demo(a)
    if a.cmd == "run":
        inner = a.inner[1:] if a.inner[:1] == ["--"] else a.inner
        return _run(a, inner or ["bastion", "status"])
    return 2


if __name__ == "__main__":
    sys.exit(main())
