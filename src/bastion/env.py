"""Runtime environment detection (the "honesty layer").

Every benchmark JSON and every UI metric carries one of these tags:

  raspberry-pi-5        real Pi 5 hardware (/proc/device-tree/model)
  arm64-native          real Arm cores (Apple Silicon, Graviton, other Arm boards):
                        real Arm timing, but NOT Raspberry Pi 5 timing
  arm64-emulated-qemu   aarch64 userland emulated on an x86 host (the digital twin on a laptop):
                        functionally correct, timings are NOT representative of any Arm device
  x86-native            x86-64 laptop/desktop, no emulation
  unknown-<machine>     anything else

Detection uses only local files; it never touches the network.

How emulation is detected (verified empirically in the twin, see docs/TWIN.md):
QEMU user-mode synthesises /proc/cpuinfo with "CPU implementer : 0x00" (an ID reserved
for software use). Real Arm silicon reports a vendor ID such as 0x41 (Arm Ltd, e.g. the
Pi 5's Cortex-A76 = part 0xd0b) or 0x61 (Apple). As a backstop, the twin launcher passes
BASTION_HOST_ARCH (the host's architecture) into the container.
"""

from __future__ import annotations

import os
import platform
import re
import socket
from pathlib import Path

TAGS = ("raspberry-pi-5", "arm64-native", "arm64-emulated-qemu", "x86-native")

TIMING_NOTES = {
    "raspberry-pi-5": "Measured on a real Raspberry Pi 5.",
    "arm64-native": "Real Arm cores, but not a Raspberry Pi 5: do not report as Pi 5 performance.",
    "arm64-emulated-qemu": "aarch64 emulated by QEMU on x86: functionally correct, timings NOT representative "
                           "of a Raspberry Pi 5 (usually many times slower).",
    "x86-native": "x86 development machine: not Arm, not representative of a Raspberry Pi 5.",
}

# Known Arm CPU implementer IDs (from the Arm architecture reference / Linux arch/arm64 cputype.h).
_ARM_IMPLEMENTERS = {
    "0x41": "Arm", "0x42": "Broadcom", "0x43": "Cavium", "0x46": "Fujitsu", "0x48": "HiSilicon",
    "0x4e": "NVIDIA", "0x50": "APM", "0x51": "Qualcomm", "0x61": "Apple", "0x6d": "Microsoft",
    "0xc0": "Ampere",
}
_X86_HOSTS = {"x86_64", "amd64", "i686", "i386", "x86"}


def _read(path: str | Path) -> str | None:
    try:
        return Path(path).read_text(errors="replace")
    except OSError:
        return None


def _cpuinfo_implementer(cpuinfo: str | None) -> str | None:
    if not cpuinfo:
        return None
    m = re.search(r"^CPU implementer\s*:\s*(0x[0-9a-fA-F]+)", cpuinfo, re.M)
    return m.group(1).lower() if m else None


def _cpuinfo_part(cpuinfo: str | None) -> str | None:
    if not cpuinfo:
        return None
    m = re.search(r"^CPU part\s*:\s*(0x[0-9a-fA-F]+)", cpuinfo, re.M)
    return m.group(1).lower() if m else None


def classify(machine: str, device_model: str | None, cpuinfo: str | None,
             host_arch: str | None) -> tuple[str, str]:
    """Pure function: return (tag, reason). Unit-tested with recorded inputs."""
    m = machine.lower()
    if m in ("aarch64", "arm64"):
        if device_model and "raspberry pi 5" in device_model.lower():
            return "raspberry-pi-5", f"device-tree model: {device_model.strip(chr(0)).strip()}"
        impl = _cpuinfo_implementer(cpuinfo)
        if impl == "0x00":
            return "arm64-emulated-qemu", "cpuinfo CPU implementer 0x00 (QEMU synthetic CPU)"
        if host_arch and host_arch.lower() in _X86_HOSTS:
            return "arm64-emulated-qemu", f"launcher reports x86 host ({host_arch})"
        if impl in _ARM_IMPLEMENTERS:
            return "arm64-native", f"CPU implementer {impl} ({_ARM_IMPLEMENTERS[impl]})"
        if platform.system() == "Darwin":
            return "arm64-native", "macOS on Apple Silicon"
        return "arm64-native", "aarch64 with no emulation signal"
    if m in _X86_HOSTS:
        return "x86-native", f"machine {m}"
    return f"unknown-{m}", f"machine {m}"


def cpu_limit() -> float:
    """Effective CPU count: cgroup quota (docker --cpus) if set, else affinity / os count."""
    raw = _read("/sys/fs/cgroup/cpu.max")  # cgroup v2: "<quota> <period>" or "max <period>"
    if raw:
        parts = raw.split()
        if len(parts) == 2 and parts[0] != "max":
            return round(int(parts[0]) / int(parts[1]), 2)
    q, p = _read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"), _read("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if q and p and int(q) > 0:
        return round(int(q) / int(p), 2)
    try:
        return float(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return float(os.cpu_count() or 1)


def memory_limit_bytes() -> int | None:
    """cgroup memory limit (docker --memory) or total physical RAM."""
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        raw = _read(path)
        if raw and raw.strip().isdigit() and int(raw) < (1 << 60):
            return int(raw)
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except Exception:
        return None


def network_status() -> dict:
    """Offline proof without sending any traffic.

    1. Linux: a default route in /proc/net/route (destination 00000000) or ipv6_route.
    2. Portable: connect() a UDP socket to a TEST-NET address (RFC 5737). UDP connect only asks the
       kernel's routing table; no packet is sent. "Network unreachable" => no route.
    """
    has_default = None
    route = _read("/proc/net/route")
    if route is not None:
        has_default = any(ln.split()[1] == "00000000" for ln in route.splitlines()[1:] if len(ln.split()) > 1)
        r6 = _read("/proc/net/ipv6_route") or ""
        has_default = has_default or any(ln.split()[0] == "0" * 32 and ln.split()[1] == "00"
                                         for ln in r6.splitlines() if len(ln.split()) > 2
                                         and ln.split()[-1] != "lo")
    routable = True
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))
    except OSError:
        routable = False
    offline = (has_default is False) or not routable
    return {"offline": offline, "default_route": has_default, "routable": routable}


def detect() -> dict:
    machine = platform.machine()
    tag, reason = classify(machine, _read("/proc/device-tree/model"), _read("/proc/cpuinfo"),
                           os.environ.get("BASTION_HOST_ARCH"))
    cpuinfo = _read("/proc/cpuinfo")
    return {
        "environment": tag,
        "reason": reason,
        "timing_note": TIMING_NOTES.get(tag, "Unknown environment: do not compare timings."),
        "machine": machine,
        "system": platform.system(),
        "python": platform.python_version(),
        "cpu_implementer": _cpuinfo_implementer(cpuinfo),
        "cpu_part": _cpuinfo_part(cpuinfo),
        "cpu_limit": cpu_limit(),
        "memory_limit_bytes": memory_limit_bytes(),
        "host_arch_hint": os.environ.get("BASTION_HOST_ARCH"),
        "twin_profile": os.environ.get("BASTION_TWIN_PROFILE"),
    }


if __name__ == "__main__":  # python -m bastion.env
    import json
    print(json.dumps({**detect(), "network": network_status()}, indent=2))
