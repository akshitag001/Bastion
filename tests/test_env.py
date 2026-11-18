"""Environment tagging (honesty layer). Inputs mirror what each platform really exposes;
the QEMU cpuinfo below was recorded inside the arm64 twin on an x86 host."""

from bastion import env

QEMU_CPUINFO = """processor\t: 0
model name\t: ARMv8 Processor rev 0 (v8l)
BogoMIPS\t: 100.00
Features\t: fp asimd aes pmull sha1 sha2 crc32 atomics
CPU implementer\t: 0x00
CPU architecture: 8
CPU variant\t: 0x0
CPU part\t: 0x051
CPU revision\t: 0
"""
PI5_CPUINFO = QEMU_CPUINFO.replace("0x00", "0x41").replace("0x051", "0xd0b")   # Cortex-A76
GRAVITON2_CPUINFO = QEMU_CPUINFO.replace("0x00", "0x41").replace("0x051", "0xd0c")  # Neoverse N1


def test_raspberry_pi_5_from_device_tree():
    tag, _ = env.classify("aarch64", "Raspberry Pi 5 Model B Rev 1.0\x00", PI5_CPUINFO, None)
    assert tag == "raspberry-pi-5"


def test_qemu_emulation_detected_from_cpuinfo():
    tag, reason = env.classify("aarch64", None, QEMU_CPUINFO, None)
    assert tag == "arm64-emulated-qemu" and "0x00" in reason


def test_qemu_emulation_detected_from_host_hint():
    tag, _ = env.classify("aarch64", None, None, "x86_64")
    assert tag == "arm64-emulated-qemu"


def test_emulation_beats_host_hint_absent_and_real_arm_is_native():
    assert env.classify("aarch64", None, GRAVITON2_CPUINFO, None)[0] == "arm64-native"


def test_real_arm_is_never_tagged_pi5_without_device_tree():
    # A Cortex-A76 cpuinfo alone is not proof of a Pi 5.
    assert env.classify("aarch64", None, PI5_CPUINFO, None)[0] == "arm64-native"


def test_x86():
    assert env.classify("AMD64", None, None, None)[0] == "x86-native"
    assert env.classify("x86_64", None, None, None)[0] == "x86-native"


def test_detect_has_required_fields():
    d = env.detect()
    for k in ("environment", "timing_note", "machine", "python", "cpu_limit"):
        assert k in d
    assert d["environment"] in env.TAGS or d["environment"].startswith("unknown-")


def test_network_status_never_raises():
    n = env.network_status()
    assert isinstance(n["offline"], bool)
