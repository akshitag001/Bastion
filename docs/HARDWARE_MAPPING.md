# Hardware mapping (fill in from real measurements)

| Item | Value |
|---|---|
| Device | Raspberry Pi 5 (RAM: _ GB) |
| OS / kernel | |
| Python | |
| ExecuTorch version | |
| Cooling / power supply | |

| Component | Runs on | Runtime | Format | Size |
|---|---|---|---|---|
| URL detector | Arm CPU | scikit-learn | joblib | |
| SMS detector | Arm CPU | scikit-learn | joblib | |
| Llama 3.2 1B explainer | Arm CPU (4x Cortex-A76) | ExecuTorch + XNNPACK | .pte (8da4w) | |
| Hailo NPU | not used in Phase 1 | | | |
