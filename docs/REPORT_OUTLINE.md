# Engineering report outline (Phase 1)

1. Problem and goals (PS3 scope; why on-device: privacy, latency, offline, cost)
2. System architecture (pipeline diagram; detector vs Llama roles; verdict lock)
3. Hardware mapping: Raspberry Pi 5 specs, OS, what runs where (detectors and Llama on Arm CPU via ExecuTorch/XNNPACK; NPU not used and why)
4. Data: datasets, licences, cleaning, dedup, splits, leakage checks (leak probe result)
5. Detectors: features, models compared (table: F1, recall, FPR, FNR, AUC, size, latency), selection rationale, thresholds (recall-first)
6. Explanation layer: Llama model + quantisation settings (8da4w, group size, seq len), prompt design, prompt-injection mitigation, guard, fallback rate
7. On-device KPIs: Pi vs laptop (latency p50/p95, RAM, model size, CPU, throughput, Llama tokens/s, end-to-end time), offline proof
8. User effectiveness: sample explanations, advice quality
9. Limitations and failure cases (honest)
10. Phase 2 roadmap (network anomaly, persistent Llama runtime, Android app, NPU)

Every number must reference a JSON file in `benchmarks/results/`.
