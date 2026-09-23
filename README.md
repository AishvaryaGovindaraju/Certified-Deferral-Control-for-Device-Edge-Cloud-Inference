# Certified Deferral Control for Device–Edge–Cloud Inference

> **Distribution-Free Risk Guarantees from Self-Labelled IoT Streams**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Status: Experimental Results](https://img.shields.io/badge/Status-Benchmark%20Data-brightgreen.svg)]()

---

## 📌 Overview

**CoDeC** (**C**ertified **D**eferral **C**ontrol) provides finite-sample distribution-free risk guarantees ($P(\text{end-to-end selective risk} \le \alpha) \ge 1 - \delta$) for multi-tier Device–Edge–Cloud cascaded inference systems under realistic communication, energy, latency, and monetary cost constraints.

In modern IoT deployments, sensor nodes (device) filter simple inputs, edge gateways process moderate complexities, and the cloud serves high-capacity neural networks. However, standard cascade thresholds (e.g., fixed confidence, entropy, or reinforcement learning) lack formal safety/error guarantees. Furthermore, stream recalibration often suffers from logging bias since the edge-cloud cascade only receives true labels for samples escalated to the cloud.

CoDeC solves this via:
1. **Multi-Dimensional Fixed-Sequence Learn-then-Test (LTT):** Jointly certifies threshold vectors across device, gateway, and cloud tiers while minimizing heterogeneous resource cost (mJ, uplink payload bytes, latency, USD).
2. **$\epsilon$-Exploration Deferral & Propensity-Weighted Conformal Recalibration:** Eliminates logged-bandit selection bias in self-labelled streams using inverse propensity weighting (IPW) and stratified bounds.
3. **Drift-Adaptive Online Control:** Regulates empirical risk under severe temporal distribution shift (e.g., 36-month sensor drift) subject to hard Lagrangian uplink budget constraints.

---

## 📊 Repository Contents & Dataset Profiles

This repository contains the empirical benchmark evaluation artifacts, statistical significance test suites, and tier profiling matrices for three standard IoT stream benchmarks:
- **HAR** (Human Activity Recognition)
- **Gas Sensor Array** (Long-term sensor drift across 36 months)
- **RT-IoT2022** (Real-Time Internet of Things Network Attack & Intrusion Detection)

### Summary of Data Files

| File | Description |
| :--- | :--- |
| [`tier_profile.csv`](tier_profile.csv) | Per-tier hardware profile (input dimensions, hidden layers, parameter count, MACs, transmission payload bytes, measured microsecond latency, modelled energy in mJ, latency in ms, and USD monetary cost). |
| [`table_main.csv`](table_main.csv) | Aggregated benchmark results (accuracy, fidelity risk, risk violation rates, normalized cost scalar, cloud escalation rate, energy, latency, USD) across all baseline strategies. |
| [`all_runs.csv`](all_runs.csv) | Raw per-run evaluation traces across varying $\alpha, \delta, \epsilon$, outage probabilities, random seeds, and sliding evaluation windows. |
| [`estimator_audit.csv`](estimator_audit.csv) | Risk estimator audit comparing Ground Truth Buffer risk, Stratified Upper Confidence Bound (UCB), IPW-UCB, and Naive (biased) empirical estimates across calibration windows. |
| [`stats_tests.csv`](stats_tests.csv) | Paired statistical significance tests (Wilcoxon signed-rank tests with Holm-Bonferroni correction) comparing CoDeC violation rates against baselines. |
| [`related_work_map.csv`](related_work_map.csv) | Systematic taxonomy and research gap mapping against state-of-the-art early exit, split inference, and conformal prediction methods. |

---

## 🔬 Experimental Results Summary

### 1. Tier Architecture Profile

```
+------------------+         +-------------------+         +-----------------+
|   Device Tier    |  ====>  |   Gateway Tier    |  ====>  |   Cloud Tier    |
| (Microcontroller)|         |    (Edge Server)  |         | (High-Perf GPU) |
+------------------+         +-------------------+         +-----------------+
  - Ultra-low latency          - Moderate capacity           - Highest capacity
  - Nano/Micro-joules          - Sub-10ms processing         - Cloud billing / API cost
  - Raw sensor features        - Aggregated features         - Full feature representation
```

### 2. Risk Guarantee & Violation Comparison

CoDeC controls risk strictly below the user-specified tolerance $\alpha = 0.05$ with statistical guarantee $1 - \delta = 0.90$. In contrast, heuristic baselines (fixed confidence, entropy) exhibit severe empirical violation rates under distribution shift and uncalibrated thresholds ($p < 10^{-10}$ via Holm-adjusted Wilcoxon tests).

---