# 🌬️ Wind Turbine Acoustic Monitoring

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![Librosa](https://img.shields.io/badge/Librosa-Audio%20DSP-purple.svg)](https://librosa.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research-orange.svg)]()
[![Tests](https://img.shields.io/badge/Tests-pytest-yellow.svg)](tests/)

> **Research-grade acoustic condition monitoring for wind turbines** — from raw microphone signals to fault diagnosis, remaining useful life (RUL) prediction, and real-time alerting.

---

## 📖 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Why Acoustic Monitoring Matters](#-why-acoustic-monitoring-matters)
- [Fault Types Detected](#-fault-types-detected)
- [System Architecture](#-system-architecture)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Repository Structure](#-repository-structure)
- [Research Contribution](#-research-contribution-windnoisegenerator)
- [Future Roadmap](#-future-roadmap)

---

## 🔭 Project Overview

This project implements an **end-to-end acoustic monitoring pipeline** for wind turbine health assessment. Using audio captured near the nacelle and rotor, the system denoises environmental interference, extracts discriminative spectral features, detects anomalous behavior, classifies specific fault types, and estimates remaining useful life — all surfaced through a monitoring dashboard with automated alerts.

The work targets **research-grade reproducibility**: every stage is configurable, unit-tested, and benchmarked, with a novel synthetic wind-noise generator enabling controlled robustness experiments.

---

## ❗ Problem Statement

Wind turbines operate in remote, harsh environments where mechanical faults often progress silently until catastrophic failure:

- **Unplanned downtime** costs operators thousands of dollars per turbine per day.
- **Gearbox and bearing failures** account for the largest share of repair costs and downtime.
- **Traditional vibration-based SCADA monitoring** requires intrusive sensor installation on every component and often misses early-stage acoustic signatures.
- **Manual inspection** is expensive, infrequent, and dangerous at hub heights exceeding 100 m.

**Goal:** detect, classify, and forecast turbine faults *early* and *non-intrusively* from acoustic emissions, under real-world wind noise conditions.

---

## 🎧 Why Acoustic Monitoring Matters

| Property | Acoustic Monitoring | Vibration Sensors | Visual Inspection |
|---|---|---|---|
| **Non-contact** | ✅ Microphones placed externally | ❌ Mounted per-component | ✅ |
| **Early fault sensitivity** | ✅ Captures high-frequency precursors | ⚠️ Component-local only | ❌ Late-stage only |
| **Installation cost** | ✅ Low | ⚠️ High (per-component) | ⚠️ Recurring labor |
| **Continuous operation** | ✅ 24/7 | ✅ 24/7 | ❌ Periodic |
| **Whole-system coverage** | ✅ One sensor hears many components | ❌ | ⚠️ Surface only |

Acoustic signatures of faults (impact transients, harmonic sidebands, broadband friction noise) appear **before** measurable performance degradation — making sound a powerful early-warning modality. The key challenge, and the focus of this research, is **separating fault signatures from dominant wind noise**.

---

## ⚙️ Fault Types Detected

The classifier distinguishes **five operational states**:

| Class | Acoustic Signature | Typical Cause |
|---|---|---|
| 🟢 **Normal operation** | Smooth broadband aerodynamic noise, stable blade-pass frequency | Healthy baseline |
| 🔴 **Blade imbalance** | Amplitude modulation at rotor frequency (1P) and harmonics | Ice/dirt accretion, pitch error, blade damage |
| 🟠 **Bearing wear** | High-frequency impulsive bursts at characteristic defect frequencies (BPFO/BPFI) | Lubrication failure, spalling, fatigue |
| 🟡 **Gearbox fault** | Gear-mesh frequency sidebands, tonal harmonics, modulation patterns | Tooth wear, pitting, misalignment |
| 🔵 **Electrical fault** | Tonal hum at line frequency harmonics (50/100/150 Hz), arcing transients | Generator winding faults, converter issues |

---

## 🏗️ System Architecture

```mermaid
flowchart LR
    A["🎤 Raw Audio"] --> B["🔇 Denoising"]
    B --> C["📊 Feature Extraction"]
    C --> D["🚨 Anomaly Detection"]
    D --> E["🏷️ Fault Classification"]
    E --> F["⏳ RUL Prediction"]
    F --> G["📈 Dashboard & Alerts"]
```

```
Raw Audio
   │  16 kHz mono recordings from nacelle-mounted microphones
   ▼
Denoising
   │  Spectral gating + adaptive wind-noise suppression
   ▼
Feature Extraction
   │  Log-mel spectrograms, MFCCs, spectral kurtosis, envelope spectra
   ▼
Anomaly Detection
   │  Autoencoder reconstruction error → healthy/anomalous gate
   ▼
Fault Classification
   │  CNN classifier → {normal, blade imbalance, bearing, gearbox, electrical}
   ▼
RUL Prediction
   │  Degradation trend regression → remaining useful life estimate
   ▼
Dashboard & Alerts
      Live health indicators, fault probabilities, maintenance alerts
```

**Pipeline stages map directly to source modules:**

| Stage | Module |
|---|---|
| Denoising, feature extraction | `src/preprocessing/` |
| Anomaly detection, classification, RUL | `src/models/` |
| Experiment training loops | `src/training/` |
| Real-time scoring | `src/inference/` |

---

## 📂 Dataset

| Property | Value |
|---|---|
| **Source recordings** | Field recordings + public turbine acoustic datasets |
| **Sample rate** | 16 kHz (resampled), mono |
| **Clip length** | 10 s windows, 50% overlap |
| **Classes** | 5 (normal + 4 fault types) |
| **Augmentation** | Synthetic wind noise via `WindNoiseGenerator` (see [Research Contribution](#-research-contribution-windnoisegenerator)) |

**Data layout:**

```
data/
├── raw/         # Original immutable recordings (never modified)
├── processed/   # Denoised, segmented, feature-extracted tensors
└── augmented/   # SNR-controlled wind-noise mixtures for robust training
```

> ⚠️ Audio files are excluded from version control (see `.gitignore`). Place raw recordings in `data/raw/` and run the preprocessing scripts in `scripts/`.

---

## 🚀 Installation

**Prerequisites:** Python 3.10+, `pip`, and (optionally) a CUDA-capable GPU for training.

```bash
# 1. Clone the repository
git clone https://gitlab.com/noohkhan7232-group/noohkhan7232-project.git
cd noohkhan7232-project

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify the setup
pytest tests/
```

---

## 🗂️ Repository Structure

```
├── config/             # Experiment & hyperparameter configurations (YAML)
├── data/
│   ├── raw/            # Original, immutable acoustic recordings
│   ├── processed/      # Cleaned and feature-extracted data
│   └── augmented/      # Wind-noise-augmented training mixtures
├── docs/               # Extended documentation
├── notebooks/          # Exploratory analysis & result visualization
├── paper/              # Research paper drafts, figures, LaTeX sources
├── scripts/            # CLI scripts (preprocess, train, evaluate, export)
├── src/
│   ├── preprocessing/  # Denoising, segmentation, feature extraction
│   ├── models/         # Autoencoder, CNN classifier, RUL regressor
│   ├── training/       # Training loops, schedulers, experiment tracking
│   ├── inference/      # Real-time scoring & dashboard backends
│   └── utils/          # Audio I/O, metrics, plotting, seeding
└── tests/              # Unit & integration tests (pytest)
```

---

## 🔬 Research Contribution: `WindNoiseGenerator`

A central obstacle in turbine acoustics research is the lack of **labeled recordings across diverse wind conditions**. This project contributes a physically-motivated synthetic wind noise generator used to systematically stress-test model robustness:

| Component | Description |
|---|---|
| **1/f² spectrum** | Generates noise with a Brownian (red) power spectral density matching the low-frequency energy roll-off of natural atmospheric wind |
| **Gust modulation** | Low-frequency stochastic amplitude envelopes simulating realistic gust events and lulls |
| **Turbulence simulation** | Band-limited stochastic fluctuations modeling micro-scale turbulence interacting with the microphone and structures |
| **SNR-controlled mixing** | Deterministic mixing of fault signals with generated noise at precise target SNRs (e.g., −5 dB to +20 dB) for controlled robustness curves |

**Why it matters:**

- 📈 Enables **robustness-vs-SNR benchmark curves** for every model — a reproducible evaluation protocol.
- 🔁 Provides **unlimited augmentation data** without expensive field campaigns.
- 🧪 Supports **ablation studies** isolating the effect of each noise phenomenon (spectrum shape vs. gusting vs. turbulence) on classifier degradation.

Implementation lives in `src/preprocessing/`, with augmented mixtures written to `data/augmented/`.

---

## 🗺️ Future Roadmap

- [ ] **Self-supervised pretraining** on unlabeled field audio (wav2vec-style) for label-efficient fine-tuning
- [ ] **Multi-microphone array support** with beamforming for component localization
- [ ] **Edge deployment** — quantized ONNX models for on-nacelle inference hardware
- [ ] **SCADA fusion** — combine acoustic features with operational telemetry (wind speed, power, pitch)
- [ ] **Probabilistic RUL** — calibrated uncertainty intervals for maintenance scheduling
- [ ] **Live dashboard** — Grafana/web dashboard with streaming inference and alert webhooks
- [ ] **Public benchmark release** — publish the WindNoiseGenerator evaluation protocol and baseline results

---

## 📄 License & Citation

Released for research purposes. If you use the `WindNoiseGenerator` or evaluation protocol in your work, please cite this repository.

---

<p align="center"><i>Listening to turbines, so failures never go unheard.</i> 🌬️🎧</p>
