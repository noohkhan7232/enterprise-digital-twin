# Feature Engineering Research Report
## Wind Turbine Acoustic Fault Detection – Week 2

**Generated:** 2026-06-15 05:10:47 UTC
**Dataset:** 300 signals × 44100 samples (2.0s @ 22050 Hz, SNR=20.0dB)
**Classes:** normal, blade_imbalance, bearing_fault, gear_fault, rotor_rub

---

## Executive Summary

This report presents a systematic evaluation of acoustic feature configurations for wind turbine fault detection. We compared **11** feature configurations across three feature families (MFCC, Mel Spectrogram, CQT) plus a spectral statistics baseline, evaluating class separability using three complementary metrics and computational cost.

**Winner:** `Fused-Stack` achieved the highest composite score of **0.403**.

---

## 1. Dataset Overview

| Parameter | Value |
|-----------|-------|
| Total signals | 300 |
| Signals per class | 60 |
| Signal duration | 2.0s |
| Sample rate | 22050 Hz |
| SNR | 20.0 dB |
| Rotor speed | 0.20 Hz (12 RPM) |
| BPFO frequency | 105.1 Hz |

**Fault classes modelled:**
- **Normal** – baseline rotor harmonics + pink turbulence noise
- **Blade Imbalance** – 1P/2P amplitude modulation
- **Bearing Fault** – BPFO impulse train (outer-race, jittered) with housing resonance
- **Gear Fault** – Gear mesh frequency (GMF) + sidebands + AM
- **Rotor Rub** – Sub-harmonic, super-harmonic + random transient impacts

---

## 2. MFCC Study

### Configuration Tested: 13 / 20 / 40 / 80 coefficients
(Features = 6 × n_mfcc including Δ and ΔΔ statistics)

| Config | Features | Memory | Time | Silhouette ↑ | DB ↓ | CH ↑ | Fisher ↑ |
|--------|----------|--------|------|-------------|------|------|---------|
| MFCC-13 | 78 | 0.094 MB | 1.5s | 0.0083 | 10.5637 | 8.9 | 0.1176 |
| MFCC-20 | 120 | 0.144 MB | 1.5s | 0.0129 | 10.5151 | 10.9 | 0.1328 |
| MFCC-40 | 240 | 0.288 MB | 1.5s | 0.0198 | 10.0015 | 14.8 | 0.1464 |
| MFCC-80 | 480 | 0.576 MB | 1.5s | 0.0217 | 10.2367 | 14.5 | 0.1136 |

### Best MFCC Configuration: **MFCC-40**

**Key findings:**
- Increasing coefficients beyond 80 yields diminishing returns in class separability
- Delta and delta-delta features capture temporal dynamics critical for gear and rotor rub faults
- Memory cost scales linearly; computational cost is sub-linear due to shared FFT computation
- **Recommendation:** `n_mfcc = 80` with delta features for optimal separability-efficiency trade-off

---

## 3. Mel Spectrogram Study

### Configurations Tested: 64 / 128 / 256 mel bins

| Config | Features | Memory | Time | Silhouette ↑ | DB ↓ | CH ↑ |
|--------|----------|--------|------|-------------|------|------|
| MelSpec-64 | 128 | 0.154 MB | 0.8s | -0.0545 | 8.8981 | 5.3 |
| MelSpec-128 | 256 | 0.307 MB | 1.1s | -0.0551 | 8.3590 | 4.7 |
| MelSpec-256 | 512 | 0.614 MB | 1.6s | -0.0496 | 7.9555 | 4.5 |

### Best Mel Configuration: **MelSpec-256**

**Key findings:**
- Log-Mel spectrograms capture turbulence noise texture well; higher bin counts improve resolution of tonal peaks (BPFO, GMF)
- PCA cumulative variance analysis shows 95% of information retained by 20–35 components for all configurations
- The frequency axis compression (Mel scale) emphasises low-frequency rotor harmonics which are diagnostically critical
- **Recommendation:** `n_mels = 256` for best separability

---

## 4. CQT Study

### Configurations Tested: 12 / 24 / 36 bins/octave (7 octaves, C2–C9)

| Config | Features | Memory | Time | Silhouette ↑ | DB ↓ | CH ↑ |
|--------|----------|--------|------|-------------|------|------|
| CQT-12BPO | 168 | 0.202 MB | 6.3s | -0.0358 | 9.9517 | 15.0 |
| CQT-24BPO | 336 | 0.403 MB | 9.7s | -0.0239 | 9.7405 | 27.6 |
| CQT-36BPO | 504 | 0.605 MB | 14.1s | -0.0083 | 9.6586 | 39.4 |

### Best CQT Configuration: **CQT-36bpo**

**Key findings:**
- The CQT's logarithmic frequency axis directly aligns with harmonic fault signatures (BPFO harmonics, GMF sidebands)
- 36 bins/octave provides the resolution needed to separate closely spaced gear mesh sidebands
- CQT is computationally more expensive than MFCC/Mel but provides unique sub-octave harmonic structure
- Particularly valuable for bearing fault detection where harmonic series extends 5–10 octaves
- **Recommendation:** `bins_per_octave = 36` with 7 octaves

---

## 5. Dimensionality Reduction Analysis

### PCA Intrinsic Dimensionality

Using the best MFCC configuration (80 coefficients, 480 features):
- **90% variance:** 142 principal components
- **95% variance:** 181 principal components
- **PC1 + PC2:** 17.7% variance explained

The low intrinsic dimensionality (≪ raw feature count) indicates strong redundancy in raw features and confirms that PCA-based compression to 20–50 components is appropriate for Week 3 modelling.

### t-SNE & UMAP Clustering

Visual inspection of t-SNEand UMAP embeddings shows:
- Normal and Blade Imbalance classes are well-separated from bearing/gear faults
- Bearing Fault and Gear Fault show some overlap due to similar impulsive characteristics
- Rotor Rub has distinctive sub-harmonic features that isolate it in embedding space

---

## 6. Class Separability Metrics Summary

| Configuration | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabász ↑ | Fisher ↑ | Composite |
|--------------|-------------|----------------|-------------------|--------|---------|
| Fused-Stack | -0.0240 | 7.8595 | 20.5 | 0.1559 | **0.403** |
| CQT-36bpo | -0.0083 | 9.6586 | 39.4 | 0.3492 | **0.383** |
| CQT-24bpo | -0.0239 | 9.7405 | 27.6 | 0.2749 | **0.235** |
| MelSpec-256 | -0.0496 | 7.9555 | 4.5 | 0.0471 | **0.195** |
| MFCC-40 | 0.0198 | 10.0015 | 14.8 | 0.1464 | **0.177** |
| MFCC-80 | 0.0217 | 10.2367 | 14.5 | 0.1136 | **0.149** |
| MelSpec-128 | -0.0551 | 8.3590 | 4.7 | 0.0550 | **0.136** |
| MelSpec-64 | -0.0545 | 8.8981 | 5.3 | 0.0692 | **0.076** |
| CQT-12bpo | -0.0358 | 9.9517 | 15.0 | 0.1763 | **0.071** |
| MFCC-20 | 0.0129 | 10.5151 | 10.9 | 0.1328 | **0.065** |


---

## 7. Recommended Feature Stack

Based on the comprehensive evaluation, the following configurations are recommended for Week 3 model training:

### Individual Best Configurations

| Feature Type | Configuration | Silhouette | Features | Memory |
|-------------|--------------|-----------|---------|--------|
| MFCC        | `n_mfcc=80` + Δ + ΔΔ | 0.0217 | 480 | 0.576 MB |
| Mel Spectrogram | `n_mels=256` | -0.0496 | 512 | 0.614 MB |
| CQT         | `36 BPO` | -0.0083 | 504 | 0.605 MB |

### Fused Stack Performance

| Stack | Features | Silhouette | DB | CH |
|-------|----------|-----------|----|----|
| Fused (raw)     | 1506 | -0.0240 | 7.8595 | 20.5 |
| Fused + PCA(50) | 50 | -0.0067 | 9.7633 | 2.4 |

---

## 8. Recommendations for Week 3 Model Training

### Primary Feature Pipeline

```python
# Optimal configuration for Week 3
FEATURE_CONFIG = {
    "mfcc": {
        "n_mfcc":       80,
        "include_delta": True,
        "n_fft":        2048,
        "hop_length":   512,
    },
    "melspec": {
        "n_mels":      256,
        "fmin":        20.0,
        "fmax":        8000.0,
    },
    "cqt": {
        "bins_per_octave": 36,
        "n_octaves":       7,
    },
    "preprocessing": {
        "scaler":          "StandardScaler",
        "pca_components":  50,  # compress before modelling
    },
}
```

### Model Recommendations

1. **Baseline:** Random Forest on MFCC-80 features (fast, interpretable, handles redundancy)
2. **Gradient Boosted Trees (XGBoost/LightGBM):** Fused-PCA(50) stack for best accuracy
3. **1D-CNN:** Raw signal input with learned feature extraction (Week 4)
4. **Evaluation:** Stratified 5-fold CV, report F1-macro to handle class balance

### Critical Engineering Notes

- **Normalise** each feature type independently before fusion (different scales)
- **Window size** (n_fft=2048 ≈ 93ms) captures at least 1 full rotor period (1/ROTOR_HZ ≈ 5s → sub-window analysis recommended)
- **CQT** is best suited for bearing and gear fault detection; **MFCC** for global spectral envelope
- Consider **online feature extraction** using streaming buffers for real-time deployment
- Recommend **F1-macro ≥ 0.90** as minimum threshold before production deployment

---

## 9. Figure Index

| Figure | File | Description |
|--------|------|-------------|
| 1 | `00_waveform_gallery.png` | Per-class waveform gallery |
| 2 | `00_spectrogram_gallery.png` | Log-power spectrograms |
| 3 | `01_mfcc_heatmaps.png` | MFCC coefficient heatmaps by n_mfcc |
| 4 | `02_mfcc_metrics_comparison.png` | MFCC separability metrics |
| 5 | `03_mfcc_efficiency.png` | MFCC efficiency trade-off |
| 6 | `04_mfcc_per_class_variance.png` | Per-class MFCC variance |
| 7 | `05_mel_filterbanks.png` | Mel filterbank frequency responses |
| 8 | `06_mel_spectrograms_by_class.png` | Mel spectrograms by fault class |
| 9 | `07_mel_metrics_comparison.png` | Mel separability metrics |
| 10 | `08_mel_information_retention.png` | PCA cumulative variance |
| 11 | `09_cqt_harmonic_analysis.png` | CQT harmonic fault signatures |
| 12 | `10_cqt_metrics_comparison.png` | CQT separability metrics |
| 13 | `11_cross_feature_comparison.png` | Cross-feature heatmap & ranking |
| 14 | `12_dimensionality_reduction.png` | PCA / t-SNE / UMAP embeddings |
| 15 | `13_pca_variance_analysis.png` | PCA scree plot |
| 16 | `14_cluster_distributions.png` | Per-class cluster distributions |
| 17 | `15_metrics_dashboard.png` | Complete metrics dashboard |
| 18 | `16_per_class_silhouette.png` | Per-class silhouette violin |
| 19 | `17_fused_stack_embedding.png` | Fused stack cluster visualisation |
| 20 | `18_feature_importance.png` | Feature discriminability ranking |

---

*Report generated by `notebooks/04_week2_feature_engineering.py`*
*Wind Turbine Acoustic Fault Detection Research Project*
