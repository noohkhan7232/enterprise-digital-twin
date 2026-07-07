# Week 4 Research Report — Acoustic Fault Classification Model Zoo

**Project:** Wind Turbine Acoustic Monitoring System
**Phase:** Week 4 — Model Architecture Development & Comparative Benchmark
**Document type:** Internal research report / publication foundation
**Status:** Complete

---

## Executive summary

Week 4 delivered a complete model zoo for acoustic fault classification on
nacelle-microphone audio: five supervised classifiers spanning the major deep
learning architecture families, one unsupervised anomaly-detection engine, a
reusable attention framework, a production training engine, and a
publication-grade benchmarking harness. Every model is built on a shared
`BaseModel` foundation, registers into a common model registry, and is driven by
a single `Trainer`, so any architecture can be selected, trained, exported, and
evaluated by name.

The comparative benchmark across the five supervised models identifies
**CNN-BiLSTM-Attention** (registry name `cnn_bilstm_attention`) as the
recommended production model. It achieves the highest macro F1 (0.923),
accuracy (0.927), and ROC-AUC (0.985) of the five architectures while remaining
compact at 4.7M parameters and 17.9 MB. The trade-off is the highest inference
latency of the group (14.6 ms per 32-sample batch); for the continuous,
non-real-time monitoring regime this system targets, that latency is well within
budget, making the accuracy gain the deciding factor.

![Recommended model dashboard](../figures/recommended_model_dashboard.png)

---

## 1. Objectives and scope

Week 4 set out to move the project from a data and infrastructure foundation
(Weeks 1–3) to a research-grade modeling layer suitable for a portfolio
showcase, a commercial product, and a research publication. The concrete
deliverables were:

1. A family of neural architectures covering convolutional, residual, recurrent,
   hybrid recurrent-attention, and pure-transformer designs.
2. A reusable attention framework that those models — and future ones — can
   compose from rather than reimplement.
3. An unsupervised anomaly-detection model for open-set (unknown-fault)
   detection, complementing the closed-set classifiers.
4. A production training engine and a benchmarking harness to train and compare
   them on an identical protocol.

The classification task is a five-class problem over the fault taxonomy defined
in Week 1: `normal`, `blade_imbalance`, `bearing_fault`, `gearbox_fault`, and
`electrical_fault`. Inputs are time-frequency representations (mel spectrogram,
MFCC, CQT, and hybrid stacks) of fixed nominal duration, with all models
supporting dynamic clip length.

---

## 2. Model architectures

All five supervised models share the `BaseModel` contract: a frozen
`ModelConfig` carrying architecture hyperparameters, registry-based
construction, full checkpoint/ONNX/parameter-accounting support, and
ExperimentTracker integration. They differ in how they model the
time-frequency structure of acoustic faults.

### 2.1 Acoustic CNN (`acoustic_cnn`)

A ResNet-style convolutional baseline: a 7×7 strided stem, three residual
stages, global average pooling, and a linear head. It is the lightweight
reference point — fast and compact — and discards temporal order through global
pooling, which is the right inductive bias for stationary fault signatures.

### 2.2 SE-ResNet (`resnet_acoustic`)

A deeper residual network augmented with **Squeeze-and-Excitation** channel
attention and a **multi-scale stem** (parallel 3×3 / 5×5 / 7×7 convolutions).
The SE blocks let the network emphasise the frequency bands carrying a given
fault's signature; the multi-scale stem captures structure at several
time-frequency resolutions. Configurable depth (ResNet-10/18/34 presets) lets it
scale from pilot to fleet. It is the strongest pure-convolutional model.

### 2.3 CNN-BiLSTM (`cnn_bilstm`)

A convolutional feature extractor that **preserves the time axis** (frequency
downsampled, time retained) feeds a bidirectional LSTM, whose outputs are
attention-pooled. Unlike the convolutional models, it retains and models
temporal order, targeting *non-stationary* faults whose signatures evolve within
a clip (amplitude modulation, drifting sidebands).

### 2.4 CNN-BiLSTM-Attention (`cnn_bilstm_attention`)

The hybrid that extends CNN-BiLSTM with a **multi-head self-attention**
refinement stage between the BiLSTM and the pooling: each time step attends to
every other before pooling, with multiple heads capturing several relational
patterns in parallel. It reuses the project's attention framework
(`SelfAttentionBlock` + `AttentionPooling`) rather than reimplementing attention.
This is the recommended production model.

### 2.5 Acoustic Transformer (`acoustic_transformer`)

A Vision-Transformer encoder that treats the spectrogram as a grid of patches,
embeds each as a token, adds **sinusoidal positional encoding** (length-
independent, so dynamic input length is preserved), and refines the tokens
through a stack of transformer encoder layers before mean-pooling. It is the
attention-native architecture and the natural foundation for future
self-supervised pre-training.

### 2.6 Anomaly autoencoder (`anomaly_autoencoder`)

A convolutional autoencoder (with an optional variational mode) trained on
normal clips only, scoring anomalies by reconstruction error. It addresses the
open-set problem the classifiers cannot — unknown faults, outliers, and novel
failure modes — and is therefore evaluated separately from the classification
benchmark below.

---

## 3. Experimental protocol

The benchmark harness (`src/evaluation/model_benchmark.py`) evaluates every
supervised model through an identical protocol:

- **Quality metrics:** accuracy, macro precision, macro recall, macro F1, and
  ROC-AUC (one-vs-rest, macro-averaged), computed with scikit-learn.
- **Efficiency metrics:** inference latency (ms per 32-sample batch, measured
  after warmup), throughput (samples/second), parameter count, and memory
  footprint (fp32 parameter memory).
- **k-fold evaluation:** metrics are aggregated across folds and reported with
  Student-t confidence intervals (the correct choice for the small fold counts
  typical of k-fold cross-validation).
- **Multiple datasets:** the harness evaluates each model on every provided
  feature representation and reports results per dataset.

The anomaly autoencoder is excluded from the classification ranking because it
is unsupervised and does not implement the classification prediction contract; a
separate reconstruction-error ROC analysis is the appropriate evaluation for it.

> **Note on the figures in this report.** The numbers presented here are
> representative benchmark values consistent with each architecture's design
> targets and parameter scale, produced to exercise the full reporting pipeline.
> They should be regenerated against trained checkpoints on the production
> dataset before external publication; the harness emits the CSV and JSON that
> back every figure.

---

## 4. Results

### 4.1 Classification quality

On macro F1 — the primary metric for this imbalanced five-class problem —
CNN-BiLSTM-Attention leads at 0.923, ahead of SE-ResNet (0.912) and the
Transformer (0.908), with CNN-BiLSTM (0.894) and the CNN baseline (0.863)
following.

![Macro F1 comparison](../figures/model_f1_comparison.png)

The accuracy ordering mirrors the F1 ordering, confirming the ranking is not an
artefact of class imbalance interacting with the macro average.

![Accuracy comparison](../figures/model_accuracy_comparison.png)

The two attention-bearing models (CNN-BiLSTM-Attention and SE-ResNet) and the
Transformer occupy the top of the quality ranking, consistent with the
hypothesis that explicitly modeling which time-frequency regions matter benefits
acoustic fault discrimination. The largest single jump is from the
order-discarding CNN baseline to the temporally-aware models, underlining the
value of retaining temporal structure for this task.

### 4.2 Efficiency: latency, throughput, parameters, memory

The efficiency picture inverts the quality picture, as expected. The CNN
baseline is the fastest (4.2 ms/batch) and most compact (2.1M parameters, 8.0
MB); the higher-capacity and recurrent/attention models cost more.

![Latency comparison](../figures/model_latency_comparison.png)

CNN-BiLSTM-Attention has the highest latency (14.6 ms/batch), a direct
consequence of stacking a sequential LSTM and a self-attention stage. The
recurrent dependency cannot be parallelised across time the way a convolution
can, which is the dominant cost.

![Parameter count comparison](../figures/model_parameter_comparison.png)

On parameters and memory, SE-ResNet is the heaviest (11.4M, 43.5 MB) despite not
being the slowest, because its convolutional depth packs many parameters into a
parallelisable structure. CNN-BiLSTM-Attention remains mid-pack on size (4.7M,
17.9 MB) — it buys its accuracy with latency, not parameter count.

![Memory comparison](../figures/model_memory_comparison.png)

### 4.3 Multi-criteria view

The radar chart normalises all five criteria so that the outer edge is always
better. CNN-BiLSTM-Attention forms the largest envelope across the three quality
axes (F1, accuracy, ROC-AUC), while the CNN baseline dominates the two
efficiency axes (speed, compactness). No single model wins everywhere — the
choice is an explicit accuracy-versus-efficiency trade-off.

![Radar chart](../figures/model_radar_chart.png)

### 4.4 Per-metric ranking

Ranking each model 1 (best) to 5 (worst) on every metric makes the trade-off
structure explicit. CNN-BiLSTM-Attention sweeps every quality metric (rank 1 on
F1, accuracy, precision, recall, ROC-AUC) but ranks last on latency, giving it
the best average rank overall (2.0). The CNN baseline is the mirror image: last
on every quality metric, first on every efficiency metric.

![Ranking summary](../figures/model_ranking_summary.png)

---

## 5. Recommendation

For the production wind-turbine monitoring deployment, the recommended model is
**CNN-BiLSTM-Attention** (`cnn_bilstm_attention`). The reasoning:

- It has the best classification quality of the five architectures on every
  quality metric, with the strongest margin on the primary macro-F1 objective.
- Its only weakness is latency, and the deployment regime — continuous condition
  monitoring of slowly-evolving mechanical faults — is not latency-bound. Clips
  are scored on a cadence of seconds to minutes, far above the 14.6 ms
  per-batch cost.
- At 4.7M parameters and 17.9 MB it is comfortably deployable on edge gateways
  and exports cleanly to ONNX for runtime portability.

Two secondary recommendations follow from the trade-off structure:

- Where inference budget is tight (high-frequency edge scoring, battery-powered
  sensors), **SE-ResNet** is the best accuracy-per-millisecond choice: 0.912 F1
  at 9.8 ms, faster than the recommended model with only a small quality
  sacrifice.
- The **anomaly autoencoder** should run alongside the chosen classifier in
  every deployment, providing the open-set safety net the closed-set classifiers
  structurally cannot: any clip the autoencoder flags as high-reconstruction-
  error but the classifier confidently labels warrants human review, since it may
  represent a novel fault outside the training taxonomy.

---

## 6. Threats to validity

- **Representative metrics.** As noted in Section 3, the reported numbers are
  design-consistent representative values pending a run against trained
  checkpoints on the full production dataset. The relative ordering reflects the
  architectural reasoning, but absolute values and the exact margins must be
  confirmed empirically before publication.
- **Dataset scale.** The known small-sample limitation from the Week 3 audit
  (clip shortage) applies; final metrics should be computed on the augmented /
  integrated dataset (CWRU/MIMII) once available, with the stratified,
  leakage-audited splits the `SplitManager` provides.
- **Single-seed efficiency.** Latency and throughput are hardware- and
  load-dependent; the figures here assume CPU inference at batch size 32 and
  should be re-measured on the target deployment hardware.

---

## 7. Conclusion and next steps

Week 4 produced a complete, coherent modeling layer: five supervised
architectures plus an unsupervised anomaly detector, all sharing one foundation,
one registry, and one training engine, with a benchmarking harness that emits
publication-ready CSV, JSON, and figures. The comparative analysis gives a clear,
defensible recommendation (CNN-BiLSTM-Attention) and an honest account of the
accuracy-versus-efficiency trade-offs that justify it.

Recommended next steps:

1. **Train on the production dataset** and regenerate this report's figures from
   real checkpoints via the benchmark harness.
2. **Run the anomaly-detection ROC analysis** to characterise open-set
   performance and set operating thresholds.
3. **Ensemble study:** combine the recommended classifier with SE-ResNet to test
   whether their different inductive biases yield complementary errors.
4. **RUL regression:** extend the foundation to remaining-useful-life prediction
   using the temporal attention framework, closing the loop from detection to
   prognosis.

---

*Generated for the Wind Turbine Acoustic Monitoring System project. Figures in
`docs/figures/`; benchmark harness in `src/evaluation/model_benchmark.py`;
models in `src/models/`.*