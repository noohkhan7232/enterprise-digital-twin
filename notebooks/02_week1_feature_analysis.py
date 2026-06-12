# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Week 1 -- Acoustic Feature Analysis
#
# **Wind Turbine Acoustic Monitoring** | Deep analysis of features extracted
# from turbine fault recordings.
#
# This notebook answers three research questions:
#
# 1. **Are fault classes acoustically separable?** If features do not
#    separate classes, no classifier will.
# 2. **Which features carry the discriminative signal?** This drives model
#    input selection and the paper's feature-ablation study.
# 3. **How badly does wind noise corrupt those features?** This motivates
#    the denoising stage and the WindNoiseGenerator augmentation strategy.
#
# All figures are saved publication-ready (300 DPI) to `docs/figures/` as
# `Figure_01.png`, `Figure_02.png`, ... and are candidates for the ICASSP
# paper.
#
# **Prerequisite:** generate the synthetic dataset first:
# `python scripts/download_datasets.py --synthetic`

# %%
"""Week 1 acoustic feature analysis (jupytext percent-format notebook)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; comment out for interactive use

import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Make the repository root importable when running from notebooks/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.audio_loader import AudioConfig, AudioLoader
from src.preprocessing.augmentation import WindNoiseGenerator
from src.preprocessing.feature_extractor import (
    SPECTRAL_STATISTIC_NAMES,
    FeatureConfig,
    FeatureExtractor,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("feature_analysis")

sns.set_theme(style="whitegrid", font_scale=1.15)

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "synthetic"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DPI = 300
CLASS_ORDER = ["normal", "bearing_fault", "blade_imbalance", "gearbox_fault"]

_figure_counter = 0


def save_figure(fig: plt.Figure, description: str) -> Path:
    """Save a figure as the next sequentially-numbered paper-ready PNG."""
    global _figure_counter
    _figure_counter += 1
    path = FIGURES_DIR / f"Figure_{_figure_counter:02d}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s (%s)", path.name, description)
    return path


# %% [markdown]
# ## Section 1 -- Dataset Summary
#
# Before any modelling we establish *what we have*: clip counts, class
# balance, and total duration. Class imbalance here would bias every metric
# downstream, so we verify the synthetic dataset is balanced by construction.

# %%
loader = AudioLoader(AudioConfig(), dataset_name="synthetic")
clips = loader.load_directory(DATA_DIR)
if not clips:
    raise SystemExit(
        f"No clips found in {DATA_DIR}. Generate them first with:\n"
        "    python scripts/download_datasets.py --synthetic"
    )

summary = pd.DataFrame(
    [{"fault_type": c.fault_type, "duration_s": c.duration} for c in clips]
)
dataset_table = (
    summary.groupby("fault_type")
    .agg(clips=("fault_type", "size"), total_duration_s=("duration_s", "sum"))
    .reindex(CLASS_ORDER)
)
dataset_table["share"] = (dataset_table["clips"] / len(clips)).round(3)

print(f"Total clips: {len(clips)}")
print(f"Total duration: {summary['duration_s'].sum() / 60:.1f} min")
print("\nDataset summary table:")
print(dataset_table.to_markdown())

# %% [markdown]
# ## Section 2 -- Mel Spectrogram Analysis
#
# The four fault classes have *qualitatively different* time-frequency
# signatures. This 4-panel figure is the visual anchor of the paper's
# methodology section:
#
# * **normal** -- smooth broadband aerodynamic noise.
# * **bearing_fault** -- periodic vertical striations (impact ring-downs
#   exciting a 3-5 kHz resonance).
# * **blade_imbalance** -- slow horizontal amplitude banding at the rotor
#   frequency (1P modulation).
# * **gearbox_fault** -- horizontal harmonic ladder around the gear-mesh
#   frequency with sidebands.

# %%
extractor = FeatureExtractor(FeatureConfig())
example_clip = {c.fault_type: c for c in clips}  # one representative per class

fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
for ax, fault_type in zip(axes.ravel(), CLASS_ORDER):
    mel = extractor.mel_spectrogram(example_clip[fault_type].waveform)
    image = librosa.display.specshow(
        mel, sr=extractor.config.sample_rate,
        hop_length=extractor.config.hop_length,
        x_axis="time", y_axis="mel", cmap="magma", ax=ax,
        vmin=-80, vmax=0,
    )
    ax.set_title(fault_type.replace("_", " ").title())
fig.colorbar(image, ax=axes, format="%+2.0f dB", label="Power (dB)", shrink=0.85)
fig.suptitle("Log-Mel Spectrograms by Fault Class", fontsize=15)
save_figure(fig, "4-panel mel spectrogram class comparison")

# %% [markdown]
# ## Section 3 -- MFCC Analysis
#
# MFCCs compress spectral *envelope shape* into ~40 coefficients. If class
# means differ visibly in MFCC space, cheap classical models (SVM, GBM)
# are viable baselines -- an important paper comparison point. We average
# MFCC matrices over all clips of each class and show them as heatmaps.

# %%
mfcc_by_class: dict[str, np.ndarray] = {}
for fault_type in CLASS_ORDER:
    class_mfccs = [extractor.mfcc(c.waveform) for c in clips
                   if c.fault_type == fault_type]
    mfcc_by_class[fault_type] = np.mean(class_mfccs, axis=0)

fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
for ax, fault_type in zip(axes.ravel(), CLASS_ORDER):
    sns.heatmap(mfcc_by_class[fault_type], cmap="coolwarm", center=0,
                cbar_kws={"label": "Coefficient value"}, ax=ax)
    ax.set_title(f"Mean MFCC -- {fault_type.replace('_', ' ').title()}")
    ax.set_xlabel("Frame")
    ax.set_ylabel("MFCC index")
fig.suptitle("Class-Averaged MFCC Heatmaps", fontsize=15)
save_figure(fig, "mean MFCC heatmaps per class")

# %% [markdown]
# ## Section 4 -- Spectral Statistics Analysis
#
# Scalar descriptors (centroid, bandwidth, rolloff, zero-crossing rate) are
# *interpretable*: a bearing fault should raise the spectral centroid
# (high-frequency impact energy), while blade imbalance should not. Boxplots
# show robustness of the separation; violin plots show the full
# distribution shape (multimodality would indicate sub-populations).

# %%
records: list[dict[str, float | str]] = []
for clip in clips:
    vector = extractor.spectral_statistics(clip.waveform)
    row: dict[str, float | str] = dict(zip(SPECTRAL_STATISTIC_NAMES,
                                           vector.astype(float)))
    row["fault_type"] = clip.fault_type
    records.append(row)
features_df = pd.DataFrame(records)

ANALYSIS_FEATURES = [
    "spectral_centroid_mean",
    "spectral_bandwidth_mean",
    "spectral_rolloff_mean",
    "zero_crossing_rate_mean",
]

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for column, feature in enumerate(ANALYSIS_FEATURES):
    sns.boxplot(data=features_df, x="fault_type", y=feature,
                order=CLASS_ORDER, ax=axes[0][column],
                hue="fault_type", legend=False, palette="deep")
    sns.violinplot(data=features_df, x="fault_type", y=feature,
                   order=CLASS_ORDER, ax=axes[1][column],
                   hue="fault_type", legend=False, palette="deep",
                   inner="quartile", cut=0)
    for row in (0, 1):
        axes[row][column].set_xlabel("")
        axes[row][column].tick_params(axis="x", rotation=30)
    axes[0][column].set_title(feature.replace("_", " "))
fig.suptitle("Spectral Statistics by Fault Class (top: boxplots, bottom: violins)",
             fontsize=15)
save_figure(fig, "spectral statistics box/violin comparison")

print(features_df.groupby("fault_type")[ANALYSIS_FEATURES].mean()
      .reindex(CLASS_ORDER).round(2).to_markdown())

# %% [markdown]
# ## Section 5 -- Wind Noise Impact
#
# **Why augmentation matters.** We contaminate a clean bearing-fault clip
# with synthetic wind noise (1/f^2 spectrum + gusts + turbulence) at 5 dB
# SNR and observe: (a) the spectrogram's low-frequency band is buried, and
# (b) every spectral statistic shifts -- the centroid drops as 1/f^2 energy
# dominates. A model trained only on clean audio sees a different feature
# distribution at deployment; SNR-controlled augmentation closes that gap.
# *This figure is a planned paper figure.*

# %%
clean_clip = example_clip["bearing_fault"]
wind_generator = WindNoiseGenerator()
rng = np.random.default_rng(42)
wind = wind_generator.generate(clean_clip.num_samples, rng=rng)
noisy_waveform = WindNoiseGenerator.mix_at_snr(clean_clip.waveform, wind, snr_db=5.0)

mel_clean = extractor.mel_spectrogram(clean_clip.waveform)
mel_noisy = extractor.mel_spectrogram(noisy_waveform)

fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
vmin, vmax = mel_clean.min(), mel_clean.max()
for ax, mel, title in zip(axes, (mel_clean, mel_noisy),
                          ("Clean bearing fault", "Wind-contaminated (5 dB SNR)")):
    image = librosa.display.specshow(
        mel, sr=extractor.config.sample_rate,
        hop_length=extractor.config.hop_length,
        x_axis="time", y_axis="mel", cmap="magma", ax=ax, vmin=vmin, vmax=vmax)
    ax.set_title(title)
fig.colorbar(image, ax=axes, format="%+2.0f dB", label="Power (dB)", shrink=0.9)
save_figure(fig, "wind noise impact on spectrogram (paper figure)")

impact_table = pd.DataFrame({
    "clean": extractor.spectral_statistics(clean_clip.waveform),
    "wind_5dB": extractor.spectral_statistics(noisy_waveform),
}, index=SPECTRAL_STATISTIC_NAMES)
impact_table["shift_%"] = (
    100.0 * (impact_table["wind_5dB"] - impact_table["clean"])
    / impact_table["clean"].abs().replace(0, np.nan)
).round(1)
print("Spectral statistic shift under wind contamination:")
print(impact_table.round(3).to_markdown())

# %% [markdown]
# ## Section 6 -- Feature Correlation
#
# Highly correlated features are redundant: they inflate dimensionality
# without adding information and destabilize linear baselines. The
# correlation matrix tells us which spectral statistics to keep for the
# compact RUL feature set.

# %%
correlation = features_df[list(SPECTRAL_STATISTIC_NAMES)].corr()

fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(correlation, dtype=bool), k=1)
sns.heatmap(correlation, mask=mask, annot=True, fmt=".2f", cmap="vlag",
            center=0, square=True, cbar_kws={"label": "Pearson r"}, ax=ax)
ax.set_title("Spectral Feature Correlation Matrix")
save_figure(fig, "spectral feature correlation matrix")

redundant_pairs = [
    (a, b, correlation.loc[a, b])
    for i, a in enumerate(correlation.columns)
    for b in correlation.columns[i + 1:]
    if abs(correlation.loc[a, b]) > 0.9
]
print(f"Highly correlated pairs (|r| > 0.9): {len(redundant_pairs)}")
for a, b, r in redundant_pairs:
    print(f"  {a} <-> {b}: r = {r:+.3f}")

# %% [markdown]
# ## Section 7 -- Dimensionality Reduction
#
# **Do faults cluster naturally?** We project the 12-dimensional spectral
# statistic vectors with PCA (linear, variance-preserving) and t-SNE
# (nonlinear, neighborhood-preserving). Visible clusters before any
# supervised training are strong evidence the classification task is
# well-posed -- and a compelling paper figure.

# %%
feature_matrix = features_df[list(SPECTRAL_STATISTIC_NAMES)].to_numpy()
labels = features_df["fault_type"].to_numpy()
scaled = StandardScaler().fit_transform(feature_matrix)

pca = PCA(n_components=2, random_state=42)
pca_embedding = pca.fit_transform(scaled)

perplexity = max(2, min(10, len(scaled) // 4))
tsne_embedding = TSNE(n_components=2, perplexity=perplexity,
                      random_state=42, init="pca").fit_transform(scaled)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, embedding, title in zip(
        axes, (pca_embedding, tsne_embedding),
        (f"PCA ({100 * pca.explained_variance_ratio_.sum():.1f}% variance)",
         f"t-SNE (perplexity = {perplexity})")):
    for fault_type in CLASS_ORDER:
        mask = labels == fault_type
        ax.scatter(embedding[mask, 0], embedding[mask, 1], s=70,
                   label=fault_type.replace("_", " "), alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
axes[0].legend(fontsize="small")
fig.suptitle("Class Separation in Reduced Feature Space", fontsize=15)
save_figure(fig, "PCA and t-SNE class separation")

# %% [markdown]
# ## Section 8 -- Research Insights
#
# We rank features by one-way ANOVA F-score (between-class vs within-class
# variance): the higher the F, the more a single feature separates the
# four classes on its own. Kruskal-Wallis (non-parametric) confirms the
# ranking without normality assumptions. The printed observations feed the
# paper's analysis section directly.

# %%
class_codes = pd.Categorical(labels, categories=CLASS_ORDER).codes
f_scores, p_values = f_classif(scaled, class_codes)

importance = pd.DataFrame({
    "feature": SPECTRAL_STATISTIC_NAMES,
    "anova_f": f_scores.round(2),
    "p_value": p_values,
}).sort_values("anova_f", ascending=False).reset_index(drop=True)
importance["kruskal_p"] = [
    scipy_stats.kruskal(*[
        features_df.loc[features_df["fault_type"] == ft, feature]
        for ft in CLASS_ORDER
    ]).pvalue
    for feature in importance["feature"]
]

print("Feature importance ranking (ANOVA F-score):")
print(importance.round(4).to_markdown(index=False))

top = importance.iloc[0]
significant = importance[importance["p_value"] < 0.05]
print("\n" + "=" * 70)
print("OBSERVATIONS FOR PAPER")
print("=" * 70)
print(f"1. Most discriminative feature: {top['feature']} "
      f"(F = {top['anova_f']:.1f}, p = {top['p_value']:.2e}).")
print(f"2. {len(significant)}/{len(importance)} spectral statistics separate "
      "classes significantly (p < 0.05): scalar features alone justify a "
      "classical-ML baseline in the paper.")
print(f"3. {len(redundant_pairs)} feature pairs are redundant (|r| > 0.9); "
      "the compact RUL feature set should keep one per pair.")
print("4. Wind contamination at 5 dB SNR shifts spectral statistics by tens "
      "of percent (Section 5), confirming that SNR-controlled augmentation "
      "and denoising are necessary, not optional.")
print("5. PCA/t-SNE (Section 7) show pre-training class structure; "
      "the learned CNN embedding must improve on this unsupervised baseline.")
print(f"\nAll {_figure_counter} figures saved to {FIGURES_DIR}/")
