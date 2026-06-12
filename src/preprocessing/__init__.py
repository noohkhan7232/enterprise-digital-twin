"""Signal preprocessing for wind turbine acoustic monitoring.

This package contains denoising, segmentation, feature extraction, and
synthetic wind-noise augmentation (``WindNoiseGenerator``) used to prepare
raw audio for anomaly detection and fault classification.
"""

from src.preprocessing.audio_loader import (
    FAULT_LABELS,
    AudioClip,
    AudioConfig,
    AudioLoader,
    CWRULoader,
    MIMIILoader,
)
from src.preprocessing.augmentation import (
    AugmentationConfig,
    AugmentationPipeline,
    WindNoiseGenerator,
)
from src.preprocessing.denoiser import (
    SUPPORTED_METHODS,
    SUPPORTED_WAVELETS,
    Denoiser,
    DenoiserConfig,
)
from src.preprocessing.feature_extractor import (
    SPECTRAL_STATISTIC_NAMES,
    FeatureConfig,
    FeatureExtractor,
)

__all__ = [
    "FAULT_LABELS",
    "SPECTRAL_STATISTIC_NAMES",
    "SUPPORTED_METHODS",
    "SUPPORTED_WAVELETS",
    "AudioClip",
    "AudioConfig",
    "AudioLoader",
    "AugmentationConfig",
    "AugmentationPipeline",
    "CWRULoader",
    "Denoiser",
    "DenoiserConfig",
    "FeatureConfig",
    "FeatureExtractor",
    "MIMIILoader",
    "WindNoiseGenerator",
]
