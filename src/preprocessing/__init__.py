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

__all__ = [
    "FAULT_LABELS",
    "AudioClip",
    "AudioConfig",
    "AudioLoader",
    "CWRULoader",
    "MIMIILoader",
]
