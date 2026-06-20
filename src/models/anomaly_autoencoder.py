#!/usr/bin/env python3
"""Anomaly-detection autoencoder for wind turbine acoustic monitoring.

This module implements :class:`AnomalyAutoencoder`, a convolutional
autoencoder (with an optional variational mode) that detects acoustic
anomalies by reconstruction error.  Unlike the supervised classifiers in this
package, it is trained on **normal clips only** (via
``WindTurbineDataset.normal_only()``) and learns to reconstruct healthy machine
sound.  At inference, clips that reconstruct poorly — i.e. that the model has
never learned to represent — are flagged as anomalous.

This is the right tool for the open-set problems supervised classifiers cannot
solve:

* **Unknown fault detection** — faults absent from the training labels still
  produce high reconstruction error.
* **Outlier / novelty detection** — sensor drift, new operating regimes, and
  previously-unseen machine states surface as anomalies.
* **Novel fault discovery** — clustering high-error clips reveals emerging
  failure modes for the research pipeline.
* **Continuous health monitoring** — the anomaly score is a single, calibrated
  health indicator that trends over a turbine's life.

It inherits the full operational contract from
:class:`~src.models.base_model.BaseModel` (checkpointing, ONNX export,
parameter accounting, device management, reproducible construction,
ExperimentTracker integration) and is registered as ``"anomaly_autoencoder"``.

Architecture
------------
::

    Input  (B, C, F, T)
      │  Convolutional encoder (stride-2 blocks, freq+time downsample)
      ▼
    (B, C_enc, F', T')
      │  AdaptiveAvgPool(4×4) + flatten        decouples latent from length
      ▼
    Latent code  (B, latent_dim)              [VAE: (mu, logvar) → reparam z]
      │  Linear → reshape to (C_enc, 4, 4)
      ▼
    Transposed-conv decoder (stride-2 blocks)
      │  F.interpolate(size=(F, T))            exact-shape reconstruction
      ▼
    Reconstruction  (B, C, F, T)

Two modes are selected by ``ModelConfig.extra["mode"]``:

* ``"ae"`` — deterministic autoencoder; loss is reconstruction MSE.
* ``"vae"`` — variational autoencoder; the encoder emits ``(mu, logvar)``, the
  latent is sampled with the reparameterisation trick, and the loss adds a
  ``beta``-weighted KL term (β-VAE).  For deterministic inference / ONNX export
  the mean is used directly (no sampling), so exported graphs have no random
  ops.

Anomaly scoring
---------------
The per-sample anomaly score is the mean reconstruction error (MSE by default,
MAE optionally).  Normal clips score low; faults score high.  Thresholds are
estimated from the score distribution of a held-out *normal* set, either as
``mean + k·std`` or as a high percentile.  :meth:`predict_anomaly` compares
scores to the threshold and :meth:`anomaly_confidence` maps the signed distance
to the threshold through a logistic squashing function for a calibrated
``[0, 1]`` confidence.

Dynamic length / ONNX / mixed precision
---------------------------------------
The latent bottleneck uses adaptive pooling, so the encoder accepts any
frequency/time size; the decoder upsamples from a fixed seed and a final
``interpolate`` matches the exact input shape — both ONNX-exportable (Resize
op).  No dtype is hardcoded in the forward path, so the model is
mixed-precision safe.

Configuration (via ``ModelConfig.extra``)
------------------------------------------
* ``mode`` (str): ``"ae"`` or ``"vae"`` (default ``"ae"``)
* ``latent_dim`` (int): latent code size (default 128)
* ``enc_channels`` (list[int]): encoder channels per block (default ``[32,64,128,128]``)
* ``seed_size`` (int): decoder seed spatial size (default 4)
* ``beta`` (float): KL weight for the VAE (default 1.0)
* ``score_metric`` (str): ``"mse"`` or ``"mae"`` (default ``"mse"``)

Usage::

    from src.models.anomaly_autoencoder import build_anomaly_autoencoder
    ae = build_anomaly_autoencoder(mode="vae", latent_dim=64)

    recon = ae.reconstruct(x)
    scores = ae.anomaly_score(x)                 # (B,)
    thr = ae.estimate_threshold(normal_scores)   # from a normal validation set
    flags = ae.predict_anomaly(x, threshold=thr) # (B,) bool

CLI::

    python src/models/anomaly_autoencoder.py --summary --mode vae
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

# ---------------------------------------------------------------------------
# Optional PyTorch import
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE: bool = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_AVAILABLE: bool = False

# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------
_MODULE_DIR: Final[Path] = Path(__file__).resolve().parent
_PROJECT_ROOT: Final[Path] = _MODULE_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.base_model import BaseModel, ModelConfig, register_model

logger = logging.getLogger("anomaly_autoencoder")

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

MODEL_NAME: Final[str] = "anomaly_autoencoder"

#: Supported autoencoder modes.
MODES: Final[tuple[str, ...]] = ("ae", "vae")

#: Supported anomaly-score metrics.
SCORE_METRICS: Final[tuple[str, ...]] = ("mse", "mae")

#: Supported threshold-estimation methods.
THRESHOLD_METHODS: Final[tuple[str, ...]] = ("std", "percentile", "iqr", "max")

DEFAULT_LATENT_DIM:   Final[int] = 128
DEFAULT_ENC_CHANNELS: Final[tuple[int, ...]] = (32, 64, 128, 128)
DEFAULT_SEED_SIZE:    Final[int] = 4
DEFAULT_BETA:         Final[float] = 1.0


# ---------------------------------------------------------------------------
# Threshold container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnomalyThreshold:
    """An estimated anomaly threshold and the statistics behind it.

    Attributes:
        value: The decision threshold; scores above it are anomalies.
        method: The estimation method used.
        mean: Mean of the reference (normal) scores.
        std: Standard deviation of the reference scores.
        n_samples: Number of reference scores used.
    """

    value:     float
    method:    str
    mean:      float
    std:       float
    n_samples: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary.

        Returns:
            Dictionary representation of the threshold.
        """
        return {
            "value": self.value, "method": self.method, "mean": self.mean,
            "std": self.std, "n_samples": self.n_samples,
        }


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class _EncoderBlock(nn.Module):
        """Strided convolution → BatchNorm → activation (downsampling).

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
        """

        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=3,
                stride=2, padding=1, bias=False,
            )
            self.bn = nn.BatchNorm2d(out_channels)
            self.act = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """Downsample the input by a factor of two.

            Args:
                x: Input tensor ``(B, C, H, W)``.

            Returns:
                Output tensor ``(B, C_out, H/2, W/2)``.
            """
            return self.act(self.bn(self.conv(x)))

    class _DecoderBlock(nn.Module):
        """Transposed convolution → BatchNorm → activation (upsampling).

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            final: When ``True``, omit norm/activation (raw reconstruction head).
        """

        def __init__(
            self, in_channels: int, out_channels: int, final: bool = False
        ) -> None:
            super().__init__()
            self.deconv = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=3,
                stride=2, padding=1, output_padding=1, bias=final,
            )
            self.final = final
            if not final:
                self.bn = nn.BatchNorm2d(out_channels)
                self.act = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """Upsample the input by a factor of two.

            Args:
                x: Input tensor ``(B, C, H, W)``.

            Returns:
                Output tensor ``(B, C_out, ~2H, ~2W)``.
            """
            x = self.deconv(x)
            if self.final:
                return x
            return self.act(self.bn(x))

else:  # pragma: no cover
    _EncoderBlock = None  # type: ignore[assignment,misc]
    _DecoderBlock = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# AnomalyAutoencoder
# ---------------------------------------------------------------------------


@register_model(MODEL_NAME)
class AnomalyAutoencoder(BaseModel):
    """Convolutional (optionally variational) autoencoder for anomaly detection.

    Trained on normal clips, it scores anomalies by reconstruction error.
    Inherits the full lifecycle contract from
    :class:`~src.models.base_model.BaseModel`.

    Configuration is read from ``ModelConfig.extra`` (see module docstring).

    Args:
        config: Model configuration. ``input_shape`` is the feature shape
            ``(F, T)`` and ``in_channels`` the channel count.

    Raises:
        RuntimeError: When instantiated without PyTorch installed.
        ValueError: When ``mode`` or ``score_metric`` is invalid.
    """

    def build_layers(self) -> None:
        """Construct the encoder, latent projections, and decoder.

        Raises:
            ValueError: When ``mode`` or ``score_metric`` is unsupported.
        """
        cfg = self.config
        extra = cfg.extra or {}

        self._mode: str = str(extra.get("mode", "ae"))
        if self._mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got '{self._mode}'")
        self._latent_dim: int = int(extra.get("latent_dim", DEFAULT_LATENT_DIM))
        enc_channels: list[int] = list(extra.get("enc_channels", DEFAULT_ENC_CHANNELS))
        self._seed_size: int = int(extra.get("seed_size", DEFAULT_SEED_SIZE))
        self._beta: float = float(extra.get("beta", DEFAULT_BETA))
        self._score_metric: str = str(extra.get("score_metric", "mse"))
        if self._score_metric not in SCORE_METRICS:
            raise ValueError(
                f"score_metric must be one of {SCORE_METRICS}, "
                f"got '{self._score_metric}'"
            )

        in_ch = cfg.in_channels
        self._enc_out_channels = enc_channels[-1]

        # ── Encoder ───────────────────────────────────────────────────────────
        enc_blocks: list[nn.Module] = []
        c = in_ch
        for out_ch in enc_channels:
            enc_blocks.append(_EncoderBlock(c, out_ch))
            c = out_ch
        self.encoder = nn.Sequential(*enc_blocks)

        # Adaptive pool decouples the latent from the input length.
        self.enc_pool = nn.AdaptiveAvgPool2d((self._seed_size, self._seed_size))
        flat = self._enc_out_channels * self._seed_size * self._seed_size

        # ── Latent projection ────────────────────────────────────────────────
        if self._mode == "vae":
            self.fc_mu = nn.Linear(flat, self._latent_dim)
            self.fc_logvar = nn.Linear(flat, self._latent_dim)
        else:
            self.fc_latent = nn.Linear(flat, self._latent_dim)
        self.fc_decode = nn.Linear(self._latent_dim, flat)

        # ── Decoder ───────────────────────────────────────────────────────────
        dec_blocks: list[nn.Module] = []
        rev = list(reversed(enc_channels))
        c = self._enc_out_channels
        for i, out_ch in enumerate(rev[1:]):
            dec_blocks.append(_DecoderBlock(c, out_ch))
            c = out_ch
        # One extra upsample, then the reconstruction head to in_channels.
        dec_blocks.append(_DecoderBlock(c, max(c // 2, in_ch)))
        c = max(c // 2, in_ch)
        self.decoder = nn.Sequential(*dec_blocks)
        self.recon_head = nn.Conv2d(c, in_ch, kernel_size=3, padding=1)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialise conv/deconv (Kaiming) and linear (Xavier) weights."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(
                    module.weight, a=0.2, mode="fan_out", nonlinearity="leaky_relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def _ensure_channel_dim(self, x: "torch.Tensor") -> "torch.Tensor":
        """Insert a channel dimension when the input is 3-D.

        Args:
            x: Input tensor, 3-D ``(B, F, T)`` or 4-D ``(B, C, F, T)``.

        Returns:
            4-D tensor ``(B, C, F, T)``.
        """
        if x.dim() == 3:
            return x.unsqueeze(1)
        return x

    def _reparameterize(
        self, mu: "torch.Tensor", logvar: "torch.Tensor"
    ) -> "torch.Tensor":
        """Sample the latent with the reparameterisation trick.

        During evaluation the mean is returned directly so inference and ONNX
        export are deterministic (no random ops in the exported graph).

        Args:
            mu: Latent means ``(B, latent_dim)``.
            logvar: Latent log-variances ``(B, latent_dim)``.

        Returns:
            Latent sample ``(B, latent_dim)``.
        """
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ------------------------------------------------------------------
    # Public: encode / decode / forward / reconstruct
    # ------------------------------------------------------------------

    def encode(
        self, x: "torch.Tensor"
    ) -> "torch.Tensor | tuple[torch.Tensor, torch.Tensor]":
        """Encode an input to its latent representation.

        Args:
            x: Input batch ``(B, C, F, T)`` or ``(B, F, T)``.

        Returns:
            For ``"ae"`` mode, the latent code ``(B, latent_dim)``.
            For ``"vae"`` mode, the tuple ``(mu, logvar)``.
        """
        x = self._ensure_channel_dim(x)
        h = self.encoder(x)
        h = self.enc_pool(h)
        h = torch.flatten(h, 1)
        if self._mode == "vae":
            return self.fc_mu(h), self.fc_logvar(h)
        return self.fc_latent(h)

    def decode(
        self, z: "torch.Tensor", output_size: "tuple[int, int]"
    ) -> "torch.Tensor":
        """Decode a latent code back to a reconstruction.

        Args:
            z: Latent code ``(B, latent_dim)``.
            output_size: Target ``(F, T)`` spatial size to reconstruct exactly.

        Returns:
            Reconstruction ``(B, C, F, T)``.
        """
        b = z.shape[0]
        h = self.fc_decode(z)
        h = h.view(b, self._enc_out_channels, self._seed_size, self._seed_size)
        h = self.decoder(h)
        h = self.recon_head(h)
        # Exact-shape match for any input length (ONNX Resize op).
        return F.interpolate(
            h, size=output_size, mode="bilinear", align_corners=False
        )

    def forward(
        self, x: "torch.Tensor"
    ) -> "tuple[torch.Tensor, dict[str, torch.Tensor]]":
        """Run the full autoencoder forward pass.

        Args:
            x: Input batch ``(B, C, F, T)`` or ``(B, F, T)``.

        Returns:
            Tuple ``(reconstruction, info)`` where *info* carries the latent
            (and, for the VAE, ``mu``/``logvar``) for loss computation.
        """
        x4 = self._ensure_channel_dim(x)
        size = (x4.shape[-2], x4.shape[-1])
        if self._mode == "vae":
            mu, logvar = self.encode(x4)
            z = self._reparameterize(mu, logvar)
            recon = self.decode(z, size)
            return recon, {"z": z, "mu": mu, "logvar": logvar}
        z = self.encode(x4)
        recon = self.decode(z, size)
        return recon, {"z": z}

    def reconstruct(self, x: "torch.Tensor") -> "torch.Tensor":
        """Reconstruct the input, returning only the reconstruction tensor.

        Args:
            x: Input batch ``(B, C, F, T)`` or ``(B, F, T)``.

        Returns:
            Reconstruction ``(B, C, F, T)``.
        """
        recon, _ = self.forward(x)
        return recon

    # ------------------------------------------------------------------
    # Public: loss
    # ------------------------------------------------------------------

    def loss_function(
        self,
        x: "torch.Tensor",
        recon: "torch.Tensor",
        info: "dict[str, torch.Tensor]",
    ) -> "tuple[torch.Tensor, dict[str, float]]":
        """Compute the training loss (reconstruction + optional KL).

        Args:
            x: Original input ``(B, C, F, T)`` or ``(B, F, T)``.
            recon: Reconstruction ``(B, C, F, T)``.
            info: The info dict returned by :meth:`forward`.

        Returns:
            Tuple ``(total_loss, components)`` where *components* maps
            ``"recon"`` / ``"kl"`` / ``"total"`` to float values.
        """
        x4 = self._ensure_channel_dim(x)
        recon_loss = F.mse_loss(recon, x4, reduction="mean")
        if self._mode == "vae" and "mu" in info:
            mu, logvar = info["mu"], info["logvar"]
            kl = -0.5 * torch.mean(
                torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            )
            total = recon_loss + self._beta * kl
            return total, {
                "recon": float(recon_loss.detach()),
                "kl": float(kl.detach()),
                "total": float(total.detach()),
            }
        return recon_loss, {
            "recon": float(recon_loss.detach()),
            "kl": 0.0,
            "total": float(recon_loss.detach()),
        }

    # ------------------------------------------------------------------
    # Public: anomaly scoring
    # ------------------------------------------------------------------

    def anomaly_score(self, x: "torch.Tensor") -> "torch.Tensor":
        """Compute the per-sample anomaly score (reconstruction error).

        Args:
            x: Input batch ``(B, C, F, T)`` or ``(B, F, T)``.

        Returns:
            Anomaly scores ``(B,)``; higher means more anomalous.
        """
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                x4 = self._ensure_channel_dim(x)
                recon, _ = self.forward(x4)
                diff = recon - x4
                if self._score_metric == "mae":
                    score = diff.abs().flatten(1).mean(dim=1)
                else:
                    score = diff.pow(2).flatten(1).mean(dim=1)
        finally:
            if was_training:
                self.train()
        return score

    def reconstruction_error_map(self, x: "torch.Tensor") -> "torch.Tensor":
        """Return the per-pixel squared-error map for visualization.

        The error map localises *where* in the time-frequency plane a clip
        deviates from normal — directly usable as a diagnostic heatmap.

        Args:
            x: Input batch ``(B, C, F, T)`` or ``(B, F, T)``.

        Returns:
            Error map ``(B, F, T)`` averaged over channels.
        """
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                x4 = self._ensure_channel_dim(x)
                recon, _ = self.forward(x4)
                err = (recon - x4).pow(2).mean(dim=1)
        finally:
            if was_training:
                self.train()
        return err

    def predict_anomaly(
        self, x: "torch.Tensor", threshold: "float | AnomalyThreshold"
    ) -> "torch.Tensor":
        """Flag samples whose anomaly score exceeds the threshold.

        Args:
            x: Input batch ``(B, C, F, T)`` or ``(B, F, T)``.
            threshold: A scalar threshold or an :class:`AnomalyThreshold`.

        Returns:
            Boolean tensor ``(B,)``; ``True`` marks an anomaly.
        """
        thr = threshold.value if isinstance(threshold, AnomalyThreshold) else float(threshold)
        return self.anomaly_score(x) > thr

    def anomaly_confidence(
        self, x: "torch.Tensor", threshold: "float | AnomalyThreshold"
    ) -> "torch.Tensor":
        """Return a calibrated ``[0, 1]`` confidence that each sample is anomalous.

        The signed distance between the score and the threshold (scaled by the
        reference spread when available) is squashed through a logistic
        function: scores far above threshold approach 1, far below approach 0.

        Args:
            x: Input batch ``(B, C, F, T)`` or ``(B, F, T)``.
            threshold: A scalar threshold or an :class:`AnomalyThreshold`.

        Returns:
            Confidence scores ``(B,)`` in ``[0, 1]``.
        """
        if isinstance(threshold, AnomalyThreshold):
            thr = threshold.value
            scale = threshold.std if threshold.std > 1e-12 else 1.0
        else:
            thr = float(threshold)
            scale = abs(thr) if abs(thr) > 1e-12 else 1.0
        scores = self.anomaly_score(x)
        return torch.sigmoid((scores - thr) / scale)

    # ------------------------------------------------------------------
    # Public: threshold estimation
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_threshold(
        normal_scores: "Any",
        *,
        method: str = "std",
        k: float = 3.0,
        percentile: float = 95.0,
    ) -> "AnomalyThreshold":
        """Estimate an anomaly threshold from normal-only reference scores.

        Args:
            normal_scores: 1-D array/tensor/sequence of anomaly scores computed
                on a held-out *normal* set.
            method: ``"std"`` (mean + k·std), ``"percentile"`` (the *percentile*
                of the scores), ``"iqr"`` (Q3 + 1.5·IQR), or ``"max"`` (the
                maximum normal score).
            k: Multiplier for the ``"std"`` method.
            percentile: Percentile for the ``"percentile"`` method.

        Returns:
            An :class:`AnomalyThreshold`.

        Raises:
            ValueError: When *method* is unsupported or *normal_scores* is empty.
        """
        if method not in THRESHOLD_METHODS:
            raise ValueError(
                f"method must be one of {THRESHOLD_METHODS}, got '{method}'"
            )
        scores = np.asarray(_to_numpy(normal_scores), dtype=np.float64).ravel()
        if scores.size == 0:
            raise ValueError("normal_scores must be non-empty")

        mean = float(scores.mean())
        std = float(scores.std())

        if method == "std":
            value = mean + k * std
        elif method == "percentile":
            value = float(np.percentile(scores, percentile))
        elif method == "iqr":
            q1, q3 = np.percentile(scores, [25, 75])
            value = float(q3 + 1.5 * (q3 - q1))
        else:  # max
            value = float(scores.max())

        return AnomalyThreshold(
            value=value, method=method, mean=mean, std=std,
            n_samples=int(scores.size),
        )

    # ------------------------------------------------------------------
    # Public: feature embedding
    # ------------------------------------------------------------------

    def extract_features(self, x: "torch.Tensor") -> "torch.Tensor":
        """Return the latent embedding (the bottleneck code).

        For the VAE this returns the mean ``mu`` (the deterministic latent),
        which is the natural embedding for clustering, novelty discovery, and
        the digital-twin similarity search on the roadmap.

        Args:
            x: Input batch ``(B, C, F, T)`` or ``(B, F, T)``.

        Returns:
            Latent embedding ``(B, latent_dim)``.
        """
        out = self.encode(x)
        if self._mode == "vae":
            return out[0]  # mu
        return out

    @property
    def feature_dim(self) -> int:
        """Dimensionality of the latent embedding.

        Returns:
            The configured latent dimension.
        """
        return self._latent_dim

    @property
    def mode(self) -> str:
        """The autoencoder mode (``"ae"`` or ``"vae"``).

        Returns:
            The configured mode string.
        """
        return self._mode

    # ------------------------------------------------------------------
    # Overrides for autoencoder semantics
    # ------------------------------------------------------------------

    def export_onnx(  # type: ignore[override]
        self,
        path: "str | Path",
        *,
        opset: int = 17,
        dynamic_batch: bool = True,
        input_names: "list[str] | None" = None,
        output_names: "list[str] | None" = None,
        experiment_tracker: Any = None,
    ) -> "Path":
        """Export a reconstruction-only ONNX graph.

        The training ``forward`` returns ``(reconstruction, info)``; for
        deployment only the reconstruction is needed, so this override wraps the
        model so the exported graph has a single tensor output.  The time axis
        is marked dynamic in addition to the batch axis.

        Args:
            path: Destination ``.onnx`` file path.
            opset: ONNX opset version.
            dynamic_batch: Mark the batch axis as dynamic.
            input_names: Graph input names (default ``["input"]``).
            output_names: Graph output names (default ``["reconstruction"]``).
            experiment_tracker: Optional tracker; the file is logged when set.

        Returns:
            The resolved :class:`Path` written.

        Raises:
            RuntimeError: When PyTorch is unavailable.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("export_onnx requires PyTorch")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        in_names = input_names or ["input"]
        out_names = output_names or ["reconstruction"]

        dynamic_axes: dict[str, dict[int, str]] | None = None
        if dynamic_batch:
            dynamic_axes = {
                in_names[0]:  {0: "batch", 2: "freq", 3: "time"},
                out_names[0]: {0: "batch", 2: "freq", 3: "time"},
            }

        parent = self

        class _ReconWrapper(nn.Module):
            """Wraps the autoencoder to output only the reconstruction."""

            def __init__(self) -> None:
                super().__init__()
                self.model = parent

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                recon, _ = self.model(x)
                return recon

        wrapper = _ReconWrapper()
        was_training = self.training
        wrapper.eval()
        dummy = torch.randn(1, *self.config.batched_input_shape, device=self.device)
        try:
            torch.onnx.export(
                wrapper, dummy, str(path),
                export_params=True, opset_version=opset,
                do_constant_folding=True,
                input_names=in_names, output_names=out_names,
                dynamic_axes=dynamic_axes,
            )
        finally:
            if was_training:
                self.train()

        logger.info("Exported anomaly autoencoder to ONNX -> %s", path)
        if experiment_tracker is not None:
            try:
                experiment_tracker.log_artifact(
                    str(path), description="ONNX autoencoder",
                    artifact_type="onnx",
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Tracker log_artifact failed: %s", exc)
        return path

    def predict(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[override]
        """Disabled for the autoencoder; use :meth:`predict_anomaly` instead.

        Args:
            x: Unused.

        Raises:
            NotImplementedError: Always — the autoencoder is unsupervised.
        """
        raise NotImplementedError(
            "AnomalyAutoencoder is unsupervised; use predict_anomaly(x, threshold) "
            "or anomaly_score(x) instead of predict()."
        )

    def predict_proba(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[override]
        """Disabled for the autoencoder; use :meth:`anomaly_confidence` instead.

        Args:
            x: Unused.

        Raises:
            NotImplementedError: Always — the autoencoder is unsupervised.
        """
        raise NotImplementedError(
            "AnomalyAutoencoder is unsupervised; use anomaly_confidence(x, threshold) "
            "instead of predict_proba()."
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_numpy(values: Any) -> "Any":
    """Convert a tensor / sequence / array to a NumPy array.

    Args:
        values: A torch tensor, NumPy array, or Python sequence.

    Returns:
        A NumPy array.
    """
    if _TORCH_AVAILABLE and isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


# ---------------------------------------------------------------------------
# Convenience constructor
# ---------------------------------------------------------------------------


def build_anomaly_autoencoder(
    input_shape: tuple[int, ...] = (128, 431),
    *,
    in_channels: int = 1,
    mode: str = "ae",
    latent_dim: int = DEFAULT_LATENT_DIM,
    enc_channels: tuple[int, ...] = DEFAULT_ENC_CHANNELS,
    seed_size: int = DEFAULT_SEED_SIZE,
    beta: float = DEFAULT_BETA,
    score_metric: str = "mse",
    dropout: float = 0.0,
    random_seed: int = 42,
    num_classes: int = 5,
) -> "AnomalyAutoencoder":
    """Construct an :class:`AnomalyAutoencoder` with explicit hyperparameters.

    Args:
        input_shape: Feature shape ``(F, T)`` (or ``(C, F, T)``).
        in_channels: Input channels (1 for mel/MFCC/CQT, 3 for hybrid stacks).
        mode: ``"ae"`` (deterministic) or ``"vae"`` (variational).
        latent_dim: Latent code size (configurable bottleneck).
        enc_channels: Encoder channels per block.
        seed_size: Decoder seed spatial size.
        beta: KL weight for the VAE (β-VAE).
        score_metric: ``"mse"`` or ``"mae"`` for the anomaly score.
        dropout: Unused by the AE head but kept for ``ModelConfig`` parity.
        random_seed: Seed for deterministic construction.
        num_classes: Retained for ``ModelConfig`` compatibility (unused by the
            autoencoder, which is unsupervised).

    Returns:
        A constructed :class:`AnomalyAutoencoder`.

    Raises:
        ValueError: When ``mode`` or ``score_metric`` is invalid.
        RuntimeError: When PyTorch is unavailable.
    """
    config = ModelConfig(
        model_name=MODEL_NAME,
        num_classes=num_classes,
        input_shape=input_shape,
        in_channels=in_channels,
        dropout=dropout,
        random_seed=random_seed,
        extra={
            "mode": mode,
            "latent_dim": latent_dim,
            "enc_channels": list(enc_channels),
            "seed_size": seed_size,
            "beta": beta,
            "score_metric": score_metric,
        },
    )
    return AnomalyAutoencoder(config)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: print summary and run a reconstruction smoke test.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code 0 on success, 1 on failure.
    """
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Anomaly autoencoder")
    parser.add_argument("--mode", default="ae", choices=list(MODES))
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--onnx", type=Path, default=None)
    args = parser.parse_args(argv)

    if not _TORCH_AVAILABLE:
        logger.error("PyTorch not installed; cannot instantiate model.")
        return 1

    model = build_anomaly_autoencoder(
        mode=args.mode, latent_dim=args.latent_dim, in_channels=args.in_channels,
    )
    if args.summary or True:
        model.print_summary()

    dummy = torch.randn(2, *model.config.batched_input_shape)
    recon = model.reconstruct(dummy)
    scores = model.anomaly_score(dummy)
    logger.info("Reconstruct OK: %s -> %s | scores %s | latent %d | mode %s",
                tuple(dummy.shape), tuple(recon.shape),
                tuple(scores.shape), model.feature_dim, model.mode)
    assert recon.shape == dummy.shape, "reconstruction must match input shape"

    if args.onnx is not None:
        model.export_onnx(args.onnx)
        logger.info("Exported ONNX to %s", args.onnx)

    return 0


if __name__ == "__main__":
    sys.exit(main())