"""The convolutional network used for both digits (10 classes) and characters (62 classes).

Architecture (input ``(B, 1, 28, 28)``)::

    Conv2d(1, 32, 3, padding=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2)    # 28x28 -> 14x14
    Conv2d(32, 64, 3, padding=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2)   # 14x14 -> 7x7
    Flatten -> Linear(64*7*7, 128) -> ReLU -> Dropout(0.3) -> Linear(128, num_classes)

The network returns raw logits; apply ``softmax`` for probabilities (``CrossEntropyLoss``
expects logits). When batch normalisation is enabled the convolutions have no bias, because
the BatchNorm shift makes a conv bias redundant.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch
from torch import nn

from src.config import IMAGE_SIZE

logger = logging.getLogger(__name__)

ARCHITECTURE_NAME = "DigitCNN"


class DigitCNN(nn.Module):
    """Two conv blocks + MLP head. Input ``(B, 1, 28, 28)`` -> logits ``(B, num_classes)``."""

    def __init__(
        self,
        num_classes: int = 10,
        conv_channels: tuple[int, int] = (32, 64),
        hidden_units: int = 128,
        dropout: float = 0.3,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        c1, c2 = (int(c) for c in conv_channels)
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        if c1 < 1 or c2 < 1 or hidden_units < 1:
            raise ValueError("conv_channels and hidden_units must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.num_classes = int(num_classes)
        self.conv_channels = (c1, c2)
        self.hidden_units = int(hidden_units)
        self.dropout = float(dropout)
        self.batch_norm = bool(batch_norm)

        def conv_block(in_ch: int, out_ch: int) -> list[nn.Module]:
            layers: list[nn.Module] = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not batch_norm)]
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers += [nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            return layers

        self.features = nn.Sequential(*conv_block(1, c1), *conv_block(c1, c2))
        pooled = IMAGE_SIZE // 4  # two 2x2 max-pools: 28 -> 14 -> 7
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c2 * pooled * pooled, self.hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_units, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape ``(B, num_classes)`` (no softmax)."""
        return self.classifier(self.features(x))

    def get_config(self) -> dict:
        """JSON-safe constructor arguments, stored in checkpoints to rebuild the model."""
        return {
            "architecture": ARCHITECTURE_NAME,
            "num_classes": self.num_classes,
            "conv_channels": list(self.conv_channels),
            "hidden_units": self.hidden_units,
            "dropout": self.dropout,
            "batch_norm": self.batch_norm,
        }

    @classmethod
    def from_config(cls, config: dict) -> DigitCNN:
        """Rebuild a model from :meth:`get_config` output (unknown keys are ignored with a warning)."""
        config = dict(config)
        architecture = config.pop("architecture", ARCHITECTURE_NAME)
        if architecture != ARCHITECTURE_NAME:
            raise ValueError(f"Unsupported architecture {architecture!r}; expected {ARCHITECTURE_NAME!r}.")
        known = {"num_classes", "conv_channels", "hidden_units", "dropout", "batch_norm"}
        unknown = sorted(set(config) - known)
        if unknown:
            logger.warning("Ignoring unknown model config keys: %s", unknown)
        kwargs = {k: v for k, v in config.items() if k in known}
        if "conv_channels" in kwargs:
            kwargs["conv_channels"] = tuple(kwargs["conv_channels"])
        return cls(**kwargs)

    def count_parameters(self) -> int:
        """Total number of parameters (all of them are trainable)."""
        return sum(p.numel() for p in self.parameters())


def architecture_summary(config: dict | DigitCNN) -> str:
    """One-line description, e.g.
    ``"Conv(32,3x3)+BN+ReLU+MaxPool → Conv(64,3x3)+BN+ReLU+MaxPool → FC(128)+ReLU+Dropout(0.3) → FC(62)"``.
    """
    if isinstance(config, DigitCNN):
        config = config.get_config()
    channels: Sequence[int] = config.get("conv_channels", (32, 64))
    bn = "+BN" if config.get("batch_norm", True) else ""
    convs = [f"Conv({int(c)},3x3){bn}+ReLU+MaxPool" for c in channels]
    hidden = int(config.get("hidden_units", 128))
    dropout = float(config.get("dropout", 0.3))
    num_classes = int(config.get("num_classes", 10))
    head = f"FC({hidden})+ReLU+Dropout({dropout:g}) → FC({num_classes})"
    return " → ".join([*convs, head])
