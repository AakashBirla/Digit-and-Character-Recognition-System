"""Small shared helpers: device selection, reproducible seeding and logging setup."""

from __future__ import annotations

import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def resolve_device(preference: str | torch.device | None = "auto") -> torch.device:
    """Turn a device preference into a concrete :class:`torch.device`.

    * ``"auto"`` / ``None`` -> CUDA when available, otherwise CPU.
    * ``"cpu"``             -> CPU.
    * ``"cuda"`` / ``"cuda:N"`` -> that GPU; raises ``ValueError`` with a helpful message
      if CUDA is not available or the index does not exist.
    """
    if preference is None:
        preference = "auto"
    if isinstance(preference, str):
        name = preference.strip().lower()
        if name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            device = torch.device(name)
        except RuntimeError as exc:
            raise ValueError(f"Unknown device {preference!r}. Use 'auto', 'cpu' or 'cuda'.") from exc
    elif isinstance(preference, torch.device):
        device = preference
    else:
        raise ValueError(f"Unsupported device preference {preference!r}. Use 'auto', 'cpu' or 'cuda'.")

    if device.type == "cpu":
        return device
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA was requested but is not available (torch.cuda.is_available() is False). "
                "Install a CUDA-enabled PyTorch build from https://pytorch.org or use --device cpu / auto."
            )
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device index {device.index} does not exist "
                f"(found {torch.cuda.device_count()} device(s))."
            )
        return device
    raise ValueError(f"Unsupported device {preference!r}. Use 'auto', 'cpu' or 'cuda'.")


def describe_device(device: str | torch.device) -> str:
    """Human-readable device description, e.g. ``"cuda (NVIDIA GeForce RTX 3050 Laptop GPU)"`` or ``"cpu"``."""
    device = torch.device(device) if isinstance(device, str) else device
    if device.type != "cuda":
        return str(device)
    try:
        index = device.index if device.index is not None else torch.cuda.current_device()
        return f"{device} ({torch.cuda.get_device_name(index)})"
    except (RuntimeError, AssertionError):  # CUDA not initialised / not available
        return str(device)


def set_seed(seed: int) -> None:
    """Seed Python's ``random``, NumPy and PyTorch (CPU and all GPUs) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.debug("Random seed set to %d", seed)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once with a consistent ``time | level | logger | message`` format.

    Calling it again only updates the level, so scripts and libraries can both call it safely.
    """
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
