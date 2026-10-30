"""Saving and loading model checkpoints safely.

A checkpoint is a plain ``dict`` that contains only JSON-safe primitives (str, int, float,
bool, None, lists and dicts) plus ``model_state_dict`` (CPU tensors). That lets us load it with
``torch.load(..., weights_only=True)``, which refuses to unpickle arbitrary Python objects
and therefore cannot execute code hidden in a malicious file.

Payload keys::

    format_version, model_state_dict, model_config, num_classes, dataset, dataset_display_name,
    mode, class_labels, class_mapping_version, preprocessing, training_config, val_metrics,
    test_metrics, best_epoch, created_at, torch_version   (+ updated_at after metadata updates)

Writes are atomic: the file is written to a temporary name in the same directory and moved
into place with ``os.replace``, so a crash never leaves a half-written checkpoint behind.
"""

from __future__ import annotations

import contextlib
import dataclasses
import enum
import logging
import math
import os
import uuid
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from pathlib import Path, PurePath
from typing import Any

import numpy as np
import torch

from src.class_mapping import CLASS_MAPPING_VERSION
from src.config import get_spec
from src.models import DigitCNN
from src.utils import resolve_device

logger = logging.getLogger(__name__)

CHECKPOINT_FORMAT_VERSION = 1

REQUIRED_KEYS: tuple[str, ...] = (
    "format_version",
    "model_state_dict",
    "model_config",
    "num_classes",
    "dataset",
    "class_labels",
    "preprocessing",
)
OPTIONAL_KEYS: tuple[str, ...] = (
    "dataset_display_name",
    "mode",
    "class_mapping_version",
    "training_config",
    "val_metrics",
    "test_metrics",
    "best_epoch",
    "created_at",
    "torch_version",
)
PREPROCESSING_KEYS: tuple[str, ...] = ("mean", "std", "target_box", "centering", "stroke_width", "image_size")
_PROTECTED_KEYS = frozenset({"model_state_dict", "format_version"})


class CheckpointError(RuntimeError):
    """The checkpoint exists but cannot be used (corrupt, wrong format or inconsistent)."""


class CheckpointNotFoundError(CheckpointError):
    """The checkpoint file does not exist."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sanitize_metadata(obj: Any) -> Any:
    """Return a JSON-safe copy of ``obj``.

    NumPy scalars/arrays and tensors become Python numbers/lists, ``Path`` becomes ``str``,
    tuples and sets become lists, dataclasses become dicts, dict keys become strings and
    non-finite floats (NaN/inf) become ``None``. Unknown objects fall back to ``str(obj)``.
    """
    if obj is None or isinstance(obj, bool):
        return obj
    if isinstance(obj, str):
        return str(obj)  # exact str: subclasses (e.g. torch's TorchVersion) are rejected by weights_only loading
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, np.generic):
        return sanitize_metadata(obj.item())
    if isinstance(obj, np.ndarray):
        return sanitize_metadata(obj.tolist())
    if isinstance(obj, torch.Tensor):
        return sanitize_metadata(obj.detach().cpu().tolist())
    if isinstance(obj, PurePath):
        return str(obj)
    if isinstance(obj, (torch.device, torch.dtype)):
        return str(obj)
    if isinstance(obj, enum.Enum):
        return sanitize_metadata(obj.value)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return sanitize_metadata(dataclasses.asdict(obj))
    if isinstance(obj, Mapping):
        return {str(key): sanitize_metadata(value) for key, value in obj.items()}
    if isinstance(obj, (set, frozenset)):
        items = [sanitize_metadata(value) for value in obj]
        with contextlib.suppress(TypeError):
            items.sort()
        return items
    if isinstance(obj, (list, tuple)):
        return [sanitize_metadata(value) for value in obj]
    logger.warning("sanitize_metadata: converting unsupported %s to str", type(obj).__name__)
    return str(obj)


def _atomic_torch_save(payload: dict, path: Path) -> None:
    """``torch.save`` to a temp file in the same directory, fsync, then ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp_path, "xb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            tmp_path.unlink()
        raise


def save_checkpoint(
    path: Path | str,
    model: DigitCNN,
    *,
    dataset_key: str,
    labels: Sequence[str],
    preprocessing: dict,
    training_config: dict | None = None,
    val_metrics: dict | None = None,
    test_metrics: dict | None = None,
    best_epoch: int | None = None,
) -> None:
    """Atomically save ``model`` and its metadata to ``path`` (parent directories are created).

    Raises:
        ValueError: if the labels do not match the model's or the dataset's number of classes.
    """
    path = Path(path)
    model = getattr(model, "_orig_mod", model)  # unwrap torch.compile() wrappers
    spec = get_spec(dataset_key)
    labels = [str(label) for label in labels]
    if len(labels) != model.num_classes:
        raise ValueError(f"Got {len(labels)} labels for a model with {model.num_classes} outputs.")
    if len(labels) != spec.num_classes:
        raise ValueError(f"{spec.display_name} has {spec.num_classes} classes but {len(labels)} labels were given.")

    if hasattr(preprocessing, "to_metadata"):  # accept a PreprocessConfig directly
        preprocessing = preprocessing.to_metadata()
    missing = [key for key in PREPROCESSING_KEYS if key not in preprocessing]
    if missing:
        logger.warning("Preprocessing metadata is missing keys %s; inference will fall back to defaults", missing)

    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": {name: t.detach().cpu().clone() for name, t in model.state_dict().items()},
        "model_config": sanitize_metadata(model.get_config()),
        "num_classes": int(model.num_classes),
        "dataset": spec.key,
        "dataset_display_name": spec.display_name,
        "mode": spec.mode,
        "class_labels": sanitize_metadata(labels),
        "class_mapping_version": CLASS_MAPPING_VERSION,
        "preprocessing": sanitize_metadata(preprocessing),
        "training_config": sanitize_metadata(training_config),
        "val_metrics": sanitize_metadata(val_metrics),
        "test_metrics": sanitize_metadata(test_metrics),
        "best_epoch": None if best_epoch is None else int(best_epoch),
        "created_at": _utc_now(),
        "torch_version": str(torch.__version__),
    }
    _atomic_torch_save(payload, path)
    logger.info("Saved checkpoint to %s", path)


def load_checkpoint(path: Path | str, map_location: str | torch.device = "cpu") -> dict:
    """Load and validate a checkpoint dict with ``torch.load(..., weights_only=True)``.

    Raises:
        CheckpointNotFoundError: if the file does not exist.
        CheckpointError: if the file is corrupt, not a weights-only checkpoint, or has the wrong format.
    """
    path = Path(path)
    if not path.is_file():
        raise CheckpointNotFoundError(f"Checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:  # corrupt zip, truncated file, disallowed pickled objects, ...
        raise CheckpointError(
            f"Could not read checkpoint {path}: the file is corrupt, truncated or not a weights-only "
            f"checkpoint created by this project ({type(exc).__name__}: {exc})"
        ) from exc

    if not isinstance(checkpoint, dict):
        raise CheckpointError(f"Checkpoint {path} does not contain a dict (found {type(checkpoint).__name__}).")
    missing = [key for key in REQUIRED_KEYS if key not in checkpoint]
    if missing:
        raise CheckpointError(
            f"Checkpoint {path} is missing required keys {missing}; it was probably created by an older "
            "version of this project. Re-train the model to create a compatible checkpoint."
        )
    version = checkpoint["format_version"]
    if version != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointError(
            f"Checkpoint {path} has format_version {version!r}, expected {CHECKPOINT_FORMAT_VERSION}."
        )
    if not isinstance(checkpoint["model_state_dict"], Mapping):
        raise CheckpointError(f"Checkpoint {path} has an invalid model_state_dict.")
    try:
        get_spec(checkpoint["dataset"])
    except ValueError as exc:
        raise CheckpointError(f"Checkpoint {path} refers to an unknown dataset: {exc}") from exc
    for key in OPTIONAL_KEYS:
        checkpoint.setdefault(key, None)
    return checkpoint


def load_model_from_checkpoint(path: Path | str, device: str | torch.device = "cpu") -> tuple[DigitCNN, dict]:
    """Rebuild a :class:`DigitCNN` from a checkpoint and move it to ``device`` in eval mode.

    Returns ``(model, metadata)`` where ``metadata`` is the checkpoint dict without the weights.
    Raises :class:`CheckpointError` (or :class:`CheckpointNotFoundError`) on any problem.
    """
    checkpoint = load_checkpoint(path, map_location="cpu")
    labels = checkpoint["class_labels"]
    num_classes = checkpoint["num_classes"]
    if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
        raise CheckpointError(f"Checkpoint {path} has invalid class_labels.")
    if num_classes != len(labels):
        raise CheckpointError(
            f"Checkpoint {path} is inconsistent: num_classes={num_classes} but {len(labels)} class labels."
        )
    try:
        model = DigitCNN.from_config(checkpoint["model_config"])
    except (TypeError, ValueError) as exc:
        raise CheckpointError(f"Checkpoint {path} has an invalid model_config: {exc}") from exc
    if model.num_classes != num_classes:
        raise CheckpointError(
            f"Checkpoint {path} is inconsistent: model has {model.num_classes} outputs, metadata says {num_classes}."
        )
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except RuntimeError as exc:
        raise CheckpointError(f"Checkpoint {path} weights do not match the model architecture: {exc}") from exc

    model.to(resolve_device(device)).eval()
    metadata = {key: value for key, value in checkpoint.items() if key != "model_state_dict"}
    return model, metadata


def update_checkpoint_metadata(path: Path | str, **fields: Any) -> None:
    """Atomically update metadata fields (e.g. ``test_metrics=...``) of an existing checkpoint."""
    protected = _PROTECTED_KEYS.intersection(fields)
    if protected:
        raise ValueError(f"Refusing to overwrite protected checkpoint keys: {sorted(protected)}")
    path = Path(path)
    checkpoint = load_checkpoint(path, map_location="cpu")
    checkpoint.update({key: sanitize_metadata(value) for key, value in fields.items()})
    checkpoint["updated_at"] = _utc_now()
    _atomic_torch_save(checkpoint, path)
    logger.info("Updated checkpoint metadata %s in %s", sorted(fields), path)
