"""Model registry and prediction service used by the web app.

:class:`Predictor` loads one model per *mode* ("digits" -> MNIST, "characters" -> EMNIST
ByClass) once at startup and serves predictions:

* Loading never raises: a missing or broken checkpoint only marks that mode as ``"missing"`` /
  ``"error"`` with a user-safe message (including the command that trains the model), so the
  app still starts and the other mode keeps working.
* Each model is validated (dataset, class labels vs. ``src.class_mapping``, input size) and
  warmed up with one forward pass, so the first real request is not slow.
* Inference runs under ``torch.inference_mode()`` with the model in eval mode, and the forward
  pass is serialised with a lock, so the predictor is safe to share between Flask threads.
* :meth:`Predictor.health` / :meth:`ModelEntry.info` are JSON-serialisable and report test
  metrics only when the checkpoint actually contains measured ones - nothing is invented.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from src.checkpoints import CheckpointError, CheckpointNotFoundError, load_model_from_checkpoint, sanitize_metadata
from src.class_mapping import get_labels, label_category
from src.config import IMAGE_SIZE, MODES, PROJECT_ROOT, DatasetSpec, get_spec
from src.models import DigitCNN, architecture_summary
from src.preprocessing import ImageInput, PreprocessConfig, preprocess_image
from src.utils import describe_device, resolve_device

logger = logging.getLogger(__name__)

_MODE_HINT = " or ".join(f'"{mode}"' for mode in MODES)


# --------------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------------
class InvalidModeError(ValueError):
    """The requested mode is not one of ``MODES``. ``.reason`` is safe to show to users."""

    code = "invalid_mode"

    def __init__(self, mode: Any) -> None:
        self.mode = mode
        shown = repr(mode) if len(repr(mode)) <= 40 else repr(mode)[:37] + "..."
        self.reason = f"Unknown mode {shown}. Use {_MODE_HINT}."
        super().__init__(self.reason)


class ModelUnavailableError(RuntimeError):
    """The model for a mode is not loaded (missing or broken checkpoint). ``.reason`` is user-safe."""

    code = "model_unavailable"

    def __init__(self, mode: str, reason: str) -> None:
        self.mode = mode
        self.reason = reason
        super().__init__(reason)


class PredictionError(RuntimeError):
    """The model produced an unusable output (e.g. NaN). ``str(exc)`` is user-safe."""

    code = "prediction_error"


# --------------------------------------------------------------------------------------------
# Registry entry
# --------------------------------------------------------------------------------------------
def _display_path(path: Path) -> str:
    """Path relative to the project root; just the file name for paths outside it.

    These strings reach browser users (health, 503 messages), so server directory layouts are never shown.
    """
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except (ValueError, OSError):
        return path.name


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


@dataclass
class ModelEntry:
    """Everything the app knows about the model of one mode."""

    mode: str
    spec: DatasetSpec
    checkpoint_path: Path
    status: str = "missing"  # "loaded" | "missing" | "error"
    error: str | None = None  # user-safe message when status != "loaded"
    model: DigitCNN | None = None
    labels: tuple[str, ...] = ()
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    metadata: dict = field(default_factory=dict)  # checkpoint metadata (without weights)

    @property
    def loaded(self) -> bool:
        return self.status == "loaded" and self.model is not None

    def info(self) -> dict:
        """JSON-safe summary for ``/api/health`` and the model-information card."""
        meta = self.metadata or {}
        training_config = meta.get("training_config") if isinstance(meta.get("training_config"), Mapping) else {}
        val_metrics = meta.get("val_metrics") if isinstance(meta.get("val_metrics"), Mapping) else {}
        trained_on = training_config.get("device_name") or training_config.get("device")
        best_epoch = meta.get("best_epoch")
        created_at = meta.get("created_at")
        info = {
            "mode": self.mode,
            "dataset": self.spec.key,
            "display_name": self.spec.display_name,
            "num_classes": self.spec.num_classes,
            "status": self.status,
            "loaded": self.loaded,
            "error": self.error,
            "checkpoint": _display_path(self.checkpoint_path),
            "train_command": self.spec.train_command,
            "architecture": architecture_summary(self.model) if self.model is not None else None,
            "parameters": self.model.count_parameters() if self.model is not None else None,
            "test_metrics": _measured_test_metrics(meta.get("test_metrics")),
            "val_accuracy": _finite_number(val_metrics.get("accuracy")),
            "best_epoch": best_epoch if isinstance(best_epoch, int) and not isinstance(best_epoch, bool) else None,
            "trained_on": trained_on if isinstance(trained_on, str) and trained_on else None,
            "created_at": created_at if isinstance(created_at, str) else None,
            "labels": list(self.labels or get_labels(self.spec.key)),
        }
        return sanitize_metadata(info)


def _measured_test_metrics(metrics: Any) -> dict | None:
    """Scalar test metrics from the checkpoint, or ``None`` if no measured accuracy is recorded."""
    if not isinstance(metrics, Mapping) or _finite_number(metrics.get("accuracy")) is None:
        return None
    return {
        str(key): value
        for key, value in metrics.items()
        if value is None or isinstance(value, (str, bool, int)) or _finite_number(value) is not None
    }


# --------------------------------------------------------------------------------------------
# Predictor
# --------------------------------------------------------------------------------------------
class Predictor:
    """Loads both models once and serves predictions (thread-safe)."""

    def __init__(
        self,
        checkpoint_paths: Mapping[str, Path | str | None] | None = None,
        device: str | torch.device = "auto",
        debug_dir: Path | str | None = None,
        load: bool = True,
    ) -> None:
        """
        Args:
            checkpoint_paths: checkpoint per mode (``{"digits": ..., "characters": ...}``); missing
                modes use ``spec.default_checkpoint``.
            device: ``"auto"``, ``"cpu"`` or ``"cuda"`` (``ValueError`` if CUDA is unavailable).
            debug_dir: if set, every preprocessed input is saved as ``<debug_dir>/latest_<mode>.png``.
            load: load the models immediately (``load_all``).
        """
        self.device: torch.device = resolve_device(device)
        self.debug_dir = Path(debug_dir) if debug_dir is not None else None
        self._lock = threading.RLock()

        paths: dict[str, Path] = {mode: get_spec(mode).default_checkpoint for mode in MODES}
        for key, path in (checkpoint_paths or {}).items():
            spec = get_spec(key)  # accepts modes (and dataset keys); ValueError otherwise
            if path is not None:
                paths[spec.mode] = Path(path)
        self.checkpoint_paths = paths
        self._entries: dict[str, ModelEntry] = {
            mode: self._unloaded_entry(mode, "The model has not been loaded yet.") for mode in MODES
        }
        if load:
            self.load_all()

    # -- loading ---------------------------------------------------------------------------
    def _unloaded_entry(self, mode: str, error: str, status: str = "missing") -> ModelEntry:
        spec = get_spec(mode)
        return ModelEntry(
            mode=mode,
            spec=spec,
            checkpoint_path=self.checkpoint_paths[mode],
            status=status,
            error=error,
            labels=get_labels(spec.key),
            preprocess=PreprocessConfig.for_dataset(spec),
        )

    def load_all(self) -> None:
        """(Re)load every mode independently. Never raises; failures are recorded per entry."""
        entries = {}
        for mode in MODES:
            try:
                entries[mode] = self._load_entry(mode)
            except Exception:  # last line of defence: load_all must never raise
                logger.exception("Unexpected error while loading the %s model", mode)
                entries[mode] = self._unloaded_entry(
                    mode, f"The {mode} model could not be loaded because of an internal error; see the server log.",
                    status="error",
                )
        with self._lock:
            self._entries = entries
        for entry in entries.values():
            if entry.loaded:
                logger.info("%s model ready (%s, %d parameters) on %s", entry.spec.display_name,
                            _display_path(entry.checkpoint_path), entry.model.count_parameters(), self.device)
            else:
                logger.warning("%s model unavailable: %s", entry.spec.display_name, entry.error)

    def _load_entry(self, mode: str) -> ModelEntry:
        spec = get_spec(mode)
        path = self.checkpoint_paths[mode]
        shown = _display_path(path)
        retrain = f"Train it with: {spec.train_command}"

        if not path.is_file():
            return self._unloaded_entry(mode, f"Checkpoint not found: {shown}. {retrain}")
        try:
            model, metadata = load_model_from_checkpoint(path, device=self.device)
        except CheckpointNotFoundError:
            return self._unloaded_entry(mode, f"Checkpoint not found: {shown}. {retrain}")
        except CheckpointError as exc:
            logger.error("Could not load %s: %s", path, exc)
            return self._unloaded_entry(
                mode, f"Checkpoint {shown} could not be loaded (corrupt, incompatible or created by an older "
                      f"version of this project). Re-train it with: {spec.train_command}", status="error")

        problem = self._validate(model, metadata, spec)
        if problem:
            logger.error("Rejecting checkpoint %s: %s", path, problem)
            return self._unloaded_entry(mode, f"Checkpoint {shown} {problem}. {retrain}", status="error")
        try:
            preprocess = PreprocessConfig.from_metadata(metadata.get("preprocessing"), spec)
        except ValueError as exc:
            logger.error("Invalid preprocessing metadata in %s: %s", path, exc)
            return self._unloaded_entry(
                mode, f"Checkpoint {shown} has invalid preprocessing settings. {retrain}", status="error")
        if preprocess.image_size != IMAGE_SIZE:
            return self._unloaded_entry(
                mode, f"Checkpoint {shown} expects {preprocess.image_size}px inputs, not {IMAGE_SIZE}px. {retrain}",
                status="error")

        try:
            self._warm_up(model, spec)
        except Exception:  # CUDA errors, invalid output ...
            logger.exception("Warm-up forward pass failed for %s", path)
            return self._unloaded_entry(
                mode, f"The model in {shown} failed a test prediction on {self.device}; see the server log.",
                status="error")
        return ModelEntry(
            mode=mode, spec=spec, checkpoint_path=path, status="loaded", error=None, model=model,
            labels=tuple(metadata["class_labels"]), preprocess=preprocess, metadata=metadata,
        )

    @staticmethod
    def _validate(model: DigitCNN, metadata: dict, spec: DatasetSpec) -> str | None:
        """Return a user-safe description of why the checkpoint does not fit ``spec``, or ``None``."""
        dataset = metadata.get("dataset")
        if dataset != spec.key:
            found = get_spec(dataset).display_name if dataset else "an unknown dataset"
            return (f"holds a model trained on {found}, but the '{spec.mode}' mode needs one trained on "
                    f"{spec.display_name}")
        if model.num_classes != spec.num_classes:
            return f"has {model.num_classes} outputs instead of {spec.num_classes}"
        if tuple(metadata.get("class_labels") or ()) != get_labels(spec.key):
            return "has class labels that do not match the official class mapping"
        return None

    def _warm_up(self, model: DigitCNN, spec: DatasetSpec) -> None:
        """One forward pass (initialises CUDA kernels/allocator) and a sanity check of the output."""
        with self._lock, torch.inference_mode():
            logits = model(torch.zeros(1, 1, IMAGE_SIZE, IMAGE_SIZE, device=self.device))
        if tuple(logits.shape) != (1, spec.num_classes) or not torch.isfinite(logits).all():
            raise PredictionError(f"warm-up produced invalid logits of shape {tuple(logits.shape)}")

    # -- queries ---------------------------------------------------------------------------
    def get_entry(self, mode: Any) -> ModelEntry:
        """Entry for ``mode`` ("digits" / "characters", case-insensitive). ``InvalidModeError`` otherwise."""
        key = mode.strip().lower() if isinstance(mode, str) else None
        with self._lock:
            entry = self._entries.get(key) if key else None
        if entry is None:
            raise InvalidModeError(mode)
        return entry

    @property
    def entries(self) -> dict[str, ModelEntry]:
        with self._lock:
            return dict(self._entries)

    def health(self) -> dict:
        """JSON-safe service status: ``"ok"`` when every model is loaded, otherwise ``"degraded"``."""
        entries = self.entries
        return {
            "status": "ok" if all(entry.loaded for entry in entries.values()) else "degraded",
            "device": str(self.device),
            "device_name": describe_device(self.device),
            "models": {mode: entry.info() for mode, entry in entries.items()},
            "loaded_models": [entry.spec.display_name for entry in entries.values() if entry.loaded],
            "torch_version": str(torch.__version__),
        }

    # -- inference -------------------------------------------------------------------------
    def predict(self, image: ImageInput, mode: str, *, top_k: int = 5) -> dict:
        """Preprocess ``image`` for ``mode`` and classify it.

        Raises:
            InvalidModeError, ModelUnavailableError, PreprocessingError (and subclasses),
            PredictionError if the model output is not a valid probability distribution.
        """
        entry = self.get_entry(mode)
        if not entry.loaded:
            raise ModelUnavailableError(entry.mode, entry.error or f"The {entry.mode} model is not available.")
        started = time.perf_counter()

        debug_path = self.debug_dir / f"latest_{entry.mode}.png" if self.debug_dir is not None else None
        prepared = preprocess_image(image, entry.preprocess, debug_path=debug_path)

        with self._lock, torch.inference_mode():
            logits = entry.model(prepared.tensor.to(self.device))
        # float64 softmax on the CPU: probabilities sum to 1 to ~1e-15, independent of the device.
        probabilities = torch.softmax(logits.detach().to("cpu", torch.float64), dim=1)[0]
        if probabilities.numel() != len(entry.labels) or not torch.isfinite(probabilities).all():
            logger.error("Model for %s produced invalid output: %s", entry.mode, logits)
            raise PredictionError("The model produced an invalid output for this image.")

        values = probabilities.tolist()
        class_index = int(probabilities.argmax())
        label = entry.labels[class_index]
        rows = [
            {"class_index": index, "label": entry.labels[index], "probability": float(value)}
            for index, value in enumerate(values)
        ]
        k = max(1, min(int(top_k), len(rows)))
        top = sorted(rows, key=lambda row: (-row["probability"], row["class_index"]))[:k]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.debug("Predicted %r (%.3f) for %s in %.1f ms", label, values[class_index], entry.mode, elapsed_ms)

        return {
            "mode": entry.mode,
            "dataset": entry.spec.key,
            "prediction": label,
            "class_index": class_index,
            "confidence": float(values[class_index]),
            "category": label_category(label),
            "probabilities": rows,
            "top_k": [dict(row) for row in top],
            "preprocessed_image": prepared.to_data_url(scale=1),
            "processing_time_ms": round(elapsed_ms, 2),
            "device": str(self.device),
        }
