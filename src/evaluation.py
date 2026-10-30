"""Evaluation on the official test split, report artefacts and a canvas round-trip check.

The main entry point is :func:`evaluate_checkpoint`:

1. rebuild the model from a checkpoint (:func:`src.checkpoints.load_model_from_checkpoint`);
2. load the dataset's **official test split**, which training never touches (validation is a
   seeded subset of the official *train* split);
3. collect predictions and compute accuracy, loss, macro/weighted precision-recall-F1, per-class
   metrics and the confusion matrix (:func:`compute_metrics`);
4. write ``test_metrics.json``, ``classification_report.txt``, ``per_class_metrics.csv``,
   ``confusion_matrix.csv`` and ``confusion_matrix.png`` (:func:`save_evaluation_artifacts`);
5. store a compact ``test_metrics`` dict in the checkpoint, but only after a full test-split
   evaluation, because a subset's accuracy must never be reported as "test accuracy";
6. optionally run the *canvas round-trip* (:func:`canvas_roundtrip_accuracy`). Test glyphs are
   re-drawn as crisp strokes on a large black canvas, the way the web app's canvas looks. They are
   pushed through the same ``src.preprocessing`` pipeline the app uses and classified again. A
   large gap between round-trip and direct accuracy means preprocessing does not reproduce the
   training distribution.

``src.preprocessing`` is imported lazily, only by the round-trip code, so metrics and artefacts
work without it.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from torch import nn

from src.checkpoints import (
    CheckpointError,
    load_model_from_checkpoint,
    sanitize_metadata,
    update_checkpoint_metadata,
)
from src.class_mapping import get_labels, label_category
from src.config import DATA_DIR, PROJECT_ROOT, RESULTS_DIR, get_spec
from src.datasets import TensorBatchLoader, load_split, normalize_batch
from src.utils import describe_device, resolve_device

logger = logging.getLogger(__name__)

TEST_SPLIT_NAME = "official_test"
TOP_CONFUSIONS = 20
COMPACT_METRIC_KEYS: tuple[str, ...] = (
    "accuracy",
    "loss",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "num_samples",
)
ROUNDTRIP_GLYPH_FRACTION = (0.35, 0.75)  # glyph's longest side as a fraction of the simulated canvas

# Chart styling: a single-hue sequential blue ramp (surface -> dark) and recessive ink colours.
_HEATMAP_RAMP = ("#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b")
_INK = "#0b0b0b"
_MUTED_INK = "#52514e"
_GRID = "#d9d8d3"


# --------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------
def utc_now() -> str:
    """Current time as an ISO-8601 UTC string (seconds precision)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def display_path(path: Path | str) -> str:
    """Path relative to the project root when possible (shorter, and leaks no home directory)."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def write_json(path: Path | str, data: Any) -> Path:
    """Write ``data`` as pretty JSON. NumPy values, paths and NaN are made JSON-safe first."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(sanitize_metadata(data), indent=2, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def _seeded_subset_indices(total: int, max_samples: int | None, seed: int) -> torch.Tensor | None:
    """Sorted indices of a seeded random subset, or ``None`` when every sample is used."""
    if max_samples is None or max_samples >= total:
        return None
    if max_samples < 1:
        raise ValueError(f"max_test_samples must be >= 1, got {max_samples}")
    permutation = torch.randperm(total, generator=torch.Generator().manual_seed(seed))
    return permutation[:max_samples].sort().values


# --------------------------------------------------------------------------------------------
# Predictions and metrics
# --------------------------------------------------------------------------------------------
@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: str | torch.device,
) -> dict:
    """Run ``model`` over ``loader``.

    Returns ``{"y_true", "y_pred", "loss", "num_samples"}``: the two label arrays are int64 NumPy
    arrays and ``loss`` is the mean cross-entropy. Probabilities are not kept, to save memory.
    """
    device = resolve_device(device)
    model.eval()
    total_loss = torch.zeros((), dtype=torch.float64, device=device)
    true_batches: list[torch.Tensor] = []
    pred_batches: list[torch.Tensor] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        total_loss += F.cross_entropy(logits, y, reduction="sum").double()
        true_batches.append(y)
        pred_batches.append(logits.argmax(dim=1))
    if not true_batches:
        raise ValueError("The evaluation loader yielded no batches.")
    y_true = torch.cat(true_batches).cpu().numpy().astype(np.int64)
    y_pred = torch.cat(pred_batches).cpu().numpy().astype(np.int64)
    num_samples = int(y_true.shape[0])
    mean_loss = float(total_loss.item()) / num_samples
    return {"y_true": y_true, "y_pred": y_pred, "loss": mean_loss, "num_samples": num_samples}


def compute_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    labels: Sequence[str],
    loss: float | None = None,
) -> dict:
    """Compute classification metrics for a fixed label set (all values are JSON-safe).

    Every metric uses ``labels=range(len(labels))``, so the confusion matrix and the per-class
    table always have one row per class, even if a class is absent. Undefined precision or
    recall (a class never predicted, or with no samples) counts as 0 (``zero_division=0``).
    """
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    labels = [str(label) for label in labels]
    num_classes = len(labels)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true has {y_true.size} entries but y_pred has {y_pred.size}.")
    if y_true.size == 0:
        raise ValueError("Cannot compute metrics without samples.")
    for name, values in (("y_true", y_true), ("y_pred", y_pred)):
        if values.min() < 0 or values.max() >= num_classes:
            raise ValueError(f"{name} contains class indices outside [0, {num_classes}).")

    class_ids = list(range(num_classes))
    common = {"labels": class_ids, "zero_division": 0}
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, **common)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", **common)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", **common
    )
    matrix = confusion_matrix(y_true, y_pred, labels=class_ids)
    predicted_count = matrix.sum(axis=0)

    per_class = [
        {
            "class_index": index,
            "label": labels[index],
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
            "predicted_count": int(predicted_count[index]),
        }
        for index in class_ids
    ]

    # Most frequent mistakes: largest off-diagonal cells (ties broken by class index for stability).
    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0)
    cells = sorted(
        ((int(off_diagonal[i, j]), int(i), int(j)) for i, j in zip(*np.nonzero(off_diagonal))),
        key=lambda cell: (-cell[0], cell[1], cell[2]),
    )
    top_confusions = [{"true": labels[i], "pred": labels[j], "count": count} for count, i, j in cells[:TOP_CONFUSIONS]]

    # Case-insensitive accuracy is informative for EMNIST ByClass, where letters like c/C, o/O and
    # s/S are written almost identically. Not applicable when no two labels differ only by case.
    folded = [label.lower() for label in labels]
    case_insensitive_accuracy = None
    if len(set(folded)) < num_classes:
        same_letter = np.array([[a == b for b in folded] for a in folded])
        case_insensitive_accuracy = float(matrix[same_letter].sum() / y_true.size)

    return {
        "num_samples": int(y_true.size),
        "num_classes": num_classes,
        "accuracy": float(np.trace(matrix) / y_true.size),
        "loss": None if loss is None else float(loss),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "case_insensitive_accuracy": case_insensitive_accuracy,
        "classes_never_predicted": [labels[i] for i in class_ids if predicted_count[i] == 0],
        "classes_without_support": [labels[i] for i in class_ids if support[i] == 0],
        "top_confusions": top_confusions,
        "per_class": per_class,
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def compact_test_metrics(metrics: dict, *, evaluated_at: str | None = None) -> dict:
    """The small ``test_metrics`` dict that is stored in a checkpoint (and shown by the web app)."""
    compact = {key: metrics.get(key) for key in COMPACT_METRIC_KEYS}
    compact["evaluated_at"] = evaluated_at or metrics.get("evaluated_at") or utc_now()
    compact["split"] = TEST_SPLIT_NAME
    return compact


# --------------------------------------------------------------------------------------------
# Artefacts
# --------------------------------------------------------------------------------------------
def _pairs_from_confusion_matrix(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Expand a confusion matrix back into (y_true, y_pred) arrays.

    The matrix fully determines every per-class metric, so the sklearn report built from these
    arrays matches one built from the original predictions exactly.
    """
    true_idx, pred_idx = np.nonzero(matrix)
    counts = matrix[true_idx, pred_idx]
    return np.repeat(true_idx, counts), np.repeat(pred_idx, counts)


def _label_category_or_none(label: str) -> str | None:
    try:
        return label_category(label)
    except ValueError:
        return None


def plot_confusion_matrix(matrix: np.ndarray, labels: Sequence[str], path: Path | str, *, title: str) -> Path:
    """Save a row-normalised confusion-matrix heatmap; every class label is shown on both axes.

    Each row is divided by its true-class count, so a cell is "fraction of class *row* predicted
    as *column*" and rare classes stay as visible as frequent ones. Matrices with up to 20 classes
    are annotated with raw counts. Thin lines separate digits, uppercase and lowercase letters.
    """
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.figure import Figure  # no pyplot: no global state, works headless (Agg)

    matrix = np.asarray(matrix, dtype=np.int64)
    labels = [str(label) for label in labels]
    n = len(labels)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalised = np.divide(matrix, row_totals, out=np.zeros(matrix.shape, dtype=float), where=row_totals > 0)

    large = n > 20
    side = min(max(0.26 * n + 2.5, 6.5), 19.0)
    fig = Figure(figsize=(side + 1.2, side), dpi=150 if large else 130)
    ax = fig.subplots()
    cmap = LinearSegmentedColormap.from_list("sequential_blue", _HEATMAP_RAMP)
    image = ax.imshow(normalised, cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    colorbar.set_label("Fraction of the true class", color=_MUTED_INK)
    colorbar.outline.set_visible(False)

    tick_font = 7 if large else 10
    ax.set_xticks(range(n), labels, fontsize=tick_font)
    ax.set_yticks(range(n), labels, fontsize=tick_font)
    ax.tick_params(length=0, colors=_INK)
    ax.set_xlabel("Predicted label", color=_INK)
    ax.set_ylabel("True label", color=_INK)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if not large:
        for i in range(n):
            for j in range(n):
                if matrix[i, j]:
                    colour = "white" if normalised[i, j] > 0.5 else _INK
                    ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center", fontsize=8, color=colour)

    categories = [_label_category_or_none(label) for label in labels]
    for index in range(1, n):
        if categories[index] != categories[index - 1]:
            ax.axhline(index - 0.5, color=_GRID, linewidth=0.8)
            ax.axvline(index - 0.5, color=_GRID, linewidth=0.8)

    accuracy = np.trace(matrix) / max(int(matrix.sum()), 1)
    ax.set_title(
        f"{title}\nRow-normalised confusion matrix · accuracy {accuracy:.2%}",
        fontsize=12 if large else 11,
        color=_INK,
    )
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    return path


def save_evaluation_artifacts(
    metrics: dict, labels: Sequence[str], out_dir: Path | str, *, title: str
) -> dict[str, str]:
    """Write the evaluation report files to ``out_dir`` and return ``{name: path}``.

    Files: ``test_metrics.json`` (everything, including the confusion matrix),
    ``classification_report.txt`` (sklearn, 4 digits), ``per_class_metrics.csv``,
    ``confusion_matrix.csv`` and ``confusion_matrix.png``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [str(label) for label in labels]
    num_classes = len(labels)
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    if matrix.shape != (num_classes, num_classes):
        raise ValueError(f"Confusion matrix has shape {matrix.shape}, expected ({num_classes}, {num_classes}).")
    paths: dict[str, str] = {}

    paths["test_metrics"] = str(write_json(out_dir / "test_metrics.json", metrics))

    y_true, y_pred = _pairs_from_confusion_matrix(matrix)
    report = classification_report(
        y_true, y_pred, labels=list(range(num_classes)), target_names=labels, digits=4, zero_division=0
    )
    header = [title, "=" * len(title), f"Samples:  {int(matrix.sum()):,}", f"Accuracy: {metrics['accuracy']:.4f}"]
    if metrics.get("loss") is not None:
        header.append(f"Loss:     {metrics['loss']:.4f}")
    if metrics.get("case_insensitive_accuracy") is not None:
        header.append(f"Case-insensitive accuracy: {metrics['case_insensitive_accuracy']:.4f}")
    never = metrics.get("classes_never_predicted") or []
    header.append(f"Classes never predicted: {', '.join(never) if never else 'none'}")
    report_path = out_dir / "classification_report.txt"
    report_path.write_text("\n".join(header) + "\n\n" + report, encoding="utf-8")
    paths["classification_report"] = str(report_path)

    per_class_path = out_dir / "per_class_metrics.csv"
    columns = ["class_index", "label", "precision", "recall", "f1", "support", "predicted_count"]
    with open(per_class_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in metrics["per_class"]:
            writer.writerow({key: row[key] for key in columns})
    paths["per_class_metrics"] = str(per_class_path)

    matrix_csv_path = out_dir / "confusion_matrix.csv"
    with open(matrix_csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true \\ predicted", *labels])
        for label, row in zip(labels, matrix.tolist()):
            writer.writerow([label, *row])
    paths["confusion_matrix_csv"] = str(matrix_csv_path)

    png_path = plot_confusion_matrix(matrix, labels, out_dir / "confusion_matrix.png", title=title)
    paths["confusion_matrix_png"] = str(png_path)
    logger.info("Saved evaluation artefacts to %s", out_dir)
    return paths


# --------------------------------------------------------------------------------------------
# Canvas round-trip
# --------------------------------------------------------------------------------------------
def simulate_canvas_drawing(
    img28: np.ndarray | torch.Tensor, rng: np.random.Generator, canvas_size: int = 400
) -> np.ndarray:
    """Re-draw a 28x28 glyph the way it might look on the web canvas.

    Returns a ``(canvas_size, canvas_size)`` uint8 image with white strokes on black. The glyph's
    ink is cropped and upscaled so its longest side is a random 35-75% of the canvas. It is then
    thresholded into crisp binary strokes and pasted at a random position.
    """
    glyph = img28.detach().cpu().numpy() if isinstance(img28, torch.Tensor) else np.asarray(img28)
    glyph = np.squeeze(glyph).astype(np.uint8)
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    rows, cols = np.nonzero(glyph)
    if rows.size == 0:
        return canvas  # an empty glyph stays an empty canvas

    crop = glyph[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
    height, width = crop.shape
    longest = rng.uniform(*ROUNDTRIP_GLYPH_FRACTION) * canvas_size
    scale = longest / max(height, width)
    new_w = int(np.clip(round(width * scale), 1, canvas_size))
    new_h = int(np.clip(round(height * scale), 1, canvas_size))
    enlarged = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    strokes = np.where(enlarged >= 128, 255, 0).astype(np.uint8)

    x0 = int(rng.integers(0, canvas_size - new_w + 1))
    y0 = int(rng.integers(0, canvas_size - new_h + 1))
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = strokes
    return canvas


@torch.no_grad()
def _predict_in_batches(model: nn.Module, x: torch.Tensor, device: torch.device, batch_size: int = 512) -> np.ndarray:
    """Arg-max predictions for normalised inputs ``x`` (B, 1, 28, 28) kept on the CPU."""
    model.eval()
    predictions = [
        model(x[start : start + batch_size].to(device, non_blocking=True)).argmax(dim=1).cpu()
        for start in range(0, x.shape[0], batch_size)
    ]
    return torch.cat(predictions).numpy() if predictions else np.zeros(0, dtype=np.int64)


def canvas_roundtrip_accuracy(
    model: nn.Module,
    images_uint8: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    preprocess_config: Any,
    device: str | torch.device,
    n: int,
    seed: int,
) -> dict:
    """Accuracy on ``n`` simulated canvas drawings, compared with direct accuracy on the same samples.

    Each sampled glyph goes through :func:`simulate_canvas_drawing`, then
    ``src.preprocessing.preprocess_image`` with ``preprocess_config``, then the model. A drawing
    that preprocessing rejects (e.g. as blank) counts as a wrong prediction.

    Returns ``{"samples", "accuracy", "direct_accuracy_same_samples", "preprocessing_errors", "seed"}``.
    """
    from src.preprocessing import PreprocessingError, preprocess_image  # lazy: only the round-trip needs it

    device = resolve_device(device)
    images = images_uint8.detach().cpu().numpy() if isinstance(images_uint8, torch.Tensor) else np.asarray(images_uint8)
    labels = targets.detach().cpu().numpy() if isinstance(targets, torch.Tensor) else np.asarray(targets)
    if images.ndim == 4:
        images = images[:, 0]
    count = min(int(n), images.shape[0])
    if count < 1:
        raise ValueError("canvas_roundtrip_accuracy needs n >= 1 and at least one image.")

    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(images.shape[0], size=count, replace=False))
    tensors: list[torch.Tensor] = []
    accepted: list[int] = []
    errors = 0
    for position, index in enumerate(chosen):
        canvas = simulate_canvas_drawing(images[index], rng)
        try:
            tensors.append(preprocess_image(canvas, preprocess_config).tensor)
            accepted.append(position)
        except PreprocessingError as exc:
            errors += 1
            logger.debug("Round-trip sample %d rejected by preprocessing: %s", int(index), exc)

    expected = labels[chosen].astype(np.int64)
    roundtrip_pred = np.full(count, -1, dtype=np.int64)  # -1 = rejected, never equals a label
    if tensors:
        roundtrip_pred[accepted] = _predict_in_batches(model, torch.cat(tensors).float(), device)

    direct_x = normalize_batch(
        torch.from_numpy(images[chosen]).unsqueeze(1), preprocess_config.mean, preprocess_config.std
    )
    direct_pred = _predict_in_batches(model, direct_x, device)
    return {
        "samples": count,
        "accuracy": float((roundtrip_pred == expected).mean()),
        "direct_accuracy_same_samples": float((direct_pred == expected).mean()),
        "preprocessing_errors": errors,
        "seed": int(seed),
    }


def _run_roundtrip(
    model: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    preprocessing_meta: dict | None,
    dataset_key: str,
    device: torch.device,
    n: int,
    seed: int,
) -> dict | None:
    """Build the checkpoint's PreprocessConfig and run the round-trip; ``None`` if preprocessing is unavailable."""
    try:
        from src.preprocessing import PreprocessConfig
    except ModuleNotFoundError as exc:
        if exc.name != "src.preprocessing":
            raise
        logger.warning("src.preprocessing is not available yet; skipping the canvas round-trip check.")
        return None
    config = PreprocessConfig.from_metadata(preprocessing_meta, dataset_key)
    start = time.perf_counter()
    result = canvas_roundtrip_accuracy(model, images, targets, config, device, n, seed)
    result["seconds"] = round(time.perf_counter() - start, 2)
    result["preprocessing"] = config.to_metadata()
    return result


# --------------------------------------------------------------------------------------------
# Checkpoint evaluation
# --------------------------------------------------------------------------------------------
def evaluate_checkpoint(
    checkpoint_path: Path | str,
    *,
    data_dir: Path | str = DATA_DIR,
    device: str | torch.device = "auto",
    results_dir: Path | str | None = None,
    batch_size: int = 1024,
    max_test_samples: int | None = None,
    update_checkpoint: bool = True,
    roundtrip_samples: int = 0,
    seed: int = 0,
) -> dict:
    """Evaluate a checkpoint on its dataset's official test split and save the report artefacts.

    ``max_test_samples`` evaluates a seeded subset (for quick checks). Test metrics are written into
    the checkpoint only when ``update_checkpoint`` is true *and* the whole test split was evaluated.
    Returns ``{"checkpoint", "dataset", "metrics", "test_metrics", "recorded_in_checkpoint",
    "full_test_split", "results_dir", "artifacts", "roundtrip"}``.
    """
    checkpoint_path = Path(checkpoint_path)
    device = resolve_device(device)
    model, meta = load_model_from_checkpoint(checkpoint_path, device)
    spec = get_spec(meta["dataset"])
    labels = list(meta["class_labels"])
    if tuple(labels) != get_labels(spec.key):
        raise CheckpointError(
            f"Checkpoint {checkpoint_path} stores class labels that differ from the {spec.display_name} class "
            "mapping in src/class_mapping.py; refusing to report metrics with a mismatched label order."
        )
    preprocessing = meta.get("preprocessing") or {}
    mean = float(preprocessing.get("mean", spec.mean))
    std = float(preprocessing.get("std", spec.std))
    images, targets = load_split(spec, data_dir, train=False)
    test_split_size = int(images.shape[0])
    subset = _seeded_subset_indices(test_split_size, max_test_samples, seed)
    if subset is not None:
        images, targets = images[subset], targets[subset]
    full_split = subset is None
    # A quick subset check must not overwrite the full-split report in results/<dataset>/.
    out_dir = Path(results_dir) if results_dir is not None else RESULTS_DIR / spec.results_subdir
    if results_dir is None and not full_split:
        out_dir = out_dir / "subset"
    num_samples = int(images.shape[0])
    logger.info(
        "Evaluating %s on %s %s test samples on %s",
        display_path(checkpoint_path),
        f"{num_samples:,}" if full_split else f"a seeded subset of {num_samples:,}/{test_split_size:,}",
        spec.display_name,
        describe_device(device),
    )

    start = time.perf_counter()
    loader = TensorBatchLoader(images, targets, batch_size, mean=mean, std=std, device=device)
    predictions = collect_predictions(model, loader, device)
    del loader
    evaluated_at = utc_now()
    metrics = {
        "dataset": spec.key,
        "display_name": spec.display_name,
        "split": TEST_SPLIT_NAME,
        "full_test_split": full_split,
        "test_split_size": test_split_size,
        "subset_seed": None if full_split else seed,
        "checkpoint": display_path(checkpoint_path),
        "best_epoch": meta.get("best_epoch"),
        "device": str(device),
        "evaluated_at": evaluated_at,
        **compute_metrics(predictions["y_true"], predictions["y_pred"], labels, loss=predictions["loss"]),
    }
    metrics["seconds"] = round(time.perf_counter() - start, 2)

    title = f"{spec.display_name} - official test split"
    if full_split:
        title += f" ({num_samples:,} samples)"
    else:
        title += f" (seeded subset: {num_samples:,} of {test_split_size:,})"
    artifacts = save_evaluation_artifacts(metrics, labels, out_dir, title=title)

    compact = compact_test_metrics(metrics, evaluated_at=evaluated_at)
    recorded = False
    if update_checkpoint and full_split:
        update_checkpoint_metadata(checkpoint_path, test_metrics=compact)
        recorded = True
    elif update_checkpoint:
        logger.warning(
            "Evaluated only %d of %d test samples: NOT recording these metrics as test accuracy in %s.",
            num_samples,
            test_split_size,
            display_path(checkpoint_path),
        )

    roundtrip = None
    if roundtrip_samples > 0:
        roundtrip = _run_roundtrip(model, images, targets, preprocessing, spec.key, device, roundtrip_samples, seed)
        if roundtrip is not None:
            roundtrip = {
                "description": (
                    "Test glyphs re-drawn as crisp strokes on a 400x400 black canvas at a random size "
                    f"({ROUNDTRIP_GLYPH_FRACTION[0]:.0%}-{ROUNDTRIP_GLYPH_FRACTION[1]:.0%} of the canvas) and "
                    "position, then passed through the app's preprocessing pipeline."
                ),
                "dataset": spec.key,
                "checkpoint": display_path(checkpoint_path),
                "full_test_split": full_split,
                "evaluated_at": utc_now(),
                **roundtrip,
            }
            artifacts["roundtrip"] = str(write_json(out_dir / "roundtrip.json", roundtrip))
    stale_roundtrip = out_dir / "roundtrip.json"
    if roundtrip is None and stale_roundtrip.exists():
        # It described an earlier evaluation; keeping it next to the new metrics would be misleading.
        stale_roundtrip.unlink()
        logger.info("Removed outdated %s (round-trip check skipped in this run)", display_path(stale_roundtrip))

    return {
        "checkpoint": str(checkpoint_path),
        "dataset": spec.key,
        "metrics": metrics,
        "test_metrics": compact,
        "recorded_in_checkpoint": recorded,
        "full_test_split": full_split,
        "results_dir": str(out_dir),
        "artifacts": artifacts,
        "roundtrip": roundtrip,
    }


def format_evaluation_summary(outcome: dict) -> str:
    """Human-readable multi-line summary of an :func:`evaluate_checkpoint` result."""
    metrics = outcome["metrics"]
    if outcome["full_test_split"]:
        scope = f"official test split, {metrics['num_samples']:,} samples"
    else:
        scope = (
            f"SUBSET of the official test split: {metrics['num_samples']:,} of "
            f"{metrics['test_split_size']:,} samples"
        )
    lines = [
        f"{metrics['display_name']} ({scope})",
        f"  accuracy        {metrics['accuracy']:.2%}",
        f"  loss            {metrics['loss']:.4f}",
        f"  macro F1        {metrics['macro_f1']:.4f}  (precision {metrics['macro_precision']:.4f}, "
        f"recall {metrics['macro_recall']:.4f})",
        f"  weighted F1     {metrics['weighted_f1']:.4f}",
    ]
    if metrics.get("case_insensitive_accuracy") is not None:
        lines.append(
            f"  ignoring case   {metrics['case_insensitive_accuracy']:.2%}  (c/C, o/O, s/S ... counted as correct)"
        )
    never = metrics.get("classes_never_predicted") or []
    never_text = ", ".join(never[:15]) + (f" (+{len(never) - 15} more)" if len(never) > 15 else "")
    lines.append(f"  never predicted {never_text or 'none'}")
    if metrics.get("top_confusions"):
        top = ", ".join(f"{c['true']}->{c['pred']} ({c['count']})" for c in metrics["top_confusions"][:5])
        lines.append(f"  top confusions  {top}")
    roundtrip = outcome.get("roundtrip")
    if roundtrip:
        lines.append(
            f"  canvas round-trip {roundtrip['accuracy']:.2%} on {roundtrip['samples']:,} simulated drawings "
            f"(direct {roundtrip['direct_accuracy_same_samples']:.2%} on the same samples, "
            f"{roundtrip['preprocessing_errors']} rejected)"
        )
    recorded = "yes" if outcome["recorded_in_checkpoint"] else "no"
    if not outcome["full_test_split"]:
        recorded += " (partial evaluation - never recorded as test accuracy)"
    lines.append(f"  stored in checkpoint: {recorded}")
    lines.append(f"  artefacts       {display_path(outcome['results_dir'])}/")
    return "\n".join(lines)
