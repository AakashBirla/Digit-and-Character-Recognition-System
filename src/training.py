"""Training loop for the DigitCNN on MNIST (digits) or EMNIST ByClass (characters).

:func:`fit` runs the whole experiment:

1. seed everything and pick the device ("Using device: ...");
2. load the official TRAIN split and hold out a seeded validation subset of it. The official
   TEST split is never used for training, model selection or early stopping;
3. train with ``CrossEntropyLoss`` on raw logits and Adam. Augmentation (small random affine
   transforms) is applied to the training batches only;
4. after every epoch, measure validation loss/accuracy. ``ReduceLROnPlateau`` halves the learning
   rate when validation accuracy stalls. The checkpoint is saved whenever validation accuracy
   improves, so the file on disk is always the *best* model. Early stopping ends the run after
   ``patience`` epochs without improvement;
5. write ``history.json`` and ``training_curves.png``;
6. reload the best checkpoint and evaluate it once on the official test split
   (:func:`src.evaluation.evaluate_checkpoint`).

``train_mnist.py`` and ``train_emnist.py`` are thin wrappers around :func:`run_cli`. User-facing
progress (device, dataset summary, one line per epoch) is printed to stdout because these are
interactive command-line tools. Diagnostics go through ``logging``.
"""

from __future__ import annotations

import argparse
import inspect
import logging
import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from src.checkpoints import save_checkpoint, update_checkpoint_metadata
from src.class_mapping import get_labels
from src.config import DATA_DIR, IMAGE_SIZE, RESULTS_DIR, DatasetSpec, get_spec
from src.datasets import augment_batch, build_data
from src.evaluation import display_path, evaluate_checkpoint, format_evaluation_summary, write_json
from src.models import DigitCNN, architecture_summary
from src.utils import describe_device, resolve_device, set_seed, setup_logging

logger = logging.getLogger(__name__)

PROGRESS_MIN_BATCHES = 500  # print in-epoch progress (every 25%) only for long epochs, e.g. full EMNIST
# ReduceLROnPlateau on validation accuracy. threshold=0: any increase counts as an improvement, the same
# rule used for saving the best checkpoint and for early stopping.
LR_SCHEDULER = {"mode": "max", "factor": 0.5, "patience": 1, "threshold": 0.0}

# Chart colours: categorical slots 1 and 2 of the project's validated palette, plus recessive ink.
_TRAIN_COLOR = "#2a78d6"
_VAL_COLOR = "#eb6834"
_INK = "#0b0b0b"
_MUTED_INK = "#52514e"
_GRID = "#e4e3df"


# --------------------------------------------------------------------------------------------
# Configuration and command line
# --------------------------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """All settings of one training run (see ``build_arg_parser`` for the matching CLI flags)."""

    dataset: str
    epochs: int
    batch_size: int
    lr: float = 1e-3
    weight_decay: float = 0.0
    val_fraction: float = 0.1
    patience: int = 4  # epochs without val-accuracy improvement before stopping; 0 disables early stopping
    seed: int = 42
    augment: bool = True
    balanced_loss: bool = False
    deterministic: bool = False
    device: str = "auto"
    data_dir: Path = DATA_DIR
    output: Path | None = None  # None -> spec.default_checkpoint
    results_dir: Path | None = None  # None -> RESULTS_DIR / spec.results_subdir
    max_train_samples: int | None = None
    max_test_samples: int | None = None
    eval_batch_size: int = 1024
    dropout: float = 0.3
    hidden_units: int = 128
    conv_channels: tuple[int, int] = (32, 64)
    roundtrip_samples: int = 2000

    def __post_init__(self) -> None:
        self.dataset = get_spec(self.dataset).key  # also accepts a mode ("digits"); ValueError if unknown
        self.data_dir = Path(self.data_dir)
        self.output = None if self.output is None else Path(self.output)
        self.results_dir = None if self.results_dir is None else Path(self.results_dir)
        self.conv_channels = tuple(int(c) for c in self.conv_channels)
        checks = [
            (self.epochs >= 1, f"epochs must be >= 1, got {self.epochs}"),
            (self.batch_size >= 1, f"batch_size must be >= 1, got {self.batch_size}"),
            (self.eval_batch_size >= 1, f"eval_batch_size must be >= 1, got {self.eval_batch_size}"),
            (self.lr > 0, f"lr must be > 0, got {self.lr}"),
            (self.weight_decay >= 0, f"weight_decay must be >= 0, got {self.weight_decay}"),
            (0 < self.val_fraction < 1, f"val_fraction must be in (0, 1), got {self.val_fraction}"),
            (self.patience >= 0, f"patience must be >= 0, got {self.patience}"),
            (self.max_train_samples is None or self.max_train_samples >= 2, "max_train_samples must be >= 2"),
            (self.max_test_samples is None or self.max_test_samples >= 1, "max_test_samples must be >= 1"),
            (self.roundtrip_samples >= 0, f"roundtrip_samples must be >= 0, got {self.roundtrip_samples}"),
            (0 <= self.dropout < 1, f"dropout must be in [0, 1), got {self.dropout}"),
            (len(self.conv_channels) == 2, "conv_channels must have exactly two values"),
        ]
        for ok, message in checks:
            if not ok:
                raise ValueError(message)

    @property
    def spec(self) -> DatasetSpec:
        """The dataset specification this run trains on."""
        return get_spec(self.dataset)

    @property
    def checkpoint_path(self) -> Path:
        """Where the best checkpoint is written (``output`` or the dataset's default checkpoint)."""
        return self.output if self.output is not None else self.spec.default_checkpoint

    @property
    def results_path(self) -> Path:
        """Directory for history, curves and evaluation artefacts.

        Defaults to ``results/<dataset>/`` for the default checkpoint. A run with a custom ``--output``
        (e.g. a smoke test) writes next to that checkpoint instead, so it can never overwrite the
        artefacts that describe the real model in ``models/``.
        """
        if self.results_dir is not None:
            return self.results_dir
        if self.output is not None and Path(self.output).resolve() != self.spec.default_checkpoint.resolve():
            output = Path(self.output)
            return output.parent / f"{output.stem}_results"
        return RESULTS_DIR / self.spec.results_subdir


def _int_at_least(minimum: int):
    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}") from exc
        if value < minimum:
            raise argparse.ArgumentTypeError(f"must be >= {minimum}, got {value}")
        return value

    return parse


def _float_in(low: float, high: float | None, *, low_inclusive: bool):
    def parse(text: str) -> float:
        try:
            value = float(text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"expected a number, got {text!r}") from exc
        too_low = value < low if low_inclusive else value <= low
        too_high = high is not None and value >= high
        if not math.isfinite(value) or too_low or too_high:
            bounds = f"{'[' if low_inclusive else '('}{low}, {high if high is not None else 'inf'})"
            raise argparse.ArgumentTypeError(f"must be in {bounds}, got {value}")
        return value

    return parse


def build_arg_parser(dataset_key: str, *, default_epochs: int, default_batch_size: int) -> argparse.ArgumentParser:
    """Command-line interface shared by ``train_mnist.py`` and ``train_emnist.py``."""
    spec = get_spec(dataset_key)
    script = spec.train_command.split()[-1]
    parser = argparse.ArgumentParser(
        description=(
            f"Train the DigitCNN on {spec.display_name} ({spec.num_classes} classes). The validation set is a "
            "seeded subset of the official train split; the official test split is used once, at the end, "
            "to evaluate the best checkpoint."
        ),
        epilog=(
            "examples:\n"
            f"  python {script}\n"
            f"  python {script} --epochs 3 --device cpu\n"
            f"  python {script} --epochs 1 --max-train-samples 3000 --max-test-samples 1000 "
            "--output /tmp/smoke.pth --results-dir /tmp/smoke"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    positive = _int_at_least(1)
    parser.add_argument("--epochs", type=positive, default=default_epochs,
                        help="maximum number of epochs (default: %(default)s)")
    parser.add_argument("--batch-size", type=positive, default=default_batch_size,
                        help="training batch size (default: %(default)s)")
    parser.add_argument("--lr", type=_float_in(0.0, None, low_inclusive=False), default=1e-3,
                        help="Adam learning rate (default: %(default)s)")
    parser.add_argument("--weight-decay", type=_float_in(0.0, None, low_inclusive=True), default=0.0,
                        help="Adam weight decay (default: %(default)s)")
    parser.add_argument("--val-fraction", type=_float_in(0.0, 1.0, low_inclusive=False), default=0.1,
                        help="fraction of the official train split held out for validation (default: %(default)s)")
    parser.add_argument("--patience", type=_int_at_least(0), default=4,
                        help="early stopping: epochs without val-accuracy improvement; 0 disables "
                             "(default: %(default)s)")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for weights, split, shuffling and augmentation (default: %(default)s)")
    parser.add_argument("--no-augment", action="store_true",
                        help="disable random affine augmentation of training batches")
    parser.add_argument("--balanced-loss", action="store_true",
                        help="weight the loss by inverse sqrt class frequency (helps rare EMNIST classes)")
    parser.add_argument("--deterministic", action="store_true",
                        help="make GPU training bit-reproducible for a given --seed (disables cuDNN autotuning; "
                        "CPU runs are always reproducible)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="auto = CUDA when available, else CPU (default: %(default)s)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help=f"dataset directory; downloads here on first use (default: {display_path(DATA_DIR)})")
    parser.add_argument("--output", type=Path, default=None,
                        help=f"checkpoint path (default: {display_path(spec.default_checkpoint)})")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help=f"directory for history, plots and test metrics (default: "
                             f"{display_path(RESULTS_DIR / spec.results_subdir)}; with a custom --output: "
                             f"<output folder>/<output name>_results)")
    parser.add_argument("--max-train-samples", type=_int_at_least(2), default=None,
                        help="use a seeded subset of the train split (before the val split), for quick smoke tests")
    parser.add_argument("--max-test-samples", type=positive, default=None,
                        help="evaluate a seeded subset of the test split; such partial results are NOT stored "
                             "in the checkpoint as test accuracy")
    parser.add_argument("--eval-batch-size", type=positive, default=1024,
                        help="batch size for validation and test evaluation (default: %(default)s)")
    parser.add_argument("--roundtrip-samples", type=_int_at_least(0), default=2000,
                        help="simulated canvas drawings for the preprocessing round-trip check; 0 skips it "
                             "(default: %(default)s)")
    return parser


def config_from_args(dataset_key: str, args: argparse.Namespace) -> TrainConfig:
    """Turn parsed command-line arguments into a :class:`TrainConfig`."""
    return TrainConfig(
        dataset=dataset_key,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_fraction=args.val_fraction,
        patience=args.patience,
        seed=args.seed,
        augment=not args.no_augment,
        balanced_loss=args.balanced_loss,
        deterministic=args.deterministic,
        device=args.device,
        data_dir=args.data_dir,
        output=args.output,
        results_dir=args.results_dir,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        eval_batch_size=args.eval_batch_size,
        roundtrip_samples=args.roundtrip_samples,
    )


# --------------------------------------------------------------------------------------------
# One epoch of training / evaluation
# --------------------------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
    *,
    show_progress: bool = False,
) -> tuple[float, float]:
    """Train for one pass over ``loader``; returns ``(mean loss, accuracy in [0, 1])``.

    ``criterion`` must return the batch-mean loss (PyTorch's default reduction). Running totals stay
    on the device, so the GPU never waits for a per-batch ``.item()``. With ``show_progress``, a short
    line is printed at 25%, 50% and 75% of the epoch.
    """
    device = resolve_device(device)
    model.train()
    total_loss = torch.zeros((), dtype=torch.float64, device=device)
    correct = torch.zeros((), dtype=torch.int64, device=device)
    seen = 0
    num_batches = len(loader) if hasattr(loader, "__len__") else 0
    milestones = {math.ceil(num_batches * q): q for q in (0.25, 0.5, 0.75)} if show_progress and num_batches else {}
    start = time.perf_counter()

    for step, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch = y.shape[0]
        total_loss += loss.detach().double() * batch
        correct += (logits.argmax(dim=1) == y).sum()
        seen += batch
        if step in milestones and step < num_batches:
            elapsed = time.perf_counter() - start
            remaining = elapsed * (num_batches - step) / step
            print(
                f"    {milestones[step]:>4.0%} | batch {step:>{len(str(num_batches))}}/{num_batches} | "
                f"loss {total_loss.item() / seen:.4f} | acc {correct.item() / seen:.2%} | "
                f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s left",
                flush=True,
            )

    if seen == 0:
        raise ValueError("The training loader yielded no batches.")
    mean_loss = total_loss.item() / seen
    if not math.isfinite(mean_loss):
        raise RuntimeError("Training diverged (loss is NaN or infinite). Try a lower --lr.")
    return mean_loss, correct.item() / seen


@torch.no_grad()
def evaluate_loss_accuracy(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: str | torch.device,
) -> tuple[float, float]:
    """Mean loss and accuracy of ``model`` (in eval mode, no gradients) over ``loader``."""
    device = resolve_device(device)
    model.eval()
    total_loss = torch.zeros((), dtype=torch.float64, device=device)
    correct = torch.zeros((), dtype=torch.int64, device=device)
    seen = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        total_loss += criterion(logits, y).double() * y.shape[0]
        correct += (logits.argmax(dim=1) == y).sum()
        seen += y.shape[0]
    if seen == 0:
        raise ValueError("The evaluation loader yielded no batches.")
    return total_loss.item() / seen, correct.item() / seen


# --------------------------------------------------------------------------------------------
# Helpers used by fit()
# --------------------------------------------------------------------------------------------
def _preprocessing_metadata(spec: DatasetSpec) -> dict:
    """Preprocessing settings stored in the checkpoint so inference reproduces the training data."""
    try:
        from src.preprocessing import PreprocessConfig  # lazy: the training loop itself doesn't need OpenCV
    except ModuleNotFoundError as exc:
        if exc.name != "src.preprocessing":
            raise
        logger.warning("src.preprocessing is not available; storing preprocessing metadata from the dataset spec.")
        return {
            "mean": spec.mean,
            "std": spec.std,
            "image_size": IMAGE_SIZE,
            "target_box": spec.target_box,
            "centering": spec.centering,
            "stroke_width": spec.stroke_width,
        }
    return PreprocessConfig.for_dataset(spec.key).to_metadata()


def _format_duration(seconds: float) -> str:
    """``42s``, ``3m 07s`` or ``1h 05m``."""
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _augmentation_settings() -> dict:
    """The random-affine ranges used by the training loader (the defaults of ``augment_batch``)."""
    return {
        name: parameter.default
        for name, parameter in inspect.signature(augment_batch).parameters.items()
        if parameter.default is not inspect.Parameter.empty and name != "generator"
    }


def _build_criterion(class_counts: Sequence[int], balanced: bool, device: torch.device) -> nn.CrossEntropyLoss:
    """Cross-entropy on raw logits, optionally with inverse-sqrt-frequency class weights.

    The weights are scaled so the average weight over training samples is 1, which keeps the loss
    comparable to the unweighted loss.
    """
    if not balanced:
        return nn.CrossEntropyLoss()
    counts = torch.tensor(class_counts, dtype=torch.float64).clamp(min=1)
    weights = counts.rsqrt()
    weights *= counts.sum() / (weights * counts).sum()
    return nn.CrossEntropyLoss(weight=weights.float().to(device))


def _class_count_summary(counts: Sequence[int], labels: Sequence[str]) -> str:
    """One line describing the class balance of the training subset."""
    present = [(count, label) for count, label in zip(counts, labels) if count > 0]
    if not present:
        return "no samples"
    low, high = min(present), max(present)
    text = (
        f"{len(present)}/{len(labels)} classes present, min {low[0]:,} ('{low[1]}'), "
        f"max {high[0]:,} ('{high[1]}'), imbalance {high[0] / low[0]:.1f}x"
    )
    digit_share = sum(count for count, label in zip(counts, labels) if label.isdigit()) / max(sum(counts), 1)
    if digit_share < 1.0:
        text += f", digits {digit_share:.1%} of samples"
    return text


def plot_training_curves(
    history: Sequence[dict], path: Path | str, *, title: str, best_epoch: int | None = None
) -> Path:
    """Save loss and accuracy curves (train vs validation per epoch) side by side."""
    from matplotlib.figure import Figure  # no pyplot: no global state, works headless (Agg)
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    epochs = [row["epoch"] for row in history]
    fig = Figure(figsize=(11, 4.3), dpi=130)
    ax_loss, ax_acc = fig.subplots(1, 2)
    panels = (
        (ax_loss, "loss", "Cross-entropy loss", 1.0),
        (ax_acc, "acc", "Accuracy", 100.0),
    )
    for ax, key, ylabel, scale in panels:
        style = {"linewidth": 2, "marker": "o", "markersize": 4}
        ax.plot(epochs, [row[f"train_{key}"] * scale for row in history], color=_TRAIN_COLOR,
                label="Train (augmented, dropout on)", **style)
        ax.plot(epochs, [row[f"val_{key}"] * scale for row in history], color=_VAL_COLOR,
                label="Validation", **style)
        if best_epoch is not None:
            ax.axvline(best_epoch, color=_MUTED_INK, linestyle="--", linewidth=1)
        ax.set_xlabel("Epoch", color=_INK)
        ax.set_ylabel(ylabel, color=_INK)
        ax.set_xlim(0.5, max(epochs) + 0.5)
        if len(epochs) <= 20:
            ax.set_xticks(epochs)
        else:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="y", color=_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=_MUTED_INK)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(_GRID)
        ax.legend(frameon=False)
    ax_acc.yaxis.set_major_formatter(PercentFormatter(decimals=1))
    subtitle = f" · best epoch {best_epoch} (dashed line)" if best_epoch is not None else ""
    fig.suptitle(f"{title}{subtitle}", color=_INK)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    return path


# --------------------------------------------------------------------------------------------
# The full experiment
# --------------------------------------------------------------------------------------------
def fit(config: TrainConfig) -> dict:
    """Train, keep the best checkpoint by validation accuracy, then evaluate it on the official test split.

    Returns a JSON-safe summary: checkpoint path, best epoch, validation and test metrics, timings and
    artefact paths.
    """
    spec = config.spec
    labels = get_labels(spec.key)
    checkpoint_path = config.checkpoint_path
    results_dir = config.results_path
    results_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.seed)
    device = resolve_device(config.device)
    print(f"Using device: {describe_device(device)}", flush=True)
    if device.type == "cuda":
        # cuDNN autotuning picks the fastest kernels for our fixed input size, but some of them are not
        # deterministic, so two GPU runs with the same seed differ slightly unless --deterministic is used.
        torch.backends.cudnn.benchmark = not config.deterministic
        torch.backends.cudnn.deterministic = config.deterministic

    # ---- data: train/val from the official TRAIN split; the test split is evaluated only at the end
    data = build_data(
        spec,
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        eval_batch_size=config.eval_batch_size,
        val_fraction=config.val_fraction,
        seed=config.seed,
        device=device,
        augment=config.augment,
        include_test=False,
        max_train_samples=config.max_train_samples,
    )
    sample_x, sample_y = next(iter(data.val_loader))
    print(f"Dataset: {spec.display_name} ({spec.num_classes} classes)")
    subset_note = f" (seeded subset of {config.max_train_samples:,})" if config.max_train_samples else ""
    print(f"  train {data.num_train:,} / validation {data.num_val:,} from the official train split{subset_note}")
    print("  test  official test split, evaluated once with the best checkpoint after training")
    print(
        f"  input batch {tuple(sample_x.shape)} {str(sample_x.dtype).removeprefix('torch.')} "
        f"(normalised with mean {spec.mean}, std {spec.std}); "
        f"labels {int(sample_y.min())}-{int(sample_y.max())} ({labels[0]}..{labels[-1]})"
    )
    print(f"  class balance: {_class_count_summary(data.train_class_counts, labels)}")
    del sample_x, sample_y

    # ---- model, loss, optimiser, scheduler
    model = DigitCNN(
        num_classes=spec.num_classes,
        conv_channels=config.conv_channels,
        hidden_units=config.hidden_units,
        dropout=config.dropout,
    ).to(device)
    print(f"Model: {architecture_summary(model)} ({model.count_parameters():,} parameters)")
    criterion = _build_criterion(data.train_class_counts, config.balanced_loss, device)
    val_criterion = nn.CrossEntropyLoss()  # unweighted, so validation loss is comparable across runs
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **LR_SCHEDULER)

    training_config = {
        "dataset": spec.key,
        "device": device.type,
        "device_name": describe_device(device),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "eval_batch_size": config.eval_batch_size,
        "optimizer": "Adam",
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "lr_scheduler": "ReduceLROnPlateau(mode='max', factor=0.5, patience=1, threshold=0) on validation accuracy",
        "loss": "CrossEntropyLoss" + (" with inverse-sqrt-frequency class weights" if config.balanced_loss else ""),
        "balanced_loss": config.balanced_loss,
        "deterministic": config.deterministic,
        "augment": config.augment,
        "augmentation": _augmentation_settings() if config.augment else None,
        "seed": config.seed,
        "val_fraction": config.val_fraction,
        "validation_source": "seeded subset of the official train split",
        "patience": config.patience,
        "num_train": data.num_train,
        "num_val": data.num_val,
        "max_train_samples": config.max_train_samples,
    }
    preprocessing = _preprocessing_metadata(spec)

    # ---- training loop
    show_progress = len(data.train_loader) >= PROGRESS_MIN_BATCHES
    history: list[dict] = []
    best_val_acc, best_epoch, epochs_without_improvement = -1.0, 0, 0
    stopped_early = interrupted = False
    history_path = results_dir / "history.json"
    run_start = time.perf_counter()
    print(f"Training for up to {config.epochs} epoch(s), {len(data.train_loader):,} batches per epoch "
          f"(batch size {config.batch_size}); saving the best model to {display_path(checkpoint_path)}", flush=True)

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        lr = optimizer.param_groups[0]["lr"]
        try:
            train_loss, train_acc = train_one_epoch(
                model, data.train_loader, criterion, optimizer, device, show_progress=show_progress
            )
        except KeyboardInterrupt:
            if not history:
                raise
            interrupted = True
            print(f"\nInterrupted during epoch {epoch}; keeping the best checkpoint (epoch {best_epoch}).")
            break
        val_loss, val_acc = evaluate_loss_accuracy(model, data.val_loader, val_criterion, device)
        scheduler.step(val_acc)
        seconds = time.perf_counter() - epoch_start

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc, best_epoch, epochs_without_improvement = val_acc, epoch, 0
            save_checkpoint(
                checkpoint_path,
                model,
                dataset_key=spec.key,
                labels=labels,
                preprocessing=preprocessing,
                training_config=training_config,
                val_metrics={"accuracy": val_acc, "loss": val_loss},
                best_epoch=epoch,
            )
        else:
            epochs_without_improvement += 1

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": lr,
            "seconds": round(seconds, 2),
            "improved": improved,
        })
        new_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:>{len(str(config.epochs))}}/{config.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.2%} | val loss {val_loss:.4f} acc {val_acc:.2%} | "
            f"lr {lr:.1e} | {seconds:.1f}s"
            + (" | saved best" if improved else "")
            + (f" | lr -> {new_lr:.1e}" if new_lr < lr else ""),
            flush=True,
        )
        write_json(history_path, _history_payload(spec, device, history, best_epoch, best_val_acc, False))

        if config.patience and epochs_without_improvement >= config.patience:
            stopped_early = True
            print(f"Early stopping: no validation-accuracy improvement for {config.patience} epochs.")
            break

    training_seconds = time.perf_counter() - run_start
    write_json(history_path, _history_payload(spec, device, history, best_epoch, best_val_acc, stopped_early))
    curves_path = plot_training_curves(
        history, results_dir / "training_curves.png", title=f"{spec.display_name} training", best_epoch=best_epoch
    )
    update_checkpoint_metadata(
        checkpoint_path,
        training_config={
            **training_config,
            "epochs_completed": len(history),
            "stopped_early": stopped_early,
            "interrupted": interrupted,
            "training_seconds": round(training_seconds, 1),
        },
    )

    # ---- free the training data (EMNIST is ~0.5 GB on the GPU) before evaluating
    del data, model, optimizer, scheduler, criterion
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"\nEvaluating the best checkpoint (epoch {best_epoch}) on the official test split ...", flush=True)
    evaluation = evaluate_checkpoint(
        checkpoint_path,
        data_dir=config.data_dir,
        device=device,
        results_dir=results_dir,
        batch_size=config.eval_batch_size,
        max_test_samples=config.max_test_samples,
        update_checkpoint=True,
        roundtrip_samples=config.roundtrip_samples,
    )

    print("\n" + "=" * 72)
    print(f"Training finished in {_format_duration(training_seconds)} ({len(history)} epoch(s)"
          + (", early stopped" if stopped_early else "") + (", interrupted" if interrupted else "") + ")")
    print(f"Best epoch {best_epoch}: validation accuracy {best_val_acc:.2%}")
    print(f"Checkpoint: {display_path(checkpoint_path)}")
    print(format_evaluation_summary(evaluation))
    print("=" * 72, flush=True)

    return {
        "dataset": spec.key,
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "results_dir": str(results_dir),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "interrupted": interrupted,
        "training_seconds": round(training_seconds, 1),
        "history_path": str(history_path),
        "curves_path": str(curves_path),
        "test_metrics": evaluation["test_metrics"],
        "test_metrics_recorded": evaluation["recorded_in_checkpoint"],
        "roundtrip": evaluation["roundtrip"],
        "artifacts": evaluation["artifacts"],
    }


def _history_payload(
    spec: DatasetSpec,
    device: torch.device,
    history: list[dict],
    best_epoch: int,
    best_val_acc: float,
    stopped_early: bool,
) -> dict:
    return {
        "dataset": spec.key,
        "display_name": spec.display_name,
        "device": describe_device(device),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "epochs": history,
    }


def run_cli(
    dataset_key: str, *, default_epochs: int, default_batch_size: int, argv: Sequence[str] | None = None
) -> int:
    """Entry point used by ``train_mnist.py`` / ``train_emnist.py``; returns the process exit code."""
    setup_logging()
    parser = build_arg_parser(dataset_key, default_epochs=default_epochs, default_batch_size=default_batch_size)
    args = parser.parse_args(argv)
    try:
        config = config_from_args(dataset_key, args)
        resolve_device(config.device)  # fail fast, e.g. --device cuda on a machine without CUDA
    except ValueError as exc:
        parser.error(str(exc))
    try:
        fit(config)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    return 0
