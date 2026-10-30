"""Dataset loading, train/validation splitting, augmentation and fast in-memory batching.

Key facts (measured on the real data, see ``inspect_datasets.py``):

* ``torchvision.datasets.MNIST`` images are stored upright.
* ``torchvision.datasets.EMNIST`` images are stored *transposed* (rows and columns swapped).
  :func:`to_upright` applies the single ``transpose(1, 2)`` that makes them upright; it is the
  only place in the project where this happens, so training, evaluation and inference all see
  upright glyphs and no orientation fix is needed at inference time.

Both datasets fit comfortably in memory as ``uint8`` tensors (EMNIST ByClass train is ~550 MB),
so instead of a ``DataLoader`` with per-sample Python transforms we use
:class:`TensorBatchLoader`: it slices whole batches out of one tensor, optionally on the GPU,
and applies augmentation to the batch at once with ``affine_grid`` / ``grid_sample``.

The official test split is never used for training or validation: validation is a seeded
subset of the official *training* split (:func:`train_val_split`).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.config import DATA_DIR, IMAGE_SIZE, DatasetSpec, get_spec
from src.utils import resolve_device

logger = logging.getLogger(__name__)

_DOWNLOAD_SIZES = {"mnist": "~11 MB", "emnist_byclass": "~560 MB zip"}
_AUGMENT_SEED_OFFSET = 1_000_003  # keeps the augmentation RNG stream separate from the shuffle stream


# --------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------
def _split_name(train: bool) -> str:
    return "train" if train else "test"


def _raw_files(spec: DatasetSpec, data_dir: Path, train: bool) -> list[Path]:
    """Paths of the uncompressed IDX files torchvision reads for this split."""
    if spec.key == "mnist":
        prefix = "train" if train else "t10k"
        raw_dir = data_dir / "MNIST" / "raw"
    else:
        prefix = f"emnist-byclass-{_split_name(train)}"
        raw_dir = data_dir / "EMNIST" / "raw"
    return [raw_dir / f"{prefix}-images-idx3-ubyte", raw_dir / f"{prefix}-labels-idx1-ubyte"]


def open_torchvision_dataset(
    key_or_mode: str | DatasetSpec,
    data_dir: Path | str = DATA_DIR,
    train: bool = True,
    download: bool = True,
) -> Any:
    """Return the raw ``torchvision`` dataset object (``MNIST`` or ``EMNIST(split="byclass")``).

    Its ``.data`` is exactly as stored on disk (EMNIST is *not* yet upright). Most code should use
    :func:`load_split` instead; this is exposed for inspection (e.g. reading ``.classes``).
    """
    import torchvision  # imported lazily: only needed when real data is loaded

    spec = get_spec(key_or_mode)
    data_dir = Path(data_dir)
    split = _split_name(train)
    present = all(path.is_file() for path in _raw_files(spec, data_dir, train))

    if not present and not download:
        raise RuntimeError(
            f"{spec.display_name} {split} files were not found under {data_dir}. "
            "Enable downloading (omit --no-download) or place the files as described in data/README.md."
        )
    if present:
        logger.info("Loading %s %s split from %s", spec.display_name, split, data_dir)
    else:
        logger.info(
            "Downloading/loading %s (%s) into %s ... this only happens once and may take a while.",
            spec.display_name,
            _DOWNLOAD_SIZES[spec.key],
            data_dir,
        )

    try:
        if spec.key == "mnist":
            return torchvision.datasets.MNIST(root=str(data_dir), train=train, download=download)
        return torchvision.datasets.EMNIST(root=str(data_dir), split="byclass", train=train, download=download)
    except Exception as exc:  # network errors, checksum failures, corrupt archives ...
        raise RuntimeError(
            f"Could not load {spec.display_name} ({split} split) from {data_dir}: {exc}. "
            "Check your internet connection and free disk space, or see data/README.md for manual download steps."
        ) from exc


def to_upright(images: torch.Tensor | np.ndarray, spec: DatasetSpec | str) -> torch.Tensor | np.ndarray:
    """Return images in upright orientation.

    For EMNIST (``spec.transpose_raw``) this swaps the last two axes - the single transpose that
    turns the raw torchvision data upright. MNIST is returned unchanged. Works for ``(28, 28)``,
    ``(N, 28, 28)`` and ``(N, 1, 28, 28)`` tensors or arrays.
    """
    spec = get_spec(spec)
    if not spec.transpose_raw:
        return images
    if isinstance(images, np.ndarray):
        return np.ascontiguousarray(np.swapaxes(images, -2, -1))
    return images.transpose(-2, -1).contiguous()


def _validate_split(images: torch.Tensor, targets: torch.Tensor, spec: DatasetSpec, split: str) -> None:
    """Raise ``RuntimeError`` if the loaded tensors do not look like the expected dataset."""
    where = f"{spec.display_name} {split} split"
    if images.dtype != torch.uint8:
        raise RuntimeError(f"{where}: expected uint8 images, got {images.dtype}.")
    if images.ndim != 3 or tuple(images.shape[1:]) != (IMAGE_SIZE, IMAGE_SIZE):
        raise RuntimeError(f"{where}: expected images of shape (N, 28, 28), got {tuple(images.shape)}.")
    if targets.ndim != 1 or targets.shape[0] != images.shape[0]:
        raise RuntimeError(f"{where}: {images.shape[0]} images but targets have shape {tuple(targets.shape)}.")
    if images.shape[0] == 0:
        raise RuntimeError(f"{where} is empty.")
    low, high = int(targets.min()), int(targets.max())
    if low < 0 or high >= spec.num_classes:
        raise RuntimeError(f"{where}: labels must be in [0, {spec.num_classes}), found range [{low}, {high}].")


def load_split(
    key_or_mode: str | DatasetSpec,
    data_dir: Path | str = DATA_DIR,
    train: bool = True,
    download: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load one official split as ``(images uint8 (N, 28, 28) upright, targets int64 (N,))``.

    Downloads the dataset on first use when ``download`` is true. Raises ``RuntimeError`` with a
    helpful message if the data is missing, cannot be downloaded or fails validation.
    """
    spec = get_spec(key_or_mode)
    dataset = open_torchvision_dataset(spec, data_dir, train=train, download=download)
    images = to_upright(dataset.data, spec)
    targets = dataset.targets.to(torch.int64)
    del dataset
    _validate_split(images, targets, spec, _split_name(train))
    logger.info(
        "%s %s split: %d images of shape %s, labels %d-%d",
        spec.display_name,
        _split_name(train),
        images.shape[0],
        tuple(images.shape[1:]),
        int(targets.min()),
        int(targets.max()),
    )
    return images.contiguous(), targets


def train_val_split(num_samples: int, val_fraction: float = 0.1, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    """Seeded random split of ``range(num_samples)`` into disjoint ``(train_idx, val_idx)`` (both sorted).

    ``val_fraction`` must be in (0, 1); each side receives at least one sample.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    if num_samples < 2:
        raise ValueError(f"Need at least 2 samples to split, got {num_samples}")
    num_val = min(max(1, round(num_samples * val_fraction)), num_samples - 1)
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(num_samples, generator=generator)
    val_idx, _ = permutation[:num_val].sort()
    train_idx, _ = permutation[num_val:].sort()
    return train_idx, val_idx


# --------------------------------------------------------------------------------------------
# Batch transforms
# --------------------------------------------------------------------------------------------
def normalize_batch(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """Scale uint8 pixels to [0, 1] (float input is assumed to be in [0, 1] already), then ``(x - mean) / std``."""
    x = x.float().div(255.0) if x.dtype == torch.uint8 else x.float()
    return (x - mean) / std


def augment_batch(
    x: torch.Tensor,
    generator: torch.Generator | None = None,
    max_rotation: float = 10.0,
    scale: tuple[float, float] = (0.9, 1.1),
    max_shear: float = 0.15,
    max_translate: float = 0.06,
) -> torch.Tensor:
    """Apply an independent random affine transform to every image of a batch.

    ``x`` must be float in [0, 1] with shape ``(B, 1, H, W)`` on any device. Each sample gets a
    rotation in ``±max_rotation`` degrees, isotropic scale in ``scale``, horizontal shear in
    ``±max_shear`` and translation of up to ``max_translate`` x image size. Empty space is filled
    with 0 (black background), so augment *before* normalising. Random parameters are drawn from
    ``generator`` (on its device, CPU by default) for reproducibility.
    """
    if x.ndim != 4 or x.shape[1] != 1:
        raise ValueError(f"augment_batch expects (B, 1, H, W), got {tuple(x.shape)}")
    if not x.is_floating_point():
        raise TypeError("augment_batch expects float images in [0, 1]; convert uint8 with x.float() / 255 first")
    batch = x.shape[0]
    if batch == 0:
        return x

    param_device = generator.device if generator is not None else torch.device("cpu")

    def uniform(low: float, high: float) -> torch.Tensor:
        return low + (high - low) * torch.rand(batch, generator=generator, device=param_device)

    angle = torch.deg2rad(uniform(-max_rotation, max_rotation))
    zoom = uniform(scale[0], scale[1])
    shear = uniform(-max_shear, max_shear)
    # Translation as a fraction of the image size; normalised coordinates span 2 units ([-1, 1]).
    shift = torch.stack([uniform(-max_translate, max_translate), uniform(-max_translate, max_translate)], dim=1) * 2

    cos, sin = torch.cos(angle), torch.sin(angle)
    rotation = torch.stack([torch.stack([cos, -sin], dim=1), torch.stack([sin, cos], dim=1)], dim=1)  # (B, 2, 2)
    shear_matrix = torch.eye(2, device=param_device).repeat(batch, 1, 1)
    shear_matrix[:, 0, 1] = shear
    forward = rotation @ shear_matrix * zoom[:, None, None]  # where an input point moves to

    # affine_grid wants the *inverse* map (output pixel -> input sampling location):
    #   p_in = A^-1 (p_out - t)  =>  theta = [A^-1 | -A^-1 t]
    inverse = torch.linalg.inv(forward)
    theta = torch.cat([inverse, -(inverse @ shift.unsqueeze(-1))], dim=2).to(device=x.device, dtype=x.dtype)

    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    out = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return out.clamp_(0.0, 1.0)


# --------------------------------------------------------------------------------------------
# In-memory batch loader
# --------------------------------------------------------------------------------------------
class TensorBatchLoader:
    """Fast in-memory batch iterator: keeps uint8 images on ``device`` (falls back to CPU if they don't fit),
    yields ``(x float32 normalized (B, 1, 28, 28), y int64 (B,))`` on ``device``.

    With ``shuffle=True`` every pass uses a new permutation seeded with ``seed + epoch``, so runs are
    reproducible but epochs differ. Augmentation (:func:`augment_batch`) runs only when ``augment=True``
    (use it for the training loader only).
    """

    def __init__(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        batch_size: int,
        *,
        mean: float,
        std: float,
        shuffle: bool = False,
        augment: bool = False,
        device: str | torch.device = "cpu",
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if images.dtype != torch.uint8:
            raise TypeError(f"TensorBatchLoader expects uint8 images, got {images.dtype}")
        if images.ndim == 4 and images.shape[1] == 1:
            images = images[:, 0]
        if images.ndim != 3:
            raise ValueError(f"Expected images of shape (N, H, W), got {tuple(images.shape)}")
        if targets.ndim != 1 or targets.shape[0] != images.shape[0]:
            raise ValueError(f"{images.shape[0]} images but targets have shape {tuple(targets.shape)}")
        if images.shape[0] == 0:
            raise ValueError("TensorBatchLoader needs at least one sample")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self.device = resolve_device(device)
        self.batch_size = int(batch_size)
        self.mean = float(mean)
        self.std = float(std)
        self.shuffle = bool(shuffle)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.num_samples = int(images.shape[0])
        self._epoch = 0
        self._images, self._targets = self._place(images.contiguous(), targets.to(torch.int64).contiguous())

    # -- storage ------------------------------------------------------------------------------
    def _try_move_to_gpu(self, images: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
        try:
            return images.to(self.device), targets.to(self.device)
        except torch.cuda.OutOfMemoryError:
            return None

    def _place(self, images: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Put the whole dataset on the GPU once if possible, else in (pinned) CPU memory."""
        if self.device.type == "cuda":
            placed = self._try_move_to_gpu(images, targets)
            if placed is not None:
                return placed
            torch.cuda.empty_cache()
            size_mb = images.numel() / 2**20
            logger.warning(
                "Not enough GPU memory to keep %d images (%.0f MB) on %s; using pinned CPU memory and "
                "copying each batch to the GPU instead (slower).",
                self.num_samples,
                size_mb,
                self.device,
            )
            images, targets = images.cpu(), targets.cpu()
            try:
                return images.pin_memory(), targets.pin_memory()
            except RuntimeError as exc:
                logger.warning("Could not pin CPU memory (%s); using pageable memory.", exc)
                return images, targets
        return images.cpu(), targets.cpu()

    @property
    def storage_device(self) -> torch.device:
        """Device that holds the uint8 dataset (``device`` unless the GPU fallback kicked in)."""
        return self._images.device

    @property
    def epoch(self) -> int:
        """Number of passes started so far (the next pass uses ``seed + epoch``)."""
        return self._epoch

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch counter, e.g. to reproduce a specific epoch's shuffle/augmentation."""
        self._epoch = int(epoch)

    # -- iteration ----------------------------------------------------------------------------
    def __len__(self) -> int:
        """Number of batches per pass."""
        if self.drop_last:
            return self.num_samples // self.batch_size
        return math.ceil(self.num_samples / self.batch_size)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        epoch = self._epoch
        self._epoch += 1
        on_device = self.storage_device.type == self.device.type
        order: torch.Tensor | None = None
        if self.shuffle:
            order = torch.randperm(self.num_samples, generator=torch.Generator().manual_seed(self.seed + epoch))
            order = order.to(self.storage_device)
        aug_generator = (
            torch.Generator().manual_seed(self.seed + epoch + _AUGMENT_SEED_OFFSET) if self.augment else None
        )

        for start in range(0, self.num_samples, self.batch_size):
            end = min(start + self.batch_size, self.num_samples)
            if self.drop_last and end - start < self.batch_size:
                break
            if order is None:  # contiguous slice (stays pinned for fast async copies)
                xb, yb = self._images[start:end], self._targets[start:end]
            else:
                index = order[start:end]
                xb, yb = self._images[index], self._targets[index]
            if not on_device:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

            x = xb.unsqueeze(1).float().div_(255.0)
            if self.augment:
                x = augment_batch(x, generator=aug_generator)
            x = x.sub_(self.mean).div_(self.std)
            yield x, yb


# --------------------------------------------------------------------------------------------
# Putting it together
# --------------------------------------------------------------------------------------------
@dataclass
class DataBundle:
    """Everything a training run needs: loaders plus split sizes and class counts."""

    spec: DatasetSpec
    train_loader: TensorBatchLoader
    val_loader: TensorBatchLoader
    test_loader: TensorBatchLoader | None
    num_train: int
    num_val: int
    num_test: int
    train_class_counts: list[int]


def _seeded_subset(
    images: torch.Tensor, targets: torch.Tensor, max_samples: int | None, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a seeded random subset of at most ``max_samples`` samples (all samples if ``None``)."""
    if max_samples is None or max_samples >= images.shape[0]:
        return images, targets
    if max_samples < 1:
        raise ValueError(f"max samples must be >= 1, got {max_samples}")
    selection, _ = torch.randperm(images.shape[0], generator=torch.Generator().manual_seed(seed))[:max_samples].sort()
    return images[selection], targets[selection]


def build_data(
    key_or_mode: str | DatasetSpec,
    *,
    data_dir: Path | str = DATA_DIR,
    batch_size: int = 128,
    eval_batch_size: int = 1024,
    val_fraction: float = 0.1,
    seed: int = 42,
    device: str | torch.device = "cpu",
    augment: bool = True,
    include_test: bool = True,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
    download: bool = True,
) -> DataBundle:
    """Load a dataset and build train / validation (from the TRAIN split) / test loaders.

    Only the training loader shuffles and (if ``augment``) augments. ``max_train_samples`` takes a
    seeded subset of the training split *before* the train/val split (useful for smoke tests);
    ``max_test_samples`` does the same for the test split.
    """
    spec = get_spec(key_or_mode)
    device = resolve_device(device)

    images, targets = load_split(spec, data_dir, train=True, download=download)
    images, targets = _seeded_subset(images, targets, max_train_samples, seed)
    train_idx, val_idx = train_val_split(images.shape[0], val_fraction=val_fraction, seed=seed)
    train_images, train_targets = images[train_idx], targets[train_idx]
    val_images, val_targets = images[val_idx], targets[val_idx]
    del images, targets

    train_class_counts = torch.bincount(train_targets, minlength=spec.num_classes).tolist()
    norm = {"mean": spec.mean, "std": spec.std, "device": device}
    train_loader = TensorBatchLoader(
        train_images, train_targets, batch_size, shuffle=True, augment=augment, seed=seed, **norm
    )
    val_loader = TensorBatchLoader(val_images, val_targets, eval_batch_size, shuffle=False, augment=False, **norm)

    test_loader: TensorBatchLoader | None = None
    num_test = 0
    if include_test:
        test_images, test_targets = load_split(spec, data_dir, train=False, download=download)
        test_images, test_targets = _seeded_subset(test_images, test_targets, max_test_samples, seed)
        test_loader = TensorBatchLoader(
            test_images, test_targets, eval_batch_size, shuffle=False, augment=False, **norm
        )
        num_test = test_loader.num_samples

    logger.info(
        "%s data ready: %d train / %d val (from the official train split) / %d test; stored on %s",
        spec.display_name,
        train_loader.num_samples,
        val_loader.num_samples,
        num_test,
        train_loader.storage_device,
    )
    return DataBundle(
        spec=spec,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_train=train_loader.num_samples,
        num_val=val_loader.num_samples,
        num_test=num_test,
        train_class_counts=train_class_counts,
    )


# --------------------------------------------------------------------------------------------
# Statistics and visualisation
# --------------------------------------------------------------------------------------------
def compute_mean_std(images_uint8: torch.Tensor, chunk_size: int = 16_384) -> tuple[float, float]:
    """Pixel mean and (population) standard deviation of uint8 images scaled to [0, 1].

    Processed in chunks: per-chunk sums of pixel values and squared values are accumulated
    exactly (integers), then combined in float64. Memory friendly even for the ~700k EMNIST images.
    """
    if images_uint8.dtype != torch.uint8:
        raise TypeError(f"compute_mean_std expects uint8 images, got {images_uint8.dtype}")
    if images_uint8.numel() == 0:
        raise ValueError("Cannot compute statistics of an empty tensor")
    total = 0
    total_sq = 0
    for start in range(0, images_uint8.shape[0], chunk_size):
        chunk = images_uint8[start : start + chunk_size].to(torch.int64)
        total += int(chunk.sum())
        total_sq += int((chunk * chunk).sum())
    count = images_uint8.numel()
    mean = total / (255.0 * count)
    variance = max(total_sq / (255.0**2 * count) - mean * mean, 0.0)
    return mean, math.sqrt(variance)


def save_sample_grid(
    images_uint8: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray | Sequence[int],
    labels: Sequence[str],
    path: Path | str,
    *,
    n: int = 40,
    ncols: int = 10,
    seed: int = 0,
    title: str | None = None,
) -> Path:
    """Save a grid of ``n`` randomly chosen (seeded) images, each titled with its decoded label."""
    from matplotlib.figure import Figure  # no pyplot: avoids global backend state (headless-safe)

    images = images_uint8.cpu().numpy() if isinstance(images_uint8, torch.Tensor) else np.asarray(images_uint8)
    target_array = targets.cpu().numpy() if isinstance(targets, torch.Tensor) else np.asarray(targets)
    if images.ndim == 4:
        images = images[:, 0]
    count = min(n, images.shape[0])
    chosen = np.random.default_rng(seed).choice(images.shape[0], size=count, replace=False)
    ncols = max(1, min(ncols, count))
    nrows = math.ceil(count / ncols)

    fig = Figure(figsize=(ncols * 1.1, nrows * 1.25 + (0.5 if title else 0.0)), dpi=100)
    axes = np.atleast_1d(fig.subplots(nrows, ncols, squeeze=False)).ravel()
    for axis in axes:
        axis.axis("off")
    for axis, index in zip(axes, chosen):
        axis.imshow(images[index], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axis.set_title(labels[int(target_array[index])], fontsize=10)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    logger.info("Saved sample grid to %s", path)
    return path
