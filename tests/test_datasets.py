"""Datasets: orientation, train/validation split, normalisation, augmentation and the in-memory loader.

Synthetic tensors are used everywhere except in the tests marked ``data``. Those read the real
MNIST / EMNIST files when they are present and are skipped otherwise; nothing is downloaded.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.class_mapping import get_labels
from src.config import DATA_DIR, EMNIST_BYCLASS_SPEC, MNIST_SPEC, get_spec
from src.datasets import (
    TensorBatchLoader,
    augment_batch,
    build_data,
    compute_mean_std,
    load_split,
    normalize_batch,
    open_torchvision_dataset,
    save_sample_grid,
    to_upright,
    train_val_split,
)
from tests.conftest import requires_emnist, requires_mnist


def asymmetric_images(n: int = 3) -> torch.Tensor:
    """uint8 (n, 28, 28) images with a unique value per pixel, so any transpose/flip is detectable."""
    base = torch.arange(28 * 28, dtype=torch.int64).reshape(28, 28) % 251
    return torch.stack([(base + i) % 256 for i in range(n)]).to(torch.uint8)


def synthetic_dataset(n: int = 50) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    images = torch.randint(0, 256, (n, 28, 28), generator=generator, dtype=torch.uint8)
    targets = torch.arange(n, dtype=torch.int64) % 10
    return images, targets


# --------------------------------------------------------------------------------------------
# Orientation
# --------------------------------------------------------------------------------------------
def test_to_upright_leaves_mnist_unchanged() -> None:
    images = asymmetric_images()
    assert torch.equal(to_upright(images, MNIST_SPEC), images)
    assert torch.equal(to_upright(images, "digits"), images)


def test_to_upright_transposes_emnist_exactly_once() -> None:
    images = asymmetric_images()
    upright = to_upright(images, EMNIST_BYCLASS_SPEC)
    assert upright.shape == images.shape
    for row, col in [(0, 5), (3, 27), (20, 1)]:
        assert torch.equal(upright[:, row, col], images[:, col, row])  # rows and columns swapped, no flip
    assert torch.equal(to_upright(upright, "characters"), images)  # a transpose is its own inverse
    assert upright.is_contiguous()


def test_to_upright_handles_single_images_channel_dims_and_numpy() -> None:
    images = asymmetric_images()
    single = to_upright(images[0], "characters")
    assert torch.equal(single, images[0].T)
    with_channel = to_upright(images.unsqueeze(1), "characters")
    assert torch.equal(with_channel[:, 0], images.transpose(1, 2))
    array = to_upright(images.numpy(), "characters")
    assert isinstance(array, np.ndarray) and np.array_equal(array, images.numpy().transpose(0, 2, 1))


# --------------------------------------------------------------------------------------------
# Train / validation split
# --------------------------------------------------------------------------------------------
def test_train_val_split_is_disjoint_complete_and_seeded() -> None:
    train_idx, val_idx = train_val_split(1000, val_fraction=0.1, seed=42)
    assert len(val_idx) == 100 and len(train_idx) == 900
    assert set(train_idx.tolist()).isdisjoint(val_idx.tolist())
    assert sorted(train_idx.tolist() + val_idx.tolist()) == list(range(1000))
    again_train, again_val = train_val_split(1000, val_fraction=0.1, seed=42)
    assert torch.equal(train_idx, again_train) and torch.equal(val_idx, again_val)
    assert not torch.equal(val_idx, train_val_split(1000, val_fraction=0.1, seed=7)[1])


def test_train_val_split_gives_each_side_at_least_one_sample() -> None:
    train_idx, val_idx = train_val_split(2, val_fraction=0.01)
    assert len(train_idx) == 1 and len(val_idx) == 1
    train_idx, val_idx = train_val_split(10, val_fraction=0.99)
    assert len(train_idx) == 1 and len(val_idx) == 9


@pytest.mark.parametrize("num_samples, fraction", [(100, 0.0), (100, 1.0), (100, -0.1), (1, 0.5)])
def test_train_val_split_rejects_bad_arguments(num_samples: int, fraction: float) -> None:
    with pytest.raises(ValueError):
        train_val_split(num_samples, val_fraction=fraction)


# --------------------------------------------------------------------------------------------
# Normalisation and augmentation
# --------------------------------------------------------------------------------------------
def test_normalize_batch_scales_uint8_and_applies_mean_std() -> None:
    x = torch.tensor([[0, 255, 51]], dtype=torch.uint8)
    out = normalize_batch(x, 0.1307, 0.3081)
    assert out.dtype == torch.float32
    expected = (torch.tensor([[0.0, 1.0, 0.2]]) - 0.1307) / 0.3081
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(normalize_batch(torch.tensor([[0.0, 1.0, 0.2]]), 0.1307, 0.3081), expected)


def test_augment_batch_keeps_shape_and_range_and_is_seeded() -> None:
    x = asymmetric_images(8).unsqueeze(1).float() / 255.0
    first = augment_batch(x, generator=torch.Generator().manual_seed(1))
    second = augment_batch(x, generator=torch.Generator().manual_seed(1))
    other = augment_batch(x, generator=torch.Generator().manual_seed(2))
    assert first.shape == x.shape and first.dtype == x.dtype
    assert first.min() >= 0.0 and first.max() <= 1.0
    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    assert not torch.equal(first, x)


def test_augment_batch_with_zero_ranges_is_the_identity() -> None:
    x = asymmetric_images(4).unsqueeze(1).float() / 255.0
    out = augment_batch(x, max_rotation=0.0, scale=(1.0, 1.0), max_shear=0.0, max_translate=0.0)
    torch.testing.assert_close(out, x, atol=1e-5, rtol=0)


def test_augment_batch_does_not_flip_or_transpose() -> None:
    """Small affine jitter keeps the orientation: a left-side bar stays on the left."""
    x = torch.zeros(16, 1, 28, 28)
    x[:, :, 4:24, 6:10] = 1.0  # vertical bar on the left
    out = augment_batch(x, generator=torch.Generator().manual_seed(0))
    left, right = out[..., :14].sum(dim=(1, 2, 3)), out[..., 14:].sum(dim=(1, 2, 3))
    assert (left > 5 * right).all()


def test_augment_batch_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        augment_batch(torch.zeros(2, 28, 28))
    with pytest.raises(TypeError):
        augment_batch(torch.zeros(2, 1, 28, 28, dtype=torch.uint8))


# --------------------------------------------------------------------------------------------
# TensorBatchLoader
# --------------------------------------------------------------------------------------------
def test_loader_yields_normalised_float_batches() -> None:
    images, targets = synthetic_dataset(50)
    loader = TensorBatchLoader(images, targets, 16, mean=0.5, std=0.25)
    assert len(loader) == 4 and loader.num_samples == 50
    batches = list(loader)
    assert [x.shape[0] for x, _ in batches] == [16, 16, 16, 2]
    x, y = batches[0]
    assert x.shape == (16, 1, 28, 28) and x.dtype == torch.float32
    assert y.dtype == torch.int64 and x.device.type == "cpu"
    torch.testing.assert_close(x[:, 0], (images[:16].float() / 255.0 - 0.5) / 0.25)
    assert torch.equal(torch.cat([y for _, y in batches]), targets)  # no shuffle: original order


def test_loader_shuffles_reproducibly_with_a_new_order_each_epoch() -> None:
    images, targets = synthetic_dataset(40)
    targets = torch.arange(40)  # unique labels reveal the order
    loader = TensorBatchLoader(images, targets, 8, mean=0.0, std=1.0, shuffle=True, seed=3)
    epoch1 = torch.cat([y for _, y in loader])
    epoch2 = torch.cat([y for _, y in loader])
    assert sorted(epoch1.tolist()) == list(range(40)) == sorted(epoch2.tolist())
    assert not torch.equal(epoch1, epoch2)
    fresh = TensorBatchLoader(images, targets, 8, mean=0.0, std=1.0, shuffle=True, seed=3)
    assert torch.equal(torch.cat([y for _, y in fresh]), epoch1)


def test_loader_drop_last_and_augmentation() -> None:
    images, targets = synthetic_dataset(50)
    loader = TensorBatchLoader(images, targets, 16, mean=0.0, std=1.0, drop_last=True, augment=True, seed=0)
    assert len(loader) == 3
    batches = list(loader)
    assert len(batches) == 3 and all(x.shape[0] == 16 for x, _ in batches)
    assert all(x.min() >= 0.0 and x.max() <= 1.0 for x, _ in batches)  # mean 0 / std 1: still [0, 1]


def test_loader_rejects_bad_input() -> None:
    images, targets = synthetic_dataset(10)
    with pytest.raises(TypeError):
        TensorBatchLoader(images.float(), targets, 4, mean=0.0, std=1.0)
    with pytest.raises(ValueError):
        TensorBatchLoader(images, targets[:5], 4, mean=0.0, std=1.0)
    with pytest.raises(ValueError):
        TensorBatchLoader(images, targets, 0, mean=0.0, std=1.0)


# --------------------------------------------------------------------------------------------
# Statistics, plots and loading errors
# --------------------------------------------------------------------------------------------
def test_compute_mean_std_matches_a_direct_computation() -> None:
    images, _ = synthetic_dataset(300)
    mean, std = compute_mean_std(images, chunk_size=64)
    reference = images.double() / 255.0
    assert mean == pytest.approx(reference.mean().item(), abs=1e-12)
    assert std == pytest.approx(reference.std(unbiased=False).item(), abs=1e-9)
    with pytest.raises(TypeError):
        compute_mean_std(images.float())


def test_save_sample_grid_writes_a_png(tmp_path: Path) -> None:
    images, targets = synthetic_dataset(30)
    path = save_sample_grid(images, targets, get_labels("digits"), tmp_path / "grids" / "samples.png", n=12,
                            ncols=6, title="Synthetic")
    assert path.is_file()
    with Image.open(path) as image:
        assert image.format == "PNG" and image.size[0] > 100


TRAIN_MARK, TEST_MARK = 11, 22  # pixel (0, 0) of every fake image tells which official split it came from


def fake_load_split(key_or_mode, data_dir=DATA_DIR, train: bool = True, download: bool = True):
    """Stand-in for load_split: tiny synthetic splits whose images are tagged with their split of origin."""
    spec = get_spec(key_or_mode)
    count = 120 if train else 50
    generator = torch.Generator().manual_seed(int(train))
    images = torch.randint(0, 256, (count, 28, 28), generator=generator, dtype=torch.uint8)
    images[:, 0, 0] = TRAIN_MARK if train else TEST_MARK
    return images, torch.arange(count, dtype=torch.int64) % spec.num_classes


def test_build_data_never_trains_or_validates_on_the_test_split(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validation is a subset of the official TRAIN split; the official TEST split feeds only the test loader."""
    import src.datasets

    monkeypatch.setattr(src.datasets, "load_split", fake_load_split)
    bundle = build_data("characters", batch_size=32, eval_batch_size=64, val_fraction=0.25, augment=False,
                        device="cpu")
    assert (bundle.num_train, bundle.num_val, bundle.num_test) == (90, 30, 50)

    def marks(loader: TensorBatchLoader) -> set[int]:
        first_pixels = torch.cat([x[:, 0, 0, 0] for x, _ in loader]) * get_spec("characters").std
        return set(torch.round((first_pixels + get_spec("characters").mean) * 255).int().tolist())

    assert marks(bundle.train_loader) == {TRAIN_MARK}
    assert marks(bundle.val_loader) == {TRAIN_MARK}
    assert marks(bundle.test_loader) == {TEST_MARK}


def test_missing_data_without_download_raises_a_helpful_error(tmp_path: Path) -> None:
    for mode in ("digits", "characters"):
        with pytest.raises(RuntimeError, match="not found"):
            load_split(mode, tmp_path, train=False, download=False)
    assert not any(tmp_path.iterdir()), "nothing may be created or downloaded"


# --------------------------------------------------------------------------------------------
# Real data (skipped when absent; never downloads)
# --------------------------------------------------------------------------------------------
def test_download_guard_blocks_any_dataset_download(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Safety net from conftest: even ``download=True`` cannot fetch a missing dataset during the tests."""
    caplog.set_level(logging.INFO, logger="src.datasets")
    with pytest.raises(RuntimeError, match="must never download"):
        load_split("digits", tmp_path, train=False, download=True)
    assert "Downloading/loading MNIST" in caplog.text  # the user is told before a (possible) download
    assert not (tmp_path / "MNIST" / "raw").exists() or not any((tmp_path / "MNIST" / "raw").iterdir())


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


@pytest.mark.data
@requires_mnist
def test_real_mnist_split_shapes_statistics_and_geometry() -> None:
    images, targets = load_split("digits", DATA_DIR, train=True, download=False)
    assert images.shape == (60_000, 28, 28) and images.dtype == torch.uint8
    assert targets.dtype == torch.int64 and int(targets.min()) == 0 and int(targets.max()) == 9
    mean, std = compute_mean_std(images)
    assert mean == pytest.approx(MNIST_SPEC.mean, abs=5e-4) and std == pytest.approx(MNIST_SPEC.std, abs=5e-4)

    sample = images[torch.randperm(60_000, generator=torch.Generator().manual_seed(0))[:2000]].numpy()
    ys, xs = np.mgrid[0:28, 0:28]
    weights = sample.astype(np.float64)
    com_x = (weights * xs).sum(axis=(1, 2)) / weights.sum(axis=(1, 2))
    com_y = (weights * ys).sum(axis=(1, 2)) / weights.sum(axis=(1, 2))
    assert com_x.mean() == pytest.approx(14.0, abs=0.15) and com_y.mean() == pytest.approx(14.0, abs=0.15)
    longest = [max(np.ptp(np.flatnonzero(im.any(0))), np.ptp(np.flatnonzero(im.any(1)))) + 1 for im in sample > 0]
    assert np.median(longest) == MNIST_SPEC.target_box


@pytest.mark.data
@requires_emnist
def test_real_emnist_is_stored_transposed_and_load_split_makes_it_upright() -> None:
    raw = open_torchvision_dataset("characters", DATA_DIR, train=False, download=False).data
    upright, targets = load_split("characters", DATA_DIR, train=False, download=False)
    assert upright.shape == (116_323, 28, 28) and upright.dtype == torch.uint8
    assert int(targets.min()) == 0 and int(targets.max()) == 61
    assert torch.equal(upright, raw.transpose(1, 2))

    # "I", "l" and "1" are tall vertical strokes once upright: the mean image varies far more along y than x.
    ys, xs = np.mgrid[0:28, 0:28]
    for label in ("1", "I", "l"):
        mean_image = upright[targets == get_labels("characters").index(label)][:3000].double().mean(0).numpy()
        weights = mean_image / mean_image.sum()
        var_x = ((xs - (xs * weights).sum()) ** 2 * weights).sum()
        var_y = ((ys - (ys * weights).sum()) ** 2 * weights).sum()
        assert var_y > 2.5 * var_x, f"{label!r} is not upright (var_x={var_x:.1f}, var_y={var_y:.1f})"


@pytest.mark.data
@requires_emnist
def test_real_emnist_geometry_and_statistics_match_the_spec() -> None:
    images, _ = load_split("characters", DATA_DIR, train=False, download=False)
    mean, std = compute_mean_std(images)  # test split; the spec values were measured on the train split
    assert mean == pytest.approx(get_spec("characters").mean, abs=2e-3)
    assert std == pytest.approx(get_spec("characters").std, abs=2e-3)

    sample = images[torch.randperm(len(images), generator=torch.Generator().manual_seed(0))[:2000]].numpy()
    masks = sample > 25  # > 0.1 * 255, as in the spec's measurement
    boxes = [(np.flatnonzero(m.any(0)), np.flatnonzero(m.any(1))) for m in masks if m.any()]
    centres = np.array([((c[0] + c[-1]) / 2, (r[0] + r[-1]) / 2) for c, r in boxes])
    longest = [max(c[-1] - c[0], r[-1] - r[0]) + 1 for c, r in boxes]
    assert centres.mean(axis=0) == pytest.approx((13.5, 13.5), abs=0.15)
    # Measured extent (incl. the blur halo) is 24 px; preprocessing deliberately targets 1 px less.
    assert np.median(longest) == 24 == EMNIST_BYCLASS_SPEC.target_box + 1


@pytest.mark.data
@requires_mnist
@requires_emnist
def test_upright_emnist_digits_look_like_mnist_digits() -> None:
    """Class-mean images of EMNIST digits correlate with MNIST's best in the upright orientation."""
    mnist, mnist_y = load_split("digits", DATA_DIR, train=False, download=False)
    emnist, emnist_y = load_split("characters", DATA_DIR, train=False, download=False)
    orientations = {
        "upright": lambda a: a,
        "raw (transposed)": lambda a: a.T,
        "horizontal flip": lambda a: a[:, ::-1],
        "vertical flip": lambda a: a[::-1],
        "rotated 180": lambda a: a[::-1, ::-1],
        "rotated 90": np.rot90,
    }
    scores: dict[str, list[float]] = {name: [] for name in orientations}
    for digit in range(10):
        mnist_mean = mnist[mnist_y == digit].double().mean(0).numpy()
        emnist_mean = emnist[emnist_y == digit][:3000].double().mean(0).numpy()
        for name, transform in orientations.items():
            scores[name].append(_correlation(mnist_mean, np.ascontiguousarray(transform(emnist_mean))))

    upright = np.array(scores["upright"])
    assert upright.mean() > 0.7
    for name, values in scores.items():
        if name != "upright":
            assert upright.mean() > np.mean(values) + 0.1, f"{name} scores as well as upright"
    assert (upright > np.array(scores["raw (transposed)"])).sum() >= 9


@pytest.mark.data
@requires_mnist
def test_build_data_uses_a_validation_subset_of_the_train_split() -> None:
    bundle = build_data("digits", data_dir=DATA_DIR, batch_size=64, eval_batch_size=128, max_train_samples=1000,
                        max_test_samples=200, val_fraction=0.1, seed=0, device="cpu", download=False)
    assert (bundle.num_train, bundle.num_val, bundle.num_test) == (900, 100, 200)
    assert sum(bundle.train_class_counts) == 900 and len(bundle.train_class_counts) == 10
    assert bundle.train_loader.shuffle and bundle.train_loader.augment
    for loader in (bundle.val_loader, bundle.test_loader):
        assert not loader.shuffle and not loader.augment
    x, y = next(iter(bundle.train_loader))
    assert x.shape == (64, 1, 28, 28) and y.dtype == torch.int64
    background = (0.0 - MNIST_SPEC.mean) / MNIST_SPEC.std
    assert x.min().item() == pytest.approx(background, abs=1e-5)
    without_test = build_data("digits", data_dir=DATA_DIR, max_train_samples=100, include_test=False, download=False)
    assert without_test.test_loader is None and without_test.num_test == 0
