"""Shared fixtures and helpers for the test suite.

The suite never downloads data and never needs trained models:

* checkpoints are randomly initialised ``DigitCNN`` models saved with ``save_checkpoint`` into a
  temporary directory (fixed seed, so every run sees the same weights);
* the web API is exercised with Flask's test client and an injected ``Predictor(device="cpu")``;
* drawings are synthesised with OpenCV on a 400x400 black canvas - the same size and polarity
  as the browser canvas - and encoded as PNG bytes or data URLs.

Tests that read the real MNIST / EMNIST files are marked ``data`` and are skipped when the files
are absent (``requires_mnist`` / ``requires_emnist`` / ``requires_emnist_mapping_zip``). As a
safety net, any attempt by torchvision to download a dataset during the test session fails.
"""

from __future__ import annotations

import base64
import io
import logging
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from app import create_app
from src.checkpoints import save_checkpoint
from src.class_mapping import get_labels
from src.config import DATA_DIR, get_spec
from src.models import DigitCNN
from src.predictor import Predictor
from src.preprocessing import PreprocessConfig

# --------------------------------------------------------------------------------------------
# Real-data availability (tests never download anything)
# --------------------------------------------------------------------------------------------
MNIST_RAW_DIR = DATA_DIR / "MNIST" / "raw"
EMNIST_RAW_DIR = DATA_DIR / "EMNIST" / "raw"
MNIST_FILES = tuple(
    MNIST_RAW_DIR / f"{prefix}-{kind}"
    for prefix in ("train", "t10k")
    for kind in ("images-idx3-ubyte", "labels-idx1-ubyte")
)
EMNIST_FILES = tuple(
    EMNIST_RAW_DIR / f"emnist-byclass-{split}-{kind}"
    for split in ("train", "test")
    for kind in ("images-idx3-ubyte", "labels-idx1-ubyte")
)
EMNIST_MAPPING_ZIP = EMNIST_RAW_DIR / "gzip.zip"


def _all_files_exist(paths: tuple[Path, ...]) -> bool:
    return all(path.is_file() for path in paths)


MNIST_AVAILABLE = _all_files_exist(MNIST_FILES)
EMNIST_AVAILABLE = _all_files_exist(EMNIST_FILES)

requires_mnist = pytest.mark.skipif(
    not MNIST_AVAILABLE, reason="MNIST files not found in data/MNIST/raw (the tests never download data)"
)
requires_emnist = pytest.mark.skipif(
    not EMNIST_AVAILABLE, reason="EMNIST ByClass files not found in data/EMNIST/raw (the tests never download data)"
)
requires_emnist_mapping_zip = pytest.mark.skipif(
    not EMNIST_MAPPING_ZIP.is_file(), reason="data/EMNIST/raw/gzip.zip not found (the tests never download data)"
)
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")


@pytest.fixture(autouse=True)
def _restore_root_log_level() -> Iterator[None]:
    """CLI entry points call ``setup_logging()``; keep their root-level change from leaking into other tests."""
    root = logging.getLogger()
    level = root.level
    yield
    root.setLevel(level)


@pytest.fixture(autouse=True, scope="session")
def _forbid_dataset_downloads() -> Iterator[None]:
    """For the whole session, make torchvision fail loudly instead of downloading a missing dataset."""
    import torchvision

    def refuse_download(dataset: object) -> None:
        if dataset._check_exists():  # files present: torchvision would not download anything
            return
        raise RuntimeError("The test suite must never download datasets.")

    with pytest.MonkeyPatch.context() as patch:
        for dataset_class in (torchvision.datasets.MNIST, torchvision.datasets.EMNIST):
            patch.setattr(dataset_class, "download", refuse_download)
        yield


# --------------------------------------------------------------------------------------------
# Random-init checkpoints
# --------------------------------------------------------------------------------------------
CHECKPOINT_SEED = 1234


def create_random_checkpoint(path: Path, key_or_mode: str, *, seed: int = CHECKPOINT_SEED, **metadata) -> Path:
    """Save a randomly initialised DigitCNN for ``key_or_mode`` to ``path`` with valid metadata.

    The global torch RNG state is left untouched. Extra keyword arguments (e.g. ``test_metrics``,
    ``val_metrics``, ``best_epoch``) are passed to ``save_checkpoint``.
    """
    spec = get_spec(key_or_mode)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = DigitCNN(num_classes=spec.num_classes)
    metadata.setdefault("training_config", {"device": "cpu", "device_name": "cpu", "note": "random-init test fixture"})
    save_checkpoint(
        path,
        model,
        dataset_key=spec.key,
        labels=get_labels(spec.key),
        preprocessing=PreprocessConfig.for_dataset(spec).to_metadata(),
        **metadata,
    )
    return path


@pytest.fixture(scope="session")
def checkpoint_paths(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Random-init checkpoints for both modes: ``{"digits": ..., "characters": ...}`` (read-only)."""
    directory = tmp_path_factory.mktemp("checkpoints")
    return {
        "digits": create_random_checkpoint(directory / "mnist_cnn.pth", "digits"),
        "characters": create_random_checkpoint(directory / "emnist_byclass_cnn.pth", "characters"),
    }


@pytest.fixture(scope="session")
def mnist_checkpoint(checkpoint_paths: dict[str, Path]) -> Path:
    return checkpoint_paths["digits"]


@pytest.fixture(scope="session")
def emnist_checkpoint(checkpoint_paths: dict[str, Path]) -> Path:
    return checkpoint_paths["characters"]


@pytest.fixture(scope="session")
def predictor(checkpoint_paths: dict[str, Path]) -> Predictor:
    """A CPU predictor with both random-init models loaded (shared; do not mutate)."""
    return Predictor(checkpoint_paths, device="cpu")


# --------------------------------------------------------------------------------------------
# Flask test clients
# --------------------------------------------------------------------------------------------
@pytest.fixture
def app(predictor: Predictor):
    """The Flask app with the shared CPU predictor injected (no model loading from models/)."""
    return create_app(predictor)


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def make_client() -> Callable[[Mapping[str, Path]], object]:
    """Factory: a test client whose predictor loads the given ``{mode: checkpoint}`` paths on the CPU."""

    def factory(paths: Mapping[str, Path]):
        return create_app(Predictor(dict(paths), device="cpu")).test_client()

    return factory


# --------------------------------------------------------------------------------------------
# Synthetic drawings
# --------------------------------------------------------------------------------------------
CANVAS_SIZE = 400
DEFAULT_BOX = (120, 60, 280, 340)  # x0, y0, x1, y1: a tall glyph roughly in the middle of the canvas
GLYPH_SHAPES = ("one", "dash", "zero", "seven", "rect", "P")


def blank_canvas(size: int = CANVAS_SIZE, value: int = 0) -> np.ndarray:
    """A ``size`` x ``size`` uint8 canvas filled with ``value`` (0 = black, like the web canvas)."""
    return np.full((size, size), value, dtype=np.uint8)


def draw_glyph(
    shape: str = "seven",
    *,
    box: tuple[int, int, int, int] = DEFAULT_BOX,
    thickness: int = 22,
    size: int = CANVAS_SIZE,
) -> np.ndarray:
    """Draw a simple anti-aliased white glyph inside ``box`` on a black ``size`` x ``size`` canvas.

    Shapes: ``"one"`` (vertical bar), ``"dash"`` (horizontal bar), ``"zero"`` (ellipse), ``"seven"``
    (top bar + diagonal, top-heavy), ``"rect"`` (rectangle outline), ``"P"`` (stem + loop in the
    upper half; its centre of mass and bounding-box centre differ clearly). ``thickness=-1`` fills
    ``"zero"`` / ``"rect"``. Coordinates may extend beyond the canvas (the glyph is then clipped).
    """
    canvas = blank_canvas(size)
    x0, y0, x1, y1 = box
    xc, yc = (x0 + x1) // 2, (y0 + y1) // 2
    line = {"color": 255, "thickness": thickness, "lineType": cv2.LINE_AA}
    if shape == "one":
        cv2.line(canvas, (xc, y0), (xc, y1), **line)
    elif shape == "dash":
        cv2.line(canvas, (x0, yc), (x1, yc), **line)
    elif shape == "zero":
        cv2.ellipse(canvas, (xc, yc), ((x1 - x0) // 2, (y1 - y0) // 2), 0, 0, 360, **line)
    elif shape == "seven":
        points = np.array([(x0, y0), (x1, y0), (x0 + (x1 - x0) // 3, y1)], dtype=np.int32)
        cv2.polylines(canvas, [points], False, **line)
    elif shape == "rect":
        cv2.rectangle(canvas, (x0, y0), (x1, y1), **line)
    elif shape == "P":
        cv2.line(canvas, (x0, y0), (x0, y1), **line)
        cv2.rectangle(canvas, (x0, y0), (x1, yc), **line)
    else:
        raise ValueError(f"Unknown test glyph {shape!r}; choose from {GLYPH_SHAPES}")
    return canvas


def png_bytes(image: np.ndarray) -> bytes:
    """Encode a uint8 ``(H, W)`` (L), ``(H, W, 3)`` (RGB) or ``(H, W, 4)`` (RGBA) array as PNG bytes."""
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(image)).save(buffer, format="PNG")
    return buffer.getvalue()


def png_data_url(image: np.ndarray) -> str:
    """``"data:image/png;base64,..."`` for ``image`` - what the frontend sends to /api/predict."""
    return "data:image/png;base64," + base64.b64encode(png_bytes(image)).decode("ascii")


def ink_geometry(image28: np.ndarray, threshold: int = 0) -> dict:
    """Geometry of a 28x28 uint8 result: centre of mass, bounding box (``> threshold``) and its centre."""
    weights = image28.astype(np.float64)
    ys, xs = np.mgrid[0:image28.shape[0], 0:image28.shape[1]]
    total = weights.sum()
    mask = image28 > threshold
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return {
        "com": (float((xs * weights).sum() / total), float((ys * weights).sum() / total)),
        "bbox_centre": ((cols[0] + cols[-1]) / 2.0, (rows[0] + rows[-1]) / 2.0),
        "width": int(cols[-1] - cols[0] + 1),
        "height": int(rows[-1] - rows[0] + 1),
    }


@pytest.fixture
def digit_png() -> bytes:
    """PNG bytes of a hand-drawn-looking "7" on the 400x400 black canvas."""
    return png_bytes(draw_glyph("seven"))


@pytest.fixture
def digit_data_url() -> str:
    return png_data_url(draw_glyph("seven"))


@pytest.fixture
def blank_data_url() -> str:
    return png_data_url(blank_canvas())
