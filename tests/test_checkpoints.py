"""Checkpoint save/load: weights-only safety, JSON-safe metadata, atomic writes and clear errors."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import torch

from src.checkpoints import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointError,
    CheckpointNotFoundError,
    load_checkpoint,
    load_model_from_checkpoint,
    sanitize_metadata,
    save_checkpoint,
    update_checkpoint_metadata,
)
from src.class_mapping import CLASS_MAPPING_VERSION, get_labels
from src.models import DigitCNN
from src.preprocessing import PreprocessConfig
from tests.conftest import create_random_checkpoint

PAYLOAD_KEYS = {
    "format_version", "model_state_dict", "model_config", "num_classes", "dataset", "dataset_display_name", "mode",
    "class_labels", "class_mapping_version", "preprocessing", "training_config", "val_metrics", "test_metrics",
    "best_epoch", "created_at", "torch_version",
}


def metadata_without_weights(checkpoint: dict) -> dict:
    return {key: value for key, value in checkpoint.items() if key != "model_state_dict"}


@pytest.fixture
def mnist_path(tmp_path: Path) -> Path:
    return create_random_checkpoint(tmp_path / "mnist.pth", "digits", val_metrics={"accuracy": 0.5, "loss": 1.0},
                                    best_epoch=3)


# --------------------------------------------------------------------------------------------
# Saving
# --------------------------------------------------------------------------------------------
def test_saved_checkpoint_loads_with_weights_only_and_has_all_keys(mnist_path: Path) -> None:
    raw = torch.load(mnist_path, map_location="cpu", weights_only=True)
    assert set(raw) == PAYLOAD_KEYS
    assert raw["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert raw["dataset"] == "mnist" and raw["mode"] == "digits" and raw["dataset_display_name"] == "MNIST"
    assert raw["num_classes"] == 10
    assert raw["class_labels"] == list(get_labels("mnist"))
    assert raw["class_mapping_version"] == CLASS_MAPPING_VERSION
    assert raw["best_epoch"] == 3
    assert raw["test_metrics"] is None  # nothing measured -> nothing recorded
    assert all(t.device.type == "cpu" for t in raw["model_state_dict"].values())
    datetime.fromisoformat(raw["created_at"])  # ISO-8601
    assert type(raw["torch_version"]) is str


def test_metadata_is_json_safe_primitives(mnist_path: Path) -> None:
    metadata = metadata_without_weights(torch.load(mnist_path, map_location="cpu", weights_only=True))
    json.dumps(metadata, allow_nan=False)  # raises for numpy scalars, Paths, NaN, ...
    preprocessing = metadata["preprocessing"]
    for key in ("mean", "std", "target_box", "centering", "stroke_width", "image_size"):
        assert key in preprocessing
    assert PreprocessConfig.from_metadata(preprocessing, "digits") == PreprocessConfig.for_dataset("digits")


def test_save_sanitizes_numpy_values_and_paths(tmp_path: Path) -> None:
    path = tmp_path / "model.pth"
    save_checkpoint(
        path, DigitCNN(10), dataset_key="mnist", labels=get_labels("mnist"),
        preprocessing=PreprocessConfig.for_dataset("mnist"),  # a PreprocessConfig is accepted too
        training_config={"lr": np.float32(0.001), "epochs": np.int64(3), "data_dir": tmp_path, "shape": (1, 28)},
        val_metrics={"accuracy": np.float64(0.25), "loss": float("nan")},
        best_epoch=np.int64(2),
    )
    metadata = metadata_without_weights(torch.load(path, weights_only=True))
    json.dumps(metadata, allow_nan=False)
    assert metadata["training_config"] == {"lr": pytest.approx(0.001), "epochs": 3, "data_dir": str(tmp_path),
                                           "shape": [1, 28]}
    assert metadata["val_metrics"] == {"accuracy": 0.25, "loss": None}
    assert metadata["best_epoch"] == 2 and type(metadata["best_epoch"]) is int


def test_save_creates_parent_dirs_and_leaves_no_temp_files(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "model.pth"
    create_random_checkpoint(path, "characters")
    update_checkpoint_metadata(path, test_metrics={"accuracy": 0.1})
    assert sorted(p.name for p in path.parent.iterdir()) == ["model.pth"]


def test_failed_write_keeps_the_previous_checkpoint_intact(tmp_path: Path, mnist_path: Path,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    """Atomic writes: a crash half-way through saving never corrupts the existing file or leaves temp files."""
    original_bytes = mnist_path.read_bytes()

    def crash_mid_write(payload: dict, handle) -> None:
        handle.write(b"partial checkpoint data")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", crash_mid_write)
    with pytest.raises(OSError, match="disk full"):
        update_checkpoint_metadata(mnist_path, test_metrics={"accuracy": 0.5})
    with pytest.raises(OSError):
        create_random_checkpoint(mnist_path, "digits")
    monkeypatch.undo()

    assert mnist_path.read_bytes() == original_bytes
    assert [p.name for p in tmp_path.iterdir()] == [mnist_path.name]
    assert load_checkpoint(mnist_path)["test_metrics"] is None


def test_save_rejects_labels_that_do_not_match_the_model_or_dataset(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        save_checkpoint(tmp_path / "a.pth", DigitCNN(10), dataset_key="mnist", labels=list("012345678"),
                        preprocessing={})
    with pytest.raises(ValueError):  # 62-class model but MNIST labels
        save_checkpoint(tmp_path / "b.pth", DigitCNN(62), dataset_key="mnist", labels=get_labels("mnist"),
                        preprocessing={})
    with pytest.raises(ValueError):  # 10 labels/outputs for a 62-class dataset
        save_checkpoint(tmp_path / "c.pth", DigitCNN(10), dataset_key="emnist_byclass", labels=get_labels("mnist"),
                        preprocessing={})
    assert not any(tmp_path.iterdir())


# --------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode, num_classes", [("digits", 10), ("characters", 62)])
def test_load_model_reproduces_the_saved_model(tmp_path: Path, mode: str, num_classes: int) -> None:
    torch.manual_seed(0)
    original = DigitCNN(num_classes=num_classes).eval()
    path = tmp_path / "model.pth"
    save_checkpoint(path, original, dataset_key=mode, labels=get_labels(mode),
                    preprocessing=PreprocessConfig.for_dataset(mode).to_metadata())

    model, metadata = load_model_from_checkpoint(path, device="cpu")
    assert not model.training
    assert "model_state_dict" not in metadata
    assert metadata["num_classes"] == num_classes == len(metadata["class_labels"])
    x = torch.randn(4, 1, 28, 28)
    with torch.inference_mode():
        torch.testing.assert_close(model(x), original(x), rtol=0, atol=0)


def test_load_checkpoint_fills_missing_optional_keys(tmp_path: Path, mnist_path: Path) -> None:
    raw = torch.load(mnist_path, weights_only=True)
    for key in ("test_metrics", "training_config", "created_at"):
        del raw[key]
    path = tmp_path / "minimal.pth"
    torch.save(raw, path)
    checkpoint = load_checkpoint(path)
    assert checkpoint["test_metrics"] is None and checkpoint["training_config"] is None


def test_missing_checkpoint_raises_not_found(tmp_path: Path) -> None:
    with pytest.raises(CheckpointNotFoundError, match="not found"):
        load_checkpoint(tmp_path / "nope.pth")
    with pytest.raises(CheckpointNotFoundError):
        load_model_from_checkpoint(tmp_path / "nope.pth")


def test_corrupt_file_raises_checkpoint_error(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.pth"
    path.write_bytes(b"this is not a checkpoint" * 10)
    with pytest.raises(CheckpointError, match="corrupt"):
        load_checkpoint(path)


def test_truncated_file_raises_checkpoint_error(tmp_path: Path, mnist_path: Path) -> None:
    path = tmp_path / "truncated.pth"
    path.write_bytes(mnist_path.read_bytes()[:2000])
    with pytest.raises(CheckpointError):
        load_model_from_checkpoint(path)


def test_pickled_python_objects_are_refused(tmp_path: Path, mnist_path: Path) -> None:
    """weights_only loading must reject arbitrary objects (e.g. a pickled Path) instead of unpickling them."""
    raw = torch.load(mnist_path, weights_only=True)
    raw["training_config"] = {"data_dir": Path("/tmp")}
    path = tmp_path / "unsafe.pth"
    torch.save(raw, path)
    with pytest.raises(CheckpointError):
        load_checkpoint(path)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda c: c.pop("model_state_dict"), "missing required keys"),
        (lambda c: c.pop("class_labels"), "missing required keys"),
        (lambda c: c.update(format_version=99), "format_version"),
        (lambda c: c.update(dataset="cifar10"), "unknown dataset"),
    ],
)
def test_invalid_contents_raise_checkpoint_error(tmp_path: Path, mnist_path: Path, mutate, message: str) -> None:
    raw = torch.load(mnist_path, weights_only=True)
    mutate(raw)
    path = tmp_path / "bad.pth"
    torch.save(raw, path)
    with pytest.raises(CheckpointError, match=message):
        load_model_from_checkpoint(path)


def test_non_dict_checkpoint_raises_checkpoint_error(tmp_path: Path) -> None:
    path = tmp_path / "list.pth"
    torch.save([1, 2, 3], path)
    with pytest.raises(CheckpointError):
        load_checkpoint(path)


def test_inconsistent_labels_or_weights_raise_checkpoint_error(tmp_path: Path, mnist_path: Path) -> None:
    raw = torch.load(mnist_path, weights_only=True)
    raw["class_labels"] = raw["class_labels"][:9]
    torch.save(raw, tmp_path / "labels.pth")
    with pytest.raises(CheckpointError, match="inconsistent"):
        load_model_from_checkpoint(tmp_path / "labels.pth")

    raw = torch.load(mnist_path, weights_only=True)
    raw["model_config"]["hidden_units"] = 64  # weights no longer fit the architecture
    torch.save(raw, tmp_path / "weights.pth")
    with pytest.raises(CheckpointError, match="do not match"):
        load_model_from_checkpoint(tmp_path / "weights.pth")


# --------------------------------------------------------------------------------------------
# Updating metadata
# --------------------------------------------------------------------------------------------
def test_update_metadata_stores_json_safe_values_and_keeps_weights(mnist_path: Path) -> None:
    before = torch.load(mnist_path, weights_only=True)
    update_checkpoint_metadata(mnist_path, test_metrics={"accuracy": np.float64(0.125), "num_samples": np.int32(8)})
    after = torch.load(mnist_path, weights_only=True)
    assert after["test_metrics"] == {"accuracy": 0.125, "num_samples": 8}
    assert type(after["test_metrics"]["accuracy"]) is float
    assert "updated_at" in after
    for name, tensor in before["model_state_dict"].items():
        assert torch.equal(after["model_state_dict"][name], tensor)
    _, metadata = load_model_from_checkpoint(mnist_path)
    assert metadata["test_metrics"]["accuracy"] == 0.125


@pytest.mark.parametrize("field", ["model_state_dict", "format_version"])
def test_update_metadata_refuses_protected_keys(mnist_path: Path, field: str) -> None:
    with pytest.raises(ValueError):
        update_checkpoint_metadata(mnist_path, **{field: {}})


def test_update_metadata_of_missing_checkpoint_raises(tmp_path: Path) -> None:
    with pytest.raises(CheckpointNotFoundError):
        update_checkpoint_metadata(tmp_path / "missing.pth", test_metrics={})
    assert not (tmp_path / "missing.pth").exists()


# --------------------------------------------------------------------------------------------
# sanitize_metadata
# --------------------------------------------------------------------------------------------
def test_sanitize_metadata_produces_json_safe_values() -> None:
    raw = {
        "np_float": np.float32(0.5),
        "np_int": np.int64(7),
        "np_bool": np.bool_(True),
        "array": np.arange(3),
        "tensor": torch.tensor([1.5, 2.5]),
        "scalar_tensor": torch.tensor(3),
        "path": Path("models/x.pth"),
        "tuple": (1, (2, 3)),
        "nan": float("nan"),
        "inf": np.float64(math.inf),
        "nested": {1: {"deep": np.float16(0.25)}},
        "set": {3, 1, 2},
        "device": torch.device("cpu"),
        "version": torch.__version__,
    }
    clean = sanitize_metadata(raw)
    json.dumps(clean, allow_nan=False)
    assert clean == {
        "np_float": 0.5,
        "np_int": 7,
        "np_bool": True,
        "array": [0, 1, 2],
        "tensor": [1.5, 2.5],
        "scalar_tensor": 3,
        "path": "models/x.pth",
        "tuple": [1, [2, 3]],
        "nan": None,
        "inf": None,
        "nested": {"1": {"deep": 0.25}},
        "set": [1, 2, 3],
        "device": "cpu",
        "version": str(torch.__version__),
    }
    assert type(clean["version"]) is str  # torch's TorchVersion str subclass is refused by weights_only loading
    assert type(clean["np_float"]) is float and type(clean["np_int"]) is int


def test_sanitize_metadata_does_not_modify_its_input() -> None:
    raw = {"values": [np.float32(1.0)], "path": Path("a")}
    sanitize_metadata(raw)
    assert isinstance(raw["values"][0], np.float32) and isinstance(raw["path"], Path)
