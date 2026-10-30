"""Configuration (dataset specs, modes, paths) and small utilities (device selection, seeding, logging)."""

from __future__ import annotations

import logging
import random

import numpy as np
import pytest
import torch

from src.config import (
    DATA_DIR,
    DATASET_SPECS,
    EMNIST_BYCLASS_SPEC,
    FRONTEND_DIR,
    MNIST_SPEC,
    MODE_TO_DATASET,
    MODELS_DIR,
    MODES,
    PROJECT_ROOT,
    RESULTS_DIR,
    get_spec,
)
from src.utils import describe_device, resolve_device, set_seed, setup_logging


def test_paths_derive_from_the_project_root() -> None:
    assert (PROJECT_ROOT / "src" / "config.py").is_file()
    assert DATA_DIR == PROJECT_ROOT / "data" and MODELS_DIR == PROJECT_ROOT / "models"
    assert RESULTS_DIR == PROJECT_ROOT / "results" and FRONTEND_DIR == PROJECT_ROOT / "frontend"
    assert (FRONTEND_DIR / "index.html").is_file()


def test_dataset_specs_record_the_measured_dataset_facts() -> None:
    assert MODES == ("digits", "characters")
    assert MODE_TO_DATASET == {"digits": "mnist", "characters": "emnist_byclass"}
    assert set(DATASET_SPECS) == {"mnist", "emnist_byclass"}
    mnist, emnist = MNIST_SPEC, EMNIST_BYCLASS_SPEC
    assert (mnist.num_classes, mnist.mean, mnist.std, mnist.transpose_raw) == (10, 0.1307, 0.3081, False)
    assert (mnist.target_box, mnist.centering, mnist.stroke_width) == (20, "center_of_mass", 2.44)
    assert (emnist.num_classes, emnist.mean, emnist.std, emnist.transpose_raw) == (62, 0.1736, 0.3317, True)
    assert (emnist.target_box, emnist.centering, emnist.stroke_width) == (23, "bounding_box", 2.80)
    assert mnist.default_checkpoint == MODELS_DIR / "mnist_cnn.pth"
    assert emnist.default_checkpoint == MODELS_DIR / "emnist_byclass_cnn.pth"
    assert (mnist.results_subdir, emnist.results_subdir) == ("mnist", "emnist_byclass")
    assert (mnist.display_name, emnist.display_name) == ("MNIST", "EMNIST ByClass")


@pytest.mark.parametrize("name, key", [("digits", "mnist"), ("mnist", "mnist"), ("characters", "emnist_byclass"),
                                       ("emnist_byclass", "emnist_byclass"), (" Characters ", "emnist_byclass")])
def test_get_spec_accepts_modes_and_dataset_keys(name: str, key: str) -> None:
    assert get_spec(name).key == key


@pytest.mark.parametrize("name", ["letters", "emnist", "", "MNIST-digits", None, 10])
def test_get_spec_rejects_unknown_names(name: object) -> None:
    with pytest.raises(ValueError):
        get_spec(name)


def test_resolve_device_auto_and_cpu() -> None:
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device("auto").type == expected
    assert resolve_device(None).type == expected
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device(torch.device("cpu")) == torch.device("cpu")


def test_resolve_device_explains_missing_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == torch.device("cpu")
    with pytest.raises(ValueError, match="CUDA was requested but is not available"):
        resolve_device("cuda")


@pytest.mark.parametrize("name", ["tpu", "gpu", "cuda:x", 3])
def test_resolve_device_rejects_unknown_devices(name: object) -> None:
    with pytest.raises(ValueError):
        resolve_device(name)


def test_describe_device_for_cpu() -> None:
    assert describe_device("cpu") == "cpu"
    assert describe_device(torch.device("cpu")) == "cpu"


def test_set_seed_makes_random_streams_reproducible() -> None:
    set_seed(123)
    first = (random.random(), np.random.rand(), torch.rand(3))
    set_seed(123)
    second = (random.random(), np.random.rand(), torch.rand(3))
    assert first[0] == second[0] and first[1] == second[1] and torch.equal(first[2], second[2])


def test_setup_logging_can_be_called_repeatedly() -> None:
    """A second call only changes the level (the root level is restored by an autouse fixture)."""
    setup_logging(logging.INFO)
    handlers = list(logging.getLogger().handlers)
    setup_logging(logging.WARNING)
    assert logging.getLogger().level == logging.WARNING
    assert logging.getLogger().handlers == handlers
