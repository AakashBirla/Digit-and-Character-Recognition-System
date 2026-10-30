"""Training: one-epoch helpers on a tiny synthetic CPU dataset, configuration/CLI parsing, and a fit() smoke run.

The synthetic task (a bright square in one of four quadrants) is learnable in a few epochs on the
CPU, which makes "training reduces the loss" a fast, deterministic test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import train_emnist
import train_mnist
from src.checkpoints import load_checkpoint, load_model_from_checkpoint
from src.config import MODELS_DIR, RESULTS_DIR, get_spec
from src.datasets import TensorBatchLoader
from src.models import DigitCNN
from src.preprocessing import PreprocessConfig
from src.training import (
    TrainConfig,
    build_arg_parser,
    config_from_args,
    evaluate_loss_accuracy,
    fit,
    plot_training_curves,
    train_one_epoch,
)
from tests.conftest import requires_mnist

NUM_CLASSES = 4


def quadrant_dataset(samples_per_class: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """uint8 images with a bright 8x8 square in quadrant ``label`` plus faint noise."""
    generator = torch.Generator().manual_seed(seed)
    images = torch.randint(0, 40, (NUM_CLASSES * samples_per_class, 28, 28), generator=generator, dtype=torch.uint8)
    targets = torch.arange(NUM_CLASSES).repeat_interleave(samples_per_class)
    for index, label in enumerate(targets.tolist()):
        row, col = 3 + 14 * (label // 2), 3 + 14 * (label % 2)
        images[index, row:row + 8, col:col + 8] = 255
    return images, targets


def tiny_model() -> DigitCNN:
    torch.manual_seed(0)
    return DigitCNN(num_classes=NUM_CLASSES, conv_channels=(4, 8), hidden_units=16, dropout=0.2)


@pytest.fixture
def loaders() -> tuple[TensorBatchLoader, TensorBatchLoader]:
    train_images, train_targets = quadrant_dataset(32, seed=1)
    val_images, val_targets = quadrant_dataset(16, seed=2)
    train = TensorBatchLoader(train_images, train_targets, 16, mean=0.2, std=0.3, shuffle=True, seed=0)
    val = TensorBatchLoader(val_images, val_targets, 32, mean=0.2, std=0.3)
    return train, val


# --------------------------------------------------------------------------------------------
# One epoch
# --------------------------------------------------------------------------------------------
def test_train_one_epoch_returns_mean_loss_and_accuracy(loaders) -> None:
    train, _ = loaders
    model = tiny_model()
    before = [p.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss, accuracy = train_one_epoch(model, train, nn.CrossEntropyLoss(), optimizer, "cpu")
    assert isinstance(loss, float) and loss > 0 and torch.isfinite(torch.tensor(loss))
    assert isinstance(accuracy, float) and 0.0 <= accuracy <= 1.0
    assert model.training
    assert any(not torch.equal(b, p) for b, p in zip(before, model.parameters())), "weights must be updated"


def test_training_reduces_loss_and_learns_a_simple_task(loaders) -> None:
    train, val = loaders
    model = tiny_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    initial_val_loss, _ = evaluate_loss_accuracy(model, val, criterion, "cpu")
    losses = [train_one_epoch(model, train, criterion, optimizer, "cpu")[0] for _ in range(6)]
    val_loss, val_accuracy = evaluate_loss_accuracy(model, val, criterion, "cpu")
    assert losses[-1] < 0.5 * losses[0]
    assert val_loss < 0.5 * initial_val_loss
    assert val_accuracy >= 0.95


def test_evaluate_loss_accuracy_uses_eval_mode_and_no_gradients(loaders) -> None:
    _, val = loaders
    model = tiny_model()
    model.train()
    criterion = nn.CrossEntropyLoss()
    first = evaluate_loss_accuracy(model, val, criterion, "cpu")
    assert not model.training
    assert all(p.grad is None for p in model.parameters())
    assert torch.is_grad_enabled(), "the no-grad context must not leak"
    assert evaluate_loss_accuracy(model, val, criterion, "cpu") == first  # dropout off -> deterministic

    x = torch.cat([xb for xb, _ in val])
    y = torch.cat([yb for _, yb in val])
    with torch.no_grad():
        logits = model(x)
    assert first[0] == pytest.approx(nn.functional.cross_entropy(logits, y).item(), rel=1e-6)
    assert first[1] == pytest.approx((logits.argmax(1) == y).float().mean().item())


def test_epoch_helpers_accept_a_standard_dataloader() -> None:
    images, targets = quadrant_dataset(8, seed=3)
    dataset = TensorDataset((images.unsqueeze(1).float() / 255.0 - 0.2) / 0.3, targets)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(0))
    model = tiny_model()
    loss, accuracy = train_one_epoch(model, loader, nn.CrossEntropyLoss(), torch.optim.SGD(model.parameters(), 0.1),
                                     torch.device("cpu"))
    assert loss > 0 and 0 <= accuracy <= 1
    loss, accuracy = evaluate_loss_accuracy(model, loader, nn.CrossEntropyLoss(), "cpu")
    assert loss > 0 and 0 <= accuracy <= 1


def test_empty_loader_raises_value_error() -> None:
    model = tiny_model()
    with pytest.raises(ValueError):
        train_one_epoch(model, [], nn.CrossEntropyLoss(), torch.optim.SGD(model.parameters(), 0.1), "cpu")
    with pytest.raises(ValueError):
        evaluate_loss_accuracy(model, [], nn.CrossEntropyLoss(), "cpu")


def test_diverging_training_raises_a_clear_error() -> None:
    images, targets = quadrant_dataset(4, seed=0)
    batch = [((images.unsqueeze(1).float() / 255.0) * float("nan"), targets)]
    model = tiny_model()
    with pytest.raises(RuntimeError, match="diverged"):
        train_one_epoch(model, batch, nn.CrossEntropyLoss(), torch.optim.SGD(model.parameters(), 0.1), "cpu")


# --------------------------------------------------------------------------------------------
# Configuration and CLI
# --------------------------------------------------------------------------------------------
def test_train_config_defaults_follow_the_spec() -> None:
    config = TrainConfig(dataset="digits", epochs=10, batch_size=128)
    assert config.dataset == "mnist"
    assert (config.lr, config.weight_decay, config.val_fraction, config.patience, config.seed) == (
        1e-3, 0.0, 0.1, 4, 42)
    assert config.augment is True and config.balanced_loss is False and config.device == "auto"
    assert config.deterministic is False
    assert config.conv_channels == (32, 64) and config.hidden_units == 128 and config.dropout == 0.3
    assert config.checkpoint_path == get_spec("mnist").default_checkpoint == MODELS_DIR / "mnist_cnn.pth"
    assert config.results_path == RESULTS_DIR / "mnist"
    assert config.roundtrip_samples == 2000


def test_custom_output_keeps_artefacts_out_of_the_canonical_results(tmp_path: Path) -> None:
    """A smoke run with its own --output must not overwrite results/<dataset>/ of the real model."""
    smoke = TrainConfig(dataset="characters", epochs=1, batch_size=8, output=tmp_path / "smoke.pth")
    assert smoke.results_path == tmp_path / "smoke_results"
    explicit = TrainConfig(dataset="characters", epochs=1, batch_size=8, output=tmp_path / "smoke.pth",
                           results_dir=tmp_path / "r")
    assert explicit.results_path == tmp_path / "r"
    default_output = TrainConfig(dataset="characters", epochs=1, batch_size=8,
                                 output=get_spec("characters").default_checkpoint)
    assert default_output.results_path == RESULTS_DIR / "emnist_byclass"


@pytest.mark.parametrize(
    "overrides",
    [{"epochs": 0}, {"batch_size": 0}, {"lr": 0.0}, {"val_fraction": 1.0}, {"val_fraction": 0.0}, {"patience": -1},
     {"dropout": 1.0}, {"max_train_samples": 1}, {"dataset": "cifar10"}],
)
def test_train_config_rejects_invalid_values(overrides: dict) -> None:
    settings = {"dataset": "mnist", "epochs": 1, "batch_size": 8, **overrides}
    with pytest.raises(ValueError):
        TrainConfig(**settings)


@pytest.mark.parametrize("dataset, epochs, batch", [("mnist", 10, 128), ("emnist_byclass", 15, 256)])
def test_arg_parser_defaults(dataset: str, epochs: int, batch: int) -> None:
    args = build_arg_parser(dataset, default_epochs=epochs, default_batch_size=batch).parse_args([])
    config = config_from_args(dataset, args)
    assert (config.dataset, config.epochs, config.batch_size) == (dataset, epochs, batch)
    assert config.output is None and config.results_dir is None
    assert config.augment is True and config.roundtrip_samples == 2000


def test_arg_parser_flags_map_onto_the_config(tmp_path: Path) -> None:
    parser = build_arg_parser("emnist_byclass", default_epochs=15, default_batch_size=256)
    args = parser.parse_args([
        "--epochs", "3", "--batch-size", "64", "--lr", "0.005", "--weight-decay", "0.0001", "--val-fraction", "0.2",
        "--patience", "0", "--seed", "7", "--no-augment", "--balanced-loss", "--deterministic", "--device", "cpu",
        "--data-dir", str(tmp_path / "data"), "--output", str(tmp_path / "m.pth"), "--results-dir", str(tmp_path),
        "--max-train-samples", "500", "--max-test-samples", "100", "--eval-batch-size", "512",
        "--roundtrip-samples", "0",
    ])
    config = config_from_args("emnist_byclass", args)
    assert (config.epochs, config.batch_size, config.lr, config.weight_decay) == (3, 64, 0.005, 0.0001)
    assert (config.val_fraction, config.patience, config.seed) == (0.2, 0, 7)
    assert config.augment is False and config.balanced_loss is True and config.device == "cpu"
    assert config.deterministic is True
    assert config.data_dir == tmp_path / "data" and config.checkpoint_path == tmp_path / "m.pth"
    assert config.results_path == tmp_path
    assert (config.max_train_samples, config.max_test_samples, config.eval_batch_size) == (500, 100, 512)
    assert config.roundtrip_samples == 0


@pytest.mark.parametrize("argv", [["--epochs", "0"], ["--device", "tpu"], ["--val-fraction", "1.5"],
                                  ["--lr", "-1"], ["--batch-size", "many"]])
@pytest.mark.parametrize("script", [train_mnist, train_emnist], ids=["train_mnist", "train_emnist"])
def test_training_scripts_reject_bad_arguments(script, argv: list[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        script.main(argv)
    assert excinfo.value.code == 2


def test_plot_training_curves_writes_a_png(tmp_path: Path) -> None:
    history = [
        {"epoch": 1, "train_loss": 1.0, "train_acc": 0.6, "val_loss": 0.8, "val_acc": 0.7},
        {"epoch": 2, "train_loss": 0.6, "train_acc": 0.8, "val_loss": 0.5, "val_acc": 0.85},
    ]
    path = plot_training_curves(history, tmp_path / "plots" / "curves.png", title="Synthetic", best_epoch=2)
    assert path.is_file() and path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# --------------------------------------------------------------------------------------------
# fit() end to end on synthetic splits (no data needed)
# --------------------------------------------------------------------------------------------
def synthetic_digit_split(count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """10 classes; class k is a bright 6x6 square in slot k of a 3x4 grid, on faint noise."""
    generator = torch.Generator().manual_seed(seed)
    images = torch.randint(0, 30, (count, 28, 28), generator=generator, dtype=torch.uint8)
    targets = torch.arange(count) % 10
    for index, label in enumerate(targets.tolist()):
        row, col = 2 + 9 * (label // 4), 1 + 7 * (label % 4)
        images[index, row:row + 6, col:col + 6] = 255
    return images, targets


def test_fit_uses_the_test_split_only_for_the_final_evaluation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                               capsys: pytest.CaptureFixture) -> None:
    import src.datasets
    import src.evaluation

    requested: list[str] = []

    def fake_load_split(key_or_mode, data_dir=None, train: bool = True, download: bool = True):
        requested.append("train" if train else "test")
        return synthetic_digit_split(200 if train else 60, seed=int(train))

    monkeypatch.setattr(src.datasets, "load_split", fake_load_split)
    monkeypatch.setattr(src.evaluation, "load_split", fake_load_split)
    checkpoint = tmp_path / "mnist_synthetic.pth"
    config = TrainConfig(dataset="digits", epochs=2, batch_size=32, device="cpu", output=checkpoint,
                         results_dir=tmp_path / "results", data_dir=tmp_path / "no-data", roundtrip_samples=10,
                         eval_batch_size=64)
    summary = fit(config)

    assert requested == ["train", "test"], "training/validation must never load the official test split"
    assert "Using device: cpu" in capsys.readouterr().out
    metadata = load_checkpoint(checkpoint)
    training = metadata["training_config"]
    assert (training["num_train"], training["num_val"]) == (180, 20)  # validation carved out of the train split
    assert summary["test_metrics_recorded"] is True  # the whole (synthetic) test split was evaluated
    assert metadata["test_metrics"]["num_samples"] == 60 and metadata["test_metrics"]["split"] == "official_test"
    assert metadata["test_metrics"]["accuracy"] == pytest.approx(summary["test_metrics"]["accuracy"])


# --------------------------------------------------------------------------------------------
# fit() smoke run on real MNIST (skipped when the data is absent; never downloads)
# --------------------------------------------------------------------------------------------
@pytest.mark.data
@pytest.mark.slow
@requires_mnist
def test_fit_smoke_run_on_a_small_mnist_subset(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    models_before = sorted(MODELS_DIR.iterdir())
    checkpoint = tmp_path / "models" / "mnist_smoke.pth"
    results = tmp_path / "results"
    config = TrainConfig(dataset="digits", epochs=2, batch_size=64, device="cpu", max_train_samples=600,
                         max_test_samples=300, roundtrip_samples=40, output=checkpoint, results_dir=results,
                         eval_batch_size=256)
    summary = fit(config)

    printed = capsys.readouterr().out
    assert "Using device: cpu" in printed and "Epoch 1/2" in printed and "Epoch 2/2" in printed
    json.dumps(summary, allow_nan=False)
    assert summary["epochs_completed"] == 2 and summary["best_epoch"] in (1, 2)
    assert 0.0 <= summary["best_val_accuracy"] <= 1.0
    assert summary["test_metrics_recorded"] is False, "a test subset must never be recorded as test accuracy"
    assert summary["roundtrip"]["samples"] == 40

    model, metadata = load_model_from_checkpoint(checkpoint)
    assert model.num_classes == 10 and metadata["dataset"] == "mnist"
    assert metadata["test_metrics"] is None
    assert metadata["best_epoch"] == summary["best_epoch"]
    assert metadata["val_metrics"]["accuracy"] == pytest.approx(summary["best_val_accuracy"])
    assert PreprocessConfig.from_metadata(metadata["preprocessing"], "digits") == PreprocessConfig.for_dataset("digits")
    training = metadata["training_config"]
    assert training["device"] == "cpu" and training["epochs_completed"] == 2
    assert training["num_train"] + training["num_val"] == 600 and training["num_val"] == 60

    history = json.loads((results / "history.json").read_text())
    assert [row["epoch"] for row in history["epochs"]] == [1, 2]
    assert {"train_loss", "train_acc", "val_loss", "val_acc", "lr", "seconds"} <= set(history["epochs"][0])
    test_metrics = json.loads((results / "test_metrics.json").read_text())
    assert test_metrics["full_test_split"] is False and test_metrics["num_samples"] == 300
    for name in ("training_curves.png", "classification_report.txt", "per_class_metrics.csv",
                 "confusion_matrix.csv", "confusion_matrix.png", "roundtrip.json"):
        assert (results / name).is_file(), name
    assert load_checkpoint(checkpoint)["format_version"] == 1
    assert sorted(MODELS_DIR.iterdir()) == models_before, "fit() must only write where it was told to"
