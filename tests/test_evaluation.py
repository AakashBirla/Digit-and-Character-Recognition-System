"""Evaluation: metrics (including classes that are never predicted), report artefacts, canvas round-trip,
and evaluate_checkpoint / evaluate.py on real data when it is present."""

from __future__ import annotations

import csv
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import evaluate
from src.checkpoints import load_checkpoint
from src.class_mapping import get_labels
from src.datasets import TensorBatchLoader
from src.evaluation import (
    canvas_roundtrip_accuracy,
    collect_predictions,
    compact_test_metrics,
    compute_metrics,
    evaluate_checkpoint,
    save_evaluation_artifacts,
    simulate_canvas_drawing,
)
from src.models import DigitCNN
from src.preprocessing import PreprocessConfig
from src.training import evaluate_loss_accuracy
from tests.conftest import create_random_checkpoint, requires_mnist

LABELS = ["a", "b", "c", "d"]


@pytest.fixture
def never_predicted_case() -> dict:
    """Class "c" has samples but is never predicted; class "d" has no samples and is never predicted."""
    y_true = [0, 0, 0, 1, 1, 2, 2, 2]
    y_pred = [0, 0, 1, 1, 1, 0, 1, 0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # zero_division=0: no UndefinedMetricWarning may be emitted
        metrics = compute_metrics(y_true, y_pred, LABELS, loss=0.75)
    return metrics


# --------------------------------------------------------------------------------------------
# compute_metrics
# --------------------------------------------------------------------------------------------
def test_metrics_with_a_never_predicted_class_are_zero_division_safe(never_predicted_case: dict) -> None:
    metrics = never_predicted_case
    assert metrics["num_samples"] == 8 and metrics["num_classes"] == 4
    assert metrics["accuracy"] == pytest.approx(4 / 8)
    assert metrics["loss"] == 0.75
    assert metrics["classes_never_predicted"] == ["c", "d"]
    assert metrics["classes_without_support"] == ["d"]
    per_class = {row["label"]: row for row in metrics["per_class"]}
    assert [row["class_index"] for row in metrics["per_class"]] == [0, 1, 2, 3]
    assert per_class["c"] == {"class_index": 2, "label": "c", "precision": 0.0, "recall": 0.0, "f1": 0.0,
                              "support": 3, "predicted_count": 0}
    assert per_class["d"]["precision"] == per_class["d"]["recall"] == per_class["d"]["f1"] == 0.0
    assert per_class["a"]["precision"] == pytest.approx(2 / 4) and per_class["a"]["recall"] == pytest.approx(2 / 3)
    assert per_class["b"]["precision"] == pytest.approx(2 / 4) and per_class["b"]["recall"] == pytest.approx(1.0)
    assert metrics["macro_precision"] == pytest.approx((0.5 + 0.5 + 0 + 0) / 4)
    assert metrics["macro_recall"] == pytest.approx((2 / 3 + 1 + 0 + 0) / 4)
    for key in ("macro_f1", "weighted_f1", "weighted_precision", "weighted_recall"):
        assert np.isfinite(metrics[key]) and 0.0 <= metrics[key] <= 1.0
    json.dumps(metrics, allow_nan=False)


def test_confusion_matrix_has_a_fixed_shape_even_for_absent_classes(never_predicted_case: dict) -> None:
    matrix = np.array(never_predicted_case["confusion_matrix"])
    assert matrix.shape == (4, 4)
    assert matrix.tolist() == [[2, 1, 0, 0], [0, 2, 0, 0], [2, 1, 0, 0], [0, 0, 0, 0]]
    assert never_predicted_case["top_confusions"] == [
        {"true": "c", "pred": "a", "count": 2},
        {"true": "a", "pred": "b", "count": 1},
        {"true": "c", "pred": "b", "count": 1},
    ]


def test_perfect_predictions() -> None:
    labels = list(get_labels("digits"))
    y = np.arange(10).repeat(3)
    metrics = compute_metrics(y, y, labels)
    assert metrics["accuracy"] == metrics["macro_f1"] == metrics["weighted_f1"] == 1.0
    assert metrics["classes_never_predicted"] == [] and metrics["top_confusions"] == []
    assert metrics["loss"] is None and metrics["case_insensitive_accuracy"] is None  # digits have no case


def test_case_insensitive_accuracy_for_emnist_letters() -> None:
    labels = list(get_labels("characters"))
    c_upper, c_lower, o_upper, zero = labels.index("C"), labels.index("c"), labels.index("O"), labels.index("0")
    y_true = [c_upper, c_lower, o_upper, zero]
    y_pred = [c_lower, c_lower, zero, zero]  # c/C confusion counts as correct ignoring case, O->0 does not
    metrics = compute_metrics(y_true, y_pred, labels)
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["case_insensitive_accuracy"] == pytest.approx(0.75)
    assert len(metrics["confusion_matrix"]) == 62


@pytest.mark.parametrize(
    "y_true, y_pred",
    [([0, 1], [0]), ([], []), ([0, 4], [0, 1]), ([0, -1], [0, 1])],
    ids=["length-mismatch", "empty", "label-out-of-range", "negative-label"],
)
def test_compute_metrics_rejects_bad_input(y_true: list[int], y_pred: list[int]) -> None:
    with pytest.raises(ValueError):
        compute_metrics(y_true, y_pred, LABELS)


def test_compact_test_metrics_keeps_only_the_headline_numbers(never_predicted_case: dict) -> None:
    compact = compact_test_metrics(never_predicted_case, evaluated_at="2026-01-01T00:00:00+00:00")
    assert set(compact) == {"accuracy", "loss", "macro_precision", "macro_recall", "macro_f1", "weighted_f1",
                            "num_samples", "evaluated_at", "split"}
    assert compact["split"] == "official_test" and compact["num_samples"] == 8


# --------------------------------------------------------------------------------------------
# Predictions and artefacts
# --------------------------------------------------------------------------------------------
def test_collect_predictions_matches_the_loss_and_accuracy_helper() -> None:
    torch.manual_seed(0)
    model = DigitCNN(num_classes=10, conv_channels=(4, 8), hidden_units=16)
    images = torch.randint(0, 256, (40, 28, 28), generator=torch.Generator().manual_seed(1), dtype=torch.uint8)
    targets = torch.arange(40) % 10
    loader = TensorBatchLoader(images, targets, 16, mean=0.1307, std=0.3081)
    result = collect_predictions(model, loader, "cpu")
    assert result["y_true"].dtype == np.int64 and result["y_pred"].dtype == np.int64
    assert result["y_true"].tolist() == targets.tolist()
    assert result["num_samples"] == 40 and result["y_pred"].shape == (40,)
    loss, accuracy = evaluate_loss_accuracy(model, loader, nn.CrossEntropyLoss(), "cpu")
    assert result["loss"] == pytest.approx(loss, rel=1e-6)
    assert float((result["y_pred"] == result["y_true"]).mean()) == pytest.approx(accuracy)


def test_save_evaluation_artifacts_writes_every_report(tmp_path: Path, never_predicted_case: dict) -> None:
    paths = save_evaluation_artifacts(never_predicted_case, LABELS, tmp_path / "out", title="Synthetic test")
    expected = {"test_metrics.json", "classification_report.txt", "per_class_metrics.csv", "confusion_matrix.csv",
                "confusion_matrix.png"}
    assert {Path(p).name for p in paths.values()} == expected
    assert {p.name for p in (tmp_path / "out").iterdir()} == expected

    saved = json.loads((tmp_path / "out" / "test_metrics.json").read_text())
    assert saved["accuracy"] == pytest.approx(0.5) and saved["confusion_matrix"] == never_predicted_case[
        "confusion_matrix"]
    report = (tmp_path / "out" / "classification_report.txt").read_text()
    assert "Synthetic test" in report and "Classes never predicted: c, d" in report
    for label in LABELS:  # one row per class, even for a class without support
        assert re.search(rf"^\s+{label}\s+\d\.\d{{4}}", report, re.MULTILINE), label
    with open(tmp_path / "out" / "per_class_metrics.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["label"] for row in rows] == LABELS
    with open(tmp_path / "out" / "confusion_matrix.csv", newline="") as handle:
        matrix_rows = list(csv.reader(handle))
    assert matrix_rows[0][1:] == LABELS and len(matrix_rows) == 5
    assert (tmp_path / "out" / "confusion_matrix.png").read_bytes()[:4] == b"\x89PNG"


def test_save_evaluation_artifacts_rejects_a_wrongly_shaped_matrix(tmp_path: Path, never_predicted_case: dict) -> None:
    with pytest.raises(ValueError):
        save_evaluation_artifacts(never_predicted_case, LABELS[:3], tmp_path, title="bad")


def test_confusion_matrix_plot_handles_62_classes(tmp_path: Path) -> None:
    labels = list(get_labels("characters"))
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 62, 500)
    y_pred = np.where(rng.random(500) < 0.7, y_true, rng.integers(0, 62, 500))
    metrics = compute_metrics(y_true, y_pred, labels)
    paths = save_evaluation_artifacts(metrics, labels, tmp_path, title="EMNIST synthetic")
    assert Path(paths["confusion_matrix_png"]).stat().st_size > 10_000


# --------------------------------------------------------------------------------------------
# Canvas round-trip
# --------------------------------------------------------------------------------------------
def synthetic_glyph28(kind: int) -> np.ndarray:
    """A 28x28 uint8 MNIST-like glyph: 0 = vertical bar, 1 = horizontal bar, 2 = box outline."""
    glyph = np.zeros((28, 28), np.uint8)
    if kind == 0:
        glyph[4:24, 12:15] = 255
    elif kind == 1:
        glyph[12:15, 4:24] = 255
    else:
        glyph[6:22, 6:22] = 255
        glyph[9:19, 9:19] = 0
    return glyph


def test_simulate_canvas_drawing_produces_crisp_strokes_at_a_random_place() -> None:
    glyph = synthetic_glyph28(2)
    canvas = simulate_canvas_drawing(glyph, np.random.default_rng(0))
    assert canvas.shape == (400, 400) and canvas.dtype == np.uint8
    assert set(np.unique(canvas).tolist()) == {0, 255}
    rows, cols = np.flatnonzero(canvas.any(1)), np.flatnonzero(canvas.any(0))
    longest = max(rows[-1] - rows[0], cols[-1] - cols[0]) + 1
    assert 0.33 * 400 <= longest <= 0.77 * 400
    again = simulate_canvas_drawing(glyph, np.random.default_rng(0))
    assert np.array_equal(canvas, again)
    assert not np.array_equal(canvas, simulate_canvas_drawing(glyph, np.random.default_rng(1)))


def test_simulate_canvas_drawing_keeps_an_empty_glyph_empty() -> None:
    assert not simulate_canvas_drawing(np.zeros((28, 28), np.uint8), np.random.default_rng(0)).any()


def test_canvas_roundtrip_accuracy_reports_comparable_accuracies() -> None:
    images = np.stack([synthetic_glyph28(i % 3) for i in range(12)])
    targets = np.arange(12) % 3
    torch.manual_seed(0)
    model = DigitCNN(num_classes=10, conv_channels=(4, 8), hidden_units=16)
    result = canvas_roundtrip_accuracy(model, images, targets, PreprocessConfig.for_dataset("digits"), "cpu", n=9,
                                       seed=0)
    assert result["samples"] == 9 and result["seed"] == 0
    assert result["preprocessing_errors"] == 0
    assert 0.0 <= result["accuracy"] <= 1.0 and 0.0 <= result["direct_accuracy_same_samples"] <= 1.0
    again = canvas_roundtrip_accuracy(model, images, targets, PreprocessConfig.for_dataset("digits"), "cpu", n=9,
                                      seed=0)
    assert again == result


def test_canvas_roundtrip_counts_rejected_drawings_as_wrong() -> None:
    images = np.zeros((4, 28, 28), np.uint8)  # empty glyphs -> blank canvases -> rejected by preprocessing
    model = DigitCNN(num_classes=10)
    result = canvas_roundtrip_accuracy(model, images, np.zeros(4, np.int64), PreprocessConfig.for_dataset("digits"),
                                       "cpu", n=4, seed=0)
    assert result["preprocessing_errors"] == 4 and result["accuracy"] == 0.0


# --------------------------------------------------------------------------------------------
# evaluate.py command line (no data needed for these paths)
# --------------------------------------------------------------------------------------------
def test_evaluate_cli_reports_a_missing_checkpoint(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    code = evaluate.main(["--mode", "digits", "--checkpoint", str(tmp_path / "missing.pth"), "--device", "cpu"])
    assert code == 1
    assert "python train_mnist.py" in capsys.readouterr().err


def test_evaluate_cli_requires_a_mode_with_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        evaluate.main(["--checkpoint", str(tmp_path / "x.pth")])
    assert excinfo.value.code == 2


def test_evaluate_cli_rejects_a_checkpoint_for_the_other_dataset(tmp_path: Path, capsys) -> None:
    checkpoint = create_random_checkpoint(tmp_path / "emnist.pth", "characters")
    code = evaluate.main(["--mode", "digits", "--checkpoint", str(checkpoint), "--device", "cpu",
                          "--results-dir", str(tmp_path / "results")])
    assert code == 1
    assert "expects MNIST" in capsys.readouterr().err
    assert load_checkpoint(checkpoint)["test_metrics"] is None


# --------------------------------------------------------------------------------------------
# evaluate_checkpoint on the real MNIST test split (skipped when absent; never downloads)
# --------------------------------------------------------------------------------------------
@pytest.mark.data
@requires_mnist
def test_partial_evaluation_is_never_recorded_but_a_full_one_is(tmp_path: Path) -> None:
    checkpoint = create_random_checkpoint(tmp_path / "mnist.pth", "digits")

    partial = evaluate_checkpoint(checkpoint, device="cpu", results_dir=tmp_path / "partial", max_test_samples=500,
                                  update_checkpoint=True)
    assert partial["full_test_split"] is False and partial["recorded_in_checkpoint"] is False
    assert partial["metrics"]["num_samples"] == 500 and partial["metrics"]["test_split_size"] == 10_000
    assert load_checkpoint(checkpoint)["test_metrics"] is None

    full = evaluate_checkpoint(checkpoint, device="cpu", results_dir=tmp_path / "full", roundtrip_samples=30)
    assert full["full_test_split"] is True and full["recorded_in_checkpoint"] is True
    stored = load_checkpoint(checkpoint)["test_metrics"]
    assert stored["split"] == "official_test" and stored["num_samples"] == 10_000
    assert stored["accuracy"] == pytest.approx(full["metrics"]["accuracy"])
    assert full["roundtrip"]["samples"] == 30 and (tmp_path / "full" / "roundtrip.json").is_file()
    assert (tmp_path / "full" / "confusion_matrix.png").is_file()
    json.dumps(full["metrics"], allow_nan=False)

    # Re-evaluating without the round-trip removes the now outdated roundtrip.json.
    evaluate_checkpoint(checkpoint, device="cpu", results_dir=tmp_path / "full", roundtrip_samples=0,
                        update_checkpoint=False)
    assert not (tmp_path / "full" / "roundtrip.json").exists()


@pytest.mark.data
@requires_mnist
def test_subset_evaluation_defaults_to_a_separate_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without --results-dir a quick subset check must not overwrite the full-split report."""
    monkeypatch.setattr("src.evaluation.RESULTS_DIR", tmp_path)
    checkpoint = create_random_checkpoint(tmp_path / "mnist.pth", "digits")
    result = evaluate_checkpoint(checkpoint, device="cpu", max_test_samples=200, update_checkpoint=False)
    assert Path(result["results_dir"]) == tmp_path / "mnist" / "subset"
    assert (tmp_path / "mnist" / "subset" / "test_metrics.json").is_file()
    assert not (tmp_path / "mnist" / "test_metrics.json").exists()
