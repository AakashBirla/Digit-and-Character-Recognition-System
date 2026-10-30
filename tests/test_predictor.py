"""Predictor: loading both models, health reporting, predictions and failure handling (CPU, random-init models)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.checkpoints import update_checkpoint_metadata
from src.class_mapping import EMNIST_BYCLASS_LABELS, MNIST_LABELS, label_category
from src.config import MODES, PROJECT_ROOT
from src.predictor import InvalidModeError, ModelUnavailableError, PredictionError, Predictor
from src.preprocessing import (
    BlankImageError,
    InvalidImageError,
    PreprocessConfig,
    decode_image_payload,
    preprocess_image,
)
from tests.conftest import (
    blank_canvas,
    create_random_checkpoint,
    draw_glyph,
    png_bytes,
    png_data_url,
    requires_cuda,
)

EXPECTED_LABELS = {"digits": MNIST_LABELS, "characters": EMNIST_BYCLASS_LABELS}
# Keys promised by the spec (implementations may add more).
PREDICTION_KEYS = {"mode", "dataset", "prediction", "class_index", "confidence", "category", "probabilities", "top_k",
                   "preprocessed_image", "processing_time_ms", "device"}
INFO_KEYS = {"mode", "dataset", "display_name", "num_classes", "status", "loaded", "error", "checkpoint",
             "architecture", "parameters", "test_metrics", "val_accuracy", "best_epoch", "trained_on", "created_at",
             "labels"}
HEALTH_KEYS = {"status", "device", "device_name", "models", "loaded_models", "torch_version"}


def assert_valid_prediction(result: dict, mode: str, top_k: int = 5) -> None:
    labels = EXPECTED_LABELS[mode]
    probabilities = result["probabilities"]
    assert PREDICTION_KEYS <= set(result)
    assert result["mode"] == mode
    assert [row["class_index"] for row in probabilities] == list(range(len(labels)))
    assert [row["label"] for row in probabilities] == list(labels)
    values = np.array([row["probability"] for row in probabilities])
    assert np.isfinite(values).all() and (values >= 0).all()
    assert values.sum() == pytest.approx(1.0, abs=1e-6)
    assert result["class_index"] == int(values.argmax())
    assert result["prediction"] == labels[result["class_index"]]
    assert result["confidence"] == pytest.approx(values.max())
    assert result["category"] == label_category(result["prediction"])
    top = result["top_k"]
    assert len(top) == min(top_k, len(labels))
    assert [row["probability"] for row in top] == sorted((row["probability"] for row in top), reverse=True)
    assert top[0]["class_index"] == result["class_index"]
    preview = decode_image_payload(result["preprocessed_image"])
    assert preview.size == (28, 28)
    assert isinstance(result["processing_time_ms"], float) and result["processing_time_ms"] >= 0
    assert result["processing_time_ms"] == round(result["processing_time_ms"], 2)
    assert result["device"] == "cpu"
    json.dumps(result, allow_nan=False)


# --------------------------------------------------------------------------------------------
# Loading and health
# --------------------------------------------------------------------------------------------
def test_both_models_load_on_cpu(predictor: Predictor) -> None:
    assert predictor.device == torch.device("cpu")
    for mode in MODES:
        entry = predictor.get_entry(mode)
        assert entry.status == "loaded" and entry.loaded and entry.error is None
        assert entry.model is not None and not entry.model.training
        assert next(entry.model.parameters()).device.type == "cpu"
        assert entry.labels == EXPECTED_LABELS[mode]
        assert entry.preprocess == PreprocessConfig.for_dataset(mode)


def test_health_is_json_serialisable_and_ok(predictor: Predictor) -> None:
    health = predictor.health()
    json.dumps(health, allow_nan=False)
    assert HEALTH_KEYS <= set(health)
    assert all(INFO_KEYS <= set(info) for info in health["models"].values())
    assert health["models"]["digits"]["checkpoint"].endswith("mnist_cnn.pth")
    assert health["torch_version"] == str(torch.__version__)
    assert health["status"] == "ok"
    assert health["device"] == "cpu" and health["device_name"] == "cpu"
    assert health["loaded_models"] == ["MNIST", "EMNIST ByClass"]
    assert set(health["models"]) == set(MODES)
    digits, characters = health["models"]["digits"], health["models"]["characters"]
    assert digits["num_classes"] == 10 and characters["num_classes"] == 62
    assert digits["parameters"] == 421_738 and characters["parameters"] == 428_446
    assert characters["labels"] == list(EMNIST_BYCLASS_LABELS)
    assert characters["architecture"].endswith("→ FC(62)")
    assert digits["trained_on"] == "cpu"
    for info in (digits, characters):
        assert info["status"] == "loaded" and info["loaded"] is True and info["error"] is None
        assert info["test_metrics"] is None, "random-init fixtures have no measured test metrics"
        assert info["val_accuracy"] is None


def test_test_metrics_are_reported_only_when_measured(tmp_path: Path) -> None:
    """Health shows test metrics only if the checkpoint stores a measured accuracy (never invented)."""
    measured = create_random_checkpoint(tmp_path / "measured.pth", "digits", val_metrics={"accuracy": 0.5})
    update_checkpoint_metadata(measured, test_metrics={"accuracy": 0.25, "num_samples": 4, "split": "official_test",
                                                       "confusion_matrix": [[1, 0], [0, 1]]})
    without_accuracy = create_random_checkpoint(tmp_path / "partial.pth", "characters",
                                                test_metrics={"note": "no accuracy here"})
    models = Predictor({"digits": measured, "characters": without_accuracy}, device="cpu").health()["models"]
    assert models["digits"]["test_metrics"] == {"accuracy": 0.25, "num_samples": 4, "split": "official_test"}
    assert models["digits"]["val_accuracy"] == 0.5
    assert models["characters"]["test_metrics"] is None


def test_missing_checkpoint_marks_mode_missing_and_health_degraded(tmp_path: Path, emnist_checkpoint: Path) -> None:
    missing = PROJECT_ROOT / "models" / "__pytest_does_not_exist__.pth"  # never created
    predictor = Predictor({"digits": missing, "characters": emnist_checkpoint}, device="cpu")
    entry = predictor.get_entry("digits")
    assert entry.status == "missing" and not entry.loaded and entry.model is None
    assert entry.error == ("Checkpoint not found: models/__pytest_does_not_exist__.pth. "
                           "Train it with: python train_mnist.py")
    health = predictor.health()
    json.dumps(health, allow_nan=False)
    assert health["status"] == "degraded"
    assert health["loaded_models"] == ["EMNIST ByClass"]
    assert health["models"]["digits"]["status"] == "missing"
    assert health["models"]["digits"]["labels"] == list(MNIST_LABELS)  # still known without a model
    assert health["models"]["characters"]["status"] == "loaded"

    with pytest.raises(ModelUnavailableError) as excinfo:
        predictor.predict(png_data_url(draw_glyph("seven")), "digits")
    assert excinfo.value.mode == "digits"
    assert "python train_mnist.py" in excinfo.value.reason
    assert_valid_prediction(predictor.predict(draw_glyph("P"), "characters"), "characters")


def test_default_checkpoint_paths_come_from_the_config() -> None:
    predictor = Predictor(device="cpu", load=False)
    assert predictor.checkpoint_paths == {"digits": PROJECT_ROOT / "models" / "mnist_cnn.pth",
                                          "characters": PROJECT_ROOT / "models" / "emnist_byclass_cnn.pth"}
    assert all(entry.status == "missing" for entry in predictor.entries.values())


@pytest.mark.parametrize("content", [b"not a checkpoint", b""])
def test_corrupt_checkpoint_marks_mode_error_without_raising(tmp_path: Path, mnist_checkpoint: Path,
                                                             content: bytes) -> None:
    broken = tmp_path / "broken.pth"
    broken.write_bytes(content)
    predictor = Predictor({"digits": mnist_checkpoint, "characters": broken}, device="cpu")
    entry = predictor.get_entry("characters")
    assert entry.status == "error" and not entry.loaded
    assert "python train_emnist.py" in entry.error
    assert "Traceback" not in entry.error
    assert predictor.health()["status"] == "degraded"
    assert predictor.get_entry("digits").loaded
    with pytest.raises(ModelUnavailableError):
        predictor.predict(draw_glyph("P"), "characters")


def test_checkpoint_for_the_wrong_dataset_is_rejected(mnist_checkpoint: Path, emnist_checkpoint: Path) -> None:
    swapped = Predictor({"digits": emnist_checkpoint, "characters": mnist_checkpoint}, device="cpu")
    for mode in MODES:
        entry = swapped.get_entry(mode)
        assert entry.status == "error"
        assert "trained on" in entry.error


def test_model_that_outputs_nan_fails_the_warm_up(tmp_path: Path) -> None:
    path = create_random_checkpoint(tmp_path / "nan.pth", "digits")
    raw = torch.load(path, weights_only=True)
    raw["model_state_dict"]["classifier.4.bias"][:] = float("nan")
    torch.save(raw, path)
    predictor = Predictor({"digits": path}, device="cpu")
    entry = predictor.get_entry("digits")
    assert entry.status == "error" and "test prediction" in entry.error
    with pytest.raises(ModelUnavailableError):
        predictor.predict(draw_glyph("one"), "digits")


def test_invalid_model_output_at_prediction_time_raises_prediction_error(
    checkpoint_paths: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    predictor = Predictor(checkpoint_paths, device="cpu")
    entry = predictor.get_entry("digits")
    monkeypatch.setattr(entry, "model", lambda x: torch.full((x.shape[0], 10), float("nan")))
    with pytest.raises(PredictionError) as excinfo:
        predictor.predict(draw_glyph("one"), "digits")
    assert excinfo.value.code == "prediction_error"


def test_checkpoint_with_wrong_label_order_is_rejected(tmp_path: Path) -> None:
    path = create_random_checkpoint(tmp_path / "labels.pth", "digits")
    raw = torch.load(path, weights_only=True)
    raw["class_labels"] = raw["class_labels"][::-1]
    torch.save(raw, path)
    entry = Predictor({"digits": path}, device="cpu").get_entry("digits")
    assert entry.status == "error" and "class labels" in entry.error


# --------------------------------------------------------------------------------------------
# Predictions
# --------------------------------------------------------------------------------------------
def test_digit_prediction_returns_10_probabilities(predictor: Predictor) -> None:
    assert_valid_prediction(predictor.predict(png_data_url(draw_glyph("seven")), "digits"), "digits")


def test_character_prediction_returns_62_labelled_probabilities(predictor: Predictor) -> None:
    result = predictor.predict(png_data_url(draw_glyph("P")), "characters")
    assert_valid_prediction(result, "characters")
    assert result["dataset"] == "emnist_byclass"


@pytest.mark.parametrize("top_k, expected", [(1, 1), (3, 3), (10, 10), (100, 62), (0, 1)])
def test_top_k_is_clamped_to_the_number_of_classes(predictor: Predictor, top_k: int, expected: int) -> None:
    result = predictor.predict(draw_glyph("zero"), "characters", top_k=top_k)
    assert len(result["top_k"]) == expected


def test_predict_accepts_every_supported_image_type(predictor: Predictor) -> None:
    glyph = draw_glyph("seven")
    reference = predictor.predict(glyph, "digits")["probabilities"]
    for image in (png_data_url(glyph), png_bytes(glyph), Image.fromarray(glyph), torch.from_numpy(glyph)):
        assert predictor.predict(image, "digits")["probabilities"] == reference


def test_predictions_are_deterministic(predictor: Predictor) -> None:
    url = png_data_url(draw_glyph("P"))
    first, second = predictor.predict(url, "characters"), predictor.predict(url, "characters")
    assert first["probabilities"] == second["probabilities"]
    assert first["preprocessed_image"] == second["preprocessed_image"]


def test_preview_is_the_preprocessed_28x28_image(predictor: Predictor) -> None:
    glyph = draw_glyph("zero")
    result = predictor.predict(glyph, "digits")
    preview = np.asarray(decode_image_payload(result["preprocessed_image"]))
    assert preview.shape == (28, 28) and preview.dtype == np.uint8
    assert np.array_equal(preview, preprocess_image(glyph, "digits").image)


def test_mode_is_case_insensitive_and_unknown_modes_are_rejected(predictor: Predictor) -> None:
    assert predictor.predict(draw_glyph("one"), " Digits ")["mode"] == "digits"
    for mode in ("letters", "", None, 3, "mnist"):
        with pytest.raises(InvalidModeError) as excinfo:
            predictor.predict(draw_glyph("one"), mode)
        assert excinfo.value.code == "invalid_mode"
        assert "digits" in excinfo.value.reason and "characters" in excinfo.value.reason


def test_blank_and_invalid_images_raise_preprocessing_errors(predictor: Predictor) -> None:
    with pytest.raises(BlankImageError):
        predictor.predict(png_data_url(blank_canvas()), "digits")
    with pytest.raises(InvalidImageError):
        predictor.predict("data:image/png;base64,AAAA", "characters")


def test_debug_dir_receives_the_latest_preprocessed_input(tmp_path: Path, checkpoint_paths: dict) -> None:
    predictor = Predictor(checkpoint_paths, device="cpu", debug_dir=tmp_path / "debug")
    result = predictor.predict(draw_glyph("seven"), "digits")
    predictor.predict(draw_glyph("P"), "characters")
    assert sorted(p.name for p in (tmp_path / "debug").iterdir()) == ["latest_characters.png", "latest_digits.png"]
    saved = np.asarray(Image.open(tmp_path / "debug" / "latest_digits.png"))
    assert np.array_equal(saved, np.asarray(decode_image_payload(result["preprocessed_image"])))


def test_probabilities_match_a_manual_forward_pass(predictor: Predictor) -> None:
    glyph = draw_glyph("zero")
    entry = predictor.get_entry("digits")
    with torch.inference_mode():
        expected = torch.softmax(entry.model(preprocess_image(glyph, entry.preprocess).tensor).double(), dim=1)[0]
    values = [row["probability"] for row in predictor.predict(glyph, "digits")["probabilities"]]
    assert all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(values, expected.tolist()))


@pytest.mark.gpu
@requires_cuda
def test_cuda_predictions_match_cpu_predictions(predictor: Predictor, checkpoint_paths: dict) -> None:
    gpu = Predictor(checkpoint_paths, device="cuda")
    assert gpu.health()["status"] == "ok" and gpu.health()["device"].startswith("cuda")
    for mode, shape in (("digits", "seven"), ("characters", "P")):
        on_gpu = gpu.predict(draw_glyph(shape), mode)
        on_cpu = predictor.predict(draw_glyph(shape), mode)
        assert on_gpu["device"].startswith("cuda")
        assert on_gpu["preprocessed_image"] == on_cpu["preprocessed_image"]
        for a, b in zip(on_gpu["probabilities"], on_cpu["probabilities"]):
            assert a["probability"] == pytest.approx(b["probability"], abs=1e-4)
