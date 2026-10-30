"""Flask API (Flask test client + injected CPU predictor): health, predictions, error codes, no leakage."""

from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch
from flask import Flask

import app as webapp
from app import create_app
from src.class_mapping import EMNIST_BYCLASS_LABELS, MNIST_LABELS
from src.config import PROJECT_ROOT
from src.predictor import Predictor
from tests.conftest import blank_canvas, draw_glyph, png_bytes, png_data_url

PREDICT_URL = "/api/predict"
MISSING_MNIST_CHECKPOINT = PROJECT_ROOT / "models" / "__pytest_missing_mnist_cnn__.pth"  # never created

# Signs of a leaked stack trace or server-side path in a response body.
LEAK_PATTERNS = (
    re.compile(r"Traceback"),
    re.compile(r'File "'),
    re.compile(re.escape(str(PROJECT_ROOT))),
    re.compile(r"(?:^|[\s'\"(=:])/(?!api/)(?:[\w.-]+/)+[\w.-]*"),  # an absolute filesystem path (not an /api/ URL)
    re.compile(r"\b\w+(?:Error|Exception)\b(?!\")"),  # exception class names
    re.compile(r"site-packages"),
)


def assert_no_leak(text: str, allowed_paths: tuple[Path, ...] = ()) -> None:
    """Fail if ``text`` contains a stack trace, exception name or server path.

    ``allowed_paths``: checkpoint paths the operator configured. The spec lets the "model unavailable"
    message name the checkpoint (relative to the project when possible), so only those are tolerated.
    """
    for path in allowed_paths:
        text = text.replace(str(path), "<configured checkpoint>")
    for pattern in LEAK_PATTERNS:
        assert not pattern.search(text), f"response leaks server details ({pattern.pattern!r}): {text[:300]}"


def assert_api_error(response, status: int, code: str, allowed_paths: tuple[Path, ...] = ()) -> dict:
    """Check the single JSON error shape and that nothing internal leaks; return the body."""
    assert response.status_code == status, response.get_data(as_text=True)[:300]
    assert response.mimetype == "application/json"
    body = response.get_json()
    assert body["success"] is False
    assert body["code"] == code
    assert isinstance(body["error"], str) and body["error"].strip()
    assert_no_leak(response.get_data(as_text=True), allowed_paths)
    return body


def assert_prediction(response, mode: str) -> dict:
    assert response.status_code == 200, response.get_data(as_text=True)[:300]
    body = response.get_json()
    labels = MNIST_LABELS if mode == "digits" else EMNIST_BYCLASS_LABELS
    assert body["success"] is True and body["mode"] == mode
    assert len(body["probabilities"]) == len(labels)
    assert [row["label"] for row in body["probabilities"]] == list(labels)
    total = sum(row["probability"] for row in body["probabilities"])
    assert total == pytest.approx(1.0, abs=1e-6)
    assert body["prediction"] in labels and body["prediction"] == body["top_k"][0]["label"]
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["top_k"]) == 5
    assert body["preprocessed_image"].startswith("data:image/png;base64,")
    assert body["device"] == "cpu"
    return body


# --------------------------------------------------------------------------------------------
# App factory and command line
# --------------------------------------------------------------------------------------------
def test_injected_predictor_is_stored_on_the_app(predictor) -> None:
    app = create_app(predictor)
    assert webapp.get_predictor(app) is predictor
    assert app.extensions["predictor"] is predictor
    assert app.config["MAX_CONTENT_LENGTH"] == 8 * 1024 * 1024


def test_create_app_builds_its_own_predictor_once(checkpoint_paths: dict) -> None:
    app = create_app(device="cpu", checkpoint_paths=checkpoint_paths)
    predictor = webapp.get_predictor(app)
    assert isinstance(predictor, Predictor) and predictor.device.type == "cpu"
    assert predictor.health()["status"] == "ok"
    client = app.test_client()
    client.get("/api/health")
    assert webapp.get_predictor(app) is predictor  # not re-created per request


def test_main_starts_the_server_with_the_spec_settings(monkeypatch: pytest.MonkeyPatch, checkpoint_paths: dict,
                                                       capsys: pytest.CaptureFixture) -> None:
    calls = []
    monkeypatch.setattr(Flask, "run", lambda self, **kwargs: calls.append(kwargs))  # never start a real server
    code = webapp.main(["--device", "cpu", "--port", "5123", "--mnist-checkpoint", str(checkpoint_paths["digits"]),
                        "--emnist-checkpoint", str(checkpoint_paths["characters"])])
    assert code == 0
    assert calls == [{"host": "127.0.0.1", "port": 5123, "threaded": True, "debug": False, "use_reloader": False}]
    assert "Open http://127.0.0.1:5123 in your browser" in capsys.readouterr().out


def test_main_exits_with_code_2_when_cuda_is_unavailable(monkeypatch: pytest.MonkeyPatch,
                                                         capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(Flask, "run", lambda self, **kwargs: pytest.fail("the server must not start"))
    assert webapp.main(["--device", "cuda"]) == 2
    assert "CUDA" in capsys.readouterr().err


def test_cli_defaults_follow_the_spec() -> None:
    args = webapp.build_arg_parser().parse_args([])
    assert (args.host, args.port, args.device) == ("127.0.0.1", 5000, "auto")
    assert args.mnist_checkpoint == PROJECT_ROOT / "models" / "mnist_cnn.pth"
    assert args.emnist_checkpoint == PROJECT_ROOT / "models" / "emnist_byclass_cnn.pth"
    assert args.debug is False and args.debug_preprocessing is False


# --------------------------------------------------------------------------------------------
# Frontend and health
# --------------------------------------------------------------------------------------------
def test_index_and_static_assets_are_served(client) -> None:
    index = client.get("/")
    assert index.status_code == 200 and index.mimetype == "text/html"
    assert b"/static/script.js" in index.data and b"/static/styles.css" in index.data
    for asset in ("/static/script.js", "/static/styles.css"):
        assert client.get(asset).status_code == 200


def test_health_reports_both_models_and_is_json_serialisable(client) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.get_json()
    json.dumps(body, allow_nan=False)
    assert body["success"] is True and body["api_status"] == "ok"
    assert body["status"] == "ok"
    assert body["device"] == "cpu"
    assert body["models"]["digits"]["loaded"] is True and body["models"]["characters"]["loaded"] is True
    assert body["models"]["digits"]["num_classes"] == 10 and body["models"]["characters"]["num_classes"] == 62
    assert body["models"]["digits"]["test_metrics"] is None  # random-init fixture: nothing measured
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Cache-Control"] == "no-store"


# --------------------------------------------------------------------------------------------
# Successful predictions
# --------------------------------------------------------------------------------------------
def test_valid_digit_prediction_returns_10_probabilities(client, digit_data_url: str) -> None:
    assert_prediction(client.post(PREDICT_URL, json={"image": digit_data_url, "mode": "digits"}), "digits")


def test_valid_character_prediction_returns_62_labelled_probabilities(client) -> None:
    response = client.post(PREDICT_URL, json={"image": png_data_url(draw_glyph("P")), "mode": "characters"})
    body = assert_prediction(response, "characters")
    assert body["dataset"] == "emnist_byclass"
    assert [row["class_index"] for row in body["probabilities"]] == list(range(62))
    assert body["category"] in {"digit", "uppercase", "lowercase"}


def test_bare_base64_and_case_insensitive_mode_are_accepted(client, digit_png: bytes) -> None:
    payload = {"image": base64.b64encode(digit_png).decode("ascii"), "mode": " Digits "}
    assert_prediction(client.post(PREDICT_URL, json=payload), "digits")


@pytest.mark.parametrize("mode", ["digits", "characters"])
def test_multipart_file_upload(client, digit_png: bytes, mode: str) -> None:
    data = {"image": (io.BytesIO(digit_png), "drawing.png", "image/png"), "mode": mode}
    assert_prediction(client.post(PREDICT_URL, data=data, content_type="multipart/form-data"), mode)


def test_multipart_upload_of_a_white_background_photo(client) -> None:
    photo = np.dstack([255 - draw_glyph("seven")] * 3)  # black ink on white paper, RGB
    data = {"image": (io.BytesIO(png_bytes(photo)), "photo.png"), "mode": "digits"}
    assert_prediction(client.post(PREDICT_URL, data=data, content_type="multipart/form-data"), "digits")


def test_multipart_data_url_text_field(client, digit_data_url: str) -> None:
    data = {"image": digit_data_url, "mode": "digits"}
    assert_prediction(client.post(PREDICT_URL, data=data, content_type="multipart/form-data"), "digits")


# --------------------------------------------------------------------------------------------
# Client errors
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "image",
    [
        pytest.param("data:image/png;base64,@@@not base64@@@", id="invalid-base64"),
        pytest.param(base64.b64encode(b"hello, I am not an image").decode(), id="not-an-image"),
        pytest.param("data:image/svg+xml;base64," + base64.b64encode(b"<svg/>").decode(), id="svg"),
        pytest.param(png_data_url(np.zeros((4, 4), np.uint8)), id="too-small"),
        pytest.param(png_data_url(draw_glyph("one"))[:400], id="truncated"),
        pytest.param(12345, id="number"),
        pytest.param(["data:image/png;base64,AAAA"], id="list"),
    ],
)
def test_invalid_image_returns_400(client, image: object) -> None:
    assert_api_error(client.post(PREDICT_URL, json={"image": image, "mode": "digits"}), 400, "invalid_image")


@pytest.mark.parametrize(
    "payload",
    [{"mode": "digits"}, {"image": "", "mode": "digits"}, {"image": "   ", "mode": "digits"},
     {"image": None, "mode": "characters"}, {}],
    ids=["no-image-key", "empty", "whitespace", "null", "empty-object"],
)
def test_missing_image_returns_400(client, payload: dict) -> None:
    body = assert_api_error(client.post(PREDICT_URL, json=payload), 400, "missing_image")
    assert "No image" in body["error"]


def test_multipart_without_image_returns_missing_image(client) -> None:
    response = client.post(PREDICT_URL, data={"mode": "digits"}, content_type="multipart/form-data")
    assert_api_error(response, 400, "missing_image")
    empty_file = {"image": (io.BytesIO(b""), "empty.png"), "mode": "digits"}
    assert_api_error(client.post(PREDICT_URL, data=empty_file, content_type="multipart/form-data"), 400,
                     "missing_image")


@pytest.mark.parametrize("mode", ["letters", "mnist", 3, "", None], ids=["letters", "dataset-key", "number", "empty",
                                                                        "null"])
def test_invalid_or_missing_mode_returns_400(client, digit_data_url: str, mode: object) -> None:
    body = assert_api_error(client.post(PREDICT_URL, json={"image": digit_data_url, "mode": mode}), 400,
                            "invalid_mode")
    assert '"digits"' in body["error"] and '"characters"' in body["error"]


def test_mode_field_absent_returns_invalid_mode(client, digit_data_url: str) -> None:
    assert_api_error(client.post(PREDICT_URL, json={"image": digit_data_url}), 400, "invalid_mode")


@pytest.mark.parametrize(
    "data, content_type",
    [
        pytest.param("image=abc&mode=digits", "text/plain", id="text-plain"),
        pytest.param('{"image": "abc", "mode": ', "application/json", id="malformed-json"),
        pytest.param('["image", "mode"]', "application/json", id="json-array"),
        pytest.param("image=abc&mode=digits", "application/x-www-form-urlencoded", id="urlencoded-form"),
    ],
)
def test_unsupported_or_malformed_body_returns_invalid_request(client, data: str, content_type: str) -> None:
    assert_api_error(client.post(PREDICT_URL, data=data, content_type=content_type), 400, "invalid_request")


@pytest.mark.parametrize("canvas", [blank_canvas(), blank_canvas(value=255)], ids=["black", "white"])
@pytest.mark.parametrize("mode", ["digits", "characters"])
def test_blank_canvas_returns_422(client, canvas: np.ndarray, mode: str) -> None:
    body = assert_api_error(client.post(PREDICT_URL, json={"image": png_data_url(canvas), "mode": mode}), 422,
                            "blank_canvas")
    assert "No drawing found" in body["error"]


def test_oversized_json_payload_returns_413(client) -> None:
    oversized = json.dumps({"image": "A" * (9 * 1024 * 1024), "mode": "digits"})
    body = assert_api_error(client.post(PREDICT_URL, data=oversized, content_type="application/json"), 413,
                            "payload_too_large")
    assert "8 MB" in body["error"]


def test_oversized_multipart_upload_returns_413(client) -> None:
    data = {"image": (io.BytesIO(b"\0" * (9 * 1024 * 1024)), "huge.png"), "mode": "digits"}
    assert_api_error(client.post(PREDICT_URL, data=data, content_type="multipart/form-data"), 413,
                     "payload_too_large")


def test_unknown_api_route_and_wrong_method_return_json_errors(client) -> None:
    assert_api_error(client.get("/api/does-not-exist"), 404, "not_found")
    response = client.get(PREDICT_URL)
    assert_api_error(response, 405, "method_not_allowed")
    assert "POST" in response.headers["Allow"]
    assert client.get("/no-such-page").mimetype == "text/html"  # non-API routes keep Flask's HTML pages


# --------------------------------------------------------------------------------------------
# Missing / broken models
# --------------------------------------------------------------------------------------------
def test_missing_checkpoint_gives_503_with_training_command_and_degraded_health(
    make_client, emnist_checkpoint: Path, digit_data_url: str
) -> None:
    client = make_client({"digits": MISSING_MNIST_CHECKPOINT, "characters": emnist_checkpoint})

    health = client.get("/api/health")
    assert health.status_code == 200
    body = health.get_json()
    json.dumps(body, allow_nan=False)
    assert body["success"] is True and body["status"] == "degraded"
    assert body["models"]["digits"]["status"] == "missing" and body["models"]["digits"]["loaded"] is False
    assert "python train_mnist.py" in body["models"]["digits"]["error"]
    assert body["models"]["characters"]["status"] == "loaded"
    assert body["loaded_models"] == ["EMNIST ByClass"]

    error = assert_api_error(client.post(PREDICT_URL, json={"image": digit_data_url, "mode": "digits"}), 503,
                             "model_unavailable")
    assert error["error"] == ("Checkpoint not found: models/__pytest_missing_mnist_cnn__.pth. "
                              "Train it with: python train_mnist.py")
    # The other mode keeps working.
    assert_prediction(client.post(PREDICT_URL, json={"image": digit_data_url, "mode": "characters"}), "characters")


def test_corrupt_checkpoint_gives_503_model_unavailable(make_client, tmp_path: Path, mnist_checkpoint: Path,
                                                        digit_data_url: str) -> None:
    broken = tmp_path / "emnist_byclass_cnn.pth"
    broken.write_bytes(b"\x00garbage" * 100)
    client = make_client({"digits": mnist_checkpoint, "characters": broken})
    assert client.get("/api/health").get_json()["models"]["characters"]["status"] == "error"
    body = assert_api_error(client.post(PREDICT_URL, json={"image": digit_data_url, "mode": "characters"}), 503,
                            "model_unavailable", allowed_paths=(broken,))
    assert "python train_emnist.py" in body["error"]


def test_both_models_missing_still_serves_health_and_frontend(digit_data_url: str) -> None:
    missing = {"digits": MISSING_MNIST_CHECKPOINT, "characters": MISSING_MNIST_CHECKPOINT.with_name("__nope__.pth")}
    client = create_app(Predictor(missing, device="cpu")).test_client()
    assert client.get("/").status_code == 200
    body = client.get("/api/health").get_json()
    assert body["status"] == "degraded" and body["loaded_models"] == []
    for mode in ("digits", "characters"):
        assert_api_error(client.post(PREDICT_URL, json={"image": digit_data_url, "mode": mode}), 503,
                         "model_unavailable")


# --------------------------------------------------------------------------------------------
# Internal errors never leak details
# --------------------------------------------------------------------------------------------
SECRET_FAILURE = f'Traceback (most recent call last): File "{PROJECT_ROOT}/src/predictor.py", line 1, secret'


@pytest.mark.parametrize("debug", [False, True], ids=["production", "flask-debug"])
def test_prediction_failure_returns_500_without_stack_trace(predictor, monkeypatch, digit_data_url: str,
                                                            debug: bool) -> None:
    def explode(*_args, **_kwargs):
        raise RuntimeError(SECRET_FAILURE)

    monkeypatch.setattr(predictor, "predict", explode)
    app = create_app(predictor)
    app.debug = debug
    response = app.test_client().post(PREDICT_URL, json={"image": digit_data_url, "mode": "digits"})
    body = assert_api_error(response, 500, "prediction_error")
    assert "secret" not in response.get_data(as_text=True)
    assert body["error"] == "The prediction failed because of an internal error."


@pytest.mark.parametrize("debug", [False, True], ids=["production", "flask-debug"])
def test_unexpected_error_returns_500_internal_error_without_details(predictor, monkeypatch, debug: bool) -> None:
    def explode():
        raise KeyError(SECRET_FAILURE)

    monkeypatch.setattr(predictor, "health", explode)
    app = create_app(predictor)
    app.debug = debug
    response = app.test_client().get("/api/health")
    assert_api_error(response, 500, "internal_error")
    assert "secret" not in response.get_data(as_text=True)


def test_error_messages_from_every_failure_path_are_user_safe(client, make_client, tmp_path: Path,
                                                              digit_data_url: str) -> None:
    """One sweep over the 4xx/5xx paths: every body is the JSON error shape with no internals."""
    broken = tmp_path / "broken.pth"
    broken.write_bytes(b"PK\x03\x04 truncated zip")
    broken_client = make_client({"digits": broken, "characters": MISSING_MNIST_CHECKPOINT})
    requests = [
        (client, {"image": "data:image/png;base64,iVBORw0KGgo=", "mode": "digits"}, 400),
        (client, {"image": png_data_url(blank_canvas()), "mode": "characters"}, 422),
        (client, {"image": digit_data_url, "mode": "klingon"}, 400),
        (broken_client, {"image": digit_data_url, "mode": "digits"}, 503),
        (broken_client, {"image": digit_data_url, "mode": "characters"}, 503),
    ]
    for test_client, payload, status in requests:
        response = test_client.post(PREDICT_URL, json=payload)
        assert response.status_code == status
        assert_no_leak(response.get_data(as_text=True), allowed_paths=(broken,))


# --------------------------------------------------------------------------------------------
# Regression tests from the code review
# --------------------------------------------------------------------------------------------
def test_deeply_nested_json_is_a_client_error(client) -> None:
    body = '{"image": ' + "[" * 100_000 + "]" * 100_000 + ', "mode": "digits"}'
    response = client.post("/api/predict", data=body, content_type="application/json")
    assert_api_error(response, 400, "invalid_request")


@pytest.mark.parametrize("port", ["0", "70000", "-1", "http"])
def test_cli_rejects_invalid_ports(port: str) -> None:
    with pytest.raises(SystemExit):
        webapp.build_arg_parser().parse_args(["--port", port])


def test_checkpoint_outside_the_project_is_shown_by_name_only(make_client, tmp_path: Path) -> None:
    missing = tmp_path / "secret_dir" / "nope.pth"
    client = make_client({"digits": missing})
    health = client.get("/api/health").get_json()
    assert health["models"]["digits"]["checkpoint"] == "nope.pth"
    assert str(tmp_path) not in json.dumps(health)
    response = client.post("/api/predict", json={"image": png_data_url(draw_glyph()), "mode": "digits"})
    body = assert_api_error(response, 503, "model_unavailable")
    assert str(tmp_path) not in body["error"] and "nope.pth" in body["error"]
