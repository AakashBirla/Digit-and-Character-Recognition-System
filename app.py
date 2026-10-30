"""Flask web app: serves the drawing UI and a small JSON API around :class:`src.predictor.Predictor`.

Run it with::

    python app.py                      # http://127.0.0.1:5000
    python app.py --device cpu --port 8000 --debug-preprocessing

Endpoints
---------
``GET  /``            the single-page frontend (``frontend/index.html``; assets under ``/static``)
``GET  /api/health``  service + model status (which models are loaded, device, test metrics if measured)
``POST /api/predict`` JSON ``{"image": "<data URL or base64>", "mode": "digits" | "characters"}``
                      or multipart/form-data with a file field ``image`` and a form field ``mode``

Every API error is ``{"success": false, "error": "<user-safe message>", "code": "<code>"}``:

=====================  ======  =================================================================
code                   status  meaning
=====================  ======  =================================================================
missing_image          400     no image in the request
invalid_image          400     not a decodable PNG/JPEG/WEBP/BMP/GIF, bad base64, bad dimensions
invalid_mode           400     mode missing or not "digits"/"characters"
invalid_request        400     body is not JSON/multipart, or not a JSON object
blank_canvas           422     the image contains no drawing
preprocessing_error    422     the drawing could not be turned into a model input
model_unavailable      503     the checkpoint for that mode is missing or broken (message says how to train it)
payload_too_large      413     request body above 8 MB
not_found              404     unknown /api/ route
method_not_allowed     405     wrong HTTP method for an /api/ route
prediction_error       500     the model failed on this input
internal_error         500     anything else (details only in the server log, never in the response)
=====================  ======  =================================================================
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from src.config import DEBUG_DIR, FRONTEND_DIR, MODES, get_spec
from src.predictor import InvalidModeError, ModelUnavailableError, Predictor
from src.preprocessing import BlankImageError, InvalidImageError, PreprocessingError, decode_image_payload
from src.utils import setup_logging

logger = logging.getLogger("app")

MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB: a 400x400 canvas PNG is ~10-40 KB
_MODE_HINT = " or ".join(f'"{mode}"' for mode in MODES)
_HTTP_ERROR_CODES = {400: "invalid_request", 404: "not_found", 405: "method_not_allowed", 413: "payload_too_large"}
_HTTP_ERROR_MESSAGES = {
    400: "The request could not be understood.",
    404: "This API endpoint does not exist. Available: GET /api/health, POST /api/predict.",
    405: "This HTTP method is not allowed for this endpoint.",
}


class ApiError(Exception):
    """An expected, user-facing API error."""

    def __init__(self, code: str, message: str, status: int) -> None:
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


def error_response(code: str, message: str, status: int) -> tuple[Response, int]:
    """The single JSON error shape used by every API endpoint."""
    return jsonify({"success": False, "error": message, "code": code}), status


def _is_api_request() -> bool:
    return request.path == "/api" or request.path.startswith("/api/")


def create_app(
    predictor: Predictor | None = None,
    *,
    device: str = "auto",
    checkpoint_paths: Mapping[str, Path | str | None] | None = None,
    debug_dir: Path | str | None = None,
) -> Flask:
    """Build the Flask app.

    The :class:`Predictor` (and therefore both models) is created once here, unless one is
    injected (tests inject a predictor that points at temporary checkpoints).
    """
    app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
    app.config["MAX_FORM_MEMORY_SIZE"] = MAX_CONTENT_LENGTH  # allow a data URL sent as a multipart text field
    app.json.sort_keys = False  # keep "success" first and probabilities in class order

    if predictor is None:
        predictor = Predictor(checkpoint_paths, device=device, debug_dir=debug_dir)
    app.extensions["predictor"] = predictor

    _register_routes(app)
    _register_error_handlers(app)

    @app.after_request
    def _security_headers(response: Response) -> Response:
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        if _is_api_request():
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    return app


def get_predictor(app: Flask) -> Predictor:
    """The predictor attached to ``app`` by :func:`create_app`."""
    return app.extensions["predictor"]


# --------------------------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------------------------
def _register_routes(app: Flask) -> None:
    @app.get("/")
    def index() -> Response:
        return app.send_static_file("index.html")

    @app.get("/api/health")
    def health() -> tuple[Response, int]:
        return jsonify({"success": True, "api_status": "ok", **get_predictor(app).health()}), 200

    @app.post("/api/predict")
    def predict() -> tuple[Response, int]:
        try:
            raw_image, raw_mode = _read_predict_request()
            mode = _validate_mode(raw_mode)
            image = decode_image_payload(raw_image)
            result = get_predictor(app).predict(image, mode)
        except ApiError as exc:
            return error_response(exc.code, exc.message, exc.status)
        except InvalidModeError as exc:
            return error_response(InvalidModeError.code, exc.reason, 400)
        except ModelUnavailableError as exc:
            return error_response(ModelUnavailableError.code, exc.reason, 503)
        except BlankImageError as exc:
            return error_response(exc.code, str(exc), 422)
        except InvalidImageError as exc:
            return error_response(exc.code, str(exc), 400)
        except PreprocessingError as exc:
            return error_response(PreprocessingError.code, str(exc), 422)
        except HTTPException:
            raise  # e.g. 413 while reading the body -> handled by the error handlers below
        except Exception:
            logger.exception("Prediction failed")
            return error_response("prediction_error", "The prediction failed because of an internal error.", 500)

        logger.info("Predicted %r for %s (confidence %.3f, %.1f ms)", result["prediction"], mode,
                    result["confidence"], result["processing_time_ms"])
        return jsonify({"success": True, **result}), 200


def _read_predict_request() -> tuple[Any, Any]:
    """Extract ``(image, mode)`` from a JSON or multipart request (validated only for presence)."""
    if request.is_json:
        try:
            body = request.get_json(silent=True)  # None for malformed JSON; 413 is still raised for big bodies
        except RecursionError:  # absurdly deep nesting is not "malformed" to the JSON parser
            body = None
        if not isinstance(body, dict):
            raise ApiError("invalid_request", 'The request body must be a JSON object like '
                           '{"image": "data:image/png;base64,...", "mode": "digits"}.', 400)
        image, mode = body.get("image"), body.get("mode")
    elif request.mimetype == "multipart/form-data":
        upload = request.files.get("image")
        image = upload.read() if upload is not None else request.form.get("image")  # file, or data URL text
        mode = request.form.get("mode")
    else:
        raise ApiError("invalid_request", "Unsupported content type. Send JSON {\"image\": ..., \"mode\": ...} "
                       "or multipart/form-data with an 'image' file and a 'mode' field.", 400)

    if image is None or (isinstance(image, (str, bytes)) and not image.strip()):
        raise ApiError("missing_image", "No image was provided. Draw something and try again.", 400)
    if not isinstance(image, (str, bytes)):
        raise ApiError("invalid_image", "The 'image' field must be a data URL or base64 string.", 400)
    return image, mode


def _validate_mode(mode: Any) -> str:
    if not isinstance(mode, str) or mode.strip().lower() not in MODES:
        if mode is None or (isinstance(mode, str) and not mode.strip()):
            raise ApiError("invalid_mode", f"No mode was provided. Use {_MODE_HINT}.", 400)
        raise InvalidModeError(mode)
    return mode.strip().lower()


# --------------------------------------------------------------------------------------------
# Error handlers: JSON for /api/*, Flask defaults elsewhere, never a stack trace
# --------------------------------------------------------------------------------------------
def _register_error_handlers(app: Flask) -> None:
    @app.errorhandler(RequestEntityTooLarge)
    def payload_too_large(_exc: RequestEntityTooLarge) -> tuple[Response, int]:
        limit_mb = MAX_CONTENT_LENGTH // (1024 * 1024)
        return error_response("payload_too_large", f"The upload is too large (limit {limit_mb} MB).", 413)

    @app.errorhandler(HTTPException)
    def http_error(exc: HTTPException) -> Any:
        if not _is_api_request():
            return exc  # normal HTML error page for the frontend routes
        status = exc.code or 500
        code = _HTTP_ERROR_CODES.get(status, "internal_error" if status >= 500 else "invalid_request")
        message = _HTTP_ERROR_MESSAGES.get(status, "The request could not be processed.")
        response, status = error_response(code, message, status)
        if status == 405 and getattr(exc, "valid_methods", None):
            response.headers["Allow"] = ", ".join(exc.valid_methods)
        return response, status

    @app.errorhandler(Exception)
    def unhandled_error(_exc: Exception) -> Any:
        logger.exception("Unhandled error for %s %s", request.method, request.path)
        if _is_api_request():
            return error_response("internal_error", "An internal error occurred. Please try again.", 500)
        return Response("Internal Server Error", status=500, mimetype="text/plain")


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------
def _port(text: str) -> int:
    try:
        port = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid port {text!r}") from None
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError(f"port must be between 1 and 65535, got {port}")
    return port


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Handwritten digit & character recognition web app (PyTorch + Flask).")
    parser.add_argument("--host", default="127.0.0.1", help="interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=_port, default=5000, help="port to listen on (default: 5000)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="inference device (default: auto = CUDA if available, else CPU)")
    parser.add_argument("--mnist-checkpoint", type=Path, default=get_spec("digits").default_checkpoint,
                        help="checkpoint for digit recognition (default: models/mnist_cnn.pth)")
    parser.add_argument("--emnist-checkpoint", type=Path, default=get_spec("characters").default_checkpoint,
                        help="checkpoint for character recognition (default: models/emnist_byclass_cnn.pth)")
    parser.add_argument("--debug-preprocessing", action="store_true",
                        help="save every preprocessed 28x28 input to results/debug/latest_<mode>.png")
    parser.add_argument("--debug", action="store_true", help="enable Flask debug mode (development only)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    setup_logging()
    try:
        app = create_app(
            device=args.device,
            checkpoint_paths={"digits": args.mnist_checkpoint, "characters": args.emnist_checkpoint},
            debug_dir=DEBUG_DIR if args.debug_preprocessing else None,
        )
    except ValueError as exc:  # e.g. --device cuda without a GPU
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    health = get_predictor(app).health()
    print(f"Inference device: {health['device_name']}")
    for info in health["models"].values():
        state = "loaded" if info["loaded"] else f"NOT AVAILABLE - {info['error']}"
        print(f"  {info['mode']:<10} ({info['display_name']}): {state}")
    if args.debug_preprocessing:
        print(f"Saving preprocessed inputs to {DEBUG_DIR}")
    shown_host = "127.0.0.1" if args.host in ("0.0.0.0", "::", "") else args.host
    if ":" in shown_host:
        shown_host = f"[{shown_host}]"
    print(f"Open http://{shown_host}:{args.port} in your browser", flush=True)

    app.run(host=args.host, port=args.port, threaded=True, debug=args.debug, use_reloader=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
