"""Turn a drawing or uploaded image into the 28x28 tensor the CNN was trained on.

A CNN only works well on inputs that look like its training data. A browser canvas drawing
(400x400, any brush size, anywhere on the canvas) looks nothing like an MNIST or EMNIST
sample, so this module rebuilds the *dataset conventions* measured in ``inspect_datasets.py``:

==============  =====================================  ======================================
                MNIST ("digits")                       EMNIST ByClass ("characters")
==============  =====================================  ======================================
glyph size      longest side = 20 px                   longest side = 23 px (24 with halo)
centring        intensity centre of mass at (14, 14)   bounding-box centre at (13.5, 13.5)
stroke width    ~2.44 px                               ~2.80 px
==============  =====================================  ======================================

Pipeline of :func:`preprocess_image`::

    decode (data URL / bytes / PIL / ndarray)
      -> to_ink_array: grayscale "white ink on black" (alpha + polarity handled)
      -> size check -> background-level removal -> ink mask (> ink_threshold)
      -> blank check -> remove isolated specks -> blank check again
      -> crop to the ink bounding box -> contrast stretch (faint ink -> full white)
      -> stroke-width normalisation (dilate thin strokes / gently erode very thick ones)
      -> resize so the LONGEST side == target_box (aspect ratio preserved, never stretched)
      -> place in a 28x28 frame (COM or bbox centring)
      -> normalise with the dataset mean/std -> tensor (1, 1, 28, 28)

Everything is deterministic and the orientation is never changed: the training data was made
upright once in ``src.datasets.to_upright``, so a drawing is already in the right orientation.

All exceptions raised for bad input derive from :class:`PreprocessingError` and carry a
user-safe message plus a machine-readable ``code`` that the web API returns as-is.
"""

from __future__ import annotations

import base64
import binascii
import io
import logging
import math
import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Union

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from src.config import IMAGE_SIZE, DatasetSpec, get_spec

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------------
#: Image container formats we decode. Everything else (SVG, TIFF, ICO, PDF, HEIC, ...) is refused.
ALLOWED_FORMATS: tuple[str, ...] = ("PNG", "JPEG", "WEBP", "BMP", "GIF")
#: Largest encoded image accepted by :func:`decode_image_payload` (the web API also caps requests at 8 MB).
MAX_IMAGE_BYTES: int = 8 * 1024 * 1024
_MAX_BASE64_CHARS: int = 4 * math.ceil(MAX_IMAGE_BYTES / 3) + 4
_ALLOWED_DATA_URL_TYPES = frozenset({
    "", "image/png", "image/apng", "image/jpeg", "image/jpg", "image/pjpeg", "image/webp",
    "image/bmp", "image/x-bmp", "image/x-ms-bmp", "image/gif",
})
#: Magic-byte signatures of common formats we do NOT support, to give a precise error message.
_UNSUPPORTED_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"II*\x00", "TIFF"), (b"MM\x00*", "TIFF"), (b"\x00\x00\x01\x00", "ICO"), (b"\x00\x00\x02\x00", "CUR"),
    (b"%PDF", "PDF"), (b"8BPS", "PSD"), (b"\x00\x00\x00\x0cjP  ", "JPEG 2000"), (b"\xff\x4f\xff\x51", "JPEG 2000"),
)
_SUPPORTED_FORMATS_TEXT = "PNG, JPEG, WEBP, BMP or GIF"

CENTERING_MODES: tuple[str, ...] = ("center_of_mass", "bounding_box")

# Speck removal: connected ink components smaller than max(SPECK_MIN_AREA, SPECK_FRACTION * total ink)
# are treated as noise (stray pixels, JPEG artefacts). The dot of a hand-drawn "i"/"j" is >= ~0.7 % of
# the ink even with a thin brush, so it survives. The largest component is never removed.
SPECK_MIN_AREA: int = 3
SPECK_FRACTION: float = 0.005

# Stroke-width normalisation (see _normalize_stroke_width for the full explanation).
THIN_STROKE_RATIO: float = 0.75  # predicted output width < 0.75 x target -> dilate (up to the target)
# The erosion trigger was tuned on simulated canvas drawings with a quickly trained model: 1.4 (instead of a
# first guess of 1.75) lifted 40 px-brush round-trip accuracy from ~95 % to ~98 % (MNIST) without changing the
# results for 10-30 px brushes. Erosion stays conservative through the caps in _erode_safely.
THICK_STROKE_RATIO: float = 1.4  # predicted output width > 1.4 x target -> erode ...
ERODE_TARGET_RATIO: float = 1.0  # ... towards the target width (capped, see MAX_ERODED_INK_FRACTION)
MAX_ERODED_INK_FRACTION: float = 0.5  # erosion may remove at most half of the ink
WORK_MAX_SIDE: int = 512  # larger crops are shrunk first: plenty for a 20-24 px output, keeps morphology fast
#: Inputs larger than this are box-downscaled by an integer factor before any float conversion, so a
#: tiny PNG that declares 4096x4096 pixels cannot make every request allocate hundreds of megabytes.
INPUT_MAX_WORK_SIDE: int = 1024
_GEOMETRY_THRESHOLD: int = 127  # mid-grey: "is this pixel part of the stroke?" for width/topology analysis
_HOLE_MIN_RADIUS_OUT: float = 0.5  # a loop is "visible" at 28x28 if its inradius is >= 0.5 output px
_SIGNIFICANT_GAP_OUT: float = 1.0  # gaps narrower than 1 output px are invisible at 28x28 anyway
_COMPONENT_MIN_FRACTION: float = 0.005  # components smaller than this share of the ink are ignored
_RADIUS_BACKOFF: float = 0.7  # multiply the radius by this when a morphology step is rejected

ImageInput = Union[Image.Image, np.ndarray, torch.Tensor, str, bytes, bytearray, memoryview]


# --------------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------------
class PreprocessingError(ValueError):
    """The input could not be turned into a model input. ``str(exc)`` is safe to show to users."""

    code = "preprocessing_error"


class InvalidImageError(PreprocessingError):
    """The payload is not a usable image (bad base64, unsupported format, wrong size, corrupt, ...)."""

    code = "invalid_image"


class BlankImageError(PreprocessingError):
    """The image contains no (or almost no) ink."""

    code = "blank_canvas"


# --------------------------------------------------------------------------------------------
# Configuration and result types
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PreprocessConfig:
    """All knobs of the pipeline. Use :meth:`for_dataset` to get the values matching a dataset."""

    mean: float = 0.1307
    std: float = 0.3081
    image_size: int = IMAGE_SIZE
    target_box: int = 20
    centering: str = "center_of_mass"  # or "bounding_box"
    stroke_width: float = 2.44  # target stroke width in output (28 px) space, used when normalize_stroke
    normalize_stroke: bool = True
    ink_threshold: int = 40  # 0-255; a pixel is ink if > threshold (after polarity normalisation)
    min_ink_pixels: int = 20  # fewer ink pixels -> BlankImageError
    min_side: int = 8
    max_side: int = 4096

    def __post_init__(self) -> None:
        if not (math.isfinite(self.mean) and math.isfinite(self.std) and self.std > 0):
            raise ValueError(f"mean/std must be finite with std > 0 (got {self.mean}, {self.std})")
        if self.image_size < 8:
            raise ValueError(f"image_size must be >= 8, got {self.image_size}")
        if not 1 <= self.target_box <= self.image_size:
            raise ValueError(f"target_box must be in [1, image_size], got {self.target_box}")
        if self.centering not in CENTERING_MODES:
            raise ValueError(f"centering must be one of {CENTERING_MODES}, got {self.centering!r}")
        if not (math.isfinite(self.stroke_width) and 0 < self.stroke_width < self.target_box):
            raise ValueError(f"stroke_width must be in (0, target_box), got {self.stroke_width}")
        if not 0 <= self.ink_threshold < 255:
            raise ValueError(f"ink_threshold must be in [0, 255), got {self.ink_threshold}")
        if self.min_ink_pixels < 1:
            raise ValueError(f"min_ink_pixels must be >= 1, got {self.min_ink_pixels}")
        if not 1 <= self.min_side <= self.max_side:
            raise ValueError(f"need 1 <= min_side <= max_side, got {self.min_side}, {self.max_side}")

    @classmethod
    def for_dataset(cls, key_or_mode: str | DatasetSpec) -> PreprocessConfig:
        """Config reproducing the conventions of a dataset (``"mnist"``/``"digits"``, ``"emnist_byclass"``/...)."""
        spec = get_spec(key_or_mode)
        return cls(
            mean=spec.mean,
            std=spec.std,
            image_size=IMAGE_SIZE,
            target_box=spec.target_box,
            centering=spec.centering,
            stroke_width=spec.stroke_width,
        )

    @classmethod
    def from_metadata(cls, meta: Mapping[str, Any] | None, fallback_key_or_mode: str | DatasetSpec) -> PreprocessConfig:
        """Rebuild a config from checkpoint metadata, starting from the dataset defaults.

        Unknown keys are ignored; a known key with an unusable value is skipped with a warning.
        Raises ``ValueError`` if the resulting combination is invalid (e.g. target_box > image_size).
        """
        base = cls.for_dataset(fallback_key_or_mode)
        if not meta:
            return base
        if not isinstance(meta, Mapping):
            logger.warning("Ignoring preprocessing metadata of type %s", type(meta).__name__)
            return base
        overrides: dict[str, Any] = {}
        for field in fields(cls):
            if meta.get(field.name) is None:
                continue
            try:
                overrides[field.name] = _coerce_field(meta[field.name], type(getattr(base, field.name)))
            except (TypeError, ValueError):
                logger.warning("Ignoring invalid preprocessing value %s=%r", field.name, meta[field.name])
        return replace(base, **overrides)

    def to_metadata(self) -> dict:
        """JSON-safe dict stored in checkpoints (``from_metadata`` reverses it)."""
        return asdict(self)


def _coerce_field(value: Any, kind: type) -> Any:
    """Convert a metadata value to the type of a config field (bool, int, float or str)."""
    if kind is bool:
        if isinstance(value, bool) or value in (0, 1):
            return bool(value)
        raise ValueError(value)
    if kind is int:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(value)
        if isinstance(value, float) and not (math.isfinite(value) and value.is_integer()):
            raise ValueError(value)
        return int(value)
    if kind is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(value)
        return float(value)
    if not isinstance(value, str):
        raise TypeError(value)
    return value


@dataclass
class PreprocessResult:
    """Output of :func:`preprocess_image`."""

    tensor: torch.Tensor  # (1, 1, 28, 28) float32, normalised with the dataset mean/std
    image: np.ndarray  # (28, 28) uint8: what the CNN sees BEFORE normalisation (for preview/debug)
    bbox: tuple[int, int, int, int]  # x, y, w, h of the detected ink in the source image
    source_size: tuple[int, int]  # (width, height) of the source image
    ink_pixels: int  # number of ink pixels found in the source image

    def to_png_bytes(self, scale: int = 1) -> bytes:
        """Encode :attr:`image` as PNG, enlarged ``scale`` times with nearest-neighbour (crisp pixels)."""
        scale = int(scale)
        if scale < 1:
            raise ValueError(f"scale must be >= 1, got {scale}")
        image = np.repeat(np.repeat(self.image, scale, axis=0), scale, axis=1) if scale > 1 else self.image
        ok, buffer = cv2.imencode(".png", np.ascontiguousarray(image))
        if not ok:  # pragma: no cover - only fails on out-of-memory
            raise PreprocessingError("Could not encode the preprocessed image.")
        return buffer.tobytes()

    def to_data_url(self, scale: int = 1) -> str:
        """``"data:image/png;base64,..."`` version of :meth:`to_png_bytes`."""
        return "data:image/png;base64," + base64.b64encode(self.to_png_bytes(scale)).decode("ascii")


# --------------------------------------------------------------------------------------------
# Decoding untrusted payloads
# --------------------------------------------------------------------------------------------
def decode_image_payload(
    payload: str | bytes | bytearray | memoryview,
    *,
    min_side: int = PreprocessConfig.min_side,
    max_side: int = PreprocessConfig.max_side,
) -> Image.Image:
    """Decode an untrusted image payload into a fully loaded PIL image.

    ``payload`` is a data URL (``"data:image/png;base64,..."``), bare base64 text, or raw file
    bytes. Only PNG, JPEG, WEBP, BMP and GIF are accepted (the first frame of animations is
    used). The image dimensions are read from the header and checked against
    ``[min_side, max_side]`` *before* any pixel data is decoded, which together with
    Pillow's own limits protects against decompression bombs. EXIF orientation is honoured.

    Raises:
        InvalidImageError: with a user-safe message for anything that is not a usable image.
    """
    data = _payload_bytes(payload)
    if len(data) > MAX_IMAGE_BYTES:
        raise InvalidImageError(f"The image is too large (limit {MAX_IMAGE_BYTES // (1024 * 1024)} MB).")
    for signature, name in _UNSUPPORTED_SIGNATURES:
        if data.startswith(signature):
            raise InvalidImageError(
                f"Unsupported image format ({name}). Please send a {_SUPPORTED_FORMATS_TEXT} image.")
    if data.lstrip()[:1] == b"<":
        raise InvalidImageError(f"Unsupported image format (SVG/XML). Please send a {_SUPPORTED_FORMATS_TEXT} image.")
    if data[4:8] == b"ftyp":
        raise InvalidImageError(f"Unsupported image format (HEIC/AVIF). Please send a {_SUPPORTED_FORMATS_TEXT} image.")

    with warnings.catch_warnings():
        # Pillow warns (and only later raises) for very large images; treat the warning as an error.
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        try:
            image = Image.open(io.BytesIO(data), formats=list(ALLOWED_FORMATS))  # lazy: reads the header only
        except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise InvalidImageError("The image dimensions are too large to process safely.") from exc
        except Exception as exc:  # UnidentifiedImageError, truncated headers, plugin errors ...
            raise InvalidImageError(
                f"The data is not a readable image. Please send a {_SUPPORTED_FORMATS_TEXT} image."
            ) from exc

        width, height = image.size
        _check_dimensions(width, height, min_side, max_side)
        try:
            image.load()  # decode the pixels of the first frame (only now, after the size check)
        except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise InvalidImageError("The image dimensions are too large to process safely.") from exc
        except Exception as exc:  # truncated/corrupt pixel data
            raise InvalidImageError("The image data is corrupt or truncated.") from exc

    return _apply_exif_orientation(image)


def _payload_bytes(payload: Any) -> bytes:
    """Raw image bytes from a data URL, bare base64 text or bytes."""
    if isinstance(payload, (bytearray, memoryview)):
        payload = bytes(payload)
    if isinstance(payload, bytes):
        if payload[:5].lower() == b"data:":  # a data URL that arrived as bytes (e.g. a text upload)
            try:
                payload = payload.decode("ascii")
            except UnicodeDecodeError as exc:
                raise InvalidImageError("The data URL contains non-ASCII characters.") from exc
        else:
            if not payload:
                raise InvalidImageError("The image is empty.")
            return payload
    if not isinstance(payload, str):
        raise InvalidImageError("The image must be a data URL, base64 text or raw image bytes.")

    text = payload.strip()
    if text[:5].lower() == "data:":
        header, comma, text = text.partition(",")
        if not comma:
            raise InvalidImageError("Malformed data URL: missing ',' before the image data.")
        parts = [part.strip().lower() for part in header[5:].split(";")]
        if "base64" not in parts[1:]:
            raise InvalidImageError("The data URL must be base64-encoded (data:image/png;base64,...).")
        if parts[0] not in _ALLOWED_DATA_URL_TYPES:
            raise InvalidImageError(f"Unsupported image type. Please send a {_SUPPORTED_FORMATS_TEXT} image.")

    text = "".join(text.split())  # base64 may be wrapped over several lines
    if not text:
        raise InvalidImageError("The image is empty.")
    if len(text) > _MAX_BASE64_CHARS:
        raise InvalidImageError(f"The image is too large (limit {MAX_IMAGE_BYTES // (1024 * 1024)} MB).")
    if "-" in text or "_" in text:  # URL-safe alphabet
        text = text.replace("-", "+").replace("_", "/")
    if len(text) % 4:
        text += "=" * (-len(text) % 4)
    try:
        data = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidImageError("The image data is not valid base64.") from exc
    if not data:
        raise InvalidImageError("The image is empty.")
    return data


def _check_dimensions(width: int, height: int, min_side: int, max_side: int) -> None:
    if width < min_side or height < min_side:
        raise InvalidImageError(
            f"The image is too small ({width}x{height} px); each side must be at least {min_side} px.")
    if width > max_side or height > max_side:
        raise InvalidImageError(
            f"The image is too large ({width}x{height} px); each side must be at most {max_side} px.")


def _apply_exif_orientation(image: Image.Image) -> Image.Image:
    """Rotate photos that carry an EXIF orientation tag so they look the way the user sees them."""
    try:
        orientation = image.getexif().get(0x0112, 1)
    except Exception:  # malformed EXIF: ignore it
        return image
    if orientation in (None, 1):
        return image
    image_format = image.format
    try:
        rotated = ImageOps.exif_transpose(image)
    except Exception:  # malformed EXIF: keep the pixels as stored
        logger.debug("Ignoring unusable EXIF orientation %r", orientation)
        return image
    rotated.format = image_format
    return rotated


# --------------------------------------------------------------------------------------------
# Image -> "white ink on black" uint8 array
# --------------------------------------------------------------------------------------------
def to_ink_array(image: Image.Image | np.ndarray | torch.Tensor, *, ink_threshold: int = 40,
                 min_ink_pixels: int = 20) -> np.ndarray:
    """Convert an image to a uint8 ``(H, W)`` array of white ink on a black background.

    Accepts PIL images (modes L, LA, RGB, RGBA, P, PA, I;16, I, F, 1, CMYK, ...) and arrays shaped
    ``(H, W)``, ``(H, W, 1-4)`` or channel-first ``(1-4, H, W)`` of any numeric dtype.

    * Transparency (minimum alpha < 255): strokes on a mostly transparent background are
      composited onto BLACK, so white ink on a transparent canvas works; if that composite has no
      ink but the alpha channel does, the drawing is dark ink on a transparent background and the
      alpha channel itself is the ink. A mostly opaque image with transparent margins, corners or
      holes is instead filled with its own background colour there, so they never become "ink".
    * Polarity: the image is inverted only if both the border AND the whole image are light
      (median > 127), so black-on-white scans and photos become white-on-black, while a canvas
      drawing whose strokes run along the edges is left alone (handwriting covers far less than
      half of the image). Only global statistics are used, so the result never depends on where
      the glyph was drawn.

    ``ink_threshold``/``min_ink_pixels`` define "has ink" for the alpha decision.
    """
    gray, alpha = _gray_and_alpha(image)
    ink = gray
    if alpha is not None and float(alpha.min()) < 255.0:
        weight = alpha / 255.0
        opaque = alpha > 127
        if np.count_nonzero(opaque) > 0.5 * opaque.size:
            # Mostly opaque "paper" with transparent margins/corners/holes: fill those with the
            # paper's own colour (white for a black-on-white page, black for a dark canvas).
            paper = float(np.median(gray[opaque]))
            ink = gray * weight + paper * (1.0 - weight)
        else:
            composite = gray * weight
            composite_blank = np.count_nonzero(composite > ink_threshold) < min_ink_pixels
            alpha_has_ink = np.count_nonzero(alpha > ink_threshold) >= min_ink_pixels
            ink = alpha if (composite_blank and alpha_has_ink) else composite

    if float(np.median(_border_pixels(ink))) > 127.0 and float(np.median(ink)) > 127.0:
        ink = 255.0 - ink
    return np.clip(np.rint(ink), 0, 255).astype(np.uint8)


def _gray_and_alpha(image: Image.Image | np.ndarray | torch.Tensor) -> tuple[np.ndarray, np.ndarray | None]:
    """Return ``(gray, alpha)`` as float32 arrays scaled to 0..255 (``alpha`` is None if absent)."""
    if isinstance(image, Image.Image):
        array = _pil_to_array(image)
    elif isinstance(image, torch.Tensor):
        array = image.detach().cpu().numpy()
    elif isinstance(image, np.ndarray):
        array = image
    else:
        raise InvalidImageError("Unsupported image type; expected a PIL image or a numeric array.")

    array = _to_hwc(array)
    values = _scale_to_255(array)
    channels = values.shape[2]
    if channels in (1, 2):
        gray = values[:, :, 0]
    else:  # ITU-R 601-2 luma, the same weights as PIL's convert("L")
        gray = values[:, :, 0] * 0.299 + values[:, :, 1] * 0.587 + values[:, :, 2] * 0.114
    alpha = values[:, :, -1] if channels in (2, 4) else None
    return gray.astype(np.float32), None if alpha is None else alpha.astype(np.float32)


def _pil_to_array(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to an array with 1 (L), 2 (LA), 3 (RGB) or 4 (RGBA) channels."""
    mode = image.mode
    try:
        if mode == "1":
            image = image.convert("L")
        elif mode in ("P", "PA"):
            image = image.convert("RGBA")  # honours palette transparency (tRNS)
        elif mode == "La":
            image = image.convert("LA")
        elif mode in ("RGBa", "RGBX", "RGBA;15"):
            image = image.convert("RGBA")
        elif mode not in ("L", "LA", "RGB", "RGBA", "I", "F") and not mode.startswith("I;16"):
            image = image.convert("RGB")  # CMYK, YCbCr, LAB, HSV, ...
        if image.mode in ("L", "RGB") and "transparency" in image.info:  # colour-key transparency
            image = image.convert("RGBA")
        return np.asarray(image)
    except Exception as exc:
        raise InvalidImageError("The image uses an unsupported colour mode.") from exc


def _to_hwc(array: np.ndarray) -> np.ndarray:
    """Reshape to ``(H, W, C)`` with C in 1..4 (accepts ``(H, W)``, channel-last or channel-first)."""
    array = np.asarray(array)
    while array.ndim > 3 and array.shape[0] == 1:  # (1, C, H, W) / (1, H, W, C)
        array = array[0]
    if array.ndim == 2:
        array = array[:, :, None]
    elif array.ndim == 3 and array.shape[2] not in (1, 2, 3, 4) and array.shape[0] in (1, 2, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.ndim != 3 or array.shape[2] not in (1, 2, 3, 4) or array.shape[0] == 0 or array.shape[1] == 0:
        raise InvalidImageError(f"Unsupported image array shape {tuple(np.shape(array))}.")
    if not (np.issubdtype(array.dtype, np.number) or array.dtype == np.bool_) or np.iscomplexobj(array):
        raise InvalidImageError(f"Unsupported image array dtype {array.dtype}.")
    return array


def _scale_to_255(array: np.ndarray) -> np.ndarray:
    """Map any numeric dtype to float32 in 0..255 (uint8 as-is, uint16 / 257, floats in [0, 1] x 255)."""
    if array.dtype == np.bool_:
        return array.astype(np.float32) * 255.0
    if array.dtype == np.uint8:
        return array.astype(np.float32)
    if array.dtype == np.uint16:
        return array.astype(np.float32) / 257.0
    values = np.nan_to_num(array.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    low, high = float(values.min()), float(values.max())
    if np.issubdtype(array.dtype, np.floating) and low >= 0.0 and high <= 1.0:
        values = values * 255.0
    elif low >= 0.0 and high <= 255.0:
        pass
    elif low >= 0.0 and high <= 65535.0 and np.issubdtype(array.dtype, np.integer):
        values = values / 257.0
    elif high > low:  # unknown range: stretch to 0..255
        values = (values - low) * (255.0 / (high - low))
    else:
        values = np.zeros_like(values)
    return np.clip(values, 0.0, 255.0).astype(np.float32)


def _border_pixels(image: np.ndarray) -> np.ndarray:
    """1-D array of the pixels on the outermost rows and columns."""
    return np.concatenate([image[0, :], image[-1, :], image[:, 0], image[:, -1]])


# --------------------------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------------------------
def preprocess_image(
    image: ImageInput,
    config: PreprocessConfig,
    *,
    debug_path: Path | str | None = None,
) -> PreprocessResult:
    """Run the full pipeline (see the module docstring) and return a :class:`PreprocessResult`.

    ``image`` may be a PIL image, a numpy array / tensor, or an untrusted payload (data URL,
    base64 text or raw bytes, decoded with :func:`decode_image_payload`). If ``debug_path`` is
    given, the 28x28 uint8 image is also saved there as a PNG.

    Raises:
        InvalidImageError: unreadable input or dimensions outside ``[min_side, max_side]``.
        BlankImageError: fewer than ``config.min_ink_pixels`` ink pixels (after removing specks).
    """
    if not isinstance(config, PreprocessConfig):
        config = PreprocessConfig.for_dataset(config)  # also accept a dataset key / mode / DatasetSpec

    if isinstance(image, (str, bytes, bytearray, memoryview)):
        image = decode_image_payload(image, min_side=config.min_side, max_side=config.max_side)
    if isinstance(image, Image.Image):  # check before converting the pixels
        _check_dimensions(*image.size, config.min_side, config.max_side)

    try:
        image, factor, (width, height) = _reduce_large_input(image)
        _check_dimensions(width, height, config.min_side, config.max_side)
        ink = to_ink_array(image, ink_threshold=config.ink_threshold, min_ink_pixels=config.min_ink_pixels)

        ink = _remove_background_level(ink)
        mask = ink > config.ink_threshold
        if np.count_nonzero(mask) < config.min_ink_pixels:
            raise BlankImageError("No drawing found: the image is empty. Draw something and try again.")
        mask = _remove_specks(mask)
        ink_pixels = int(np.count_nonzero(mask))
        if ink_pixels < config.min_ink_pixels:
            raise BlankImageError("No drawing found: only a few stray pixels. Draw something and try again.")

        x, y, w, h = _mask_bbox(mask)
        # Keep the anti-aliased fringe right next to the strokes, but drop everything else
        # (removed specks, background noise) so it cannot leak into the resized glyph.
        near_ink = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        crop = np.where(near_ink, ink, 0)[y:y + h, x:x + w].astype(np.uint8)
        crop = _stretch_contrast(crop, mask[y:y + h, x:x + w])

        if max(h, w) > WORK_MAX_SIDE:
            scale = WORK_MAX_SIDE / max(h, w)
            size = (max(1, round(w * scale)), max(1, round(h * scale)))
            crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
            crop = _stretch_contrast(crop, crop > config.ink_threshold)
        if config.normalize_stroke:
            crop = _normalize_stroke_width(crop, config)

        glyph, box_centre = _resize_longest_side(crop, config.target_box)
        frame = _place_in_frame(glyph, box_centre, config)
    except cv2.error as exc:  # should not happen for valid input; never leak OpenCV internals
        logger.exception("OpenCV failed while preprocessing an image")
        raise PreprocessingError("The image could not be processed.") from exc

    output = np.clip(np.rint(frame), 0, 255).astype(np.uint8)
    tensor = torch.from_numpy(output.astype(np.float32) / 255.0)
    tensor = ((tensor - config.mean) / config.std).reshape(1, 1, config.image_size, config.image_size).contiguous()

    result = PreprocessResult(  # geometry in source-image pixels (approximate if the input was reduced)
        tensor=tensor,
        image=output,
        bbox=(int(x * factor), int(y * factor), int(w * factor), int(h * factor)),
        source_size=(int(width), int(height)),
        ink_pixels=int(ink_pixels * factor * factor),
    )
    if debug_path is not None:
        _save_debug_image(result, Path(debug_path))
    return result


def preprocess_to_tensor(image: ImageInput, config: PreprocessConfig) -> torch.Tensor:
    """Shortcut for ``preprocess_image(image, config).tensor`` -> ``(1, 1, 28, 28)`` float32."""
    return preprocess_image(image, config).tensor


def _reduce_large_input(
    image: Image.Image | np.ndarray | torch.Tensor,
) -> tuple[Image.Image | np.ndarray, int, tuple[int, int]]:
    """Box-downscale inputs larger than ``INPUT_MAX_WORK_SIDE`` by an integer factor.

    Returns ``(image, factor, (source_width, source_height))``. Small inputs are returned
    unchanged with factor 1. Area averaging keeps thin strokes (as fainter lines, which the
    contrast stretch later restores) and keeps memory bounded for huge images.
    """
    if isinstance(image, Image.Image):
        width, height = image.size
        if max(width, height) <= INPUT_MAX_WORK_SIDE:
            return image, 1, (width, height)
        factor = math.ceil(max(width, height) / INPUT_MAX_WORK_SIDE)
        try:
            # Box-reduce inside PIL, so no full-resolution numpy/float copy is ever made.
            return image.reduce(factor), factor, (width, height)
        except (ValueError, NotImplementedError):
            pass  # modes PIL cannot reduce (palette, bilevel, ...) take the array path below
        array = _pil_to_array(image)
    else:
        if not isinstance(image, (np.ndarray, torch.Tensor)):
            raise InvalidImageError("Unsupported image type; expected a PIL image or a numeric array.")
        array = _to_hwc(image.detach().cpu().numpy() if isinstance(image, torch.Tensor) else image)
        height, width = array.shape[:2]
        if max(width, height) <= INPUT_MAX_WORK_SIDE:
            return array, 1, (width, height)

    factor = math.ceil(max(width, height) / INPUT_MAX_WORK_SIDE)
    array = _to_hwc(array)
    if array.dtype not in (np.uint8, np.uint16, np.float32):
        array = _scale_to_255(array)
    channels = array.shape[2]
    if channels == 2:  # cv2.resize handles 1, 3 and 4 channels; resize gray and alpha separately
        planes = [cv2.resize(array[:, :, c], (max(1, width // factor), max(1, height // factor)),
                             interpolation=cv2.INTER_AREA) for c in range(2)]
        reduced = np.stack(planes, axis=-1)
    else:
        reduced = cv2.resize(array, (max(1, width // factor), max(1, height // factor)), interpolation=cv2.INTER_AREA)
    logger.debug("Reduced a %dx%d input by %d before preprocessing", width, height, factor)
    return reduced, factor, (width, height)


def _remove_background_level(ink: np.ndarray) -> np.ndarray:
    """Subtract the background grey level (median of the border) so grey paper/haze becomes black.

    A no-op for a clean canvas (border median 0); it helps photos and scans, and turns a uniformly
    grey image into an empty one instead of a solid block of "ink". The whole-image median caps the
    estimate, so strokes that run along the canvas edges are not mistaken for the background.
    """
    background = int(min(np.median(_border_pixels(ink)), np.median(ink)))
    return cv2.subtract(ink, background) if background > 0 else ink


def _remove_specks(mask: np.ndarray) -> np.ndarray:
    """Drop tiny isolated components (noise); see SPECK_MIN_AREA / SPECK_FRACTION."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 2:  # background + a single component
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    min_area = max(SPECK_MIN_AREA, SPECK_FRACTION * float(areas.sum()))
    keep = areas >= min_area
    keep[int(np.argmax(areas))] = True  # never remove the main stroke
    if keep.all():
        return mask
    return np.concatenate([[False], keep])[labels]


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """``(x, y, w, h)`` of the True pixels of a non-empty mask."""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return int(cols[0]), int(rows[0]), int(cols[-1] - cols[0] + 1), int(rows[-1] - rows[0] + 1)


def _stretch_contrast(crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Scale intensities so the ink peaks at 255 (no-op for canvas strokes, which already do).

    Faint ink (grey pencil, a light colour) would otherwise produce a dim glyph unlike the bright
    MNIST/EMNIST strokes. The 99th percentile of the ink pixels is used so one bright outlier
    does not decide the scale. Only pixels already classified as ink drive this, so a blank
    canvas with faint noise is never amplified into "ink".
    """
    if not mask.any():
        return crop
    peak = float(np.percentile(crop[mask], 99))
    if peak <= 0 or peak >= 250:
        return crop
    return np.clip(crop.astype(np.float32) * (255.0 / peak), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------------------------
# Stroke-width normalisation
# --------------------------------------------------------------------------------------------
def estimate_stroke_width(mask: np.ndarray) -> float | None:
    """Mean stroke width of a binary mask as ``2 * area / perimeter``.

    For a long stroke of width w and length L, area = w*L and perimeter ~ 2L, so the ratio is ~w.
    The perimeter is the summed length of all contours (outer and holes, ``RETR_LIST``) - the same
    estimator ``inspect_datasets.py`` used to measure 2.44 px (MNIST) and 2.80 px (EMNIST).
    Returns ``None`` if the mask has no measurable perimeter.
    """
    mask_u8 = np.ascontiguousarray(mask, dtype=np.uint8)
    area = int(np.count_nonzero(mask_u8))
    if area == 0:
        return None
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
    return 2.0 * area / perimeter if perimeter > 0 else None


def _normalize_stroke_width(crop: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    """Bring the stroke width (as it will appear at 28x28) closer to the dataset's typical width.

    Why: users draw with any brush. A 4 px brush on a 300 px glyph becomes a 0.3 px hairline
    after shrinking to 20 px, i.e. a faint grey trace unlike the bright ~2.4 px MNIST strokes;
    a 40 px brush on a small glyph becomes a blob.

    Heuristic (conservative by design):
      1. Estimate the source width ``w`` (2*area/perimeter) and predict the output width
         ``w * target_box / longest_side``.
      2. Inside [0.75, 1.4] x target: leave the drawing alone (natural variation).
      3. Too thin: grey-scale dilation with an elliptical kernel of radius r, where r solves
         ``(w + 2r) * B / (L + 2r) = target`` (the bbox grows with the stroke). The dilation is
         rejected - and retried with a 30 % smaller radius - if it changes the glyph's topology as
         seen at 28x28: two separate strokes merging (e.g. the dot of an "i") or an enclosed loop
         appearing/disappearing (a "C" closing into an "O", an "e" eye filling up). Gaps narrower
         than one output pixel are invisible at 28x28 anyway, so the topology baseline is taken
         after bridging those.
      4. Too thick: grey-scale erosion towards the target width, but never removing more than a
         quarter of the width per side, never more than half of the ink, and never splitting the
         glyph into more pieces or deleting a piece; otherwise the radius is halved and retried.
      If no safe radius exists the crop is returned unchanged, so this step can never empty the
      image or melt a glyph into a blob.
    """
    target, box = config.stroke_width, config.target_box
    mask = crop > _GEOMETRY_THRESHOLD
    if np.count_nonzero(mask) < config.min_ink_pixels:
        return crop
    width = estimate_stroke_width(mask)
    if width is None:
        return crop
    longest = max(crop.shape)
    predicted = width * box / longest

    if predicted < THIN_STROKE_RATIO * target:
        radius = (target * longest - width * box) / (2.0 * (box - target))
        return _dilate_safely(crop, int(round(radius)), config)
    if predicted > THICK_STROKE_RATIO * target:
        goal = ERODE_TARGET_RATIO * target
        radius = (width * box - goal * longest) / (2.0 * (box - goal))
        radius = min(radius, width / 4.0)
        return _erode_safely(crop, int(round(radius)), config)
    return crop


def _dilate_safely(crop: np.ndarray, radius: int, config: PreprocessConfig) -> np.ndarray:
    """Dilate by ``radius`` (or a smaller radius) without changing the visible topology."""
    if radius < 1:
        return crop
    box = config.target_box
    padded = cv2.copyMakeBorder(crop, radius + 1, radius + 1, radius + 1, radius + 1, cv2.BORDER_CONSTANT, value=0)
    mask = padded > _GEOMETRY_THRESHOLD
    longest = max(crop.shape)

    # Topology bounds. Gaps narrower than one output pixel are invisible at 28x28, so a baseline in
    # which exactly those gaps are bridged (a dilation by half that width - note that a morphological
    # closing would NOT bridge the end-to-end gaps of a broken thin line) tells us what the real
    # dilation may do at most:
    #   * strokes: no more merging than in the baseline        -> components_after >= components_base
    #   * loops:   keep every visible loop of the drawing      -> holes_after >= holes_original
    #              create loops only by bridging invisible gaps -> holes_after <= holes_base
    invisible_gap = int((_SIGNIFICANT_GAP_OUT * longest / box - 1) // 2)
    base = mask
    if invisible_gap >= 1:
        base = _morph(np.pad(mask, invisible_gap + 1).astype(np.uint8), cv2.MORPH_DILATE, invisible_gap) > 0
    min_area = max(SPECK_MIN_AREA, _COMPONENT_MIN_FRACTION * np.count_nonzero(mask))
    min_components = _count_components(base, min_area)
    min_holes = _count_holes(mask, box / longest)
    max_holes = max(min_holes, _count_holes(base, box / (longest + 2 * max(invisible_gap, 0))))

    while radius >= 1:
        dilated = _crop_to_ink(_morph(padded, cv2.MORPH_DILATE, radius), config.ink_threshold)
        new_mask = dilated > _GEOMETRY_THRESHOLD
        holes = _count_holes(new_mask, box / max(dilated.shape))
        if _count_components(new_mask, min_area) >= min_components and min_holes <= holes <= max_holes:
            logger.debug("Stroke normalisation: dilated by %d px", radius)
            return dilated
        radius = int(radius * _RADIUS_BACKOFF)
    logger.debug("Stroke normalisation: no topology-preserving dilation found; keeping thin strokes")
    return crop


def _erode_safely(crop: np.ndarray, radius: int, config: PreprocessConfig) -> np.ndarray:
    """Erode by ``radius`` (or less) as long as the glyph keeps >= half its ink and its pieces."""
    if radius < 1:
        return crop
    mask = crop > _GEOMETRY_THRESHOLD
    ink_before = np.count_nonzero(mask)
    min_area = max(SPECK_MIN_AREA, _COMPONENT_MIN_FRACTION * ink_before)
    pieces_before = _count_components(mask, min_area)

    while radius >= 1:
        eroded = _morph(crop, cv2.MORPH_ERODE, radius)
        new_mask = eroded > _GEOMETRY_THRESHOLD
        ink_after = np.count_nonzero(new_mask)
        if (ink_after >= max(config.min_ink_pixels, (1.0 - MAX_ERODED_INK_FRACTION) * ink_before)
                and _count_components(new_mask, min_area) == pieces_before):  # nothing split or vanished
            logger.debug("Stroke normalisation: eroded by %d px", radius)
            return _crop_to_ink(eroded, config.ink_threshold)
        radius //= 2
    return crop


def _morph(image: np.ndarray, operation: int, radius: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.morphologyEx(image, operation, kernel)


def _crop_to_ink(image: np.ndarray, threshold: int) -> np.ndarray:
    mask = image > threshold
    if not mask.any():
        return image
    x, y, w, h = _mask_bbox(mask)
    return image[y:y + h, x:x + w]


def _count_components(mask: np.ndarray, min_area: float) -> int:
    """Number of 8-connected ink components with at least ``min_area`` pixels."""
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return int(np.count_nonzero(stats[1:, cv2.CC_STAT_AREA] >= min_area))


def _count_holes(mask: np.ndarray, output_scale: float) -> int:
    """Number of enclosed loops that would still be visible at output scale.

    A hole is a background region not connected to the image border; it is "visible" if its
    inradius (largest distance to the ink) times ``output_scale`` is >= 0.5 output pixel.
    """
    background = np.pad(~mask, 1, constant_values=True).astype(np.uint8)
    count, labels = cv2.connectedComponents(background, connectivity=4)
    if count <= 2:  # only the outside region
        return 0
    outside = np.unique(_border_pixels(labels))
    distance = cv2.distanceTransform(background, cv2.DIST_L2, 5)
    deep = distance * output_scale >= _HOLE_MIN_RADIUS_OUT
    hole_labels = np.unique(labels[deep & (background > 0)])
    return int(np.count_nonzero(~np.isin(hole_labels, outside)))


# --------------------------------------------------------------------------------------------
# Resize and placement
# --------------------------------------------------------------------------------------------
def _resize_longest_side(crop: np.ndarray, target_box: int) -> tuple[np.ndarray, tuple[float, float]]:
    """Scale the crop so its longest side becomes ``target_box`` pixels, preserving the aspect ratio.

    Both axes use the SAME scale factor (``fx == fy``) instead of rounding each side to whole
    pixels, which would stretch a glyph by up to half a pixel per side. A blank margin of at least
    one output pixel is added first, so rounding the output array size only trims empty margin.
    A side that would become thinner than one output pixel (a hairline "1") is scaled to 1 px.
    INTER_AREA when shrinking (proper anti-aliasing, like the datasets), INTER_LINEAR when enlarging
    (cubic interpolation would add faint ringing around the glyph, which skews the centring).

    Returns ``(glyph, centre)``: the float32 image and the exact centre ``(x, y)`` of the glyph's
    box in the glyph's pixel-index coordinates.
    """
    height, width = crop.shape
    scale = target_box / max(height, width)
    fx, fy = max(scale, 1.0 / width), max(scale, 1.0 / height)
    pad = int(math.ceil(1.0 / min(fx, fy)))
    source = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0).astype(np.float32)
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    glyph = np.clip(cv2.resize(source, None, fx=fx, fy=fy, interpolation=interpolation), 0, 255)
    # A source edge at e maps to e * f; pixel-index coordinates are edge coordinates - 0.5.
    centre = ((pad + width / 2.0) * fx - 0.5, (pad + height / 2.0) * fy - 0.5)
    return glyph, centre


def _place_in_frame(glyph: np.ndarray, box_centre: tuple[float, float], config: PreprocessConfig) -> np.ndarray:
    """Translate the glyph into an ``image_size`` x ``image_size`` black frame (float32).

    * ``bounding_box`` (EMNIST): the glyph's box centre lands on ((S-1)/2, (S-1)/2) = (13.5, 13.5).
    * ``center_of_mass`` (MNIST): the intensity centre of mass lands on (S/2, S/2) = (14.0, 14.0)
      in pixel-index coordinates.
    The (sub-pixel) shift is applied with bilinear interpolation and clamped so that no ink leaves
    the frame (only an extremely lopsided glyph can end up slightly off-centre because of that).
    """
    size = config.image_size
    if float(glyph.max()) < 0.5:  # cannot happen for a non-blank crop; keeps the maths below well-defined
        return np.zeros((size, size), np.float32)
    if config.centering == "bounding_box":
        shift_x, shift_y = (size - 1) / 2.0 - box_centre[0], (size - 1) / 2.0 - box_centre[1]
    else:
        total = float(glyph.sum())
        ys, xs = np.mgrid[0:glyph.shape[0], 0:glyph.shape[1]]
        shift_x = size / 2.0 - float((xs * glyph).sum()) / total
        shift_y = size / 2.0 - float((ys * glyph).sum()) / total
    visible = glyph >= 0.5  # anything that survives rounding to uint8
    cols = np.flatnonzero(visible.any(axis=0))
    rows = np.flatnonzero(visible.any(axis=1))
    shift_x = _clamp_shift(shift_x, -float(cols[0]), float(size - 1 - cols[-1]))
    shift_y = _clamp_shift(shift_y, -float(rows[0]), float(size - 1 - rows[-1]))
    matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    return cv2.warpAffine(glyph, matrix, (size, size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _clamp_shift(shift: float, low: float, high: float) -> float:
    """Clamp ``shift`` to [low, high]; if that range is empty the glyph cannot fit anyway, so keep it."""
    return min(max(shift, low), high) if low <= high else shift


def _save_debug_image(result: PreprocessResult, path: Path) -> None:
    """Best-effort save of the 28x28 image; a failure is logged, never raised."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(result.to_png_bytes())
    except OSError as exc:
        logger.warning("Could not save debug image to %s: %s", path, exc)
