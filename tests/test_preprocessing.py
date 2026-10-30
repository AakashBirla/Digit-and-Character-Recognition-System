"""Image preprocessing: canvas drawing / upload -> 28x28 tensor that matches MNIST or EMNIST conventions.

MNIST ("digits"): glyph fits a 20x20 box and its intensity centre of mass sits at (14.0, 14.0).
EMNIST ByClass ("characters"): glyph fits a 24x24 box and its bounding-box centre sits at (13.5, 13.5).
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from src.config import get_spec
from src.preprocessing import (
    BlankImageError,
    InvalidImageError,
    PreprocessConfig,
    PreprocessingError,
    PreprocessResult,
    decode_image_payload,
    estimate_stroke_width,
    preprocess_image,
    preprocess_to_tensor,
    to_ink_array,
)
from tests.conftest import GLYPH_SHAPES, blank_canvas, draw_glyph, ink_geometry, png_bytes, png_data_url

MODES = ("digits", "characters")


@pytest.fixture(params=MODES)
def config(request: pytest.FixtureRequest) -> PreprocessConfig:
    """The dataset-specific config of each mode."""
    return PreprocessConfig.for_dataset(request.param)


def half_max_extent(image28: np.ndarray) -> tuple[int, int]:
    """(width, height) of the region brighter than half intensity - a robust glyph size."""
    mask = image28 > 127
    return int(mask.any(axis=0).sum()), int(mask.any(axis=1).sum())


def assert_glyph_fills_target_box(image28: np.ndarray, config: PreprocessConfig) -> None:
    width, height = half_max_extent(image28)
    assert config.target_box - 2 <= max(width, height) <= config.target_box + 1, (width, height)


def assert_centred(image28: np.ndarray, config: PreprocessConfig) -> None:
    """MNIST: centre of mass at (14, 14). EMNIST: bounding-box centre at (13.5, 13.5) (+-0.5 px pixel grid)."""
    geometry = ink_geometry(image28)
    if config.centering == "center_of_mass":
        assert geometry["com"] == pytest.approx((14.0, 14.0), abs=0.05)
    else:
        assert geometry["bbox_centre"] == pytest.approx((13.5, 13.5), abs=0.5)


# --------------------------------------------------------------------------------------------
# Output contract
# --------------------------------------------------------------------------------------------
def test_output_is_a_normalised_1x1x28x28_float_tensor(config: PreprocessConfig) -> None:
    result = preprocess_image(draw_glyph("seven"), config)
    assert isinstance(result, PreprocessResult)
    assert result.tensor.shape == (1, 1, 28, 28)
    assert result.tensor.dtype == torch.float32
    assert result.image.shape == (28, 28) and result.image.dtype == np.uint8
    assert torch.isfinite(result.tensor).all()


def test_tensor_values_are_consistent_with_the_dataset_normalisation(config: PreprocessConfig) -> None:
    result = preprocess_image(draw_glyph("P"), config)
    background, full_ink = (0.0 - config.mean) / config.std, (1.0 - config.mean) / config.std
    assert result.tensor.min().item() == pytest.approx(background, abs=1e-6)  # black background
    assert result.tensor.max().item() <= full_ink + 1e-6
    assert result.tensor.max().item() > (0.9 - config.mean) / config.std  # strokes are (nearly) full white
    expected = (result.image.astype(np.float32) / 255.0 - config.mean) / config.std
    np.testing.assert_allclose(result.tensor[0, 0].numpy(), expected, atol=1e-6)


def test_dataset_configs_use_the_dataset_conventions() -> None:
    for mode in MODES:
        spec, config = get_spec(mode), PreprocessConfig.for_dataset(mode)
        assert (config.mean, config.std) == (spec.mean, spec.std)
        assert (config.target_box, config.centering, config.stroke_width) == (
            spec.target_box, spec.centering, spec.stroke_width)
        assert config.image_size == 28
    assert PreprocessConfig.for_dataset("digits").target_box == 20
    assert PreprocessConfig.for_dataset("characters").target_box == 23


@pytest.mark.parametrize("shape", [(400, 400), (200, 300), (300, 200), (64, 64), (1000, 600), (1500, 1200)])
def test_any_input_size_gives_the_same_output_dimensions(config: PreprocessConfig, shape: tuple[int, int]) -> None:
    height, width = shape
    canvas = np.zeros(shape, np.uint8)
    cv2.line(canvas, (width // 2, height // 5), (width // 2, 4 * height // 5), 255, max(2, width // 25))
    result = preprocess_image(canvas, config)
    assert result.tensor.shape == (1, 1, 28, 28)
    assert result.source_size == (width, height)
    assert_glyph_fills_target_box(result.image, config)


def test_preprocessing_is_deterministic(config: PreprocessConfig) -> None:
    canvas = draw_glyph("seven", thickness=9)
    first, second = preprocess_image(canvas, config), preprocess_image(canvas.copy(), config)
    assert np.array_equal(first.image, second.image)
    assert torch.equal(first.tensor, second.tensor)


def test_result_does_not_depend_on_where_the_glyph_was_drawn(config: PreprocessConfig) -> None:
    top_left = preprocess_image(draw_glyph("seven", box=(30, 20, 190, 280)), config)
    bottom_right = preprocess_image(draw_glyph("seven", box=(220, 120, 380, 380)), config)
    assert np.array_equal(top_left.image, bottom_right.image)


def test_preprocess_to_tensor_matches_preprocess_image(config: PreprocessConfig) -> None:
    canvas = draw_glyph("zero")
    assert torch.equal(preprocess_to_tensor(canvas, config), preprocess_image(canvas, config).tensor)


def test_result_reports_source_geometry() -> None:
    result = preprocess_image(draw_glyph("rect", box=(100, 50, 300, 350), thickness=10), "digits")
    x, y, w, h = result.bbox
    assert result.source_size == (400, 400)
    assert abs(x - 95) <= 2 and abs(y - 45) <= 2 and abs(w - 211) <= 3 and abs(h - 311) <= 3
    assert result.ink_pixels > 1000


# --------------------------------------------------------------------------------------------
# Blank / invalid input
# --------------------------------------------------------------------------------------------
def _stray_pixels() -> np.ndarray:
    canvas = blank_canvas()
    for x, y in [(10, 10), (300, 200), (350, 100), (50, 50)]:
        canvas[y, x] = 255
    return canvas


@pytest.mark.parametrize(
    "image",
    [
        pytest.param(blank_canvas(), id="black-canvas"),
        pytest.param(blank_canvas(value=255), id="white-canvas"),
        pytest.param(blank_canvas(value=128), id="uniform-grey"),
        pytest.param(_stray_pixels(), id="stray-pixels"),
        pytest.param(draw_glyph("one") // 8, id="ink-too-faint"),
        pytest.param(np.zeros((400, 400, 4), np.uint8), id="fully-transparent-rgba"),
    ],
)
def test_blank_images_are_detected(config: PreprocessConfig, image: np.ndarray) -> None:
    with pytest.raises(BlankImageError) as excinfo:
        preprocess_image(image, config)
    assert excinfo.value.code == "blank_canvas"
    assert isinstance(excinfo.value, PreprocessingError) and isinstance(excinfo.value, ValueError)
    assert "No drawing found" in str(excinfo.value)


def test_blank_png_data_url_is_detected() -> None:
    with pytest.raises(BlankImageError):
        preprocess_image(png_data_url(blank_canvas()), "characters")


@pytest.mark.parametrize("shape", [(4, 400), (400, 5), (5000, 20)])
def test_images_with_unusable_dimensions_are_rejected(shape: tuple[int, int]) -> None:
    image = np.zeros(shape, np.uint8)
    image[1:3, 1:3] = 255
    with pytest.raises(InvalidImageError) as excinfo:
        preprocess_image(image, "digits")
    assert excinfo.value.code == "invalid_image"


# --------------------------------------------------------------------------------------------
# Size, stroke width and edges
# --------------------------------------------------------------------------------------------
def test_small_drawing_is_scaled_up_to_the_target_box(config: PreprocessConfig) -> None:
    small = draw_glyph("seven", box=(180, 180, 205, 210), thickness=3)  # ~25x30 px on a 400 px canvas
    result = preprocess_image(small, config)
    assert_glyph_fills_target_box(result.image, config)
    assert result.image.max() >= 250


def test_large_drawing_is_scaled_down_to_the_target_box(config: PreprocessConfig) -> None:
    large = draw_glyph("zero", box=(5, 5, 395, 395), thickness=40)  # fills the whole canvas
    result = preprocess_image(large, config)
    assert_glyph_fills_target_box(result.image, config)
    assert_centred(result.image, config)
    assert result.image[14, 14] == 0, "the loop of the 0 must stay open"


def test_thin_strokes_are_thickened_into_visible_strokes(config: PreprocessConfig) -> None:
    thin = draw_glyph("seven", thickness=2)  # a 2 px brush on a 280 px glyph -> ~0.15 px after resizing
    normalised = preprocess_image(thin, config).image
    untouched = preprocess_image(thin, replace(config, normalize_stroke=False)).image
    assert normalised.max() >= 250
    assert np.count_nonzero(normalised > 127) > 3 * max(1, np.count_nonzero(untouched > 127))
    width = estimate_stroke_width(normalised > 127)
    assert width is not None and 1.5 <= width <= 4.0


def test_very_thick_strokes_do_not_become_a_blob(config: PreprocessConfig) -> None:
    thick = draw_glyph("zero", thickness=80)
    result = preprocess_image(thick, config).image
    width = estimate_stroke_width(result > 127)
    assert width is not None and width <= 4.0
    assert result[13:15, 13:15].max() == 0, "the hole of the 0 must survive"


@pytest.mark.parametrize("brush", [2, 6, 12, 22, 40])
def test_output_stroke_width_is_similar_for_any_brush_size(config: PreprocessConfig, brush: int) -> None:
    result = preprocess_image(draw_glyph("rect", thickness=brush), config).image
    width = estimate_stroke_width(result > 127)
    assert width is not None and 1.5 <= width <= 4.0


@pytest.mark.parametrize(
    "box",
    [
        pytest.param((-40, 0, 250, 400), id="clipped-at-left-and-bottom"),
        pytest.param((150, -60, 399, 300), id="clipped-at-top-touching-right"),
        pytest.param((0, 0, 399, 399), id="touching-all-edges"),
    ],
)
def test_glyph_touching_or_crossing_the_canvas_edge(config: PreprocessConfig, box: tuple[int, int, int, int]) -> None:
    result = preprocess_image(draw_glyph("seven", box=box, thickness=30), config)
    assert result.tensor.shape == (1, 1, 28, 28)
    assert_glyph_fills_target_box(result.image, config)
    if config.centering == "bounding_box":
        assert_centred(result.image, config)


def test_glyph_touching_the_edge_matches_the_same_glyph_drawn_inside() -> None:
    """A stroke that just touches the border is not clipped, so it must give the same result as inside."""
    at_edge = blank_canvas()
    cv2.rectangle(at_edge, (0, 50), (25, 350), 255, -1)
    inside = blank_canvas()
    cv2.rectangle(inside, (200, 20), (225, 320), 255, -1)
    for mode in MODES:
        assert np.array_equal(preprocess_image(at_edge, mode).image, preprocess_image(inside, mode).image)


# --------------------------------------------------------------------------------------------
# Aspect ratio and centring
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "box",
    [(150, 20, 250, 380), (20, 150, 380, 250), (100, 100, 300, 300), (50, 150, 350, 250), (180, 40, 220, 360)],
    ids=["tall", "wide", "square", "wide-3to1", "very-tall"],
)
def test_aspect_ratio_is_preserved(config: PreprocessConfig, box: tuple[int, int, int, int]) -> None:
    """A filled rectangle keeps its proportions: the longest side becomes target_box, the other scales along."""
    canvas = blank_canvas()
    cv2.rectangle(canvas, box[:2], box[2:], 255, -1)
    width_in, height_in = box[2] - box[0] + 1, box[3] - box[1] + 1
    scale = config.target_box / max(width_in, height_in)
    geometric = replace(config, normalize_stroke=False)  # pure resize, no stroke-width change
    width, height = half_max_extent(preprocess_image(canvas, geometric).image)
    assert width == pytest.approx(width_in * scale, abs=1.0)
    assert height == pytest.approx(height_in * scale, abs=1.0)


@pytest.mark.parametrize("shape", ["one", "dash"])
def test_narrow_glyphs_are_not_stretched_to_a_square(config: PreprocessConfig, shape: str) -> None:
    width, height = half_max_extent(preprocess_image(draw_glyph(shape), config).image)
    long_side, short_side = max(width, height), min(width, height)
    assert long_side >= config.target_box - 2
    assert short_side <= 6, "a 1 or a - must stay thin, never stretched to fill the box"
    assert (height > width) == (shape == "one")


@pytest.mark.parametrize("shape", GLYPH_SHAPES)
def test_glyph_fills_the_dataset_target_box(config: PreprocessConfig, shape: str) -> None:
    assert_glyph_fills_target_box(preprocess_image(draw_glyph(shape), config).image, config)


@pytest.mark.parametrize("shape", ["one", "dash", "zero", "rect", "P"])
def test_mnist_centre_of_mass_is_at_14_14(shape: str) -> None:
    image = preprocess_image(draw_glyph(shape), "digits").image
    assert ink_geometry(image)["com"] == pytest.approx((14.0, 14.0), abs=0.05)


@pytest.mark.parametrize("shape", GLYPH_SHAPES)
def test_emnist_bounding_box_centre_is_at_13_5(shape: str) -> None:
    image = preprocess_image(draw_glyph(shape), "characters").image
    assert ink_geometry(image)["bbox_centre"] == pytest.approx((13.5, 13.5), abs=0.5)


def test_centring_rule_differs_between_datasets() -> None:
    """A "P" (heavy top) separates the two rules: COM-centred for MNIST, box-centred for EMNIST."""
    mnist = ink_geometry(preprocess_image(draw_glyph("P"), "digits").image)
    emnist = ink_geometry(preprocess_image(draw_glyph("P"), "characters").image)
    assert mnist["com"] == pytest.approx((14.0, 14.0), abs=0.05)
    assert max(abs(mnist["bbox_centre"][0] - 13.5), abs(mnist["bbox_centre"][1] - 13.5)) >= 1.0
    assert emnist["bbox_centre"] == pytest.approx((13.5, 13.5), abs=0.5)
    assert emnist["com"][1] < 12.5, "the EMNIST P keeps its natural (high) centre of mass"


def test_orientation_is_never_changed(config: PreprocessConfig) -> None:
    """A "P" stays a P: tall, stem on the left, loop at the top (no transpose/flip/rotation)."""
    image = preprocess_image(draw_glyph("P", thickness=14), config).image
    mask = image > 127
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    box = mask[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    height, width = box.shape
    assert height > width
    column_heights = box.sum(axis=0)
    assert np.argmax(column_heights) < width / 3, "the stem must be on the left"
    assert box[0].sum() > 0.6 * width, "the top row is the loop's top bar"
    assert box[-1].sum() < 0.4 * width, "the bottom row is only the stem"


# --------------------------------------------------------------------------------------------
# Polarity, alpha and input types
# --------------------------------------------------------------------------------------------
def test_black_ink_on_white_background_matches_white_on_black(config: PreprocessConfig) -> None:
    glyph = draw_glyph("zero", thickness=20)
    expected = preprocess_image(glyph, config).image
    assert np.array_equal(preprocess_image(255 - glyph, config).image, expected)
    rgb_white_paper = np.dstack([255 - glyph] * 3)
    assert np.array_equal(preprocess_image(rgb_white_paper, config).image, expected)
    assert np.array_equal(preprocess_image(png_data_url(rgb_white_paper), config).image, expected)


def test_transparent_rgba_input_is_handled(config: PreprocessConfig) -> None:
    glyph = draw_glyph("zero", thickness=20)
    expected = preprocess_image(glyph, config).image
    white_ink = np.dstack([np.full_like(glyph, 255)] * 3 + [glyph])  # white strokes on a transparent canvas
    black_ink = np.dstack([np.zeros_like(glyph)] * 3 + [glyph])  # black strokes on a transparent canvas
    for rgba in (white_ink, black_ink):
        assert np.array_equal(preprocess_image(rgba, config).image, expected)
        assert np.array_equal(preprocess_image(png_bytes(rgba), config).image, expected)


def test_to_ink_array_normalises_polarity_and_alpha() -> None:
    glyph = draw_glyph("one")
    assert np.array_equal(to_ink_array(glyph), glyph)
    assert np.array_equal(to_ink_array(255 - glyph), glyph)
    assert np.array_equal(to_ink_array(np.dstack([np.zeros_like(glyph)] * 3 + [glyph])), glyph)
    assert to_ink_array(Image.fromarray(glyph).convert("RGB")).shape == (400, 400)


def test_all_supported_input_types_give_the_same_result(config: PreprocessConfig) -> None:
    glyph = draw_glyph("P")
    expected = preprocess_image(glyph, config).image
    inputs = {
        "pil": Image.fromarray(glyph),
        "tensor": torch.from_numpy(glyph),
        "float01": glyph.astype(np.float32) / 255.0,
        "png-bytes": png_bytes(glyph),
        "data-url": png_data_url(glyph),
        "bare-base64": base64.b64encode(png_bytes(glyph)).decode("ascii"),
    }
    for name, value in inputs.items():
        assert np.array_equal(preprocess_image(value, config).image, expected), name


def _encode(image: Image.Image, image_format: str, **options) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=image_format, **options)
    return buffer.getvalue()


@pytest.mark.parametrize("image_format, options", [("PNG", {}), ("BMP", {}), ("GIF", {}),
                                                   ("WEBP", {"lossless": True})])
def test_every_allowed_lossless_format_gives_the_same_result(image_format: str, options: dict) -> None:
    glyph = draw_glyph("seven")
    expected = preprocess_image(glyph, "digits").image
    payload = _encode(Image.fromarray(glyph), image_format, **options)
    assert decode_image_payload(payload).format == image_format
    assert np.array_equal(preprocess_image(payload, "digits").image, expected)


@pytest.mark.parametrize("mode", ["1", "L", "LA", "P", "RGB", "RGBA", "I", "F", "I;16", "CMYK"])
def test_to_ink_array_supports_common_pil_modes(mode: str) -> None:
    binary = np.where(draw_glyph("seven") > 127, 255, 0).astype(np.uint8)
    if mode == "I;16":
        image = Image.fromarray(binary.astype(np.uint16) * 257)  # 16-bit greyscale
        assert image.mode == "I;16"
    else:
        image = Image.fromarray(binary).convert(mode)
    assert np.array_equal(to_ink_array(image), binary), mode


def test_exif_orientation_of_photos_is_applied() -> None:
    """A phone photo stored sideways with EXIF orientation 6 is decoded the way the user sees it."""
    upright = draw_glyph("P")
    stored = np.ascontiguousarray(np.rot90(upright, k=1))  # rotated 90 deg counter-clockwise on disk
    exif = Image.new("L", (1, 1)).getexif()
    exif[0x0112] = 6  # "rotate 90 deg clockwise to display"
    payload = _encode(Image.fromarray(stored), "PNG", exif=exif)
    decoded = np.asarray(decode_image_payload(payload))
    assert np.array_equal(decoded, upright)


def test_jpeg_input_gives_nearly_the_same_result() -> None:
    glyph = draw_glyph("P")
    buffer = io.BytesIO()
    Image.fromarray(glyph).save(buffer, format="JPEG", quality=90)
    expected = preprocess_image(glyph, "digits").image.astype(int)
    from_jpeg = preprocess_image(buffer.getvalue(), "digits").image.astype(int)
    assert np.abs(expected - from_jpeg).mean() < 1.0


# --------------------------------------------------------------------------------------------
# Decoding untrusted payloads
# --------------------------------------------------------------------------------------------
def test_decode_image_payload_accepts_data_url_base64_and_bytes() -> None:
    raw = png_bytes(draw_glyph("one", size=64, box=(20, 10, 40, 50), thickness=4))
    for payload in (raw, "data:image/png;base64," + base64.b64encode(raw).decode(), base64.b64encode(raw).decode()):
        image = decode_image_payload(payload)
        assert isinstance(image, Image.Image)
        assert image.size == (64, 64) and image.format == "PNG"


def _tiff_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(draw_glyph("one")).save(buffer, format="TIFF")
    return buffer.getvalue()


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param("", id="empty-string"),
        pytest.param(b"", id="empty-bytes"),
        pytest.param("data:image/png;base64,@@not-base64@@", id="invalid-base64"),
        pytest.param("data:image/png;base64,", id="empty-data-url"),
        pytest.param(base64.b64encode(b"definitely not an image").decode(), id="base64-of-text"),
        pytest.param(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50, id="garbage-png"),
        pytest.param("data:text/plain;base64," + base64.b64encode(b"hello").decode(), id="text-data-url"),
        pytest.param(b'<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"/>', id="svg"),
        pytest.param(_tiff_bytes(), id="tiff"),
        pytest.param(png_bytes(np.zeros((4, 4), np.uint8)), id="too-small"),
        pytest.param(png_bytes(np.zeros((10, 5000), np.uint8)), id="too-wide"),
        pytest.param(png_bytes(draw_glyph("one"))[:300], id="truncated-png"),
        pytest.param(12345, id="not-a-string"),
    ],
)
def test_decode_image_payload_rejects_bad_payloads(payload: object) -> None:
    with pytest.raises(InvalidImageError) as excinfo:
        decode_image_payload(payload)
    assert excinfo.value.code == "invalid_image"
    message = str(excinfo.value)
    assert message and "Traceback" not in message and "/home/" not in message


# --------------------------------------------------------------------------------------------
# Configuration and result helpers
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_config_metadata_round_trip(mode: str) -> None:
    config = PreprocessConfig.for_dataset(mode)
    metadata = config.to_metadata()
    json.dumps(metadata, allow_nan=False)
    assert PreprocessConfig.from_metadata(json.loads(json.dumps(metadata)), mode) == config


def test_from_metadata_ignores_unknown_keys_and_falls_back_to_dataset_defaults() -> None:
    assert PreprocessConfig.from_metadata(None, "characters") == PreprocessConfig.for_dataset("characters")
    config = PreprocessConfig.from_metadata({"target_box": 22, "future_option": 1, "mean": "oops"}, "characters")
    assert config.target_box == 22
    assert config.mean == get_spec("characters").mean  # invalid value skipped
    assert config.centering == "bounding_box"


@pytest.mark.parametrize("kwargs", [{"centering": "middle"}, {"target_box": 30}, {"std": 0.0}, {"ink_threshold": 255}])
def test_invalid_config_values_raise(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        PreprocessConfig(**kwargs)


def test_result_png_and_data_url_round_trip() -> None:
    result = preprocess_image(draw_glyph("seven"), "digits")
    decoded = np.asarray(Image.open(io.BytesIO(result.to_png_bytes())))
    assert np.array_equal(decoded, result.image)
    url = result.to_data_url()
    assert url.startswith("data:image/png;base64,")
    assert np.array_equal(np.asarray(decode_image_payload(url)), result.image)
    assert Image.open(io.BytesIO(result.to_png_bytes(scale=5))).size == (140, 140)


def test_debug_path_saves_the_28x28_image(tmp_path: Path) -> None:
    path = tmp_path / "debug" / "latest.png"
    result = preprocess_image(draw_glyph("zero"), "characters", debug_path=path)
    assert np.array_equal(np.asarray(Image.open(path)), result.image)


# --------------------------------------------------------------------------------------------
# Regression tests from the code review
# --------------------------------------------------------------------------------------------
def _draw_square_c(margin: int) -> np.ndarray:
    """A square 'C' (top, left and bottom strokes) drawn ``margin`` px from the canvas edges."""
    canvas = blank_canvas()
    far = canvas.shape[0] - 1 - margin
    corners = np.array([(far, margin), (margin, margin), (margin, far), (far, far)], dtype=np.int32)
    cv2.polylines(canvas, [corners], False, 255, 22, cv2.LINE_AA)
    return canvas


def _border_median(image: np.ndarray) -> float:
    return float(np.median(np.concatenate([image[0], image[-1], image[:, 0], image[:, -1]])))


@pytest.mark.parametrize("mode", ["digits", "characters"])
def test_strokes_along_the_canvas_edges_are_not_inverted(mode: str) -> None:
    """Strokes covering most of the border must not flip the polarity (that produced a white block)."""
    edge = _draw_square_c(margin=8)
    assert _border_median(edge) > 127  # the drawing really covers most of the border
    near_edge = preprocess_image(edge, mode).image
    inset = preprocess_image(_draw_square_c(margin=60), mode).image
    assert near_edge.mean() < 100  # mostly black frame, not an inverted block
    assert np.abs(near_edge.astype(float) - inset.astype(float)).mean() < 25


def test_black_on_white_with_transparent_margins_is_read_as_black_ink() -> None:
    """Transparent margins/corners of a black-on-white upload are paper, not ink."""
    rgb = 255 - draw_glyph("seven")
    alpha = np.full_like(rgb, 255)
    alpha[:30, :] = alpha[-30:, :] = alpha[:, :30] = alpha[:, -30:] = 0
    rgba = np.dstack([rgb, rgb, rgb, alpha])
    reference = preprocess_image(draw_glyph("seven"), "digits").image
    result = preprocess_image(Image.fromarray(rgba, "RGBA"), "digits").image
    assert np.abs(result.astype(float) - reference.astype(float)).mean() < 5


def test_tiny_glyph_is_centred_in_bounding_box_mode() -> None:
    """A glyph smaller than the target box is enlarged without ringing that would skew the centring."""
    image = np.zeros((16, 16), np.uint8)
    image[4:11, 4:11] = 255  # a 7x7 blob
    geometry = ink_geometry(preprocess_image(image, replace(PreprocessConfig.for_dataset("characters"),
                                                            normalize_stroke=False, min_ink_pixels=1)).image, 0)
    assert geometry["bbox_centre"] == pytest.approx((13.5, 13.5), abs=0.6)


def test_huge_upload_is_reduced_without_changing_the_result() -> None:
    """A 4096 px image is box-reduced inside PIL first; it must still give the same glyph."""
    small = draw_glyph("seven", size=400)
    big = cv2.resize(small, (4000, 4000), interpolation=cv2.INTER_NEAREST)
    a = preprocess_image(Image.fromarray(small), "digits").image.astype(float)
    b = preprocess_image(Image.fromarray(big).convert("RGBA"), "digits").image.astype(float)
    assert np.abs(a - b).mean() < 6
