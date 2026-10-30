"""Class-index -> character tables for MNIST and EMNIST ByClass, and the official mapping checks."""

from __future__ import annotations

import string
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.class_mapping import (
    EMNIST_BYCLASS_ASCII,
    EMNIST_BYCLASS_LABELS,
    MNIST_LABELS,
    decode_label,
    get_labels,
    label_category,
    load_official_emnist_mapping,
    parse_mapping_lines,
    verify_emnist_mapping,
)
from src.config import DATA_DIR
from tests.conftest import requires_emnist, requires_emnist_mapping_zip


def official_mapping_text() -> str:
    """The official ``emnist-byclass-mapping.txt`` contents: ``"<index> <ascii>"`` per line (CRLF, as shipped)."""
    codes = [*range(48, 58), *range(65, 91), *range(97, 123)]  # 0-9, A-Z, a-z
    return "".join(f"{index} {code}\r\n" for index, code in enumerate(codes))


# --------------------------------------------------------------------------------------------
# Label tables
# --------------------------------------------------------------------------------------------
def test_mnist_has_the_ten_digit_labels() -> None:
    assert MNIST_LABELS == tuple("0123456789")
    assert get_labels("digits") == get_labels("mnist") == MNIST_LABELS


def test_emnist_byclass_has_62_unique_labels_in_official_order() -> None:
    assert len(EMNIST_BYCLASS_LABELS) == 62
    assert len(set(EMNIST_BYCLASS_LABELS)) == 62, "duplicate labels"
    assert "".join(EMNIST_BYCLASS_LABELS) == string.digits + string.ascii_uppercase + string.ascii_lowercase
    assert get_labels("characters") == get_labels("emnist_byclass") == EMNIST_BYCLASS_LABELS


def test_full_ascii_table_matches_the_official_mapping() -> None:
    """Index i -> ASCII 48+i (digits), 65+(i-10) (A-Z), 97+(i-36) (a-z). The index is NOT the ASCII code."""
    assert len(EMNIST_BYCLASS_ASCII) == 62
    for index, (label, code) in enumerate(zip(EMNIST_BYCLASS_LABELS, EMNIST_BYCLASS_ASCII)):
        if index < 10:
            expected = 48 + index
        elif index < 36:
            expected = 65 + index - 10
        else:
            expected = 97 + index - 36
        assert code == expected, f"class {index}"
        assert ord(label) == code
        assert decode_label(index, "characters") == chr(expected)


@pytest.mark.parametrize(
    "index, label",
    [(0, "0"), (9, "9"), (10, "A"), (11, "B"), (25, "P"), (35, "Z"), (36, "a"), (37, "b"), (50, "o"), (61, "z")],
)
def test_spot_check_character_indices(index: int, label: str) -> None:
    assert decode_label(index, "characters") == label
    assert decode_label(index, "emnist_byclass") == label


@pytest.mark.parametrize("index", range(10))
def test_digit_indices_decode_to_the_same_digit(index: int) -> None:
    assert decode_label(index, "digits") == str(index)


def test_decode_label_accepts_numpy_and_tensor_integers() -> None:
    assert decode_label(np.int64(36), "characters") == "a"
    assert decode_label(torch.tensor(7), "digits") == "7"


@pytest.mark.parametrize("index, mode", [(-1, "characters"), (62, "characters"), (10, "digits"), (100, "digits")])
def test_decode_label_out_of_range_raises_index_error(index: int, mode: str) -> None:
    with pytest.raises(IndexError):
        decode_label(index, mode)


def test_unknown_mode_raises_value_error() -> None:
    with pytest.raises(ValueError):
        get_labels("letters")


@pytest.mark.parametrize("label, category", [("0", "digit"), ("9", "digit"), ("A", "uppercase"), ("Z", "uppercase"),
                                             ("a", "lowercase"), ("z", "lowercase")])
def test_label_category(label: str, category: str) -> None:
    assert label_category(label) == category


def test_every_label_has_the_category_of_its_index_range() -> None:
    categories = [label_category(label) for label in EMNIST_BYCLASS_LABELS]
    assert categories == ["digit"] * 10 + ["uppercase"] * 26 + ["lowercase"] * 26


@pytest.mark.parametrize("label", ["", "AB", "!", "é"])
def test_label_category_rejects_non_alphanumeric_labels(label: str) -> None:
    with pytest.raises(ValueError):
        label_category(label)


# --------------------------------------------------------------------------------------------
# Mapping-file parsing and verification
# --------------------------------------------------------------------------------------------
def test_parse_mapping_lines_reads_the_official_format() -> None:
    mapping = parse_mapping_lines(official_mapping_text().splitlines())
    assert mapping == dict(enumerate(EMNIST_BYCLASS_LABELS))


def test_parse_mapping_lines_skips_blank_lines() -> None:
    assert parse_mapping_lines(["", "0 48", "   ", "10 65", ""]) == {0: "0", 10: "A"}


@pytest.mark.parametrize("lines", [["0 48 1"], ["0"], ["a 48"], ["0 x"], ["-1 48"], ["0 48", "0 49"]])
def test_parse_mapping_lines_rejects_malformed_input(lines: list[str]) -> None:
    with pytest.raises(ValueError):
        parse_mapping_lines(lines)


def test_verify_emnist_mapping_accepts_matching_sources() -> None:
    report = verify_emnist_mapping(dict(enumerate(EMNIST_BYCLASS_LABELS)), list(EMNIST_BYCLASS_LABELS))
    assert report["matches_official"] is True
    assert report["matches_torchvision"] is True
    assert report["mismatches"] == []


def test_verify_emnist_mapping_never_raises_for_missing_sources() -> None:
    report = verify_emnist_mapping(None, None)
    assert report["matches_official"] is None
    assert report["matches_torchvision"] is None
    assert report["mismatches"] == []


def test_verify_emnist_mapping_reports_mismatches() -> None:
    swapped = dict(enumerate(EMNIST_BYCLASS_LABELS))
    swapped[10], swapped[36] = "a", "A"  # upper/lower case swapped
    report = verify_emnist_mapping(swapped, list(EMNIST_BYCLASS_LABELS[:61]))  # torchvision list too short
    assert report["matches_official"] is False
    assert report["matches_torchvision"] is False
    indices = {(m["source"], m["class_index"]) for m in report["mismatches"]}
    assert indices == {("official", 10), ("official", 36), ("torchvision", 61)}


def _make_raw_dir(root: Path) -> Path:
    raw = root / "EMNIST" / "raw"
    raw.mkdir(parents=True)
    return raw


def test_load_official_mapping_from_zip_member(tmp_path: Path) -> None:
    raw = _make_raw_dir(tmp_path)
    with zipfile.ZipFile(raw / "gzip.zip", "w") as archive:
        archive.writestr("gzip/emnist-byclass-mapping.txt", official_mapping_text())
    assert load_official_emnist_mapping(tmp_path) == dict(enumerate(EMNIST_BYCLASS_LABELS))


def test_load_official_mapping_prefers_extracted_text_file(tmp_path: Path) -> None:
    raw = _make_raw_dir(tmp_path)
    (raw / "emnist-byclass-mapping.txt").write_text(official_mapping_text(), encoding="ascii")
    mapping = load_official_emnist_mapping(tmp_path)
    assert verify_emnist_mapping(mapping, None)["matches_official"] is True


def test_load_official_mapping_returns_none_when_absent(tmp_path: Path) -> None:
    assert load_official_emnist_mapping(tmp_path) is None
    raw = _make_raw_dir(tmp_path)
    with zipfile.ZipFile(raw / "gzip.zip", "w") as archive:
        archive.writestr("gzip/something-else.txt", "")
    assert load_official_emnist_mapping(tmp_path) is None


# --------------------------------------------------------------------------------------------
# Real data (skipped when the files are absent; never downloads)
# --------------------------------------------------------------------------------------------
@pytest.mark.data
@requires_emnist_mapping_zip
def test_official_mapping_file_in_downloaded_zip_matches_labels() -> None:
    official = load_official_emnist_mapping(DATA_DIR)
    assert official is not None
    report = verify_emnist_mapping(official, None)
    assert report["matches_official"] is True, report["mismatches"]
    assert [ord(official[i]) for i in range(62)] == list(EMNIST_BYCLASS_ASCII)


@pytest.mark.data
@requires_emnist
def test_torchvision_emnist_classes_match_labels() -> None:
    from src.datasets import open_torchvision_dataset

    dataset = open_torchvision_dataset("characters", DATA_DIR, train=False, download=False)
    report = verify_emnist_mapping(None, dataset.classes)
    assert report["matches_torchvision"] is True, report["mismatches"]
    assert tuple(dataset.classes) == EMNIST_BYCLASS_LABELS
    assert int(dataset.targets.min()) == 0 and int(dataset.targets.max()) == 61
