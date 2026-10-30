"""Class-index -> character tables for MNIST and EMNIST ByClass.

A network only outputs class *indices*; this module turns them into characters.

* MNIST: index ``i`` is simply the digit ``str(i)``.
* EMNIST ByClass (62 classes): indices 0-9 are digits, 10-35 are ``A``-``Z`` and 36-61 are
  ``a``-``z``. The official ``emnist-byclass-mapping.txt`` stores lines ``"<index> <ascii code>"``
  (e.g. ``"10 65"`` means index 10 is ``chr(65) == "A"``). Note that the class index is *not*
  the ASCII code.

The official mapping file ships inside ``data/EMNIST/raw/gzip.zip`` (member
``gzip/emnist-byclass-mapping.txt``); :func:`load_official_emnist_mapping` reads it from there so
:func:`verify_emnist_mapping` can prove that the constants below match the dataset.
"""

from __future__ import annotations

import logging
import operator
import string
import zipfile
from collections.abc import Iterable, Sequence
from pathlib import Path

from src.config import get_spec

logger = logging.getLogger(__name__)

MNIST_LABELS: tuple[str, ...] = tuple(string.digits)
EMNIST_BYCLASS_LABELS: tuple[str, ...] = tuple(string.digits + string.ascii_uppercase + string.ascii_lowercase)
EMNIST_BYCLASS_ASCII: tuple[int, ...] = tuple(ord(label) for label in EMNIST_BYCLASS_LABELS)
CLASS_MAPPING_VERSION = "emnist-byclass-mapping.txt (official EMNIST v1.0, 2017) / MNIST digits 0-9"

_LABELS_BY_DATASET: dict[str, tuple[str, ...]] = {
    "mnist": MNIST_LABELS,
    "emnist_byclass": EMNIST_BYCLASS_LABELS,
}


def get_labels(key_or_mode: str) -> tuple[str, ...]:
    """Return the ordered class labels for a dataset key or mode (``"digits"`` / ``"characters"``)."""
    return _LABELS_BY_DATASET[get_spec(key_or_mode).key]


def decode_label(class_index: int, key_or_mode: str) -> str:
    """Map a class index to its character.

    Raises:
        IndexError: if ``class_index`` is outside ``[0, num_classes)``.
        TypeError: if ``class_index`` is not an integer.
    """
    index = operator.index(class_index)  # accepts int, numpy ints and integer 0-d tensors
    labels = get_labels(key_or_mode)
    if not 0 <= index < len(labels):
        raise IndexError(f"Class index {index} is out of range for {len(labels)} classes.")
    return labels[index]


def label_category(label: str) -> str:
    """Return ``"digit"``, ``"uppercase"`` or ``"lowercase"`` for a single ASCII alphanumeric label."""
    if len(label) == 1:
        if label in string.digits:
            return "digit"
        if label in string.ascii_uppercase:
            return "uppercase"
        if label in string.ascii_lowercase:
            return "lowercase"
    raise ValueError(f"Label {label!r} is not a single ASCII digit or letter.")


def parse_mapping_lines(lines: Iterable[str]) -> dict[int, str]:
    """Parse EMNIST mapping lines ``"<index> <ascii code>"`` into ``{index: character}``.

    Blank lines are skipped. Raises ``ValueError`` on malformed lines, duplicate indices or
    invalid character codes.
    """
    mapping: dict[int, str] = {}
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"Malformed mapping line {line_number}: {raw_line!r} (expected '<index> <ascii>').")
        try:
            index, code = int(parts[0]), int(parts[1])
        except ValueError as exc:
            raise ValueError(f"Malformed mapping line {line_number}: {raw_line!r} (non-integer value).") from exc
        if index < 0 or not 0 <= code <= 0x10FFFF:
            raise ValueError(f"Malformed mapping line {line_number}: {raw_line!r} (value out of range).")
        if index in mapping:
            raise ValueError(f"Duplicate class index {index} on mapping line {line_number}.")
        mapping[index] = chr(code)
    return mapping


def load_official_emnist_mapping(data_dir: Path | str, split: str = "byclass") -> dict[int, str] | None:
    """Load the official EMNIST mapping for ``split`` from the downloaded data.

    Looks for ``<data_dir>/EMNIST/raw/emnist-<split>-mapping.txt`` first, then for the member
    ``gzip/emnist-<split>-mapping.txt`` inside ``<data_dir>/EMNIST/raw/gzip.zip`` (where
    torchvision leaves it). Returns ``None`` if neither is available.
    """
    raw_dir = Path(data_dir) / "EMNIST" / "raw"
    filename = f"emnist-{split}-mapping.txt"

    text_file = raw_dir / filename
    if text_file.is_file():
        logger.debug("Reading EMNIST mapping from %s", text_file)
        return parse_mapping_lines(text_file.read_text(encoding="ascii").splitlines())

    zip_path = raw_dir / "gzip.zip"
    if zip_path.is_file():
        member = f"gzip/{filename}"
        try:
            with zipfile.ZipFile(zip_path) as archive:
                data = archive.read(member)
        except KeyError:
            logger.warning("%s does not contain %s", zip_path, member)
            return None
        except zipfile.BadZipFile:
            logger.warning("%s is not a valid zip archive; cannot read the EMNIST mapping", zip_path)
            return None
        logger.debug("Reading EMNIST mapping from %s:%s", zip_path, member)
        return parse_mapping_lines(data.decode("ascii").splitlines())

    logger.info("Official EMNIST mapping not found under %s", raw_dir)
    return None


def verify_emnist_mapping(
    official: dict[int, str] | None,
    torchvision_classes: Sequence[str] | None,
) -> dict:
    """Compare the official mapping and torchvision's ``.classes`` against ``EMNIST_BYCLASS_LABELS``.

    Returns ``{"matches_official": bool | None, "matches_torchvision": bool | None,
    "mismatches": [...], "num_classes": 62}``. A ``None`` input yields ``None`` for that check
    (never raises). Each mismatch is ``{"source", "class_index", "expected", "found"}``.
    """
    expected = EMNIST_BYCLASS_LABELS
    mismatches: list[dict] = []

    def compare(source: str, found_by_index: dict[int, str]) -> bool:
        ok = True
        for index in sorted(set(range(len(expected))) | set(found_by_index)):
            want = expected[index] if index < len(expected) else None
            got = found_by_index.get(index)
            if want != got:
                ok = False
                mismatches.append({"source": source, "class_index": index, "expected": want, "found": got})
        return ok

    matches_official = None if official is None else compare("official", dict(official))
    matches_torchvision = (
        None if torchvision_classes is None else compare("torchvision", dict(enumerate(torchvision_classes)))
    )
    return {
        "matches_official": matches_official,
        "matches_torchvision": matches_torchvision,
        "mismatches": mismatches,
        "num_classes": len(expected),
    }
