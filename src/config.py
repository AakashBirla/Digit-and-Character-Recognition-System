"""Central configuration: project paths and per-dataset specifications.

Every path used by the project derives from the constants defined here (CLI flags may
override them). The :class:`DatasetSpec` records hold the facts that training, evaluation
and inference must agree on - number of classes, normalisation statistics, orientation of
the raw data, and the glyph geometry that preprocessing has to reproduce.

The two application *modes* map onto datasets as follows::

    "digits"     -> "mnist"           (10 classes, 0-9)
    "characters" -> "emnist_byclass"  (62 classes, 0-9 A-Z a-z)

The geometry values were measured on the real training data (see ``inspect_datasets.py``):

* MNIST glyphs are scaled so their longest side is 20 px and are centred by their intensity
  centre of mass at pixel (14.0, 14.0).
* EMNIST glyphs (after transposing the raw data upright) fill roughly a 24 px box and are
  centred by their bounding box at pixel (13.5, 13.5). The 24 px includes the soft halo that
  EMNIST's Gaussian blur adds around every stroke; a crisp canvas drawing has no halo, so
  preprocessing scales it to 23 px instead. On simulated canvas drawings of validation images
  (held-out train split, never the test split) with the trained model this gave ~2 points higher
  accuracy than 24 px (and 22 px was worse again).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = PROJECT_ROOT / "models"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
FRONTEND_DIR: Path = PROJECT_ROOT / "frontend"
DEBUG_DIR: Path = RESULTS_DIR / "debug"  # preprocessed inputs saved by `app.py --debug-preprocessing`

IMAGE_SIZE: int = 28


@dataclass(frozen=True)
class DatasetSpec:
    """Immutable description of one supported dataset and how its images look."""

    key: str  # "mnist" | "emnist_byclass"
    mode: str  # "digits" | "characters"
    display_name: str  # "MNIST" | "EMNIST ByClass"
    num_classes: int  # 10 | 62
    mean: float  # pixel mean over the training split (pixels scaled to [0, 1])
    std: float  # pixel standard deviation over the training split
    transpose_raw: bool  # raw torchvision tensor needs data.transpose(1, 2) to be upright
    target_box: int  # longest glyph side (px) inside the 28x28 frame
    centering: str  # "center_of_mass" | "bounding_box"
    stroke_width: float  # median stroke width (px) in 28x28 space
    default_checkpoint: Path  # where training saves / the app looks for the model
    results_subdir: str  # sub-folder of RESULTS_DIR for this dataset's artefacts

    @property
    def train_command(self) -> str:
        """Shell command that trains the model for this dataset (used in user-facing messages)."""
        return f"python {_TRAIN_SCRIPTS[self.key]}"


_TRAIN_SCRIPTS: dict[str, str] = {"mnist": "train_mnist.py", "emnist_byclass": "train_emnist.py"}

MNIST_SPEC = DatasetSpec(
    key="mnist",
    mode="digits",
    display_name="MNIST",
    num_classes=10,
    mean=0.1307,
    std=0.3081,
    transpose_raw=False,
    target_box=20,
    centering="center_of_mass",
    stroke_width=2.44,
    default_checkpoint=MODELS_DIR / "mnist_cnn.pth",
    results_subdir="mnist",
)

EMNIST_BYCLASS_SPEC = DatasetSpec(
    key="emnist_byclass",
    mode="characters",
    display_name="EMNIST ByClass",
    num_classes=62,
    mean=0.1736,
    std=0.3317,
    transpose_raw=True,
    target_box=23,  # measured extent is 24 px incl. the blur halo; 23 px fits crisp drawings best
    centering="bounding_box",
    stroke_width=2.80,
    default_checkpoint=MODELS_DIR / "emnist_byclass_cnn.pth",
    results_subdir="emnist_byclass",
)

DATASET_SPECS: dict[str, DatasetSpec] = {spec.key: spec for spec in (MNIST_SPEC, EMNIST_BYCLASS_SPEC)}
MODE_TO_DATASET: dict[str, str] = {"digits": "mnist", "characters": "emnist_byclass"}
MODES: tuple[str, ...] = ("digits", "characters")


def get_spec(key_or_mode: str | DatasetSpec) -> DatasetSpec:
    """Return the :class:`DatasetSpec` for a dataset key ("mnist") or a mode ("digits").

    Matching ignores surrounding whitespace and case. A ``DatasetSpec`` is returned unchanged.

    Raises:
        ValueError: if the value is neither a known dataset key nor a known mode.
    """
    if isinstance(key_or_mode, DatasetSpec):
        return key_or_mode
    if isinstance(key_or_mode, str):
        name = key_or_mode.strip().lower()
        if name in DATASET_SPECS:
            return DATASET_SPECS[name]
        if name in MODE_TO_DATASET:
            return DATASET_SPECS[MODE_TO_DATASET[name]]
    valid = ", ".join(repr(v) for v in (*MODES, *DATASET_SPECS))
    raise ValueError(f"Unknown dataset or mode {key_or_mode!r}. Expected one of: {valid}.")
