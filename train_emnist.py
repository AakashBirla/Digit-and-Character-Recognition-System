"""Train the character recogniser (DigitCNN on EMNIST ByClass, 62 classes: 0-9, A-Z, a-z).

Usage::

    python train_emnist.py                     # 15 epochs, batch 256, auto device
    python train_emnist.py --balanced-loss     # up-weight rare classes
    python train_emnist.py --help              # all options

The first run downloads EMNIST (~560 MB zip) into ``data/``. Writes the best checkpoint (by
validation accuracy) to ``models/emnist_byclass_cnn.pth``. Training history, curves and
official-test-split metrics go to ``results/emnist_byclass/``.
"""

import sys

from src.training import run_cli


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, train, evaluate; returns the exit code."""
    return run_cli("emnist_byclass", default_epochs=15, default_batch_size=256, argv=argv)


if __name__ == "__main__":
    sys.exit(main())
