"""Train the digit recogniser (DigitCNN on MNIST, 10 classes).

Usage::

    python train_mnist.py                      # 10 epochs, batch 128, auto device
    python train_mnist.py --epochs 3 --device cpu
    python train_mnist.py --help               # all options

Writes the best checkpoint (by validation accuracy) to ``models/mnist_cnn.pth``. Training
history, curves and official-test-split metrics go to ``results/mnist/``.
"""

import sys

from src.training import run_cli


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, train, evaluate; returns the exit code."""
    return run_cli("mnist", default_epochs=10, default_batch_size=128, argv=argv)


if __name__ == "__main__":
    sys.exit(main())
