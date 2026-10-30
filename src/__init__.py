"""Handwritten digit (MNIST) and character (EMNIST ByClass) recognition with PyTorch.

Package layout:

* ``config``         - paths, dataset specifications and the mode -> dataset mapping
* ``utils``          - device selection, seeding and logging helpers
* ``class_mapping``  - class-index -> character tables (and official EMNIST mapping checks)
* ``models``         - the ``DigitCNN`` convolutional network
* ``checkpoints``    - safe (``weights_only=True``) checkpoint save/load with JSON-safe metadata
* ``datasets``       - dataset loading (upright orientation), splits, augmentation, fast batch loaders
* ``preprocessing``  - canvas/photo image -> 28x28 tensor that matches the training distribution
* ``predictor``      - model registry and prediction service used by the web app
* ``training`` / ``evaluation`` - training loop and test-set evaluation

Submodules are intentionally not imported here so that ``import src`` stays cheap.
"""

__version__ = "2.0.0"
