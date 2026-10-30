"""Evaluate trained checkpoints on the official test split (plus a canvas round-trip check).

Usage::

    python evaluate.py                                   # both modes, default checkpoints in models/
    python evaluate.py --mode characters                 # EMNIST ByClass only
    python evaluate.py --mode digits --checkpoint path/to/model.pth
    python evaluate.py --max-test-samples 1000           # quick check on a seeded subset

Each evaluation writes ``test_metrics.json``, ``classification_report.txt``,
``per_class_metrics.csv``, ``confusion_matrix.csv``, ``confusion_matrix.png`` and
``roundtrip.json`` to ``results/<dataset>/``. It also stores compact test metrics in the
checkpoint, which the web app displays. This happens only for a full test-split evaluation.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.checkpoints import CheckpointError, load_checkpoint
from src.config import DATA_DIR, MODES, RESULTS_DIR, get_spec
from src.evaluation import display_path, evaluate_checkpoint, format_evaluation_summary
from src.utils import describe_device, resolve_device, setup_logging

logger = logging.getLogger("evaluate")


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def _non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface of ``evaluate.py``."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained checkpoints on the official MNIST / EMNIST ByClass test split.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python evaluate.py\n"
            "  python evaluate.py --mode digits --max-test-samples 1000 --roundtrip-samples 0\n"
            "  python evaluate.py --mode characters --checkpoint models/emnist_byclass_cnn.pth"
        ),
    )
    parser.add_argument("--mode", choices=(*MODES, "all"), default="all",
                        help="which model(s) to evaluate; 'all' skips missing checkpoints (default: %(default)s)")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="checkpoint path override (requires --mode digits or --mode characters)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help=f"dataset directory (default: {display_path(DATA_DIR)})")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help=f"output directory; with --mode all a per-dataset sub-folder is used "
                             f"(default: {display_path(RESULTS_DIR)}/<dataset>)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="auto = CUDA when available, else CPU (default: %(default)s)")
    parser.add_argument("--batch-size", type=_positive_int, default=1024,
                        help="evaluation batch size (default: %(default)s)")
    parser.add_argument("--max-test-samples", type=_positive_int, default=None,
                        help="evaluate a seeded subset only; partial results are never stored in the checkpoint "
                             "and by default go to results/<dataset>/subset/")
    parser.add_argument("--roundtrip-samples", type=_non_negative_int, default=2000,
                        help="simulated canvas drawings for the preprocessing round-trip; 0 skips it "
                             "(default: %(default)s)")
    parser.add_argument("--no-update-checkpoint", action="store_true",
                        help="do not write test metrics into the checkpoint")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Evaluate the requested checkpoint(s); returns the process exit code."""
    setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.checkpoint is not None and args.mode == "all":
        parser.error("--checkpoint evaluates a single model: add --mode digits or --mode characters")
    try:
        device = resolve_device(args.device)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Using device: {describe_device(device)}", flush=True)

    modes = MODES if args.mode == "all" else (args.mode,)
    evaluated, failed = 0, 0
    for mode in modes:
        spec = get_spec(mode)
        checkpoint = args.checkpoint if args.checkpoint is not None else spec.default_checkpoint
        if not checkpoint.is_file():
            message = f"Checkpoint not found: {display_path(checkpoint)}. Train it with: {spec.train_command}"
            if len(modes) > 1:
                logger.warning("Skipping %s mode. %s", mode, message)
                continue
            print(f"Error: {message}", file=sys.stderr)
            return 1

        results_dir = args.results_dir
        if results_dir is not None and len(modes) > 1:
            results_dir = results_dir / spec.results_subdir
        try:
            stored_dataset = load_checkpoint(checkpoint)["dataset"]
            if stored_dataset != spec.key:
                raise CheckpointError(
                    f"{display_path(checkpoint)} is a {get_spec(stored_dataset).display_name} checkpoint, "
                    f"but --mode {mode} expects {spec.display_name}."
                )
            print(f"\nEvaluating {display_path(checkpoint)} ...", flush=True)
            outcome = evaluate_checkpoint(
                checkpoint,
                data_dir=args.data_dir,
                device=device,
                results_dir=results_dir,
                batch_size=args.batch_size,
                max_test_samples=args.max_test_samples,
                update_checkpoint=not args.no_update_checkpoint,
                roundtrip_samples=args.roundtrip_samples,
            )
        except (CheckpointError, RuntimeError) as exc:  # unusable checkpoint, data download/validation failure
            print(f"Error evaluating {mode} model: {exc}", file=sys.stderr)
            failed += 1
            continue
        evaluated += 1
        print(format_evaluation_summary(outcome), flush=True)

    if evaluated == 0 and failed == 0:
        print("No checkpoints found. Train a model first: python train_mnist.py / python train_emnist.py",
              file=sys.stderr)
        return 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
