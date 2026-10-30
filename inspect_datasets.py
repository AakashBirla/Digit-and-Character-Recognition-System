"""Inspect MNIST and EMNIST ByClass before training: orientation, label mapping, statistics, geometry.

What it checks
--------------
* Both official splits load with the expected shape ``(N, 28, 28)``, dtype ``uint8`` and label range.
* EMNIST orientation: ``src.datasets.load_split`` returns the raw torchvision tensor transposed
  (upright). An automatic check classifies EMNIST digits against MNIST digit templates in several
  orientations - the upright one should win by a wide margin. Sample grids let you confirm visually.
* Class mapping: the constants in ``src.class_mapping`` vs torchvision's ``.classes`` vs the
  official ``emnist-byclass-mapping.txt`` (read from inside ``data/EMNIST/raw/gzip.zip``).
* Normalisation statistics: training-split pixel mean/std vs the constants in ``src.config``.
* Glyph geometry on a random sample: longest side, bounding-box centre, intensity centre of mass
  and stroke width - the facts the inference preprocessing has to reproduce.

Outputs (in ``--results-dir``)::

    dataset_inspection.json
    samples_mnist.png
    samples_emnist_byclass.png                    (upright - what the model is trained on)
    samples_emnist_byclass_raw_untransposed.png   (raw torchvision tensor, for comparison)

Usage::

    python inspect_datasets.py [--data-dir data] [--results-dir results] [--no-download]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision

from src.checkpoints import sanitize_metadata
from src.class_mapping import (
    EMNIST_BYCLASS_LABELS,
    MNIST_LABELS,
    get_labels,
    load_official_emnist_mapping,
    verify_emnist_mapping,
)
from src.config import DATA_DIR, EMNIST_BYCLASS_SPEC, MNIST_SPEC, PROJECT_ROOT, RESULTS_DIR, DatasetSpec
from src.datasets import compute_mean_std, load_split, open_torchvision_dataset, save_sample_grid, to_upright
from src.utils import setup_logging

logger = logging.getLogger("inspect_datasets")

BOX_THRESHOLD = 0.1  # pixel > 0.1 (of 1.0) counts as ink for bounding boxes (includes blurred edges)
STROKE_THRESHOLD = 0.5  # binary mask used for the stroke-width estimate
STATS_TOLERANCE = 5e-4  # allowed |measured - configured| for mean/std
CENTRE_TOLERANCE = 0.15  # px
STROKE_TOLERANCE = 0.1  # px
# Longest glyph side expected in the training data (pixels > BOX_THRESHOLD, i.e. including EMNIST's blur
# halo). Preprocessing reproduces it for MNIST, but targets 1 px less for EMNIST (crisp canvas strokes have
# no halo) - see the target_box comment in src/config.py.
EXPECTED_LONGEST_SIDE = {"mnist": 20, "emnist_byclass": 24}


# ------------------------------------------------------------------------------------------------
# Measurements
# ------------------------------------------------------------------------------------------------
def split_summary(images: torch.Tensor, targets: torch.Tensor, spec: DatasetSpec) -> dict:
    """Shape/dtype/range facts and class counts for one split."""
    labels = get_labels(spec.key)
    counts = torch.bincount(targets, minlength=spec.num_classes).tolist()
    order = np.argsort(counts)
    summary = {
        "num_samples": int(images.shape[0]),
        "image_shape": list(images.shape[1:]),
        "dtype": str(images.dtype).replace("torch.", ""),
        "pixel_min": int(images.min()),
        "pixel_max": int(images.max()),
        "label_min": int(targets.min()),
        "label_max": int(targets.max()),
        "classes_present": int(sum(c > 0 for c in counts)),
        "class_counts": {labels[i]: counts[i] for i in range(spec.num_classes)},
        "min_class": {"label": labels[order[0]], "count": counts[order[0]]},
        "max_class": {"label": labels[order[-1]], "count": counts[order[-1]]},
        "imbalance_ratio": round(counts[order[-1]] / max(counts[order[0]], 1), 2),
        "digit_fraction": round(sum(counts[:10]) / images.shape[0], 4),
    }
    summary["valid"] = (
        summary["image_shape"] == [28, 28]
        and summary["dtype"] == "uint8"
        and 0 <= summary["label_min"]
        and summary["label_max"] < spec.num_classes
        and summary["classes_present"] == spec.num_classes
    )
    return summary


def stroke_width(mask: np.ndarray) -> float | None:
    """Estimate stroke width as ``2 * area / perimeter`` of a binary mask (contours via RETR_LIST)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
    return 2.0 * float(mask.sum()) / perimeter if perimeter > 0 else None


def geometry_stats(images: torch.Tensor, spec: DatasetSpec, num_samples: int, seed: int) -> dict:
    """Glyph size / position / stroke statistics on a seeded random sample of the (upright) images."""
    rng = np.random.default_rng(seed)
    chosen = rng.choice(images.shape[0], size=min(num_samples, images.shape[0]), replace=False)
    pixels = images[torch.from_numpy(np.sort(chosen))].numpy().astype(np.float64) / 255.0  # (n, 28, 28)

    ink = pixels > BOX_THRESHOLD
    has_ink = ink.any(axis=(1, 2))
    pixels, ink = pixels[has_ink], ink[has_ink]
    rows, cols = ink.any(axis=2), ink.any(axis=1)
    size = pixels.shape[1]
    top, bottom = rows.argmax(axis=1), size - 1 - rows[:, ::-1].argmax(axis=1)
    left, right = cols.argmax(axis=1), size - 1 - cols[:, ::-1].argmax(axis=1)
    height, width = bottom - top + 1, right - left + 1
    longest = np.maximum(height, width)
    bbox_cx, bbox_cy = (left + right) / 2.0, (top + bottom) / 2.0

    ys, xs = np.mgrid[0:size, 0:size]
    mass = pixels.sum(axis=(1, 2))
    com_x = (pixels * xs).sum(axis=(1, 2)) / mass
    com_y = (pixels * ys).sum(axis=(1, 2)) / mass

    widths = [stroke_width((p > STROKE_THRESHOLD).astype(np.uint8)) for p in pixels]
    widths = np.array([w for w in widths if w is not None])

    def mean_std(values: np.ndarray) -> dict:
        return {"mean": round(float(values.mean()), 3), "std": round(float(values.std()), 3)}

    stats = {
        "num_samples": int(pixels.shape[0]),
        "seed": seed,
        "methods": {
            "longest_side": f"max(h, w) of the bounding box of pixels > {BOX_THRESHOLD}",
            "bbox_centre": "((left + right) / 2, (top + bottom) / 2) in pixel-index coordinates",
            "centre_of_mass": "intensity-weighted mean of np.mgrid pixel-index coordinates",
            "stroke_width": f"2 * area / perimeter of the pixels > {STROKE_THRESHOLD} mask "
            "(area = pixel count, perimeter = sum of cv2.arcLength over RETR_LIST contours)",
        },
        "longest_side": {
            "median": float(np.median(longest)),
            "mean": round(float(longest.mean()), 3),
            "p5": float(np.percentile(longest, 5)),
            "p95": float(np.percentile(longest, 95)),
            "expected": EXPECTED_LONGEST_SIDE[spec.key],
            "fraction_equal_expected": round(float((longest == EXPECTED_LONGEST_SIDE[spec.key]).mean()), 4),
            "spec_target_box": spec.target_box,
        },
        "bbox_centre": {"x": mean_std(bbox_cx), "y": mean_std(bbox_cy)},
        "centre_of_mass": {"x": mean_std(com_x), "y": mean_std(com_y)},
        "stroke_width": {
            "median": round(float(np.median(widths)), 3),
            "mean": round(float(widths.mean()), 3),
            "spec": spec.stroke_width,
            "diff_median_minus_spec": round(float(np.median(widths)) - spec.stroke_width, 3),
        },
        "spec_centering": spec.centering,
    }

    # Is the glyph centred the way the spec says (and therefore the way preprocessing reproduces it)?
    if spec.centering == "center_of_mass":
        target = size / 2.0  # 14.0
        centre = stats["centre_of_mass"]
    else:
        target = (size - 1) / 2.0  # 13.5
        centre = stats["bbox_centre"]
    stats["centring_target"] = target
    stats["centring_matches_spec"] = bool(
        abs(centre["x"]["mean"] - target) <= CENTRE_TOLERANCE and abs(centre["y"]["mean"] - target) <= CENTRE_TOLERANCE
    )
    stats["longest_side_matches_spec"] = stats["longest_side"]["median"] == EXPECTED_LONGEST_SIDE[spec.key]
    stats["stroke_width_matches_spec"] = abs(stats["stroke_width"]["diff_median_minus_spec"]) <= STROKE_TOLERANCE
    return stats


def orientation_check(
    mnist_images: torch.Tensor,
    mnist_targets: torch.Tensor,
    emnist_images: torch.Tensor,
    emnist_targets: torch.Tensor,
    num_samples: int,
    seed: int,
) -> dict:
    """Classify EMNIST digits with MNIST class-mean templates under several orientations.

    MNIST is upright by construction, so the orientation of the (already "upright") EMNIST digits
    that matches the templates best is the correct one. Expect "upright" to win clearly.
    """
    templates = torch.stack([mnist_images[mnist_targets == d].float().mean(0) for d in range(10)])  # (10, 28, 28)
    templates = templates.flatten(1)
    templates = templates / templates.norm(dim=1, keepdim=True)

    digit_idx = torch.nonzero(emnist_targets < 10).squeeze(1)
    generator = torch.Generator().manual_seed(seed)
    pick = digit_idx[torch.randperm(digit_idx.numel(), generator=generator)[:num_samples]]
    digits = emnist_images[pick].float()
    labels = emnist_targets[pick]

    variants = {
        "upright (as loaded)": digits,
        "transposed (raw torchvision tensor)": digits.transpose(1, 2),
        "horizontal flip": digits.flip(2),
        "vertical flip": digits.flip(1),
        "rotated 90 deg clockwise": torch.rot90(digits, k=-1, dims=(1, 2)),
        "rotated 90 deg counter-clockwise": torch.rot90(digits, k=1, dims=(1, 2)),
        "rotated 180 deg": torch.rot90(digits, k=2, dims=(1, 2)),
    }
    accuracy = {}
    for name, batch in variants.items():
        flat = batch.flatten(1)
        flat = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-8)
        predictions = (flat @ templates.T).argmax(dim=1)
        accuracy[name] = round(float((predictions == labels).float().mean()), 4)
    best = max(accuracy, key=accuracy.get)
    return {
        "method": "nearest MNIST class-mean template (cosine similarity) on EMNIST digit samples",
        "num_samples": int(labels.numel()),
        "accuracy_by_orientation": accuracy,
        "best_orientation": best,
        "upright_is_best": best == "upright (as loaded)",
    }


def mnist_mapping_check() -> dict:
    """torchvision MNIST classes look like '0 - zero'; compare their leading digit with MNIST_LABELS."""
    torchvision_labels = [name.split(" ")[0] for name in torchvision.datasets.MNIST.classes]
    return {"matches_torchvision": torchvision_labels == list(MNIST_LABELS), "torchvision_classes": torchvision_labels}


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


# ------------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect MNIST and EMNIST ByClass (orientation, mapping, stats).")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="dataset root (default: data/)")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help="output directory (default: results/)")
    parser.add_argument("--no-download", action="store_true", help="fail instead of downloading missing datasets")
    parser.add_argument("--geometry-samples", type=int, default=20_000, help="images sampled for geometry stats")
    parser.add_argument("--seed", type=int, default=0, help="seed for sampling and sample grids")
    return parser.parse_args(argv)


def inspect_dataset(spec: DatasetSpec, args: argparse.Namespace) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Load both splits of one dataset and compute every per-dataset statistic. Returns (report, train x, train y)."""
    download = not args.no_download
    print(f"\n=== {spec.display_name} ===")
    train_x, train_y = load_split(spec, args.data_dir, train=True, download=download)
    test_x, test_y = load_split(spec, args.data_dir, train=False, download=download)

    mean, std = compute_mean_std(train_x)
    normalization = {
        "measured_mean": round(mean, 5),
        "measured_std": round(std, 5),
        "spec_mean": spec.mean,
        "spec_std": spec.std,
        "mean_diff": round(mean - spec.mean, 5),
        "std_diff": round(std - spec.std, 5),
    }
    normalization["matches_spec"] = (
        abs(normalization["mean_diff"]) <= STATS_TOLERANCE and abs(normalization["std_diff"]) <= STATS_TOLERANCE
    )
    report = {
        "key": spec.key,
        "mode": spec.mode,
        "display_name": spec.display_name,
        "num_classes": spec.num_classes,
        "transpose_raw": spec.transpose_raw,
        "splits": {"train": split_summary(train_x, train_y, spec), "test": split_summary(test_x, test_y, spec)},
        "normalization": normalization,
        "geometry": geometry_stats(train_x, spec, args.geometry_samples, args.seed),
    }
    return report, train_x, train_y


def print_dataset_summary(report: dict) -> None:
    for split in ("train", "test"):
        s = report["splits"][split]
        print(
            f"  {split:5s}: {s['num_samples']:>7,} images {s['image_shape']} {s['dtype']}, pixels {s['pixel_min']}-"
            f"{s['pixel_max']}, labels {s['label_min']}-{s['label_max']} ({s['classes_present']} classes) "
            f"{'OK' if s['valid'] else 'INVALID'}"
        )
    s = report["splits"]["train"]
    print(
        f"  class counts (train): min {s['min_class']['count']:,} ('{s['min_class']['label']}'), "
        f"max {s['max_class']['count']:,} ('{s['max_class']['label']}'), imbalance x{s['imbalance_ratio']}, "
        f"digits {s['digit_fraction']:.1%}"
    )
    n = report["normalization"]
    print(
        f"  mean/std (train): {n['measured_mean']:.4f} / {n['measured_std']:.4f}"
        f"   spec {n['spec_mean']} / {n['spec_std']}   diff {n['mean_diff']:+.5f} / {n['std_diff']:+.5f}"
        f"  {'OK' if n['matches_spec'] else 'DIFFERS'}"
    )
    g = report["geometry"]
    ls, sw = g["longest_side"], g["stroke_width"]
    print(
        f"  geometry ({g['num_samples']:,} samples): longest side median {ls['median']:.0f} px "
        f"(p5 {ls['p5']:.0f}, p95 {ls['p95']:.0f}; expected {ls['expected']}, "
        f"preprocessing target {ls['spec_target_box']}) "
        f"{'OK' if g['longest_side_matches_spec'] else 'DIFFERS'}"
    )
    print(
        f"    bbox centre ({g['bbox_centre']['x']['mean']:.2f}, {g['bbox_centre']['y']['mean']:.2f}) "
        f"± ({g['bbox_centre']['x']['std']:.2f}, {g['bbox_centre']['y']['std']:.2f}); "
        f"COM ({g['centre_of_mass']['x']['mean']:.2f}, {g['centre_of_mass']['y']['mean']:.2f}) "
        f"± ({g['centre_of_mass']['x']['std']:.2f}, {g['centre_of_mass']['y']['std']:.2f}); "
        f"spec centring '{g['spec_centering']}' at {g['centring_target']} "
        f"{'OK' if g['centring_matches_spec'] else 'DIFFERS'}"
    )
    print(
        f"    stroke width median {sw['median']:.2f} px (spec {sw['spec']}, diff {sw['diff_median_minus_spec']:+.2f}) "
        f"{'OK' if g['stroke_width_matches_spec'] else 'DIFFERS'}"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(logging.WARNING)  # keep the console readable; this script prints its own summary
    results_dir: Path = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
        "data_dir": relative(args.data_dir),
        "datasets": {},
        "mapping": {},
        "artifacts": {},
    }

    # ---- MNIST -------------------------------------------------------------------------------
    mnist_report, mnist_x, mnist_y = inspect_dataset(MNIST_SPEC, args)
    report["datasets"]["mnist"] = mnist_report
    print_dataset_summary(mnist_report)
    path = save_sample_grid(
        mnist_x, mnist_y, MNIST_LABELS, results_dir / "samples_mnist.png", seed=args.seed, title="MNIST train (upright)"
    )
    report["artifacts"]["samples_mnist"] = relative(path)
    report["mapping"]["mnist"] = mnist_mapping_check()

    # ---- EMNIST ByClass ------------------------------------------------------------------------
    emnist_report, emnist_x, emnist_y = inspect_dataset(EMNIST_BYCLASS_SPEC, args)
    report["datasets"]["emnist_byclass"] = emnist_report
    print_dataset_summary(emnist_report)
    path = save_sample_grid(
        emnist_x,
        emnist_y,
        EMNIST_BYCLASS_LABELS,
        results_dir / "samples_emnist_byclass.png",
        seed=args.seed,
        title="EMNIST ByClass train - upright (as used for training)",
    )
    report["artifacts"]["samples_emnist_byclass"] = relative(path)

    # Raw tensor exactly as torchvision stores it (same sample indices, so the grids are comparable).
    raw_dataset = open_torchvision_dataset(EMNIST_BYCLASS_SPEC, args.data_dir, train=True, download=False)
    raw_x = raw_dataset.data
    check_idx = torch.from_numpy(np.random.default_rng(args.seed).choice(raw_x.shape[0], 20_000, replace=False))
    emnist_report["load_split_equals_transposed_raw"] = bool(
        torch.equal(to_upright(raw_x[check_idx], EMNIST_BYCLASS_SPEC), emnist_x[check_idx])
    )
    path = save_sample_grid(
        raw_x,
        raw_dataset.targets,
        EMNIST_BYCLASS_LABELS,
        results_dir / "samples_emnist_byclass_raw_untransposed.png",
        seed=args.seed,
        title="EMNIST ByClass train - RAW torchvision tensor (not transposed)",
    )
    report["artifacts"]["samples_emnist_byclass_raw_untransposed"] = relative(path)
    del raw_dataset, raw_x

    # Mapping: constants vs torchvision .classes vs the official mapping file.
    test_dataset = open_torchvision_dataset(EMNIST_BYCLASS_SPEC, args.data_dir, train=False, download=False)
    torchvision_classes = list(test_dataset.classes)
    del test_dataset
    official = load_official_emnist_mapping(args.data_dir, "byclass")
    emnist_mapping = verify_emnist_mapping(official, torchvision_classes)
    emnist_mapping["official_mapping_found"] = official is not None
    emnist_mapping["ascii_ranges"] = {"0-9": "48-57", "10-35": "65-90 (A-Z)", "36-61": "97-122 (a-z)"}
    report["mapping"]["emnist_byclass"] = emnist_mapping

    orientation = orientation_check(mnist_x, mnist_y, emnist_x, emnist_y, num_samples=5000, seed=args.seed)
    report["orientation_check"] = orientation

    # ---- Summary -----------------------------------------------------------------------------
    critical = {
        "mnist splits valid": all(s["valid"] for s in mnist_report["splits"].values()),
        "emnist splits valid": all(s["valid"] for s in emnist_report["splits"].values()),
        "mnist labels match torchvision": report["mapping"]["mnist"]["matches_torchvision"],
        "emnist labels match torchvision": emnist_mapping["matches_torchvision"] is True,
        "emnist labels match official mapping": emnist_mapping["matches_official"] is not False,
        "emnist load_split == transpose(raw)": emnist_report["load_split_equals_transposed_raw"],
        "emnist upright orientation best": orientation["upright_is_best"],
    }
    report["critical_checks"] = critical
    report["all_critical_checks_passed"] = all(critical.values())

    out_path = results_dir / "dataset_inspection.json"
    out_path.write_text(json.dumps(sanitize_metadata(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")

    print("\n=== Class mapping ===")
    print(f"  MNIST labels vs torchvision: {'OK' if critical['mnist labels match torchvision'] else 'MISMATCH'}")
    official_status = {True: "OK", False: "MISMATCH", None: "not found (skipped)"}[emnist_mapping["matches_official"]]
    torchvision_status = "OK" if emnist_mapping["matches_torchvision"] else "MISMATCH"
    print(f"  EMNIST ByClass labels vs torchvision .classes: {torchvision_status}")
    print(f"  EMNIST ByClass labels vs official emnist-byclass-mapping.txt: {official_status}")
    if emnist_mapping["mismatches"]:
        print(f"  mismatches: {emnist_mapping['mismatches'][:10]}")
    print("\n=== EMNIST orientation check (MNIST-template accuracy on EMNIST digits) ===")
    for name, acc in orientation["accuracy_by_orientation"].items():
        print(f"  {name:36s} {acc:6.1%}")
    print(f"  best: {orientation['best_orientation']}")
    print("\n=== Critical checks ===")
    for name, ok in critical.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("\nSaved:")
    for artifact in (*report["artifacts"].values(), relative(out_path)):
        print(f"  {artifact}")
    return 0 if report["all_critical_checks_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
