"""Preprocessing utilities for training data."""

import logging
from pathlib import Path
from typing import TypedDict

import numpy as np

from sam_trainer.config import PreprocessingConfig
from sam_trainer.io import get_image_paths, read_image, write_image

logger = logging.getLogger(__name__)


class PreprocessingStats(TypedDict):
    n_pairs: int
    max_shape: tuple[int, ...]
    patch_shape: tuple[int, int]
    output_dir: Path
    output_images_dir: Path
    output_labels_dir: Path


def _build_stem_map(paths: list[Path]) -> dict[str, Path]:
    return {p.stem: p for p in paths}


def _pair_paths(images_dir: Path, labels_dir: Path) -> list[tuple[str, Path, Path]]:
    image_paths = [p for p in get_image_paths(images_dir) if p.is_file()]
    label_paths = [p for p in get_image_paths(labels_dir) if p.is_file()]

    if len(image_paths) == 0:
        raise ValueError(f"No image files found in {images_dir}")
    if len(label_paths) == 0:
        raise ValueError(f"No label files found in {labels_dir}")

    image_map = _build_stem_map(image_paths)
    label_map = _build_stem_map(label_paths)

    image_stems = set(image_map.keys())
    label_stems = set(label_map.keys())

    only_images = sorted(image_stems - label_stems)
    only_labels = sorted(label_stems - image_stems)

    if only_images or only_labels:
        details = []
        if only_images:
            details.append(f"images-only stems (first 10): {only_images[:10]}")
        if only_labels:
            details.append(f"labels-only stems (first 10): {only_labels[:10]}")
        raise ValueError(
            "Image/label filenames must match by stem; " + "; ".join(details)
        )

    stems = sorted(image_stems)
    return [(stem, image_map[stem], label_map[stem]) for stem in stems]


def _center_pad_to_shape(
    arr: np.ndarray,
    target_shape: tuple[int, ...],
    pad_value: int,
) -> np.ndarray:
    if arr.ndim != len(target_shape):
        raise ValueError(
            f"Array ndim mismatch: got {arr.ndim}, target has {len(target_shape)} dims"
        )

    pad_width: list[tuple[int, int]] = []
    for current, target in zip(arr.shape, target_shape):
        if current > target:
            raise ValueError(
                f"Cannot pad from shape {arr.shape} to smaller target {target_shape}"
            )
        total = target - current
        before = total // 2
        after = total - before
        pad_width.append((before, after))

    return np.pad(arr, pad_width=pad_width, mode="constant", constant_values=pad_value)


def run_preprocessing(config: PreprocessingConfig) -> PreprocessingStats:
    """Run final preprocessing stage before training."""
    if config.mode != "pad_to_max":
        raise ValueError(f"Unsupported preprocessing mode: {config.mode}")

    pairs = _pair_paths(config.input_images_dir, config.input_labels_dir)
    logger.info("Found %d image-label pairs for preprocessing", len(pairs))

    max_shape: tuple[int, ...] | None = None

    for stem, img_path, label_path in pairs:
        image = read_image(img_path)
        label = read_image(label_path)

        if image.shape != label.shape:
            raise ValueError(
                f"Shape mismatch for '{stem}': image {image.shape} vs label {label.shape}"
            )

        if max_shape is None:
            max_shape = tuple(int(v) for v in image.shape)
        else:
            if image.ndim != len(max_shape):
                raise ValueError(
                    "All samples must have the same ndim for pad_to_max; "
                    f"got {image.ndim} and {len(max_shape)}"
                )
            max_shape = tuple(
                max(curr, int(v)) for curr, v in zip(max_shape, image.shape)
            )

    if max_shape is None:
        raise ValueError("No data found for preprocessing")

    output_dir = config.output_dir
    if output_dir is None:
        raise ValueError("output_dir must be resolved before preprocessing")

    output_images_dir = output_dir / "images"
    output_labels_dir = output_dir / "labels"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Padding all samples to max shape %s with center alignment", max_shape)

    for stem, img_path, label_path in pairs:
        image = read_image(img_path)
        label = read_image(label_path)

        padded_image = _center_pad_to_shape(image, max_shape, pad_value=0)
        padded_label = _center_pad_to_shape(label, max_shape, pad_value=0)

        write_image(padded_image, output_images_dir / stem, "tif")
        write_image(padded_label, output_labels_dir / stem, "tif")

    patch_shape = tuple(int(v) for v in max_shape[-2:])

    return {
        "n_pairs": len(pairs),
        "max_shape": max_shape,
        "patch_shape": patch_shape,
        "output_dir": output_dir,
        "output_images_dir": output_images_dir,
        "output_labels_dir": output_labels_dir,
    }
