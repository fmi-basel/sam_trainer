"""Shared utilities for SAM inference across different input formats.

This module provides common functions for model loading, image processing,
and segmentation that are used by various inference scripts.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from micro_sam.automatic_segmentation import (
    automatic_instance_segmentation,
    get_predictor_and_segmenter,
)
from micro_sam.instance_segmentation import InstanceSegmentationWithDecoder
from micro_sam.util import get_sam_model

from sam_trainer.utils.logging import get_logger
from sam_trainer.utils.normalization import PercentileNormalizer

logger = get_logger(__name__)


def load_model_with_decoder(
    model_type: str,
    device: str,
    model_path: Optional[str] = None,
    use_amg: bool = False,
    is_tiled: bool = False,
    **amg_kwargs,
) -> Tuple:
    """Load an exported model with either decoder or AMG segmentation.

    Args:
        model_type: SAM model type (e.g., 'vit_b_lm', 'vit_l_lm')
        device: Device to load model on ('cuda' or 'cpu')
        model_path: Path to a custom model checkpoint (.pt file). If None, the
            pre-trained micro-SAM model for `model_type` is downloaded/used from cache.
        use_amg: If True, use AMG instead of decoder-based segmentation
        is_tiled: If True, build a segmenter that supports tiled image embeddings
            (`TiledInstanceSegmentationWithDecoder`/`TiledAutomaticMaskGenerator`) instead
            of the whole-image one. Required for tiled inference — the whole-image segmenter
            can't be tiled after the fact by passing `tile_shape` to `initialize()`. Must be
            set whenever a `tile_shape` will be passed to `segment_image()`.
        **amg_kwargs: Additional kwargs for AMG (pred_iou_thresh, stability_score_thresh, etc.)

    Returns:
        Tuple of (predictor, segmenter)

    Raises:
        Exception: If model loading fails
    """
    mode = "AMG" if use_amg else "AIS (decoder-based)"
    logger.info(f"Loading model with {mode} segmentation" + (" (tiled)" if is_tiled else ""))
    segmentation_mode = "amg" if use_amg else "ais"

    if model_path is not None:
        state = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "decoder_state" in state:
            # Exported instance-segmentation model (encoder + UNETR decoder).
            # get_predictor_and_segmenter passes the checkpoint to get_sam_model which
            # calls sam.load_state_dict on the full dict — failing on decoder_state.
            # Fix: load SAM encoder with flexible_load_checkpoint (ignores unknown keys),
            # then supply the pre-loaded state so get_predictor_and_segmenter can pick
            # up decoder_state without re-loading the file.
            predictor = get_sam_model(
                model_type=model_type,
                device=device,
                checkpoint_path=model_path,
                flexible_load_checkpoint=True,
            )
            _, segmenter = get_predictor_and_segmenter(
                model_type=model_type,
                predictor=predictor,
                state=state,
                device=device,
                segmentation_mode=segmentation_mode,
                is_tiled=is_tiled,
                **amg_kwargs,
            )
            return predictor, segmenter

    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_type,
        checkpoint=model_path,
        device=device,
        segmentation_mode=segmentation_mode,
        is_tiled=is_tiled,
        **amg_kwargs,
    )
    return predictor, segmenter


def resolve_channel_index(
    channel_labels: list,
    channel: Optional[str],
    wavelength_ids: Optional[list] = None,
) -> int:
    """Resolve a channel specification to a 0-based integer index.

    Lookup order for string names:
    1. ``channel_labels`` (omero channel names, e.g. 'BF', 'DAPI')
    2. ``wavelength_ids`` (e.g. 'A02_C01')

    Args:
        channel_labels: List of channel label strings from image metadata (e.g., from NGIO).
            Can be empty or None if metadata is unavailable.
        channel: Channel specification. One of:
            - None: use the first channel (index 0, default behaviour)
            - An integer string like "0" or "2": use that channel index
            - A channel name like "BF" or "DAPI": look up in channel_labels
            - A wavelength ID like "A02_C01": look up in wavelength_ids
        wavelength_ids: Optional list of wavelength ID strings from image metadata
            (e.g., from NGIO ``image_data.wavelength_ids``). Can be empty or None.

    Returns:
        0-based channel index.

    Raises:
        ValueError: If the channel name is not found or the index is out of range.
    """
    if channel is None:
        return 0

    # Try integer index first
    try:
        idx = int(channel)
        n = (
            len(channel_labels)
            if channel_labels
            else (len(wavelength_ids) if wavelength_ids else 0)
        )
        if n and idx >= n:
            raise ValueError(f"Channel index {idx} out of range for {n} channels")
        return idx
    except ValueError as int_err:
        if "out of range" in str(int_err):
            raise

    # Look up by name: try channel_labels first, then wavelength_ids
    if channel_labels and channel in channel_labels:
        return channel_labels.index(channel)

    if wavelength_ids and channel in wavelength_ids:
        return wavelength_ids.index(channel)

    # Not found — build a helpful error message
    available: list[str] = []
    if channel_labels:
        available += [f"channel_labels: {channel_labels}"]
    if wavelength_ids:
        available += [f"wavelength_ids: {wavelength_ids}"]
    if not available:
        raise ValueError(
            f"Cannot resolve channel name '{channel}': no channel metadata available. "
            "Use a numeric index instead."
        )
    raise ValueError(
        f"Channel '{channel}' not found. Available — " + "; ".join(available)
    )


def _to_2d(image: np.ndarray, channel_index: int = 0) -> np.ndarray:
    """Reduce an ND image to 2D by squeezing singletons and selecting a channel.

    OME-Zarr images arriving from NGIO iterators can have extra leading dimensions
    (e.g. C, Z) even after MIP. SAM only accepts 2D (H, W) or 3D (Z, H, W) input.

    Args:
        image: Input image array of any dimensionality.
        channel_index: Index of the channel to select when the image has multiple
            channels after squeezing. Default: 0 (first channel).
    """
    original_shape = image.shape
    # Squeeze all size-1 dimensions first
    image = np.squeeze(image)
    if image.ndim > 2:
        if channel_index >= image.shape[0]:
            raise ValueError(
                f"Channel index {channel_index} out of range for shape {image.shape} "
                f"(original: {original_shape})"
            )
        logger.debug(
            f"Image shape {original_shape} → squeezed to {image.shape}, "
            f"selecting channel {channel_index} along axis 0"
        )
        image = image[channel_index]
        # Handle remaining extra dims (e.g., Z axis after channel selection)
        while image.ndim > 2:
            image = image[0]
    elif image.ndim != len(original_shape):
        logger.debug(f"Image shape {original_shape} → squeezed to {image.shape}")
    return image


def _merge_along_seam_band(band: np.ndarray, min_run: int, union) -> None:
    """Union label pairs from long, straight runs of exactly-2-labels-present columns.

    `band` is a strip of pixels straddling a tile seam, shape (band_width, n_positions) —
    i.e. each column `i` is the cross-section of pixel values at position `i` along the seam,
    spanning a small margin on both sides of it. The split boundary from a seam artifact
    doesn't necessarily sit exactly on the seam pixel (empirically it can land a few pixels
    off, presumably wherever the two tiles' stitched decoder outputs happen to disagree most),
    so scanning a margin band and checking "exactly 2 distinct labels present in this
    cross-section" catches it even when the two label regions aren't directly touching at the
    exact seam line. A short run of a given pair is normal (two genuinely distinct cells
    happen to pass close to the same seam), but a long straight run of the *same* pair is the
    signature of one cell getting cut by the tiling seam.
    """
    n_positions = band.shape[1]
    pairs: list = [None] * n_positions
    for i in range(n_positions):
        labels = np.unique(band[:, i])
        labels = labels[labels != 0]
        if len(labels) == 2:
            pairs[i] = (int(labels[0]), int(labels[1]))

    run_start = 0
    run_pair = pairs[0] if n_positions else None
    for i in range(1, n_positions + 1):
        current = pairs[i] if i < n_positions else None
        if current == run_pair:
            continue
        if run_pair is not None and i - run_start >= min_run:
            union(*run_pair)
        run_start = i
        run_pair = current


def _merge_tile_seam_splits(
    masks: np.ndarray, tile_shape: Tuple[int, int], margin: int = 16, min_run: int = 10
) -> np.ndarray:
    """Merge instances that tiled AIS inference split along tile-grid seams.

    `TiledInstanceSegmentationWithDecoder` stitches each tile's decoder output (foreground/
    center-distance/boundary-distance maps) with a hard cut at the tile's inner boundary —
    the halo only gives each tile's encoder extra context, predictions from adjacent tiles
    are never blended. A cell straddling a seam can get slightly different center/boundary
    predictions on each side, and the watershed-style instance decoding then treats that
    mismatch as a real boundary, splitting one cell into two near the seam line — the actual
    split boundary can land a few pixels off the exact seam coordinate (confirmed empirically
    2026-08-19: one real split's true label transition sat 2px from the nominal seam, which a
    naive single-pixel-either-side check missed entirely), so this scans a small band around
    each seam rather than just the immediate seam pixel.

    Confirmed empirically on real Jessica plate inference output: several masks had long,
    straight bands (10-280+ px) where exactly 2 labels co-occurred near a tile-grid seam,
    clearly distinct from the noise floor of genuinely separate adjacent cells that happen to
    pass close to the same seam.

    Tile seams are deterministic — the tiling grid always starts at (0, 0) (see
    `TiledInstanceSegmentationWithDecoder.initialize` → `blocking([0, 0], original_size,
    tile_shape)`), so seam lines fall at exact multiples of `tile_shape` regardless of halo.

    Args:
        masks: Instance segmentation labels from tiled AIS inference.
        tile_shape: The tile shape used for inference (same value passed to `initialize`).
        margin: Half-width (pixels) of the band scanned on each side of a seam line for
            candidate split-label pairs.
        min_run: Minimum contiguous run length (pixels) of the same 2-label pair along a
            seam band to treat as a split rather than coincidental cell-to-cell proximity.

    Returns:
        Masks with seam-split instances merged back into a single label (labels unchanged
        for anything not connected to a seam split).
    """
    if masks.max() == 0:
        return masks

    parent = {int(label): int(label) for label in np.unique(masks) if label != 0}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_a] = root_b

    height, width = masks.shape
    for seam in range(tile_shape[0], height, tile_shape[0]):
        lo, hi = max(0, seam - margin), min(height, seam + margin)
        _merge_along_seam_band(masks[lo:hi, :], min_run, union)
    for seam in range(tile_shape[1], width, tile_shape[1]):
        lo, hi = max(0, seam - margin), min(width, seam + margin)
        _merge_along_seam_band(masks[:, lo:hi].T, min_run, union)

    roots = {label: find(label) for label in parent}
    if all(root == label for label, root in roots.items()):
        return masks

    merged = masks.copy()
    for label, root in roots.items():
        if root != label:
            merged[masks == label] = root
    return merged


def segment_image(
    image: np.ndarray,
    predictor,
    segmenter,
    use_amg: bool = False,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    generate_kwargs: Optional[Dict[str, Any]] = None,
    channel_index: int = 0,
    normalize: bool = True,
    normalize_lower_percentile: float = 1.0,
    normalize_upper_percentile: float = 99.5,
    invert: bool = False,
) -> np.ndarray:
    """Run instance segmentation on a single image.

    Args:
        image: Input image as numpy array (any dimensionality; reduced to 2D internally)
        predictor: SAM predictor
        segmenter: SAM segmenter (AMG or InstanceSegmentationWithDecoder)
        use_amg: Whether using AMG mode (affects processing)
        tile_shape: Optional tile shape for large images (e.g., (512, 512))
        halo: Optional overlap for stitching tiles (e.g., (64, 64))
        generate_kwargs: Optional parameters for generate() method (decoder thresholds)
        channel_index: Index of the channel to select from multi-channel patches.
            Default: 0 (first channel).
        normalize: Apply the same percentile normalization used during training before
            segmentation. Must match training settings, otherwise the model sees a
            different input distribution than it was trained on.
        normalize_lower_percentile: Lower percentile for intensity clipping.
        normalize_upper_percentile: Upper percentile for intensity clipping.
        invert: Invert intensities after normalization. Must match training settings
            (e.g. for dark-foreground brightfield images).

    Returns:
        Instance segmentation masks as 2D numpy array with integer labels
    """
    generate_kwargs = generate_kwargs or {}
    image = _to_2d(image, channel_index=channel_index)

    if normalize:
        image = PercentileNormalizer(
            normalize_lower_percentile, normalize_upper_percentile, invert=invert
        )(image)

    if isinstance(segmenter, InstanceSegmentationWithDecoder) and not use_amg:
        # generate() returns a 2D integer label array directly (output_mode="instance_segmentation")
        # Without tile_shape, initialize() feeds the whole image through SAM's encoder, which
        # resizes the longest side to 1024px — for images much larger than the training patch
        # size, this shrinks objects far below the scale the decoder was trained on. Tiling
        # keeps each tile near the encoder's native resolution instead. Only works if `segmenter`
        # was built with `is_tiled=True` (see load_model_with_decoder) — the whole-image
        # InstanceSegmentationWithDecoder.initialize() doesn't accept tile_shape/halo at all.
        if tile_shape is not None:
            segmenter.initialize(image, tile_shape=tile_shape, halo=halo or (0, 0))
        else:
            segmenter.initialize(image)
        masks = segmenter.generate(**generate_kwargs)
        if tile_shape is not None:
            masks = _merge_tile_seam_splits(masks, tile_shape)
    else:
        # AMG-based segmentation: use automatic_instance_segmentation
        masks = automatic_instance_segmentation(
            predictor=predictor,
            segmenter=segmenter,
            input_path=image,
            ndim=2,
            tile_shape=tile_shape,
            halo=halo,
            verbose=False,
        )

    return masks


def postprocess_masks(
    masks: np.ndarray,
    min_area: int = 0,
    border_margin: int = 0,
    max_instances: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """Filter and relabel instance masks based on size and position.

    Args:
        masks: Instance masks as 2D array with integer labels
        min_area: Minimum area in pixels (instances smaller are removed)
        border_margin: Margin in pixels from border (touching instances removed)
        max_instances: Maximum number of instances to keep (largest by area)

    Returns:
        Tuple of (filtered_masks, num_removed) where:
            - filtered_masks: Relabeled masks array
            - num_removed: Number of instances that were filtered out
    """
    label_ids = np.unique(masks)
    label_ids = label_ids[label_ids != 0]

    if label_ids.size == 0:
        return masks.astype(np.uint16, copy=False), 0

    kept: list[Tuple[int, int]] = []
    height, width = masks.shape

    for label_id in label_ids:
        region_mask = masks == label_id
        area = int(region_mask.sum())

        # Filter by minimum area
        if min_area > 0 and area < min_area:
            continue

        # Filter by border touching
        if border_margin > 0:
            ys, xs = np.nonzero(region_mask)
            if ys.size == 0:
                continue
            touches_border = (
                (ys < border_margin).any()
                or (ys >= height - border_margin).any()
                or (xs < border_margin).any()
                or (xs >= width - border_margin).any()
            )
            if touches_border:
                continue

        kept.append((area, label_id))

    if not kept:
        return np.zeros_like(masks, dtype=np.uint16), label_ids.size

    # Sort by area (largest first)
    kept.sort(key=lambda item: item[0], reverse=True)

    # Limit number of instances
    if max_instances is not None and max_instances > 0:
        kept = kept[:max_instances]

    # Relabel sequentially
    filtered = np.zeros_like(masks, dtype=np.uint16)
    for new_idx, (_, label_id) in enumerate(kept, start=1):
        filtered[masks == label_id] = new_idx

    removed = label_ids.size - len(kept)
    return filtered, removed
