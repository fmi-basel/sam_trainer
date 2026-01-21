"""
Script to preprocess Zarr datasets for SAM training.
Performs:
1. Polarity Correction (using Ground Truth labels)
2. Percentile Normalization (Robust to padding)
3. One-time conversion to uint8 (Speeds up training)
"""

import logging
from pathlib import Path

import numpy as np
import zarr
from skimage import exposure

# Setup logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)


def preprocess_zarr(
    input_path: Path,
    output_path: Path,
    raw_key: str,
    label_key: str,
    target_dark_object: bool = True,
    enhance_contrast: bool = False,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.5,
    limit_fraction: float = None,
):
    logger.info(f"Preprocessing {input_path} -> {output_path}")

    # Open Input
    try:
        store_in = zarr.open(str(input_path), mode="r")
        if raw_key not in store_in or label_key not in store_in:
            logger.error(f"Missing keys {raw_key} or {label_key} in {input_path}")
            return

        raw_arr_in = store_in[raw_key]
        lbl_arr_in = store_in[label_key]

    except Exception as e:
        logger.error(f"Failed to open input: {e}")
        return

    # Create Output
    store_out = zarr.open(str(output_path), mode="w")

    # Init Output Arrays
    # We save processed raw as uint8 (optimized)
    # Labels remain same type (usually uint32/16)

    shape = raw_arr_in.shape
    if limit_fraction:
        n_samples = int(shape[0] * limit_fraction)
        shape = (n_samples,) + shape[1:]
        logger.info(f"Limiting to {limit_fraction:.0%} of data: {n_samples} samples")
    else:
        n_samples = shape[0]

    chunks = raw_arr_in.chunks

    # Compressor
    compressor = zarr.Blosc(cname="lz4", clevel=5, shuffle=zarr.Blosc.SHUFFLE)

    raw_arr_out = store_out.create_dataset(
        raw_key,
        shape=shape,
        chunks=chunks,
        dtype="uint8",
        compressor=compressor,
        overwrite=True,
    )

    lbl_arr_out = store_out.create_dataset(
        label_key,
        shape=shape,
        chunks=chunks,
        dtype=lbl_arr_in.dtype,
        compressor=compressor,
        overwrite=True,
    )

    # Process Loop
    # n_samples is already set above
    inverted_count = 0

    logger.info(f"Processing {n_samples} samples...")
    log_interval = max(1, n_samples // 20)  # Log every ~5%

    for i in range(n_samples):
        # Log progress periodically
        if i % log_interval == 0:
            logger.info(f"  Progress: {i}/{n_samples} ({i / n_samples * 100:.1f}%)")

        # Load
        raw = raw_arr_in[i]
        lbl = lbl_arr_in[i]

        # 1. Determine Polarity (Label Guided)
        processed_raw = raw.copy()

        # Create foreground/background masks from labels
        fg_mask = lbl > 0  # Foreground = instances
        bg_mask = (lbl == 0) & (raw > 0)  # Background = non-padded areas without labels

        # Let's Normalize first (Robust to padding), then Invert, then Mask Padding back to 0

        # --- Normalization ---
        arr = raw.astype(np.float32)
        valid_mask = arr > 0

        if valid_mask.any():
            lo, hi = np.percentile(
                arr[valid_mask], [lower_percentile, upper_percentile]
            )
        else:
            lo, hi = np.percentile(arr, [lower_percentile, upper_percentile])

        if hi <= lo:
            lo, hi = float(arr.min()), float(arr.max())
            if hi == lo:
                hi = lo + 1.0

        arr = np.clip(arr, lo, hi)
        arr = (arr - lo) / (hi - lo)

        # --- Inversion (on normalized 0-1 float) ---
        # Check if we need to invert based on foreground vs background brightness
        if np.any(fg_mask) and np.any(bg_mask):
            mean_fg = np.mean(arr[fg_mask])
            mean_bg = np.mean(arr[bg_mask])
            is_bright_obj = mean_fg > mean_bg

            if target_dark_object and is_bright_obj:
                arr = 1.0 - arr
                inverted_count += 1

        # --- CLAHE (Optional) ---
        if enhance_contrast:
            # arr is float 0-1
            arr = exposure.equalize_adapthist(arr, clip_limit=0.03)

        # --- Finalize ---
        arr = (arr * 255.0).astype(np.uint8)

        # Restore Padding (0 in original = 0 in output)
        # This is critical for sparse data
        arr[~valid_mask] = 0

        # Write
        raw_arr_out[i] = arr
        lbl_arr_out[i] = lbl  # Copy label as is

    logger.info(
        f"Finished. Inverted {inverted_count}/{n_samples} ({inverted_count / n_samples:.1%}) samples."
    )


if __name__ == "__main__":
    # Config
    train_in = Path(
        "/tachyon/groups/scratch/gmicro_prefect/ggrossha/ggrossha_SWI/training_data/accumulated_train.zarr"
    )
    val_in = Path(
        "/tachyon/groups/scratch/gmicro_prefect/ggrossha/ggrossha_SWI/training_data/accumulated_val.zarr"
    )

    logger.info(f"Loading Training data: {train_in}")
    logger.info(f"Loading Validation data: {val_in}")

    # Output names
    train_out = train_in.parent / f"{train_in.stem}_preprocessed_noclahe.zarr"
    val_out = val_in.parent / f"{val_in.stem}_preprocessed_noclahe.zarr"

    preprocess_zarr(train_in, train_out, "0", "labels/mask/0", enhance_contrast=False)
    preprocess_zarr(val_in, val_out, "0", "labels/mask/0", enhance_contrast=False)
