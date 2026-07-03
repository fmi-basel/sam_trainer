"""Intensity normalization shared between training and inference.

Training and inference must apply identical normalization, otherwise the
decoder sees a systematically different input distribution than it was
trained on. Keep this in one place and import it from both.
"""

import numpy as np


class PercentileNormalizer:
    """Rescale intensities to uint8 using percentiles computed over non-zero pixels."""

    def __init__(self, lower: float, upper: float):
        self.lower = lower
        self.upper = upper

    def __call__(self, raw):
        arr = np.asarray(raw, dtype=np.float32)

        # Robust normalization: Calculate percentiles only on non-padded (non-zero) pixels
        # to avoid skewing contrast due to zero-padding.
        mask = arr > 0
        if mask.any():
            lo, hi = np.percentile(arr[mask], [self.lower, self.upper])
        else:
            # Fallback if image is all zeros
            lo, hi = np.percentile(arr, [self.lower, self.upper])

        if hi <= lo:
            lo = float(arr.min())
            hi = float(arr.max())
            if hi == lo:
                hi = lo + 1.0
        arr = np.clip(arr, lo, hi)
        arr = (arr - lo) / (hi - lo)
        arr = (arr * 255.0).astype(np.uint8)
        return arr
