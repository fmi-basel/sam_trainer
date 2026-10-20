"""One-off fix for images_unstacked / masks_unstacked filename mismatch.

images_unstacked mixes two naming conventions across stacks
(stack_BF_min_ome_s{1,2,3,5,8}_sub20_sliceNNNN.tif vs.
stack_BF_mip_min_s{30-34}_sub20_sliceNNNN.tif), while masks_unstacked uses
one uniform scheme (mask_s{N}_sliceNNNN.tif). Plain lexicographic sort of
each directory independently groups these differently, so positional
pairing (get_image_paths + zip, used by both run_augmentation and
prepare_data_splits) matched the wrong stack's mask to 160/200 images.

This script writes copies of the mask files into a new directory, renamed
to share the exact stem of their corresponding image (matched by the
(stack, slice) key extracted from each filename), so downstream code's
existing stem-based/positional pairing becomes correct without any code
changes. See docs/swi_training_overview.md Known issues for the general
fix (Fix A) deferred to a future release.

Usage:
    python scripts/fix_unstacked_mask_pairing.py
"""

import re
import shutil
from pathlib import Path

IMAGES_DIR = Path("W:/scratch/gmicro_ipa/ggrossha/ancneagu/swi_annotations/images_unstacked")
MASKS_DIR = Path("W:/scratch/gmicro_ipa/ggrossha/ancneagu/swi_annotations/masks_unstacked")
OUTPUT_DIR = Path("W:/scratch/gmicro_ipa/ggrossha/ancneagu/swi_annotations/masks_unstacked_matched")

KEY_RE = re.compile(r"s(\d+).*?slice(\d+)")


def key(path: Path) -> tuple[int, int]:
    match = KEY_RE.search(path.stem)
    if not match:
        raise ValueError(f"Could not extract (stack, slice) key from {path.name}")
    return int(match.group(1)), int(match.group(2))


def main() -> None:
    images = list(IMAGES_DIR.glob("*.tif"))
    masks = list(MASKS_DIR.glob("*.tif"))

    image_keys = [key(p) for p in images]
    mask_by_key = {key(p): p for p in masks}

    if len(mask_by_key) != len(masks):
        raise ValueError("Duplicate (stack, slice) keys found in masks_unstacked")
    if len(set(image_keys)) != len(images):
        raise ValueError("Duplicate (stack, slice) keys found in images_unstacked")
    if set(image_keys) != set(mask_by_key):
        missing = set(image_keys) - set(mask_by_key)
        extra = set(mask_by_key) - set(image_keys)
        raise ValueError(f"Key sets differ. Missing masks: {missing}, extra masks: {extra}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for image_path, k in zip(images, image_keys):
        mask_path = mask_by_key[k]
        shutil.copy2(mask_path, OUTPUT_DIR / f"{image_path.stem}{mask_path.suffix}")

    print(f"Wrote {len(images)} matched mask files to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
