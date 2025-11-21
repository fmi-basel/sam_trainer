from collections import Counter
from pathlib import Path

import numpy as np
import tifffile
import torch
from micro_sam.training.training import default_sam_loader
from torch_em.data import MinInstanceSampler

labels_dir = Path("dat/augmented_training_data/labels")
images_dir = Path("dat/augmented_training_data/images")


def count_instances(mask):
    # assumes 0 is background, instances encoded with unique integers
    vals = np.unique(mask)
    return (vals > 0).sum()


def summarize_distribution():
    counts = []
    for p in labels_dir.glob("*.tif"):
        m = tifffile.imread(p)
        if m.ndim > 2:  # if stacked, pick first slice or max projection
            m = m[0]
        counts.append(count_instances(m))
    ctr = Counter(counts)
    print("Instance count distribution (instances_per_image: frequency):")
    for k in sorted(ctr):
        print(f"{k}: {ctr[k]}")
    print("Images with 0 instances:", ctr.get(0, 0))
    print("Mean instances/image:", np.mean(counts))
    print("Median:", np.median(counts))


def simulate_empty_rate(
    n_samples=200, patch_shape=(512, 512), min_instances=1, min_size=15
):
    # Build loader (train=True to enable sampling)
    loader = default_sam_loader(
        raw_paths=[str(p) for p in images_dir.glob("*.tif")],
        label_paths=[str(p) for p in labels_dir.glob("*.tif")],
        is_train=True,
        sampler=MinInstanceSampler(min_instances, min_size=min_size),
        raw_key=None,
        label_key=None,
        batch_size=4,
        patch_shape=patch_shape,
        with_segmentation_decoder=True,
        train_instance_segmentation_only=False,
        n_samples=n_samples,
        num_workers=0,
        raw_transform=None,
    )
    empty_batches = 0
    checked = 0
    for x, y in loader:
        # y shape (B, 1, H, W)
        if torch.sum(y > 0) == 0:
            empty_batches += 1
        checked += 1
        if checked >= 200:  # limit scan
            break
    print(
        f"Checked {checked} batches; empty batches: {empty_batches} ({empty_batches / checked * 100:.2f}%)"
    )


if __name__ == "__main__":
    summarize_distribution()
    simulate_empty_rate()
