"""Quick diagnostic to check label file shapes."""

from pathlib import Path

import numpy as np

from sam_trainer.io import read_image

labels_dir = Path("dat/augmented_training_data/labels")
images_dir = Path("dat/augmented_training_data/images")

print("Checking first 5 label files:")
for i, label_path in enumerate(sorted(labels_dir.glob("*.tif"))[:5]):
    label = read_image(label_path)
    image_path = images_dir / label_path.name
    image = read_image(image_path)

    print(f"\n{label_path.name}:")
    print(f"  Label shape: {label.shape}, dtype: {label.dtype}")
    print(f"  Image shape: {image.shape}, dtype: {image.dtype}")
    print(f"  Label unique values: {np.unique(label)[:10]}...")  # first 10
    print(f"  Label ndim: {label.ndim}")

    if i >= 4:
        break
