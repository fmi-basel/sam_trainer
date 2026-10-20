"""Check whether the current train/val split leaks source stacks across both sets.

Reproduces `prepare_data_splits` from `sam_trainer/training.py` for a given config,
then groups filenames back to their source stack ID (stripping augmentation/slice
suffixes) to report train/val stack overlap. Read-only, CPU-only.

Usage:
    python scripts/diagnosis/check_split_leakage.py configs/swi_decoder-only_lr5e-5.yaml
"""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sam_trainer.training import prepare_data_splits, stack_id  # noqa: E402


def main(config_path: str) -> None:
    config = yaml.safe_load(Path(config_path).read_text())
    training_cfg = config["training"]

    images_dir = Path(training_cfg["images_dir"])
    labels_dir = Path(training_cfg["labels_dir"])

    train_images, _, val_images, _ = prepare_data_splits(
        images_dir,
        labels_dir,
        training_cfg["val_split"],
        shuffle=training_cfg.get("shuffle_data", True),
        seed=training_cfg.get("shuffle_seed"),
    )

    train_stacks = {stack_id(p) for p in train_images}
    val_stacks = {stack_id(p) for p in val_images}
    overlap = train_stacks & val_stacks

    print(f"Train tiles: {len(train_images)} ({len(train_stacks)} unique stacks)")
    print(f"Val tiles:   {len(val_images)} ({len(val_stacks)} unique stacks)")
    print(f"Stacks in both train and val: {len(overlap)} / {len(val_stacks)} val stacks")
    if overlap:
        print("LEAKAGE DETECTED. Overlapping stack IDs:")
        for stack in sorted(overlap):
            print(f"  {stack}")
    else:
        print("No stack-level overlap between train and val.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <config.yaml>")
        sys.exit(1)
    main(sys.argv[1])
