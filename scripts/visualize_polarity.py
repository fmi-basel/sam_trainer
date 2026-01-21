from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr
from scipy.stats import skew


def analyze_and_visualize(path_str, raw_key, label_key, output_dir_str):
    print(f"\nAnalyzing {path_str}...")
    path = Path(path_str)
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        store = zarr.open(str(path), mode="r")
        if raw_key not in store or label_key not in store:
            print("Missing keys")
            return

        raw_arr = store[raw_key]
        lbl_arr = store[label_key]

        # Analyze first 50 samples or so
        num_samples = min(50, raw_arr.shape[0])

        results = []

        print(f"Scanning {num_samples} samples for polarity issues...")

        for idx in range(num_samples):
            raw = raw_arr[idx]
            lbl = lbl_arr[idx]

            # Mask out padding (exact zeros)
            valid_mask = raw > 0
            if not np.any(valid_mask):
                continue

            pixels = raw[valid_mask]

            # 1. Calculate Histogram Skewness
            # Brightfield (Dark obj on Bright BG) -> Mass at high values, tail at low -> Negative Skew
            # Inverted (Bright obj on Dark BG) -> Mass at low values, tail at high -> Positive Skew
            sk = skew(pixels)

            # 2. Calculate Label-Based Contrast (Ground Truth)
            fg_mask = lbl > 0
            bg_mask = (lbl == 0) & (raw > 0)

            if np.any(fg_mask) and np.any(bg_mask):
                mean_fg = np.mean(raw[fg_mask])
                mean_bg = np.mean(raw[bg_mask])
                contrast_diff = (
                    mean_fg - mean_bg
                )  # Negative = Dark Object, Positive = Bright Object
            else:
                contrast_diff = np.nan

            results.append(
                {
                    "id": idx,
                    "skew": sk,
                    "contrast_diff": contrast_diff,
                    "is_bright_object": contrast_diff > 0
                    if not np.isnan(contrast_diff)
                    else False,
                    "raw": raw,
                    "lbl": lbl,
                }
            )

        # Correlation Analysis
        valid_res = [r for r in results if not np.isnan(r["contrast_diff"])]
        correct_predictions = 0

        print("\n--- Correlation Analysis (Skewness vs GT Contrast) ---")
        print(
            f"{'ID':<5} | {'Skew':<8} | {'GT Contrast':<12} | {'Prediction':<15} | {'Actual Type':<15}"
        )
        print("-" * 70)

        for r in valid_res:
            # Predict: Negative Skew = Dark Object (Normal), Positive Skew = Bright Object (Inverted)
            pred_inverted = r["skew"] > 0
            is_inverted = r["is_bright_object"]

            if pred_inverted == is_inverted:
                correct_predictions += 1

            pred_str = "Bright Obj" if pred_inverted else "Dark Obj"
            act_str = "Bright Obj" if is_inverted else "Dark Obj"

            print(
                f"{r['id']:<5} | {r['skew']:<8.2f} | {r['contrast_diff']:<12.2f} | {pred_str:<15} | {act_str:<15}"
            )

            # Save interesting cases (Top 3 Bright, Top 3 Dark)

        accuracy = correct_predictions / len(valid_res) if valid_res else 0
        print(f"\nSkewness Heuristic Accuracy: {accuracy:.2%}")

        # Save visual examples
        # Sort by contrast to find extreme inverted vs extreme normal
        sorted_res = sorted(valid_res, key=lambda x: x["contrast_diff"])

        # Save top 3 Dark Objects (Normal)
        save_visuals(sorted_res[:3], output_dir, "normal_dark_obj")

        # Save top 3 Bright Objects (Inverted)
        save_visuals(sorted_res[-3:], output_dir, "inverted_bright_obj")

    except Exception:
        import traceback

        traceback.print_exc()


def save_visuals(samples, output_dir, prefix):
    for i, item in enumerate(samples):
        raw = item["raw"]
        lbl = item["lbl"]

        # Normalize for png
        raw_float = raw.astype(float)
        # Robust min/max
        valid = raw_float[raw_float > 0]
        p2, p98 = np.percentile(valid, [2, 98])
        raw_norm = np.clip((raw_float - p2) / (p98 - p2), 0, 1)

        # Create overlay
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(raw_norm, cmap="gray")
        plt.title(
            f"ID {item['id']} Raw\nSkew: {item['skew']:.2f} (Pred: {'Bright' if item['skew'] > 0 else 'Dark'})"
        )
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(raw_norm, cmap="gray")
        plt.imshow(lbl > 0, cmap="jet", alpha=0.3)
        plt.title(
            f"GT Contrast: {item['contrast_diff']:.1f}\nType: {'Bright' if item['is_bright_object'] else 'Dark'}"
        )
        plt.axis("off")

        out_path = output_dir / f"{prefix}_{i}_id{item['id']}.png"
        plt.savefig(out_path)
        plt.close()
        print(f"Saved {out_path}")


if __name__ == "__main__":
    train_zarr = r"W:/groups/scratch/gmicro_prefect/ggrossha/ggrossha_SWI/training_data/accumulated_train.zarr"
    out_dir = r"results/diagnosis_vis"

    raw_key = "0"
    label_key = "labels/mask/0"

    analyze_and_visualize(train_zarr, raw_key, label_key, out_dir)
