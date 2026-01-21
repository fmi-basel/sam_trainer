import traceback
from pathlib import Path

import numpy as np
import zarr


def analyze_zarr(path_str, raw_key, label_key):
    print(f"\nAnalyzing Zarr container: {path_str}")
    path = Path(path_str)
    if not path.exists():
        print(f"ERROR: Path does not exist: {path}")
        return

    try:
        store = zarr.open(str(path), mode="r")
        print("  Zarr tree structure:")
        print(store.tree())

        # Analyze Raw
        if raw_key in store:
            raw_arr = store[raw_key]
            print(f"\n  [Raw Data] Key: '{raw_key}'")
            print(f"    Shape: {raw_arr.shape}")
            print(f"    Dtype: {raw_arr.dtype}")
            print(f"    Chunks: {raw_arr.chunks}")

            # Sample first image/volume slice
            # Assuming (N, H, W) or (N, C, H, W)
            sample_idx = 0
            if raw_arr.shape[0] > 0:
                sample_data = raw_arr[sample_idx]
                print(f"    Sample {sample_idx} Min: {np.min(sample_data)}")
                print(f"    Sample {sample_idx} Max: {np.max(sample_data)}")
                print(f"    Sample {sample_idx} Mean: {np.mean(sample_data):.4f}")
                print(f"    Sample {sample_idx} Std: {np.std(sample_data):.4f}")

                # Check dynamic range issues
                if np.max(sample_data) < 1.0 and np.issubdtype(
                    raw_arr.dtype, np.floating
                ):
                    print("    NOTE: Data seems to be in [0, 1] float range.")
                elif np.max(sample_data) > 255:
                    print(
                        f"    NOTE: Data max > 255. Likely 16-bit or unnormalized. (Max: {np.max(sample_data)})"
                    )

        else:
            print(f"  ERROR: raw_key '{raw_key}' not found in zarr.")

        # Analyze Labels
        if label_key in store:
            lbl_arr = store[label_key]
            print(f"\n  [Labels] Key: '{label_key}'")
            print(f"    Shape: {lbl_arr.shape}")
            print(f"    Dtype: {lbl_arr.dtype}")
            print(f"    Chunks: {lbl_arr.chunks}")

            if lbl_arr.shape[0] > 0:
                sample_lbl = lbl_arr[sample_idx]
                unique_vals = np.unique(sample_lbl)
                print(f"    Sample {sample_idx} Unique Values: {unique_vals}")
                print(
                    f"    Number of objects: {len(unique_vals) - 1 if 0 in unique_vals else len(unique_vals)}"
                )

                if np.issubdtype(lbl_arr.dtype, np.floating):
                    print("    WARNING: Labels stored as FLOAT.")
                    if np.all(np.mod(unique_vals, 1) == 0):
                        print("      But all values appear to be integers.")
                    else:
                        print("      Values have decimals!")
        else:
            print(f"  ERROR: label_key '{label_key}' not found in zarr.")

    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    # Hardcoded paths from the user's config context
    train_zarr = r"W:/groups/scratch/gmicro_prefect/ggrossha/ggrossha_SWI/training_data/accumulated_train.zarr"
    val_zarr = r"W:/groups/scratch/gmicro_prefect/ggrossha/ggrossha_SWI/training_data/accumulated_val.zarr"

    raw_key = "0"
    label_key = "labels/mask/0"

    analyze_zarr(train_zarr, raw_key, label_key)
    analyze_zarr(val_zarr, raw_key, label_key)
