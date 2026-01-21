
import sys
import numpy as np
import zarr
import traceback
from pathlib import Path

def analyze_zarr(path_str, raw_key, label_key):
    print(f"\nAnalyzing Zarr container: {path_str}")
    path = Path(path_str)
    if not path.exists():
        print(f"ERROR: Path does not exist: {path}")
        return

    try:
        store = zarr.open(str(path), mode='r')
        
        if raw_key not in store or label_key not in store:
            print("  Missing keys.")
            return

        raw_arr = store[raw_key]
        lbl_arr = store[label_key]

        num_samples = min(20, raw_arr.shape[0])
        print(f"  Checking first {num_samples} samples...")
        
        dark_count = 0
        bright_count = 0
        contrasts = []
        zero_counts_pct = []
        
        for idx in range(num_samples):
            raw = raw_arr[idx]
            lbl = lbl_arr[idx]
            
            # Zero count analysis
            zero_count = np.sum(raw == 0)
            zero_pct = zero_count / lbl.size
            zero_counts_pct.append(zero_pct)
            
            fg_mask = (lbl > 0)
            if not np.any(fg_mask):
                continue
                
            raw_fg = raw[fg_mask]
            
            # BG mask (non-zero)
            bg_mask = (lbl == 0) & (raw > 0)
            
            if not np.any(bg_mask):
                continue

            mean_bg = np.mean(raw[bg_mask])
            mean_fg = np.mean(raw_fg)
            diff = mean_fg - mean_bg
            contrasts.append(diff)
            
            if diff < 0:
                dark_count += 1
            else:
                bright_count += 1
        
        if contrasts:
            print(f"    Dark Objects (FG < BG): {dark_count}")
            print(f"    Bright Objects (FG > BG): {bright_count}")
            print(f"    Mean Contrast (FG - BG): {np.mean(contrasts):.2f}")
            print(f"    Min Contrast: {np.min(contrasts):.2f}")
            print(f"    Max Contrast: {np.max(contrasts):.2f}")
        else:
            print("    No Valid Samples found (empty masks?)")
            
        print(f"    Avg Zero (Padding) Ratio: {np.mean(zero_counts_pct):.2%}")

    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    train_zarr = r"W:/groups/scratch/gmicro_prefect/ggrossha/ggrossha_SWI/training_data/accumulated_train.zarr"
    val_zarr = r"W:/groups/scratch/gmicro_prefect/ggrossha/ggrossha_SWI/training_data/accumulated_val.zarr"
    
    raw_key = "0"
    label_key = "labels/mask/0"
    
    analyze_zarr(train_zarr, raw_key, label_key)
    analyze_zarr(val_zarr, raw_key, label_key)
