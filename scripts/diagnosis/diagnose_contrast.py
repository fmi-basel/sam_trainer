
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
        # print("  Zarr tree structure:")
        # print(store.tree())
        
        # Check keys
        if raw_key not in store or label_key not in store:
            print("  Missing keys.")
            return

        raw_arr = store[raw_key]
        lbl_arr = store[label_key]

        # Sample 0
        idx = 0
        if raw_arr.shape[0] > 0:
            raw = raw_arr[idx]
            lbl = lbl_arr[idx]
            
            print(f"  [Sample {idx}]")
            print(f"    Raw Range: {raw.min()} - {raw.max()}")
            
            zero_count = np.sum(raw == 0)
            print(f"    Zero Count: {zero_count} ({zero_count/lbl.size:.2%})")

            print(f"    Label Unique: {np.unique(lbl)}")
            
            mask_size = np.sum(lbl > 0)
            total_size = lbl.size
            
            if mask_size > 0:
                fg_mask = (lbl > 0)
                bg_mask = (lbl == 0) & (raw > 0)  # Exclude padding from BG stats
                
                mean_fg = np.mean(raw[fg_mask])
                if np.sum(bg_mask) > 0:
                    mean_bg = np.mean(raw[bg_mask])
                    print(f"    Foreground (Label > 0) Mean: {mean_fg:.2f}")
                    print(f"    Background (Label==0 & Raw>0) Mean: {mean_bg:.2f}")
                    print(f"    Background (Label==0 including 0s) Mean: {np.mean(raw[lbl == 0]):.2f}")
                else:
                    mean_bg = 0
                    print(f"    Foreground (Label > 0) Mean: {mean_fg:.2f}")
                    print("    No non-zero background pixels found!")

                print(f"    Fill Ratio: {mask_size / total_size:.2%}")
                
                if mean_fg < mean_bg:
                    print("    Type: Dark Object on Bright Background (Brightfield CHECK PASS)")
                else:
                    print("    Type: Bright Object on Dark Background (Fluorescence?)")
            else:
                print("    WARNING: Mask is empty!")

    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    train_zarr = r"W:/groups/scratch/gmicro_prefect/ggrossha/ggrossha_SWI/training_data/accumulated_train.zarr"
    val_zarr = r"W:/groups/scratch/gmicro_prefect/ggrossha/ggrossha_SWI/training_data/accumulated_val.zarr"
    
    raw_key = "0"
    label_key = "labels/mask/0"
    
    analyze_zarr(train_zarr, raw_key, label_key)
    analyze_zarr(val_zarr, raw_key, label_key)
