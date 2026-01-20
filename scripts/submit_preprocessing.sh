#!/bin/bash
#SBATCH --job-name=preprocess_zarr
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err

# Load environmental vars (Pixi)
WD="$(pwd)"
export PATH="$PATH:$WD/infrastructure/apps/pixi/bin"
export PIXI_CACHE_DIR="$WD/infrastructure/apps/pixi/.pixi_cache"
export TMPDIR="$WD/infrastructure/.tmp_$USER"
mkdir -p "$TMPDIR"

PIXIBIN="$WD/infrastructure/apps/pixi/bin/pixi"
if [[ ! -x "$PIXIBIN" ]]; then
    # Fallback if local pixi missing
    echo "Pixi binary not found at $PIXIBIN"
    exit 1
fi

# Run script
echo "Running preprocessing..."
pixi run python scripts/preprocess_dataset.py
