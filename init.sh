#!/usr/bin/env bash
# Initialize pixi and set cache and temporary directories.
WD="$(realpath .)"
export PATH=$PATH:"$WD/infrastructure/apps/pixi/bin"
export PIXI_CACHE_DIR="$WD/infrastructure/apps/pixi/.pixi_cache"
export TMPDIR="$WD/infrastructure/.tmp_$USER"
mkdir -p "$TMPDIR"
