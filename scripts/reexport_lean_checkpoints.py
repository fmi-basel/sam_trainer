"""Re-export legacy final_models/*.pt checkpoints into micro_sam's lean, portable
inference format, without touching the originals.

These files predate the branching export logic in `training.py::run_training`
(see commit eeff18e) and were built by hand-copying the raw training checkpoint
(model_state, optimizer_state, scheduler_state, scaler_state, iteration, epoch,
train_time, timestamp, and an `init` dict holding train_dataset/val_dataset objects
tied to sam_trainer.training:PercentileNormalizer) and merging a decoder_state key
in by hand. That raw state can't be unpickled outside an environment with
sam_trainer installed. This script drives `micro_sam.util.export_custom_sam_model`
(the same function sam_trainer's own training pipeline uses) against each raw file
and writes the lean {"model_state", "decoder_state"} result to a new
"<name>_lean.pt" file, leaving the original untouched. Non-tensor training metadata
that would otherwise be discarded (epoch, iteration, best/current metric,
timestamp, wall-clock train time) is written to a "<name>_lean.provenance.json"
sidecar so it isn't lost.

`model_type` is accepted for parity with `export_custom_sam_model`'s signature but
is not actually used by its current implementation (verified against the installed
micro_sam version) — correctness of this script does not depend on it being exact.

Usage:
    python scripts/reexport_lean_checkpoints.py <model_type> <path.pt> [<path.pt> ...]

Example:
    python scripts/reexport_lean_checkpoints.py vit_b_lm \
        final_models/full_img_vit_b_lm.pt \
        final_models/hydra_tsiairis.pt \
        final_models/trunk-like-structures_buehler.pt \
        final_models/ais_vit_b_a40_full_run.pt
"""

import json
import sys
from pathlib import Path

import torch
from loguru import logger
from micro_sam.util import export_custom_sam_model

PROVENANCE_KEYS = (
    "iteration",
    "epoch",
    "best_epoch",
    "best_metric",
    "current_metric",
    "train_time",
    "timestamp",
)


def reexport_lean(path: Path, model_type: str) -> None:
    """Strip a raw sam_trainer checkpoint down to a lean, portable model file."""
    raw_state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw_state, dict) or "model_state" not in raw_state:
        logger.warning(f"{path}: no 'model_state' key found, skipping")
        return
    if set(raw_state) <= {"model_state", "decoder_state"}:
        logger.info(f"{path.name}: already lean ({sorted(raw_state)}), skipping")
        return

    lean_path = path.with_name(f"{path.stem}_lean{path.suffix}")
    export_custom_sam_model(
        checkpoint_path=str(path),
        model_type=model_type,
        save_path=str(lean_path),
        with_segmentation_decoder="decoder_state" in raw_state,
    )

    provenance = {k: raw_state[k] for k in PROVENANCE_KEYS if k in raw_state}
    if provenance:
        provenance_path = lean_path.with_suffix(".provenance.json")
        provenance_path.write_text(json.dumps(provenance, indent=2))

    before_mb = path.stat().st_size / 1e6
    after_mb = lean_path.stat().st_size / 1e6
    logger.info(
        f"{path.name}: {before_mb:.0f} MB -> {lean_path.name}: {after_mb:.0f} MB"
        + (f" (+ {provenance_path.name})" if provenance else "")
    )


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(
            "Usage: python scripts/reexport_lean_checkpoints.py <model_type> <path.pt> [<path.pt> ...]"
        )
    model_type = sys.argv[1]
    for arg in sys.argv[2:]:
        reexport_lean(Path(arg), model_type)


if __name__ == "__main__":
    main()
