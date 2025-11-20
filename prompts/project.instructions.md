# sam_trainer Project Instructions

> Read the global AI instructions (`vscode-userdata:/c:/Users/khosnikl/AppData/Roaming/Code/User/prompts/global-copilot.instructions.md`) first. This document augments them with repo-specific guidance that every LLM agent must follow before editing this project.

## Canonical Workflows
- The repository must support four CLI entry points: data augmentation, training, evaluation, and inference. Leverage `sam_trainer/cli.py` and the scripts under `scripts/` when extending or testing any of these flows.
- Prefer keeping new logic inside `sam_trainer/` modules; scripts in `scripts/` should stay as thin shells that collect parameters and call library code.

## Data Assumptions
- Unless explicitly stated otherwise, **training uses the augmented dataset** under `dat/augmented_training_data/{images,labels}`. Keep raw/original data paths untouched for reproducibility.
- Images are 2048×2048 brightfield captures where the foreground objects are darker than the background. Any preprocessing or visualization utilities must respect this contrast convention.
- Current preprocessing consists of percentile-based normalization and clipping only. Leave clear TODO placeholders before introducing additional steps later.

## Config & Naming Conventions
- Copy the existing YAML structure in `configs/` when adding new experiments. Keep names descriptive (`full_sam_<backbone>_<accelerator>[_variant].yaml`).
- Distinguish **A100 vs. A40** configs explicitly; keep A100 defaults aligned with the working files already in the repo (e.g., batch size 4 for ViT-B, workers ≤8). Adjust values only when there is a documented reason (GPU memory, throughput, etc.).
- Always set `checkpoint_name`, `export_path`, and `output_base_dir` so checkpoint and final-model locations are deterministic.

## Outputs & Artifacts
- Training runs must populate `runs/<experiment>/` with `config.yaml`, checkpoints under `runs/<experiment>/checkpoints/<checkpoint_name>/`, and mirrored log files inside `runs/<experiment>/logs/`.
- Final exported weights belong in `final_models/` and should share the experiment’s descriptive name (e.g., `final_models/full_sam_vit_b_a100.pt`).
- Keep everything relative to the repo root so SLURM jobs launched from `scripts/` remain portable.

## Evaluation & Inference
- Future work will define concrete metrics and report formats. For now, leave a placeholder comment or section when extending evaluation scripts (e.g., “TODO: add F1/IoU once metrics are finalized”).
- Use `scripts/run_inference.py` as the canonical entry point for checkpoint validation; new utilities should wrap or extend it rather than duplicate functionality.

## HPC Usage
- Default to the A100 partition (`scripts/submit_training_a100.sh`). There are up to 4 GPUs per node and typically 2 nodes per job allocation; design configs with that capacity in mind.
- Maximum wall time is 56h. If a job may exceed that, surface it early and coordinate with the user before changing scheduler parameters.
- Memory can scale up to ~600 GB per node; CPUs per task may also increase when justified. Document any deviations from the default SLURM resources inside the relevant script or config commit message.
- Keep SLURM script proliferation to a minimum. Reuse the existing submission script and add flags/logic there instead of creating new files unless absolutely necessary.

## Additional Notes
- Downloading SAM backbones requires internet only once; cache checkpoints under `~/.cache/micro_sam` or document any alternative cache path in the README before running on air-gapped nodes.
- When adding notebooks or interactive steps, place them under `notebooks/` (create if missing) and ensure they read/write from deterministic locations (e.g., `runs/<experiment>/analysis/`).
- Always validate configs with Pydantic (`sam_trainer/config.py`) before SLURM submission to avoid wasting cluster time.
