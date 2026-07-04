# SWI Training Overview

Training and inference of micro-SAM models on C. elegans SWI (spinning wheel imaging) data.
All paths are relative to the project root unless stated otherwise.

## Training data

| Split | Path |
|-------|------|
| Images | `/tachyon/groups/scratch/gmicro_ipa/ggrossha/ancneagu/swi_annotations/augmented/images` |
| Labels | `/tachyon/groups/scratch/gmicro_ipa/ggrossha/ancneagu/swi_annotations/augmented/labels` |
| Test images | `/tachyon/scratch/gmicro_ipa/ggrossha/ancneagu/swi_annotations/test_data` |

1134 total samples (1078 train / 56 val, 5% split). Images are 3D TIFF stacks.

## Training runs

### Decoder-only (UNETR, AIS inference)

| Config | Experiment dir | Exported model | Notes |
|--------|---------------|----------------|-------|
| `configs/swi_decoder-only_lr5e-5.yaml` | `runs/grosshans_SWI_aug6_decoder-only_lr5e-5/` | `runs/SWI/grosshans_SWI_aug6_decoder-only_lr5e-5_model.pt` | Fresh run, 200 epochs, fixed sampler |

### Full SAM fine-tune (mask decoder, AMG inference)

| Config | Warm-start checkpoint | Experiment dir | Exported model |
|--------|-----------------------|---------------|----------------|
| `configs/swi_full-sam_lr1e-5_resume.yaml` | `runs/grosshans_SWI_aug6_vit_b_lm/checkpoints/grosshans_SWI_aug6_checkpoint/best.pt` | `runs/grosshans_SWI_aug6_full-sam_lr1e-5/` | `runs/SWI/grosshans_SWI_aug6_full-sam_lr1e-5_model.pt` |
| `configs/swi_full-sam_lr5e-5_resume.yaml` | `runs/grosshans_SWI_aug6_vit_b_lm_B/checkpoints/grosshans_SWI_aug6_B_checkpoint/best.pt` | `runs/grosshans_SWI_aug6_full-sam_lr5e-5/` | `runs/SWI/grosshans_SWI_aug6_full-sam_lr5e-5_model.pt` |

Full-SAM warm-start checkpoints are from runs that reached ~50 epochs before crashing (NFS instability on old cluster).

### Split-fix retraining runs (post val-split-leakage fix, see Known issues)

| Config | Experiment dir | Exported model | Notes |
|--------|---------------|----------------|-------|
| `configs/swi_decoder-only_lr5e-5_splitfix.yaml` | `runs/grosshans_SWI_splitfix_decoder-only_lr5e-5/` | `runs/SWI/grosshans_SWI_splitfix_decoder-only_lr5e-5_model.pt` | Clean test of the split fix: fresh experiment name/dir, `resume_from_checkpoint: null`, same hyperparameters as `swi_decoder-only_lr5e-5.yaml` otherwise |
| `configs/swi_full-sam_lr5e-5_resume_splitfix.yaml` | `runs/grosshans_SWI_splitfix_full-sam_lr5e-5/` | `runs/SWI/grosshans_SWI_splitfix_full-sam_lr5e-5_model.pt` | **Not a clean test of the fix** — still warm-starts from `grosshans_SWI_aug6_vit_b_lm_B/checkpoints/.../best.pt`, which was itself trained under the old leaky split. Renamed only to avoid checkpoint collisions; useful as a secondary signal, not proof the fix alone resolves full-SAM behavior. |

New experiment names/dirs are required, not just cosmetic: the pre-existing `runs/grosshans_SWI_aug6_decoder-only_lr5e-5/checkpoints/.../{best,latest}.pt` were trained under the old leaky split, and `train_instance_segmentation`'s auto-resume (`overwrite_training` unset when `resume_from_checkpoint` is `null`) could otherwise silently resume from that checkpoint instead of training cleanly on the fixed split.

## Submitting

```bash
# All three can run in parallel on H100 nodes
sbatch scripts/submit_training_new_cluster.sh configs/swi_decoder-only_lr5e-5.yaml
sbatch scripts/submit_training_new_cluster.sh configs/swi_full-sam_lr1e-5_resume.yaml
sbatch scripts/submit_training_new_cluster.sh configs/swi_full-sam_lr5e-5_resume.yaml
```

Split-fix retraining runs (can run in parallel with each other and the above):
```bash
sbatch scripts/submit_training_new_cluster.sh configs/swi_decoder-only_lr5e-5_splitfix.yaml
sbatch scripts/submit_training_new_cluster.sh configs/swi_full-sam_lr5e-5_resume_splitfix.yaml
```

## Inference

**Decoder-only model (AIS)** — use the exported `.pt` from `runs/SWI/`:

```bash
sbatch scripts/submit_inference.sh \
    runs/SWI/grosshans_SWI_aug6_decoder-only_lr5e-5_model.pt \
    /tachyon/scratch/gmicro_ipa/ggrossha/ancneagu/swi_annotations/test_data \
    /tachyon/scratch/gmicro_ipa/ggrossha/ancneagu/swi_annotations/test_inference \
    --pattern "*.tiff"
```

Adjust decoder thresholds if results are over/under-segmented (defaults 0.5):
```bash
    --center-dist-thresh 0.3 --boundary-dist-thresh 0.3 --foreground-thresh 0.4
```

**Full-SAM model (AMG)** — same script, but you must pass `--use-amg` explicitly: `get_predictor_and_segmenter` raises `RuntimeError` if `segmentation_mode="ais"` is requested (the CLI default) against a checkpoint with no `decoder_state`. It is not auto-detected unless `segmentation_mode` is left as `None`/`"auto"`, which the CLI never does.

```bash
sbatch scripts/submit_inference.sh \
    runs/SWI/grosshans_SWI_splitfix_full-sam_lr5e-5_model.pt \
    /tachyon/scratch/gmicro_ipa/ggrossha/ancneagu/swi_annotations/test_data \
    /tachyon/scratch/gmicro_ipa/ggrossha/ancneagu/swi_annotations/test_inference_splitfix \
    --use-amg --pattern "*.tiff"
```

## Known issues

- **Val-split leakage (confirmed 2026-07-03, fixed in `a0dc7d6`).** `prepare_data_splits` in `training.py` used to split the flat list of augmented tiles rather than source stacks. The decoder-only dataset has only 16 unique source stacks (`mip_164_{0,8}_z000-007`), each expanded to ~120 tiles via slicing + 6x augmentation, so augmented variants of the same slice — and adjacent z-slices from the same stack — landed in both train and val. This explained why `grosshans_SWI_aug6_decoder-only_lr5e-5` reached low loss and segmented training data perfectly but failed on genuinely unseen `test_data`: the val loss was measuring near-duplicate recognition, not generalization. Diagnostic: `python scripts/diagnosis/check_split_leakage.py <config.yaml>`.
  - **Confirmed by split-fix retraining (2026-07-04):** `swi_decoder-only_lr5e-5_splitfix.yaml` plateaued around loss ~1.06-1.07 and early-stopped after ~28 epochs (vs. the old leaky-split 0.06) — consistent with decoder-only genuinely struggling to generalize from only 16 source stacks once val is a real holdout. Not yet usable for production; treat as confirming the diagnosis, not as a working model.
  - `swi_full-sam_lr5e-5_resume_splitfix.yaml` ran the full 100 epochs, best epoch 93, best val loss ~0.0376 — a much better generalization signal, though still confounded by warm-starting from a checkpoint (`grosshans_SWI_aug6_vit_b_lm_B`) trained under the old leaky split. Exported to `runs/SWI/grosshans_SWI_splitfix_full-sam_lr5e-5_model.pt`.
- **Train/inference normalization mismatch (fixed in `69698df`).** `PercentileNormalizer` is now shared between training (`training.py`) and inference (`run_inference.py` / `run_inference_hcs.py` via `inference_utils.segment_image`), applied by default. Override with `--no-normalize` / `--normalize-lower-percentile` / `--normalize-upper-percentile` if a model was trained with different settings.

## Key fixes on branch `swi-training-fixes`

- `training.py`: decoder-only resume now uses `overwrite_training=False` (torch-em auto-resume) instead of passing the UNETR checkpoint as `checkpoint_path` to `get_trainable_sam_model`, which misidentified it as `vit_t`
- `training.py`: `MinInstanceSampler(1, min_size=1)` always enforced for decoder-only training
- `config.py`: `early_stopping` accepts `null`
- `augmentation.py`: skip image/label pairs with empty label masks
- `run_inference.py`: 3D TIFF stacks processed slice-by-slice, results saved as 3D label volume
- `inference_utils.py`: decoder model loading uses `flexible_load_checkpoint=True` + pre-supplied state to avoid `sam.load_state_dict` failing on `decoder_state` key; fixed `amg=` → `segmentation_mode=` API change in micro-SAM 1.8
