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

### Full SAM fine-tune (mask decoder + jointly-trained AIS decoder)

`training.py` always passes `with_segmentation_decoder=True` to `train_sam` and to the
`export_custom_sam_model` export call, regardless of training mode — so full-SAM training
here also jointly trains and exports the AIS instance-segmentation decoder, same as
decoder-only training. Verified directly: `grosshans_SWI_splitfix_full-sam_lr5e-5_model.pt`
contains `['model_state', 'decoder_state']`. Use the default AIS mode (no `--use-amg`)
unless you specifically want AMG.

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

### Pair-fix retraining runs (post image/label pairing fix, see Known issues — supersedes split-fix runs above)

| Config | Experiment dir | Exported model | Notes |
|--------|---------------|----------------|-------|
| `configs/swi_decoder-only_lr5e-5_unstacked.yaml` | `runs/grosshans_SWI_unstacked_decoder-only_lr5e-5/` | `runs/SWI/grosshans_SWI_unstacked_decoder-only_lr5e-5_model.pt` | Cold, no augmentation: trains directly on `images_unstacked` + `masks_unstacked_matched` (200 pairs). Sanity baseline. |
| `configs/swi_full-sam_lr5e-5_unstacked_cold.yaml` | `runs/grosshans_SWI_unstacked_full-sam_lr5e-5_cold/` | `runs/SWI/grosshans_SWI_unstacked_full-sam_lr5e-5_cold_model.pt` | Cold, no augmentation, `resume_from_checkpoint: null` — genuinely fresh from pretrained `vit_b_lm`, not warm-started from any prior (pairing-bug-contaminated) checkpoint. |
| `configs/aug_full-img_vit-b_pairfix.yaml` | `runs/grosshans_SWI_pairfix_vit_b_lm/` | `runs/SWI/grosshans_SWI_pairfix_full-sam_lr5e-5_cold_model.pt` | Combined augmentation (writes `augmented_pairfix/`) + full-SAM cold training in one job. Run this (or the standalone `augment` CLI command) first — `swi_decoder-only_lr5e-5_pairfix.yaml` depends on `augmented_pairfix/` existing. |
| `configs/swi_decoder-only_lr5e-5_pairfix.yaml` | `runs/grosshans_SWI_pairfix_decoder-only_lr5e-5/` | `runs/SWI/grosshans_SWI_pairfix_decoder-only_lr5e-5_model.pt` | Training-only; requires `augmented_pairfix/images` + `/labels` to already exist. |

All prior SWI checkpoints (including the split-fix runs above) were trained on data affected
by the image/label pairing bug and should not be trusted or warm-started from.

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

Pair-fix retraining runs (see Known issues — supersedes split-fix runs, use these instead):
```bash
# Cold, no augmentation, directly on the 200 corrected unstacked pairs — can run in parallel
sbatch scripts/submit_training_new_cluster.sh configs/swi_decoder-only_lr5e-5_unstacked.yaml
sbatch scripts/submit_training_new_cluster.sh configs/swi_full-sam_lr5e-5_unstacked_cold.yaml

# Fixed augmentation + full-SAM cold training in one job — run this before the next line
sbatch scripts/submit_training_new_cluster.sh configs/aug_full-img_vit-b_pairfix.yaml
# Requires augmented_pairfix/ to exist (from the job above, or a standalone `augment` CLI run)
sbatch scripts/submit_training_new_cluster.sh configs/swi_decoder-only_lr5e-5_pairfix.yaml
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

**Full-SAM model (AIS by default)** — same script, no `--use-amg` needed: these checkpoints
contain a `decoder_state` (see above), so the CLI default (`segmentation_mode="ais"`) works
directly. Only pass `--use-amg` if you specifically want AMG instead — note
`get_predictor_and_segmenter` raises `RuntimeError` for `segmentation_mode="ais"` against a
checkpoint that genuinely has no `decoder_state` (e.g. a full-SAM run from before
`with_segmentation_decoder=True` was added), so this only works because these particular
checkpoints do have one.

```bash
sbatch scripts/submit_inference.sh \
    runs/SWI/grosshans_SWI_splitfix_full-sam_lr5e-5_model.pt \
    /tachyon/scratch/gmicro_ipa/ggrossha/ancneagu/swi_annotations/test_data \
    /tachyon/scratch/gmicro_ipa/ggrossha/ancneagu/swi_annotations/test_inference_splitfix \
    --pattern "*.tiff"
```

## Known issues

- **Val-split leakage (confirmed 2026-07-03, fixed in `a0dc7d6`).** `prepare_data_splits` in `training.py` used to split the flat list of augmented tiles rather than source stacks. The decoder-only dataset has only 16 unique source stacks (`mip_164_{0,8}_z000-007`), each expanded to ~120 tiles via slicing + 6x augmentation, so augmented variants of the same slice — and adjacent z-slices from the same stack — landed in both train and val. This explained why `grosshans_SWI_aug6_decoder-only_lr5e-5` reached low loss and segmented training data perfectly but failed on genuinely unseen `test_data`: the val loss was measuring near-duplicate recognition, not generalization. Diagnostic: `python scripts/diagnosis/check_split_leakage.py <config.yaml>`.
  - **Confirmed by split-fix retraining (2026-07-04):** `swi_decoder-only_lr5e-5_splitfix.yaml` plateaued around loss ~1.06-1.07 and early-stopped after ~28 epochs (vs. the old leaky-split 0.06) — consistent with decoder-only genuinely struggling to generalize from only 16 source stacks once val is a real holdout. Not yet usable for production; treat as confirming the diagnosis, not as a working model.
  - `swi_full-sam_lr5e-5_resume_splitfix.yaml` ran the full 100 epochs, best epoch 93, best val loss ~0.0376 — a much better generalization signal, though still confounded by warm-starting from a checkpoint (`grosshans_SWI_aug6_vit_b_lm_B`) trained under the old leaky split. Exported to `runs/SWI/grosshans_SWI_splitfix_full-sam_lr5e-5_model.pt`.
- **Train/inference normalization mismatch (fixed in `69698df`).** `PercentileNormalizer` is now shared between training (`training.py`) and inference (`run_inference.py` / `run_inference_hcs.py` via `inference_utils.segment_image`), applied by default. Override with `--no-normalize` / `--normalize-lower-percentile` / `--normalize-upper-percentile` if a model was trained with different settings.
- **Image/label pairing bug (confirmed 2026-07-04, one-off fix applied, general fix deferred).** `get_image_paths` sorts each directory independently with plain `sorted()`, and both `run_augmentation` and `prepare_data_splits` pair images to labels purely by list position. `images_unstacked` mixes two naming conventions across stacks (`stack_BF_min_ome_s{1,2,3,5,8}_sub20_sliceNNNN.tif` vs. `stack_BF_mip_min_s{30-34}_sub20_sliceNNNN.tif`) while `masks_unstacked` uses one uniform scheme (`mask_s{N}_sliceNNNN.tif`), so lexicographic sort grouped them differently and positional pairing matched the wrong stack's mask to **160/200 images (80%)** — only stacks 1 and 2 (sorted first in both directories) were correct. This is more severe than the val-split leakage above: most training supervision was wrong-stack masks, not just a bad validation signal, and explains why only one C. elegans developmental stage segmented correctly while the others (larval stages) failed.
  - **Fix applied for the current run:** `scripts/fix_unstacked_mask_pairing.py` matches images↔masks by an extracted `(stack, slice)` key and writes renamed copies to `masks_unstacked_matched` (verified 0/200 mismatches). New pair-fix configs (see Training runs above) use this corrected directory and fresh experiment names, since all prior checkpoints are downstream of this bug.
  - **General fix (Fix A) deferred to a future release:** replace positional pairing with explicit key-based matching in `run_augmentation`/`prepare_data_splits` so this bug class can't recur silently for other datasets. Not implemented now, since a fully general matching strategy would need testing against filename conventions this project hasn't encountered yet. Revisit if the symptom recurs on another dataset.

## Key fixes on branch `swi-training-fixes`

- `training.py`: decoder-only resume now uses `overwrite_training=False` (torch-em auto-resume) instead of passing the UNETR checkpoint as `checkpoint_path` to `get_trainable_sam_model`, which misidentified it as `vit_t`
- `training.py`: `MinInstanceSampler(1, min_size=1)` always enforced for decoder-only training
- `config.py`: `early_stopping` accepts `null`
- `augmentation.py`: skip image/label pairs with empty label masks
- `run_inference.py`: 3D TIFF stacks processed slice-by-slice, results saved as 3D label volume
- `inference_utils.py`: decoder model loading uses `flexible_load_checkpoint=True` + pre-supplied state to avoid `sam.load_state_dict` failing on `decoder_state` key; fixed `amg=` → `segmentation_mode=` API change in micro-SAM 1.8
