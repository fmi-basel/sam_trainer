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

## Submitting

```bash
# All three can run in parallel on H100 nodes
sbatch scripts/submit_training_new_cluster.sh configs/swi_decoder-only_lr5e-5.yaml
sbatch scripts/submit_training_new_cluster.sh configs/swi_full-sam_lr1e-5_resume.yaml
sbatch scripts/submit_training_new_cluster.sh configs/swi_full-sam_lr5e-5_resume.yaml
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

**Full-SAM model (AMG)** — same script; mode auto-detected from checkpoint (no `decoder_state` → AMG).

## Key fixes on branch `swi-training-fixes`

- `training.py`: decoder-only resume now uses `overwrite_training=False` (torch-em auto-resume) instead of passing the UNETR checkpoint as `checkpoint_path` to `get_trainable_sam_model`, which misidentified it as `vit_t`
- `training.py`: `MinInstanceSampler(1, min_size=1)` always enforced for decoder-only training
- `config.py`: `early_stopping` accepts `null`
- `augmentation.py`: skip image/label pairs with empty label masks
- `run_inference.py`: 3D TIFF stacks processed slice-by-slice, results saved as 3D label volume
- `inference_utils.py`: decoder model loading uses `flexible_load_checkpoint=True` + pre-supplied state to avoid `sam.load_state_dict` failing on `decoder_state` key; fixed `amg=` → `segmentation_mode=` API change in micro-SAM 1.8
