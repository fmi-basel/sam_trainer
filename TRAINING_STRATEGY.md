# Training Strategy & Experiment Plan

This document outlines the strategy for comparing SAM model architectures and hyperparameters on the A100 cluster.

## Phase 1: Baselines (Capacity Check)

**Goal:** Determine if the larger ViT-H model provides enough benefit to justify its computational cost over ViT-B.

* **ViT-B (Base):** Faster, lighter.
* **ViT-H (Huge):** More capacity, slower, requires smaller batch size.

### Commands

```bash
# 1. ViT-B Baseline
sbatch scripts/submit_full_sam_training.sh configs/full_sam_vit_b_a100.yaml

# 2. ViT-H Baseline
sbatch scripts/submit_full_sam_training.sh configs/full_sam_vit_h_a100.yaml
```

## Phase 2: Hyperparameter Variations (on ViT-B)

**Goal:** Investigate if the ViT-B baseline can be improved with more aggressive learning rates or longer training, which is cheaper to iterate on than ViT-H.

* **High LR:** Testing `5e-5` (vs default `1e-5`) to see if convergence is faster or better.
* **More Samples:** Testing `5000` samples (vs `2500`) to see if the model is under-trained.

### Commands

```bash
# 3. High Learning Rate
sbatch scripts/submit_full_sam_training.sh configs/full_sam_vit_b_a100_high_lr.yaml

# 4. More Samples
sbatch scripts/submit_full_sam_training.sh configs/full_sam_vit_b_a100_more_samples.yaml
```

## Phase 3: Evaluation

**Goal:** Compare the best checkpoints from all 4 runs on the test set.

Once training is complete, use the inference script to generate predictions and compute metrics.

```bash
# Example inference command (update with actual checkpoint paths later)
python scripts/run_inference.py -c configs/full_sam_vit_b_a100.yaml -m runs/full_sam_vit_b_a100/checkpoints/best.pt
```
