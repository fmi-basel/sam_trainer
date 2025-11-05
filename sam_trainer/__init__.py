"""SAM Trainer - A lightweight package for training micro-SAM models with data augmentation.

This package provides tools for:
- Data augmentation with multiple output formats
- Training SAM models for instance segmentation
- Easy configuration via CLI or Python API
"""

__version__ = "0.1.0"
__author__ = "Niklas Khoss"

from sam_trainer.config import AugmentationConfig, PipelineConfig, TrainingConfig

__all__ = ["AugmentationConfig", "TrainingConfig", "PipelineConfig"]
