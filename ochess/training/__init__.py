"""
Training module for Torch o'Chess.

Provides:
    - Trainer: Main training loop with checkpointing
    - Callbacks: Training callbacks (logging, early stopping)
    - Metrics: Training metrics tracking
"""

from ochess.training.trainer import Trainer, TrainingConfig
from ochess.training.callbacks import EarlyStopping, ModelCheckpoint, MetricsLogger
from ochess.training.metrics import AccuracyMetric, compute_move_accuracy

__all__ = [
    "Trainer",
    "TrainingConfig",
    "EarlyStopping",
    "ModelCheckpoint",
    "MetricsLogger",
    "AccuracyMetric",
    "compute_move_accuracy",
]
