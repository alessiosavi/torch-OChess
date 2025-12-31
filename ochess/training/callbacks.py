"""
Training callbacks for monitoring and control.
"""

import logging
from typing import Dict, Optional, Callable

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping callback."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, trainer, epoch: int, train_metrics: Dict, val_metrics: Optional[Dict]):
        if val_metrics is None:
            return

        current_loss = val_metrics.get('loss', float('inf'))

        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            logger.info(f"Early stopping: no improvement for {self.patience} epochs")


class ModelCheckpoint:
    """Save model checkpoints."""

    def __init__(self, save_best_only: bool = True, metric: str = 'val_loss'):
        self.save_best_only = save_best_only
        self.metric = metric
        self.best_value = float('inf') if 'loss' in metric else 0

    def __call__(self, trainer, epoch: int, train_metrics: Dict, val_metrics: Optional[Dict]):
        metrics = val_metrics if val_metrics else train_metrics
        current = metrics.get(self.metric.replace('val_', ''), 0)

        is_better = (
            current < self.best_value if 'loss' in self.metric
            else current > self.best_value
        )

        if is_better or not self.save_best_only:
            if is_better:
                self.best_value = current
            trainer.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")


class MetricsLogger:
    """Log metrics to file or wandb."""

    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file

    def __call__(self, trainer, epoch: int, train_metrics: Dict, val_metrics: Optional[Dict]):
        if self.log_file:
            with open(self.log_file, 'a') as f:
                line = f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}"
                if val_metrics:
                    line += f", val_loss={val_metrics['loss']:.4f}"
                f.write(line + "\n")
