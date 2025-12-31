"""
Training loop for chess neural networks.

Provides:
    - Trainer: Main training class with checkpointing
    - Support for mixed precision training
    - Learning rate scheduling
    - Metrics tracking
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from typing import Dict, Optional, List, Callable
from dataclasses import dataclass, field
import time
import json
import logging

from ochess.model.losses import ChessLoss

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 100

    # Loss weights
    move_loss_weight: float = 3.0
    score_loss_weight: float = 0.5
    capture_loss_weight: float = 1.0
    outcome_loss_weight: float = 1.0

    # Learning rate schedule
    lr_scheduler: str = "cosine"  # "cosine", "step", "plateau", "none"
    lr_warmup_epochs: int = 5
    lr_min: float = 1e-6

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 5
    save_best_only: bool = False

    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val_loss"

    # Hardware
    device: str = "cuda"
    use_amp: bool = True  # Automatic mixed precision
    num_workers: int = 4
    gradient_clip: float = 1.0

    # Logging
    log_every_n_steps: int = 50


class Trainer:
    """
    Training loop for chess neural networks.

    Features:
    - Automatic mixed precision (AMP) for faster training
    - Gradient clipping for stability
    - Learning rate scheduling
    - Checkpointing and resume
    - Early stopping
    - Metrics tracking
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        callbacks: Optional[List[Callable]] = None
    ):
        """
        Initialize trainer.

        Args:
            model: Neural network model
            config: Training configuration
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            callbacks: List of callback functions
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.callbacks = callbacks or []

        # Setup device
        self.device = torch.device(config.device)
        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Loss function
        self.criterion = ChessLoss(
            move_weight=config.move_loss_weight,
            score_weight=config.score_loss_weight,
            capture_weight=config.capture_loss_weight,
            outcome_weight=config.outcome_loss_weight
        )

        # Mixed precision
        self.scaler = GradScaler() if config.use_amp else None

        # Checkpointing
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # State tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

        # Metrics history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'move_accuracy': [],
            'learning_rate': []
        }

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.lr_scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.lr_min
            )
        elif self.config.lr_scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif self.config.lr_scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
        else:
            return None

    def train(self) -> Dict:
        """
        Run full training loop.

        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training for {self.config.epochs} epochs")
        logger.info(f"Model has {self.model.num_parameters:,} parameters")

        start_time = time.time()

        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch

            # Training epoch
            train_metrics = self._train_epoch()

            # Validation epoch
            val_metrics = None
            if self.val_loader is not None:
                val_metrics = self._validate_epoch()

            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['move_accuracy'].append(train_metrics['move_accuracy'])
            self.history['learning_rate'].append(self._get_lr())

            if val_metrics:
                self.history['val_loss'].append(val_metrics['loss'])

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    metric = val_metrics['loss'] if val_metrics else train_metrics['loss']
                    self.scheduler.step(metric)
                else:
                    self.scheduler.step()

            # Checkpointing
            self._checkpoint(epoch, val_metrics)

            # Early stopping check
            if self._check_early_stopping(val_metrics):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

            # Logging
            self._log_epoch(epoch, train_metrics, val_metrics)

            # Callbacks
            for callback in self.callbacks:
                callback(self, epoch, train_metrics, val_metrics)

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time / 60:.1f} minutes")

        return self.history

    def _train_epoch(self) -> Dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct_moves = 0
        total_moves = 0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            batch = self._to_device(batch)

            # Forward pass
            self.optimizer.zero_grad()

            if self.config.use_amp and self.scaler is not None:
                with autocast():
                    outputs = self.model(batch['boards'], batch['colors'])
                    loss = self.criterion(outputs, batch)

                # Backward with scaling
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.config.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(batch['boards'], batch['colors'])
                loss = self.criterion(outputs, batch)
                loss.backward()

                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )

                self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            pred_moves = outputs['move_logits'].argmax(dim=-1)
            correct_moves += (pred_moves == batch['target_move']).sum().item()
            total_moves += batch['target_move'].size(0)
            num_batches += 1
            self.global_step += 1

            # Logging
            if batch_idx % self.config.log_every_n_steps == 0:
                logger.debug(
                    f"Epoch {self.current_epoch} Step {batch_idx}: "
                    f"Loss={loss.item():.4f}"
                )
        return {
            'loss': total_loss / num_batches,
            'move_accuracy': correct_moves / total_moves if total_moves > 0 else 0
        }

    def _validate_epoch(self) -> Dict:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct_moves = 0
        total_moves = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._to_device(batch)

                if self.config.use_amp:
                    with autocast():
                        outputs = self.model(batch['boards'], batch['colors'])
                        loss = self.criterion(outputs, batch)
                else:
                    outputs = self.model(batch['boards'], batch['colors'])
                    loss = self.criterion(outputs, batch)

                total_loss += loss.item()
                pred_moves = outputs['move_logits'].argmax(dim=-1)
                correct_moves += (pred_moves == batch['target_move']).sum().item()
                total_moves += batch['target_move'].size(0)
                num_batches += 1

        return {
            'loss': total_loss / num_batches if num_batches > 0 else 0,
            'move_accuracy': correct_moves / total_moves if total_moves > 0 else 0
        }

    def _to_device(self, batch: Dict) -> Dict:
        """Move batch to device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']

    def _checkpoint(self, epoch: int, val_metrics: Optional[Dict]):
        """Save checkpoint."""
        # Determine if this is the best model
        is_best = False
        if val_metrics is not None:
            current_metric = val_metrics.get('loss', float('inf'))
            if current_metric < self.best_val_loss:
                self.best_val_loss = current_metric
                is_best = True
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

        # Save periodic checkpoint
        if (epoch + 1) % self.config.save_every_n_epochs == 0:
            if not self.config.save_best_only:
                self.save_checkpoint(f"epoch_{epoch + 1}.pt")

        # Save best model
        if is_best:
            self.save_checkpoint("best_model.pt")

    def _check_early_stopping(self, val_metrics: Optional[Dict]) -> bool:
        """Check if early stopping should trigger."""
        if val_metrics is None:
            return False

        return self.epochs_without_improvement >= self.config.early_stopping_patience

    def _log_epoch(
        self,
        epoch: int,
        train_metrics: Dict,
        val_metrics: Optional[Dict]
    ):
        """Log epoch results."""
        msg = f"Epoch {epoch + 1}/{self.config.epochs}"
        msg += f" | Train Loss: {train_metrics['loss']:.4f}"
        msg += f" | Move Acc: {train_metrics['move_accuracy']:.4f}"

        if val_metrics:
            msg += f" | Val Loss: {val_metrics['loss']:.4f}"
            msg += f" | Val Acc: {val_metrics['move_accuracy']:.4f}"

        msg += f" | LR: {self._get_lr():.2e}"

        logger.info(msg)
        print(msg)

    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        path = self.checkpoint_dir / filename

        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.config
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
