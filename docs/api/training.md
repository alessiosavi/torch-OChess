# Training API Reference

The `ochess.training` module provides training utilities including the trainer class, callbacks, and metrics.

## Module Overview

```python
from ochess.training import (
    Trainer,
    TrainingConfig,

    # Callbacks
    Callback,
    CheckpointCallback,
    EarlyStoppingCallback,
    LoggingCallback,
    LRSchedulerCallback,

    # Metrics
    MetricsTracker,
    compute_accuracy,
    compute_top_k_accuracy
)
```

---

## TrainingConfig

Configuration for training.

### Definition

```python
@dataclass
class TrainingConfig:
    """Training configuration."""

    # Basic
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3

    # Optimizer
    optimizer: str = "adamw"           # "adam", "adamw", "sgd"
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)

    # Learning rate schedule
    scheduler: str = "cosine"          # "cosine", "step", "plateau", "none"
    warmup_steps: int = 1000
    min_lr: float = 1e-6

    # Training
    gradient_clip: float = 1.0
    accumulation_steps: int = 1
    mixed_precision: bool = True

    # Data
    num_workers: int = 4
    pin_memory: bool = True

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 5                # Save every N epochs
    keep_best: int = 3                 # Keep top N checkpoints

    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 1e-4

    # Logging
    log_every: int = 100               # Log every N steps
    eval_every: int = 1                # Evaluate every N epochs

    # Device
    device: str = "auto"               # "auto", "cuda", "cpu"
```

---

## Trainer

Main training class.

### Class Definition

```python
class Trainer:
    """Handles the training loop."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        criterion: Optional[nn.Module] = None,
        callbacks: Optional[List[Callback]] = None
    ):
        ...
```

### Constructor Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | nn.Module | Model to train |
| `config` | TrainingConfig | Training configuration |
| `train_loader` | DataLoader | Training data |
| `val_loader` | DataLoader | Validation data (optional) |
| `criterion` | nn.Module | Loss function (default: ChessLoss) |
| `callbacks` | List[Callback] | Training callbacks |

### Methods

#### `train() -> Dict`

Run full training loop.

```python
from ochess.training import Trainer, TrainingConfig
from ochess.model import ChessResNet
from ochess.data import ChessDataset, create_dataloader

# Setup
model = ChessResNet()
config = TrainingConfig(epochs=100, learning_rate=1e-3)

train_dataset = ChessDataset("train.parquet")
val_dataset = ChessDataset("val.parquet")

train_loader = create_dataloader(train_dataset, batch_size=256)
val_loader = create_dataloader(val_dataset, batch_size=256, shuffle=False)

# Create trainer
trainer = Trainer(
    model=model,
    config=config,
    train_loader=train_loader,
    val_loader=val_loader
)

# Train
history = trainer.train()

print(history.keys())
# ['train_loss', 'val_loss', 'train_accuracy', 'val_accuracy', ...]
```

#### `train_epoch() -> Dict`

Train for one epoch.

```python
metrics = trainer.train_epoch()
print(f"Train loss: {metrics['loss']:.4f}")
print(f"Move accuracy: {metrics['move_accuracy']:.2%}")
```

#### `validate() -> Dict`

Run validation.

```python
val_metrics = trainer.validate()
print(f"Val loss: {val_metrics['loss']:.4f}")
```

#### `save_checkpoint(path: str)`

Save training state.

```python
trainer.save_checkpoint("checkpoint_epoch_50.pt")
```

#### `load_checkpoint(path: str)`

Resume from checkpoint.

```python
trainer.load_checkpoint("checkpoint_epoch_50.pt")
trainer.train()  # Continues from epoch 51
```

#### `get_lr() -> float`

Get current learning rate.

```python
print(f"Current LR: {trainer.get_lr()}")
```

---

## Callbacks

### Callback Base Class

```python
class Callback:
    """Base class for training callbacks."""

    def on_train_begin(self, trainer): pass
    def on_train_end(self, trainer): pass
    def on_epoch_begin(self, trainer, epoch): pass
    def on_epoch_end(self, trainer, epoch, logs): pass
    def on_batch_begin(self, trainer, batch_idx): pass
    def on_batch_end(self, trainer, batch_idx, logs): pass
```

### CheckpointCallback

Save model checkpoints.

```python
from ochess.training import CheckpointCallback

callback = CheckpointCallback(
    checkpoint_dir="checkpoints",
    save_every=5,           # Save every 5 epochs
    keep_best=3,            # Keep top 3 by val_loss
    monitor="val_loss",     # Metric to monitor
    mode="min"              # "min" for loss, "max" for accuracy
)

trainer = Trainer(model, config, loader, callbacks=[callback])
```

### EarlyStoppingCallback

Stop training when metric stops improving.

```python
from ochess.training import EarlyStoppingCallback

callback = EarlyStoppingCallback(
    monitor="val_loss",
    patience=10,            # Stop after 10 epochs without improvement
    min_delta=1e-4,         # Minimum change to qualify as improvement
    mode="min"
)
```

### LoggingCallback

Log training progress.

```python
from ochess.training import LoggingCallback

callback = LoggingCallback(
    log_every=100,          # Log every 100 batches
    log_to_file="training.log",
    use_tensorboard=True,
    tensorboard_dir="runs/"
)
```

### LRSchedulerCallback

Adjust learning rate during training.

```python
from ochess.training import LRSchedulerCallback

# Cosine annealing
callback = LRSchedulerCallback(
    scheduler_type="cosine",
    warmup_steps=1000,
    min_lr=1e-6
)

# Step decay
callback = LRSchedulerCallback(
    scheduler_type="step",
    step_size=30,
    gamma=0.1
)

# Reduce on plateau
callback = LRSchedulerCallback(
    scheduler_type="plateau",
    factor=0.5,
    patience=5
)
```

### Custom Callback

```python
class CustomCallback(Callback):
    def on_epoch_end(self, trainer, epoch, logs):
        if logs['val_accuracy'] > 0.8:
            print("Achieved 80% accuracy!")

        # Access trainer state
        model = trainer.model
        optimizer = trainer.optimizer

    def on_batch_end(self, trainer, batch_idx, logs):
        if batch_idx % 500 == 0:
            # Custom logging
            print(f"Batch {batch_idx}: loss={logs['loss']:.4f}")
```

---

## MetricsTracker

Track and aggregate metrics.

### Class Definition

```python
class MetricsTracker:
    """Track training metrics."""

    def __init__(self):
        self.reset()
```

### Methods

```python
tracker = MetricsTracker()

# During training
for batch in loader:
    loss = compute_loss(...)
    accuracy = compute_accuracy(...)

    tracker.update({
        'loss': loss.item(),
        'accuracy': accuracy
    }, batch_size=len(batch))

# Get averages
metrics = tracker.get_metrics()
print(f"Avg loss: {metrics['loss']:.4f}")
print(f"Avg accuracy: {metrics['accuracy']:.2%}")

# Reset for next epoch
tracker.reset()
```

---

## Utility Functions

### compute_accuracy

```python
from ochess.training import compute_accuracy

logits = model(boards, colors)['move_logits']  # [B, 4096]
targets = batch['target_move']                  # [B]

accuracy = compute_accuracy(logits, targets)
print(f"Accuracy: {accuracy:.2%}")
```

### compute_top_k_accuracy

```python
from ochess.training import compute_top_k_accuracy

top1 = compute_top_k_accuracy(logits, targets, k=1)
top3 = compute_top_k_accuracy(logits, targets, k=3)
top5 = compute_top_k_accuracy(logits, targets, k=5)

print(f"Top-1: {top1:.2%}")
print(f"Top-3: {top3:.2%}")
print(f"Top-5: {top5:.2%}")
```

---

## Complete Training Example

```python
import torch
from ochess.model import ChessResNet, ChessResNetConfig
from ochess.model.losses import ChessLoss
from ochess.data import ChessDataset, create_dataloader
from ochess.training import (
    Trainer,
    TrainingConfig,
    CheckpointCallback,
    EarlyStoppingCallback,
    LoggingCallback
)

# 1. Create model
model_config = ChessResNetConfig(
    num_residual_blocks=8,
    hidden_channels=256
)
model = ChessResNet(model_config)

# 2. Load data
train_dataset = ChessDataset("datasets/train.parquet")
val_dataset = ChessDataset("datasets/val.parquet")

train_loader = create_dataloader(train_dataset, batch_size=256, num_workers=4)
val_loader = create_dataloader(val_dataset, batch_size=256, shuffle=False)

# 3. Configure training
training_config = TrainingConfig(
    epochs=100,
    learning_rate=1e-3,
    weight_decay=0.01,
    scheduler="cosine",
    warmup_steps=500,
    mixed_precision=True,
    gradient_clip=1.0,
    early_stopping=True,
    patience=15
)

# 4. Setup callbacks
callbacks = [
    CheckpointCallback(
        checkpoint_dir="checkpoints",
        save_every=5,
        keep_best=3,
        monitor="val_loss"
    ),
    EarlyStoppingCallback(
        monitor="val_loss",
        patience=15
    ),
    LoggingCallback(
        log_every=100,
        use_tensorboard=True
    )
]

# 5. Create trainer
trainer = Trainer(
    model=model,
    config=training_config,
    train_loader=train_loader,
    val_loader=val_loader,
    callbacks=callbacks
)

# 6. Train
print("Starting training...")
history = trainer.train()

# 7. Analyze results
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.plot(history['train_move_accuracy'], label='Train')
plt.plot(history['val_move_accuracy'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Move Accuracy')

plt.tight_layout()
plt.savefig('training_history.png')

# 8. Load best model
best_model = ChessResNet(model_config)
best_model.load_state_dict(torch.load("checkpoints/best_model.pt"))
```

---

## Command Line Training

```bash
# Basic training
python scripts/train.py \
    --data datasets/train.parquet \
    --model resnet \
    --epochs 100

# Advanced options
python scripts/train.py \
    --data datasets/train.parquet \
    --val-data datasets/val.parquet \
    --model resnet \
    --epochs 100 \
    --batch-size 256 \
    --lr 1e-3 \
    --scheduler cosine \
    --warmup-steps 1000 \
    --checkpoint-dir checkpoints \
    --tensorboard-dir runs \
    --mixed-precision \
    --num-workers 4

# Resume training
python scripts/train.py \
    --data datasets/train.parquet \
    --resume checkpoints/checkpoint_epoch_50.pt
```
