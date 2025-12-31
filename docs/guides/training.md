# Training Guide

This guide covers how to effectively train chess neural networks.

## Quick Start

```bash
python scripts/train.py \
    --data datasets/train_data.parquet \
    --model resnet \
    --epochs 100
```

## Training Configuration

### Model Selection

Three architectures available:

| Model | Parameters | Speed | Accuracy | Best For |
|-------|------------|-------|----------|----------|
| `resnet` | ~2M | Fast | Good | Production |
| `transformer` | ~5M | Slow | Better | Research |
| `hybrid` | ~3M | Medium | Very Good | Balanced |

```bash
# ResNet (recommended for starting)
python scripts/train.py --model resnet

# Transformer (needs more data)
python scripts/train.py --model transformer

# Hybrid (balanced)
python scripts/train.py --model hybrid
```

### Hyperparameters

| Parameter | Default | Recommended Range |
|-----------|---------|-------------------|
| `--epochs` | 100 | 50-200 |
| `--batch-size` | 256 | 64-512 |
| `--lr` | 1e-3 | 1e-4 to 1e-2 |
| `--weight-decay` | 0.01 | 0.001-0.1 |

```bash
python scripts/train.py \
    --data datasets/train.parquet \
    --model resnet \
    --epochs 100 \
    --batch-size 256 \
    --lr 0.001 \
    --weight-decay 0.01
```

### Learning Rate Schedule

```bash
# Cosine annealing (default, recommended)
python scripts/train.py --scheduler cosine --warmup-steps 1000

# Step decay
python scripts/train.py --scheduler step --step-size 30 --gamma 0.1

# Reduce on plateau
python scripts/train.py --scheduler plateau --patience 5 --factor 0.5
```

### Mixed Precision Training

Use automatic mixed precision for faster training:

```bash
python scripts/train.py --mixed-precision
```

Benefits:
- 2-3x faster training
- 50% less memory
- Minimal accuracy loss

### Early Stopping

Stop training when validation loss stops improving:

```bash
python scripts/train.py \
    --early-stopping \
    --patience 15 \
    --min-delta 0.0001
```

## Data Configuration

### Train/Validation Split

```bash
# Automatic 90/10 split
python scripts/train.py --data datasets/train.parquet

# Explicit validation set
python scripts/train.py \
    --data datasets/train.parquet \
    --val-data datasets/val.parquet
```

### Data Loading

```bash
# Parallel data loading (recommended)
python scripts/train.py --num-workers 4

# Enable caching (first epoch slower, rest faster)
python scripts/train.py --cache-dir datasets/cache
```

### Sequence Length

Number of previous positions to consider:

```bash
# Default: 5 positions of history
python scripts/train.py --sequence-length 5

# Less context (faster)
python scripts/train.py --sequence-length 3

# More context (more memory)
python scripts/train.py --sequence-length 10
```

## Loss Function

The model uses multi-task learning:

```
Total Loss = move_weight * move_loss +
             score_weight * score_loss +
             capture_weight * capture_loss +
             outcome_weight * outcome_loss
```

Default weights:
| Task | Weight | Purpose |
|------|--------|---------|
| Move | 3.0 | Main task - predict best move |
| Score | 0.5 | Auxiliary - evaluate position |
| Capture | 1.0 | Auxiliary - tactical awareness |
| Outcome | 1.0 | Auxiliary - long-term planning |

```bash
# Custom weights
python scripts/train.py \
    --move-weight 3.0 \
    --score-weight 0.5 \
    --capture-weight 1.0 \
    --outcome-weight 1.0
```

## Checkpointing

### Automatic Saving

```bash
# Save every 5 epochs, keep best 3
python scripts/train.py \
    --checkpoint-dir checkpoints \
    --save-every 5 \
    --keep-best 3
```

### Resume Training

```bash
python scripts/train.py \
    --data datasets/train.parquet \
    --resume checkpoints/checkpoint_epoch_50.pt
```

## Monitoring

### TensorBoard

```bash
# Enable TensorBoard logging
python scripts/train.py --tensorboard-dir runs

# View in browser
tensorboard --logdir runs
```

### Console Output

```bash
# Verbose output
python scripts/train.py --verbose

# Log every N batches
python scripts/train.py --log-every 100
```

### Metrics Tracked

- Train/Val loss (total and per-task)
- Move accuracy (top-1, top-3, top-5)
- Score MSE
- Capture accuracy
- Learning rate

## GPU Training

### Single GPU

```bash
# Auto-detect GPU
python scripts/train.py --device auto

# Force specific device
python scripts/train.py --device cuda:0
python scripts/train.py --device cpu
```

### Multi-GPU (Data Parallel)

```python
import torch
from ochess.model import ChessResNet

model = ChessResNet()
model = torch.nn.DataParallel(model)  # Use all GPUs
model = model.to('cuda')
```

## Training Recipes

### Recipe 1: Quick Experiment

```bash
python scripts/train.py \
    --data datasets/small.parquet \
    --model resnet \
    --epochs 20 \
    --batch-size 128 \
    --lr 0.001
```

### Recipe 2: Production Training

```bash
python scripts/train.py \
    --data datasets/train.parquet \
    --val-data datasets/val.parquet \
    --model resnet \
    --epochs 100 \
    --batch-size 256 \
    --lr 0.001 \
    --scheduler cosine \
    --warmup-steps 1000 \
    --mixed-precision \
    --early-stopping \
    --patience 15 \
    --checkpoint-dir checkpoints \
    --tensorboard-dir runs \
    --num-workers 4 \
    --cache-dir datasets/cache
```

### Recipe 3: Large Transformer

```bash
python scripts/train.py \
    --data datasets/large.parquet \
    --model transformer \
    --epochs 200 \
    --batch-size 64 \
    --lr 0.0001 \
    --weight-decay 0.05 \
    --scheduler cosine \
    --warmup-steps 2000 \
    --mixed-precision
```

## Python Training

```python
from ochess.model import ChessResNet, ChessResNetConfig
from ochess.data import ChessDataset, create_dataloader
from ochess.training import Trainer, TrainingConfig

# Model
model = ChessResNet(ChessResNetConfig(
    num_residual_blocks=8,
    hidden_channels=256
))

# Data
train_dataset = ChessDataset("datasets/train.parquet")
val_dataset = ChessDataset("datasets/val.parquet")

train_loader = create_dataloader(train_dataset, batch_size=256)
val_loader = create_dataloader(val_dataset, batch_size=256, shuffle=False)

# Training config
config = TrainingConfig(
    epochs=100,
    learning_rate=1e-3,
    scheduler="cosine",
    warmup_steps=1000,
    mixed_precision=True,
    early_stopping=True,
    patience=15
)

# Train
trainer = Trainer(
    model=model,
    config=config,
    train_loader=train_loader,
    val_loader=val_loader
)

history = trainer.train()
```

## Troubleshooting

### Loss Not Decreasing

**Problem:** Loss stays flat or increases

**Solutions:**
1. Lower learning rate: `--lr 0.0001`
2. Add warmup: `--warmup-steps 1000`
3. Check data quality
4. Reduce batch size

### Overfitting

**Problem:** Train loss << Val loss

**Solutions:**
1. Add regularization: `--weight-decay 0.1`
2. Use smaller model
3. Add dropout: `--dropout 0.2`
4. Get more training data
5. Use early stopping

### Out of Memory

**Problem:** CUDA out of memory

**Solutions:**
1. Reduce batch size: `--batch-size 64`
2. Use gradient accumulation
3. Enable mixed precision: `--mixed-precision`
4. Use smaller model

### Slow Training

**Problem:** Training takes too long

**Solutions:**
1. Enable mixed precision: `--mixed-precision`
2. Use more workers: `--num-workers 4`
3. Enable caching: `--cache-dir datasets/cache`
4. Reduce sequence length: `--sequence-length 3`

### Unstable Training

**Problem:** Loss oscillates wildly

**Solutions:**
1. Reduce learning rate
2. Add gradient clipping: `--gradient-clip 1.0`
3. Use warmup: `--warmup-steps 1000`
4. Try different optimizer

## Expected Results

### Move Accuracy by Dataset Size

| Positions | Top-1 | Top-3 | Top-5 |
|-----------|-------|-------|-------|
| 10,000 | ~35% | ~55% | ~65% |
| 50,000 | ~45% | ~65% | ~75% |
| 100,000 | ~50% | ~70% | ~80% |
| 500,000 | ~55% | ~75% | ~85% |

### Training Time Estimates

| Model | 10K pos | 100K pos | GPU |
|-------|---------|----------|-----|
| ResNet | 10 min | 1.5 hours | RTX 3080 |
| Transformer | 30 min | 5 hours | RTX 3080 |
| Hybrid | 15 min | 2.5 hours | RTX 3080 |

CPU training is approximately 5-10x slower.
