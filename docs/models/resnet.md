# ChessResNet Architecture

The ResNet-style architecture is inspired by AlphaZero and is the recommended model for most use cases.

## Overview

ChessResNet uses residual connections to enable deep networks without degradation. The architecture processes the chess board through multiple residual blocks before producing predictions.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CHESS RESNET                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: boards [B, T, 8, 8] + colors [B, T]                                 │
│                    │                                                         │
│                    ▼                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        EMBEDDING LAYER                               │    │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐            │    │
│  │  │ Piece Embed   │  │ Position Embed│  │ Color Embed   │            │    │
│  │  │ 13 → 64 dims  │  │ 64 → 64 dims  │  │ 2 → 64 dims   │            │    │
│  │  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘            │    │
│  │          │                  │                  │                     │    │
│  │          └──────────────────┼──────────────────┘                     │    │
│  │                             │                                        │    │
│  │                     Concatenate → [B, T, 8, 8, 128]                  │    │
│  └─────────────────────────────┼───────────────────────────────────────┘    │
│                                │                                             │
│                    Reshape to [B*T, 128, 8, 8]                              │
│                                │                                             │
│                                ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                       INITIAL CONVOLUTION                            │    │
│  │                                                                      │    │
│  │  Conv2D(128 → 256, kernel=3, padding=1)                             │    │
│  │  BatchNorm2D(256)                                                    │    │
│  │  ReLU                                                                │    │
│  │                                                                      │    │
│  │  Output: [B*T, 256, 8, 8]                                           │    │
│  └─────────────────────────────┼───────────────────────────────────────┘    │
│                                │                                             │
│                                ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      RESIDUAL TOWER (×8)                             │    │
│  │                                                                      │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │                    RESIDUAL BLOCK                            │    │    │
│  │  │                                                              │    │    │
│  │  │  Input ─────────────────────────────────────┐                │    │    │
│  │  │    │                                        │                │    │    │
│  │  │    ▼                                        │                │    │    │
│  │  │  Conv2D(256, 256, kernel=3, padding=1)     │  Skip          │    │    │
│  │  │    │                                        │  Connection    │    │    │
│  │  │    ▼                                        │                │    │    │
│  │  │  BatchNorm2D(256)                          │                │    │    │
│  │  │    │                                        │                │    │    │
│  │  │    ▼                                        │                │    │    │
│  │  │  ReLU                                       │                │    │    │
│  │  │    │                                        │                │    │    │
│  │  │    ▼                                        │                │    │    │
│  │  │  Conv2D(256, 256, kernel=3, padding=1)     │                │    │    │
│  │  │    │                                        │                │    │    │
│  │  │    ▼                                        │                │    │    │
│  │  │  BatchNorm2D(256)                          │                │    │    │
│  │  │    │                                        │                │    │    │
│  │  │    └──────────────────┬─────────────────────┘                │    │    │
│  │  │                       │                                      │    │    │
│  │  │                       ▼                                      │    │    │
│  │  │                    Add (residual)                            │    │    │
│  │  │                       │                                      │    │    │
│  │  │                       ▼                                      │    │    │
│  │  │                     ReLU                                     │    │    │
│  │  │                       │                                      │    │    │
│  │  │                    Output                                    │    │    │
│  │  └───────────────────────┼──────────────────────────────────────┘    │    │
│  │                          │                                           │    │
│  │  Output: [B*T, 256, 8, 8]                                           │    │
│  └──────────────────────────┼──────────────────────────────────────────┘    │
│                             │                                                │
│                 Reshape to [B, T, 256, 8, 8]                                │
│                             │                                                │
│                             ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    TEMPORAL AGGREGATION                              │    │
│  │                                                                      │    │
│  │  Method 1 (default): Mean over T dimension                          │    │
│  │  Method 2: Conv3D with kernel (T, 1, 1)                             │    │
│  │  Method 3: Attention over sequence                                   │    │
│  │                                                                      │    │
│  │  Output: [B, 256, 8, 8]                                             │    │
│  └─────────────────────────┼───────────────────────────────────────────┘    │
│                            │                                                 │
│                            ▼                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                       OUTPUT HEADS                                   │    │
│  │                                                                      │    │
│  │  ┌─────────────────┐  Global Average Pool → Flatten                 │    │
│  │  │                 │         │                                       │    │
│  │  │  MOVE HEAD      │         ▼                                       │    │
│  │  │  Conv 1×1 → 73  │    [B, 256*8*8] = [B, 16384]                   │    │
│  │  │  Flatten        │         │                                       │    │
│  │  │  → [B, 4672]    │         ├──────────────────────┐               │    │
│  │  │  Linear → 4096  │         │                      │               │    │
│  │  └────────┬────────┘         ▼                      ▼               │    │
│  │           │           ┌─────────────┐        ┌─────────────┐        │    │
│  │           │           │ SCORE HEAD  │        │CAPTURE HEAD │        │    │
│  │           │           │ Lin → 256   │        │ Lin → 64    │        │    │
│  │           │           │ ReLU        │        │ ReLU        │        │    │
│  │           │           │ Lin → 1     │        │ Lin → 2     │        │    │
│  │           │           └──────┬──────┘        └──────┬──────┘        │    │
│  │           │                  │                      │               │    │
│  │           │                  │        ┌─────────────┐               │    │
│  │           │                  │        │OUTCOME HEAD │               │    │
│  │           │                  │        │ Lin → 64    │               │    │
│  │           │                  │        │ ReLU        │               │    │
│  │           │                  │        │ Lin → 3     │               │    │
│  │           │                  │        └──────┬──────┘               │    │
│  │           │                  │               │                      │    │
│  └───────────┼──────────────────┼───────────────┼──────────────────────┘    │
│              │                  │               │                            │
│              ▼                  ▼               ▼                            │
│         move_logits         score          capture/outcome                  │
│          [B, 4096]          [B, 1]          [B, 2]/[B, 3]                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Why Residual Connections?

### The Degradation Problem

Without residual connections, deep networks suffer from degradation:

```
Depth  │  Training Error
───────┼─────────────────
  10   │     5.2%
  20   │     4.8%
  30   │     5.5%  ← Gets worse!
  50   │     6.2%
```

### The Solution

Residual connections allow the network to learn identity mappings:

```
x ─────────────────────────────┐
│                              │
▼                              │
F(x)                           │  If F(x) = 0, output = x
│                              │  Network can "skip" layers
└──────────── + ───────────────┘
              │
              ▼
           x + F(x)
```

This enables training much deeper networks:

```
Depth  │  Training Error (with ResNet)
───────┼──────────────────────────────
  10   │     5.2%
  20   │     4.3%
  30   │     3.9%
  50   │     3.5%  ← Keeps improving!
```

## Configuration

```python
from dataclasses import dataclass

@dataclass
class ChessResNetConfig:
    # Piece embedding
    num_piece_types: int = 13      # Empty + 6 white + 6 black
    piece_embed_dim: int = 64      # Embedding dimension

    # Position embedding
    num_squares: int = 64          # 8×8 board
    position_embed_dim: int = 64   # Embedding dimension

    # Color embedding
    num_colors: int = 2            # White (0) and Black (1)
    color_embed_dim: int = 64      # Embedding dimension

    # Convolutional backbone
    initial_channels: int = 128    # After embedding concat
    hidden_channels: int = 256     # ResNet tower channels
    num_residual_blocks: int = 8   # Number of residual blocks

    # Temporal
    sequence_length: int = 5       # Number of positions in sequence

    # Output
    num_moves: int = 4096          # 64 * 64 possible moves

    # Board handling
    flip_board_for_black: bool = True  # Normalize perspective
```

## Parameter Count

```
Component                    │ Parameters
─────────────────────────────┼───────────────
Piece Embedding              │     832
Position Embedding           │   4,096
Color Embedding              │     128
Initial Conv + BN            │  33,024
Residual Blocks (×8)         │ 1,180,672
Move Head                    │   82,944
Score Head                   │  278,785
Capture Head                 │   16,450
Outcome Head                 │   16,579
─────────────────────────────┼───────────────
TOTAL                        │ ~2.1M
```

## Usage

### Basic

```python
from ochess.model import ChessResNet, ChessResNetConfig

# Create model with defaults
model = ChessResNet()

# Forward pass
boards = torch.randint(0, 13, (4, 5, 8, 8))  # [B, T, 8, 8]
colors = torch.randint(0, 2, (4, 5))          # [B, T]

outputs = model(boards, colors)
```

### Custom Configuration

```python
config = ChessResNetConfig(
    num_residual_blocks=12,    # Deeper
    hidden_channels=384,       # Wider
    sequence_length=3,         # Shorter history
)

model = ChessResNet(config)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Get Features

```python
# Get intermediate features for analysis
outputs = model(boards, colors, return_features=True)

features = outputs['features']  # [B, hidden_channels, 8, 8]
```

### Move Prediction

```python
# Get move probabilities
probs = model.predict_move(boards, colors)  # Softmax applied
best_move_idx = probs.argmax(dim=-1)
```

## Training Tips

### Learning Rate Schedule

```python
# Warm-up + cosine annealing works well
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,
    total_steps=num_epochs * len(train_loader),
    pct_start=0.1  # 10% warm-up
)
```

### Batch Size

- Recommended: 128-512
- Larger batches benefit from batch normalization
- Use gradient accumulation if memory limited

### Data Augmentation

The model uses board flipping for Black, but you can add:

```python
# Random horizontal flip (symmetric positions)
if random.random() < 0.5:
    board = board.flip(1)  # Flip along file axis
    # Remember to adjust move indices!
```

### Regularization

```python
config = ChessResNetConfig(
    # Built-in: BatchNorm provides regularization
)

# Additional options:
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01  # L2 regularization
)
```

## Common Issues

### Issue: Loss Not Decreasing

**Cause**: Learning rate too high or too low

**Solution**:

```python
# Try learning rate finder
from torch_lr_finder import LRFinder
lr_finder = LRFinder(model, optimizer, criterion)
lr_finder.range_test(train_loader, end_lr=1, num_iter=100)
lr_finder.plot()
```

### Issue: Overfitting

**Cause**: Model too large for dataset

**Solution**:

```python
# Use smaller model
config = ChessResNetConfig(
    num_residual_blocks=4,  # Fewer blocks
    hidden_channels=128     # Narrower
)
```

### Issue: Slow Training

**Cause**: CPU bottleneck in data loading

**Solution**:

```python
# Enable caching in dataset
dataset = ChessDataset(path, cache_dir="./cache")

# Use multiple workers
loader = DataLoader(dataset, num_workers=4, pin_memory=True)
```

## Comparison with AlphaZero

| Aspect | AlphaZero | ChessResNet |
|--------|-----------|-------------|
| Residual Blocks | 19-40 | 8 (configurable) |
| Channels | 256 | 256 |
| Input | 119 planes | 13 piece types |
| Output | Policy + Value | Move + Score + Capture + Outcome |
| Training | Self-play | Stockfish supervision |

Our model is smaller for faster experimentation, but the architecture principles are the same.
