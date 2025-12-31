# ChessHybrid Architecture

The Hybrid architecture combines convolutional layers for efficient local pattern extraction with attention layers for global reasoning.

## Overview

ChessHybrid uses a CNN backbone to extract local features (tactical patterns, piece configurations) and then applies self-attention to reason about global relationships. This provides the best of both worlds: CNN efficiency and attention expressiveness.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CHESS HYBRID                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: boards [B, T, 8, 8] + colors [B, T]                                 │
│                    │                                                         │
│                    ▼                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        EMBEDDING LAYER                               │    │
│  │                                                                      │    │
│  │  Piece Embed (13 → 64) + Position Embed (64 → 64)                   │    │
│  │  Output: [B, T, 8, 8, 128]                                          │    │
│  │                                                                      │    │
│  └─────────────────────────────┼───────────────────────────────────────┘    │
│                                │                                             │
│                    Reshape to [B*T, 128, 8, 8]                              │
│                                │                                             │
│                                ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    CNN BACKBONE (4 ResBlocks)                        │    │
│  │                                                                      │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │  Initial Conv: 128 → 128, 3×3, padding=1                      │  │    │
│  │  │  BatchNorm + ReLU                                              │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  │                          │                                           │    │
│  │                          ▼                                           │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │               RESIDUAL BLOCK ×4                                │  │    │
│  │  │                                                                │  │    │
│  │  │  x ──────────────────────────────────────────┐                 │  │    │
│  │  │  │                                           │                 │  │    │
│  │  │  ▼                                           │                 │  │    │
│  │  │  Conv 3×3 → BN → ReLU → Conv 3×3 → BN       │                 │  │    │
│  │  │  │                                           │                 │  │    │
│  │  │  └─────────────────── + ─────────────────────┘                 │  │    │
│  │  │                       │                                        │  │    │
│  │  │                       ▼                                        │  │    │
│  │  │                     ReLU                                       │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  │                                                                      │    │
│  │  Output: [B*T, 128, 8, 8]                                           │    │
│  │                                                                      │    │
│  │  ★ CNN extracts local patterns:                                     │    │
│  │    - Pawn structures                                                │    │
│  │    - Piece configurations                                           │    │
│  │    - Tactical motifs (forks, pins)                                  │    │
│  │                                                                      │    │
│  └─────────────────────────┼───────────────────────────────────────────┘    │
│                            │                                                 │
│            Flatten spatial: [B*T, 128, 64] → [B*T, 64, 128]                 │
│            Linear project: [B*T, 64, 256]                                   │
│                            │                                                 │
│                            ▼                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                   ATTENTION LAYERS (×2)                              │    │
│  │                                                                      │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │              SELF-ATTENTION LAYER                              │  │    │
│  │  │                                                                │  │    │
│  │  │  Multi-Head Attention (4 heads, dim=256)                      │  │    │
│  │  │  │                                                             │  │    │
│  │  │  ▼                                                             │  │    │
│  │  │  Add & LayerNorm                                               │  │    │
│  │  │  │                                                             │  │    │
│  │  │  ▼                                                             │  │    │
│  │  │  FFN: Linear(256→512) → GELU → Linear(512→256)                │  │    │
│  │  │  │                                                             │  │    │
│  │  │  ▼                                                             │  │    │
│  │  │  Add & LayerNorm                                               │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  │                                                                      │    │
│  │  Output: [B*T, 64, 256]                                             │    │
│  │                                                                      │    │
│  │  ★ Attention adds global reasoning:                                 │    │
│  │    - Long diagonal relationships                                    │    │
│  │    - King safety across the board                                   │    │
│  │    - Coordination between distant pieces                            │    │
│  │                                                                      │    │
│  └─────────────────────────┼───────────────────────────────────────────┘    │
│                            │                                                 │
│              Mean pool over 64 squares: [B*T, 256]                          │
│              Reshape: [B, T, 256]                                           │
│                            │                                                 │
│                            ▼                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                   TEMPORAL AGGREGATION                               │    │
│  │                                                                      │    │
│  │  Option 1: Mean over T: [B, 256]                                    │    │
│  │  Option 2: Temporal attention                                        │    │
│  │  Option 3: Last position only                                        │    │
│  │                                                                      │    │
│  └─────────────────────────┼───────────────────────────────────────────┘    │
│                            │                                                 │
│                            ▼                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                       OUTPUT HEADS                                   │    │
│  │                                                                      │    │
│  │     ┌──────────────┐    ┌──────────────┐    ┌──────────────┐        │    │
│  │     │  MOVE HEAD   │    │  SCORE HEAD  │    │ CAPTURE/OUT  │        │    │
│  │     │ Lin → 512    │    │ Lin → 128    │    │ Lin → 64     │        │    │
│  │     │ ReLU         │    │ ReLU         │    │ ReLU         │        │    │
│  │     │ Lin → 4096   │    │ Lin → 1      │    │ Lin → 2/3    │        │    │
│  │     └──────┬───────┘    └──────┬───────┘    └──────┬───────┘        │    │
│  │            │                   │                   │                 │    │
│  └────────────┼───────────────────┼───────────────────┼─────────────────┘    │
│               │                   │                   │                      │
│               ▼                   ▼                   ▼                      │
│          move_logits          score            capture/outcome              │
│           [B, 4096]           [B, 1]            [B, 2]/[B, 3]               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## The Hybrid Advantage

### Why Combine CNN + Attention?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PATTERN RECOGNITION                                   │
├───────────────────────────┬─────────────────────────────────────────────────┤
│      CNN EXCELS AT        │           ATTENTION EXCELS AT                   │
├───────────────────────────┼─────────────────────────────────────────────────┤
│                           │                                                  │
│  Local patterns:          │  Global patterns:                               │
│                           │                                                  │
│  ┌─────┐                  │     a b c d e f g h                             │
│  │ . p .│  Pawn chain     │   8 . . . . . . . r ←─┐                        │
│  │ . P .│                 │   7 . . . . . . . . │  │                        │
│  │ . . .│                 │   6 . . . . . . . . │  │ Rook on open file     │
│  └─────┘                  │   5 . . . . . . . . │  │ attacking king        │
│                           │   4 . . . . . . . . │  │                        │
│  ┌─────┐                  │   3 . . . . . . . . │  │                        │
│  │ N . .│  Knight fork    │   2 . . . . . . . . │  │                        │
│  │ . . .│  potential      │   1 . . . . K . . .←┘  │                        │
│  │ K . Q│                 │                        │                        │
│  └─────┘                  │                        │                        │
│                           │                                                  │
│  Efficient: O(k²×H×W)     │  Expensive: O(N²) but global                    │
│  k = kernel size          │  N = 64 squares                                 │
│                           │                                                  │
└───────────────────────────┴─────────────────────────────────────────────────┘

HYBRID SOLUTION:
  1. Use CNN to efficiently extract local features
  2. Use attention (on reduced representation) for global reasoning
  3. Best of both: fast + expressive
```

### Feature Flow

```
Raw Board               CNN Features            Attention-Enhanced
[8×8 pieces]     →     [8×8×128 local]    →    [8×8×256 global]

┌─────────────┐        ┌─────────────┐        ┌─────────────┐
│ r n b q k . │        │ ████████░░░ │        │ ████████████ │
│ p p p . . . │   →    │ ████░░░░░░░ │   →    │ ████████████ │
│ . . . . . . │  CNN   │ ░░░░░░░░░░░ │  Attn  │ ██░░░░░░░░██ │
│ . . . . . . │        │ ░░░░░░░░░░░ │        │ ░░░░░░░░░░░░ │
│ . . . P . . │        │ ░░░░████░░░ │        │ ░░░░████░░░░ │
│ . . . . . . │        │ ░░░░░░░░░░░ │        │ ░░░░░░░░░░░░ │
│ P P P . P P │        │ ████░░████░ │        │ ████████████ │
│ R N B Q K . │        │ ████████░░░ │        │ ████████████ │
└─────────────┘        └─────────────┘        └─────────────┘

Local patterns          Global context added
extracted              (piece coordination,
                       long-range threats)
```

## Configuration

```python
from dataclasses import dataclass

@dataclass
class ChessHybridConfig:
    # Embedding
    num_piece_types: int = 13
    piece_embed_dim: int = 64
    position_embed_dim: int = 64

    # CNN backbone
    initial_channels: int = 128
    hidden_channels: int = 128     # Narrower than pure ResNet
    num_residual_blocks: int = 4   # Fewer than pure ResNet

    # Attention layers
    hidden_dim: int = 256          # Attention dimension
    num_attention_layers: int = 2  # Fewer than pure Transformer
    num_heads: int = 4
    ff_dim: int = 512
    dropout: float = 0.1

    # Sequence
    sequence_length: int = 5

    # Output
    num_moves: int = 4096

    # Options
    flip_board_for_black: bool = True
```

## Parameter Count

```
Component                    │ Parameters
─────────────────────────────┼───────────────
Piece Embedding              │       832
Position Embedding           │     4,096
Initial Conv + BN            │    16,512
Residual Blocks (×4)         │   295,168
Linear Projection            │    33,024
Attention Layers (×2)        │
  ├─ Self-Attention          │   263,168 × 2
  ├─ Feed-Forward            │   263,168 × 2
  └─ LayerNorms              │     1,024 × 2
Move Head                    │ 2,101,248
Score Head                   │    33,025
Capture Head                 │    16,450
Outcome Head                 │    16,579
─────────────────────────────┼───────────────
TOTAL                        │ ~3.1M
```

## Usage

### Basic

```python
from ochess.model import ChessHybrid, ChessHybridConfig

# Create model with defaults
model = ChessHybrid()

# Forward pass
boards = torch.randint(0, 13, (4, 5, 8, 8))
colors = torch.randint(0, 2, (4, 5))

outputs = model(boards, colors)
```

### Custom Configuration

```python
# Emphasize CNN (faster, more local)
config = ChessHybridConfig(
    num_residual_blocks=6,
    num_attention_layers=1,
    hidden_channels=192
)

# Emphasize Attention (slower, more global)
config = ChessHybridConfig(
    num_residual_blocks=2,
    num_attention_layers=4,
    hidden_dim=384
)
```

### Extract Both CNN and Attention Features

```python
outputs = model(boards, colors, return_features=True)

cnn_features = outputs['cnn_features']      # [B, T, 128, 8, 8]
attn_features = outputs['attn_features']    # [B, T, 64, 256]
final_features = outputs['features']        # [B, 256]
```

## Training Tips

### Balanced Learning

The hybrid needs both CNN and attention to learn well:

```python
# Use moderate learning rate (between ResNet and Transformer)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-4,              # Not too high (attention), not too low (CNN)
    weight_decay=0.01
)
```

### Batch Size

- Recommended: 64-256
- CNN part benefits from batch norm
- Attention part benefits from larger batches

### Progressive Training (Optional)

Train CNN first, then add attention:

```python
# Phase 1: Train CNN backbone
for param in model.attention_layers.parameters():
    param.requires_grad = False
train(model, epochs=10)

# Phase 2: Train full model
for param in model.parameters():
    param.requires_grad = True
train(model, epochs=90)
```

## Comparison

| Aspect | ResNet | Transformer | Hybrid |
|--------|--------|-------------|--------|
| Parameters | 2.1M | 5.2M | 3.1M |
| Local patterns | ★★★★★ | ★★★☆☆ | ★★★★★ |
| Global patterns | ★★★☆☆ | ★★★★★ | ★★★★☆ |
| Training speed | ★★★★★ | ★★☆☆☆ | ★★★★☆ |
| Memory usage | ★★★★★ | ★★☆☆☆ | ★★★★☆ |
| Interpretability | ★★☆☆☆ | ★★★★★ | ★★★★☆ |

## When to Use ChessHybrid

**Best choice when:**

- You want balanced performance
- Dataset is medium-sized (10k-100k positions)
- Both tactical and strategic reasoning matter
- You're unsure which architecture to use

**Consider alternatives when:**

- Maximum speed needed → Use ResNet
- Maximum expressiveness needed → Use Transformer
- Very limited compute → Use smaller ResNet
- Research on attention patterns → Use Transformer

## Architecture Variants

### Variant 1: CNN-Heavy

```python
config = ChessHybridConfig(
    num_residual_blocks=6,
    num_attention_layers=1,
    hidden_channels=256,
    hidden_dim=256
)
# Faster, better for tactical puzzles
```

### Variant 2: Attention-Heavy

```python
config = ChessHybridConfig(
    num_residual_blocks=2,
    num_attention_layers=4,
    hidden_channels=128,
    hidden_dim=384
)
# Slower, better for strategic positions
```

### Variant 3: Balanced (Default)

```python
config = ChessHybridConfig()  # 4 res blocks, 2 attn layers
# Good all-around performance
```
