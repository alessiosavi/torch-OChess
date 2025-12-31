# ChessTransformer Architecture

The Transformer-based architecture uses self-attention to model relationships between all squares on the chess board simultaneously.

## Overview

ChessTransformer treats each square as a token and uses self-attention to learn which squares are relevant to each other. This allows the model to capture long-range dependencies (like rook lines, bishop diagonals, and king safety) from the first layer.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CHESS TRANSFORMER                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: boards [B, T, 8, 8] + colors [B, T]                                 │
│                    │                                                         │
│                    ▼                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        EMBEDDING LAYER                               │    │
│  │                                                                      │    │
│  │  For each square (i, j):                                            │    │
│  │    embed = PieceEmbed(board[i,j]) + PositionEmbed(i*8 + j)         │    │
│  │                                                                      │    │
│  │  Output: [B, T, 64, 128]  (64 squares, 128 embed dims)              │    │
│  └─────────────────────────────┼───────────────────────────────────────┘    │
│                                │                                             │
│                                ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      LINEAR PROJECTION                               │    │
│  │                                                                      │    │
│  │  Linear(128 → embed_dim)  # Project to transformer dimension        │    │
│  │                                                                      │    │
│  │  Output: [B, T, 64, 256]                                            │    │
│  └─────────────────────────────┼───────────────────────────────────────┘    │
│                                │                                             │
│                    Reshape to [B*T, 64, 256]                                │
│                                │                                             │
│                    Add [CLS] token → [B*T, 65, 256]                         │
│                                │                                             │
│                                ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                  TRANSFORMER ENCODER (×4 layers)                     │    │
│  │                                                                      │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │                MULTI-HEAD SELF-ATTENTION                       │  │    │
│  │  │                                                                │  │    │
│  │  │  Input: X [B*T, 65, 256]                                      │  │    │
│  │  │                                                                │  │    │
│  │  │  Q = X @ W_Q    [B*T, 65, 256]                                │  │    │
│  │  │  K = X @ W_K    [B*T, 65, 256]                                │  │    │
│  │  │  V = X @ W_V    [B*T, 65, 256]                                │  │    │
│  │  │                                                                │  │    │
│  │  │  Split into 8 heads: [B*T, 8, 65, 32]                         │  │    │
│  │  │                                                                │  │    │
│  │  │  Attention(Q, K, V):                                          │  │    │
│  │  │    scores = Q @ K^T / sqrt(32)   [B*T, 8, 65, 65]            │  │    │
│  │  │    weights = softmax(scores)                                  │  │    │
│  │  │    output = weights @ V          [B*T, 8, 65, 32]            │  │    │
│  │  │                                                                │  │    │
│  │  │  Concat heads → [B*T, 65, 256]                                │  │    │
│  │  │  Output projection: Linear(256 → 256)                         │  │    │
│  │  │                                                                │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  │                         │                                            │    │
│  │                         ▼                                            │    │
│  │                    Add & LayerNorm                                   │    │
│  │                         │                                            │    │
│  │                         ▼                                            │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │                  FEED-FORWARD NETWORK                          │  │    │
│  │  │                                                                │  │    │
│  │  │  Linear(256 → 1024)                                           │  │    │
│  │  │  GELU activation                                               │  │    │
│  │  │  Dropout(0.1)                                                  │  │    │
│  │  │  Linear(1024 → 256)                                           │  │    │
│  │  │  Dropout(0.1)                                                  │  │    │
│  │  │                                                                │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  │                         │                                            │    │
│  │                         ▼                                            │    │
│  │                    Add & LayerNorm                                   │    │
│  │                                                                      │    │
│  └─────────────────────────┼───────────────────────────────────────────┘    │
│                            │                                                 │
│                            ▼                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    TEMPORAL AGGREGATION                              │    │
│  │                                                                      │    │
│  │  Extract [CLS] token: [B*T, 256] → reshape [B, T, 256]              │    │
│  │  Temporal attention or mean pooling → [B, 256]                      │    │
│  │                                                                      │    │
│  └─────────────────────────┼───────────────────────────────────────────┘    │
│                            │                                                 │
│                            ▼                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                       OUTPUT HEADS                                   │    │
│  │                                                                      │    │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐        │    │
│  │  │   MOVE HEAD    │  │   SCORE HEAD   │  │ CAPTURE/OUTCOME│        │    │
│  │  │                │  │                │  │                │        │    │
│  │  │ Linear → 1024  │  │ Linear → 256   │  │ Linear → 64    │        │    │
│  │  │ ReLU           │  │ ReLU           │  │ ReLU           │        │    │
│  │  │ Linear → 4096  │  │ Linear → 1     │  │ Linear → 2/3   │        │    │
│  │  │                │  │                │  │                │        │    │
│  │  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘        │    │
│  │          │                   │                   │                  │    │
│  └──────────┼───────────────────┼───────────────────┼──────────────────┘    │
│             │                   │                   │                       │
│             ▼                   ▼                   ▼                       │
│        move_logits          score            capture/outcome               │
│         [B, 4096]           [B, 1]            [B, 2]/[B, 3]                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Self-Attention Visualization

### How Attention Works for Chess

```
Query: "What should I attend to from square e4?"

     a    b    c    d    e    f    g    h
  ┌────┬────┬────┬────┬────┬────┬────┬────┐
8 │ .2 │ .1 │ .3 │ .4 │ .8 │ .3 │ .1 │ .2 │  ← High attention on e8 (queen)
  ├────┼────┼────┼────┼────┼────┼────┼────┤
7 │ .1 │ .1 │ .2 │ .5 │ .6 │ .4 │ .1 │ .1 │
  ├────┼────┼────┼────┼────┼────┼────┼────┤
6 │ .1 │ .1 │ .4 │ .3 │ .5 │ .3 │ .3 │ .1 │
  ├────┼────┼────┼────┼────┼────┼────┼────┤
5 │ .2 │ .2 │ .3 │ .6 │ .7 │ .5 │ .2 │ .1 │  ← Diagonal attention
  ├────┼────┼────┼────┼────┼────┼────┼────┤
4 │ .3 │ .4 │ .5 │ .7 │ ■  │ .7 │ .4 │ .3 │  ← e4 = query square
  ├────┼────┼────┼────┼────┼────┼────┼────┤
3 │ .2 │ .2 │ .4 │ .6 │ .7 │ .5 │ .3 │ .2 │
  ├────┼────┼────┼────┼────┼────┼────┼────┤
2 │ .1 │ .1 │ .3 │ .4 │ .5 │ .4 │ .2 │ .1 │
  ├────┼────┼────┼────┼────┼────┼────┼────┤
1 │ .1 │ .1 │ .2 │ .3 │ .6 │ .3 │ .2 │ .1 │  ← High on e1 (king)
  └────┴────┴────┴────┴────┴────┴────┴────┘

Attention naturally learns:
- File/rank relationships (rook-like)
- Diagonal relationships (bishop-like)
- Knight-jump patterns
- King safety zones
```

### Multi-Head Attention

Different heads learn different patterns:

```
Head 1: File attention        Head 2: Diagonal attention
     a b c d e f g h              a b c d e f g h
   ┌─────────────────┐          ┌─────────────────┐
 8 │ . . . . ■ . . . │        8 │ ■ . . . . . . . │
 7 │ . . . . ■ . . . │        7 │ . ■ . . . . . . │
 6 │ . . . . ■ . . . │        6 │ . . ■ . . . . . │
 5 │ . . . . ■ . . . │        5 │ . . . ■ . . . . │
 4 │ . . . . ★ . . . │        4 │ . . . . ★ . . . │
 3 │ . . . . ■ . . . │        3 │ . . . . . ■ . . │
 2 │ . . . . ■ . . . │        2 │ . . . . . . ■ . │
 1 │ . . . . ■ . . . │        1 │ . . . . . . . ■ │
   └─────────────────┘          └─────────────────┘

Head 3: Knight-like           Head 4: King safety
     a b c d e f g h              a b c d e f g h
   ┌─────────────────┐          ┌─────────────────┐
 8 │ . . . . . . . . │        8 │ . . . . . . . . │
 7 │ . . . . . . . . │        7 │ . . . . . . . . │
 6 │ . . . ■ . ■ . . │        6 │ . . . . . . . . │
 5 │ . . ■ . . . ■ . │        5 │ . . . ■ ■ ■ . . │
 4 │ . . . . ★ . . . │        4 │ . . . ■ ★ ■ . . │
 3 │ . . ■ . . . ■ . │        3 │ . . . ■ ■ ■ . . │
 2 │ . . . ■ . ■ . . │        2 │ . . . . . . . . │
 1 │ . . . . . . . . │        1 │ . . . . . . . . │
   └─────────────────┘          └─────────────────┘
```

## Configuration

```python
from dataclasses import dataclass

@dataclass
class ChessTransformerConfig:
    # Piece embedding
    num_piece_types: int = 13
    piece_embed_dim: int = 64

    # Position embedding
    num_squares: int = 64
    position_embed_dim: int = 64

    # Transformer
    embed_dim: int = 256           # Model dimension
    num_heads: int = 8             # Attention heads
    num_layers: int = 4            # Encoder layers
    ff_dim: int = 1024             # Feed-forward dimension
    dropout: float = 0.1           # Dropout rate

    # Sequence
    sequence_length: int = 5

    # Output
    num_moves: int = 4096

    # Options
    use_cls_token: bool = True     # Use [CLS] for classification
    flip_board_for_black: bool = True
```

## Parameter Count

```
Component                    │ Parameters
─────────────────────────────┼───────────────
Piece Embedding              │       832
Position Embedding           │     4,096
CLS Token                    │       256
Linear Projection            │    33,024
Transformer Layer (×4)       │
  ├─ Self-Attention          │   263,168 × 4
  ├─ Feed-Forward            │   525,312 × 4
  └─ LayerNorms              │     1,024 × 4
Move Head                    │ 4,198,400
Score Head                   │    65,793
Capture Head                 │    16,450
Outcome Head                 │    16,579
─────────────────────────────┼───────────────
TOTAL                        │ ~5.2M
```

## Usage

### Basic

```python
from ochess.model import ChessTransformer, ChessTransformerConfig

# Create model
model = ChessTransformer()

# Forward pass
boards = torch.randint(0, 13, (4, 5, 8, 8))
colors = torch.randint(0, 2, (4, 5))

outputs = model(boards, colors)
```

### Custom Configuration

```python
config = ChessTransformerConfig(
    num_layers=6,          # Deeper
    num_heads=16,          # More heads
    embed_dim=512,         # Wider
    ff_dim=2048,
    dropout=0.2            # More regularization
)

model = ChessTransformer(config)
```

### Extract Attention Weights

```python
# Get attention maps for visualization
outputs = model(boards, colors, return_attention=True)

attention_weights = outputs['attention']  # List of [B, H, N, N]

# Visualize attention from e4 (square 28) in layer 0, head 0
import matplotlib.pyplot as plt

attn = attention_weights[0][0, 0, 28, :64]  # [64]
attn = attn.reshape(8, 8).detach().numpy()

plt.imshow(attn, cmap='hot')
plt.title("Attention from e4")
plt.colorbar()
```

## Training Tips

### Learning Rate

Transformers are sensitive to learning rate:

```python
# Use warmup + decay
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.98),    # Transformer defaults
    weight_decay=0.01
)

# Linear warmup + cosine decay
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1000,
    num_training_steps=total_steps
)
```

### Batch Size

- Transformers benefit from larger batches
- Recommended: 64-256 (memory permitting)
- Use gradient accumulation for effective larger batches

```python
accumulation_steps = 4
for i, batch in enumerate(loader):
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Regularization

Transformers can overfit easily:

```python
config = ChessTransformerConfig(
    dropout=0.2,           # Increase dropout
)

# Also use:
# - Label smoothing
# - Weight decay
# - Early stopping
```

### Mixed Precision

Recommended for memory efficiency:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(boards, colors)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Advantages Over ResNet

1. **Global receptive field**: See entire board from layer 1
2. **Explicit relational reasoning**: Attention directly models piece relationships
3. **Interpretability**: Attention weights show what the model "looks at"
4. **Flexible sequence handling**: Natural handling of variable-length history

## Disadvantages

1. **Slower training**: O(N²) attention complexity
2. **Higher memory**: Stores attention matrices
3. **Needs more data**: More parameters to train
4. **Position encoding**: Must learn chess-specific positional patterns

## When to Use

**Use ChessTransformer when:**
- You have a large dataset (100k+ positions)
- Long-range patterns matter (endgames, long diagonals)
- You want to analyze attention patterns
- Research/experimentation is the goal

**Use ChessResNet when:**
- Speed is important
- Dataset is smaller
- Production deployment is the goal
