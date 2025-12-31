# Neural Network Models Overview

Torch o'Chess implements three distinct neural network architectures for chess move prediction. Each has different strengths and trade-offs.

## Architecture Comparison

| Feature | ChessResNet | ChessTransformer | ChessHybrid |
|---------|-------------|------------------|-------------|
| Parameters | ~2M | ~5M | ~3M |
| Training Speed | Fast | Slow | Medium |
| Inference Speed | Fast | Medium | Medium |
| Local Patterns | Excellent | Good | Excellent |
| Global Patterns | Good | Excellent | Very Good |
| Memory Usage | Low | High | Medium |
| Recommended For | Production | Research | Balanced |

## Quick Selection Guide

```
Do you need fast training?
├── Yes → ChessResNet
└── No
    ├── Is long-range reasoning critical?
    │   ├── Yes → ChessTransformer
    │   └── No → ChessHybrid (best of both)
    └── Want balanced performance? → ChessHybrid
```

## Shared Architecture Elements

All three models share:

### Input Format

```python
boards: torch.Tensor  # Shape: [batch, sequence, 8, 8]
colors: torch.Tensor  # Shape: [batch, sequence]
```

### Output Format

```python
{
    'move_logits': torch.Tensor,  # [batch, 4096]
    'score': torch.Tensor,        # [batch, 1]
    'capture': torch.Tensor,      # [batch, 2]
    'outcome': torch.Tensor       # [batch, 3]
}
```

### Embeddings

```
Piece Embedding: 13 piece types → 64 dimensions
Position Embedding: 64 squares → 64 dimensions
Color Embedding: 2 colors → 64 dimensions
```

### Output Heads

```
┌───────────────────────────────────────────────────────────────┐
│                      SHARED OUTPUT HEADS                      │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  Move Head:     Linear(hidden → 4096)                         │
│                 Predicts probability over all possible moves  │
│                                                               │
│  Score Head:    Linear(hidden → 64) → Linear(64 → 1)          │
│                 Predicts position evaluation                  │
│                                                               │
│  Capture Head:  Linear(hidden → 2)                            │
│                 Predicts if best move is a capture            │
│                                                               │
│  Outcome Head:  Linear(hidden → 3)                            │
│                 Predicts game outcome (loss/draw/win)         │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

## Model 1: ChessResNet

**Philosophy**: Proven architecture from AlphaZero, optimized for chess.

```
Input [B, T, 8, 8]
        │
        ▼
┌───────────────────┐
│  Piece Embedding  │  [B, T, 8, 8, 64]
│  + Pos Embedding  │  [B, T, 8, 8, 64]
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Initial Conv2D   │  [B*T, 256, 8, 8]
│  3×3, stride=1    │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Residual Block   │ ×8
│  ┌─────────────┐  │
│  │ Conv 3×3    │  │
│  │ BatchNorm   │  │
│  │ ReLU        │  │
│  │ Conv 3×3    │  │
│  │ BatchNorm   │  │
│  │ + Skip      │  │
│  │ ReLU        │  │
│  └─────────────┘  │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Temporal Pooling  │  Aggregate sequence
└───────────────────┘
        │
        ▼
    Output Heads
```

**Strengths**:

- Fast training and inference
- Strong at tactical patterns (forks, pins, skewers)
- Well-understood architecture
- Low memory footprint

**Weaknesses**:

- Limited receptive field growth
- May miss very long-range dependencies

**Best for**: Production deployments, resource-constrained environments

---

## Model 2: ChessTransformer

**Philosophy**: Leverage self-attention to see the entire board at once.

```
Input [B, T, 8, 8]
        │
        ▼
┌───────────────────┐
│  Piece Embedding  │  [B, T, 8, 8, 64]
│  + Pos Embedding  │  [B, T, 8, 8, 64]
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Flatten to       │  [B, T, 64, 128]
│  64 square tokens │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Linear Project   │  [B, T, 64, embed_dim]
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Transformer      │ ×4
│  Encoder Layer    │
│  ┌─────────────┐  │
│  │ Self-Attn   │  │  Each square attends
│  │ (8 heads)   │  │  to all other squares
│  │ + Residual  │  │
│  │ LayerNorm   │  │
│  │ FFN         │  │
│  │ + Residual  │  │
│  │ LayerNorm   │  │
│  └─────────────┘  │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  [CLS] token or   │
│  Mean Pooling     │
└───────────────────┘
        │
        ▼
    Output Heads
```

**Strengths**:

- Global receptive field from layer 1
- Excellent at long-range patterns (rook/queen/bishop lines)
- Can learn complex positional relationships
- Attention weights are interpretable

**Weaknesses**:

- Slower training (O(N²) attention)
- Higher memory usage
- Needs more data to train effectively
- May overfit on small datasets

**Best for**: Research, large datasets, analysis of model attention

---

## Model 3: ChessHybrid

**Philosophy**: Combine CNN's local pattern efficiency with attention's global reasoning.

```
Input [B, T, 8, 8]
        │
        ▼
┌───────────────────┐
│  Piece Embedding  │
│  + Pos Embedding  │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Initial Conv2D   │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Residual Block   │ ×4 (fewer than pure ResNet)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Flatten to       │  CNN features → tokens
│  Sequence         │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Self-Attention   │ ×2 (fewer than pure Transformer)
│  Layers           │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Temporal Pool    │
└───────────────────┘
        │
        ▼
    Output Heads
```

**Strengths**:

- Best of both worlds
- CNN extracts local features efficiently
- Attention adds global reasoning on top
- Balanced parameter count
- Good performance across all pattern types

**Weaknesses**:

- More complex architecture
- Slightly slower than pure ResNet
- May need tuning to balance CNN vs attention

**Best for**: General-purpose use, when unsure which architecture fits

---

## Detailed Specifications

### ChessResNet Configuration

```python
@dataclass
class ChessResNetConfig:
    # Embedding
    num_piece_types: int = 13
    piece_embed_dim: int = 64
    position_embed_dim: int = 64

    # ResNet
    num_residual_blocks: int = 8
    hidden_channels: int = 256

    # Sequence
    sequence_length: int = 5

    # Board handling
    flip_board_for_black: bool = True

# Default model: ~2.1M parameters
model = ChessResNet(ChessResNetConfig())
```

### ChessTransformer Configuration

```python
@dataclass
class ChessTransformerConfig:
    # Embedding
    num_piece_types: int = 13
    piece_embed_dim: int = 64
    position_embed_dim: int = 64

    # Transformer
    embed_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    ff_dim: int = 1024
    dropout: float = 0.1

    # Sequence
    sequence_length: int = 5

# Default model: ~5.2M parameters
model = ChessTransformer(ChessTransformerConfig())
```

### ChessHybrid Configuration

```python
@dataclass
class ChessHybridConfig:
    # Embedding
    num_piece_types: int = 13
    piece_embed_dim: int = 64
    position_embed_dim: int = 64

    # CNN backbone
    num_residual_blocks: int = 4
    hidden_channels: int = 128

    # Attention layers
    num_attention_layers: int = 2
    num_heads: int = 4
    hidden_dim: int = 256

    # Sequence
    sequence_length: int = 5

# Default model: ~3.1M parameters
model = ChessHybrid(ChessHybridConfig())
```

## Usage Examples

### Basic Usage

```python
import torch
from ochess.model import ChessResNet, ChessTransformer, ChessHybrid

# Create model
model = ChessResNet()  # or ChessTransformer() or ChessHybrid()

# Sample input
batch_size = 4
seq_len = 5
boards = torch.randint(0, 13, (batch_size, seq_len, 8, 8))
colors = torch.randint(0, 2, (batch_size, seq_len))

# Forward pass
outputs = model(boards, colors)

# Get predictions
move_probs = torch.softmax(outputs['move_logits'], dim=-1)
best_moves = move_probs.argmax(dim=-1)
```

### With Configuration

```python
from ochess.model import ChessResNet, ChessResNetConfig

config = ChessResNetConfig(
    num_residual_blocks=12,  # Deeper
    hidden_channels=384,      # Wider
    flip_board_for_black=True
)

model = ChessResNet(config)
```

### Move Prediction with Legal Masking

```python
from ochess.data import MoveEncoder
import chess

encoder = MoveEncoder()
board = chess.Board()

# Get model output
outputs = model(boards, colors)
logits = outputs['move_logits'][0]  # First in batch

# Mask illegal moves
filtered = encoder.filter_to_legal(logits, board)

# Get best legal move
best_idx = filtered.argmax().item()
best_move = encoder.decode(best_idx)
print(f"Best move: {best_move}")
```

---

## Choosing a Model

### For Beginners

Start with **ChessResNet**. It's fast, well-understood, and gives good results.

### For Research

Use **ChessTransformer** if you want to:

- Study attention patterns
- Experiment with architecture modifications
- Have a large dataset (100k+ positions)

### For Production

- **ChessResNet**: Best speed/quality ratio
- **ChessHybrid**: If quality is more important than speed

### For Limited Resources

Use **ChessResNet** with reduced configuration:

```python
config = ChessResNetConfig(
    num_residual_blocks=4,
    hidden_channels=128
)
# Results in ~0.5M parameters
```
