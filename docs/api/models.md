# Models API Reference

The `ochess.model` module provides neural network architectures and supporting components.

## Module Overview

```python
from ochess.model import (
    # Main models
    ChessResNet,
    ChessResNetConfig,
    ChessTransformer,
    ChessTransformerConfig,
    ChessHybrid,
    ChessHybridConfig,

    # Loss functions
    ChessLoss,
    ChessLossConfig,

    # Components (advanced usage)
    PieceEmbedding,
    PositionEmbedding,
    ResidualBlock,
    MoveHead,
    ScoreHead
)
```

---

## ChessResNet

ResNet-style architecture for chess.

### Class Definition

```python
class ChessResNet(nn.Module):
    """AlphaZero-inspired residual network for chess."""

    def __init__(self, config: ChessResNetConfig = None):
        ...
```

### Configuration

```python
@dataclass
class ChessResNetConfig:
    # Embeddings
    num_piece_types: int = 13
    piece_embed_dim: int = 64
    position_embed_dim: int = 64

    # Architecture
    hidden_channels: int = 256
    num_residual_blocks: int = 8

    # Sequence
    sequence_length: int = 5

    # Output
    num_moves: int = 4096

    # Options
    flip_board_for_black: bool = True
```

### Methods

#### `forward(boards, colors, return_features=False) -> Dict[str, Tensor]`

```python
model = ChessResNet()

boards = torch.randint(0, 13, (4, 5, 8, 8))  # [B, T, H, W]
colors = torch.randint(0, 2, (4, 5))          # [B, T]

outputs = model(boards, colors)

# outputs contains:
# - 'move_logits': [4, 4096]
# - 'score': [4, 1]
# - 'capture': [4, 2]
# - 'outcome': [4, 3]

# With features
outputs = model(boards, colors, return_features=True)
features = outputs['features']  # [4, 256, 8, 8]
```

#### `predict_move(boards, colors) -> Tensor`

Get move probabilities (softmax applied).

```python
probs = model.predict_move(boards, colors)  # [B, 4096]
best_moves = probs.argmax(dim=-1)           # [B]
```

#### `get_config() -> ChessResNetConfig`

Get model configuration.

```python
config = model.get_config()
print(config.num_residual_blocks)  # 8
```

---

## ChessTransformer

Transformer architecture for chess.

### Class Definition

```python
class ChessTransformer(nn.Module):
    """Self-attention based chess model."""

    def __init__(self, config: ChessTransformerConfig = None):
        ...
```

### Configuration

```python
@dataclass
class ChessTransformerConfig:
    # Embeddings
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

    # Output
    num_moves: int = 4096

    # Options
    use_cls_token: bool = True
    flip_board_for_black: bool = True
```

### Methods

#### `forward(boards, colors, return_attention=False) -> Dict[str, Tensor]`

```python
model = ChessTransformer()

outputs = model(boards, colors)
# Same outputs as ResNet

# Get attention weights
outputs = model(boards, colors, return_attention=True)
attention = outputs['attention']  # List of [B, heads, N, N]
```

---

## ChessHybrid

Hybrid CNN + Attention architecture.

### Class Definition

```python
class ChessHybrid(nn.Module):
    """Hybrid architecture combining CNN and attention."""

    def __init__(self, config: ChessHybridConfig = None):
        ...
```

### Configuration

```python
@dataclass
class ChessHybridConfig:
    # Embeddings
    num_piece_types: int = 13
    piece_embed_dim: int = 64
    position_embed_dim: int = 64

    # CNN backbone
    hidden_channels: int = 128
    num_residual_blocks: int = 4

    # Attention
    hidden_dim: int = 256
    num_attention_layers: int = 2
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

### Methods

Same interface as other models, plus:

#### `forward(..., return_features=True)`

```python
outputs = model(boards, colors, return_features=True)

cnn_features = outputs['cnn_features']    # CNN output
attn_features = outputs['attn_features']  # After attention
features = outputs['features']            # Final features
```

---

## ChessLoss

Multi-task loss function.

### Class Definition

```python
class ChessLoss(nn.Module):
    """Combined loss for multi-task chess learning."""

    def __init__(self, config: ChessLossConfig = None):
        ...
```

### Configuration

```python
@dataclass
class ChessLossConfig:
    # Loss weights
    move_weight: float = 3.0
    score_weight: float = 0.5
    capture_weight: float = 1.0
    outcome_weight: float = 1.0

    # Loss functions
    score_loss_type: str = "huber"  # "huber" or "mse"
    label_smoothing: float = 0.0

    # Optional
    use_focal_loss: bool = False
    focal_gamma: float = 2.0
```

### Methods

#### `forward(outputs, targets) -> Dict[str, Tensor]`

```python
criterion = ChessLoss()

outputs = model(boards, colors)
targets = {
    'move': target_moves,      # [B]
    'score': target_scores,    # [B, 1]
    'capture': is_capture,     # [B]
    'outcome': outcomes        # [B]
}

losses = criterion(outputs, targets)

print(losses.keys())
# ['total', 'move', 'score', 'capture', 'outcome']

total_loss = losses['total']
total_loss.backward()
```

### Accessing Individual Losses

```python
losses = criterion(outputs, targets)

print(f"Total: {losses['total'].item():.4f}")
print(f"Move:  {losses['move'].item():.4f}")
print(f"Score: {losses['score'].item():.4f}")
```

---

## Components (Advanced)

### PieceEmbedding

```python
from ochess.model.components import PieceEmbedding

embed = PieceEmbedding(num_pieces=13, embed_dim=64)
pieces = torch.randint(0, 13, (4, 8, 8))
embedded = embed(pieces)  # [4, 8, 8, 64]
```

### PositionEmbedding

```python
from ochess.model.components import PositionEmbedding

embed = PositionEmbedding(num_positions=64, embed_dim=64)
# Adds positional information to input
```

### ResidualBlock

```python
from ochess.model.components import ResidualBlock

block = ResidualBlock(channels=256)
x = torch.randn(4, 256, 8, 8)
out = block(x)  # Same shape, with residual connection
```

### Output Heads

```python
from ochess.model.components import MoveHead, ScoreHead, CaptureHead

move_head = MoveHead(in_features=256, num_moves=4096)
score_head = ScoreHead(in_features=256)
capture_head = CaptureHead(in_features=256)

features = torch.randn(4, 256)
move_logits = move_head(features)  # [4, 4096]
score = score_head(features)       # [4, 1]
capture = capture_head(features)   # [4, 2]
```

---

## Model Factory

Create models by name.

```python
from ochess.model import create_model

# By name
model = create_model("resnet")
model = create_model("transformer")
model = create_model("hybrid")

# With config
model = create_model("resnet", num_residual_blocks=12)
model = create_model("transformer", num_layers=6, embed_dim=512)
```

---

## Saving and Loading

### Save Model

```python
# Save state dict
torch.save(model.state_dict(), "model.pt")

# Save complete checkpoint
torch.save({
    'model_state_dict': model.state_dict(),
    'config': model.get_config(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch
}, "checkpoint.pt")
```

### Load Model

```python
# Load state dict
model = ChessResNet()
model.load_state_dict(torch.load("model.pt"))

# Load from checkpoint
checkpoint = torch.load("checkpoint.pt")
config = checkpoint['config']
model = ChessResNet(config)
model.load_state_dict(checkpoint['model_state_dict'])
```

### Load for Inference

```python
from ochess.model import load_model

# Auto-detect model type
model = load_model("checkpoints/best_model.pt")
model.eval()

with torch.no_grad():
    outputs = model(boards, colors)
```

---

## Complete Example

```python
import torch
from ochess.model import (
    ChessResNet,
    ChessResNetConfig,
    ChessLoss,
    ChessLossConfig
)
from ochess.data import ChessDataset, create_dataloader

# Create model
config = ChessResNetConfig(
    num_residual_blocks=8,
    hidden_channels=256,
    flip_board_for_black=True
)
model = ChessResNet(config)

# Move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Create loss
criterion = ChessLoss(ChessLossConfig(
    move_weight=3.0,
    score_weight=0.5
))

# Create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Load data
dataset = ChessDataset("train_data.parquet")
loader = create_dataloader(dataset, batch_size=256)

# Training step
model.train()
for batch in loader:
    boards = batch['boards'].to(device)
    colors = batch['colors'].to(device)

    # Forward
    outputs = model(boards, colors)

    # Compute loss
    targets = {
        'move': batch['target_move'].to(device),
        'score': batch['target_score'].to(device),
        'capture': batch['is_capture'].to(device)
    }
    losses = criterion(outputs, targets)

    # Backward
    optimizer.zero_grad()
    losses['total'].backward()
    optimizer.step()

    print(f"Loss: {losses['total'].item():.4f}")
    break

# Inference
model.eval()
with torch.no_grad():
    test_boards = torch.randint(0, 13, (1, 5, 8, 8)).to(device)
    test_colors = torch.zeros(1, 5, dtype=torch.long).to(device)

    outputs = model(test_boards, test_colors)
    probs = torch.softmax(outputs['move_logits'], dim=-1)

    best_move_idx = probs.argmax().item()
    confidence = probs.max().item()

    print(f"Best move index: {best_move_idx}")
    print(f"Confidence: {confidence:.2%}")
```
