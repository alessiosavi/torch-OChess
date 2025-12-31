# Data Module API Reference

The `ochess.data` module provides utilities for handling chess data: FEN parsing, move encoding, score encoding, and PyTorch datasets.

## Module Overview

```python
from ochess.data import (
    FenParser,
    MoveEncoder,
    ScoreEncoder,
    ChessDataset,
    create_dataloader
)
```

---

## FenParser

Converts FEN strings to tensors and vice versa.

### Class Definition

```python
class FenParser:
    """Parse FEN strings to tensors and back."""

    # Piece mapping
    PIECE_TO_INDEX = {
        ' ': 0,   # Empty
        'P': 1,   # White pawn
        'N': 2,   # White knight
        'B': 3,   # White bishop
        'R': 4,   # White rook
        'Q': 5,   # White queen
        'K': 6,   # White king
        'p': 7,   # Black pawn
        'n': 8,   # Black knight
        'b': 9,   # Black bishop
        'r': 10,  # Black rook
        'q': 11,  # Black queen
        'k': 12,  # Black king
    }

    INDEX_TO_PIECE = {v: k for k, v in PIECE_TO_INDEX.items()}
```

### Methods

#### `to_tensor(fen: str) -> torch.Tensor`

Convert FEN string to 8x8 tensor of piece indices.

```python
parser = FenParser()

# Starting position
fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
tensor = parser.to_tensor(fen)

print(tensor.shape)  # torch.Size([8, 8])
print(tensor.dtype)  # torch.int64

# Check white king at e1
print(tensor[7, 4])  # 6 (white king)
```

#### `to_fen(tensor: torch.Tensor) -> str`

Convert tensor back to FEN board string.

```python
tensor = parser.to_tensor(fen)
board_fen = parser.to_fen(tensor)

print(board_fen)  # "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
```

#### `get_active_color(fen: str) -> int`

Extract active color from FEN (0 = White, 1 = Black).

```python
white_fen = "8/8/8/8/8/8/8/8 w - - 0 1"
black_fen = "8/8/8/8/8/8/8/8 b - - 0 1"

parser.get_active_color(white_fen)  # 0
parser.get_active_color(black_fen)  # 1
```

#### `is_white_to_move(fen: str) -> bool`

Check if White is to move.

```python
parser.is_white_to_move(white_fen)  # True
parser.is_white_to_move(black_fen)  # False
```

#### `flip_board(tensor: torch.Tensor) -> torch.Tensor`

Flip board 180 degrees (rotate).

```python
flipped = parser.flip_board(tensor)
# tensor[0, 0] == flipped[7, 7]
```

#### `swap_colors(tensor: torch.Tensor) -> torch.Tensor`

Swap White pieces (1-6) with Black pieces (7-12).

```python
swapped = parser.swap_colors(tensor)
# White king (6) becomes Black king (12)
# Black pawn (7) becomes White pawn (1)
```

#### `normalize_for_player(tensor: torch.Tensor, white_to_move: bool) -> torch.Tensor`

Normalize board to current player's perspective.

```python
# For White: no change
# For Black: flip board AND swap colors
normalized = parser.normalize_for_player(tensor, white_to_move=False)
```

#### `display(tensor: torch.Tensor) -> str`

Get ASCII representation of board.

```python
print(parser.display(tensor))
#   a b c d e f g h
# 8 r n b q k b n r
# 7 p p p p p p p p
# 6 . . . . . . . .
# 5 . . . . . . . .
# 4 . . . . . . . .
# 3 . . . . . . . .
# 2 P P P P P P P P
# 1 R N B Q K B N R
```

---

## MoveEncoder

Encode chess moves as indices (0-4095).

### Class Definition

```python
class MoveEncoder:
    """Encode UCI moves as indices."""

    NUM_MOVES = 4096  # 64 * 64
```

### Methods

#### `encode(uci: str) -> int`

Encode UCI move string to index.

```python
encoder = MoveEncoder()

idx = encoder.encode("e2e4")
print(idx)  # 796 (12 * 64 + 28)

idx = encoder.encode("g1f3")
print(idx)  # 405
```

#### `decode(index: int) -> str`

Decode index to UCI move string.

```python
uci = encoder.decode(796)
print(uci)  # "e2e4"
```

#### `encode_move(move: chess.Move) -> int`

Encode python-chess Move object.

```python
import chess

move = chess.Move.from_uci("e2e4")
idx = encoder.encode_move(move)
```

#### `decode_to_move(index: int) -> chess.Move`

Decode index to python-chess Move.

```python
move = encoder.decode_to_move(796)
print(move.uci())  # "e2e4"
```

#### `encode_batch(moves: List[str]) -> torch.Tensor`

Batch encode multiple moves.

```python
moves = ["e2e4", "d2d4", "g1f3"]
indices = encoder.encode_batch(moves)

print(indices.shape)  # torch.Size([3])
print(indices.dtype)  # torch.int64
```

#### `decode_batch(indices: torch.Tensor) -> List[str]`

Batch decode indices.

```python
indices = torch.tensor([796, 843, 405])
moves = encoder.decode_batch(indices)

print(moves)  # ['e2e4', 'd2d4', 'g1f3']
```

#### `get_legal_move_mask(board: chess.Board) -> torch.Tensor`

Get boolean mask of legal moves.

```python
board = chess.Board()
mask = encoder.get_legal_move_mask(board)

print(mask.shape)  # torch.Size([4096])
print(mask.sum())  # 20 (starting position has 20 legal moves)
```

#### `filter_to_legal(logits: torch.Tensor, board: chess.Board) -> torch.Tensor`

Mask illegal moves to -inf.

```python
logits = model(board_tensor)['move_logits']
filtered = encoder.filter_to_legal(logits, board)

# Illegal moves are now -inf
best_legal = filtered.argmax()
```

#### `to_onehot(index: int) -> torch.Tensor`

Convert index to one-hot vector.

```python
onehot = encoder.to_onehot(796)
print(onehot.shape)  # torch.Size([4096])
print(onehot.sum())  # 1.0
```

#### `get_from_to_squares(index: int) -> Tuple[int, int]`

Extract from and to squares from index.

```python
from_sq, to_sq = encoder.get_from_to_squares(796)
print(chess.square_name(from_sq))  # "e2"
print(chess.square_name(to_sq))    # "e4"
```

#### `normalize_move_for_black(index: int) -> int`

Normalize move index for Black's perspective.

```python
# When board is flipped for Black, move indices need adjustment
white_idx = encoder.encode("e2e4")
black_idx = encoder.normalize_move_for_black(white_idx)

# Reverse operation
restored = encoder.denormalize_move_for_black(black_idx)
assert restored == white_idx
```

---

## ScoreEncoder

Encode evaluation scores with proper perspective handling.

### Class Definition

```python
@dataclass
class ScoreEncoderConfig:
    normalize: bool = True       # Normalize to [-15, 15] range
    scale: float = 1000.0        # Normalization scale
    max_score: int = 15000       # Maximum centipawn value
    mate_score: int = 10000      # Base value for mate scores

class ScoreEncoder:
    """Encode chess scores with side-to-move perspective."""

    def __init__(self, normalize: bool = True, scale: float = 1000.0):
        ...
```

### Methods

#### `encode(score_str: str, white_to_move: bool) -> float`

Encode score string to normalized value.

```python
encoder = ScoreEncoder()

# White winning +150cp, White to move
score = encoder.encode("+150", white_to_move=True)
print(score)  # 0.15 (good for me)

# White winning +150cp, Black to move
score = encoder.encode("+150", white_to_move=False)
print(score)  # -0.15 (bad for me)  ← THE CRITICAL FIX
```

#### `encode_from_centipawns(cp: int, white_to_move: bool) -> float`

Encode integer centipawn value.

```python
score = encoder.encode_from_centipawns(150, white_to_move=True)
print(score)  # 0.15
```

#### `decode(value: float, white_to_move: bool) -> int`

Decode normalized value back to centipawns.

```python
cp = encoder.decode(0.15, white_to_move=True)
print(cp)  # 150
```

#### `is_mate_score(score_str: str) -> bool`

Check if score represents mate.

```python
encoder.is_mate_score("#+5")   # True
encoder.is_mate_score("#-3")   # True
encoder.is_mate_score("M5")    # True
encoder.is_mate_score("+150")  # False
```

#### `get_mate_distance(score_str: str) -> Optional[int]`

Get mate distance from score string.

```python
encoder.get_mate_distance("#+5")   # 5 (White mates in 5)
encoder.get_mate_distance("#-3")   # -3 (Black mates in 3)
encoder.get_mate_distance("+150")  # None (not a mate)
```

#### `score_to_win_probability(score: float) -> float`

Convert score to estimated win probability.

```python
# Equal position
encoder.score_to_win_probability(0.0)  # ~0.5

# Winning (+1000cp)
encoder.score_to_win_probability(1.0)  # ~0.9

# Losing (-1000cp)
encoder.score_to_win_probability(-1.0)  # ~0.1
```

#### `explain_score(centipawns: int) -> str`

Get human-readable explanation.

```python
ScoreEncoder.explain_score(150)    # "Slight advantage (+1.5 pawns)"
ScoreEncoder.explain_score(-300)   # "Clear disadvantage (-3 pawns)"
ScoreEncoder.explain_score(9500)   # "Mate in sight"
```

---

## ChessDataset

PyTorch Dataset for chess positions with caching.

### Class Definition

```python
class ChessDataset(torch.utils.data.Dataset):
    """Dataset for chess positions with tensor caching."""

    def __init__(
        self,
        data_path: str,
        sequence_length: int = 5,
        cache_dir: Optional[str] = None,
        flip_for_black: bool = True
    ):
        ...
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_path` | str | required | Path to parquet file |
| `sequence_length` | int | 5 | Number of positions per sample |
| `cache_dir` | str | None | Directory for tensor cache |
| `flip_for_black` | bool | True | Normalize perspective |

### Methods

#### `__len__() -> int`

Get dataset size.

```python
dataset = ChessDataset("data.parquet")
print(len(dataset))  # Number of samples
```

#### `__getitem__(idx: int) -> Dict[str, torch.Tensor]`

Get single sample.

```python
sample = dataset[0]

print(sample.keys())
# dict_keys(['boards', 'colors', 'target_move', 'target_score',
#            'is_capture', 'game_id'])

print(sample['boards'].shape)       # [5, 8, 8]
print(sample['colors'].shape)       # [5]
print(sample['target_move'].shape)  # []
print(sample['target_score'].shape) # []
```

### Usage Example

```python
from ochess.data import ChessDataset, create_dataloader

# Create dataset with caching
dataset = ChessDataset(
    data_path="datasets/train_data.parquet",
    sequence_length=5,
    cache_dir="datasets/cache"
)

# Create dataloader
train_loader = create_dataloader(
    dataset,
    batch_size=256,
    shuffle=True,
    num_workers=4
)

# Training loop
for batch in train_loader:
    boards = batch['boards']        # [B, T, 8, 8]
    colors = batch['colors']        # [B, T]
    target_move = batch['target_move']  # [B]
    target_score = batch['target_score']  # [B]

    outputs = model(boards, colors)
    loss = criterion(outputs, targets)
```

---

## create_dataloader

Convenience function to create DataLoader.

### Function Signature

```python
def create_dataloader(
    dataset: ChessDataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False
) -> DataLoader:
    ...
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | ChessDataset | required | Dataset instance |
| `batch_size` | int | 64 | Samples per batch |
| `shuffle` | bool | True | Shuffle data |
| `num_workers` | int | 0 | Parallel data loading workers |
| `pin_memory` | bool | True | Pin memory for GPU transfer |
| `drop_last` | bool | False | Drop incomplete last batch |

### Example

```python
from ochess.data import ChessDataset, create_dataloader

dataset = ChessDataset("data.parquet")

# For training
train_loader = create_dataloader(
    dataset,
    batch_size=256,
    shuffle=True,
    num_workers=4
)

# For validation (no shuffle)
val_loader = create_dataloader(
    dataset,
    batch_size=256,
    shuffle=False,
    num_workers=4
)
```

---

## Complete Example

```python
import torch
from ochess.data import (
    FenParser,
    MoveEncoder,
    ScoreEncoder,
    ChessDataset,
    create_dataloader
)

# Initialize utilities
parser = FenParser()
move_encoder = MoveEncoder()
score_encoder = ScoreEncoder()

# Parse a FEN
fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
board_tensor = parser.to_tensor(fen)
color = parser.get_active_color(fen)

# Encode a move
move_idx = move_encoder.encode("e7e5")

# Encode a score (Black to move, White slightly better)
score = score_encoder.encode("+50", white_to_move=False)
print(f"Score from Black's view: {score}")  # -0.05 (bad for me)

# Load dataset
dataset = ChessDataset(
    "datasets/train_data.parquet",
    cache_dir="datasets/cache"
)

loader = create_dataloader(dataset, batch_size=64)

for batch in loader:
    print(f"Batch boards: {batch['boards'].shape}")
    print(f"Batch moves: {batch['target_move'].shape}")
    break
```
