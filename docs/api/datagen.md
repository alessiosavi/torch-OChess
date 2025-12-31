# Data Generation API Reference

The `ochess.datagen` module provides tools for generating labeled chess positions using Stockfish.

## Module Overview

```python
from ochess.datagen import (
    StockfishWrapper,
    PositionGenerator,
    DataGenerationPipeline,
    LabeledPosition
)
```

---

## LabeledPosition

Data class representing a labeled chess position.

### Definition

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class LabeledPosition:
    """A chess position with Stockfish labels."""

    # Position
    fen: str                          # FEN string
    board_tensor: Optional[torch.Tensor] = None  # [8, 8] tensor

    # Labels
    best_move_uci: str                # Stockfish best move
    score_cp: int                     # Centipawns (White's view)
    score_normalized: float           # Normalized (side-to-move view)

    # Metadata
    white_to_move: bool               # Active color
    is_capture: bool                  # Best move is capture
    is_check: bool                    # Best move gives check
    is_mate: bool                     # Position is mate

    # Contrastive learning (optional)
    bad_move_uci: Optional[str] = None        # A suboptimal move
    bad_move_score_cp: Optional[int] = None   # Score after bad move
```

### Usage

```python
# Create manually
position = LabeledPosition(
    fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    best_move_uci="e7e5",
    score_cp=30,
    score_normalized=-0.03,  # Black to move, slightly worse
    white_to_move=False,
    is_capture=False,
    is_check=False,
    is_mate=False
)

# Convert to dict for dataframe
row = position.to_dict()
```

---

## StockfishWrapper

Interface to Stockfish chess engine.

### Class Definition

```python
class StockfishWrapper:
    """Wrapper for Stockfish engine."""

    def __init__(
        self,
        stockfish_path: str = "/usr/local/bin/stockfish",
        depth: int = 20,
        threads: int = 1,
        hash_mb: int = 128
    ):
        ...
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `stockfish_path` | str | "/usr/local/bin/stockfish" | Path to Stockfish binary |
| `depth` | int | 20 | Search depth |
| `threads` | int | 1 | CPU threads |
| `hash_mb` | int | 128 | Hash table size in MB |

### Methods

#### `analyze(board: chess.Board) -> Dict`

Get Stockfish analysis of position.

```python
import chess
from ochess.datagen import StockfishWrapper

sf = StockfishWrapper(stockfish_path="/usr/local/bin/stockfish")
board = chess.Board()

result = sf.analyze(board)
print(result)
# {
#     'best_move': 'e2e4',
#     'score_cp': 30,
#     'score_mate': None,
#     'pv': ['e2e4', 'e7e5', 'g1f3', ...]
# }
```

#### `get_best_move(board: chess.Board) -> str`

Get just the best move.

```python
best = sf.get_best_move(board)
print(best)  # "e2e4"
```

#### `get_score(board: chess.Board) -> int`

Get evaluation in centipawns.

```python
score = sf.get_score(board)
print(score)  # 30 (White's view)
```

#### `get_top_moves(board: chess.Board, n: int = 3) -> List[Dict]`

Get top N moves with evaluations.

```python
top_moves = sf.get_top_moves(board, n=3)
print(top_moves)
# [
#     {'move': 'e2e4', 'score_cp': 30},
#     {'move': 'd2d4', 'score_cp': 25},
#     {'move': 'g1f3', 'score_cp': 20}
# ]
```

#### `is_position_tactical(board: chess.Board) -> bool`

Check if position has tactical elements.

```python
# Returns True if:
# - Pieces are hanging
# - There are checks available
# - Material is imbalanced
tactical = sf.is_position_tactical(board)
```

#### `set_skill_level(level: int)`

Set Stockfish skill level (0-20).

```python
sf.set_skill_level(10)  # Medium strength
```

#### `close()`

Close Stockfish process.

```python
sf.close()  # Always call when done
```

### Context Manager

```python
# Recommended usage
with StockfishWrapper() as sf:
    result = sf.analyze(board)
    # Automatically closes
```

---

## PositionGenerator

Generate diverse chess positions.

### Class Definition

```python
class PositionGenerator:
    """Generate chess positions using various methods."""

    def __init__(
        self,
        random_seed: Optional[int] = None
    ):
        ...
```

### Methods

#### `generate_random_game(max_moves: int = 80) -> List[chess.Board]`

Generate positions from a random game.

```python
gen = PositionGenerator(random_seed=42)

boards = gen.generate_random_game(max_moves=60)
print(len(boards))  # Up to 60 positions

for board in boards[:5]:
    print(board.fen())
```

#### `generate_from_opening(opening_moves: List[str], random_moves: int = 20) -> List[chess.Board]`

Generate positions starting from an opening.

```python
# Sicilian Defense
opening = ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4"]
boards = gen.generate_from_opening(opening, random_moves=20)
```

#### `generate_endgame(pieces: Dict[str, int]) -> chess.Board`

Generate random endgame position.

```python
# King + Queen vs King
board = gen.generate_endgame({
    'K': 1, 'Q': 1,  # White
    'k': 1           # Black
})
```

#### `generate_tactical(num_positions: int = 100) -> List[chess.Board]`

Generate positions with tactical themes.

```python
tactical_positions = gen.generate_tactical(num_positions=50)
# Positions with forks, pins, skewers, etc.
```

#### `generate_mixed(num_positions: int, ratios: Dict[str, float]) -> List[chess.Board]`

Generate positions using multiple methods.

```python
positions = gen.generate_mixed(
    num_positions=1000,
    ratios={
        'random': 0.4,     # 40% random games
        'opening': 0.3,    # 30% from openings
        'tactical': 0.2,   # 20% tactical
        'endgame': 0.1     # 10% endgames
    }
)
```

### Built-in Openings

```python
# Access common openings
from ochess.datagen import COMMON_OPENINGS

print(COMMON_OPENINGS.keys())
# ['sicilian', 'french', 'caro_kann', 'queens_gambit',
#  'kings_indian', 'english', 'italian', 'ruy_lopez']

print(COMMON_OPENINGS['sicilian'])
# ['e2e4', 'c7c5', 'g1f3', 'd7d6', 'd2d4', 'c5d4', 'f3d4']
```

---

## DataGenerationPipeline

Complete pipeline for generating labeled datasets.

### Class Definition

```python
class DataGenerationPipeline:
    """End-to-end data generation pipeline."""

    def __init__(
        self,
        stockfish_path: str = "/usr/local/bin/stockfish",
        analysis_depth: int = 20,
        include_bad_moves: bool = True,
        random_seed: Optional[int] = None
    ):
        ...
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `stockfish_path` | str | "/usr/local/bin/stockfish" | Path to Stockfish |
| `analysis_depth` | int | 20 | Search depth |
| `include_bad_moves` | bool | True | Generate bad move examples |
| `random_seed` | int | None | Random seed for reproducibility |

### Methods

#### `generate_dataset(num_positions: int, output_name: str, **kwargs) -> str`

Generate complete dataset.

```python
from ochess.datagen import DataGenerationPipeline

pipeline = DataGenerationPipeline(
    stockfish_path="/usr/local/bin/stockfish",
    analysis_depth=20,
    include_bad_moves=True
)

output_path = pipeline.generate_dataset(
    num_positions=10000,
    output_name="train_data",
    generation_method="mixed",
    output_dir="datasets/",
    save_format="parquet"
)

print(output_path)  # "datasets/train_data.parquet"
```

#### Parameters for `generate_dataset`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_positions` | int | required | Number of positions |
| `output_name` | str | required | Output file name |
| `generation_method` | str | "mixed" | "random", "opening", "tactical", "mixed" |
| `output_dir` | str | "datasets/" | Output directory |
| `save_format` | str | "parquet" | "parquet" or "csv" |
| `min_ply` | int | 10 | Minimum game ply to include |
| `max_ply` | int | 200 | Maximum game ply |
| `skip_boring` | bool | True | Skip drawish/one-sided positions |

#### `label_position(board: chess.Board) -> LabeledPosition`

Label a single position.

```python
import chess

board = chess.Board()
board.push_san("e4")
board.push_san("e5")

labeled = pipeline.label_position(board)

print(labeled.best_move_uci)       # Best move
print(labeled.score_cp)            # Centipawn evaluation
print(labeled.score_normalized)    # Side-to-move perspective
print(labeled.is_capture)          # Is best move a capture?
```

#### `label_positions(boards: List[chess.Board], show_progress: bool = True) -> List[LabeledPosition]`

Label multiple positions in batch.

```python
boards = gen.generate_random_game(max_moves=50)
labeled = pipeline.label_positions(boards, show_progress=True)

print(f"Labeled {len(labeled)} positions")
```

#### `generate_bad_move(board: chess.Board, good_moves: List[str]) -> Tuple[str, int]`

Generate a suboptimal move for contrastive learning.

```python
board = chess.Board()
good_moves = ["e2e4", "d2d4", "g1f3"]  # Top 3 moves

bad_move, bad_score = pipeline.generate_bad_move(board, good_moves)
print(f"Bad move: {bad_move}, score: {bad_score}")
```

#### `save_dataset(positions: List[LabeledPosition], path: str, format: str = "parquet")`

Save labeled positions to file.

```python
pipeline.save_dataset(labeled_positions, "my_data.parquet")
```

#### `load_dataset(path: str) -> pd.DataFrame`

Load previously saved dataset.

```python
df = pipeline.load_dataset("datasets/train_data.parquet")
print(df.columns)
# ['fen', 'best_move_uci', 'score_cp', 'score_normalized',
#  'white_to_move', 'is_capture', 'is_check', 'is_mate',
#  'bad_move_uci', 'bad_move_score_cp']
```

---

## Complete Example

```python
from ochess.datagen import (
    StockfishWrapper,
    PositionGenerator,
    DataGenerationPipeline
)
import chess

# Method 1: Use pipeline (recommended)
pipeline = DataGenerationPipeline(
    stockfish_path="/usr/local/bin/stockfish",
    analysis_depth=20,
    include_bad_moves=True,
    random_seed=42
)

# Generate full dataset
output_path = pipeline.generate_dataset(
    num_positions=10000,
    output_name="training_data",
    generation_method="mixed",
    output_dir="datasets/"
)

print(f"Dataset saved to: {output_path}")

# Method 2: Manual control
gen = PositionGenerator(random_seed=42)

# Generate positions
boards = gen.generate_mixed(
    num_positions=1000,
    ratios={'random': 0.5, 'opening': 0.3, 'tactical': 0.2}
)

# Label with Stockfish
with StockfishWrapper(depth=15) as sf:
    for board in boards[:5]:
        analysis = sf.analyze(board)
        print(f"FEN: {board.fen()}")
        print(f"Best move: {analysis['best_move']}")
        print(f"Score: {analysis['score_cp']} cp")
        print("---")

# Method 3: Custom pipeline
pipeline = DataGenerationPipeline(stockfish_path="/usr/local/bin/stockfish")

# Generate from specific opening
sicilian_boards = gen.generate_from_opening(
    opening_moves=["e2e4", "c7c5", "g1f3"],
    random_moves=30
)

# Label them
labeled = pipeline.label_positions(sicilian_boards)

# Filter for tactical positions
tactical = [p for p in labeled if p.is_capture or p.is_check]
print(f"Found {len(tactical)} tactical positions")

# Save
pipeline.save_dataset(tactical, "datasets/sicilian_tactical.parquet")
```

---

## Command Line Usage

```bash
# Generate 10000 positions
python scripts/generate_data.py \
    --stockfish-path /usr/local/bin/stockfish \
    --num-positions 10000 \
    --depth 20 \
    --output-name train_data \
    --method mixed

# Generate tactical positions only
python scripts/generate_data.py \
    --num-positions 5000 \
    --method tactical \
    --output-name tactical_data

# With specific random seed for reproducibility
python scripts/generate_data.py \
    --num-positions 10000 \
    --seed 42 \
    --output-name reproducible_data
```

---

## Performance Tips

### Parallel Generation

```python
# Use multiple Stockfish instances
from concurrent.futures import ProcessPoolExecutor

def analyze_batch(boards):
    with StockfishWrapper() as sf:
        return [sf.analyze(b) for b in boards]

# Split work
board_batches = [boards[i::4] for i in range(4)]

with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(analyze_batch, board_batches))
```

### Depth vs Speed Trade-off

| Depth | Quality | Speed (pos/sec) |
|-------|---------|-----------------|
| 10 | Good for training | ~50 |
| 15 | Very good | ~20 |
| 20 | Excellent | ~8 |
| 25 | Near perfect | ~3 |

Recommendation: Use depth 15-20 for training data.

### Memory Management

```python
# Process in chunks to avoid memory issues
for chunk_start in range(0, num_positions, 1000):
    chunk = generate_positions(1000)
    labeled = pipeline.label_positions(chunk)
    save_chunk(labeled, f"data_chunk_{chunk_start}.parquet")

# Merge chunks later
merge_parquet_files("data_chunk_*.parquet", "final_data.parquet")
```
