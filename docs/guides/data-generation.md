# Data Generation Guide

This guide explains how to generate high-quality training data using Stockfish.

## Overview

Training a chess neural network requires labeled data:
- **Positions**: Chess board states (FEN strings)
- **Best moves**: Stockfish's recommended moves
- **Evaluations**: Position scores in centipawns
- **Bad moves**: Suboptimal moves for contrastive learning

## Quick Start

```bash
python scripts/generate_data.py \
    --stockfish-path /usr/local/bin/stockfish \
    --num-positions 10000 \
    --output-name train_data
```

## Generation Methods

### 1. Random Games (Default: 40%)

Generate positions by playing random legal moves.

**Pros:**
- Diverse positions
- Covers unusual game states
- Good for generalization

**Cons:**
- Unrealistic games
- Many boring positions

```bash
python scripts/generate_data.py --method random --num-positions 5000
```

### 2. From Openings (Default: 30%)

Start from known openings, then play randomly.

**Pros:**
- Realistic opening structures
- Common tactical themes
- Better mid-game positions

**Cons:**
- Limited opening diversity

```bash
python scripts/generate_data.py --method opening --num-positions 5000
```

**Available Openings:**
- Sicilian Defense
- French Defense
- Caro-Kann
- Queen's Gambit
- King's Indian
- English Opening
- Italian Game
- Ruy Lopez

### 3. Tactical Positions (Default: 20%)

Generate positions with tactical elements.

**Pros:**
- Rich in captures and checks
- Trains tactical awareness
- Higher quality positions

**Cons:**
- Computationally expensive
- May overfit to tactics

```bash
python scripts/generate_data.py --method tactical --num-positions 5000
```

### 4. Endgames (Default: 10%)

Generate random endgame positions.

**Pros:**
- Covers endgame theory
- Clear winning/losing positions
- Fewer pieces = faster analysis

**Cons:**
- Less complex
- Limited piece diversity

```bash
python scripts/generate_data.py --method endgame --num-positions 5000
```

### 5. Mixed (Recommended)

Combine all methods with configurable ratios.

```bash
python scripts/generate_data.py \
    --method mixed \
    --num-positions 10000 \
    --ratios "random:0.4,opening:0.3,tactical:0.2,endgame:0.1"
```

## Configuration Options

### Stockfish Settings

| Option | Default | Description |
|--------|---------|-------------|
| `--stockfish-path` | /usr/local/bin/stockfish | Path to Stockfish |
| `--depth` | 20 | Analysis depth |
| `--threads` | 1 | CPU threads for Stockfish |
| `--hash` | 128 | Hash table size (MB) |

```bash
# Faster but lower quality
python scripts/generate_data.py --depth 10 --threads 4

# Higher quality but slower
python scripts/generate_data.py --depth 25 --threads 1
```

### Position Filtering

| Option | Default | Description |
|--------|---------|-------------|
| `--min-ply` | 10 | Minimum game ply to include |
| `--max-ply` | 200 | Maximum game ply |
| `--skip-boring` | True | Skip drawish positions |
| `--skip-one-sided` | True | Skip positions where one side is winning by >10 pawns |

```bash
# Include early game positions
python scripts/generate_data.py --min-ply 1

# Include longer games
python scripts/generate_data.py --max-ply 300
```

### Bad Move Generation

Bad moves are useful for contrastive learning (teaching the network what NOT to do).

| Option | Default | Description |
|--------|---------|-------------|
| `--include-bad-moves` | True | Generate bad move examples |
| `--bad-move-threshold` | 100 | Min cp loss for "bad" move |

```bash
# Generate without bad moves (faster)
python scripts/generate_data.py --no-bad-moves

# Stricter bad move definition
python scripts/generate_data.py --bad-move-threshold 200
```

### Output Settings

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir` | datasets/ | Output directory |
| `--output-name` | train_data | Output file name |
| `--format` | parquet | Output format (parquet/csv) |

```bash
python scripts/generate_data.py \
    --output-dir my_data/ \
    --output-name custom_dataset \
    --format parquet
```

## Depth vs Quality Trade-off

| Depth | Quality | Speed (pos/sec) | Recommended For |
|-------|---------|-----------------|-----------------|
| 10 | Good | ~50 | Quick experiments |
| 15 | Very good | ~20 | Development |
| 20 | Excellent | ~8 | Production |
| 25 | Near perfect | ~3 | Research |

**Recommendation:** Use depth 20 for final training datasets.

## Dataset Size Guidelines

| Positions | Training Quality | Time (depth 20) |
|-----------|------------------|-----------------|
| 1,000 | Toy model | ~2 min |
| 10,000 | Basic model | ~20 min |
| 50,000 | Good model | ~1.5 hours |
| 100,000 | Strong model | ~3 hours |
| 500,000+ | Expert model | ~15 hours |

**Recommendation:** Start with 10,000 for development, scale to 100,000+ for production.

## Output Format

The generated parquet file contains:

| Column | Type | Description |
|--------|------|-------------|
| `fen` | string | FEN position string |
| `best_move_uci` | string | Stockfish best move |
| `score_cp` | int | Centipawns (White's view) |
| `score_normalized` | float | Normalized (side-to-move view) |
| `white_to_move` | bool | Active color |
| `is_capture` | bool | Best move is capture |
| `is_check` | bool | Best move gives check |
| `is_mate` | bool | Position is mate |
| `bad_move_uci` | string | A suboptimal move |
| `bad_move_score_cp` | int | Score after bad move |

## Python API

```python
from ochess.datagen import DataGenerationPipeline

# Create pipeline
pipeline = DataGenerationPipeline(
    stockfish_path="/usr/local/bin/stockfish",
    analysis_depth=20,
    include_bad_moves=True
)

# Generate dataset
output_path = pipeline.generate_dataset(
    num_positions=10000,
    output_name="my_dataset",
    generation_method="mixed"
)

print(f"Saved to: {output_path}")
```

### Custom Generation

```python
from ochess.datagen import (
    StockfishWrapper,
    PositionGenerator,
    DataGenerationPipeline
)
import chess

# Manual control
gen = PositionGenerator(random_seed=42)
pipeline = DataGenerationPipeline(stockfish_path="/usr/local/bin/stockfish")

# Generate positions from specific opening
sicilian = gen.generate_from_opening(
    opening_moves=["e2e4", "c7c5", "g1f3", "d7d6"],
    random_moves=30
)

# Label them
labeled = pipeline.label_positions(sicilian)

# Filter tactical positions
tactical = [p for p in labeled if p.is_capture or p.is_check]

# Save
pipeline.save_dataset(tactical, "datasets/sicilian_tactical.parquet")
```

## Parallel Generation

Speed up generation using multiple processes:

```python
from concurrent.futures import ProcessPoolExecutor

def generate_batch(seed):
    pipeline = DataGenerationPipeline(stockfish_path="/usr/local/bin/stockfish")
    return pipeline.generate_positions(1000, random_seed=seed)

# Generate in parallel
with ProcessPoolExecutor(max_workers=4) as executor:
    batches = list(executor.map(generate_batch, range(4)))

# Combine
all_positions = [p for batch in batches for p in batch]
```

## Validation

After generation, validate your dataset:

```python
import pandas as pd

df = pd.read_parquet("datasets/train_data.parquet")

print(f"Total positions: {len(df)}")
print(f"Unique positions: {df['fen'].nunique()}")
print(f"White to move: {df['white_to_move'].mean():.1%}")
print(f"Captures: {df['is_capture'].mean():.1%}")
print(f"Checks: {df['is_check'].mean():.1%}")
print(f"\nScore distribution:")
print(df['score_cp'].describe())
```

**Expected output:**
```
Total positions: 10000
Unique positions: 9987
White to move: 50.2%
Captures: 18.5%
Checks: 4.2%

Score distribution:
count    10000.00
mean        12.34
std        285.67
min      -5000.00
25%        -95.00
50%         10.00
75%        120.00
max       5000.00
```

## Tips and Best Practices

### 1. Use Consistent Random Seeds

```bash
python scripts/generate_data.py --seed 42
```

### 2. Split Train/Val/Test

```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_parquet("datasets/all_data.parquet")

# Split by game to avoid leakage
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

train_df.to_parquet("datasets/train.parquet")
val_df.to_parquet("datasets/val.parquet")
test_df.to_parquet("datasets/test.parquet")
```

### 3. Balance Your Dataset

```python
# Ensure 50/50 White/Black to move
white_df = df[df['white_to_move'] == True].sample(5000)
black_df = df[df['white_to_move'] == False].sample(5000)
balanced = pd.concat([white_df, black_df]).sample(frac=1)
```

### 4. Monitor Generation

```bash
# Watch progress
python scripts/generate_data.py --num-positions 10000 --verbose

# Check intermediate results
ls -la datasets/
```
