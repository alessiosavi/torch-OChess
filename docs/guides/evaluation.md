# Evaluation Guide

This guide covers how to evaluate your trained model's performance.

## Quick Start

```bash
python scripts/evaluate.py \
    --model checkpoints/best_model.pt \
    --num-games 100
```

## Evaluation Methods

### 1. Playing Against Stockfish

The primary evaluation method: play games against Stockfish at various skill levels.

```bash
python scripts/evaluate.py \
    --model checkpoints/best_model.pt \
    --num-games 100 \
    --levels 1,5,10,15,20
```

**Stockfish Skill Levels:**

| Level | Approximate Rating |
|-------|-------------------|
| 0-3 | 800-1200 (Beginner) |
| 4-7 | 1200-1600 (Intermediate) |
| 8-12 | 1600-2000 (Advanced) |
| 13-17 | 2000-2400 (Expert) |
| 18-20 | 2400+ (Master) |

### 2. Move Agreement

Compare model's choices with Stockfish's best moves:

```bash
python scripts/evaluate.py \
    --model checkpoints/best_model.pt \
    --benchmark \
    --num-positions 1000
```

### 3. Centipawn Loss

Measure average evaluation loss per move:

```bash
python scripts/evaluate.py \
    --model checkpoints/best_model.pt \
    --analyze-games \
    --num-games 50
```

## Interpreting Results

### Win Rate

```
Level  5: W:18 D:2 L:0  (95%)
Level 10: W:12 D:5 L:3  (72%)
Level 15: W:6  D:8 L:6  (50%)
```

**Guidelines:**

| Win Rate | Interpretation |
|----------|----------------|
| >90% | Model is stronger than this level |
| 60-90% | Model is competitive |
| 40-60% | Model is roughly equal |
| <40% | Model is weaker |

### Agreement Rate

How often model picks same move as Stockfish:

| Rate | Interpretation |
|------|----------------|
| >70% | Excellent - near master level |
| 60-70% | Good - expert level |
| 50-60% | Average - intermediate |
| <50% | Below average |

### Centipawn Loss (CPL)

Average position evaluation loss per move:

| CPL | Rating Equivalent |
|-----|-------------------|
| <10 | Super GM (2700+) |
| 10-25 | GM (2500-2700) |
| 25-50 | IM/FM (2200-2500) |
| 50-100 | Expert (1800-2200) |
| 100-200 | Intermediate (1400-1800) |
| >200 | Beginner (<1400) |

## Detailed Evaluation

### Full Evaluation Run

```bash
python scripts/evaluate.py \
    --model checkpoints/best_model.pt \
    --num-games 100 \
    --levels 1,3,5,7,10,12,15,17,20 \
    --analyze-games \
    --save-pgn evaluation_games.pgn \
    --verbose
```

### Output Example

```
=== Evaluation Results ===

Overall: 55W 22D 23L (66% points)

By Level:
  Level  1: 10W  0D  0L (100.0%)
  Level  3:  9W  1D  0L ( 95.0%)
  Level  5:  8W  2D  0L ( 90.0%)
  Level  7:  6W  3D  1L ( 75.0%)
  Level 10:  5W  4D  1L ( 70.0%)
  Level 12:  4W  4D  2L ( 60.0%)
  Level 15:  3W  4D  3L ( 50.0%)
  Level 17:  2W  2D  6L ( 30.0%)
  Level 20:  1W  2D  7L ( 20.0%)

Analysis:
  Average CPL: 45.2
  Blunders: 23 (2.3 per game)
  Mistakes: 89 (8.9 per game)

Agreement with Stockfish (depth 20): 62.3%

Games saved to: evaluation_games.pgn
```

## Python API

```python
from ochess.evaluation import StockfishEvaluator, evaluate_model
from ochess.model import ChessResNet
import torch

# Load model
model = ChessResNet()
model.load_state_dict(torch.load("checkpoints/best_model.pt"))
model.eval()

# Quick evaluation
results = evaluate_model(
    model_path="checkpoints/best_model.pt",
    num_games=50,
    skill_levels=[1, 5, 10, 15]
)

# Detailed evaluation
evaluator = StockfishEvaluator(model)

# Play single game
game = evaluator.play_game(
    skill_level=10,
    model_color="white",
    verbose=True
)

print(f"Result: {game.outcome}")
print(f"Moves: {game.num_moves}")
print(f"Blunders: {game.blunders}")

# Benchmark agreement
benchmark = evaluator.benchmark(
    depths=[10, 15, 20],
    num_positions=500
)

for depth, rate in benchmark['agreement_rate'].items():
    print(f"Depth {depth}: {rate:.1%} agreement")
```

## Comparing Models

```bash
# Evaluate multiple models
for model in checkpoints/model_*.pt; do
    echo "Evaluating $model"
    python scripts/evaluate.py --model $model --num-games 50 --levels 10
done
```

Or in Python:

```python
models = [
    "checkpoints/resnet_8block.pt",
    "checkpoints/resnet_12block.pt",
    "checkpoints/transformer.pt"
]

for model_path in models:
    results = evaluate_model(model_path, num_games=50)
    print(f"{model_path}: {results['overall']['win_rate']:.1%}")
```

## Saving and Analyzing Games

### Save Games as PGN

```bash
python scripts/evaluate.py \
    --model checkpoints/best_model.pt \
    --num-games 20 \
    --save-pgn games.pgn
```

### View Games

```bash
# Install a PGN viewer or use online tools
cat games.pgn

# Or analyze in Python
import chess.pgn

with open("games.pgn") as f:
    while (game := chess.pgn.read_game(f)):
        print(f"Result: {game.headers['Result']}")
        print(f"Moves: {len(list(game.mainline_moves()))}")
```

## Tips

### 1. Test Against Multiple Levels

Don't just test at one level - the model might be strong at some levels but weak at others.

### 2. Play Both Colors

```bash
# Ensure model plays both white and black
python scripts/evaluate.py --model best.pt --color random
```

### 3. Use Enough Games

Statistical significance requires enough games:

| Games | Confidence |
|-------|------------|
| 10 | Low - rough estimate |
| 50 | Medium - decent estimate |
| 100 | Good - reliable |
| 500+ | High - very accurate |

### 4. Check for Weaknesses

Look at:

- Opening performance
- Endgame performance
- Tactical positions
- Quiet positions

```python
# Analyze where model fails
for game in evaluator.results['games']:
    if game.outcome == 'loss':
        print(f"Lost against level {game.stockfish_level}")
        print(f"Final position: {game.final_fen}")
        print(f"Blunders: {game.blunders}")
```
