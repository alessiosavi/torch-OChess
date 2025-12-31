# Evaluation API Reference

The `ochess.evaluation` module provides tools for evaluating model performance against Stockfish.

## Module Overview

```python
from ochess.evaluation import (
    StockfishEvaluator,
    EvaluationConfig,
    GameResult,
    evaluate_model
)
```

---

## StockfishEvaluator

Evaluate model by playing games against Stockfish.

### Class Definition

```python
class StockfishEvaluator:
    """Evaluate model against Stockfish."""

    def __init__(
        self,
        model: nn.Module,
        stockfish_path: str = "/usr/local/bin/stockfish",
        device: str = "auto"
    ):
        ...
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | nn.Module | required | Trained model |
| `stockfish_path` | str | "/usr/local/bin/stockfish" | Path to Stockfish |
| `device` | str | "auto" | Device for inference |

### Methods

#### `evaluate(num_games: int, skill_levels: List[int], ...) -> Dict`

Run evaluation games.

```python
from ochess.evaluation import StockfishEvaluator
from ochess.model import ChessResNet

# Load trained model
model = ChessResNet()
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

# Create evaluator
evaluator = StockfishEvaluator(
    model=model,
    stockfish_path="/usr/local/bin/stockfish"
)

# Run evaluation
results = evaluator.evaluate(
    num_games=100,
    skill_levels=[1, 5, 10, 15, 20],
    time_control=None,  # Unlimited time
    show_progress=True
)

print(results)
# {
#     'overall': {'wins': 45, 'draws': 20, 'losses': 35, 'win_rate': 0.55},
#     'by_level': {
#         1: {'wins': 10, 'draws': 0, 'losses': 0, 'win_rate': 1.0},
#         5: {'wins': 9, 'draws': 1, 'losses': 0, 'win_rate': 0.95},
#         10: {'wins': 8, 'draws': 5, 'losses': 7, 'win_rate': 0.525},
#         15: {'wins': 5, 'draws': 8, 'losses': 7, 'win_rate': 0.45},
#         20: {'wins': 3, 'draws': 6, 'losses': 11, 'win_rate': 0.30}
#     },
#     'games': [...]  # List of GameResult objects
# }
```

#### `play_game(skill_level: int, model_color: str, ...) -> GameResult`

Play a single game.

```python
result = evaluator.play_game(
    skill_level=10,
    model_color="white",  # or "black" or "random"
    max_moves=200,
    verbose=True
)

print(result)
# GameResult(
#     outcome='win',        # 'win', 'loss', 'draw'
#     model_color='white',
#     num_moves=67,
#     pgn='1. e4 e5 2. Nf3 ...',
#     final_fen='...',
#     stockfish_level=10
# )
```

#### `benchmark(depths: List[int], num_positions: int) -> Dict`

Benchmark model against Stockfish at various depths.

```python
benchmark = evaluator.benchmark(
    depths=[5, 10, 15, 20],
    num_positions=1000
)

print(benchmark)
# {
#     'agreement_rate': {5: 0.85, 10: 0.72, 15: 0.65, 20: 0.58},
#     'avg_score_diff': {5: 15, 10: 35, 15: 55, 20: 80}
# }
```

---

## GameResult

Data class for game results.

### Definition

```python
@dataclass
class GameResult:
    """Result of a single game."""

    outcome: str            # 'win', 'loss', 'draw'
    model_color: str        # 'white' or 'black'
    num_moves: int          # Number of moves played
    pgn: str                # PGN notation
    final_fen: str          # Final position
    stockfish_level: int    # Stockfish skill level
    termination: str        # 'checkmate', 'stalemate', 'repetition', etc.

    # Analysis
    blunders: int = 0       # Model blunders (>300cp loss)
    mistakes: int = 0       # Model mistakes (>100cp loss)
    inaccuracies: int = 0   # Model inaccuracies (>50cp loss)
    avg_centipawn_loss: float = 0.0  # Average cp loss per move
```

### Methods

```python
result = evaluator.play_game(skill_level=10, model_color="white")

# Check outcome
if result.is_win():
    print("Model won!")
elif result.is_draw():
    print("Game was drawn")
else:
    print("Model lost")

# Get PGN
print(result.pgn)
# [Event "Model vs Stockfish"]
# [White "ChessNet"]
# [Black "Stockfish Level 10"]
# [Result "1-0"]
# 1. e4 e5 2. Nf3 Nc6 ...

# Analyze quality
print(f"Blunders: {result.blunders}")
print(f"Avg CP loss: {result.avg_centipawn_loss:.1f}")
```

---

## EvaluationConfig

Configuration for evaluation.

```python
@dataclass
class EvaluationConfig:
    """Evaluation configuration."""

    # Games
    num_games_per_level: int = 20
    skill_levels: List[int] = field(default_factory=lambda: [1, 5, 10, 15, 20])

    # Game settings
    max_moves: int = 200
    time_control: Optional[str] = None  # e.g., "1+0" for 1 minute

    # Model settings
    temperature: float = 0.0           # 0 = deterministic
    use_legal_filtering: bool = True

    # Analysis
    analyze_games: bool = True         # Run post-game analysis
    stockfish_analysis_depth: int = 15

    # Output
    save_pgn: bool = True
    pgn_file: str = "evaluation_games.pgn"
    verbose: bool = True
```

---

## evaluate_model Function

Convenience function for quick evaluation.

```python
from ochess.evaluation import evaluate_model

results = evaluate_model(
    model_path="checkpoints/best_model.pt",
    num_games=100,
    skill_levels=[1, 5, 10, 15, 20],
    stockfish_path="/usr/local/bin/stockfish"
)

# Print summary
print("\n=== Evaluation Results ===")
print(f"Overall win rate: {results['overall']['win_rate']:.1%}")
print("\nBy Stockfish level:")
for level, stats in results['by_level'].items():
    print(f"  Level {level:2d}: {stats['win_rate']:.1%} "
          f"(W:{stats['wins']} D:{stats['draws']} L:{stats['losses']})")
```

---

## Complete Example

```python
import torch
from ochess.model import ChessResNet, ChessResNetConfig
from ochess.evaluation import (
    StockfishEvaluator,
    EvaluationConfig,
    evaluate_model
)

# 1. Load model
config = ChessResNetConfig()
model = ChessResNet(config)
model.load_state_dict(torch.load("checkpoints/best_model.pt"))
model.eval()

# 2. Quick evaluation
print("=== Quick Evaluation ===")
results = evaluate_model(
    model_path="checkpoints/best_model.pt",
    num_games=50,
    skill_levels=[1, 5, 10]
)

# 3. Detailed evaluation
print("\n=== Detailed Evaluation ===")
evaluator = StockfishEvaluator(model, stockfish_path="/usr/local/bin/stockfish")

# Play games at each level
for level in [1, 5, 10, 15, 20]:
    print(f"\nLevel {level}:")

    wins, draws, losses = 0, 0, 0
    for _ in range(10):
        result = evaluator.play_game(skill_level=level, model_color="random")
        if result.is_win():
            wins += 1
        elif result.is_draw():
            draws += 1
        else:
            losses += 1

    print(f"  W: {wins}, D: {draws}, L: {losses}")
    print(f"  Win rate: {(wins + draws * 0.5) / 10:.1%}")

# 4. Benchmark move accuracy
print("\n=== Benchmark vs Stockfish Depths ===")
benchmark = evaluator.benchmark(depths=[10, 15, 20], num_positions=500)

for depth, agreement in benchmark['agreement_rate'].items():
    print(f"  Depth {depth}: {agreement:.1%} agreement")

# 5. Save games
print("\n=== Sample Games ===")
for i in range(3):
    result = evaluator.play_game(skill_level=10, model_color="white", verbose=False)
    print(f"\nGame {i+1}: {result.outcome} in {result.num_moves} moves")
    print(f"Blunders: {result.blunders}, Mistakes: {result.mistakes}")

    # Save PGN
    with open(f"game_{i+1}.pgn", "w") as f:
        f.write(result.pgn)
```

---

## Command Line Evaluation

```bash
# Basic evaluation
python scripts/evaluate.py \
    --model checkpoints/best_model.pt \
    --num-games 100

# Specific levels
python scripts/evaluate.py \
    --model checkpoints/best_model.pt \
    --num-games 50 \
    --levels 1,5,10,15,20

# Save games
python scripts/evaluate.py \
    --model checkpoints/best_model.pt \
    --num-games 100 \
    --save-pgn evaluation_games.pgn

# Verbose output
python scripts/evaluate.py \
    --model checkpoints/best_model.pt \
    --num-games 10 \
    --verbose
```

---

## Interpreting Results

### Win Rate by Level

| Level | Expected | Meaning |
|-------|----------|---------|
| 1-3 | >90% | Model beats beginners |
| 5-7 | >70% | Model is intermediate |
| 10-12 | >50% | Model is advanced |
| 15-17 | >30% | Model is expert |
| 20 | >10% | Model approaches Stockfish |

### Agreement Rate

How often model's move matches Stockfish's best move:

| Rate | Quality |
|------|---------|
| >80% | Excellent |
| 70-80% | Very good |
| 60-70% | Good |
| 50-60% | Average |
| <50% | Needs improvement |

### Centipawn Loss

Average evaluation loss per move:

| CPL | Quality |
|-----|---------|
| <20 | GM level |
| 20-40 | Expert |
| 40-70 | Intermediate |
| 70-100 | Beginner |
| >100 | Needs improvement |
