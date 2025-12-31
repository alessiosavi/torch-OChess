# Play API Reference

The `ochess.play` module provides interfaces for playing against the trained model.

## Module Overview

```python
from ochess.play import (
    ChessEngine,
    CLIInterface,
    BoardDisplay,
    play_game
)
```

---

## ChessEngine

Model wrapper for playing chess.

### Class Definition

```python
class ChessEngine:
    """Chess engine powered by neural network."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "auto",
        temperature: float = 0.0,
        use_legal_filtering: bool = True
    ):
        ...
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | nn.Module | required | Trained model |
| `device` | str | "auto" | Inference device |
| `temperature` | float | 0.0 | Sampling temperature (0 = deterministic) |
| `use_legal_filtering` | bool | True | Mask illegal moves |

### Methods

#### `get_move(board: chess.Board, history: List[chess.Board] = None) -> chess.Move`

Get best move for position.

```python
import chess
from ochess.play import ChessEngine
from ochess.model import ChessResNet

# Load model
model = ChessResNet()
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

# Create engine
engine = ChessEngine(model, temperature=0.0)

# Get move
board = chess.Board()
move = engine.get_move(board)
print(f"Best move: {move.uci()}")  # e.g., "e2e4"
```

#### `get_move_with_score(board, history) -> Tuple[chess.Move, float]`

Get move and position evaluation.

```python
move, score = engine.get_move_with_score(board)
print(f"Move: {move.uci()}, Score: {score:.2f}")
```

#### `get_top_moves(board, n: int = 5) -> List[Tuple[chess.Move, float]]`

Get top N moves with probabilities.

```python
top_moves = engine.get_top_moves(board, n=5)

for move, prob in top_moves:
    print(f"{move.uci()}: {prob:.1%}")
# e2e4: 35.2%
# d2d4: 28.1%
# g1f3: 15.3%
# c2c4: 12.8%
# e2e3: 8.6%
```

#### `evaluate_position(board) -> Dict`

Get full position evaluation.

```python
eval_result = engine.evaluate_position(board)

print(eval_result)
# {
#     'score': 0.15,           # Side-to-move perspective
#     'win_probability': 0.58, # Estimated win chance
#     'is_capture_likely': False,
#     'predicted_outcome': 'draw'  # loss/draw/win
# }
```

#### `analyze_move(board, move) -> Dict`

Analyze a specific move.

```python
import chess

board = chess.Board()
move = chess.Move.from_uci("e2e4")

analysis = engine.analyze_move(board, move)
print(analysis)
# {
#     'move': 'e2e4',
#     'probability': 0.35,
#     'rank': 1,              # Best move
#     'is_best': True
# }
```

#### `set_temperature(temperature: float)`

Adjust move randomness.

```python
engine.set_temperature(0.0)   # Always best move
engine.set_temperature(0.5)   # Some randomness
engine.set_temperature(1.0)   # More creative
```

---

## CLIInterface

Command-line interface for playing.

### Class Definition

```python
class CLIInterface:
    """Terminal interface for human vs model games."""

    def __init__(
        self,
        engine: ChessEngine,
        human_color: str = "white"
    ):
        ...
```

### Methods

#### `play() -> GameResult`

Start an interactive game.

```python
from ochess.play import ChessEngine, CLIInterface

engine = ChessEngine(model)
cli = CLIInterface(engine, human_color="white")

# Start game (blocks until game ends)
result = cli.play()

print(f"Game over: {result.outcome}")
```

#### `show_board()`

Display current board state.

```python
cli.show_board()
#     a   b   c   d   e   f   g   h
#   +---+---+---+---+---+---+---+---+
# 8 | r | n | b | q | k | b | n | r | 8
#   +---+---+---+---+---+---+---+---+
# 7 | p | p | p | p | p | p | p | p | 7
#   +---+---+---+---+---+---+---+---+
# 6 |   |   |   |   |   |   |   |   | 6
#   +---+---+---+---+---+---+---+---+
# 5 |   |   |   |   |   |   |   |   | 5
#   +---+---+---+---+---+---+---+---+
# 4 |   |   |   |   |   |   |   |   | 4
#   +---+---+---+---+---+---+---+---+
# 3 |   |   |   |   |   |   |   |   | 3
#   +---+---+---+---+---+---+---+---+
# 2 | P | P | P | P | P | P | P | P | 2
#   +---+---+---+---+---+---+---+---+
# 1 | R | N | B | Q | K | B | N | R | 1
#   +---+---+---+---+---+---+---+---+
#     a   b   c   d   e   f   g   h
```

#### `get_human_move() -> chess.Move`

Prompt human for move input.

```python
# Called internally during play()
# Handles input validation and error messages
```

### Interactive Commands

During play, these commands are available:

| Command | Description |
|---------|-------------|
| `quit` | Exit game |
| `resign` | Resign the game |
| `draw` | Offer/accept draw |
| `undo` | Take back last move |
| `hint` | Get move suggestion |
| `eval` | Show position evaluation |
| `moves` | Show legal moves |
| `pgn` | Show game PGN |
| `fen` | Show current FEN |

---

## BoardDisplay

Utilities for displaying boards.

### Class Definition

```python
class BoardDisplay:
    """Board visualization utilities."""

    @staticmethod
    def to_ascii(board: chess.Board) -> str:
        """Convert board to ASCII representation."""

    @staticmethod
    def to_unicode(board: chess.Board) -> str:
        """Convert board to Unicode representation."""

    @staticmethod
    def highlight_squares(board, squares: List[int]) -> str:
        """Display board with highlighted squares."""
```

### Methods

#### `to_ascii(board) -> str`

ASCII board representation.

```python
from ochess.play import BoardDisplay
import chess

board = chess.Board()
print(BoardDisplay.to_ascii(board))
```

#### `to_unicode(board) -> str`

Unicode board with chess symbols.

```python
print(BoardDisplay.to_unicode(board))
#   a b c d e f g h
# 8 ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜
# 7 ♟ ♟ ♟ ♟ ♟ ♟ ♟ ♟
# 6 · · · · · · · ·
# 5 · · · · · · · ·
# 4 · · · · · · · ·
# 3 · · · · · · · ·
# 2 ♙ ♙ ♙ ♙ ♙ ♙ ♙ ♙
# 1 ♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖
```

#### `highlight_squares(board, squares) -> str`

Highlight specific squares.

```python
# Highlight e4 and d5
squares = [chess.E4, chess.D5]
print(BoardDisplay.highlight_squares(board, squares))
# Squares marked with [*]
```

#### `show_attacks(board, square) -> str`

Show squares attacked from a position.

```python
# Show knight attacks from g1
attacks = BoardDisplay.show_attacks(board, chess.G1)
print(attacks)
```

---

## play_game Function

Convenience function to start a game.

```python
from ochess.play import play_game

# Quick start
result = play_game(
    model_path="checkpoints/best_model.pt",
    human_color="white"
)

# With options
result = play_game(
    model_path="checkpoints/best_model.pt",
    human_color="black",
    temperature=0.3,    # Add some randomness
    show_analysis=True  # Show engine evaluation
)
```

---

## Complete Example

```python
import torch
import chess
from ochess.model import ChessResNet
from ochess.play import ChessEngine, CLIInterface, BoardDisplay

# 1. Load model
model = ChessResNet()
model.load_state_dict(torch.load("checkpoints/best_model.pt"))
model.eval()

# 2. Create engine
engine = ChessEngine(
    model,
    temperature=0.2,  # Slight randomness
    use_legal_filtering=True
)

# 3. Play interactively
print("Starting game...")
cli = CLIInterface(engine, human_color="white")
result = cli.play()

print(f"\nGame over: {result.outcome}")
print(f"Moves played: {result.num_moves}")
print(f"\nPGN:\n{result.pgn}")

# Or: Manual game control
board = chess.Board()

print("\n=== Manual Game ===")
while not board.is_game_over():
    print(BoardDisplay.to_unicode(board))

    if board.turn == chess.WHITE:
        # Human move
        move_str = input("Your move: ")
        try:
            move = board.parse_san(move_str)
            board.push(move)
        except ValueError:
            print("Invalid move!")
            continue
    else:
        # Engine move
        move = engine.get_move(board)
        print(f"Engine plays: {board.san(move)}")
        board.push(move)

print("\nFinal position:")
print(BoardDisplay.to_unicode(board))
print(f"Result: {board.result()}")
```

---

## Command Line Play

```bash
# Play as white
python scripts/play.py \
    --model checkpoints/best_model.pt \
    --color white

# Play as black
python scripts/play.py \
    --model checkpoints/best_model.pt \
    --color black

# Random color
python scripts/play.py \
    --model checkpoints/best_model.pt \
    --color random

# With analysis
python scripts/play.py \
    --model checkpoints/best_model.pt \
    --color white \
    --show-eval

# Adjust difficulty (temperature)
python scripts/play.py \
    --model checkpoints/best_model.pt \
    --temperature 0.5  # More random/easier
```

---

## Game Session Example

```
$ python scripts/play.py --model best_model.pt --color white

=== Chess Game ===
You are playing as White
Type 'help' for commands

    a   b   c   d   e   f   g   h
  +---+---+---+---+---+---+---+---+
8 | r | n | b | q | k | b | n | r | 8
  +---+---+---+---+---+---+---+---+
7 | p | p | p | p | p | p | p | p | 7
  +---+---+---+---+---+---+---+---+
...

Your move: e4

Engine thinking...
Engine plays: e5

    a   b   c   d   e   f   g   h
  +---+---+---+---+---+---+---+---+
8 | r | n | b | q | k | b | n | r | 8
  +---+---+---+---+---+---+---+---+
7 | p | p | p | p |   | p | p | p | 7
  +---+---+---+---+---+---+---+---+
6 |   |   |   |   |   |   |   |   | 6
  +---+---+---+---+---+---+---+---+
5 |   |   |   |   | p |   |   |   | 5
  +---+---+---+---+---+---+---+---+
4 |   |   |   |   | P |   |   |   | 4
  +---+---+---+---+---+---+---+---+
...

Your move: hint
Suggested: Nf3 (probability: 32%)

Your move: Nf3

Engine plays: Nc6

...

Your move: eval
Position evaluation: +0.25 (slight advantage)
Win probability: 58%

Your move: quit

Game ended by resignation.
Final position saved to: game_2024_01_15.pgn
```
