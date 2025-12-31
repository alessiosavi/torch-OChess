# Playing Guide

This guide covers how to play against your trained model.

## Quick Start

```bash
python scripts/play.py \
    --model checkpoints/best_model.pt \
    --color white
```

## Playing Options

### Choose Your Color

```bash
# Play as White
python scripts/play.py --model best.pt --color white

# Play as Black
python scripts/play.py --model best.pt --color black

# Random color
python scripts/play.py --model best.pt --color random
```

### Adjust Difficulty

Control the model's playing style with temperature:

```bash
# Deterministic (strongest)
python scripts/play.py --model best.pt --temperature 0.0

# Slight randomness
python scripts/play.py --model best.pt --temperature 0.3

# More creative/weaker
python scripts/play.py --model best.pt --temperature 0.7
```

### Show Analysis

```bash
# Show model's evaluation
python scripts/play.py --model best.pt --show-eval

# Show top moves considered
python scripts/play.py --model best.pt --show-analysis
```

## Game Interface

### Making Moves

Enter moves in standard algebraic notation (SAN) or UCI format:

```
Your move: e4      # SAN
Your move: e2e4    # UCI
Your move: Nf3     # SAN
Your move: g1f3    # UCI
```

Special moves:
```
Your move: O-O     # Kingside castling
Your move: O-O-O   # Queenside castling
Your move: e8=Q    # Pawn promotion
```

### Commands

During the game, use these commands:

| Command | Description |
|---------|-------------|
| `quit` | Exit the game |
| `resign` | Resign the game |
| `draw` | Offer/accept draw |
| `undo` | Take back last move |
| `hint` | Get move suggestion |
| `eval` | Show position evaluation |
| `moves` | Show legal moves |
| `pgn` | Display game in PGN format |
| `fen` | Show current FEN |
| `help` | Show all commands |

### Example Session

```
$ python scripts/play.py --model best.pt --color white

=== Chess Game ===
You are playing as White
Type 'help' for commands

    a   b   c   d   e   f   g   h
  +---+---+---+---+---+---+---+---+
8 | r | n | b | q | k | b | n | r |
  +---+---+---+---+---+---+---+---+
7 | p | p | p | p | p | p | p | p |
  +---+---+---+---+---+---+---+---+
6 |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+
5 |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+
4 |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+
3 |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+
2 | P | P | P | P | P | P | P | P |
  +---+---+---+---+---+---+---+---+
1 | R | N | B | Q | K | B | N | R |
  +---+---+---+---+---+---+---+---+
    a   b   c   d   e   f   g   h

Your move: e4

Engine thinking...
Engine plays: e5

Your move: Nf3

Engine plays: Nc6

Your move: hint
Suggested: Bb5 (probability: 28%)

Your move: Bb5

Engine plays: a6

Your move: eval
Position evaluation: +0.32 (slight advantage)
Win probability: 58%

Your move: Ba4

...

Game over: White wins by checkmate!

Save game? (y/n): y
Saved to: game_2024_01_15_143022.pgn
```

## Python API

```python
from ochess.play import ChessEngine, CLIInterface
from ochess.model import ChessResNet
import torch

# Load model
model = ChessResNet()
model.load_state_dict(torch.load("checkpoints/best_model.pt"))
model.eval()

# Create engine
engine = ChessEngine(model, temperature=0.2)

# Option 1: Interactive game
cli = CLIInterface(engine, human_color="white")
result = cli.play()

# Option 2: Programmatic game
import chess

board = chess.Board()
history = []

while not board.is_game_over():
    # Get model's move
    move = engine.get_move(board, history)
    print(f"Engine: {board.san(move)}")
    board.push(move)
    history.append(board.copy())

    if board.is_game_over():
        break

    # Human move (from input or another source)
    human_move = input("Your move: ")
    board.push_san(human_move)
    history.append(board.copy())

print(f"Result: {board.result()}")
```

## Analysis Mode

Analyze positions without playing:

```python
from ochess.play import ChessEngine
import chess

engine = ChessEngine(model)

# Analyze a specific position
fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
board = chess.Board(fen)

# Get evaluation
eval_result = engine.evaluate_position(board)
print(f"Score: {eval_result['score']:.2f}")
print(f"Win probability: {eval_result['win_probability']:.1%}")

# Get top moves
top_moves = engine.get_top_moves(board, n=5)
for move, prob in top_moves:
    print(f"{board.san(move)}: {prob:.1%}")
```

## Tips

### 1. Start with Temperature 0

The strongest play is deterministic (temperature=0). Increase temperature to make the model weaker or more varied.

### 2. Use Hints Wisely

The `hint` command shows what the model would play - useful for learning!

### 3. Analyze Your Losses

When you lose, use `pgn` to save the game and analyze it later:

```bash
# After a loss
Your move: pgn
[Event "Human vs Model"]
[White "Human"]
[Black "ChessNet"]
[Result "0-1"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 ...
```

### 4. Try Different Openings

Test the model's strength in various openings:
- Open games (1. e4 e5)
- Closed games (1. d4 d5)
- Indian defenses (1. d4 Nf6)
- Flank openings (1. c4, 1. Nf3)

## Saving Games

Games can be saved automatically:

```bash
python scripts/play.py \
    --model best.pt \
    --auto-save \
    --save-dir my_games/
```

Or manually during the game:
```
Your move: pgn > my_game.pgn
Game saved to my_game.pgn
```

## Known Limitations

1. **No opening book**: The model plays from knowledge only
2. **No endgame tables**: May miss perfect endgame play
3. **No time control**: Thinks for same time regardless of position
4. **Move notation**: Some complex notations may not parse correctly

## Troubleshooting

### "Invalid move" Error

Make sure your move is:
- Legal in the current position
- Properly formatted (SAN or UCI)

Examples:
```
# Correct
e4, e2e4, Nf3, g1f3, O-O, Qxe5+

# Incorrect
E4 (use lowercase)
Ng3 (if knight can't go there)
```

### Model Plays Poorly

- Try a better-trained model
- Check if model was trained on diverse positions
- Lower temperature for stronger play

### Slow Response

- Use GPU: `--device cuda`
- Use smaller model
- Reduce sequence length in model config
