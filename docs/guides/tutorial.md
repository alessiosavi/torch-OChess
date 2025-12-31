# Tutorial: Building a Chess AI from Scratch

This tutorial walks you through the complete process of building a chess AI using Torch o'Chess.

## What You'll Build

By the end of this tutorial, you'll have:
1. Generated a training dataset using Stockfish
2. Trained a neural network to play chess
3. Evaluated your model against Stockfish
4. Played games against your AI

## Prerequisites

- Python 3.8+
- PyTorch installed
- Stockfish installed
- ~30 minutes

## Step 1: Generate Training Data

First, we'll create training data by having Stockfish analyze chess positions.

### Command Line

```bash
python scripts/generate_data.py \
    --stockfish-path /usr/local/bin/stockfish \
    --num-positions 10000 \
    --depth 15 \
    --output-name tutorial_data \
    --verbose
```

### Python API

```python
from ochess.datagen import DataGenerationPipeline

# Create pipeline
pipeline = DataGenerationPipeline(
    stockfish_path="/usr/local/bin/stockfish",
    analysis_depth=15,
    include_bad_moves=True
)

# Generate data
output_path = pipeline.generate_dataset(
    num_positions=10000,
    output_name="tutorial_data",
    generation_method="mixed"
)

print(f"Data saved to: {output_path}")
```

### Understanding the Data

Let's examine what we generated:

```python
import pandas as pd

df = pd.read_parquet("datasets/tutorial_data.parquet")
print(f"Total positions: {len(df)}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nSample row:")
print(df.iloc[0])
```

Output:
```
Total positions: 10000

Columns: ['fen', 'best_move_uci', 'score_cp', 'score_normalized',
          'white_to_move', 'is_capture', 'is_check', 'bad_move_uci']

Sample row:
fen                    rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b...
best_move_uci          e7e5
score_cp               25
score_normalized       -0.025
white_to_move          False
is_capture             False
is_check               False
bad_move_uci           a7a6
```

**Key insight:** Notice `score_normalized` is negative because it's Black's turn and White is slightly ahead. This is the critical fix!

## Step 2: Understand the Score Encoding

This is the most important concept:

```python
from ochess.data import ScoreEncoder

encoder = ScoreEncoder()

# Position: White is better by 150 centipawns
# White's turn: "I'm winning" = positive
white_score = encoder.encode("+150", white_to_move=True)
print(f"White's view: {white_score}")  # 0.15

# Black's turn: "I'm losing" = negative
black_score = encoder.encode("+150", white_to_move=False)
print(f"Black's view: {black_score}")  # -0.15

# The old broken code would return 0.15 for both!
# This is why the fix is critical.
```

## Step 3: Create the Model

Let's create a neural network:

```python
from ochess.model import ChessResNet, ChessResNetConfig

# Configure model
config = ChessResNetConfig(
    num_residual_blocks=6,    # Moderate depth
    hidden_channels=192,      # Moderate width
    sequence_length=5         # Use 5 positions of history
)

# Create model
model = ChessResNet(config)

# Count parameters
params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {params:,}")  # ~1.5M
```

### Understanding the Model

```python
import torch

# Sample input
boards = torch.randint(0, 13, (2, 5, 8, 8))  # [batch, seq, 8, 8]
colors = torch.randint(0, 2, (2, 5))          # [batch, seq]

# Forward pass
outputs = model(boards, colors)

print("Output shapes:")
for key, value in outputs.items():
    print(f"  {key}: {value.shape}")
```

Output:
```
Output shapes:
  move_logits: torch.Size([2, 4096])
  score: torch.Size([2, 1])
  capture: torch.Size([2, 2])
  outcome: torch.Size([2, 3])
```

## Step 4: Train the Model

### Command Line

```bash
python scripts/train.py \
    --data datasets/tutorial_data.parquet \
    --model resnet \
    --num-blocks 6 \
    --channels 192 \
    --epochs 30 \
    --batch-size 128 \
    --lr 0.001 \
    --checkpoint-dir checkpoints/tutorial \
    --verbose
```

### Python API

```python
from ochess.data import ChessDataset, create_dataloader
from ochess.training import Trainer, TrainingConfig

# Load data
dataset = ChessDataset(
    "datasets/tutorial_data.parquet",
    sequence_length=5
)

# Split into train/val
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

train_loader = create_dataloader(train_dataset, batch_size=128)
val_loader = create_dataloader(val_dataset, batch_size=128, shuffle=False)

# Configure training
training_config = TrainingConfig(
    epochs=30,
    learning_rate=1e-3,
    scheduler="cosine",
    warmup_steps=500,
    early_stopping=True,
    patience=5
)

# Train
trainer = Trainer(
    model=model,
    config=training_config,
    train_loader=train_loader,
    val_loader=val_loader
)

history = trainer.train()
```

### Monitor Progress

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(history['train_move_accuracy'], label='Train')
plt.plot(history['val_move_accuracy'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Move Accuracy')

plt.tight_layout()
plt.savefig('training_progress.png')
plt.show()
```

## Step 5: Evaluate the Model

### Against Stockfish

```bash
python scripts/evaluate.py \
    --model checkpoints/tutorial/best_model.pt \
    --num-games 20 \
    --levels 1,5,10 \
    --verbose
```

### Python API

```python
from ochess.evaluation import StockfishEvaluator

# Load trained model
model.load_state_dict(torch.load("checkpoints/tutorial/best_model.pt"))
model.eval()

# Create evaluator
evaluator = StockfishEvaluator(model)

# Play games
results = evaluator.evaluate(
    num_games=20,
    skill_levels=[1, 5, 10]
)

print("\n=== Results ===")
for level, stats in results['by_level'].items():
    win_rate = (stats['wins'] + 0.5 * stats['draws']) / \
               (stats['wins'] + stats['draws'] + stats['losses'])
    print(f"Level {level}: {win_rate:.1%}")
```

### Check Move Agreement

```python
benchmark = evaluator.benchmark(depths=[10, 15], num_positions=500)

print("\nAgreement with Stockfish:")
for depth, rate in benchmark['agreement_rate'].items():
    print(f"  Depth {depth}: {rate:.1%}")
```

## Step 6: Play Against Your AI

### Command Line

```bash
python scripts/play.py \
    --model checkpoints/tutorial/best_model.pt \
    --color white \
    --show-eval
```

### Python API

```python
from ochess.play import ChessEngine, CLIInterface

# Create engine
engine = ChessEngine(model, temperature=0.2)

# Play interactively
cli = CLIInterface(engine, human_color="white")
result = cli.play()

print(f"\nGame result: {result.outcome}")
print(f"Moves: {result.num_moves}")
```

### Programmatic Play

```python
import chess

board = chess.Board()
history = []

# Play 10 moves
for _ in range(10):
    if board.is_game_over():
        break

    move = engine.get_move(board, history)
    print(f"Move: {board.san(move)}")

    board.push(move)
    history.append(board.copy())

print(f"\nPosition after 10 moves:")
print(board)
```

## Step 7: Improve Your Model

### More Data

```bash
# Generate more training data
python scripts/generate_data.py \
    --num-positions 50000 \
    --depth 20 \
    --output-name large_data
```

### Longer Training

```bash
python scripts/train.py \
    --data datasets/large_data.parquet \
    --model resnet \
    --epochs 100 \
    --mixed-precision \
    --early-stopping
```

### Larger Model

```bash
python scripts/train.py \
    --data datasets/large_data.parquet \
    --model resnet \
    --num-blocks 12 \
    --channels 256 \
    --epochs 100
```

## Understanding What the Model Learned

### Visualize Attention (Transformer)

```python
from ochess.model import ChessTransformer

transformer = ChessTransformer()
# ... train ...

outputs = transformer(boards, colors, return_attention=True)
attention = outputs['attention'][0]  # First layer

import matplotlib.pyplot as plt

# Attention from e4 (square 28)
attn_map = attention[0, 0, 28, :64].reshape(8, 8)
plt.imshow(attn_map.detach(), cmap='hot')
plt.title("What e4 attends to")
plt.colorbar()
plt.show()
```

### Analyze Model Decisions

```python
# Get top moves with probabilities
board = chess.Board()
top_moves = engine.get_top_moves(board, n=5)

print("Model's top moves:")
for move, prob in top_moves:
    print(f"  {board.san(move)}: {prob:.1%}")
```

## Summary

You've learned how to:

1. **Generate data** using Stockfish with proper score encoding
2. **Understand the critical fix** for side-to-move perspective
3. **Create and configure** neural network models
4. **Train models** with proper loss functions
5. **Evaluate** against Stockfish at various levels
6. **Play games** against your trained AI

## Next Steps

- Try different architectures (Transformer, Hybrid)
- Experiment with hyperparameters
- Generate more diverse training data
- Add opening book knowledge
- Implement time controls

Happy chess AI building!
