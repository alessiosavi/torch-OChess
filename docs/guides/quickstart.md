# Quick Start Guide

Get Torch o'Chess running in 5 minutes.

## Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Stockfish chess engine

## Installation

### 1. Clone and Install

```bash
# Clone repository
git clone https://github.com/yourrepo/torch-ochess.git
cd torch-ochess

# Install package
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

### 2. Install Stockfish

**macOS:**
```bash
brew install stockfish
# Usually installs to /usr/local/bin/stockfish
```

**Ubuntu/Debian:**
```bash
sudo apt install stockfish
# Usually installs to /usr/bin/stockfish
```

**Windows:**
Download from https://stockfishchess.org/download/

### 3. Verify Installation

```bash
# Check Stockfish
stockfish --version

# Check Python package
python -c "from ochess.model import ChessResNet; print('OK')"
```

## Quick Workflow

### Step 1: Generate Training Data

```bash
python scripts/generate_data.py \
    --stockfish-path /usr/local/bin/stockfish \
    --num-positions 10000 \
    --output-name train_data
```

This creates `datasets/train_data.parquet` with:
- 10,000 labeled chess positions
- Best moves from Stockfish
- Position evaluations
- Bad move examples for contrastive learning

**Time:** ~10-30 minutes depending on CPU

### Step 2: Train a Model

```bash
python scripts/train.py \
    --data datasets/train_data.parquet \
    --model resnet \
    --epochs 50
```

This trains a ChessResNet model and saves:
- Checkpoints in `checkpoints/`
- Best model as `checkpoints/best_model.pt`

**Time:** ~1-2 hours on GPU, longer on CPU

### Step 3: Evaluate Against Stockfish

```bash
python scripts/evaluate.py \
    --model checkpoints/best_model.pt \
    --num-games 20 \
    --levels 1,5,10
```

This plays games against Stockfish at different skill levels.

### Step 4: Play Against the Model

```bash
python scripts/play.py \
    --model checkpoints/best_model.pt \
    --color white
```

Play an interactive game against your trained model!

## Example Session

```bash
# Complete workflow
$ python scripts/generate_data.py --num-positions 5000
Generating positions... 100%|████████| 5000/5000
Saved to datasets/train_data.parquet

$ python scripts/train.py --data datasets/train_data.parquet --epochs 20
Epoch 1/20: loss=2.45, acc=12.3%
Epoch 2/20: loss=1.89, acc=23.1%
...
Epoch 20/20: loss=0.82, acc=45.2%
Best model saved to checkpoints/best_model.pt

$ python scripts/evaluate.py --model checkpoints/best_model.pt --num-games 10
Level 1: 10W 0D 0L (100%)
Level 5: 8W 1D 1L (85%)
Level 10: 4W 3D 3L (55%)

$ python scripts/play.py --model checkpoints/best_model.pt
Your move: e4
Engine plays: e5
Your move: Nf3
...
```

## Python API Quick Start

```python
from ochess.model import ChessResNet
from ochess.data import ChessDataset, create_dataloader
from ochess.training import Trainer, TrainingConfig
from ochess.play import ChessEngine
import chess

# 1. Create model
model = ChessResNet()

# 2. Load data
dataset = ChessDataset("datasets/train_data.parquet")
loader = create_dataloader(dataset, batch_size=256)

# 3. Train
trainer = Trainer(
    model=model,
    config=TrainingConfig(epochs=50),
    train_loader=loader
)
trainer.train()

# 4. Play
engine = ChessEngine(model)
board = chess.Board()
move = engine.get_move(board)
print(f"Best move: {move}")
```

## Common Issues

### Stockfish Not Found

```
Error: Stockfish not found at /usr/local/bin/stockfish
```

**Fix:** Specify correct path:
```bash
python scripts/generate_data.py --stockfish-path $(which stockfish)
```

### CUDA Not Available

```
Warning: CUDA not available, using CPU
```

**Fix:** Install PyTorch with CUDA support:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Out of Memory

```
RuntimeError: CUDA out of memory
```

**Fix:** Reduce batch size:
```bash
python scripts/train.py --batch-size 64
```

## Next Steps

- Read [Architecture Overview](../architecture.md) to understand the system
- Learn about [The Critical Fix](../critical-fix.md) for score encoding
- Explore [Model Options](../models/overview.md) for different architectures
- Check [Training Guide](training.md) for optimization tips
