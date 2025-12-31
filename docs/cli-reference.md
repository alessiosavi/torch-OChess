# CLI Reference

Complete reference for all command-line scripts.

## generate_data.py

Generate training data using Stockfish.

### Usage

```bash
python scripts/generate_data.py [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--stockfish-path` | str | /usr/local/bin/stockfish | Path to Stockfish |
| `--num-positions` | int | 10000 | Number of positions |
| `--depth` | int | 20 | Stockfish analysis depth |
| `--method` | str | mixed | Generation method |
| `--output-name` | str | train_data | Output file name |
| `--output-dir` | str | datasets/ | Output directory |
| `--format` | str | parquet | Output format |
| `--threads` | int | 1 | Stockfish threads |
| `--hash` | int | 128 | Stockfish hash MB |
| `--min-ply` | int | 10 | Minimum game ply |
| `--max-ply` | int | 200 | Maximum game ply |
| `--include-bad-moves` | flag | True | Include bad moves |
| `--no-bad-moves` | flag | False | Exclude bad moves |
| `--bad-move-threshold` | int | 100 | Min cp for bad move |
| `--skip-boring` | flag | True | Skip boring positions |
| `--ratios` | str | - | Method ratios (mixed) |
| `--seed` | int | None | Random seed |
| `--verbose` | flag | False | Verbose output |

### Examples

```bash
# Basic generation
python scripts/generate_data.py --num-positions 10000

# Custom Stockfish path
python scripts/generate_data.py --stockfish-path /usr/bin/stockfish

# Tactical positions only
python scripts/generate_data.py --method tactical --num-positions 5000

# High quality
python scripts/generate_data.py --depth 25 --num-positions 50000

# Mixed with custom ratios
python scripts/generate_data.py \
    --method mixed \
    --ratios "random:0.5,opening:0.3,tactical:0.2"

# Reproducible
python scripts/generate_data.py --seed 42
```

---

## train.py

Train a chess neural network.

### Usage

```bash
python scripts/train.py [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--data` | str | required | Training data path |
| `--val-data` | str | None | Validation data path |
| `--model` | str | resnet | Model type |
| `--epochs` | int | 100 | Training epochs |
| `--batch-size` | int | 256 | Batch size |
| `--lr` | float | 0.001 | Learning rate |
| `--weight-decay` | float | 0.01 | Weight decay |
| `--optimizer` | str | adamw | Optimizer |
| `--scheduler` | str | cosine | LR scheduler |
| `--warmup-steps` | int | 1000 | Warmup steps |
| `--gradient-clip` | float | 1.0 | Gradient clipping |
| `--mixed-precision` | flag | False | Use AMP |
| `--early-stopping` | flag | False | Enable early stopping |
| `--patience` | int | 10 | Early stopping patience |
| `--checkpoint-dir` | str | checkpoints | Checkpoint directory |
| `--save-every` | int | 5 | Save every N epochs |
| `--keep-best` | int | 3 | Keep best N models |
| `--tensorboard-dir` | str | None | TensorBoard log dir |
| `--num-workers` | int | 4 | Data loader workers |
| `--cache-dir` | str | None | Tensor cache dir |
| `--sequence-length` | int | 5 | Sequence length |
| `--device` | str | auto | Device |
| `--resume` | str | None | Resume from checkpoint |
| `--config` | str | None | YAML config file |
| `--verbose` | flag | False | Verbose output |
| `--log-every` | int | 100 | Log every N steps |

### Model-specific Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--num-blocks` | int | 8 | ResNet blocks |
| `--channels` | int | 256 | Hidden channels |
| `--num-layers` | int | 4 | Transformer layers |
| `--embed-dim` | int | 256 | Transformer dim |
| `--num-heads` | int | 8 | Attention heads |
| `--dropout` | float | 0.1 | Dropout rate |

### Loss Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--move-weight` | float | 3.0 | Move loss weight |
| `--score-weight` | float | 0.5 | Score loss weight |
| `--capture-weight` | float | 1.0 | Capture loss weight |
| `--outcome-weight` | float | 1.0 | Outcome loss weight |

### Examples

```bash
# Basic training
python scripts/train.py --data datasets/train.parquet --model resnet

# With validation
python scripts/train.py \
    --data datasets/train.parquet \
    --val-data datasets/val.parquet

# Production training
python scripts/train.py \
    --data datasets/train.parquet \
    --model resnet \
    --epochs 100 \
    --batch-size 256 \
    --lr 0.001 \
    --scheduler cosine \
    --warmup-steps 1000 \
    --mixed-precision \
    --early-stopping \
    --patience 15 \
    --tensorboard-dir runs

# Resume training
python scripts/train.py \
    --data datasets/train.parquet \
    --resume checkpoints/checkpoint_epoch_50.pt

# From YAML config
python scripts/train.py --config config/train.yaml
```

---

## evaluate.py

Evaluate model against Stockfish.

### Usage

```bash
python scripts/evaluate.py [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--model` | str | required | Model checkpoint path |
| `--stockfish-path` | str | /usr/local/bin/stockfish | Stockfish path |
| `--num-games` | int | 100 | Number of games |
| `--levels` | str | 1,5,10,15,20 | Stockfish skill levels |
| `--color` | str | random | Model color |
| `--max-moves` | int | 200 | Max moves per game |
| `--temperature` | float | 0.0 | Model temperature |
| `--analyze-games` | flag | False | Analyze games |
| `--analysis-depth` | int | 15 | Analysis depth |
| `--benchmark` | flag | False | Benchmark mode |
| `--num-positions` | int | 1000 | Positions for benchmark |
| `--save-pgn` | str | None | Save games to PGN |
| `--device` | str | auto | Device |
| `--verbose` | flag | False | Verbose output |

### Examples

```bash
# Basic evaluation
python scripts/evaluate.py --model checkpoints/best_model.pt

# Specific levels
python scripts/evaluate.py \
    --model checkpoints/best_model.pt \
    --levels 5,10,15

# With analysis
python scripts/evaluate.py \
    --model checkpoints/best_model.pt \
    --analyze-games \
    --save-pgn evaluation.pgn

# Benchmark mode
python scripts/evaluate.py \
    --model checkpoints/best_model.pt \
    --benchmark \
    --num-positions 500
```

---

## play.py

Play against the model.

### Usage

```bash
python scripts/play.py [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--model` | str | required | Model checkpoint path |
| `--color` | str | white | Your color |
| `--temperature` | float | 0.0 | Model temperature |
| `--show-eval` | flag | False | Show evaluation |
| `--show-analysis` | flag | False | Show top moves |
| `--unicode` | flag | True | Unicode board |
| `--auto-save` | flag | False | Auto-save games |
| `--save-dir` | str | games | Save directory |
| `--device` | str | auto | Device |

### Examples

```bash
# Play as White
python scripts/play.py --model checkpoints/best_model.pt --color white

# Play as Black with analysis
python scripts/play.py \
    --model checkpoints/best_model.pt \
    --color black \
    --show-eval \
    --show-analysis

# Easier opponent
python scripts/play.py \
    --model checkpoints/best_model.pt \
    --temperature 0.5

# Auto-save games
python scripts/play.py \
    --model checkpoints/best_model.pt \
    --auto-save \
    --save-dir my_games
```

---

## Common Options

These options are available in most scripts:

| Option | Description |
|--------|-------------|
| `--help`, `-h` | Show help message |
| `--verbose`, `-v` | Verbose output |
| `--quiet`, `-q` | Quiet mode |
| `--device` | Computing device |
| `--seed` | Random seed |
| `--config` | YAML config file |

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | File not found |
| 4 | Stockfish error |
| 5 | CUDA error |
