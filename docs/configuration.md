# Configuration Reference

Complete reference for all configuration options in Torch o'Chess.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OCHESS_STOCKFISH_PATH` | /usr/local/bin/stockfish | Stockfish binary path |
| `OCHESS_DATA_DIR` | datasets/ | Dataset directory |
| `OCHESS_CHECKPOINT_DIR` | checkpoints/ | Checkpoint directory |
| `OCHESS_CACHE_DIR` | datasets/cache/ | Tensor cache directory |
| `OCHESS_LOG_LEVEL` | INFO | Logging level |

```bash
# Set environment variables
export OCHESS_STOCKFISH_PATH=/path/to/stockfish
export OCHESS_DATA_DIR=/data/chess/
```

## Model Configurations

### ChessResNetConfig

```python
@dataclass
class ChessResNetConfig:
    # Embeddings
    num_piece_types: int = 13       # Don't change
    piece_embed_dim: int = 64       # Embedding size
    position_embed_dim: int = 64    # Position encoding size
    num_squares: int = 64           # Don't change

    # Architecture
    initial_channels: int = 128     # After embedding
    hidden_channels: int = 256      # ResNet width
    num_residual_blocks: int = 8    # ResNet depth

    # Sequence
    sequence_length: int = 5        # History length

    # Output
    num_moves: int = 4096           # Don't change

    # Board handling
    flip_board_for_black: bool = True  # Critical: keep True
```

### ChessTransformerConfig

```python
@dataclass
class ChessTransformerConfig:
    # Embeddings
    num_piece_types: int = 13
    piece_embed_dim: int = 64
    position_embed_dim: int = 64

    # Transformer
    embed_dim: int = 256            # Model dimension
    num_heads: int = 8              # Attention heads
    num_layers: int = 4             # Encoder layers
    ff_dim: int = 1024              # Feed-forward dimension
    dropout: float = 0.1            # Dropout rate

    # Sequence
    sequence_length: int = 5

    # Options
    use_cls_token: bool = True      # Use [CLS] for pooling
    flip_board_for_black: bool = True
```

### ChessHybridConfig

```python
@dataclass
class ChessHybridConfig:
    # Embeddings
    num_piece_types: int = 13
    piece_embed_dim: int = 64
    position_embed_dim: int = 64

    # CNN backbone
    initial_channels: int = 128
    hidden_channels: int = 128
    num_residual_blocks: int = 4

    # Attention
    hidden_dim: int = 256
    num_attention_layers: int = 2
    num_heads: int = 4
    ff_dim: int = 512
    dropout: float = 0.1

    # Sequence
    sequence_length: int = 5
    flip_board_for_black: bool = True
```

## Training Configuration

### TrainingConfig

```python
@dataclass
class TrainingConfig:
    # Basic
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3

    # Optimizer
    optimizer: str = "adamw"        # "adam", "adamw", "sgd"
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    momentum: float = 0.9           # For SGD

    # Learning rate schedule
    scheduler: str = "cosine"       # "cosine", "step", "plateau", "none"
    warmup_steps: int = 1000
    min_lr: float = 1e-6
    step_size: int = 30             # For step scheduler
    gamma: float = 0.1              # LR decay factor

    # Training tricks
    gradient_clip: float = 1.0      # Max gradient norm
    accumulation_steps: int = 1     # Gradient accumulation
    mixed_precision: bool = True    # Use AMP
    label_smoothing: float = 0.0    # Label smoothing

    # Data
    num_workers: int = 4            # DataLoader workers
    pin_memory: bool = True         # Pin memory for GPU

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 5             # Epochs
    keep_best: int = 3              # Keep top N

    # Early stopping
    early_stopping: bool = True
    patience: int = 10              # Epochs without improvement
    min_delta: float = 1e-4         # Minimum improvement

    # Logging
    log_every: int = 100            # Steps
    eval_every: int = 1             # Epochs

    # Device
    device: str = "auto"            # "auto", "cuda", "cpu", "mps"
```

## Loss Configuration

### ChessLossConfig

```python
@dataclass
class ChessLossConfig:
    # Task weights
    move_weight: float = 3.0        # Main task
    score_weight: float = 0.5       # Position evaluation
    capture_weight: float = 1.0     # Capture prediction
    outcome_weight: float = 1.0     # Game outcome

    # Loss types
    score_loss_type: str = "huber"  # "huber" or "mse"
    label_smoothing: float = 0.0    # For cross-entropy

    # Focal loss (for imbalanced data)
    use_focal_loss: bool = False
    focal_gamma: float = 2.0
```

## Data Generation Configuration

### DataGenerationConfig

```python
@dataclass
class DataGenerationConfig:
    # Stockfish
    stockfish_path: str = "/usr/local/bin/stockfish"
    analysis_depth: int = 20
    threads: int = 1
    hash_mb: int = 128

    # Generation
    num_positions: int = 10000
    generation_method: str = "mixed"  # "random", "opening", "tactical", "mixed"
    include_bad_moves: bool = True
    bad_move_threshold: int = 100     # Min cp loss

    # Position filtering
    min_ply: int = 10
    max_ply: int = 200
    skip_boring: bool = True          # Skip drawish positions
    skip_one_sided: bool = True       # Skip one-sided positions

    # Ratios for mixed method
    random_ratio: float = 0.4
    opening_ratio: float = 0.3
    tactical_ratio: float = 0.2
    endgame_ratio: float = 0.1

    # Output
    output_dir: str = "datasets"
    output_format: str = "parquet"    # "parquet" or "csv"

    # Reproducibility
    random_seed: Optional[int] = None
```

## Score Encoder Configuration

### ScoreEncoderConfig

```python
@dataclass
class ScoreEncoderConfig:
    normalize: bool = True          # Normalize to [-15, 15]
    scale: float = 1000.0           # Divide by this
    max_score: int = 15000          # Clip scores
    mate_score: int = 10000         # Base mate value
```

## Evaluation Configuration

### EvaluationConfig

```python
@dataclass
class EvaluationConfig:
    # Games
    num_games_per_level: int = 20
    skill_levels: List[int] = field(
        default_factory=lambda: [1, 5, 10, 15, 20]
    )

    # Game settings
    max_moves: int = 200
    time_control: Optional[str] = None  # e.g., "1+0"

    # Model settings
    temperature: float = 0.0        # 0 = deterministic
    use_legal_filtering: bool = True

    # Analysis
    analyze_games: bool = True
    analysis_depth: int = 15

    # Output
    save_pgn: bool = True
    pgn_file: str = "evaluation.pgn"
    verbose: bool = True
```

## Play Configuration

### PlayConfig

```python
@dataclass
class PlayConfig:
    # Model
    model_path: str = "checkpoints/best_model.pt"
    temperature: float = 0.0

    # Game
    human_color: str = "white"      # "white", "black", "random"

    # Display
    show_evaluation: bool = False
    show_analysis: bool = False
    unicode_board: bool = True

    # Saving
    auto_save: bool = False
    save_dir: str = "games"
```

## YAML Configuration Files

You can use YAML files for configuration:

### config/default.yaml

```yaml
# Default configuration

model:
  type: resnet
  num_residual_blocks: 8
  hidden_channels: 256
  flip_board_for_black: true

training:
  epochs: 100
  batch_size: 256
  learning_rate: 0.001
  optimizer: adamw
  weight_decay: 0.01
  scheduler: cosine
  warmup_steps: 1000
  mixed_precision: true
  early_stopping: true
  patience: 15

data:
  sequence_length: 5
  num_workers: 4
  cache_dir: datasets/cache

loss:
  move_weight: 3.0
  score_weight: 0.5
  capture_weight: 1.0
  outcome_weight: 1.0
```

### Loading YAML Config

```python
import yaml
from ochess.training import TrainingConfig

with open("config/default.yaml") as f:
    config_dict = yaml.safe_load(f)

training_config = TrainingConfig(**config_dict['training'])
```

### Command Line Override

```bash
python scripts/train.py \
    --config config/default.yaml \
    --epochs 50 \
    --lr 0.0001  # Override specific values
```

## Recommended Configurations

### Small/Fast Training

```python
model_config = ChessResNetConfig(
    num_residual_blocks=4,
    hidden_channels=128
)

training_config = TrainingConfig(
    epochs=50,
    batch_size=256,
    learning_rate=1e-3
)
```

### Large/Quality Training

```python
model_config = ChessResNetConfig(
    num_residual_blocks=12,
    hidden_channels=384
)

training_config = TrainingConfig(
    epochs=200,
    batch_size=512,
    learning_rate=5e-4,
    warmup_steps=2000,
    weight_decay=0.05
)
```

### Research/Experimental

```python
model_config = ChessTransformerConfig(
    num_layers=6,
    embed_dim=512,
    num_heads=16,
    dropout=0.2
)

training_config = TrainingConfig(
    epochs=300,
    batch_size=64,
    learning_rate=1e-4,
    gradient_clip=0.5
)
```
