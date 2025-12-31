# System Architecture

This document describes the high-level architecture of Torch o'Chess.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TORCH O'CHESS SYSTEM                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐ │
│  │   DATAGEN    │───►│   TRAINING   │───►│  EVALUATION  │───►│    PLAY    │ │
│  │   Pipeline   │    │   Pipeline   │    │   Pipeline   │    │  Interface │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └────────────┘ │
│         │                   │                   │                   │        │
│         ▼                   ▼                   ▼                   ▼        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         CORE MODULES                                  │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────────┐  │   │
│  │  │ FenParser  │  │MoveEncoder │  │ScoreEncoder│  │  ChessDataset  │  │   │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         NEURAL NETWORKS                               │   │
│  │  ┌────────────┐  ┌────────────────┐  ┌─────────────┐                 │   │
│  │  │ChessResNet │  │ChessTransformer│  │ ChessHybrid │                 │   │
│  │  │ (2M params)│  │   (5M params)  │  │ (3M params) │                 │   │
│  │  └────────────┘  └────────────────┘  └─────────────┘                 │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Module Dependencies

```
ochess/
├── data/           ◄── Core data handling (no dependencies)
│   ├── fen_parser.py
│   ├── move_encoder.py
│   ├── score_encoder.py
│   └── dataset.py
│
├── datagen/        ◄── Depends on: data/
│   ├── stockfish_wrapper.py
│   ├── position_generator.py
│   └── pipeline.py
│
├── model/          ◄── Depends on: data/
│   ├── components/
│   ├── chess_resnet.py
│   ├── chess_transformer.py
│   ├── chess_hybrid.py
│   └── losses.py
│
├── training/       ◄── Depends on: data/, model/
│   ├── trainer.py
│   ├── callbacks.py
│   └── metrics.py
│
├── evaluation/     ◄── Depends on: data/, model/, datagen/
│   └── stockfish_eval.py
│
└── play/           ◄── Depends on: data/, model/
    ├── engine.py
    ├── cli_interface.py
    └── board_display.py
```

## Data Flow

### 1. Data Generation Pipeline

```
Stockfish Engine
       │
       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    Position     │────►│    Stockfish    │────►│    Labeled      │
│   Generator     │     │    Analysis     │     │    Position     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
       │                                               │
       │  Random games                                 │  FEN, best_move,
       │  Opening book                                 │  score, is_capture,
       │  Mixed strategy                               │  bad_move
       │                                               │
       └───────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │    Parquet      │
                    │    Dataset      │
                    └─────────────────┘
```

### 2. Training Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    Parquet      │────►│  ChessDataset   │────►│   DataLoader    │
│    File         │     │  (with cache)   │     │   (batched)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING LOOP                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │   Forward   │───►│  Multi-Task │───►│    Backward +       │  │
│  │    Pass     │    │    Loss     │    │    Optimizer        │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
│         │                                         │              │
│         │                                         ▼              │
│         │                              ┌─────────────────────┐  │
│         │                              │     Callbacks       │  │
│         │                              │  - Checkpointing    │  │
│         │                              │  - Early stopping   │  │
│         │                              │  - Logging          │  │
│         │                              └─────────────────────┘  │
└─────────┼───────────────────────────────────────────────────────┘
          │
          ▼
   ┌─────────────┐
   │   Metrics   │
   │  Tracking   │
   └─────────────┘
```

### 3. Inference Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Chess Board   │────►│   FEN Parser    │────►│  Board Tensor   │
│   (FEN string)  │     │                 │     │   [1, 8, 8]     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │  Neural Network │
                                               │   (inference)   │
                                               └─────────────────┘
                                                        │
                        ┌───────────────┬───────────────┼───────────────┐
                        ▼               ▼               ▼               ▼
                 ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐
                 │   Move    │   │   Score   │   │  Capture  │   │  Outcome  │
                 │  Logits   │   │   Value   │   │   Probs   │   │   Probs   │
                 │  [4096]   │   │    [1]    │   │    [2]    │   │    [3]    │
                 └───────────┘   └───────────┘   └───────────┘   └───────────┘
                        │
                        ▼
                 ┌───────────┐     ┌───────────┐
                 │   Legal   │────►│   Best    │
                 │  Masking  │     │   Move    │
                 └───────────┘     └───────────┘
```

## Board Representation

### Piece Encoding (13 values)

```
Index │ Piece │ Symbol
──────┼───────┼───────
  0   │ Empty │  .
  1   │ White Pawn   │ P
  2   │ White Knight │ N
  3   │ White Bishop │ B
  4   │ White Rook   │ R
  5   │ White Queen  │ Q
  6   │ White King   │ K
  7   │ Black Pawn   │ p
  8   │ Black Knight │ n
  9   │ Black Bishop │ b
 10   │ Black Rook   │ r
 11   │ Black Queen  │ q
 12   │ Black King   │ k
```

### Board Tensor Layout

```
Shape: [batch, sequence, 8, 8]

        a   b   c   d   e   f   g   h
      ┌───┬───┬───┬───┬───┬───┬───┬───┐
   8  │ r │ n │ b │ q │ k │ b │ n │ r │  ← Row 0 (rank 8)
      ├───┼───┼───┼───┼───┼───┼───┼───┤
   7  │ p │ p │ p │ p │ p │ p │ p │ p │  ← Row 1 (rank 7)
      ├───┼───┼───┼───┼───┼───┼───┼───┤
   6  │ . │ . │ . │ . │ . │ . │ . │ . │  ← Row 2 (rank 6)
      ├───┼───┼───┼───┼───┼───┼───┼───┤
   5  │ . │ . │ . │ . │ . │ . │ . │ . │  ← Row 3 (rank 5)
      ├───┼───┼───┼───┼───┼───┼───┼───┤
   4  │ . │ . │ . │ . │ . │ . │ . │ . │  ← Row 4 (rank 4)
      ├───┼───┼───┼───┼───┼───┼───┼───┤
   3  │ . │ . │ . │ . │ . │ . │ . │ . │  ← Row 5 (rank 3)
      ├───┼───┼───┼───┼───┼───┼───┼───┤
   2  │ P │ P │ P │ P │ P │ P │ P │ P │  ← Row 6 (rank 2)
      ├───┼───┼───┼───┼───┼───┼───┼───┤
   1  │ R │ N │ B │ Q │ K │ B │ N │ R │  ← Row 7 (rank 1)
      └───┴───┴───┴───┴───┴───┴───┴───┘
        0   1   2   3   4   5   6   7
                  Column index
```

### Move Encoding (4096 moves)

```
index = from_square * 64 + to_square

Square numbering (matches python-chess):
   a1=0,  b1=1,  ... h1=7
   a2=8,  b2=9,  ... h2=15
   ...
   a8=56, b8=57, ... h8=63

Example: e2e4
   from_square = e2 = 12
   to_square   = e4 = 28
   index = 12 * 64 + 28 = 796
```

## Multi-Task Learning

The network predicts four outputs simultaneously:

```
┌─────────────────────────────────────────────────────────────────┐
│                      MULTI-TASK OUTPUTS                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────┐                                             │
│  │  Move Logits   │  Shape: [B, 4096]                           │
│  │                │  Loss: CrossEntropy                          │
│  │                │  Weight: 3.0 (most important)               │
│  └────────────────┘                                             │
│                                                                  │
│  ┌────────────────┐                                             │
│  │  Score Value   │  Shape: [B, 1]                              │
│  │                │  Loss: HuberLoss (robust to outliers)       │
│  │                │  Weight: 0.5                                │
│  └────────────────┘                                             │
│                                                                  │
│  ┌────────────────┐                                             │
│  │  Capture Prob  │  Shape: [B, 2]                              │
│  │                │  Loss: CrossEntropy                          │
│  │                │  Weight: 1.0                                │
│  └────────────────┘                                             │
│                                                                  │
│  ┌────────────────┐                                             │
│  │  Outcome Prob  │  Shape: [B, 3] (loss/draw/win)              │
│  │                │  Loss: CrossEntropy                          │
│  │                │  Weight: 1.0                                │
│  └────────────────┘                                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Total Loss = 3.0 * move_loss + 0.5 * score_loss +
             1.0 * capture_loss + 1.0 * outcome_loss
```

## Perspective Handling

### The Critical Innovation

All positions are normalized to "current player's perspective":

```
┌─────────────────────────────────────────────────────────────────┐
│                   PERSPECTIVE NORMALIZATION                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  White to Move:                   Black to Move:                │
│  ┌─────────────────┐              ┌─────────────────┐           │
│  │ Board unchanged │              │ Board flipped   │           │
│  │ Score unchanged │              │ 180° rotation   │           │
│  │                 │              │ Colors swapped  │           │
│  │ Network sees:   │              │ Score negated   │           │
│  │ "I am White"    │              │                 │           │
│  │ +150cp = good   │              │ Network sees:   │           │
│  └─────────────────┘              │ "I am White"    │           │
│                                   │ (actually Black)│           │
│                                   │ +150cp = good   │           │
│                                   └─────────────────┘           │
│                                                                  │
│  Result: Network always plays from bottom of board,             │
│          positive scores always mean "I'm winning"              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Board Flipping Visualization

```
Original (Black to move):          After Flip (normalized):
   a  b  c  d  e  f  g  h            a  b  c  d  e  f  g  h
8  r  .  .  .  k  .  .  r         8  R  .  .  .  K  .  .  R    ← Was rank 1
7  p  p  p  .  .  p  p  p         7  P  P  P  .  .  P  P  P    ← Was rank 2
6  .  .  .  .  .  .  .  .         6  .  .  .  .  .  .  .  .
5  .  .  .  p  p  .  .  .    →    5  .  .  .  P  P  .  .  .    ← Colors swapped
4  .  .  .  P  P  .  .  .         4  .  .  .  p  p  .  .  .
3  .  .  .  .  .  .  .  .         3  .  .  .  .  .  .  .  .
2  P  P  P  .  .  P  P  P         2  p  p  p  .  .  p  p  p    ← Was rank 7
1  R  .  .  .  K  .  .  R         1  r  .  .  .  k  .  .  r    ← Was rank 8

Black pieces now at "bottom" (ranks 1-2)
Network thinks it's playing as White
```

## Caching Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                       DATASET CACHING                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  First Load:                                                    │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │   Parquet   │────►│ FEN Parsing │────►│   Tensors   │       │
│  │    File     │     │   (slow)    │     │  (cached)   │       │
│  └─────────────┘     └─────────────┘     └──────┬──────┘       │
│                                                  │              │
│                                                  ▼              │
│                                          ┌─────────────┐       │
│                                          │  .pt file   │       │
│                                          │  on disk    │       │
│                                          └─────────────┘       │
│                                                                  │
│  Subsequent Loads:                                              │
│  ┌─────────────┐     ┌─────────────┐                           │
│  │  .pt file   │────►│   Tensors   │  (instant load)           │
│  │  on disk    │     │  in memory  │                           │
│  └─────────────┘     └─────────────┘                           │
│                                                                  │
│  Benefit: 10-100x faster data loading after first run          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Performance Characteristics

| Component | Time Complexity | Memory |
|-----------|-----------------|--------|
| FEN Parsing | O(64) per position | ~256 bytes |
| Move Encoding | O(1) | 8 bytes |
| ResNet Forward | O(B × C × H × W × D) | ~50MB model |
| Transformer Forward | O(B × N² × D) | ~100MB model |
| Legal Move Masking | O(legal_moves) | ~16KB |

Where:

- B = batch size
- C = channels (256)
- H, W = height, width (8)
- D = depth of network
- N = sequence length (64 squares)
