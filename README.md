# Torch o'Chess

### *A Baseline Chess Engine for Experimenting with Neural Networks*

---

## Overview

Torch o'Chess is an open-source chess engine built in PyTorch, designed for experimenting with neural networks in chess.
It's not a complete product but serves as a baseline for future research and improvements.
The core implementation resides in a single notebook, making it easy to follow, modify, and extend.

## Key Features

- **Custom Dataset**: Easily load your own chess datasets for training and evaluation trough FEN notation.
- **Game Turns Batch Creation**: Generate batches of game turns, utilizing sequences of N previous moves.
- **Duplicate Filtering**: Filter out duplicate batches to maintain clean training data.
- **FEN Parsing**: Convert FEN notation to tensors (and back).
- **Debugging & Utilities**: Utilities for printing and debugging chessboards, matrices, and tensors.
- **Conv3D Baseline Engine**: A foundational 3D convolutional neural network architecture tailored for chess move prediction.
- **Training Loop**: Simple but functional training loop, that leverage 2 different loss on 4 different output.
- **Accuracy Reporting**: Basic accuracy tracking (not deeply optimized, but functional for experimentation).
- **Turn Filtering**: Methods to filter turns where at least one prediction differs from expected (used for debug common error type of the network).
- **Stockfish Integration**: Debug and analyze network performance by playing against Stockfish.

## Motivation

Torch o'Chess is a personal project intended as a starting point for exploring how neural networks can be applied to chess.
The goal of this initial release is to track experiment, allowing others to explore different architectures and strategies.
This is not a polished product but a flexible foundation to grow and improve.

## Installation

To use Torch o'Chess, simply clone the repository and install the required dependencies. The project is organized as a single Jupyter notebook, which you can run locally.

```bash
git clone https://github.com/alessiosavi/torch-oChess.git
cd torchoChess
pip install -r requirements.txt
```

## How to Use

- Load your dataset using the provided dataset loader functions.
- Customize the network architecture or modify the training loop as needed.
- Train the model on different dataset, and evaluate its performance.
- Use Stockfish to test and debug your model’s predictions.

## Future Plans

This project is just the beginning, and there are several areas for future development:
- Experimenting with different network architectures.
- Adding advanced evaluation metrics for chess move prediction.
  - [ ] Invalid move
  - [ ] Missed move
  - [ ] No-sense move
- Standardize experiment run
- Standardize dataset generation

## Contributing

Contributions are welcome! Feel free to fork the repository, submit pull requests, or open issues if you encounter any bugs or have suggestions for improvement.

## License

Torch o'Chess is licensed under the MIT License.