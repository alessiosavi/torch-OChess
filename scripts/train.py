#!/usr/bin/env python3
"""
Train a chess neural network.

Usage:
    python scripts/train.py --data datasets/train_data.parquet --model resnet
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from ochess.data.dataset import ChessDataset, create_dataloader, train_val_split
from ochess.model.chess_resnet import ChessResNet, ChessResNetConfig
from ochess.model.chess_transformer import ChessTransformer, ChessTransformerConfig
from ochess.model.chess_hybrid import ChessHybrid, ChessHybridConfig
from ochess.training.trainer import Trainer, TrainingConfig


def create_model(model_type: str, config_overrides: dict = None):
    """Create model of specified type."""
    overrides = config_overrides or {}

    if model_type == "resnet":
        config = ChessResNetConfig(**overrides)
        return ChessResNet(config)
    elif model_type == "transformer":
        config = ChessTransformerConfig(**overrides)
        return ChessTransformer(config)
    elif model_type == "hybrid":
        config = ChessHybridConfig(**overrides)
        return ChessHybrid(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    parser = argparse.ArgumentParser(description="Train chess neural network")

    # Data
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to training data (parquet)"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split ratio"
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=5,
        help="Number of positions per sequence"
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        choices=["resnet", "transformer", "hybrid"],
        default="resnet",
        help="Model architecture"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension size"
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=8,
        help="Number of residual/transformer blocks"
    )

    # Training
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay"
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda, cpu, mps)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Data loader workers"
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision"
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Checkpoint directory"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint"
    )

    # Logging
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = "cpu"

    # Load data
    logger.info(f"Loading data from {args.data}")
    dataset = ChessDataset(
        args.data,
        sequence_length=args.sequence_length
    )

    # Split data
    train_dataset, val_dataset = train_val_split(
        dataset,
        val_ratio=args.val_split
    )
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create data loaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # Create model
    model_config = {
        'sequence_length': args.sequence_length
    }
    if args.model == "resnet":
        model_config['num_residual_blocks'] = args.num_blocks
        model_config['hidden_dim'] =  args.hidden_dim
    elif args.model == "transformer":
        model_config['num_layers'] = args.num_blocks
        model_config['embed_dim'] = args.hidden_dim
    elif args.model == "hybrid":
        model_config['hidden_dim'] =  args.hidden_dim
        model_config['num_residual_blocks'] = args.num_blocks // 2
        model_config['num_attention_layers'] = args.num_blocks // 2

    model = create_model(args.model, model_config)
    logger.info(f"Model: {args.model} with {model.num_parameters:,} parameters")

    # Training config
    train_config = TrainingConfig(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device,
        use_amp=not args.no_amp,
        num_workers=args.num_workers,
        checkpoint_dir=args.checkpoint_dir
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        config=train_config,
        train_loader=train_loader,
        val_loader=val_loader
    )

    # Resume if specified
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    history = trainer.train()

    # Save final model
    final_path = Path(args.checkpoint_dir) / "final_model.pt"
    torch.save(model, final_path)
    logger.info(f"Saved final model to {final_path}")

    print("\nTraining complete!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")


if __name__ == "__main__":
    main()
