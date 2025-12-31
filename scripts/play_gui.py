#!/usr/bin/env python3
"""
Play against a trained chess model using a graphical interface.

Usage:
    python scripts/play_gui.py --model checkpoints/best_model.pt --model-type hybrid --color white
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from ochess.play.gui_interface import GUIInterface
from ochess.model.chess_resnet import ChessResNet, ChessResNetConfig
from ochess.model.chess_transformer import ChessTransformer, ChessTransformerConfig
from ochess.model.chess_hybrid import ChessHybrid, ChessHybridConfig


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


def infer_model_config_from_state_dict(state_dict: dict, model_type: str) -> dict:
    """
    Infer model configuration from state dict keys and tensor shapes.

    Args:
        state_dict: Model state dictionary
        model_type: Type of model (resnet, transformer, hybrid)

    Returns:
        Dictionary of config overrides
    """
    config = {}

    if model_type == "resnet":
        # Count residual blocks
        block_indices = set()
        for key in state_dict.keys():
            if 'residual_tower.blocks.' in key:
                parts = key.split('.')
                for i, p in enumerate(parts):
                    if p == 'blocks' and i + 1 < len(parts):
                        try:
                            block_indices.add(int(parts[i + 1]))
                        except ValueError:
                            pass
        if block_indices:
            config['num_residual_blocks'] = max(block_indices) + 1

        # Infer hidden_dim from conv weights
        for key, tensor in state_dict.items():
            if 'residual_tower.blocks.0.conv1.weight' in key:
                config['hidden_dim'] = tensor.shape[0]
                break

    elif model_type == "transformer":
        # Count transformer layers
        layer_indices = set()
        for key in state_dict.keys():
            if 'transformer.layers.' in key:
                parts = key.split('.')
                for i, p in enumerate(parts):
                    if p == 'layers' and i + 1 < len(parts):
                        try:
                            layer_indices.add(int(parts[i + 1]))
                        except ValueError:
                            pass
        if layer_indices:
            config['num_layers'] = max(layer_indices) + 1

        # Infer embed_dim
        for key, tensor in state_dict.items():
            if 'cls_token' in key:
                config['embed_dim'] = tensor.shape[-1]
                break

    elif model_type == "hybrid":
        # Count residual blocks
        block_indices = set()
        for key in state_dict.keys():
            if 'residual_tower.blocks.' in key:
                parts = key.split('.')
                for i, p in enumerate(parts):
                    if p == 'blocks' and i + 1 < len(parts):
                        try:
                            block_indices.add(int(parts[i + 1]))
                        except ValueError:
                            pass
        if block_indices:
            config['num_residual_blocks'] = max(block_indices) + 1

        # Count attention layers
        attn_indices = set()
        for key in state_dict.keys():
            if 'attention_layers.' in key:
                parts = key.split('.')
                for i, p in enumerate(parts):
                    if p == 'attention_layers' and i + 1 < len(parts):
                        try:
                            attn_indices.add(int(parts[i + 1]))
                        except ValueError:
                            pass
        if attn_indices:
            config['num_attention_layers'] = max(attn_indices) + 1

        # Infer hidden_dim from conv weights
        for key, tensor in state_dict.items():
            if 'residual_tower.blocks.0.conv1.weight' in key:
                config['hidden_dim'] = tensor.shape[0]
                break

    return config


def load_model(model_path: str, model_type: str, device: str):
    """
    Load a trained model from checkpoint.

    Supports two formats:
    1. Checkpoint dict with 'model_state_dict' (from Trainer)
    2. Full model object (from torch.save(model, path))

    Args:
        model_path: Path to the model checkpoint
        model_type: Type of model architecture (resnet, transformer, hybrid)
        device: Device to load the model on

    Returns:
        Loaded model in eval mode
    """
    print(f"Loading {model_type} model from {model_path}...")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Check if it's a checkpoint dict or a full model
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # It's a training checkpoint - need to create model and load state dict
        print("Detected checkpoint format (state_dict)")

        state_dict = checkpoint['model_state_dict']

        # Infer model architecture from state dict
        config_overrides = infer_model_config_from_state_dict(state_dict, model_type)
        print(f"  Inferred config: {config_overrides}")

        # Create model with inferred config
        model = create_model(model_type, config_overrides)

        # Load state dict
        model.load_state_dict(state_dict)

        # Print some info
        if 'epoch' in checkpoint:
            print(f"  Loaded from epoch {checkpoint['epoch'] + 1}")
        if 'best_val_loss' in checkpoint:
            print(f"  Best validation loss: {checkpoint['best_val_loss']:.4f}")

    elif isinstance(checkpoint, dict) and not 'model_state_dict' in checkpoint:
        # It might be just a state_dict without wrapper
        print("Detected raw state_dict format")
        model = create_model(model_type)
        model.load_state_dict(checkpoint)

    else:
        # It's a full model object
        print("Detected full model format")
        model = checkpoint

    model = model.to(device)
    model.eval()

    print(f"  Model parameters: {model.num_parameters:,}")
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Play chess against trained model (GUI)"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["resnet", "transformer", "hybrid"],
        required=True,
        help="Model architecture type (resnet, transformer, hybrid)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference (cuda, cpu, mps)"
    )
    parser.add_argument(
        "--color",
        type=str,
        choices=["white", "black"],
        default="white",
        help="Your color (white or black)"
    )
    parser.add_argument(
        "--square-size",
        type=int,
        default=80,
        help="Size of each square in pixels (default: 80)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    elif args.device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        print("MPS not available, using CPU")
        args.device = "cpu"

    # Load model
    model = load_model(args.model, args.model_type, args.device)

    # Start game
    human_white = args.color == "white"
    interface = GUIInterface(model, args.device, square_size=args.square_size)
    interface.start_game(human_white)


if __name__ == "__main__":
    main()
