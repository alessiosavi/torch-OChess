"""
Model loading utilities for torch-OChess.

Provides shared functions for creating and loading models across all scripts.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def create_model(
    model_type: str, config_overrides: Optional[Dict[str, Any]] = None
) -> nn.Module:
    """
    Create a model of the specified type.

    Args:
        model_type: Model architecture type (resnet, transformer, hybrid)
        config_overrides: Optional dictionary of config overrides

    Returns:
        Initialized model

    Raises:
        ValueError: If model_type is unknown
    """
    from ochess.model.chess_hybrid import ChessHybrid, ChessHybridConfig
    from ochess.model.chess_resnet import ChessResNet, ChessResNetConfig
    from ochess.model.chess_transformer import (ChessTransformer,
                                                ChessTransformerConfig)

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


def infer_model_config_from_state_dict(
    state_dict: Dict[str, torch.Tensor], model_type: str
) -> Dict[str, Any]:
    """
    Infer model configuration from state dict keys and tensor shapes.

    This allows loading checkpoints without knowing the exact architecture
    parameters used during training.

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
            if "residual_tower.blocks." in key:
                parts = key.split(".")
                for i, p in enumerate(parts):
                    if p == "blocks" and i + 1 < len(parts):
                        try:
                            block_indices.add(int(parts[i + 1]))
                        except ValueError:
                            pass
        if block_indices:
            config["num_residual_blocks"] = max(block_indices) + 1

        # Infer hidden_dim from conv weights
        for key, tensor in state_dict.items():
            if "residual_tower.blocks.0.conv1.weight" in key:
                config["hidden_dim"] = tensor.shape[0]
                break

    elif model_type == "transformer":
        # Count transformer layers
        layer_indices = set()
        for key in state_dict.keys():
            if "transformer.layers." in key:
                parts = key.split(".")
                for i, p in enumerate(parts):
                    if p == "layers" and i + 1 < len(parts):
                        try:
                            layer_indices.add(int(parts[i + 1]))
                        except ValueError:
                            pass
        if layer_indices:
            config["num_layers"] = max(layer_indices) + 1

        # Infer embed_dim
        for key, tensor in state_dict.items():
            if "cls_token" in key:
                config["embed_dim"] = tensor.shape[-1]
                break

    elif model_type == "hybrid":
        # Count residual blocks
        block_indices = set()
        for key in state_dict.keys():
            if "residual_tower.blocks." in key:
                parts = key.split(".")
                for i, p in enumerate(parts):
                    if p == "blocks" and i + 1 < len(parts):
                        try:
                            block_indices.add(int(parts[i + 1]))
                        except ValueError:
                            pass
        if block_indices:
            config["num_residual_blocks"] = max(block_indices) + 1

        # Count attention layers
        attn_indices = set()
        for key in state_dict.keys():
            if "attention_layers." in key:
                parts = key.split(".")
                for i, p in enumerate(parts):
                    if p == "attention_layers" and i + 1 < len(parts):
                        try:
                            attn_indices.add(int(parts[i + 1]))
                        except ValueError:
                            pass
        if attn_indices:
            config["num_attention_layers"] = max(attn_indices) + 1

        # Infer hidden_dim from conv weights
        for key, tensor in state_dict.items():
            if "residual_tower.blocks.0.conv1.weight" in key:
                config["hidden_dim"] = tensor.shape[0]
                break

    return config


def load_model(
    model_path: str, model_type: str, device: str, verbose: bool = True
) -> nn.Module:
    """
    Load a trained model from checkpoint.

    Supports multiple formats:
    1. Checkpoint dict with 'model_state_dict' (from Trainer)
    2. Raw state_dict (just the weights)
    3. Full model object (from torch.save(model, path))

    Args:
        model_path: Path to the model checkpoint
        model_type: Type of model architecture (resnet, transformer, hybrid)
        device: Device to load the model on
        verbose: Whether to print loading information

    Returns:
        Loaded model in eval mode
    """
    if verbose:
        print(f"Loading {model_type} model from {model_path}...")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Check if it's a checkpoint dict or a full model
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # It's a training checkpoint - need to create model and load state dict
        if verbose:
            print("Detected checkpoint format (state_dict)")

        state_dict = checkpoint["model_state_dict"]

        # Infer model architecture from state dict
        config_overrides = infer_model_config_from_state_dict(state_dict, model_type)
        if verbose:
            print(f"  Inferred config: {config_overrides}")

        # Create model with inferred config
        model = create_model(model_type, config_overrides)

        # Load state dict
        model.load_state_dict(state_dict)

        # Print some info
        if verbose:
            if "epoch" in checkpoint:
                print(f"  Loaded from epoch {checkpoint['epoch'] + 1}")
            if "best_val_loss" in checkpoint:
                print(f"  Best validation loss: {checkpoint['best_val_loss']:.4f}")

    elif isinstance(checkpoint, dict) and "model_state_dict" not in checkpoint:
        # It might be just a state_dict without wrapper
        if verbose:
            print("Detected raw state_dict format")

        # Infer config from raw state dict
        config_overrides = infer_model_config_from_state_dict(checkpoint, model_type)
        if verbose and config_overrides:
            print(f"  Inferred config: {config_overrides}")

        model = create_model(model_type, config_overrides)
        model.load_state_dict(checkpoint)

    else:
        # It's a full model object
        if verbose:
            print("Detected full model format")
        model = checkpoint

    model = model.to(device)
    model.eval()

    if verbose:
        print(f"  Model parameters: {model.num_parameters:,}")

    return model
