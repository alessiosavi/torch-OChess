"""
Device utilities for torch-OChess.

Provides shared functions for device detection and validation.
"""

import logging
from typing import Optional

import torch


def get_available_device(
    requested: str = "cuda", logger: Optional[logging.Logger] = None
) -> str:
    """
    Get the best available device, falling back gracefully.

    Checks if the requested device is available and falls back to CPU if not.
    Supports cuda, mps (Apple Silicon), and cpu.

    Args:
        requested: Requested device (cuda, mps, cpu)
        logger: Optional logger for warning messages

    Returns:
        Available device string
    """
    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        else:
            msg = "CUDA not available, using CPU"
            if logger:
                logger.warning(msg)
            else:
                print(msg)
            return "cpu"

    elif requested == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            msg = "MPS not available, using CPU"
            if logger:
                logger.warning(msg)
            else:
                print(msg)
            return "cpu"

    elif requested == "cpu":
        return "cpu"

    else:
        msg = f"Unknown device '{requested}', using CPU"
        if logger:
            logger.warning(msg)
        else:
            print(msg)
        return "cpu"


def get_device_info(device: str) -> dict:
    """
    Get information about the specified device.

    Args:
        device: Device string (cuda, mps, cpu)

    Returns:
        Dictionary with device information
    """
    info = {
        "device": device,
        "available": True,
    }

    if device == "cuda":
        if torch.cuda.is_available():
            info["device_count"] = torch.cuda.device_count()
            info["device_name"] = torch.cuda.get_device_name(0)
            info["memory_allocated"] = torch.cuda.memory_allocated(0)
            info["memory_reserved"] = torch.cuda.memory_reserved(0)
        else:
            info["available"] = False

    elif device == "mps":
        info["available"] = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )

    return info
