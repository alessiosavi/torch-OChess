"""
Utility functions for torch-OChess.

Provides shared utilities for model loading, device handling, and other common operations.
"""

from ochess.utils.device import get_available_device, get_device_info
from ochess.utils.model_loader import (create_model,
                                       infer_model_config_from_state_dict,
                                       load_model)

__all__ = [
    # Model loading
    "create_model",
    "infer_model_config_from_state_dict",
    "load_model",
    # Device utilities
    "get_available_device",
    "get_device_info",
]
