"""
Leap Trading System - Device Utilities
Centralized PyTorch device management.
"""

import torch
from functools import lru_cache
from typing import Union
import logging

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_device() -> torch.device:
    """
    Get the best available PyTorch device (cached).

    Priority:
    1. CUDA (NVIDIA GPU)
    2. MPS (Apple Silicon)
    3. CPU (fallback)

    Returns:
        torch.device: The best available device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using Apple Silicon MPS device")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU device")
    return device


def get_device_string() -> str:
    """
    Get device as string for configuration.

    Returns:
        str: Device name ('cuda', 'mps', or 'cpu')
    """
    return str(get_device().type)


def resolve_device(device: Union[str, torch.device, None] = 'auto') -> torch.device:
    """
    Resolve a device specification to a torch.device.

    Args:
        device: Device specification. Can be:
            - 'auto': Automatically detect best device
            - 'cuda', 'mps', 'cpu': Specific device type
            - torch.device: Return as-is
            - None: Same as 'auto'

    Returns:
        torch.device: Resolved device
    """
    if device is None or device == 'auto':
        return get_device()
    elif isinstance(device, torch.device):
        return device
    else:
        return torch.device(device)


__all__ = ['get_device', 'get_device_string', 'resolve_device']
