"""Hardware detection and dtype selection."""

from __future__ import annotations
from contextlib import nullcontext

import torch


def get_device(override: str | None = None) -> torch.device:
    """Return the best available device, or the one the user forced."""
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype(device: torch.device) -> torch.dtype:
    """
    CUDA: bfloat16 on Ampere+ (A100, 4090…), else float16.
    MPS / CPU: must use float32 — MPS doesn't support float16 autocast.
    """
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def autocast_ctx(device: torch.device, dtype: torch.dtype):
    """Mixed-precision context manager. No-op on non-CUDA devices."""
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()
