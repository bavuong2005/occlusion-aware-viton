"""
GPU memory management utilities.

Provides stage-based VRAM management to allow sequential loading/unloading
of large models on a single GPU without OOM.

Extracted from the offline_preprocessing notebook's clear_memory() pattern.
"""

import gc
from contextlib import contextmanager

import torch


def clear_memory():
    """Force garbage collection and clear CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


@contextmanager
def gpu_stage(stage_name: str):
    """
    Context manager that clears GPU memory after each pipeline stage.

    Usage:
        with gpu_stage("Person Parsing"):
            model = load_model()
            result = model(input)
            del model
        # VRAM is freed here even if an exception occurs
    """
    print(f"[{stage_name}] Starting...")
    try:
        yield
    finally:
        clear_memory()
        print(f"[{stage_name}] Done. VRAM freed.")


def get_device() -> str:
    """Return 'cuda' if available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def print_vram_usage():
    """Print current VRAM usage (debug helper)."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  VRAM: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    else:
        print("  VRAM: N/A (no CUDA)")
