"""
Image I/O and conversion utilities.

Consolidated from scattered helpers across both notebooks:
- save_rgb, save_mask from offline notebook cell 1
- ensure_rgb, ensure_gray from app_online.py
- resize_rgb from offline notebook cell 1
"""

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image


# ─── Image Loading ──────────────────────────────────────────────────

def ensure_rgb(path_or_img: Union[Path, str, Image.Image]) -> Image.Image:
    """Open an image and convert to RGB."""
    if isinstance(path_or_img, Image.Image):
        return path_or_img.convert("RGB")
    return Image.open(path_or_img).convert("RGB")


def ensure_gray(path_or_img: Union[Path, str, Image.Image]) -> Image.Image:
    """Open an image and convert to grayscale."""
    if isinstance(path_or_img, Image.Image):
        return path_or_img.convert("L")
    return Image.open(path_or_img).convert("L")


# ─── Image Saving ───────────────────────────────────────────────────

def save_rgb(img: Image.Image, path: Union[Path, str]):
    """Save an RGB image, creating parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def save_mask(mask: Union[Image.Image, np.ndarray], path: Union[Path, str]):
    """
    Save a mask image (grayscale), handling various input formats:
    - PIL Image → save directly as "L"
    - bool ndarray → convert to 0/255
    - float ndarray with max <= 1 → scale to 0/255
    - uint8 ndarray → save directly
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(mask, Image.Image):
        mask.convert("L").save(path)
        return

    mask = np.array(mask)
    if mask.dtype == bool:
        mask = mask.astype(np.uint8) * 255
    elif mask.max() <= 1:
        mask = mask.astype(np.uint8) * 255
    else:
        mask = mask.astype(np.uint8)

    Image.fromarray(mask).save(path)


# ─── Image Resizing ─────────────────────────────────────────────────

def resize_rgb(img: Image.Image, size: tuple = (768, 1024)) -> Image.Image:
    """Resize an RGB image to (width, height) using bicubic interpolation."""
    return img.resize(size, Image.BICUBIC)


# ─── Validation ──────────────────────────────────────────────────────

def validate_nonempty_mask(
    mask_path: Union[Path, str], min_positive_pixels: int = 100
) -> tuple:
    """
    Check that a mask file exists and has sufficient positive (white) area.

    Returns:
        (ok: bool, message: str)
    """
    mask_path = Path(mask_path)
    if not mask_path.exists():
        return False, "file_not_found"

    arr = np.array(Image.open(mask_path).convert("L"))
    positive = int((arr > 127).sum())
    if positive < min_positive_pixels:
        return False, f"too_small_positive_area={positive}"
    return True, f"positive_area={positive}"
