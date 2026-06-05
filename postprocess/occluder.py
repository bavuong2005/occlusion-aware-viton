"""
Occluder restoration (bag/accessory preservation).

Implements the V10 refined alpha-blending approach from the online_inference
notebook. After StableVITON generates the try-on result, objects like bags
that were occluding the original clothing need to be composited back onto
the result image.
"""

import cv2
import numpy as np
from PIL import Image


def refined_restore_occluder(
    original_img: Image.Image,
    result_img: Image.Image,
    object_mask: Image.Image,
    erode_iter: int = 1,
    blur_ksize: int = 5,
) -> Image.Image:
    """
    Restore an occluding object (e.g. bag) from the original image onto
    the try-on result using refined alpha blending.

    The mask is first eroded to trim artifact edges, then blurred for
    smooth blending. Both the original image and mask are resized to
    match the result image dimensions.

    Args:
        original_img: Original person image (PIL RGB).
        result_img: StableVITON output image (PIL RGB).
        object_mask: Binary mask of the object to restore (PIL L).
        erode_iter: Number of erosion iterations (higher = more edge trimming).
        blur_ksize: Gaussian blur kernel size for mask feathering.

    Returns:
        Blended result image with the occluder restored (PIL RGB).
    """
    target_size = result_img.size  # (width, height)

    # Resize original and mask to match result dimensions
    orig_resized = original_img.resize(target_size, Image.Resampling.LANCZOS)
    mask_resized = object_mask.resize(target_size, Image.Resampling.NEAREST)

    orig_np = np.array(orig_resized).astype(np.float32)
    res_np = np.array(result_img).astype(np.float32)
    mask_np = np.array(mask_resized.convert("L"))

    # 1. Erode mask to trim artifact edges
    kernel_erode = np.ones((3, 3), np.uint8)
    eroded_mask = cv2.erode(mask_np, kernel_erode, iterations=erode_iter)

    # 2. Blur mask edges for smooth blending
    if blur_ksize > 0:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        blurred_mask = cv2.GaussianBlur(eroded_mask, (blur_ksize, blur_ksize), 0)
    else:
        blurred_mask = eroded_mask

    # 3. Alpha blending: original * alpha + result * (1 - alpha)
    alpha = blurred_mask.astype(np.float32) / 255.0
    alpha = alpha[..., np.newaxis]

    blended_np = orig_np * alpha + res_np * (1.0 - alpha)

    return Image.fromarray(blended_np.astype(np.uint8))
