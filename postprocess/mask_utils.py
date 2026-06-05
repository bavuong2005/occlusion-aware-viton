"""
Mask morphology and selection utilities.

Extracted from offline_preprocessing notebook cell 2.
Provides mask cleaning, label selection, connected component filtering,
and agnostic image/mask generation for VITON preprocessing.
"""

import cv2
import numpy as np
from PIL import Image


# ─── Morphological Operations ───────────────────────────────────────

def clean_mask(mask: np.ndarray, open_k: int = 5, close_k: int = 7, blur_k: int = 0) -> np.ndarray:
    """
    Clean a binary mask using morphological open/close and optional blur.

    Args:
        mask: Binary mask (0/255 uint8 or bool).
        open_k: Kernel size for morphological opening (removes small noise).
        close_k: Kernel size for morphological closing (fills small holes).
        blur_k: Kernel size for Gaussian blur (0 = disabled).

    Returns:
        Cleaned binary mask (0/255 uint8).
    """
    mask = mask.astype(np.uint8)

    if open_k > 0:
        kernel_open = np.ones((open_k, open_k), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    if close_k > 0:
        kernel_close = np.ones((close_k, close_k), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    if blur_k and blur_k > 1:
        if blur_k % 2 == 0:
            blur_k += 1
        mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)

    mask = (mask > 127).astype(np.uint8) * 255
    return mask


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected component in a binary mask.

    Args:
        mask: Binary mask (0/255 uint8).

    Returns:
        Mask with only the largest connected component.
    """
    mask_bin = (mask > 127).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)

    if num_labels <= 1:
        return (mask_bin * 255).astype(np.uint8)

    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest = (labels == largest_idx).astype(np.uint8) * 255
    return largest


def refine_top_mask(top_mask: np.ndarray, open_k: int = 5, close_k: int = 9) -> np.ndarray:
    """
    Refine a clothing/top mask: clean (Removed keep_largest_component to avoid dropping disconnected sleeves).
    """
    top_mask = clean_mask(top_mask, open_k=open_k, close_k=close_k)
    return top_mask


# ─── Label Selection ────────────────────────────────────────────────

def get_binary_mask_from_labels(label_map: np.ndarray, target_labels: list) -> np.ndarray:
    """Create a binary mask (0/255) from selected labels in a segmentation map."""
    mask = np.isin(label_map, target_labels).astype(np.uint8) * 255
    return mask


def pick_best_top_labels(
    label_map: np.ndarray,
    candidate_label_sets: list,
    min_area_ratio: float = 0.01,
    max_area_ratio: float = 0.55,
) -> tuple:
    """
    Choose the best set of labels for the clothing region based on mask area.

    Iterates through candidate label sets, computes the mask area for each,
    and picks the one with the largest area within acceptable bounds.

    Args:
        label_map: Segmentation map (H, W) with integer labels.
        candidate_label_sets: List of label sets to try, e.g. [[4], [5], [4,5]].
        min_area_ratio: Minimum acceptable mask area / total area.
        max_area_ratio: Maximum acceptable mask area / total area.

    Returns:
        (best_labels: list, best_mask: np.ndarray)
    """
    H, W = label_map.shape
    total_area = H * W

    best_labels = None
    best_mask = None
    best_score = -1

    for labels in candidate_label_sets:
        mask = get_binary_mask_from_labels(label_map, labels)
        mask = clean_mask(mask, open_k=3, close_k=7)
        area_ratio = (mask > 127).sum() / total_area

        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue

        score = area_ratio
        if score > best_score:
            best_score = score
            best_labels = labels
            best_mask = mask

    if best_labels is None:
        # Fallback to label [4] if no candidate is suitable
        best_labels = [4]
        best_mask = get_binary_mask_from_labels(label_map, best_labels)

    return best_labels, best_mask


# ─── Agnostic Generation ────────────────────────────────────────────

def create_robust_agnostic(
    person_img_pil: Image.Image,
    parsing_map: np.ndarray,
    top_labels: list = None,
    object_mask: np.ndarray = None,
    category: str = "Upper-body",
) -> tuple:
    """
    Create agnostic image and mask for VITON inference.

    Erases the clothing region (+ optionally occluding objects like bags)
    and exposed skin (arms, neck) from the person image, replacing
    them with a neutral gray (127).

    Args:
        person_img_pil: Original person image (RGB PIL).
        parsing_map: Segmentation map from SegFormer (H, W).
        top_labels: Label IDs for the upper-body clothing region.
        object_mask: Optional binary mask (0/255) for occluding objects.
        category: Garment category ("Upper-body", "Lower-body", "Dress").

    Returns:
        (agnostic_img: PIL.Image, agnostic_mask: PIL.Image)
    """
    if top_labels is None:
        top_labels = [4]

    person_np = np.array(person_img_pil)

    # 1. Get clothing mask
    mask_cloth = np.isin(parsing_map, top_labels).astype(np.uint8) * 255

    # 2. Merge object mask (e.g. bag) if provided
    if object_mask is not None:
        mask_object = (object_mask > 127).astype(np.uint8) * 255
        mask_cloth = np.maximum(mask_cloth, mask_object)

    # 3. Get skin mask
    # SegFormer mattmdjaga/segformer_b2_clothes labels:
    # 12 = left-leg, 13 = right-leg, 14 = right-arm, 15 = left-arm
    skin_labels = [14, 15]
    if category == "Lower-body":
        skin_labels = [12, 13]
    elif category == "Dress":
        skin_labels = [12, 13, 14, 15]
        
    mask_skin = np.isin(parsing_map, skin_labels).astype(np.uint8) * 255

    # 4. Combine clothing + skin → dilate to cover edges
    combined_mask = np.maximum(mask_cloth, mask_skin)
    kernel = np.ones((15, 15), np.uint8)
    agnostic_mask_np = cv2.dilate(combined_mask, kernel, iterations=2)

    # Prevent lower-body mask from bleeding up into the upper clothes
    if category == "Lower-body":
        mask_preserve = np.isin(parsing_map, [4]).astype(np.uint8) * 255
        agnostic_mask_np[mask_preserve > 127] = 0

    # 5. PROTECT REGIONS FROM BEING ERASED/DRAWN OVER
    # Protect head/face/hair (labels: 1=Hat, 2=Hair, 3=Sunglasses, 11=Face)
    mask_head = np.isin(parsing_map, [1, 2, 3, 11]).astype(np.uint8) * 255
    agnostic_mask_np[mask_head > 127] = 0

    # 6. Create agnostic image: fill masked regions with neutral gray
    agnostic_np = person_np.copy()
    agnostic_np[agnostic_mask_np > 127] = 127

    return Image.fromarray(agnostic_np), Image.fromarray(agnostic_mask_np)


# ─── Feather Mask (for blending) ────────────────────────────────────

def feather_mask(mask_pil: Image.Image, ksize: int = 15) -> np.ndarray:
    """
    Create a soft-edged alpha mask for blending.

    Args:
        mask_pil: Binary mask (PIL Image).
        ksize: Gaussian blur kernel size for feathering.

    Returns:
        Float alpha mask (H, W, 1) in range [0, 1].
    """
    mask = np.array(mask_pil.convert("L")).astype(np.uint8)
    mask = (mask > 127).astype(np.uint8) * 255

    if ksize % 2 == 0:
        ksize += 1

    mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
    return (mask.astype(np.float32) / 255.0)[..., None]
