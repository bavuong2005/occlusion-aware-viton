"""
Stage 3: Person Parsing using SegFormer.

Segments the person image into semantic regions (clothing, skin, hair, etc.)
using SegFormer B2 fine-tuned on clothing data. The parsing map is then
used to create the agnostic image and mask required by StableVITON.

Extracted from offline_preprocessing notebook cells 12-17.
Model: mattmdjaga/segformer_b2_clothes (~1GB VRAM)

SegFormer Label Map (mattmdjaga/segformer_b2_clothes):
    0: Background, 1: Hat, 2: Hair, 3: Sunglasses, 4: Upper-clothes,
    5: Skirt, 6: Pants, 7: Dress, 8: Belt, 9: Left-shoe, 10: Right-shoe,
    11: Face, 12: Left-leg, 13: Right-leg, 14: Left-arm, 15: Right-arm,
    16: Bag, 17: Scarf
"""

from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

from postprocess.mask_utils import (
    clean_mask,
    create_robust_agnostic,
    get_binary_mask_from_labels,
    keep_largest_component,
    pick_best_top_labels,
)
from utils.memory import clear_memory, get_device


# Candidate label sets based on Garment Category
CATEGORY_CANDIDATES = {
    "Upper-body": [[4], [4, 7], [7]],
    "Lower-body": [[6], [5], [5, 6]],
    "Dress": [[7], [4, 7], [4, 5, 6]],
}

PARSING_MODEL_ID = "mattmdjaga/segformer_b2_clothes"


class PersonParsingStage:
    """SegFormer-based person parsing and agnostic image generation."""

    def __init__(self, device: Optional[str] = None):
        self.device = device or get_device()
        self.processor = None
        self.model = None

    def load(self):
        """Load SegFormer model into VRAM."""
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

        self.processor = AutoImageProcessor.from_pretrained(PARSING_MODEL_ID)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(
            PARSING_MODEL_ID
        ).to(self.device)
        self.model.eval()
        print(f"[PersonParsing] SegFormer loaded on {self.device}")

    def unload(self):
        """Free model from VRAM."""
        del self.model, self.processor
        self.model = self.processor = None
        clear_memory()

    @torch.no_grad()
    def _run_segformer(self, image_pil: Image.Image) -> np.ndarray:
        """Run SegFormer inference and return the label map."""
        inputs = self.processor(images=image_pil, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits  # [B, C, h, w]

        upsampled = torch.nn.functional.interpolate(
            logits,
            size=image_pil.size[::-1],  # (H, W)
            mode="bilinear",
            align_corners=False,
        )
        pred = upsampled.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
        return pred

    def run(
        self,
        person_img: Image.Image,
        object_mask: Optional[np.ndarray] = None,
        category: str = "Upper-body",
    ) -> dict:
        """
        Parse person image and generate agnostic image/mask.

        Args:
            person_img: Person image (PIL RGB).
            object_mask: Optional binary mask (0/255) of occluding objects
                        to include in the agnostic mask.
            category: Garment category ("Upper-body", "Lower-body", "Dress").

        Returns:
            dict with keys:
                - "parsing_map": np.ndarray (H, W) segmentation map
                - "top_labels": list of selected label IDs
                - "person_top_mask": np.ndarray (0/255) clothing mask
                - "agnostic_img": PIL.Image agnostic person image
                - "agnostic_mask": PIL.Image agnostic mask
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # 1. Run segmentation
        parsing_map = self._run_segformer(person_img)
        print(f"[PersonParsing] Labels found: {np.unique(parsing_map)}")

        # 2. Select best clothing labels (max_area_ratio=0.95 to allow full-body Dress/Pants)
        top_labels, person_top_mask = pick_best_top_labels(
            parsing_map, CATEGORY_CANDIDATES[category],
            min_area_ratio=0.05, max_area_ratio=0.95,
        )
        person_top_mask = clean_mask(person_top_mask, open_k=5, close_k=15)
        print(f"[PersonParsing] TOP_LABELS selected: {top_labels}")

        # 3. Merge object mask into person_top_mask (V10 patch from online notebook)
        if object_mask is not None:
            object_mask_resized = cv2.resize(
                object_mask,
                (person_top_mask.shape[1], person_top_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            # Dilate object mask to fill gaps at clothing-object boundary
            kernel = np.ones((11, 11), np.uint8)
            object_mask_dilated = cv2.dilate(object_mask_resized, kernel, iterations=1)

            person_top_mask = np.maximum(person_top_mask, object_mask_dilated)
            person_top_mask = clean_mask(person_top_mask, open_k=0, close_k=15)

        # 4. Create agnostic image and mask
        agnostic_img, agnostic_mask = create_robust_agnostic(
            person_img, parsing_map,
            top_labels=top_labels,
            object_mask=object_mask,
            category=category,
        )

        return {
            "parsing_map": parsing_map,
            "top_labels": top_labels,
            "person_top_mask": person_top_mask,
            "agnostic_img": agnostic_img,
            "agnostic_mask": agnostic_mask,
        }
