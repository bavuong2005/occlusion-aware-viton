"""
Stage 5: Cloth Parsing using SegFormer.

Segments the cloth image to create a binary mask of the clothing region.
Uses the same SegFormer model as person parsing but with different
label selection logic optimized for isolated clothing images.

Extracted from offline_preprocessing notebook cells 22-23.
Model: mattmdjaga/segformer_b2_clothes (~1GB VRAM, shared with PersonParsing)
"""

from typing import Optional

import numpy as np
import torch
from PIL import Image

from postprocess.mask_utils import (
    clean_mask,
    get_binary_mask_from_labels,
    pick_best_top_labels,
    refine_top_mask,
)
from utils.memory import clear_memory, get_device


CLOTH_CATEGORY_CANDIDATES = {
    "Upper-body": [[4], [4, 7], [7], [4, 5], [4, 6]],
    "Lower-body": [[6], [5], [5, 6]],
    "Dress": [[7], [4, 7], [4, 5, 6]],
}

PARSING_MODEL_ID = "mattmdjaga/segformer_b2_clothes"


class ClothParsingStage:
    """SegFormer-based cloth mask generation."""

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
        print(f"[ClothParsing] SegFormer loaded on {self.device}")

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
        logits = outputs.logits

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
        cloth_img: Image.Image,
        person_top_labels: Optional[list] = None,
        category: str = "Upper-body",
    ) -> dict:
        """
        Parse cloth image and generate binary cloth mask.

        Args:
            cloth_img: Cloth image (PIL RGB).
            person_top_labels: If available, prioritize matching labels
                              from person parsing for consistency.
            category: Garment category ("Upper-body", "Lower-body", "Dress").

        Returns:
            dict with keys:
                - "cloth_mask": np.ndarray (0/255) binary cloth mask
                - "cloth_parsing_map": np.ndarray segmentation map
                - "cloth_labels": list of selected label IDs
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # 1. Run segmentation
        cloth_parsing_map = self._run_segformer(cloth_img)

        # 2. Build candidate list: prioritize person's labels if available
        candidates = list(CLOTH_CATEGORY_CANDIDATES[category])
        if person_top_labels:
            if person_top_labels not in candidates:
                candidates.insert(0, person_top_labels)
            else:
                candidates.remove(person_top_labels)
                candidates.insert(0, person_top_labels)

        # 3. Pick best labels
        cloth_labels, cloth_mask = pick_best_top_labels(
            cloth_parsing_map, candidates,
            min_area_ratio=0.05, max_area_ratio=0.90,
        )

        # 4. Refine mask
        cloth_mask = refine_top_mask(cloth_mask, open_k=5, close_k=9)

        print(f"[ClothParsing] Labels selected: {cloth_labels}, "
              f"mask area: {(cloth_mask > 127).sum()}")

        return {
            "cloth_mask": cloth_mask,
            "cloth_parsing_map": cloth_parsing_map,
            "cloth_labels": cloth_labels,
        }
