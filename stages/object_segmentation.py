"""
Stage 2: Object Segmentation using SAM (Segment Anything Model).

Takes bounding boxes from Grounding DINO and produces precise pixel-level
masks for occluding objects. These masks are used to:
1. Exclude the object from agnostic generation (so the model doesn't try to
   replace it with clothing)
2. Restore the object onto the final result via alpha blending

Extracted from offline_preprocessing notebook cells 9-10.
Model: SAM ViT-H (~7GB VRAM)
"""

from typing import Optional

import numpy as np
import torch
from PIL import Image

from postprocess.mask_utils import clean_mask, keep_largest_component
from utils.memory import clear_memory, get_device


class ObjectSegmentationStage:
    """SAM-based object segmentation from bounding box prompts."""

    SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
    SAM_MODEL_TYPE = "vit_h"

    def __init__(self, device: Optional[str] = None, checkpoint_path: Optional[str] = None):
        self.device = device or get_device()
        self.checkpoint_path = checkpoint_path or self.SAM_CHECKPOINT
        self.sam = None
        self.predictor = None

    def load(self):
        """Load SAM model into VRAM."""
        from segment_anything import sam_model_registry, SamPredictor

        self.sam = sam_model_registry[self.SAM_MODEL_TYPE](
            checkpoint=self.checkpoint_path
        )
        self.sam.to(self.device)
        self.predictor = SamPredictor(self.sam)
        print(f"[ObjectSegmentation] SAM {self.SAM_MODEL_TYPE} loaded on {self.device}")

    def unload(self):
        """Free model from VRAM."""
        del self.sam, self.predictor
        self.sam = self.predictor = None
        clear_memory()

    def run(
        self,
        person_img: Image.Image,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: list,
        target_labels: Optional[list[str]] = None,
        open_k: int = 3,
        close_k: int = 7,
    ) -> Optional[np.ndarray]:
        """
        Segment the best-matching objects using SAM with box prompt.

        Args:
            person_img: Person image (PIL RGB).
            boxes: Detection bounding boxes from Grounding DINO.
            scores: Detection confidence scores.
            labels: Detection labels.
            target_labels: Which labels to segment (e.g. ["bag", "cat"]).
            open_k: Morphological opening kernel size for mask cleanup.
            close_k: Morphological closing kernel size for mask cleanup.

        Returns:
            Clean binary mask (0/255 uint8 ndarray) or None if no match.
        """
        if self.predictor is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Find the matching detections for the target labels
        if not target_labels:
            matched_indices = list(range(len(labels)))
        else:
            matched_indices = []
            for target_label in target_labels:
                matched_indices.extend([
                    i for i, label in enumerate(labels)
                    if target_label.lower() in str(label).lower()
                ])
            matched_indices = list(set(matched_indices))

        if not matched_indices:
            print("[ObjectSegmentation] No detection matches target labels.")
            return None

        print(f"[ObjectSegmentation] Matched indices: {matched_indices}")
        for idx in matched_indices:
            print(f"  - label={labels[idx]}, score={float(scores[idx]):.3f}")

        # Set image and predict mask
        self.predictor.set_image(np.array(person_img))

        combined_mask = np.zeros(person_img.size[::-1], dtype=np.uint8)

        for idx in matched_indices:
            box = boxes[idx]
            masks_pred, scores_pred, _ = self.predictor.predict(
                box=box,
                multimask_output=False,
            )

            mask = masks_pred[0].astype(np.uint8)
            mask = (mask > 0).astype(np.uint8) * 255
            combined_mask = np.bitwise_or(combined_mask, mask)

        # Clean up the combined mask
        object_mask = clean_mask(combined_mask, open_k=open_k, close_k=close_k)

        print(f"[ObjectSegmentation] Mask generated: {object_mask.shape}, "
              f"positive area: {(object_mask > 127).sum()}")
        return object_mask
