"""
Stage 1: Object Detection using Grounding DINO.

Detects objects (e.g. bags, accessories) in the person image that may
occlude the clothing region. These detections are used by the SAM
segmentation stage to create precise object masks.

Extracted from offline_preprocessing notebook cells 7-8.
Model: IDEA-Research/grounding-dino-tiny (~1GB VRAM)
"""

from typing import Optional

import numpy as np
import torch
from PIL import Image

from utils.memory import clear_memory, get_device


class ObjectDetectionStage:
    """Grounding DINO object detector for finding occluding objects."""

    MODEL_ID = "IDEA-Research/grounding-dino-tiny"

    def __init__(self, device: Optional[str] = None):
        self.device = device or get_device()
        self.processor = None
        self.model = None

    def load(self):
        """Load Grounding DINO model into VRAM."""
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.MODEL_ID
        ).to(self.device)
        print(f"[ObjectDetection] Grounding DINO loaded on {self.device}")

    def unload(self):
        """Free model from VRAM."""
        del self.model, self.processor
        self.model = self.processor = None
        clear_memory()

    def run(
        self,
        person_img: Image.Image,
        text_prompt: str = "bag.",
        threshold: float = 0.3,
    ) -> tuple:
        """
        Detect objects matching the text prompt in the person image.

        Args:
            person_img: Person image (PIL RGB).
            text_prompt: Object description for zero-shot detection.
            threshold: Detection confidence threshold.

        Returns:
            (boxes: np.ndarray or None, scores: np.ndarray, labels: list)
            boxes is None if no objects detected.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        inputs = self.processor(
            images=person_img, text=text_prompt, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=threshold,
            text_threshold=threshold,
            target_sizes=[person_img.size[::-1]],  # (H, W)
        )

        result = results[0]
        boxes = result["boxes"].cpu().numpy()
        scores = result["scores"].cpu().numpy()
        labels = result["labels"]

        if len(boxes) == 0:
            print("[ObjectDetection] No objects detected.")
            return None, scores, labels

        print(f"[ObjectDetection] Detected {len(boxes)} objects: {labels}")
        return boxes, scores, labels
