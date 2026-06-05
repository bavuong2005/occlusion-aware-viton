"""
Stage 4: DensePose estimation using Detectron2.

Generates a DensePose visualization map that serves as a body pose
condition for StableVITON. The output is a colored image where each
body part is rendered with a distinct color on a black background.

Extracted from offline_preprocessing notebook cells 19-21.
Model: Detectron2 DensePose R50-FPN (~2GB VRAM)

Dependencies:
    - detectron2 (built from source)
    - DensePose project from detectron2 repo
"""

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image

from utils.memory import clear_memory, get_device


class DensePoseStage:
    """DensePose body pose estimation for VITON conditioning."""

    # Default paths — these should be overridden for Modal deployment
    DEFAULT_CFG = "densepose_rcnn_R_50_FPN_s1x.yaml"
    DEFAULT_WEIGHTS = "densepose_model.pkl"

    def __init__(
        self,
        device: Optional[str] = None,
        cfg_path: Optional[str] = None,
        weights_path: Optional[str] = None,
        detectron2_densepose_dir: Optional[str] = None,
    ):
        self.device = device or get_device()
        self.cfg_path = cfg_path or self.DEFAULT_CFG
        self.weights_path = weights_path or self.DEFAULT_WEIGHTS
        self.detectron2_densepose_dir = detectron2_densepose_dir
        self.predictor = None

    def load(self):
        """Load DensePose model into VRAM."""
        import sys

        # Add DensePose project to path if specified
        if self.detectron2_densepose_dir:
            densepose_path = str(self.detectron2_densepose_dir)
            if densepose_path not in sys.path:
                sys.path.insert(0, densepose_path)

        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        from densepose import add_densepose_config

        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(self.cfg_path)
        cfg.MODEL.WEIGHTS = self.weights_path
        cfg.MODEL.DEVICE = self.device

        self.predictor = DefaultPredictor(cfg)
        print(f"[DensePose] Model loaded on {self.device}")

    def unload(self):
        """Free model from VRAM."""
        del self.predictor
        self.predictor = None
        clear_memory()

    def run(self, person_img: Image.Image) -> Image.Image:
        """
        Generate a DensePose visualization for the person image.

        The output is a colored body-part map on a black background,
        using the PARULA colormap. This serves as the pose condition
        for StableVITON inference.

        Args:
            person_img: Person image (PIL RGB).

        Returns:
            DensePose visualization image (PIL RGB).
        """
        if self.predictor is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        from densepose.vis.extractor import DensePoseResultExtractor

        # Convert PIL to BGR for detectron2
        img_cv2 = cv2.cvtColor(np.array(person_img), cv2.COLOR_RGB2BGR)

        with torch.no_grad():
            outputs = self.predictor(img_cv2)

        if "instances" not in outputs or len(outputs["instances"]) == 0:
            raise RuntimeError("No person detected in the image for DensePose.")

        instances = outputs["instances"].to("cpu")

        if not instances.has("pred_densepose"):
            raise RuntimeError(
                "Person detected but DensePose prediction missing. "
                "Check config/weights compatibility."
            )

        # Extract DensePose results
        extractor = DensePoseResultExtractor()
        densepose_data = extractor(instances)
        results, boxes_xywh = densepose_data

        # Render DensePose on black background
        black_bg = np.zeros_like(img_cv2)
        img_h, img_w = black_bg.shape[:2]

        for i, result in enumerate(results):
            box = boxes_xywh[i]
            x, y, w, h = [int(v) for v in box]

            # Body part labels (0-24) → normalized to colormap range
            labels = result.labels.cpu().numpy()
            labels_norm = (labels * 255.0 / 24).astype(np.uint8)
            color_map = cv2.applyColorMap(labels_norm, cv2.COLORMAP_PARULA)

            # Only keep colored pixels where body parts are detected
            mask = (labels > 0)[:, :, None].astype(np.uint8)

            # Safe coordinate clipping
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(img_w, x + w), min(img_h, y + h)

            slice_h, slice_w = y2 - y1, x2 - x1
            color_map_sliced = color_map[:slice_h, :slice_w]
            mask_sliced = mask[:slice_h, :slice_w]

            roi = black_bg[y1:y2, x1:x2]
            black_bg[y1:y2, x1:x2] = roi * (1 - mask_sliced) + color_map_sliced * mask_sliced

        # Convert BGR back to RGB for PIL
        densepose_rgb = cv2.cvtColor(black_bg, cv2.COLOR_BGR2RGB)
        print(f"[DensePose] Visualization generated: {densepose_rgb.shape}")

        return Image.fromarray(densepose_rgb)
