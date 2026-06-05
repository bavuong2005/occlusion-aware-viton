"""
Stage 6: StableVITON Try-On Inference.

Wraps the StableVITON inference.py CLI to run virtual try-on. This stage:
1. Prepares the VITON-HD test data directory structure
2. Invokes StableVITON's inference.py as a subprocess
3. Collects the output image

This is the heaviest stage (~10-12GB VRAM). It is always run last so
that VRAM cleanup is unnecessary.

Extracted from online_inference notebook (patched V10 version).
Model: StableVITON VITONHD checkpoint
"""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image


class TryOnInferenceStage:
    """StableVITON inference wrapper using subprocess CLI."""

    def __init__(
        self,
        stableviton_dir: Path,
        checkpoint_path: Path,
        runtime_dir: Path,
    ):
        """
        Args:
            stableviton_dir: Path to the StableVITON repo root.
            checkpoint_path: Path to VITONHD.ckpt.
            runtime_dir: Working directory for temporary data and outputs.
        """
        self.stableviton_dir = Path(stableviton_dir)
        self.checkpoint_path = Path(checkpoint_path)
        self.runtime_dir = Path(runtime_dir)

        self.config_path = self.stableviton_dir / "configs" / "VITONHD.yaml"
        self.data_root = self.runtime_dir / "data" / "viton_hd"
        self.output_root = self.runtime_dir / "outputs"

    def _prepare_test_sample(
        self,
        person_img: Image.Image,
        cloth_img: Image.Image,
        cloth_mask: Image.Image,
        agnostic_img: Image.Image,
        agnostic_mask: Image.Image,
        densepose_img: Image.Image,
        object_mask: Optional[np.ndarray] = None,
        sample_name: str = "00000_00.jpg",
    ):
        """
        Prepare the VITON-HD test directory structure with a single sample.

        This implements the V10 patched version from the online notebook:
        - Merges object mask (bag) into agnostic mask
        - Dilates combined mask to cover edges
        - Creates clean gray-filled agnostic image

        The directory structure created is:
            data/viton_hd/test/
                image/          → person image
                cloth/          → clothing image
                cloth-mask/     → clothing binary mask
                agnostic-v3.2/  → agnostic (clothing-erased) image
                agnostic-mask/  → agnostic binary mask
                image-densepose/ → body pose map
            data/viton_hd/test_pairs.txt
        """
        test_root = self.data_root / "test"

        folders = [
            "image", "cloth", "cloth-mask",
            "agnostic-mask", "agnostic-v3.2", "image-densepose",
        ]

        # Clean previous run
        if self.data_root.exists():
            shutil.rmtree(self.data_root)

        for folder in folders:
            (test_root / folder).mkdir(parents=True, exist_ok=True)

        # V10 patch: merge object mask into agnostic if provided
        if object_mask is not None:
            agnostic_mask_np = np.array(agnostic_mask.convert("L")) > 127
            object_mask_bool = object_mask > 127

            combined_mask_np = np.maximum(
                agnostic_mask_np.astype(np.uint8),
                object_mask_bool.astype(np.uint8),
            )

            # NOTE: Removed 25x25 dilation here because mask_utils.py already dilates by 30px!
            # Adding another 25px dilation causes the mask to bleed severely.
            agnostic_mask = Image.fromarray((combined_mask_np * 255).astype(np.uint8))

            # Rebuild agnostic image with combined mask
            person_np = np.array(person_img).copy()
            person_np[combined_mask_np > 0] = 127
            agnostic_img = Image.fromarray(person_np)

        mask_name = sample_name.replace(".jpg", "_mask.png")

        # Save all inputs
        person_img.save(test_root / "image" / sample_name)
        cloth_img.save(test_root / "cloth" / sample_name)
        cloth_mask.save(test_root / "cloth-mask" / sample_name)
        agnostic_img.save(test_root / "agnostic-v3.2" / sample_name)
        densepose_img.save(test_root / "image-densepose" / sample_name)

        # Save agnostic mask in both formats (some code paths expect each)
        agnostic_mask.save(test_root / "agnostic-mask" / mask_name)
        agnostic_mask.save(test_root / "agnostic-mask" / sample_name.replace(".jpg", ".png"))

        # Write test pairs file
        with open(self.data_root / "test_pairs.txt", "w") as f:
            f.write(f"{sample_name} {sample_name}\n")

        print(f"[TryOnInference] Test sample prepared at {self.data_root}")

    def _run_subprocess(self):
        """Run StableVITON inference.py as a subprocess."""
        if self.output_root.exists():
            shutil.rmtree(self.output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

        python_exe = sys.executable

        cmd = [
            python_exe,
            "inference.py",
            "--config_path", str(self.config_path),
            "--batch_size", "1",
            "--model_load_path", str(self.checkpoint_path),
            "--save_dir", str(self.output_root),
            "--data_root_dir", str(self.data_root),
        ]

        print(f"[TryOnInference] Running: {' '.join(cmd)}")
        print(f"[TryOnInference] CWD: {self.stableviton_dir}")

        result = subprocess.run(
            cmd,
            cwd=str(self.stableviton_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        if result.returncode != 0:
            print("[TryOnInference] STDOUT:", result.stdout)
            print("[TryOnInference] STDERR:", result.stderr)
            raise RuntimeError(f"StableVITON inference failed (exit code {result.returncode})")

        print("[TryOnInference] Inference completed successfully.")

    def _find_latest_result(self) -> Path:
        """Find the most recently written output image."""
        candidates = list(self.output_root.rglob("*.png")) + list(self.output_root.rglob("*.jpg"))
        if not candidates:
            raise FileNotFoundError(f"No output found in {self.output_root}")
        return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]

    def run(
        self,
        person_img: Image.Image,
        cloth_img: Image.Image,
        cloth_mask: Image.Image,
        agnostic_img: Image.Image,
        agnostic_mask: Image.Image,
        densepose_img: Image.Image,
        object_mask: Optional[np.ndarray] = None,
    ) -> Image.Image:
        """
        Run the full StableVITON inference pipeline.

        Args:
            person_img: Original person image (PIL RGB).
            cloth_img: Target clothing image (PIL RGB).
            cloth_mask: Binary mask of the clothing (PIL L).
            agnostic_img: Agnostic person image (PIL RGB).
            agnostic_mask: Agnostic mask (PIL L).
            densepose_img: DensePose visualization (PIL RGB).
            object_mask: Optional object mask for V10 patching (np.ndarray 0/255).

        Returns:
            Try-on result image (PIL RGB).
        """
        self._prepare_test_sample(
            person_img, cloth_img, cloth_mask,
            agnostic_img, agnostic_mask, densepose_img,
            object_mask=object_mask,
        )

        self._run_subprocess()

        result_path = self._find_latest_result()
        result_img = Image.open(result_path).convert("RGB")

        print(f"[TryOnInference] Result: {result_path} ({result_img.size})")
        return result_img
