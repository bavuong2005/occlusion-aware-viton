"""
Pipeline Orchestrator — Full Virtual Try-On Pipeline.

Runs all preprocessing and inference stages sequentially with
stage-based GPU memory management. Each stage loads its model,
processes the input, then explicitly frees VRAM before the next
stage starts.

Pipeline flow:
    Stage 1: Object Detection  (Grounding DINO)   → bounding boxes
    Stage 2: Object Segmentation (SAM)             → object mask
    Stage 3: Person Parsing    (SegFormer)          → agnostic img/mask
    Stage 4: DensePose         (Detectron2)         → pose map
    Stage 5: Cloth Parsing     (SegFormer)          → cloth mask
    Stage 6: Try-On Inference  (StableVITON)        → raw result
    Stage 7: Post-processing   (CPU)                → final result
"""

from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from postprocess.occluder import refined_restore_occluder
from utils.image_utils import ensure_gray, resize_rgb
from utils.memory import gpu_stage


# Default image size expected by StableVITON
IMG_SIZE = (768, 1024)


def run_full_pipeline(
    person_img: Image.Image,
    cloth_img: Image.Image,
    stableviton_dir: Path,
    checkpoint_path: Path,
    runtime_dir: Path,
    preserve_objects: Optional[list[str]] = None,
    preserve_arms: bool = False,
    blend_edges: bool = True,
    sam_checkpoint: Optional[str] = None,
    densepose_cfg: Optional[str] = None,
    densepose_weights: Optional[str] = None,
    detectron2_densepose_dir: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
    category: str = "Upper-body",
    use_pipeline_enhancements: bool = True,
) -> dict:
    """
    Run the complete virtual try-on pipeline.

    Args:
        person_img: Original person image (PIL RGB).
        cloth_img: Target clothing image (PIL RGB).
        stableviton_dir: Path to StableVITON repo root.
        checkpoint_path: Path to VITONHD.ckpt.
        runtime_dir: Working directory for temp data/outputs.
        preserve_objects: List of objects to detect and restore (e.g. ["bag", "cat"]).
        preserve_arms: Whether to restore original arms (fixes DensePose occlusion errors).
        blend_edges: Whether to use smooth blending for object restoration.
        sam_checkpoint: Path to SAM checkpoint (optional override).
        densepose_cfg: Path to DensePose config (optional override).
        densepose_weights: Path to DensePose weights (optional override).
        detectron2_densepose_dir: Path to detectron2 DensePose project dir.
        progress_callback: Optional fn(stage_num, total_stages, stage_name)
                          for UI progress updates.

    Returns:
        dict with keys:
            - "result_img": final PIL.Image (RGB)
            - "debug": dict with intermediate results and metadata
    """
    results = {}
    total_stages = 7

    def _progress(stage_num: int, stage_name: str):
        if progress_callback:
            progress_callback(stage_num, total_stages, stage_name)

    # ─── Resize inputs ──────────────────────────────────────────
    person_img = resize_rgb(person_img.convert("RGB"), IMG_SIZE)
    cloth_img = resize_rgb(cloth_img.convert("RGB"), IMG_SIZE)

    if not use_pipeline_enhancements:
        preserve_objects = None
        preserve_arms = False

    if category == "Lower-body" and use_pipeline_enhancements:
        # Force preserve_arms to True for Lower-body to prevent hand deformation by pants dilation
        preserve_arms = True

    # ─── Stage 1-2: Object Detection + Segmentation ─────────────
    object_mask = None

    if preserve_objects:
        text_prompt = " . ".join(preserve_objects) + " ."
        _progress(1, f"Object Detection ({text_prompt})")
        with gpu_stage("Stage 1: Object Detection"):
            from stages.object_detection import ObjectDetectionStage

            detector = ObjectDetectionStage()
            detector.load()
            boxes, scores, labels = detector.run(person_img, text_prompt)
            detector.unload()

        if boxes is not None and len(boxes) > 0:
            _progress(2, "Object Segmentation (SAM)")
            with gpu_stage("Stage 2: Object Segmentation"):
                from stages.object_segmentation import ObjectSegmentationStage

                segmenter = ObjectSegmentationStage(
                    checkpoint_path=sam_checkpoint,
                )
                segmenter.load()
                object_mask = segmenter.run(
                    person_img, boxes, scores, labels, target_labels=preserve_objects
                )
                segmenter.unload()
        else:
            _progress(2, "Object Segmentation (skipped — no objects detected)")
            print("[Pipeline] No matching objects detected, skipping SAM.")
    else:
        _progress(1, "Object Detection (skipped)")
        _progress(2, "Object Segmentation (skipped)")
        print("[Pipeline] Object preservation disabled or empty list, skipping stages 1-2.")

    results["object_mask"] = object_mask

    # ─── Stage 3: Person Parsing ─────────────────────────────────
    _progress(3, "Person Parsing (SegFormer)")
    with gpu_stage("Stage 3: Person Parsing"):
        from stages.person_parsing import PersonParsingStage

        parser = PersonParsingStage()
        parser.load()
        parsing_results = parser.run(person_img, object_mask=object_mask, category=category)
        parser.unload()

    results.update(parsing_results)

    # ─── Stage 4: DensePose ──────────────────────────────────────
    _progress(4, "DensePose Estimation")
    with gpu_stage("Stage 4: DensePose"):
        from stages.densepose import DensePoseStage

        densepose = DensePoseStage(
            cfg_path=densepose_cfg,
            weights_path=densepose_weights,
            detectron2_densepose_dir=detectron2_densepose_dir,
        )
        densepose.load()
        results["densepose_img"] = densepose.run(person_img)
        densepose.unload()

    # ─── Stage 5: Cloth Parsing ──────────────────────────────────
    _progress(5, "Cloth Parsing (SegFormer)")
    with gpu_stage("Stage 5: Cloth Parsing"):
        from stages.cloth_parsing import ClothParsingStage

        cloth_parser = ClothParsingStage()
        cloth_parser.load()
        cloth_results = cloth_parser.run(
            cloth_img,
            person_top_labels=results.get("top_labels"),
            category=category,
        )
        cloth_parser.unload()

    results.update(cloth_results)

    # ─── Stage 6: StableVITON Inference ──────────────────────────
    _progress(6, "Try-On Inference (StableVITON)")
    # No gpu_stage context — this is the last GPU stage, cleanup not needed
    from stages.tryon_inference import TryOnInferenceStage

    tryon = TryOnInferenceStage(
        stableviton_dir=stableviton_dir,
        checkpoint_path=checkpoint_path,
        runtime_dir=runtime_dir,
    )

    # Convert cloth_mask from ndarray to PIL for the inference stage
    from PIL import Image as _PILImage
    import numpy as np

    cloth_mask_pil = _PILImage.fromarray(results["cloth_mask"]).convert("L")

    result_img = tryon.run(
        person_img=person_img,
        cloth_img=cloth_img,
        cloth_mask=cloth_mask_pil,
        agnostic_img=results["agnostic_img"],
        agnostic_mask=results["agnostic_mask"],
        densepose_img=results["densepose_img"],
        object_mask=object_mask,
    )

    # ─── Stage 7: Post-processing ────────────────────────────────
    _progress(7, "Post-processing (occluder restoration)")

    objects_restored = False
    
    final_restore_mask = None
    if preserve_objects and object_mask is not None:
        final_restore_mask = object_mask.copy()

    if preserve_arms and "parsing_map" in results:
        parsing_map = results["parsing_map"]
        # SegFormer labels: 14 = Right-arm, 15 = Left-arm
        arm_mask = np.isin(parsing_map, [14, 15]).astype(np.uint8) * 255
        
        # Clean the arm mask slightly to remove noise
        from postprocess.mask_utils import clean_mask
        arm_mask = clean_mask(arm_mask, open_k=3, close_k=5)
        
        if final_restore_mask is None:
            final_restore_mask = arm_mask
        else:
            final_restore_mask = np.maximum(final_restore_mask, arm_mask)

    if final_restore_mask is not None and (final_restore_mask > 0).any():
        object_mask_pil = _PILImage.fromarray(final_restore_mask).convert("L")
        erode_iter = 1
        blur_ksize = 5 if blend_edges else 1

        result_img = refined_restore_occluder(
            person_img, result_img, object_mask_pil,
            erode_iter=erode_iter, blur_ksize=blur_ksize,
        )
        objects_restored = True
        print("[Pipeline] Objects/Arms restored via alpha blending.")

    # ─── Build debug info ────────────────────────────────────────
    debug = {
        "objects_detected": object_mask is not None,
        "objects_restored": objects_restored,
        "top_labels_person": results.get("top_labels"),
        "cloth_labels": results.get("cloth_labels"),
    }

    # Collect intermediate images for debug gallery
    debug_images = {
        "person": person_img,
        "cloth": cloth_img,
        "agnostic": results.get("agnostic_img"),
        "agnostic_mask": results.get("agnostic_mask"),
        "densepose": results.get("densepose_img"),
        "cloth_mask": _PILImage.fromarray(results["cloth_mask"]) if "cloth_mask" in results else None,
    }
    if object_mask is not None:
        debug_images["object_mask"] = _PILImage.fromarray(object_mask)

    return {
        "result_img": result_img,
        "debug": debug,
        "debug_images": debug_images,
    }
