import os
import json
from pathlib import Path

def create_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

def mk_markdown(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")]
    }

def mk_code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.split("\n")]
    }

def strip_local_imports(code_str):
    lines = code_str.split('\n')
    out = []
    in_multiline_import = False
    for line in lines:
        if in_multiline_import:
            if ')' in line:
                in_multiline_import = False
            continue
            
        strip_line = line.strip()
        # Handle relative imports like "from utils...", "from stages...", "from postprocess..."
        if strip_line.startswith('from utils') or strip_line.startswith('from stages') or strip_line.startswith('from postprocess'):
            if '(' in line and ')' not in line:
                in_multiline_import = True
            continue
        
        # also ignore "import utils"
        if strip_line.startswith('import utils') or strip_line.startswith('import stages') or strip_line.startswith('import postprocess'):
            continue
            
        out.append(line)
        
    return '\n'.join(out)

def read_py_file(filepath, title):
    with open(filepath, 'r', encoding='utf-8') as f:
        code = f.read()
    code = strip_local_imports(code)
    # Add a header comment
    header = f"# {'='*50}\n# {title}\n# {'='*50}\n\n"
    return mk_code(header + code)

# ─── OFFLINE PREPROCESSING NOTEBOOK ──────────────────────────────────────────

offline_cells = [
    mk_markdown("# Offline Preprocessing (Standalone)\nThis notebook is completely standalone. It contains all the required classes and functions directly in the cells. It is ready to run on Kaggle/Colab.\n\n> **Note for Kaggle/Colab users**: Please update the checkpoint paths and dataset paths in the configuration section to point to your uploaded datasets."),
    
    mk_code("""# Required Pip Installs (Uncomment if running on Colab/Kaggle)
# !pip install -q transformers diffusers accelerate opencv-python segment-anything
# !pip install -q git+https://github.com/IDEA-Research/GroundingDINO.git
# !pip install -q git+https://github.com/facebookresearch/detectron2.git"""),

    mk_code("""import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import cv2

# Configuration - UPDATE THESE PATHS FOR KAGGLE/COLAB
person_img_path = "../inputs/test_person.jpg"
output_dir = Path("../outputs/preprocessed")
output_dir.mkdir(parents=True, exist_ok=True)

sam_checkpoint = "../checkpoints/sam_vit_h_4b8939.pth"
densepose_cfg = "../detectron2_repo/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml"
densepose_weights = "../checkpoints/densepose_model.pkl"
densepose_repo = "../detectron2_repo/projects/DensePose"

# Pipeline Config
preserve_objects = ["bag", "cat"] 
preserve_arms = False
category = "Upper-body" # Upper-body, Lower-body, Dress

print("Setup complete.")"""),

    mk_markdown("## Utilities"),
    read_py_file("utils/memory.py", "utils/memory.py"),
    read_py_file("utils/image_utils.py", "utils/image_utils.py"),
    read_py_file("postprocess/mask_utils.py", "postprocess/mask_utils.py"),

    mk_markdown("## Stage 1 & 2: Object Detection and Segmentation"),
    read_py_file("stages/object_detection.py", "stages/object_detection.py"),
    read_py_file("stages/object_segmentation.py", "stages/object_segmentation.py"),

    mk_markdown("## Stage 3: Person Parsing"),
    read_py_file("stages/person_parsing.py", "stages/person_parsing.py"),

    mk_markdown("## Stage 4: DensePose Estimation"),
    read_py_file("stages/densepose.py", "stages/densepose.py"),

    mk_markdown("## Execution\nNow we run the pipeline using the classes defined above."),
    mk_code("""
person_img = Image.open(person_img_path).convert("RGB")
IMG_SIZE = (768, 1024)
person_img = resize_rgb(person_img, IMG_SIZE)
display(person_img.resize((384, 512)))

object_mask = None
if preserve_objects:
    text_prompt = " . ".join(preserve_objects) + " ."
    print(f"Detecting: {text_prompt}")
    
    with gpu_stage("Stage 1: Object Detection"):
        detector = ObjectDetectionStage()
        detector.load()
        boxes, scores, labels = detector.run(person_img, text_prompt)
        detector.unload()
    
    if boxes is not None and len(boxes) > 0:
        with gpu_stage("Stage 2: Object Segmentation"):
            segmenter = ObjectSegmentationStage(checkpoint_path=sam_checkpoint)
            segmenter.load()
            object_mask = segmenter.run(person_img, boxes, scores, labels, target_labels=preserve_objects)
            segmenter.unload()
            if object_mask is not None:
                display(Image.fromarray(object_mask).resize((384, 512)))

with gpu_stage("Stage 3: Person Parsing"):
    parser = PersonParsingStage()
    parser.load()
    parsing_results = parser.run(person_img, object_mask=object_mask, category=category)
    parser.unload()

parsing_map = parsing_results["parsing_map"]
agnostic_img = parsing_results["agnostic_img"]
agnostic_mask = parsing_results["agnostic_mask"]
top_labels = parsing_results["top_labels"]

display(agnostic_img.resize((384, 512)))
display(agnostic_mask.resize((384, 512)))

with gpu_stage("Stage 4: DensePose"):
    densepose = DensePoseStage(cfg_path=densepose_cfg, weights_path=densepose_weights, detectron2_densepose_dir=densepose_repo)
    densepose.load()
    densepose_img = densepose.run(person_img)
    densepose.unload()

display(densepose_img.resize((384, 512)))

# Save
final_restore_mask = None
if preserve_objects and object_mask is not None:
    final_restore_mask = object_mask.copy()

if category == "Lower-body":
    preserve_arms = True

if preserve_arms:
    arm_mask = np.isin(parsing_map, [14, 15]).astype(np.uint8) * 255
    arm_mask = clean_mask(arm_mask, open_k=3, close_k=5)
    if final_restore_mask is None:
        final_restore_mask = arm_mask
    else:
        final_restore_mask = np.maximum(final_restore_mask, arm_mask)

person_img.save(output_dir / "person_img.png")
agnostic_img.save(output_dir / "agnostic_img.png")
agnostic_mask.save(output_dir / "agnostic_mask.png")
densepose_img.save(output_dir / "densepose_img.png")
np.save(output_dir / "top_labels.npy", top_labels)

if final_restore_mask is not None and (final_restore_mask > 0).any():
    Image.fromarray(final_restore_mask).save(output_dir / "final_restore_mask.png")
    
print("Offline preprocessing completed and saved to disk.")
""")
]


# ─── ONLINE INFERENCE NOTEBOOK ───────────────────────────────────────────

online_cells = [
    mk_markdown("# Online Inference (Standalone)\nThis notebook is completely standalone. It contains all the required classes and functions directly in the cells. It takes the pre-computed person data from the offline stage and a target clothing image to generate the final try-on result.\n\n> **Note for Kaggle/Colab users**: Please update the checkpoint paths and dataset paths in the configuration section to point to your uploaded datasets."),
    
    mk_code("""import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import cv2

# Configuration - UPDATE THESE PATHS FOR KAGGLE/COLAB
cloth_img_path = "../inputs/test_cloth.jpg"
preprocessed_dir = Path("../outputs/preprocessed")

stableviton_dir = "../StableVITON"
viton_checkpoint = "../checkpoints/stablevton/ckpts/VITONHD.ckpt"
runtime_dir = "../tmp/runtime"

category = "Upper-body" # Upper-body, Lower-body, Dress

print("Setup complete.")"""),

    mk_markdown("## Utilities"),
    read_py_file("utils/memory.py", "utils/memory.py"),
    read_py_file("utils/image_utils.py", "utils/image_utils.py"),
    read_py_file("postprocess/mask_utils.py", "postprocess/mask_utils.py"),

    mk_markdown("## Stage 5: Cloth Parsing"),
    read_py_file("stages/cloth_parsing.py", "stages/cloth_parsing.py"),

    mk_markdown("## Stage 6: Try-On Inference"),
    read_py_file("stages/tryon_inference.py", "stages/tryon_inference.py"),

    mk_markdown("## Stage 7: Post-processing (Alpha Blending)"),
    read_py_file("postprocess/occluder.py", "postprocess/occluder.py"),

    mk_markdown("## Execution\nLoad data and run the inference pipeline."),
    mk_code("""
cloth_img = Image.open(cloth_img_path).convert("RGB")
IMG_SIZE = (768, 1024)
cloth_img = resize_rgb(cloth_img, IMG_SIZE)
display(cloth_img.resize((384, 512)))

person_img = Image.open(preprocessed_dir / "person_img.png").convert("RGB")
agnostic_img = Image.open(preprocessed_dir / "agnostic_img.png").convert("RGB")
agnostic_mask = Image.open(preprocessed_dir / "agnostic_mask.png").convert("L")
densepose_img = Image.open(preprocessed_dir / "densepose_img.png").convert("RGB")
top_labels = np.load(preprocessed_dir / "top_labels.npy").tolist()

restore_mask_path = preprocessed_dir / "final_restore_mask.png"
final_restore_mask_pil = Image.open(restore_mask_path).convert("L") if restore_mask_path.exists() else None

with gpu_stage("Stage 5: Cloth Parsing"):
    cloth_parser = ClothParsingStage()
    cloth_parser.load()
    cloth_results = cloth_parser.run(cloth_img, person_top_labels=top_labels, category=category)
    cloth_parser.unload()

cloth_mask_np = cloth_results["cloth_mask"]
cloth_mask_pil = Image.fromarray(cloth_mask_np).convert("L")
display(cloth_mask_pil.resize((384, 512)))

# Stage 6: Try-On Inference
tryon = TryOnInferenceStage(
    stableviton_dir=Path(stableviton_dir),
    checkpoint_path=Path(viton_checkpoint),
    runtime_dir=Path(runtime_dir)
)

result_img = tryon.run(
    person_img=person_img,
    cloth_img=cloth_img,
    cloth_mask=cloth_mask_pil,
    agnostic_img=agnostic_img,
    agnostic_mask=agnostic_mask,
    densepose_img=densepose_img,
    object_mask=None 
)

display(result_img.resize((384, 512)))

# Stage 7: Post-processing
if final_restore_mask_pil is not None:
    result_img = refined_restore_occluder(person_img, result_img, final_restore_mask_pil, erode_iter=1, blur_ksize=5)

display(result_img.resize((384, 512)))
result_img.save("../outputs/final_tryon_result.png")
print("Try-on complete! Result saved.")
""")
]

os.makedirs('d:/study/3rd-year-2/CS338/final-cs338/notebooks', exist_ok=True)

with open('d:/study/3rd-year-2/CS338/final-cs338/notebooks/offline_preprocessing.ipynb', 'w', encoding='utf-8') as f:
    json.dump(create_notebook(offline_cells), f, indent=1)

with open('d:/study/3rd-year-2/CS338/final-cs338/notebooks/online_inference.ipynb', 'w', encoding='utf-8') as f:
    json.dump(create_notebook(online_cells), f, indent=1)

print("Notebooks created successfully.")
