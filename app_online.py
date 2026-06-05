import json
import shutil
import zipfile
import subprocess
from pathlib import Path
import sys
import os

import cv2
import numpy as np
from PIL import Image
import gradio as gr


# =========================
# BASE PATH (PORTABLE)
# =========================
BASE_DIR = Path(__file__).resolve().parent

# Cho phép chạy cả local + Kaggle
if str(BASE_DIR).startswith("/kaggle"):
    print("Running on Kaggle")
else:
    print("Running locally")


STABLEVITON_ROOT = BASE_DIR / "StableVITON"
CKPT_PATH = BASE_DIR / "checkpoints" / "stablevton" / "ckpts" / "VITONHD.ckpt"
CONFIG_PATH = STABLEVITON_ROOT / "configs" / "VITONHD.yaml"

RUNTIME_DIR = BASE_DIR / "runtime"
EXTRACT_DIR = RUNTIME_DIR / "offline_assets"
DATA_ROOT = RUNTIME_DIR / "data" / "viton_hd"
OUTPUT_ROOT = RUNTIME_DIR / "outputs"

RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# =========================
# UTILS
# =========================
def reset_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def ensure_rgb(path: Path):
    return Image.open(path).convert("RGB")


def ensure_gray(path: Path):
    return Image.open(path).convert("L")


# =========================
# ZIP LOADING
# =========================
def extract_zip(zip_path: str) -> Path:
    zip_path = Path(zip_path)

    if not zip_path.exists():
        raise FileNotFoundError(f"Không tìm thấy zip: {zip_path}")

    reset_dir(EXTRACT_DIR)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(EXTRACT_DIR)

    meta_files = list(EXTRACT_DIR.rglob("meta.json"))
    if not meta_files:
        raise FileNotFoundError("Không tìm thấy meta.json trong zip")

    return meta_files[0]


def load_meta_from_zip(zip_path: str):
    meta_path = extract_zip(zip_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    artifact_root = meta_path.parents[2]
    return artifact_root, meta


# =========================
# MASK + BLENDING
# =========================
def feather_mask(mask_pil: Image.Image, ksize: int = 15):
    mask = np.array(mask_pil.convert("L")).astype(np.uint8)
    mask = (mask > 127).astype(np.uint8) * 255

    if ksize % 2 == 0:
        ksize += 1

    mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
    return (mask.astype(np.float32) / 255.0)[..., None]


def restore_occluder(original_person_pil, result_pil, object_mask_pil, feather_ksize=15):
    # resize original về đúng size output
    original = np.array(
        original_person_pil.resize(result_pil.size, Image.BILINEAR)
    ).astype(np.float32)

    result = np.array(result_pil).astype(np.float32)

    # resize mask
    object_mask_pil = object_mask_pil.resize(
        result_pil.size,
        Image.NEAREST
    )

    alpha = feather_mask(object_mask_pil, feather_ksize)

    blended = original * alpha + result * (1.0 - alpha)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)


# =========================
# PREPARE DATA FOR VITON
# =========================
def prepare_viton_test_sample(
    person_img,
    cloth_img,
    cloth_mask,
    agnostic_img,
    agnostic_mask,
    densepose_img,
    sample_name="00000_00.jpg",
):
    test_root = DATA_ROOT / "test"

    folders = [
        "image",
        "cloth",
        "cloth-mask",
        "agnostic-mask",
        "agnostic-v3.2",
        "image-densepose",
    ]

    if DATA_ROOT.exists():
        shutil.rmtree(DATA_ROOT)

    for folder in folders:
        (test_root / folder).mkdir(parents=True, exist_ok=True)

    mask_name = sample_name.replace(".jpg", "_mask.png")

    person_img.save(test_root / "image" / sample_name)
    cloth_img.save(test_root / "cloth" / sample_name)
    cloth_mask.save(test_root / "cloth-mask" / sample_name)
    agnostic_img.save(test_root / "agnostic-v3.2" / sample_name)
    agnostic_mask.save(test_root / "agnostic-mask" / mask_name)
    densepose_img.save(test_root / "image-densepose" / sample_name)

    with open(DATA_ROOT / "test_pairs.txt", "w") as f:
        f.write(f"{sample_name} {sample_name}\n")


# =========================
# RUN MODEL
# =========================
def run_stableviton():
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    python_exe = sys.executable

    cmd = [
        python_exe,
        "inference.py",
        "--config_path", str(CONFIG_PATH),
        "--batch_size", "1",
        "--model_load_path", str(CKPT_PATH),
        "--save_dir", str(OUTPUT_ROOT),
        "--data_root_dir", str(DATA_ROOT),
    ]

    print("Running StableVITON...")
    print("CMD:", " ".join(cmd))

    result = subprocess.run(
        cmd,
        cwd=str(STABLEVITON_ROOT),
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError("StableVITON failed")


def find_latest_result():
    candidates = list(OUTPUT_ROOT.rglob("*.png")) + list(OUTPUT_ROOT.rglob("*.jpg"))
    if not candidates:
        raise FileNotFoundError("Không tìm thấy output")

    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


# =========================
# MAIN PIPELINE
# =========================
def run_pipeline(zip_file, preserve_bag=True, blend_edges=True):
    if zip_file is None:
        raise gr.Error("Upload offline_assets.zip")

    # FIX BUG Gradio path
    if hasattr(zip_file, "name"):
        zip_path = zip_file.name
    else:
        zip_path = str(zip_file)

    artifact_root, meta = load_meta_from_zip(zip_path)

    # Load paths
    person_img = ensure_rgb(artifact_root / meta["person_image"])
    cloth_img = ensure_rgb(artifact_root / meta["cloth_image"])
    agnostic_img = ensure_rgb(artifact_root / meta["agnostic"])
    agnostic_mask = ensure_gray(artifact_root / meta["agnostic_mask"])
    densepose_img = ensure_rgb(artifact_root / meta["densepose"])
    cloth_mask = ensure_gray(artifact_root / meta["cloth_mask"])

    prepare_viton_test_sample(
        person_img,
        cloth_img,
        cloth_mask,
        agnostic_img,
        agnostic_mask,
        densepose_img,
    )

    run_stableviton()

    result_path = find_latest_result()
    result_img = Image.open(result_path).convert("RGB")

    # ===== restore bag =====
    bag_mask = meta.get("object_masks", {}).get("bag", None)

    if preserve_bag and bag_mask:
        bag_path = artifact_root / bag_mask
        if bag_path.exists():
            k = 15 if blend_edges else 1
            result_img = restore_occluder(
                person_img,
                result_img,
                Image.open(bag_path).convert("L"),
                k
            )

    final_path = OUTPUT_ROOT / "final.png"
    result_img.save(final_path)

    debug = {
        "person_id": meta.get("person_id"),
        "cloth_id": meta.get("cloth_id"),
        "bag_restored": bool(bag_mask),
    }

    return str(final_path), debug


# =========================
# UI
# =========================
demo = gr.Interface(
    fn=run_pipeline,
    inputs=[
        gr.File(file_types=[".zip"]),
        gr.Checkbox(value=True, label="Preserve bag"),
        gr.Checkbox(value=True, label="Smooth blending"),
    ],
    outputs=[
        gr.Image(type="filepath"),
        gr.JSON(),
    ],
    title="StableVITON Demo",
)


if __name__ == "__main__":
    demo.launch(share=True)