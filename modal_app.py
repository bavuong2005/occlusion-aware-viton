"""
Modal Deployment — StableVITON Virtual Try-On.

Defines the Modal App, container image, and GPU function for
deploying the try-on pipeline as a serverless Gradio web app.

Usage:
    # Deploy as a web endpoint
    modal deploy modal_app.py

    # Test locally via Modal (health-check only)
    modal run modal_app.py

    # Serve interactively (dev mode — stops on Ctrl+C)
    modal serve modal_app.py

Notes on Modal v1.4 API changes:
    - modal.Mount removed from top-level; use include_source=True instead.
    - mounts= parameter removed from @app.function(); replaced by include_source=True.
    - include_source=True auto-uploads the local project code into /root in the container.
    - extra_options= replaces inline CLI flags in .pip_install().
"""

import modal

# ─── Modal Image Definition ─────────────────────────────────────────
#
# This builds a container image with all dependencies pre-installed.
# Model weights are NOT baked into the image (too large); they live
# in the Modal Volume and are downloaded once via upload_checkpoints().

image = (
    modal.Image.debian_slim(python_version="3.10")
    # System dependencies
    .apt_install(
        "git", "wget",
        "libgl1-mesa-glx", "libglib2.0-0",  # OpenCV deps
    )
    # Core ML dependencies — PyTorch CUDA 12.1 build
    .pip_install(
        "torch==2.1.0",
        "torchvision==0.16.0",
        extra_options="--index-url https://download.pytorch.org/whl/cu121",
    )
    # StableVITON dependencies
    .pip_install(
        "pytorch-lightning==1.9.5",
        "diffusers==0.21.4",
        "accelerate==0.24.1",
        "safetensors>=0.4.1",
        "transformers>=4.38,<4.45",   # 4.38+ required for grounding-dino support
        "huggingface-hub<0.25.0",
        "einops",
        "omegaconf",
        "open_clip_torch",
        "kornia",
        "clean-fid",
        "ipython",
        "test-tube",
        "albumentations<1.4.0",
    )
    # Preprocessing dependencies
    .pip_install(
        "opencv-python-headless<4.9.0",
        "pillow",
        "numpy>=1.24,<2.0",
        "scipy",
        "scikit-image",
        "av",
    )
    # SAM
    .pip_install("segment-anything")
    # Gradio for the web UI (pinned to v3.x; v4 has breaking changes)
    # Pinned fastapi and pydantic to avoid "unhashable type: 'dict'" template bugs in Gradio 3.x
    .pip_install("gradio>=3.38,<4.0", "fastapi==0.104.1", "pydantic==1.10.13", "starlette==0.27.0")
    # Detectron2 (DensePose dependency — must be built from source)
    .run_commands(
        "pip install 'git+https://github.com/facebookresearch/detectron2.git'",
        "git clone https://github.com/facebookresearch/detectron2.git /detectron2_repo"
    )
    # Pre-download SegFormer weights into image (avoids runtime HuggingFace download)
    .run_commands(
        'python -c "'
        "from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation; "
        "AutoImageProcessor.from_pretrained('mattmdjaga/segformer_b2_clothes'); "
        "AutoModelForSemanticSegmentation.from_pretrained('mattmdjaga/segformer_b2_clothes')"
        '"'
    )
    # Pre-download Grounding DINO weights
    .run_commands(
        'python -c "'
        "from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection; "
        "AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-tiny'); "
        "AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-tiny')"
        '"'
    )
    # Mount local project source code into /root in the container
    .add_local_dir(
        ".",
        remote_path="/root",
        ignore=["venv", ".git", "checkpoints", "runtime", "__pycache__", ".gradio", "flagged"],
    )
)

app = modal.App("stableviton-tryon", image=image)

# ─── Volumes ─────────────────────────────────────────────────────────
#
# Persistent storage for large model weights (StableVITON checkpoint,
# SAM checkpoint, DensePose weights) that are too large to bake into the image.

checkpoint_volume = modal.Volume.from_name(
    "stableviton-checkpoints",
    create_if_missing=True,
)


# ─── Helper: Upload checkpoints to volume ────────────────────────────

@app.function(
    volumes={"/checkpoints": checkpoint_volume},
    timeout=3600,   # 1 hour — downloading large checkpoints can be slow
)
def upload_checkpoints():
    """
    One-time setup: download model checkpoints to the Modal Volume.

    Run with:
        modal run modal_app.py::upload_checkpoints
    """
    import os
    import subprocess

    ckpt_dir = "/checkpoints/stablevton/ckpts"
    os.makedirs(ckpt_dir, exist_ok=True)

    # StableVITON checkpoint (~5.7GB)
    ckpt_path = os.path.join(ckpt_dir, "VITONHD.ckpt")
    if not os.path.exists(ckpt_path):
        print("Downloading StableVITON checkpoint from HuggingFace...")
        subprocess.run([
            "python", "-c",
            "from huggingface_hub import hf_hub_download; "
            "hf_hub_download("
            "  repo_id='rlawjdghek/StableVITON',"
            "  filename='ckpts/VITONHD.ckpt',"
            "  local_dir='/checkpoints/stablevton',"
            "  local_dir_use_symlinks=False,"
            ")"
        ], check=True)
        print("Checkpoint downloaded.")
    else:
        print("Checkpoint already exists, skipping.")

    # SAM ViT-H checkpoint (~2.4GB)
    sam_path = "/checkpoints/sam_vit_h_4b8939.pth"
    if not os.path.exists(sam_path):
        print("Downloading SAM ViT-H checkpoint...")
        subprocess.run([
            "wget", "-q",
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "-O", sam_path,
        ], check=True)
        print("SAM checkpoint downloaded.")
    else:
        print("SAM checkpoint already exists, skipping.")

    # DensePose model weights (~243MB)
    densepose_path = "/checkpoints/densepose_model.pkl"
    if not os.path.exists(densepose_path):
        print("Downloading DensePose weights...")
        subprocess.run([
            "wget", "-q",
            "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl",
            "-O", densepose_path,
        ], check=True)
        print("DensePose weights downloaded.")
    else:
        print("DensePose weights already exist, skipping.")

    checkpoint_volume.commit()
    print("All checkpoints ready!")


# ─── GPU Function ────────────────────────────────────────────────────
#
# include_source=True (Modal v1.4+): auto-uploads local project code
# into /root inside the container. No manual mounts needed.

@app.function(
    gpu="T4",                           # 16GB VRAM — change to "A10G" for faster inference
    timeout=600,                        # 10 min max per request
    volumes={"/checkpoints": checkpoint_volume},
    memory=16384,                       # 16GB system RAM
    include_source=True,                # Auto-mounts local code into /root
)
def run_tryon_gpu(
    person_bytes: bytes,
    cloth_bytes: bytes,
    preserve_objects: list[str],
    preserve_arms: bool = True,
    blend_edges: bool = True,
    category: str = "Upper-body",
    use_pipeline_enhancements: bool = True,
) -> bytes:
    """
    Run the full try-on pipeline on a Modal GPU instance.

    Args:
        person_bytes: Person image as bytes.
        cloth_bytes: Cloth image as bytes.
        preserve_objects: List of objects to preserve.
        preserve_arms: Whether to restore original arms to fix DensePose occlusion.
        blend_edges: Whether to smooth-blend restored objects.

    Returns:
        Result image as PNG bytes.
    """
    import io
    import sys

    # include_source=True mounts project code at /root
    sys.path.insert(0, "/root")

    from pathlib import Path
    from PIL import Image
    from pipeline import run_full_pipeline

    person_img = Image.open(io.BytesIO(person_bytes)).convert("RGB")
    cloth_img = Image.open(io.BytesIO(cloth_bytes)).convert("RGB")

    output = run_full_pipeline(
        person_img=person_img,
        cloth_img=cloth_img,
        stableviton_dir=Path("/root/StableVITON"),
        checkpoint_path=Path("/checkpoints/stablevton/ckpts/VITONHD.ckpt"),
        runtime_dir=Path("/tmp/runtime"),
        preserve_objects=preserve_objects,
        preserve_arms=preserve_arms,
        blend_edges=blend_edges,
        category=category,
        use_pipeline_enhancements=use_pipeline_enhancements,
        sam_checkpoint="/checkpoints/sam_vit_h_4b8939.pth",
        densepose_cfg="/detectron2_repo/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml",
        densepose_weights="/checkpoints/densepose_model.pkl",
        detectron2_densepose_dir="/detectron2_repo/projects/DensePose",
    )

    # Serialize result image to bytes for transfer
    buf = io.BytesIO()
    output["result_img"].save(buf, format="PNG")
    return buf.getvalue()


# ─── Gradio Web Endpoint ─────────────────────────────────────────────
#
# Runs as a CPU-only container that serves the Gradio UI.
# GPU work is offloaded to run_tryon_gpu() via .remote().

@app.function(
    timeout=600,
    include_source=True,                # Auto-mounts local code into /root
)
@modal.asgi_app()
def gradio_app():
    """
    Serve the Gradio UI as a Modal web endpoint.

    The UI calls run_tryon_gpu() remotely for GPU work.
    """
    import io
    import gradio as gr
    from PIL import Image

    def handle_tryon(person_image, cloth_image, preserve_list, custom_objects, preserve_arms, blend_edges, category, mode_radio):
        if person_image is None or cloth_image is None:
            raise gr.Error("Please upload both images.")

        if not isinstance(person_image, Image.Image):
            person_image = Image.fromarray(person_image)
        if not isinstance(cloth_image, Image.Image):
            cloth_image = Image.fromarray(cloth_image)

        # Serialize images to bytes for cross-container transfer
        person_buf = io.BytesIO()
        person_image.save(person_buf, format="PNG")

        cloth_buf = io.BytesIO()
        cloth_image.save(cloth_buf, format="PNG")

        objects_to_preserve = list(preserve_list)
        if custom_objects:
            custom_items = [x.strip() for x in custom_objects.split(",") if x.strip()]
            objects_to_preserve.extend(custom_items)

        # Call GPU function on a separate GPU container
        result_bytes = run_tryon_gpu.remote(
            person_bytes=person_buf.getvalue(),
            cloth_bytes=cloth_buf.getvalue(),
            preserve_objects=objects_to_preserve,
            preserve_arms=preserve_arms,
            blend_edges=blend_edges,
            category=category,
            use_pipeline_enhancements=(mode_radio == "Our Pipeline (Occlusion Preserved)"),
        )

        result_img = Image.open(io.BytesIO(result_bytes))
        return result_img

    with gr.Blocks(
        title="StableVITON Virtual Try-On",
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="blue"),
        css=".gradio-container { max-width: 1200px !important; }"
    ) as demo:
        gr.HTML("""
            <h1 style="text-align:center">👗 StableVITON Virtual Try-On</h1>
            <p style="text-align:center;color:#666">
                Upload a person image and a clothing image to see the virtual try-on result.
            </p>
        """)

        with gr.Row():
            # Left column: Inputs and Settings
            with gr.Column(scale=1):
                with gr.Row():
                    person_input = gr.Image(label="Person Image", type="pil", height=300)
                    cloth_input = gr.Image(label="Clothing Image", type="pil", height=300)

                with gr.Accordion("Advanced Options", open=True):
                    mode_radio = gr.Radio(
                        choices=["Our Pipeline (Occlusion Preserved)", "StableVITON Baseline (No Protection)"],
                        value="Our Pipeline (Occlusion Preserved)",
                        label="Pipeline Mode",
                    )
                    category_radio = gr.Radio(
                        choices=["Upper-body", "Lower-body"],
                        value="Upper-body",
                        label="Garment Category",
                    )
                    preserve_objects_cb = gr.CheckboxGroup(
                        choices=["bag", "cat", "dog", "cell phone", "book"],
                        value=["bag"],
                        label="🎒 Objects to preserve (occluders)",
                    )
                    custom_objects_txt = gr.Textbox(
                        placeholder="e.g. laptop, cup, umbrella...",
                        label="Other objects (comma separated)",
                    )
                    preserve_arms_cb = gr.Checkbox(value=False, label="💪 Preserve arms/hands (Fix DensePose errors)")
                    blend_edges_cb = gr.Checkbox(value=True, label="✨ Smooth blending")

                run_btn = gr.Button("🚀 Run Try-On", variant="primary", size="lg")

            # Right column: Output Result
            with gr.Column(scale=1):
                result_output = gr.Image(label="Try-On Result", type="pil", height=650)

        run_btn.click(
            fn=handle_tryon,
            inputs=[person_input, cloth_input, preserve_objects_cb, custom_objects_txt, preserve_arms_cb, blend_edges_cb, category_radio, mode_radio],
            outputs=[result_output],
        )

    return demo.app


# ─── Local entry point ───────────────────────────────────────────────

@app.local_entrypoint()
def main():
    """Health check — confirms Modal app parses correctly."""
    print("Modal app configured correctly for Modal v1.4+.")
    print("Next steps:")
    print("  1. Upload checkpoints: modal run modal_app.py::upload_checkpoints")
    print("  2. Dev mode:           modal serve modal_app.py")
    print("  3. Deploy:             modal deploy modal_app.py")
