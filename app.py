"""
StableVITON Virtual Try-On — Gradio Web Interface.

New all-in-one interface: upload person + cloth images directly,
get the try-on result. No more zip file workflow.

Replaces the old app_online.py.
"""

from pathlib import Path

import gradio as gr
from PIL import Image

from pipeline import run_full_pipeline


# ─── Path Configuration ─────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent

STABLEVITON_DIR = BASE_DIR / "StableVITON"
CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "stablevton" / "ckpts" / "VITONHD.ckpt"
RUNTIME_DIR = BASE_DIR / "runtime"

# Optional model paths (set these for local/Modal deployment)
SAM_CHECKPOINT = None        # e.g. "/path/to/sam_vit_h_4b8939.pth"
DENSEPOSE_CFG = None         # e.g. "/path/to/densepose_rcnn_R_50_FPN_s1x.yaml"
DENSEPOSE_WEIGHTS = None     # e.g. "/path/to/densepose_model.pkl"
DETECTRON2_DENSEPOSE_DIR = None  # e.g. "/path/to/detectron2/projects/DensePose"

RUNTIME_DIR.mkdir(parents=True, exist_ok=True)


# ─── Pipeline Runner ────────────────────────────────────────────────

def run_tryon(
    person_image,
    cloth_image,
    preserve_bag: bool = True,
    blend_edges: bool = True,
    show_debug: bool = False,
    mode_radio: str = "Our Pipeline (Occlusion Preserved)",
    progress=gr.Progress(track_tqdm=True),
):
    """
    Main Gradio handler: run the full try-on pipeline.

    Args:
        person_image: Uploaded person image (from gr.Image).
        cloth_image: Uploaded clothing image (from gr.Image).
        preserve_bag: Whether to detect and restore bags/accessories.
        blend_edges: Whether to smooth-blend restored objects.
        show_debug: Whether to return intermediate debug images.
        progress: Gradio progress tracker.

    Returns:
        (result_image, debug_gallery_or_json)
    """
    if person_image is None:
        raise gr.Error("Please upload a person image.")
    if cloth_image is None:
        raise gr.Error("Please upload a clothing image.")

    # Convert to PIL if needed (Gradio may pass numpy arrays)
    if not isinstance(person_image, Image.Image):
        person_image = Image.fromarray(person_image)
    if not isinstance(cloth_image, Image.Image):
        cloth_image = Image.fromarray(cloth_image)

    def progress_callback(stage_num, total_stages, stage_name):
        progress((stage_num - 1) / total_stages, desc=f"Stage {stage_num}/{total_stages}: {stage_name}")

    # Run pipeline
    output = run_full_pipeline(
        person_img=person_image,
        cloth_img=cloth_image,
        stableviton_dir=STABLEVITON_DIR,
        checkpoint_path=CHECKPOINT_PATH,
        runtime_dir=RUNTIME_DIR,
        preserve_bag=preserve_bag,
        blend_edges=blend_edges,
        sam_checkpoint=SAM_CHECKPOINT,
        densepose_cfg=DENSEPOSE_CFG,
        densepose_weights=DENSEPOSE_WEIGHTS,
        detectron2_densepose_dir=DETECTRON2_DENSEPOSE_DIR,
        progress_callback=progress_callback,
        use_pipeline_enhancements=(mode_radio == "Our Pipeline (Occlusion Preserved)"),
    )

    result_img = output["result_img"]
    debug_info = output["debug"]

    # Save final result
    final_path = RUNTIME_DIR / "final.png"
    result_img.save(final_path)

    # Build debug gallery if requested
    if show_debug:
        debug_imgs = output.get("debug_images", {})
        gallery = []
        for name, img in debug_imgs.items():
            if img is not None:
                gallery.append((img, name))
        return result_img, gallery, debug_info

    return result_img, [], debug_info


# ─── Gradio UI ───────────────────────────────────────────────────────

def create_demo() -> gr.Blocks:
    """Build the Gradio Blocks interface."""

    with gr.Blocks(
        title="StableVITON Virtual Try-On",
        theme=gr.themes.Soft(),
        css="""
            .main-title {
                text-align: center;
                margin-bottom: 0.5em;
            }
            .subtitle {
                text-align: center;
                color: #666;
                margin-bottom: 1.5em;
            }
        """,
    ) as demo:
        gr.HTML("""
            <h1 class="main-title">👗 StableVITON Virtual Try-On</h1>
            <p class="subtitle">Upload a person image and a clothing image to see the virtual try-on result.</p>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                person_input = gr.Image(
                    label="Person Image",
                    type="pil",
                    height=400,
                    elem_id="person-upload",
                )
            with gr.Column(scale=1):
                cloth_input = gr.Image(
                    label="Clothing Image",
                    type="pil",
                    height=400,
                    elem_id="cloth-upload",
                )

        with gr.Row():
            mode_radio = gr.Radio(
                choices=["Our Pipeline (Occlusion Preserved)", "StableVITON Baseline (No Protection)"],
                value="Our Pipeline (Occlusion Preserved)",
                label="Pipeline Mode",
            )
            preserve_bag_cb = gr.Checkbox(
                value=True,
                label="🎒 Preserve bag/accessories",
                elem_id="preserve-bag",
            )
            blend_edges_cb = gr.Checkbox(
                value=True,
                label="✨ Smooth blending",
                elem_id="blend-edges",
            )
            show_debug_cb = gr.Checkbox(
                value=False,
                label="🔍 Show debug stages",
                elem_id="show-debug",
            )

        run_btn = gr.Button(
            "🚀 Run Try-On",
            variant="primary",
            size="lg",
            elem_id="run-btn",
        )

        with gr.Row():
            with gr.Column(scale=2):
                result_output = gr.Image(
                    label="Try-On Result",
                    type="pil",
                    height=500,
                    elem_id="result-output",
                )
            with gr.Column(scale=1):
                debug_gallery = gr.Gallery(
                    label="Debug Stages",
                    columns=2,
                    height=400,
                    elem_id="debug-gallery",
                    visible=True,
                )
                debug_json = gr.JSON(
                    label="Pipeline Info",
                    elem_id="debug-json",
                )

        run_btn.click(
            fn=run_tryon,
            inputs=[
                person_input,
                cloth_input,
                preserve_bag_cb,
                blend_edges_cb,
                show_debug_cb,
                mode_radio,
            ],
            outputs=[result_output, debug_gallery, debug_json],
        )

    return demo


# ─── Entry Point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=True)
