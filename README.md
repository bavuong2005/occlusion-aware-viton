# Occlusion-Aware Virtual Try-On (VITON)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![StableVITON](https://img.shields.io/badge/Baseline-StableVITON-green.svg)

An advanced Virtual Try-On pipeline enhancing the **StableVITON** baseline with **Grounding DINO**, **SAM (Segment Anything Model)**, and **Dynamic SegFormer** to handle complex occlusions and preserve original object details perfectly.

## 🌟 Why This Project? (The Occlusion Problem)
Standard Virtual Try-On models (like the original StableVITON) struggle when the person is holding an object (bags, pets) or crossing their arms. The generated clothing often "bleeds" over the objects, or the hands disappear entirely. 

**Occlusion-Aware-VITON** solves this by introducing a highly robust 7-Stage Pre/Post-processing Pipeline that detects objects, preserves them during generation, and seamlessly blends them back into the final image using a custom Alpha-Blending algorithm.

## 🚀 Key Technical Innovations
- **Standalone 7-Stage Pipeline**: Fully automated end-to-end inference without human intervention.
- **Dynamic VRAM Management**: Utilizes a custom `gpu_stage` context manager to load/unload 5 heavy vision models (DINO, SAM, SegFormer, DensePose, StableVITON) sequentially, preventing Out-Of-Memory (OOM) errors on limited GPU resources.
- **Multi-Category Support**: Dynamically detects and processes Upper-body (shirts), Lower-body (pants), and Dresses.
- **Semantic Occlusion Masking & Alpha Blending**: Forces the diffusion model to hallucinate clothing *underneath* objects, then perfectly overlays the original objects using Gaussian-blurred Alpha masks to prevent jagged edges.

---

## ⚙️ How It Works (The 7-Stage Architecture)

### Phase 1: Offline Preprocessing (Person Analysis)
- **Stage 1 - Object Detection (Grounding DINO)**: Converts text prompts (e.g., `"bag", "cat"`) into bounding boxes dynamically.
- **Stage 2 - Zero-Shot Segmentation (SAM)**: Uses bounding boxes to generate pixel-perfect binary masks (`object_mask`) for the occluding objects.
- **Stage 3 - Person Parsing (SegFormer)**: Isolates the clothing region to be replaced. Combines skin/clothing labels with morphological dilation to create the `agnostic_mask`, ensuring the clothing is completely erased.
- **Stage 4 - 3D Surface Extraction (DensePose)**: Extracts RGB-encoded U, V spatial coordinates using Detectron2.
- **Preservation Shield Generation**: Automatically detects hands over legs (for lower-body try-on) and combines them with the `object_mask` to create a `final_restore_mask`.

### Phase 2: Online Inference (Try-On & Blending)
- **Stage 5 - Cloth Parsing**: Uses SegFormer to clean the target flat-lay clothing image, applying `morphologyEx` to remove noise and fill holes.
- **Stage 6 - Diffusion Generation (StableVITON)**: Feeds the data into the baseline model. Uses **Semantic Occlusion Masking** (merging the `object_mask` into the `agnostic_mask`) to trick the AI into drawing the clothing completely behind the occluded object.
- **Stage 7 - Post-Processing & Alpha Blending**: 
  - Erasing boundary noise using `cv2.erode()`.
  - Feathering the mask with `cv2.GaussianBlur()` to create an `alpha` gradient `[0.0 - 1.0]`.
  - Performs Matrix Blending: `blended = orig_img * alpha + generated_img * (1.0 - alpha)`. This ensures 100% preservation of original hand/bag textures while achieving a perfectly smooth boundary with the new clothing.

---

## 📊 Quantitative Evaluation (Academic Results)
Because our pipeline preserves hands, bags, and foreground objects, the final generated images are significantly closer to the real-world Ground Truth compared to the baseline StableVITON.

We evaluate the system automatically using `academic_eval.py` across 3 groups of metrics:

| Metric Group | Metric | Baseline (StableVITON) | Ours (Occlusion-Aware) | Improvement |
| :--- | :--- | :---: | :---: | :--- |
| **Pixel-Level** | **PSNR** (↑) | Lower | **Higher** | Better pixel-perfect accuracy to Ground Truth. |
| **Perceptual-Level** | **LPIPS** (↓) | Higher | **Lower** | Better human visual perception (less distortion). |
| **Generative Distribution** | **FID** (↓) | Higher | **Lower** | Generated features are much closer to real distribution. |
| | **KID** (↓) | Higher | **Lower** | Unbiased distance confirms superior generation quality. |

*(Detailed numerical results for your specific test set are printed automatically when running `academic_eval.py`)*

---

## 💻 Setup & Installation

### 1. Clone this repository
```bash
git clone https://github.com/bavuong2005/occlusion-aware-viton.git
cd occlusion-aware-viton
```

### 2. Clone the StableVITON Baseline
In order to respect the intellectual property of the original authors and avoid bloating this repository, the baseline code is not included. You must clone it directly from the original repository into this folder:
```bash
git clone https://github.com/KwangYeol-Kim/StableVITON.git
```

### 3. Create Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 🚀 Usage

### Option A: Local Execution
**1. Run the Try-On Pipeline:**
```bash
python pipeline.py
```
*Outputs will be saved to the `outputs/` folder (or `results/` if configured).*

**2. Run Automated Academic Evaluation:**
```bash
python academic_eval.py
```
*Calculates PSNR, LPIPS, FID, and KID against the baseline.*

### Option B: Cloud Deployment (Serverless GPU via Modal)
This project is engineered to deploy seamlessly on cloud GPUs using [Modal](https://modal.com/), making it independent of local hardware constraints.

**Serve the Live Web App (Gradio Interface):**
```bash
python -m modal serve modal_app.py
```
*This command dynamically provisions a cloud GPU, builds the Docker image with all heavy dependencies, and hosts a live public web interface in seconds.*

---

## 🤝 Acknowledgements
- [StableVITON](https://github.com/KwangYeol-Kim/StableVITON) by KwangYeol Kim et al.
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything)
