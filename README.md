# Occlusion-Aware Virtual Try-On (VITON)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![StableVITON](https://img.shields.io/badge/Baseline-StableVITON-green.svg)

An advanced Virtual Try-On pipeline enhancing the **StableVITON** baseline with **Grounding DINO**, **SAM (Segment Anything Model)**, and **Dynamic SegFormer** to handle complex occlusions and preserve original object details perfectly.

## 🌟 Key Features
- **End-to-End Inference Pipeline**: Integrates Grounding DINO and SAM to establish robust object grounding and zero-shot segmentation during image generation.
- **Occlusion Handling**: Resolves complex occlusion artifacts (e.g., preserving hands, bags, and foreground objects) by implementing a custom Alpha Blending workflow.
- **Dynamic VRAM Management**: Optimizes memory usage using an intelligent GPU Context Manager, allowing seamless sequential execution of multiple heavy vision models on limited GPU resources.
- **Automated Academic Evaluation Suite**: Benchmarks generated image quality automatically across Pixel-level (PSNR), Perceptual-level (LPIPS), and Distribution-level (FID, KID) metrics.

## 🚀 Setup & Installation

### 1. Clone this repository
```bash
git clone https://github.com/bavuong2005/occlusion-aware-viton.git
cd occlusion-aware-viton
```

### 2. Clone the StableVITON Baseline
In order to respect the intellectual property of the original authors and avoid bloating this repository, the baseline code is not included. You must clone it directly from the original repository:
```bash
# Clone StableVITON into this directory
git clone https://github.com/KwangYeol-Kim/StableVITON.git
```

### 3. Create Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

## 📊 Evaluation Metrics
This project outperforms the original StableVITON on scenarios with occlusions:
- **PSNR (Peak Signal-to-Noise Ratio)**: Measures pixel-level accuracy.
- **LPIPS**: Evaluates human perceptual image quality.
- **FID / KID**: Measures the feature distribution distance between generated and real images.

## 💻 Usage
To run the evaluation pipeline on the test dataset:
```bash
python pipeline.py
```
To run the automated academic metric calculation (PSNR, LPIPS, FID, KID):
```bash
python academic_eval.py
```

## 🤝 Acknowledgements
- [StableVITON](https://github.com/KwangYeol-Kim/StableVITON) by KwangYeol Kim et al.
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything)
