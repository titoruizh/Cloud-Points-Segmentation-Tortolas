# Cloud Point Research V2 — GeoAI Technical Portfolio

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![WandB](https://img.shields.io/badge/WandB-Experiment%20Tracking-orange.svg)](https://wandb.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Advanced Binary Segmentation (Machinery vs. Ground) on Photogrammetric Point Clouds.**

---

## 📌 Overview

**Cloud Point Research V2** is a technical R&D project focused on solving complex semantic segmentation challenges in geospatial data. The core objective is to accurately identify and segment machinery within large-scale photogrammetric point clouds, distinguishing them from the terrain (ground).

This repository serves as a **technical portfolio** demonstrating proficiency in:
- **3D Deep Learning**: Implementation and fine-tuning of PointNet++ MSG.
- **Data-Centric AI**: Strategies like "Resolution Sync" to eliminate domain gaps.
- **MLOps**: Full experiment tracking, hyperparameter sweeps, and model versioning using Weights & Biases.
- **Production Engineering**: Optimization for high-throughput inference (Torch Compile, FP16) and deployment via a Gradio interface.

> **Note:** This project is a showcase of technical capabilities and research methodology, not a standalone production package.

---

## 🚀 Key Features & Highlights

*   **Robust Architecture**: Utilizes **PointNet++ MSG** (Multi-Scale Grouping) with a 9-channel input (XYZ + RGB + Normals) to capture fine geometric details.
*   **Resolution Synchronization**: Implements a rigorous `0.25m` density synchronization between training and inference data to maximize model generalization.
*   **Handling Class Imbalance**: Addresses the scarcity of machinery points (vs. vast ground areas) using:
    *   Dynamic Class Weights (e.g., [1.0, 15.0]).
    *   Targeted Oversampling during data loading.
*   **Optimized Inference Pipeline**:
    *   Spatial Blocking (10x10m chunks) for consistent geometry.
    *   Vectorized gridding and on-the-fly normal calculation.
    *   Optimized for **RTX 5090** (CUDA, FP16 support).
*   **Interactive Demo**: Includes a `Gradio` based web application for visualizing inference results in real-time.

---

## 📊 Results & Performance

The model (V6) has achieved significant accuracy in distinguishing machinery from the terrain, validated on challenging test scenes.

**Key Metrics (Validation V6):**
| Metric | Score | Note |
| :--- | :--- | :--- |
| **mIoU** | **93.06%** | Mean Intersection over Union |
| **IoU Machinery** | **87.67%** | High precision on target class |
| **IoU Ground** | **98.46%** | Robust terrain rejection |
| **Val Loss** | 0.0227 | Stable convergence |

### Visual Comparisons (RGB vs. Segmentation)

**Scene 1**
| RGB Input | AI Segmentation |
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/a584df28-2917-4167-a05f-20556c8de400" width="100%"> | <img src="https://github.com/user-attachments/assets/f1ebb32e-e2d4-46c4-829d-bb398ad27c96" width="100%"> |

**Scene 2**
| RGB Input | AI Segmentation |
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/43a6eb27-c57d-46ab-b539-cc4895ea850b" width="100%"> | <img src="https://github.com/user-attachments/assets/b0492563-e778-49b0-8b78-a33faa36da00" width="100%"> |

*The segmentation output clearly delineates the machinery (red/color) from the surrounding ground, even in complex terrains.*

### Interactive Demo (Gradio App)

The project includes a web-based interface for easy testing and visualization of the model's performance on new data.

<img width="1046" height="467" alt="Gradio Interface" src="https://github.com/user-attachments/assets/570dd147-8202-4ba0-b075-9de12265bd68" />

For a deep dive into the technical details, see the [Technical Report V6](docs/TECHNICAL_REPORT_V6.md).

---

## 🛠 Project Structure

The codebase is organized to separate data processing, modeling, and experimentation configuration.

```
.
├── configs/                # YAML Configuration files (Versioned)
│   └── pointnet2/          # Model-specific configs (e.g., v6_0.25m.yaml)
├── src/                    # Source code
│   ├── data/               # Custom Datasets (V3-V6) & Loaders
│   ├── models/             # PyTorch Architectures (PointNet2, RandLANet)
│   └── utils/              # Metrics (IoU), Visualization, & Helpers
├── scripts/                # Utility scripts (Inference, Pre-processing)
├── docs/                   # Documentation & Technical Reports
├── TRAIN_V6.py             # Main training entry point
├── main_inference_app.py   # Gradio Web UI for inference
└── requirements.txt        # Project dependencies
```

---

## 💻 Getting Started

While this is a portfolio project, the following steps outline how to reproduce the environment and run the code.

### Prerequisites
*   Python 3.8+
*   CUDA-capable GPU (Recommended for training)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cloud-point-research-v2.git
cd cloud-point-research-v2

# Install dependencies
pip install -r requirements.txt
```

### Usage Examples

**1. Training (Reproduce V6)**
```bash
python3 TRAIN_V6.py --config configs/pointnet2/pointnet2_v6_0.25m.yaml
```

**2. Batch Inference**
```bash
python3 scripts/inference/infer_pointnet_v6.py \
  --input_file "data/raw_test/RGB/input.laz" \
  --checkpoint "checkpoints/best_model_v6.pth" \
  --output_file "data/predictions/output.laz" \
  --batch_size 64
```

**3. Interactive Web App**
```bash
# Launches the Gradio interface
python3 main_inference_app.py
```

---

## 👤 Author

**Tito Ruiz Haros**
*   **Role**: GeoAI Researcher & Developer
*   **Focus**: Computer Vision, 3D Deep Learning, MLOps

---
*Built with ❤️ using PyTorch & WandB.*
