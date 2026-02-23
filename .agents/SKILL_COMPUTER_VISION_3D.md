---
name: Computer Vision 3D Expert
description: Expert knowledge on 3D Point Cloud Segmentation, PointNet++ Architecture, and Deep Learning Training Workflows.
---

# 🧠 SKILL: Computer Vision 3D & Point Cloud Deep Learning

This skill encapsulates expert knowledge for developing, training, and optimizing 3D Semantic Segmentation models, specifically focused on **PointNet++** for aerial photogrammetry (fused LIDAR/RGB).

## 1. Core Philosophy: "Resolution Sync" 📏

The single most critical lesson from the V5/V6 evolution is **Resolution Synchronization**.

*   **The Trap**: Training on high-density data (e.g., 0.10m, 10k points) and inferring on lower-density production data (e.g., 0.25m). This creates a "Domain Gap" where the model learns micro-textures that don't exist in production.
*   **The Fix (V6)**: Downsample training data to match the **exact density** of production data.
    *   If production is **0.25m**, training blocks (10x10m) must leverage ~**2048 points** (not 10,000).
    *   This forces the model to learn robust geometric features rather than overfitting to density.

## 2. Model Architecture: PointNet++ MSG 🏗️

We use **PointNet++ with Multi-Scale Grouping (MSG)** to handle non-uniform densities.

### Inputs (9 Channels)
Do **NOT** blindly feed all available data. Feature engineering is key:
*   **Geometric**: `X, Y, Z` (Normalized per block).
*   **Visual**: `R, G, B` (Normalized [0-1]).
*   **Surface**: `Nx, Ny, Nz` (Normal Vectors). essential for distinguishing flat ground from vertical machinery sides.
*   **❌ Excluded**: Absolute Height ($Z$) or "Height above ground". While useful for analysis, including it as a feature causes the model to overfit to specific terrain elevations.

### Architecture Nuances
*   **Set Abstraction Layers**: Hierarchical feature extraction.
*   **Radius Parameters**: Crucial. `base_radius=3.5m` has proven optimal for large machinery (Bulldozers/Trucks). Smaller radii miss the context of "big objects".
*   **NoVerticality Variant**: A specialized variant that ignores Z-axis scaling/transforms to preserve the "upright" nature of gravity-aligned objects (Validation V5).

## 3. Data Preprocessing Pipeline ⚙️

### Gridding (The "Block" Strategy)
Point clouds are too massive for GPU memory. We chop them into **10m x 10m** vertical columns.
*   **Buffer/Context**: Blocks are cut *without* padding for training efficiency, but inference must handle edge cases (usually by processing slightly larger blocks or overlapping).
*   **Format**: `.npy` files containing `[N, C]` matrices are faster to load than parsing `.las` files repeatedly.

### Augmentation (The "Jitter" Secret)
To make the model robust to sensor noise:
*   **Jitter**: Add random gaussian noise (`sigma=0.005`) to XYZ coordinates.
*   **Rotation**: Random rotation around Z-axis (0-360°) is mandatory for direction-invariant detection.
*   **Scaling**: Random scaling (0.9x - 1.1x) to handle slight size variations.

## 4. Training Workflow & Hyperparameters 🔥

### Class Imbalance
In mining environments, "Ground" is 95% of data; "Machinery" is 5%.
*   **Weighted Cross Entropy**: Apply heavy weights to the minority class.
    *   *Recipe*: `Class Weights: [1.0 (Ground), 15.0 - 20.0 (Machinery)]`.
    *   Too high (e.g., 50.0) causes false positives (hallucinations).
    *   Too low (e.g., 1.0) causes the model to ignore machinery entirely.

### Optimization
*   **Optimizer**: Adam (`lr=0.001` is the sweet spot).
*   **Scheduler**: CosineAnnealingLR (Starts high, decays to 0). Helps settle into minima.
*   **Batch Size**: Maximize correct GPU usage. For RTX 4090/5090, `Batch=64` is standard.
*   **Mixed Precision (FP16)**: Mandatory for speed. Little to no loss in accuracy for this task.

## 5. Typical Files & Locations
*   `src/models/pointnet2.py`: The model definition.
*   `TRAIN_V[X].py`: Main training loop.
*   `src/data/dataset_v[X].py`: Data loaders (check for `__getitem__` logic).
*   `configs/`: YAML configuration files. Always use config files over hardcoding.

## 6. Debugging Model Performance 📉
If metrics (IoU) are low:
1.  **Check IoU per Class**: Is "Ground" 99% but "Machinery" 0%? $\rightarrow$ Increase Class Weights.
2.  **Check Normalization**: Are inputs actually normalized to [0-1] or [-1, 1]?
3.  **Check Visuals**: Use WandB 3D visualization. If prediction points are scattered randomly, the model isn't converging (check LR). If predictions are shifted, check coordinate normalization logic.
