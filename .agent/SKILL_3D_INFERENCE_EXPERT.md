---
name: 3D Inference Expert
description: Expert knowledge on Point Cloud Inference Apps, Post-Processing Pipelines, and Production Deployment.
---

# 🔨 SKILL: 3D Inference & App Expert

This skill encapsulates expert knowledge for **running, configuring, and extending** the inference applications for Point Cloud Segmentation. It covers the full pipeline from raw LAZ input to refined DTM output.

## 1. The Inference Engine ("Nitro") 🚀

The core engine (`scripts/inference` or `app_inference/core`) is built for speed vs. large datasets.

### Key Mechanisms:
*   **Vectorized Gridding**: We don't loop over points to grid them. We use NumPy vectorization (`floor(coords / block_size)`) and Hashing (`x * 1000 + y`) to assign points to blocks instantly.
*   **Torch Compile**: The model is compiled (`torch.compile(mode='reduce-overhead')`) on the first run. This adds a ~45s startup delay but speeds up batch processing by 30-40%.
*   **Lazy Loading**: We read `.las` files using `laspy` in stream mode where possible, or memory-map inputs.

### Configuration Parameters:
*   **`--confidence` (Threshold)**: The most important knob for the user.
    *   *Default*: `0.5`
    *   *High Precision*: `0.8` (Less noise, potentially fragmented machinery).
    *   *High Recall*: `0.3` (Detects everything, but more noise/rocks mistaken for machines).
*   **`batch_size`**: Tuning this depends on VRAM. `64` is safe for 24GB VRAM.

## 2. Post-Processing Pipeline 🧹

Raw deep learning predictions are rarely perfect. We use a deterministic pipeline to clean them.

### A. FIX_TECHO (Volumetric Repair)
*   **Problem**: LIDAR/Photogrammetry often misses the sides of machines, leaving "floating roofs".
*   **Solution**: **DBSCAN Clustering**.
    1.  Identify clusters of "Machinery" points.
    2.  Calculate the Bounding Box of each cluster.
    3.  Check "Ground" points *inside* that box (under the roof).
    4.  **Flip** those ground points to "Machinery".
*   **Key Params**:
    *   `eps` (e.g., 2.5m): Maximum distance between points to be considered the same machine.
    *   `min_samples`: Minimum points to form a valid machine (filters noise).

    *   `min_samples`: Minimum points to form a valid machine (filters noise).

### C. Smart Gap Filling (Anti-Halo) 🛡️
*   **Problem**: Simple filling causes "Halo Effects" (expanding machine borders) and square artifacts in DTM.
*   **Solution**: **Quadrant Neighbor Check**.
    *   A ground point is only filled if it has machinery neighbors in **3 out of 4 spatial quadrants**.
    *   This ensures the point is geometrically *inside* the cluster, not on the edge.
*   **Performance**: MUST be vectorized (NumPy). Python loops are too slow (>20 min vs <1 min).

### D. INTERPOL (The "Digital Bulldozer")
*   **Problem**: Removing a machine leaves a "hole" in the terrain (or a pile of dirt). We need a clean DTM (Digital Terrain Model).
*   **Solution**: **IDW (Inverse Distance Weighting)**.
    1.  Delete all "Machinery" points.
    2.  For every X,Y position where a machine was, look for the nearest *true* Ground neighbors.
    3.  Interpolate the Z value based on those neighbors.
*   **Key Params**:
    *   `k_neighbors` (e.g., 12): How many ground points to average.
    *   `max_dist`: Max search radius.

## 3. The Web Application (Gradio) 🌐

The UI (`main_inference_app.py`) is a wrapper around the Core Engine and Post-Processor.

### Structure
*   `ui/app.py`: Logic for the Gradio interface. Handles state, logs, and progress bars.
*   `core/inference_engine.py`: The PyTorch runner.
*   `core/postprocess.py`: The cleaning logic.

### Expert Tips for Extension
*   **Adding a Slider**: Update `create_app()` in `ui/app.py` to add the component, then pass the value to `run_pipeline()`.
*   **New Output Type**: If adding a new export format (e.g., DXF), implement it in `core/postprocess.py` and add a checkbox in `ui/app.py`.
*   **Debugging**: The app redirects `stdout` to the log window. Use `self.log()` in the `InferenceApp` class to communicate with the user.

## 4. Validated Checkpoints 🏆

Always know which checkpoint to use for which scenario:

| Version | Resolution | Best Checkpoint (Example) | Use Case |
| :--- | :--- | :--- | :--- |
| **V6** | **0.25m** | `LR0.0010_W15...BEST_IOU.pth` | **Production**. Matches monthly photogrammetry. |
| **V5** | **0.10m** | `LR0.0010_W20...BEST_IOU.pth` | Research/High-Res. Good for detailed scans. |

**Rule of Thumb**: Match the checkpoint resolution to your input file resolution!
