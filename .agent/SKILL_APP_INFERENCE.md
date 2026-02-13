---
name: App Inference Expert
description: Deep expertise in the Point Cloud Inference Application - Architecture, Pipeline, UI, Performance Optimization, and Advanced Configuration.
---

# 🎯 SKILL: App Inference Expert - Production System Mastery

This skill encapsulates comprehensive knowledge of the **Point Cloud Inference Application** (`main_inference_app.py`), covering the full stack from entry point to production deployment.

## 1. Application Architecture 🏗️

### Entry Point: `main_inference_app.py`

The application follows a **CLI-orchestrated Gradio web interface** pattern:

```
main_inference_app.py (Launcher)
    ↓
    ├─ Dependency Verification
    ├─ sys.path Configuration (PROJECT_ROOT)
    ├─ CLI Arguments (--port, --share, --no-check)
    ├─ Banner Display
    └─ launch_app() → Gradio Server (0.0.0.0:7860)
```

**Critical Design Patterns:**
- ✅ **Lazy Imports**: Core modules imported AFTER sys.path setup
- ✅ **Fail-Fast Validation**: Dependencies checked before UI launch
- ✅ **Public Share**: Gradio's share flag for tunnel access
- ✅ **Output Directory Pre-creation**: Avoids runtime errors

### Core Module Structure:

```
app_inference/
├── ui/
│   ├── app.py              # Main Gradio interface (InferenceApp class)
│   ├── components.py       # Reusable UI elements
│   └── styles.py           # Custom CSS theming
├── core/
│   ├── inference_engine.py # PyTorch model wrapper + GPU optimizations
│   ├── postprocess.py      # FIX_TECHO + INTERPOL pipeline
│   └── validators.py       # LAZ/LAS file validation
├── config/
│   └── default_config.yaml # Default parameters
└── utils/                  # (Helper functions)
```

---

## 2. UI Layer (`app_inference/ui/app.py`) 🎨

### The `InferenceApp` Class

**State Management:**
```python
class InferenceApp:
    - validator: PointCloudValidator      # File validation instance
    - inference_engine: InferenceEngine   # Model executor (lazy-initialized)
    - postprocessor: PostProcessor        # Pipeline cleaner
    - log_messages: List[str]             # Rolling log (last 50 messages)
    - all_checkpoints: List[Tuple]        # Scanned checkpoint metadata
```

**Key Methods:**

| Method | Purpose | Critical Details |
|--------|---------|------------------|
| `_scan_checkpoints()` | Scans `checkpoints/` recursively | Returns: `(display, path, version, is_winner)` |
| `_format_display()` | Parses checkpoint filename for metadata | Extracts: LR, W, R via regex |
| `get_checkpoints_for_version()` | Filters by V5/V6 | Auto-fallback to all if none match |
| `get_default_checkpoint()` | Returns winner or BEST_IOU | Priority: Winner > BEST_IOU > First |
| `validate_files()` | Pre-flight check | Requires RGB, min 1000 pts |
| `run_pipeline()` | **Main execution orchestrator** | 3-stage: Inference → FIX_TECHO → INTERPOL |
| `_generate_results_html()` | Pretty HTML results panel | Cards + file list + metrics |

### Dual Version System (V5/V6)

```python
VERSION_CONFIG = {
    "V5 (0.10m)": {"num_points": 10000, "suffix": "_PointnetV5", "filter": "V5"},
    "V6 (0.25m)": {"num_points": 2048,  "suffix": "_PointnetV6", "filter": "V6"}
}

WINNER_CHECKPOINTS = {
    "V5": "LR0.0010_W20_J0.005_R3.5_BEST_IOU.pth",  # 🏆
    "V6": "LR0.0010_W15_J0.005_R3.5_BEST_IOU.pth",  # 🏆
}
```

**UI Behavior:**
- Version selector (`gr.Radio`) triggers checkpoint dropdown update
- Default checkpoint auto-selected based on version
- 🏆 icon highlights production-validated models

### Pipeline Orchestration Flow

```
run_pipeline() sequence:
1. Validate files (RGB check)
2. Configure InferenceConfig(batch_size, num_points, use_compile)
3. Load model checkpoint
4. Configure postprocessors (FixTechoConfig, InterpolConfig)
5. For each file:
   a. INFERENCE → _PointnetV{5|6}.laz
   b. FIX_TECHO → _techos.laz (if requested)
   c. INTERPOL  → _DTM.laz (if requested)
6. Clean intermediate files (if not exported)
7. Generate HTML results
```

**Output Selection Logic:**
```python
export_classified = "📊 Clasificado" in output_types
export_techos    = "🏗️ Corregido (Techos)" in output_types
export_dtm       = "🌍 DTM (Interpolado)" in output_types
```

---

## 3. Inference Engine (`inference_engine.py`) 🚀

### Core Optimizations for RTX 5090

```python
class InferenceEngine:
    OPTIMIZATIONS:
    - torch.set_float32_matmul_precision('high')  # TensorCore optimization
    - PYTORCH_ALLOC_CONF="expandable_segments:True"  # Avoid fragmentation
    - FP16 Mixed Precision (torch.amp.autocast)
    - torch.compile (optional, +25% speed, 60s warmup)
    - Persistent DataLoader workers
    - Pin memory for faster CPU→GPU transfers
```

### Feature Extraction Pipeline

**The 9-channel input tensor:**
```
[XYZ, RGB, Normals] = 3 + 3 + 3 = 9
```

**Critical Flow:**
```python
1. Read LAZ with laspy
2. Extract XYZ (raw coordinates)
3. Extract RGB → normalize to [0-1]
   - Auto-detect scale: 65535 (16-bit) or 255 (8-bit)
4. Extract/Compute Normals:
   ✅ FAST PATH: Read native normals from LAZ (if exist)
   🔥 SLOW PATH: Compute via GPU (compute_normals_gpu, k=30, radius=R)
```

**Why normals are critical:**
> Ground is flat (normals point up). Machinery has vertical sides (normals point horizontally). This geometric distinction is what the model learns.

### Gridding Strategy (Block-based Inference)

**Problem:** Point clouds have millions of points. Can't fit in GPU memory.

**Solution:** Divide into 10m × 10m vertical columns.

```python
Vectorized Gridding (NumPy):
    grid_x = floor((xyz[:, 0] - min_x) / 10.0)
    grid_y = floor((xyz[:, 1] - min_y) / 10.0)
    hash = grid_x * 100000 + grid_y
    
    # Group by hash (fast sorting + unique)
    groups = split by unique hashes
    
    # Filter blocks with <50 points (noise)
```

**Per-block normalization:**
```python
# Center to tile origin
block[:, 0] -= (tile_origin_x + 5.0)  # Half block size
block[:, 1] -= (tile_origin_y + 5.0)
block[:, 2] -= min(block[:, 2])       # Ground-relative height
```

### DataLoader Configuration

```python
DataLoader(
    dataset=GridDatasetNitro,
    batch_size=64,              # Tunable: 16-256+ (depends on VRAM)
    num_workers=12,             # CPU cores for pre-loading
    pin_memory=True,            # Faster transfers
    persistent_workers=True,    # Keep workers alive between epochs
    prefetch_factor=2           # Pre-load 2 batches ahead
)
```

**Performance Tuning:**
- **Low VRAM (<16GB):** batch_size=32, num_workers=4
- **High VRAM (32GB+):** batch_size=128-256, num_workers=16
- **Bottleneck Check:** If GPU util <80%, increase batch_size

### Inference Loop (FP16 Optimized)

```python
with torch.no_grad():
    for batch_data, batch_indices in dataloader:
        batch_data = batch_data.to(device, non_blocking=True)
        
        with autocast(device_type='cuda'):  # FP16
            logits = model(xyz, batch_data)
            probs = softmax(logits, dim=1)[:, 1, :]  # Class 1 prob
        
        # Store results in global array
        global_probs[indices] = probs.cpu().numpy()
```

**Confidence Thresholding:**
```python
preds = (global_probs > confidence).astype(uint8)
# confidence=0.5: Balanced
# confidence=0.8: High precision (less false positives)
# confidence=0.3: High recall (catch everything)
```

---

## 4. Post-Processing Pipeline 🧹

### A. FIX_TECHO (Volumetric Roof Filling)

**The Problem:**
> Photogrammetry/LIDAR often misses the sides of machinery, leaving isolated "roof" clusters floating above the ground.

**The Solution:** 3D Bounding Box Filling

```python
ALGORITHM:
1. Extract machinery points (class=1)
2. DBSCAN clustering (eps=2.5m, min_samples=30)
3. For each cluster:
   a. Compute 3D BBox (min_x, min_y, min_z, max_x, max_y, max_z)
   b. Define fill region:
      - XY: [min_x - padding, max_x + padding]
      - Z:  [min_z + z_buffer, min_z + max_height]
   c. Find ground points inside this volume
   d. Refine with 2D KDTree (proximity check)
   e. Flip to class=1 (machinery)
```

**Key Parameters:**

| Param | Purpose | Default | Tuning |
|-------|---------|---------|--------|
| `eps` | DBSCAN radius | 2.5m | Smaller for small machines |
| `min_samples` | Min cluster size | 30 | Higher to filter noise |
| `z_buffer` | Floor protection | 1.5m | Prevents filling ground under machine |
| `max_height` | Ceiling limit | 8.0m | Match tallest expected machine |
| `padding` | XY margin | 1.5m | Expand search zone |

### B. Smart Gap Filling (Anti-Halo)

**The Problem:**
> Naive filling causes "halos" (expanding machine borders) and square artifacts.

**The Solution:** Quadrant Neighbor Check (Vectorized)

```python
ALGORITHM:
1. Build KDTree of existing machinery
2. For each ground point near machinery:
   a. Find neighbors within merge_radius (e.g., 2.5m)
   b. Check if count >= merge_neighbors (e.g., 4)
   c. **Quadrant Analysis:**
      - Q1: (+X, +Y)  [bit 0]
      - Q2: (-X, +Y)  [bit 1]
      - Q3: (-X, -Y)  [bit 2]
      - Q4: (+X, -Y)  [bit 3]
   d. Compute bitwise OR of occupied quadrants
   e. If popcount >= 3: Point is INSIDE cluster → Fill
   f. Else: Edge point → Keep as ground
```

**Vectorization (CRITICAL for speed):**
```python
# NumPy approach (1000x faster than Python loop):
diff = neighbors_xy - query_xy
quadrant_mask = compute_quadrant_bits(diff)  # Vectorized
accumulated = bitwise_or.at(reduce_indices, quadrant_mask)
popcount = count_bits(accumulated)
valid = popcount >= 3
```

**Parameters:**

| Param | Effect | Default |
|-------|--------|---------|
| `smart_merge` | Enable/disable | `True` |
| `merge_radius` | Search distance | 2.5m |
| `merge_neighbors` | Min neighbors | 4 |

### C. INTERPOL (Digital Bulldozer - DTM Generation)

**The Goal:** Generate a clean Digital Terrain Model by removing all machinery.

**Algorithm:** Inverse Distance Weighting (IDW)

```python
1. Extract ONLY ground points (class=2)
2. Build KDTree of ground
3. For each machinery position (X, Y):
   a. Query k_neighbors nearest ground points
   b. Compute distances: d_i
   c. Compute weights: w_i = 1 / d_i²
   d. Interpolate: Z = Σ(w_i * z_i) / Σ(w_i)
4. Replace machinery with interpolated ground
```

**Parameters:**

| Param | Purpose | Default |
|-------|---------|---------|
| `k_neighbors` | Averaging window | 12 |
| `max_dist` | Search limit | 50m |

**Edge Cases:**
- If no ground within `max_dist`: Use median Z of all ground
- For large excavations: Increase `k_neighbors` to smooth

---

## 5. Model (PointNet++) 🧠

### Input Format

```python
forward(xyz, features):
    # xyz: [B, N, 3] (raw coords, transformed in forward)
    # features: [B, N, 9] (XYZ + RGB + Normals)
    
    # Internal: Permute to [B, C, N] for Conv1D
```

### Architecture Summary

```
INPUT [B, N, 9]
    ↓
Set Abstraction Hierarchy:
    SA1: 1024 pts, r=0.5R → [B, 64, 1024]
    SA2: 256 pts,  r=1.0R → [B, 128, 256]
    SA3: 64 pts,   r=2.0R → [B, 256, 64]
    SA4: 16 pts,   r=4.0R → [B, 512, 16]
    ↓
Feature Propagation (Upsampling):
    FP4, FP3, FP2, FP1 → [B, 128, N]
    ↓
Classification Head:
    Conv1D(128→128) → BN → Dropout(0.5)
    Conv1D(128→2)
    ↓
OUTPUT: [B, 2, N] (Logits per point)
```

**Critical Parameter: `base_radius`**
- Extracted from checkpoint filename (e.g., `_R3.5_`)
- Scales all SA layer radii
- **3.5m is optimal for large machinery** (bulldozers, trucks)

---

## 6. Configuration System ⚙️

### `default_config.yaml`

```yaml
model:
  d_in: 9                # XYZ + RGB + Normals
  num_classes: 2         # Ground vs Machinery

inference:
  batch_size: 64         # Safe for 24GB VRAM
  num_points: 10000      # V5 default
  use_compile: true      # +25% speed (60s warmup)

fix_techo:
  eps: 2.5
  z_buffer: 1.5
  max_height: 8.0
  padding: 1.5
  min_samples: 30

interpol:
  k_neighbors: 12
  max_dist: 50.0

system:
  device: "cuda"
  num_workers: 12
  float32_matmul_precision: "high"
```

**Override Hierarchy:**
1. UI Sliders (highest priority)
2. CLI arguments
3. default_config.yaml
4. Hardcoded defaults

---

## 7. Performance Optimization Guide 🔥

### Batch Size Tuning

**Rule of Thumb:**
```
batch_size = (VRAM_GB - 4) * 8
```

**Empirical Values:**

| GPU | VRAM | Safe Batch | Aggressive |
|-----|------|------------|------------|
| RTX 3060 | 12GB | 32 | 48 |
| RTX 4090 | 24GB | 64 | 128 |
| RTX 5090 | 32GB | 96 | 192-256 |

**Symptoms of Incorrect Batch Size:**
- **Too high:** OOM crash, CUDA errors
- **Too low:** GPU util <50%, slow processing
- **Optimal:** GPU util 90-100%, steady throughput

**Benchmark Command:**
```bash
python tests/batch_benchmark.py \
    --input data/raw/test.laz \
    --checkpoint checkpoints/.../BEST_IOU.pth \
    --batches 64 96 128 160 192 224 256 \
    --num_points 2048
```

### torch.compile Trade-offs

**Pros:**
- +20-30% throughput (after warmup)
- Better kernel fusion

**Cons:**
- 45-90s compilation on first run
- Slightly higher memory usage

**When to disable:**
- Single-file inference (overhead not worth it)
- Development/debugging (recompiles on code change)

**When to enable:**
- Batch processing (100+ files)
- Production deployment

### DataLoader Workers

**Formula:**
```
num_workers = min(CPU_cores * 0.75, 16)
```

**Bottleneck Detection:**
```python
# If GPU utilization oscillates (e.g., 100% → 20% → 100%):
#   → Increase num_workers (data loading is slow)
# If GPU utilization steady at 100%:
#   → Workers are fine
```

---

## 8. Validation System 🔍

### `PointCloudValidator`

**Required Checks:**
```python
✅ File exists
✅ Extension: .las or .laz
✅ RGB channels present (MANDATORY)
✅ Point count >= 1000
⚠️  Normals present (warning if missing)
```

**Why RGB is mandatory:**
> The model is trained on RGB+XYZ+Normals. Without RGB, you'd feed zeros, causing degraded performance.

**ValidationResult:**
```python
@dataclass
class ValidationResult:
    is_valid: bool
    file_name: str
    point_count: int
    has_rgb: bool
    has_normals: bool
    rgb_range: str       # "0-255" or "0-65535"
    errors: List[str]
    warnings: List[str]
```

---

## 9. UI Design System 🎨

### Color Scheme (Dark Mode)

```css
Primary:   #60a5fa (Blue)
Success:   #10b981 (Green)
Warning:   #fbbf24 (Yellow)
Error:     #ef4444 (Red)
BG Dark:   #1e293b
BG Card:   #334155
Text:      #f1f5f9
```

### Component Patterns

**File Upload:**
```python
gr.File(
    label="📁 Archivos LAZ/LAS (con RGB)",
    file_count="multiple",
    file_types=[".las", ".laz"]
)
```

**Output Type Selector:**
```python
gr.CheckboxGroup(
    choices=["📊 Clasificado", "🏗️ Corregido", "🌍 DTM"],
    value=["📊 Clasificado", "🏗️ Corregido", "🌍 DTM"]
)
```

**Parameter Accordion:**
```python
with gr.Accordion("⚙️ Parámetros", open=False):
    # Hides complexity by default
```

---

## 10. Debugging & Troubleshooting 🔧

### Common Issues

**1. "Model not converging / Random predictions"**
```
DIAGNOSIS: Check log for "Cargando modelo" message
FIX: Verify checkpoint path, ensure state_dict loads correctly
```

**2. "OOM (Out of Memory)"**
```
DIAGNOSIS: CUDA out of memory error
FIX: Reduce batch_size (e.g., 64 → 32)
```

**3. "GPU utilization very low (<40%)"**
```
DIAGNOSIS: CPU bottleneck
FIX: Increase num_workers (e.g., 12 → 16)
     OR increase batch_size
```

**4. "Missing normals warning but inference still runs"**
```
STATUS: Normal. compute_normals_gpu will compute them (slower).
OPTIMIZATION: Pre-compute normals and save in LAZ for faster loading.
```

**5. "FIX_TECHO not filling anything"**
```
DIAGNOSIS: DBSCAN params too strict
FIX: Reduce eps (e.g., 2.5 → 2.0) or min_samples (30 → 20)
```

**6. "Smart Merge too aggressive (filling ground)"**
```
DIAGNOSIS: merge_neighbors too low
FIX: Increase merge_neighbors (e.g., 4 → 6)
```

---

## 11. Extension Patterns 🛠️

### Adding a New Output Format

**Example: Export to DXF**

1. **Add to PostProcessor:**
```python
# postprocess.py
def run_export_dxf(self, input_file, output_file, callback):
    las = laspy.read(input_file)
    # ... DXF conversion logic
    return PostprocessResult(...)
```

2. **Add UI Checkbox:**
```python
# app.py
output_types = gr.CheckboxGroup(
    choices=["📊 Clasificado", "🏗️ Corregido", "🌍 DTM", "📐 DXF"],
    ...
)
```

3. **Update Pipeline:**
```python
# app.py → run_pipeline()
if "📐 DXF" in output_types:
    self.postprocessor.run_export_dxf(...)
```

### Adding a New Parameter

**Example: Add "noise_filter" threshold**

1. **Add Slider:**
```python
noise_filter = gr.Slider(0, 100, value=50, label="Noise Filter")
```

2. **Pass to Pipeline:**
```python
run_btn.click(
    fn=app_instance.run_pipeline,
    inputs=[..., noise_filter],  # Add here
    ...
)
```

3. **Update Method Signature:**
```python
def run_pipeline(self, ..., noise_filter):
    # Use noise_filter in inference or postprocess
```

---

## 12. Production Deployment 🚀

### Recommended Setup

**Hardware:**
- GPU: RTX 4090/5090 (24-32GB VRAM)
- CPU: 16+ cores
- RAM: 64GB+
- Storage: NVMe SSD (fast LAZ I/O)

**Software:**
```bash
# Docker recommended (not yet implemented)
# For now: Virtual environment
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Launch:**
```bash
python main_inference_app.py \
    --port 8080 \
    --share  # For remote access
```

**Reverse Proxy (nginx):**
```nginx
location / {
    proxy_pass http://localhost:7860;
    proxy_set_header Host $host;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
}
```

---

## 13. Key Takeaways 📌

1. **Resolution Sync:** Always match model version (V5/V6) to input data density
2. **Batch Size:** Maximize GPU usage without OOM (monitoring is key)
3. **Normals:** Native normals = 10x faster than GPU compute
4. **Smart Merge:** Quadrant check prevents halo artifacts
5. **torch.compile:** Essential for batch processing (ignore for single files)
6. **Validation:** RGB is mandatory, normals are highly recommended
7. **Pipeline Flexibility:** Output types are independent, can mix and match
8. **Winner Checkpoints:** Trust the 🏆 - they're production-validated

---

## 14. Advanced Techniques 🎓

### Memory Management

**Problem:** Large files (>50M points) can cause memory spikes.

**Solution:**
```python
# Clear intermediate tensors
del xyz, rgb, normals
torch.cuda.empty_cache()
```

### Asynchronous File Processing

**Current:** Sequential (file1 → file2 → file3)

**Future Enhancement:**
```python
# Process files in parallel (careful with GPU memory)
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=2) as executor:
    futures = [executor.submit(process_file, f) for f in files]
```

### Caching Computed Features

**Optimization:** Save computed normals to disk
```python
# After compute_normals_gpu:
np.save(f"{input_file}.normals.npy", normals)

# On next run:
if os.path.exists(normals_cache):
    normals = np.load(normals_cache)
```

---

**This skill represents the complete operational knowledge of the Point Cloud Inference Application. Master these concepts for production-grade 3D semantic segmentation deployment.**
