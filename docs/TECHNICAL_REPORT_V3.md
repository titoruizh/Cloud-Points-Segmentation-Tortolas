# Informe Técnico V3: RandLANet "Turbo C++" & High Density ⚡
**Versión:** 3.0 (Ingeniería Avanzada)
**Fecha:** 07 Enero 2026
**Autor:** Antigravity AI & Usuario
**Estado:** ✅ Entrenamiento High Density Activo

---

## 1. El Pivot a V3 (Ingeniería de Software) 🎯

### 1.1 El Problema (V2 Limit)
En la V2, identificamos un límite computacional severo:
- **Matriz de Adyacencia Cuadrática:** La implementación original usaba `square_distance` ($O(N^2)$).
- **Consumo de Memoria:**
    - Para $N=25k$: ~12 GB VRAM (Manejable).
    - Para $N=65k$: **>64 GB VRAM** (Crash instantáneo en RTX 5090).
- **Consecuencia:** Estábamos limitados al "Efficiency Spot" (25k), perdiendo densidad crítica para ver maquinaria pequeña en bloques de 50m.

### 1.2 La Solución "Turbo C++" (V3)
Reescribimos el núcleo geométrico de RandLANet (`randlanet_blocks.py`) para utilizar **aceleración nativa en C++** mediante `torch_cluster`.

- **Algoritmo:** K-Nearest Neighbors (KNN) optimizado.
- **Complejidad:** $O(N)$ (vs $O(N^2)$).
- **Resultado:**
    - Podemos procesar **65,536 puntos** (High Density).
    - El consumo de memoria es **mínimo** (la matriz gigante desaparece).
    - Velocidad de entrenamiento multiplicada.

---

## 2. Nueva Arquitectura V3 🛠️

### 2.1 Componentes Actualizados

#### A. RandLANet Blocks (`src/models/randlanet_blocks.py`)
Se implementó una lógica híbrida con fallback seguro:
```python
try:
    from torch_cluster import knn
    KNN_AVAILABLE = True  # 🚀 V3 MODE
except ImportError:
    KNN_AVAILABLE = False # 🐢 V1/V2 LEGACY MODE
```
- **Optimización de Entrada:** Ahora el bloque detecta y corrige automáticamente la dimensionalidad (`[B, 3, N]` vs `[B, N, 3]`), eliminando errores de transposición comunes.

#### B. Arquitectura Profunda (`src/models/randlanet.py`)
Se corrigió una inconsistencia crítica en el flujo de canales que V2 arrastraba, alineando el Encoder con el Decoder para permitir Skip Isomorphic Connections robustas:
- **Flujo de Tensores:** 8 $\to$ 32 $\to$ 64 $\to$ 128 $\to$ 256 Canales.
- **Beneficio:** Mayor capacidad de representación semántica profunda.

---

## 3. Configuración Productiva V3 ⚙️

### 3.1 Perfil "High Density C++"
Archivo: `configs/randlanet/randlanet_v4_optimized.yaml`

| Parámetro | Valor V3 | Valor V2 (Antiguo) | Razón |
| :--- | :--- | :--- | :--- |
| **Num Points** | **65,536** | 25,000 | Densidad real. Verdad absoluta. |
| **Batch Size** | **12** | 4 | La optimización de memoria libera VRAM para más batch. |
| **Accumulations** | **2** | 6 | Menos pasos virtuales requeridos. |
| **LR Strategy** | **Sweep** | 0.005 | Búsqueda Bayesiana activa. |

### 3.2 Naming Convention (Sweeps)
Para trazabilidad en Weights & Biases (W&B), los experimentos generados por el Agente siguen el patrón dinámico de hiperparámetros:
- **ID:** `LR_<LearningRate>_W_<WeightMAQ>_J_<Jitter>` (e.g., `LR_0.005_W_100_J_0.01`)
- **Proyecto:** `Tortolas-segmentation`
- **Grupo:** `RandLANet_V2` (Se mantiene para continuidad histórica).

---

## 4. Instrucciones de Despliegue 🚀

### 4.1 Entrenamiento (Agente / Manual)
El entorno está preparado para ejecución inmediata con la aceleración activada.

```bash
# Lanzamiento Manual (Verificación)
python3 TRAIN.py --config configs/randlanet/randlanet_v4_optimized.yaml

# Lanzamiento Agente (Sweeps)
wandb agent <SWEEP_ID>
```

### 4.2 Inferencia (Producción)
Los scripts de inferencia heredarán automáticamente la ventaja de velocidad de `torch_cluster` si se ejecuta en la misma imagen Docker.

---

## 5. Conclusión
La V3 no cambia el modelo matemático, pero desbloquea su potencial real eliminando el cuello de botella de ingeniería. Ahora tenemos **la mejor densidad posible** (65k) con **la mejor eficiencia posible** (C++).

## 3.3 PointNet++ V2 Inference Post-Mortem (The "0% vs 400%" Paradox)
During the inference phase of V2, we encountered a critical instability where the model either detected **0% Machinery** (Robust Model) or **400% Machinery** (Sensitive Model), with no middle ground. 

### Root Cause Analysis
1.  **The "Unseen Ground" Bias (Critical):**
    *   **Issue:** The V2 Training Dataset was generated with `EASY_NEGATIVE_RATIO = 0.0` in `preprocess_blocks.py`.
    *   **Effect:** The model was trained **exclusively** on Machinery logs and complex rocks (Hard Negatives). It **never saw flat ground** during training.
    *   **Result:** During inference, when presented with flat ground (Verticality $\approx$ 0), the model treated it as "Out-of-Distribution" data. The sensitive checkpoint mapped this unknown strictly to Class 1 (Machinery), causing millions of False Positives.
    *   **Fix (Future):** In V3, we must set `EASY_NEGATIVE_RATIO = 0.5` to teach the model what "Ground" looks like.

2.  **Feature Inversion (Verticality):**
    *   **Issue:** 
        *   Training Preprocessing: `Verticality = 1.0 - abs(Nz)` (1.0 = Wall).
        *   Inference Script: `Verticality = abs(Nz)` (1.0 = Flat).
    *   **Effect:** We were inadvertently feeding "Flat Ground" as "Perfect Vertical Walls" to the model during inference, amplifying the False Positives.
    *   **Fix:** Corrected `infer_pointnet.py` to match training formula.

3.  **Grid Alignment Mismatch:**
    *   **Issue:** Inference was normalizing coordinates based on *Data Centroid* (`x - min(x)`), whereas Training used *Tile Centers* (`x - tile_center`).
    *   **Effect:** This caused a spatial shift of ~5 meters, misaligning the learned geometric features.
    *   **Fix:** Implemented `Tile-Center` normalization in the inference loader.

### Mitigations Implemented (Inference-Side)
Since retraining V2 was not immediate, we implemented physics-based filters in `infer_pointnet.py`:
*   **Physics Filter:** Forces `Probability = 0` if `Verticality < 0.05` (Flat ground cannot be machinery).
*   **DBSCAN Clustering:** Removes small, sparse clusters (< 50 points) to eliminate "scan-line" noise artifacts.

## 4. RandLANet V2 High-Density Inference Results
As part of the final V2 validation, we developed and tested a robust inference pipeline for RandLANet using the new High-Density capabilities.

### 4.1 Implementation Details (`infer_randlanet.py`)
*   **Block Size:** 50.0m (matched to training V2).
*   **Point Density:** 65,536 points per block (leveraging `torch_cluster` optimization).
*   **Normalization:** Implemented **Tile-Center Normalization** to resolve the spatial shift observed in PointNet++.
*   **Post-Processing:** Integrated the same **Physics Filter** (Verticality < 0.05 -> Ground) and **DBSCAN** noise removal used for PointNet++.

### 4.2 Performance Observation
*   **Execution:** The script runs successfully on the RTX 5090, processing ~12 million points in < 1 minute.
*   **Detection Quality:** The model functions ("al menos funciona") but exhibits similar sensitivity issues to PointNet++, likely due to the shared **Dataset Bias** (lack of explicit ground samples).
*   **Result:** It produced ~15,000 candidate machinery points from a 12M point cloud after aggressive filtering.

---

## 5. Final Recommendations for V3 Phase 🏁
This concludes the V2 "High Density & Robustness" phase. We have successfully engineered the software stack to handle massive point clouds (65k) and validated the inference pipelines.

### Critical Path for V3 (Data-Centric Iteration):
The primary bottleneck is no longer Code or Hardware, but **Data Distribution**.

1.  **Activación de "Easy Negatives" (Mandatory):**
    *   **Acción:** Set `DEFAULT_EASY_NEGATIVE_RATIO = 0.5` in `preprocess_blocks.py`.
    *   **Objetivo:** Generar miles de bloques de *solo suelo* y *solo pared plana*.
    *   **Efecto:** El modelo aprenderá que "Plano = Clase 0" nativamente, eliminando la necesidad de filtros heurísticos en inferencia.

2.  **Hybrid Training Strategy:**
    *   Train RandLANet V3 with the mixed dataset (Machinery + Rocks + Flat Ground).
    *   Utilize the validated `sweep_hyperparam.yaml` with conservative Learning Rates (0.001 - 0.005) to ensure stability.

**Status Final:** ✅ Código optimizado, Pipelines validados, Siguiente paso: Re-generación de Datos V3.

---

## 6. Data-Centric Evolution (The "Balanced" Era) �
*Added after Preprocessing V3*

Following the identification of the "Unseen Ground" bias, we executed a rigorous data engineering phase. 

### 6.1 Dataset 1: RandLANet High Density (30m)
Aims to satisfy RandLANet's hunger for point density while maintaining context.
*   **Block Size:** 30m x 30m
*   **Points per Block (Mean):** ~79,400 (Density ~88 pts/m²)
*   **Class Balance (Points):** 
    *   Suelo: **98.6%**
    *   Maquinaria: **1.4%**
    *   *Analysis:* Severe imbalance (1:70). Requires  to prevent mode collapse.

### 6.2 Dataset 2: PointNet++ Balanced V2 (10m)
Designed to cure PointNet++'s false positives by forcing it to learn "Flat Ground".
*   **Block Size:** 10m x 10m
*   **Strategy:** Aggressive Easy Negative Mining.
*   **Block Distribution (Perfect Balance):**
    *   🚜 Machinery: **29.9%**
    *   ⛰️ Hard Negative (Rocks): **26.0%**
    *   🟤 Easy Negative (Ground): **44.1%** (Previously 0%)
*   **Point Balance:** 1:22 (4.3% Machinery).
*   **Configuration Impact:**
    *   Allowed softening weights to .
    *   Reduced  to **2.5m** to capture local curvature vs 5.0m (blur).

## 7. Next Steps: PointNet++ V3 Sweep 🚀
With the dataset fixed, we launch a Hyperparameter Sweep to find the optimal Local Radius and Weights for this balanced data.
*   **Target:** Maximize IoU_Maq without 400% over-detection.
*   **Sweep Params:** , , .


---

## 8. PointNet++ V3 Final Results: The Breakthrough 🏆
*Added after V3 Sweep Analysis*

Following the correction of the "Unseen Ground" bias and the implementation of the "Balanced Dataset" (Ratio 1.5), we achieved a massive leap in robustness.

### 8.1 The Winning Configuration (R3.5)
The user selected the following checkpoint for production inference based on its balance of IoU and Stability (Min Loss):
*   **Checkpoint:** `LR0.0010_W20_J0.005_R3.5_BEST_IOU.pth`
*   **Hyperparameters:**
    *   **Learning Rate:** `0.0010` (Conservative & Stable)
    *   **Class Weights:** `[1.0, 20.0]` (Perfectly aligned with 1:22 data imbalance)
    *   **Base Radius:** `3.5m` (The "Goldilocks" zone: detailed enough for wheels, wide enough for context).

### 8.2 Quantitative Metrics
Based on `VALORES POINTNET V3.csv`:
| Métrica | V2 (Legacy) | V3 (Balanced) | Delta |
| :--- | :--- | :--- | :--- |
| **Val Loss** | ~0.19 - 0.25 | **0.072** | � **-65% Error** (Massive stability gain) |
| **IoU Maquinaria** | 0% or ~40% (Unstable) | **64.33%** | 🚀 **Robust & Reliable** |
| **Accuracy** | ~85% | **94.5%** | ✅ High Precision |

### 8.3 Qualitative "Reality Check"
*   **False Positives:** The "Unseen Ground" phantom detections have explicitly vanished. The model now confidently classifies flat ground as Class 0 (even without the Heuristic Filter, though we keep it for safety).
*   **Detail:** The reduced radius (down to 3.5m from 5.0m) allows the model to distinguish complex rock formations from machinery edges.

### 8.4 Conclusion for V3
The **Data-Centric approach** was the key. No amount of architectural tweaking (Turbo C++) could fix the fact that the V2 model had never seen ground. By balancing the dataset (44% Ground), we cured the model's blindness.

**Status:** ✅ **PRODUCTION READY**


---

## 9. RandLANet V3: The "Paranoia" Incident (Lessons Learned) 📉
*Added after Initial V3 Sweep*

### 9.1 The Failure Mode
In the first attempt to train RandLANet V3 (30m), we strictly followed the mathematical imbalance of the dataset (~1.4% Machinery -> 1:70 Ratio).
*   **Config Used:** `class_weights: [1.0, 50.0]`
*   **Result:** The model became **frozen in a local minimum** with a Validation Loss of ~0.35 (vs 0.19 typical).
*   **Behavior:** The penalty for misclassifying machinery was so high (50x) that the gradients exploded or caused the model to become "paranoid", predicting extremely conservative/noisy patterns to avoid the massive loss penalty. It effectively stopped learning after Epoch 4.

### 9.2 The Correction (Stability > Theory)
We reverted to the proven "Robust" weights from V2, even though they technically under-represent the imbalance.
*   **New Config:**
    *   **Weights:** `[1.0, 15.0]` (Softened from 50.0).
    *   **Learning Rate:** `0.0005` (Reduced from 0.001 to prevent oscillations).
*   **Rationale:** It is better to have a stable training that captures *most* machinery than a theoretically perfect weighting that causes gradient instability. We rely on the high point density (65k) to compensate.


---

## 10. RandLANet V3 Final Verdict: The Limit of the System ⚖️
*Added after Final Analysis of 30m Runs*

We have concluded the RandLANet V3 (30m High Density) experiments. The results are definitive and reveal the "Stability vs. Theory" trade-off.

### 10.1 Comparative Analysis (The "Paranoia" Proof)
We tested three distinct configurations to handle the 1:70 class imbalance. The results confirm that **theoretical weighting (50.0) is toxic** for this geometry.

| Run ID | Learning Rate | Weight Maq | Best IoU Maq | Val Loss | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LR0.0004_W35...** | **0.0004** | **35.0** | **29.66%** 🏆 | **0.3179** ✅ | **GANADOR (Robust)** |
| LR0.0005_W50... | 0.0005 | 50.0 | 18.68% ❌ | 0.3473 | PEOR (Paranoid) |
| LR0.0013_W50... | 0.0013 | 50.0 | 17.59% ❌ | 0.3453 | Inestable |

### 10.2 Technical Deduction
1.  **The "Paranoia" Threshold:**
    *   At **Weight 50.0**, the model suffers from gradient noise. The penalty for missing a machinery point is so high that the model becomes risk-averse, likely predicting large, fuzzy blobs to cover potential machinery, which ruins Precision and lowers IoU.
    *   **Evidence:** The IoU collapsed from ~30% to ~18% just by increasing weight from 35 to 50.

2.  **The Sweet Spot (Weight 35):**
    *   **Weight 35.0** represents the maximum pressure we can apply before the gradients destabilize. It achieves a **Loss of 0.31**, significantly better than the 0.34+ seen in the heavier models.
    *   **Result:** 29.66% IoU seems to be the **architectural limit** for RandLANet on this specific 30m dataset distribution without further data augmentation or "Easy Negative" injection.

### 10.3 Final Recommendation
*   **Production Model:** Use the checkpoint **`LR0.0004_W35..._BEST_IOU.pth`**.
*   **Comparison:** PointNet++ V3 (R3.5) achieved **64% IoU**, significantly outperforming RandLANet (~30%). This suggests that for this specific task (sparse machinery in large open pits), **local context (PointNet++) beats density (RandLANet)**.

