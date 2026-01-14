# Informe Técnico V5: "Geometric Purification" (Verticality Ablation) 📉
**Versión:** 5.0 (Ablation Study)
**Fecha:** 12 Enero 2026
**Autor:** Antigravity AI & Usuario
**Estado:** 🧪 Experimento en Curso

---

## 1. Hipótesis V5: ¿La Verticalidad nos Miente? 🤔
En V4 logramos excelentes métricas (IoU 83%), pero observamos problemas persistentes en:
1.  **Techos:** A menudo se confunden con suelo.
2.  **Pretiles:** Muros bajos confundidos con maquinaria.

**Teoría:** La feature de "Verticalidad" (1-|Nz|) fuerza al modelo a aprender que "Pared = Maquinaria" y "Plano = Suelo". Esto es cierto para el chasis, pero **falso para el techo** (que es plano) y **falso para el suelo inclinado**.
Al eliminar este canal explícito, forzamos al modelo (PointNet++) a aprender la *forma tridimensional completa* en lugar de depender de un "truco" local como la normal Z.

---

## 2. Definición Técnica V5 🛠️

### 2.1 Nueva Dimensionalidad (`d_in = 9`)
Eliminamos la verticalidad explicita.

| Canal | Descripción |
| :--- | :--- |
| **0-2** | X, Y, Z (Normalizados) |
| **3-5** | R, G, B (Normalizados) |
| **6-8** | Nx, Ny, Nz (Normales de superficie) |
| **~~9~~** | ~~Verticalidad~~ (ELIMINADO ❌) |

### 2.2 Estrategia de Datos
*   **Source:** `data/raw RGB`
*   **Target:** `data/processed/blocks_10m V5`
*   **Script:** `scripts/preprocessing/V5/preprocess_blocks_10m_v5.py`

---

## 3. Resultados Esperados
*   **Posible caída en IoU inicial:** La verticalidad es una feature muy fuerte ("chivato"). Sin ella, el entrenamiento podría tardar más en converger, pero el resultado debería ser más robusto geométricamente.

---

## 4. Análisis de Datos y Estrategia de Entrenamiento 📊💪

Antes de entrenar, ejecutamos un análisis exhaustivo del dataset generado (`data/processed/blocks_10m V5`).

### 4.1 Estadísticas del Dataset V5
*   **Total Bloques:** 837
*   **Distribución de Bloques:**
    *   🚜 Machinery: 246 (29.4%)
    *   ⛰️ Hard Negative: 238 (28.4%) - *Muros y pendientes fuertes*
    *   🟤 Easy Negative: 353 (42.2%)
*   **Balance de Puntos:**
    *   Suelo (0): 95.78%
    *   Maquinaria (1): 4.22%
*   **Ratio de Desbalance:** **22.7 : 1** (Por cada punto de máquina hay ~23 de suelo).

### 4.2 Configuración de Entrenamiento (`pointnet2_v5_novert.yaml`)
Basado en el análisis, ajustamos los hiperparámetros para compensar la falta de la feature "Verticalidad" y el desbalance de clases.

1.  **Class Weights `[1.0, 15.0]`:**
    *   Optamos por un peso conservador (15.0) en lugar del ratio puro (23.0) para evitar falsos positivos excesivos, confiando en el oversampling para llenar el gap.
2.  **Runtime Oversampling (Factor 4x):**
    *   Se implementó `oversample_machinery: 4` en el configuración.
    *   **Efecto:** Vemos la maquinaria 5 veces más frecuentemente por época (1 real + 4 copias), reduciendo el desbalance efectivo a ~4:1.
3.  **Scheduler: `CosineAnnealingLR`:**
    *   A diferencia de `StepLR` (V4), usamos un decaimiento coseno para una convergencia más suave, permitiendo al modelo explorar mínimos más robustos sin cambios bruscos de LR.
4.  **Loader V5 (`src.data.dataset_v5`):**
    *   Se creó un cargador específico para manejar `d_in=9` (10 columnas en disco).
    *   **Pipeline:** XYZ -> Augmentation -> Feature Stacking (XYZ+RGB+Normals) -> Tensor.

**Estado Actual:**
*   Explorando `LR [0.0005, 0.001]` y `Weights [15, 20, 25]`.

---

## 5. Inferencia V5.2: "Nitro" 🏎️💨

Para hacer frente a nubes de puntos masivas (100M+ puntos) en la RTX 5090, hemos desarrollado una nueva versión del motor de inferencia: `scripts/inference/infer_pointnet_v5.2.py`.

### 5.1 Optimizaciones Clave
1.  **Lectura Directa de Normales:** Si el archivo LAS ya trae normales (`normal_x`, `vl_x`...), el script las lee directamente sin recalcularlas con Open3D. Ahorro de tiempo: **~60-80%** en pre-procesamiento.
2.  **Gridding Vectorizado:** Reemplazo de bucles Python por operaciones vectorizadas de Numpy para dividir la nube en bloques de 10x10m.
3.  **Inferencia FP16 (AMP):** Uso de `torch.amp.autocast` para reducir el uso de memoria VRAM y aprovechar los Tensor Cores.
4.  **Carga "Nitro":** Pre-allocación de tensores en memoria continua para maximizar el ancho de banda hacia la GPU.

### 5.2 Configuración Recomendada (RTX 5090)
*   **Batch Size:** `64` (Estándar Seguro) o `96` (Agresivo).
    *   *Nota:* Intentar `256` causó OOM (>20GB VRAM de alocación) debido a la expansión de tensores internos de PointNet++.
*   **Torch Compile:** Opcional (`--no_compile false`). Acelera el grafo, pero añade overhead de inicio (1-2 min).

**Comando:**
```bash
python3 scripts/inference/infer_pointnet_v5.2.py \
  --input_file "ruta/nube.laz" \
  --checkpoint "ruta/best_model.pth" \
  --batch_size 64
```



