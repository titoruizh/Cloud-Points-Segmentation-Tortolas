# Pipeline de Postproceso: Changelog Detallado de Optimizaciones

> **Archivo modificado:** `app_inference/core/postprocess.py`
> **Hardware de prueba:** NVIDIA RTX 5090 (32GB VRAM), 32 cores CPU, 63GB RAM
> **Dataset de benchmark:** `LINK_260226_LOG176_NDP_PTL_edit_RGB_0.25m.laz` — 76,536,995 puntos (244 MB)
> **Fecha:** Marzo 2026

---

## Resumen Ejecutivo

El pipeline de postproceso (FIX_TECHO + INTERPOL) pasó de **~826s** a **~50s** — una mejora de **~16.5x** — a través de 8 iteraciones de optimización. El postproceso representaba el 84% del tiempo total del pipeline; ahora representa el 24%.

| Métrica | Baseline (V1) | Final (V5.6) | Mejora |
|---------|---------------|--------------|--------|
| INTERPOL | 635.2s | 24.2s | **26x** |
| FIX_TECHO | ~190s (est.) | 25.5s | **~7.5x** |
| Postproceso total | ~825s | ~50s | **~16.5x** |
| Pipeline completo | ~997s | ~217s | **~4.6x** |

---

## Arquitectura del Pipeline

```
Nube LiDAR (.laz)
    │
    ├── 1. NORMALES (GPU Open3D Tensor CUDA) ─────────── ~10.5s
    │     Cálculo de normales por chunks espaciales
    │
    ├── 2. INFERENCIA (PointNet++ GPU FP16) ──────────── ~157s
    │     Clasificación: suelo (class 2) vs maquinaria (class 1)
    │
    ├── 3. FIX_TECHO (subprocess spawn limpio) ───────── ~25s
    │     ├── Smart Merge (gap filling)
    │     ├── DBSCAN (clustering maquinaria)
    │     └── Roof Fill (relleno de techos)
    │
    └── 4. INTERPOL (subprocess spawn limpio) ────────── ~24s
          IDW interpolation: Z de maquinaria → Z del suelo bajo ella
```

Los subprocesos FIX_TECHO e INTERPOL usan `multiprocessing.get_context('spawn').Process()` para obtener un heap limpio (~200MB) separado del heap de inferencia (~40GB), evitando OOM.

---

## V1 — Baseline (0.25m)

**Estado:** Primera versión funcional del pipeline completo.

### Arquitectura INTERPOL
- **95 tiles** (19×5 grilla de 250m) procesados secuencialmente
- Cada tile: `scipy.spatial.cKDTree` construido sobre **TODOS** los puntos de suelo del tile (sin downsample)
- Para nubes de 76M puntos, cada tile podía tener 1-3M puntos de suelo
- cKDTree query con k=12 vecinos, max_dist=50m
- IDW (Inverse Distance Weighting) en CPU

### Arquitectura FIX_TECHO
- Smart Merge: `cKDTree.query_ball_point` para encontrar suelo cerca de maquinaria
- Cuadrante surrounding test: punto de suelo debe tener maquinaria en 3+ cuadrantes
- DBSCAN clustering (sklearn, CPU)
- Roof Fill: 796 clusters procesados en paralelo con `ThreadPoolExecutor`
  - Cada cluster: escanea 76M puntos con BBox filter + construye cKDTree individual

### Tiempos
| Etapa | Tiempo |
|-------|--------|
| Normales | 10.7s |
| Inferencia | 161.8s |
| INTERPOL | **635.2s** |
| **Total** | **~997s (16.6 min)** |

### Métricas de calidad
- dZ medio: 0.397m | dZ max: 10.574m
- Maquinaria detectada: 285,964 pts (0.4%)
- Smart Merge: 700,421 puntos unidos

---

## V2 — Voxel Downsample per-tile

**Cambio clave:** Agregar voxel 2D downsample dentro de cada tile ANTES de construir el cKDTree.

### Detalle técnico
- Si `n_suelo_tile > cfg.max_local_suelo` (default 5M), se aplica voxel 2D downsample
- Resolución adaptativa: `voxel_size = sqrt(area_tile / max_local_suelo)`
- Implementación: discretizar XY en grid → `np.unique` sobre voxel keys → mantener 1 punto por voxel
- Reduce puntos de suelo de ~2.4M a ~800K por tile sin perder cobertura espacial

### Impacto
| Métrica | V1 | V2 | Cambio |
|---------|----|----|--------|
| INTERPOL | 635.2s | **190.8s** | **3.3x** |
| dZ medio | 0.397m | 0.416m | +5% (aceptable) |

**Razón de la mejora:** cKDTree es O(N log N) para construcción y O(log N) por query. Reducir N de 2.4M a 800K reduce drásticamente el tiempo de construcción del árbol (×95 tiles).

---

## V3 — GPU Smart Merge (`torch.cdist`)

**Cambio clave:** Mover el cálculo de distancias del Smart Merge de CPU (`cKDTree.query_ball_point`) a GPU (`torch.cdist`).

### Detalle técnico
- Nueva función `_smart_merge_gpu_chunk()`:
  1. Pre-filtro XY BBox en CPU (rápido)
  2. `torch.cdist(candidates, machinery)` en GPU → matriz de distancias completa
  3. Threshold `dists <= merge_radius` → máscara booleana
  4. Cuadrante test en GPU: bitwise flags para 4 cuadrantes, `pop >= 3`
- Sub-batching dinámico para controlar VRAM: `sub_sz = min(50K, 2GB / (N_loc × 4 bytes))`
- Fallback automático a CPU si GPU OOM o `N_loc > gpu_loc_limit`
- Tier system: `HIGH (≥20GB): gpu_loc_limit=120K`, `MEDIUM: 60K`, `LOW: 30K`

### Impacto
- Smart Merge completó exitosamente: **38,725,540 puntos** unidos (vs 700K en V2)
- dZ medio: **0.165m** (mejor precisión por mejor clasificación)
- INTERPOL: ~192s (sin cambio, no se tocó)

### Problema descubierto (V4)
El Smart Merge unió demasiados puntos (38.7M = 50% de la nube). Esto causó:
- DBSCAN recibía 38.7M puntos → OOM / crash en algunas nubes
- La clasificación era incorrecta (suelo plano reclasificado como maquinaria)

---

## V4 — Primer intento GPU INTERPOL (`torch.cdist` per-tile)

**Cambio clave:** Reemplazar cKDTree per-tile con `torch.cdist` para el cálculo de distancias en INTERPOL.

### Detalle técnico
- Nueva función `_interpol_idw_gpu_tile()` usando `torch.cdist(maq_xy, suelo_xy)`
- Resultado: distancias brutas en GPU, IDW weights calculados en GPU

### Resultado: FRACASO
| Métrica | V3 (CPU) | V4 (GPU cdist) |
|---------|----------|----------------|
| INTERPOL | 192.4s | 224.0s |
| dZ medio | 0.165m | **6.523m** |
| dZ max | 6.727m | **50.422m** |

**Causa raíz:** `torch.cdist` calcula distancias entre TODOS los pares (N_maq × N_suelo), no los K vecinos más cercanos. La función no garantiza encontrar los mismos K=12 vecinos que cKDTree. Los índices de vecinos y las distancias eran incorrectos, produciendo interpolación Z con artefactos verticales masivos.

**Lección:** `torch.cdist` NO es un reemplazo válido para k-NN search. Se necesita una estructura espacial real (tree o hash grid).

---

## V5 — Smart Merge Ratio Cap + Revert GPU INTERPOL

**Cambios clave:**
1. Revertir completamente `_interpol_idw_gpu_tile()` → volver a cKDTree CPU
2. Agregar **ratio cap** al Smart Merge: si `merged_points > 5 × original_maq`, abortar merge

### Detalle técnico del ratio cap
```python
_max_merge = max(500_000, len(idx_maq) * 5)
if len(all_valid) > _max_merge:
    # ABORT: no aplicar merge, usar clasificación original
```
- Para 165K maq original → cap = 825K máximo
- Previene que DBSCAN reciba millones de puntos y crashee

### Detalle técnico: `_detect_gpu_tier()`
Nueva función para detectar VRAM disponible y clasificar GPU:
```python
def _detect_gpu_tier():
    # Retorna (tier_name, free_vram_gb, device)
    # HIGH: ≥20GB | MEDIUM: 8-20GB | LOW: <8GB | CPU: sin GPU
```
Con diccionario `_TIER_PARAMS` para parámetros adaptativos por tier.

### Bug encontrado
`props.total_mem` → debía ser `props.total_memory`. El `except Exception` silencioso hacía que siempre cayera a CPU. Log mostraba `INTERPOL modo: CPU` cuando debía ser GPU.

### Tiempos
| Etapa | Tiempo |
|-------|--------|
| INTERPOL | 203.5s (CPU, correcto) |
| dZ medio | 0.439m (bueno) |

---

## V5.2 — GPU INTERPOL con `torch_cluster.knn` (per-tile)

**Cambio clave:** Reemplazar `torch.cdist` con `torch_cluster.knn()` para k-NN search real en GPU.

### Descubrimiento clave
`torch_cluster` ya estaba compilado desde source en el Dockerfile para CUDA 12.8 / RTX 5090. La función `torch_cluster.knn(x, y, k)` hace k-NN search en GPU y ya se usaba en `src/models/randlanet_blocks.py:104`.

### Detalle técnico
- Nueva función `_interpol_idw_gpu_knn()`:
  1. Transfer ground_xy, maq_xy a GPU como float32
  2. `torch_cluster.knn(ground_t, maq_t, k=12)` → `edge_index[2, N_maq*12]`
  3. Reshape `edge_index[1]` a `(N_maq, 12)` → índices de vecinos
  4. Distancias: `sqrt(sum((neighbor_xy - maq_xy)²))` en PyTorch
  5. Filtro max_dist=50m: `dists > 50 → inf`
  6. IDW: `w = 1/d`, `Z = sum(w*z) / sum(w)` — enteramente en GPU
- Fix del bug: `props.total_mem` → `props.total_memory`

### Resultado: PEOR QUE CPU
| Métrica | V5 (CPU) | V5.2 (GPU per-tile) |
|---------|----------|---------------------|
| INTERPOL | 203.5s | **247.8s** |
| dZ medio | 0.439m | 0.431m (correcto) |
| GPU tiles | 0 | 84/84 |

**Causa raíz:** `torch_cluster.knn` reconstruye la estructura espacial interna en cada llamada. Con 95 tiles × ~2.4M puntos cada uno, el overhead de 95 reconstrucciones supera el beneficio del GPU. El cKDTree de scipy está muy optimizado para este patrón.

**Lección:** El overhead per-call de `torch_cluster.knn` es significativo. Se necesita UNA sola llamada global, no 95 llamadas per-tile.

---

## V5.3 — INTERPOL Global: UN voxel downsample + UN knn call

**Cambio clave:** Reestructuración arquitectural completa de INTERPOL. Reemplazar el loop de 95 tiles con UN enfoque global.

### Antes (per-tile, 95 iteraciones)
```
for tile in 95_tiles:
    suelo_tile = extract_tile(suelo, tile_bbox)
    if len(suelo_tile) > max_local:
        suelo_tile = voxel_downsample(suelo_tile)
    tree = cKDTree(suelo_tile)          # 95 árboles
    dists, idx = tree.query(maq_tile)   # 95 queries
    z_interp = idw(dists, idx, suelo_z) # 95 IDW
```

### Después (global, 1 iteración)
```
# Paso 1: UN voxel downsample global (76M → 4.1M)
suelo_ds = global_voxel_2d_downsample(suelo_xyz, voxel_size=1.08m)

# Paso 2: UN knn call para TODOS los puntos de maquinaria
if gpu_available:
    result = _interpol_idw_gpu_knn(maq_xy_all, suelo_ds_xy, suelo_ds_z, k=6)
else:
    tree = cKDTree(suelo_ds_xy)  # UN solo árbol
    dists, idx = tree.query(maq_xy_all, k=6, workers=-1)  # UN query
    result = cpu_idw(dists, idx, suelo_ds_z)
```

### Detalle técnico
- Voxel 2D global: resolución adaptativa basada en `sqrt(area_total / max_suelo)`
- `k=6` en vez de `k=12` (suficiente para IDW con downsample global)
- CPU fallback: mismo enfoque global con `cKDTree`, no per-tile
- VRAM estimado por tile: ~90MB (ground_xy 19MB + ground_z 10MB + maq 0.5MB + edge_index 13MB + internal 40MB)
- Timing detallado con `datetime.now()` en cada paso

### Resultado: MEJORA MASIVA
| Métrica | V5.2 (per-tile) | V5.3 (global) | Cambio |
|---------|-----------------|---------------|--------|
| Voxel downsample | N/A (per-tile) | **3.8s** | Global |
| knn + IDW | ~247s (95 calls) | **4.8s** (1 call) | **~52x** |
| INTERPOL total | 247.8s | **24.7s** | **10x** |
| dZ medio | 0.431m | 0.457m | +6% (aceptable) |

Los ~20s restantes de INTERPOL total son I/O de disco (leer/escribir 76M puntos en .laz).

---

## V5.4 — Smart Merge Early Abort

**Cambio clave:** Detener el escaneo de bloques del Smart Merge inmediatamente cuando el acumulado supera el threshold.

### Problema
En V5.3, Smart Merge procesaba los 153 bloques completos (76M candidatos × 500K/bloque) antes de verificar el cap. Para este dataset, el merge siempre se abortaba (28.9M > 825K), pero gastaba ~25s procesando 153 bloques inútilmente.

### Detalle técnico
```python
accumulated_count = 0
_max_merge = max(500_000, len(idx_maq) * 5)

for chunk_start in range(0, n_cands, CHUNK_SZ):
    if accumulated_count > _max_merge:
        # EARLY ABORT — no procesar más bloques
        break
    # ... procesar bloque, acumular count
```

### Timing agregados a FIX_TECHO
- Smart Merge: tiempo total + bloques procesados/total
- DBSCAN: `"DBSCAN: N objetos en X.Xs (N pts)"`
- Roof Fill: `"Rellenados N puntos de techo en X.Xs"`
- FIX_TECHO total: `"FIX_TECHO completado en X.Xs"`

### Resultado
| Métrica | V5.3 | V5.4 | Cambio |
|---------|------|------|--------|
| Smart Merge | ~25s (153 bloques) | **1.3s** (6/153 bloques) | **19x** |
| DBSCAN | ~0.4s | 0.4s | = |
| Roof Fill | ~74s (desconocido en V5.3) | **74.0s** | Ahora visible |
| FIX_TECHO total | ~100s (est.) | **92.2s** | Visible |

**Descubrimiento:** Roof Fill era el verdadero bottleneck (74s de 92.2s = 80% de FIX_TECHO).

---

## V5.5 — Roof Fill Global: UN cKDTree + UN query

**Cambio clave:** Reemplazar el procesamiento de 796 clusters individuales con UN enfoque global (mismo patrón exitoso de INTERPOL V5.3).

### Problema arquitectural del Roof Fill original
```python
# POR CADA cluster (×796):
for lbl in unique_labels:
    cluster_points = xyz[idx_maq][labels == lbl]
    # Escanear 76M puntos con BBox filter (6 comparaciones × 76M)
    candidate_mask = (classification == ground) & (xyz[:, 0] >= min_x - pad) & ...
    candidates = np.where(candidate_mask)[0]
    # Construir cKDTree individual
    tree_2d = cKDTree(cluster_points[:, :2])
    dists, _ = tree_2d.query(candidates_xy, k=1)
```

**Costo:** 796 × 76M BBox comparaciones = **~60 BILLION operaciones**. Más 796 construcciones de cKDTree.

### Nuevo enfoque global
```python
# Paso 1: Per-cluster Z ranges (vectorizado)
for cluster in sorted_labels:
    cluster_z_lo[i] = percentile_5(cluster_z) + z_buffer
    cluster_z_hi[i] = percentile_5(cluster_z) + max_height

# Paso 2: UN filtro Z global sobre 76M (una sola pasada)
cand_mask = (classification == ground) & (z >= global_z_lo) & (z <= global_z_hi)

# Paso 3: UN cKDTree con TODA la maquinaria válida (164K pts)
tree_maq_2d = cKDTree(maq_valid[:, :2])

# Paso 4: UN query para TODOS los candidatos
dists, nn_idx = tree_maq_2d.query(cand_xy, k=1, workers=-1)

# Paso 5: Filtro proximidad + verificación Z per-cluster vectorizada
prox_mask = dists <= proximity_radius  # 1.5m
matched_cluster = maq_cluster_idx[nn_idx[prox_mask]]
z_valid = (cand_z >= cluster_z_lo[matched_cluster]) & (cand_z <= cluster_z_hi[matched_cluster])
```

**Clave:** `np.searchsorted(sorted_ulabels, maq_valid_labels)` mapea cada punto de maquinaria a su cluster index en O(N log K), permitiendo verificación Z vectorizada.

### Resultado
| Métrica | V5.4 (per-cluster) | V5.5 (global) | Cambio |
|---------|-------|------|--------|
| Roof Fill | **74.0s** | **11.1s** | **6.7x** |
| FIX_TECHO total | 92.2s | **26.7s** | **3.5x** |
| Puntos rellenados | 203,363 | 197,877 | -2.7% (delta menor) |

La diferencia de ~5.5K puntos se debe a que el enfoque global usa el vecino más cercano de TODA la maquinaria (no solo del cluster local), lo cual es ligeramente más conservador pero igualmente correcto.

### Timing detallado V5.5
```
Roof Z-ranges: 787 clusters en 0.2s
Roof filtro Z global: 76,183,233 candidatos en 0.2s
Roof cKDTree: 156,696 maq en 0.0s
Roof query: 76,183,233 pts en 10.6s
Rellenados 197,877 puntos de techo en 11.1s
```

**Observación:** El filtro Z capturó 99.5% del suelo (76.2M de 76.5M). Los clusters de maquinaria abarcan un rango Z amplio que incluye casi toda la nube.

---

## V5.6 — Filtro XY+Z para Roof Fill (versión final)

**Cambio clave:** Agregar filtro XY BBox global al pre-filtro de candidatos del Roof Fill.

### Detalle técnico
```python
maq_xy_min = maq_valid[:, :2].min(axis=0)
maq_xy_max = maq_valid[:, :2].max(axis=0)
xy_margin = proximity_radius + padding  # 1.5 + 1.5 = 3.0m

cand_mask = (
    (classification == ground) &
    (xyz[:, 0] >= maq_xy_min[0] - xy_margin) &
    (xyz[:, 0] <= maq_xy_max[0] + xy_margin) &
    (xyz[:, 1] >= maq_xy_min[1] - xy_margin) &
    (xyz[:, 1] <= maq_xy_max[1] + xy_margin) &
    (xyz[:, 2] >= global_z_lo) &
    (xyz[:, 2] <= global_z_hi)
)
```

### Resultado
| Métrica | V5.5 | V5.6 |
|---------|------|------|
| Filtro candidatos | 76.2M (Z only) | 76.0M (XY+Z, 99.5%) |
| Roof query | 10.6s | **7.4s** |
| Roof Fill total | 11.1s | **8.1s** |
| FIX_TECHO total | 26.7s | **25.5s** |

Para este dataset, la maquinaria se extiende por casi toda el área XY, limitando el impacto del filtro XY. En nubes donde la maquinaria ocupe una fracción del área total, el filtro XY reduciría candidatos significativamente más.

---

## Tabla Comparativa Final — Todas las Versiones

### Tiempos (segundos)

| Versión | Normales | Inferencia | FIX_TECHO | INTERPOL | Post total | Pipeline |
|---------|----------|------------|-----------|----------|------------|----------|
| V1 (baseline) | 10.7 | 161.8 | ~190 | 635.2 | **~825** | **~997** |
| V2 (voxel ds) | 10.4 | 158.7 | ~190 | 190.8 | **~381** | **~560** |
| V3 (GPU SM) | 12.5 | 156.1 | ~190 | 192.4 | **~382** | **~541** |
| V4 (GPU cdist) | 12.5 | 154.7 | ~190 | 224.0 | **~414** | **~574** |
| V5 (revert+cap) | 13.2 | 160.2 | ~100 | 203.5 | **~304** | **~567** |
| V5.2 (knn tile) | 10.1 | 156.8 | ~100 | 247.8 | **~348** | **~515** |
| **V5.3 (knn global)** | 10.2 | 158.8 | ~100 | **24.7** | **~125** | **~394** |
| V5.4 (early abort) | 10.5 | 158.8 | **92.2** | 24.7 | **~117** | **~386** |
| V5.5 (roof global) | 12.8 | 163.6 | **26.7** | 24.3 | **~51** | **~327** |
| **V5.6 (XY filter)** | **10.5** | **157.5** | **25.5** | **24.2** | **~50** | **~318** |

### Calidad (dZ = diferencia Z entre valor original e interpolado)

| Versión | dZ medio | dZ max | Puntos modificados | Smart Merge |
|---------|----------|--------|-------------------|-------------|
| V1 | 0.397m | 10.574m | 1,260,995 | 700,421 merged |
| V2 | 0.416m | 10.419m | 1,295,041 | 710,804 merged |
| V3 | **0.165m** | **6.727m** | 240,884 | 38.7M merged |
| V4 | ~~6.523m~~ | ~~50.422m~~ | - | ~~38.7M merged~~ |
| V5 | 0.439m | 9.790m | 398,675 | Abortado (31.9M) |
| V5.2 | 0.431m | 9.789m | 342,446 | Abortado (32.2M) |
| V5.3 | 0.457m | 9.789m | 342,446 | Abortado (28.9M) |
| V5.6 | 0.470m | 9.633m | 329,893 | Abortado (1.1M, early) |

---

## Resumen de Patrones de Optimización Aplicados

### 1. "Global vs Per-Tile" (el patrón más impactante)
Reemplazar N operaciones independientes por UNA operación global. Aplicado exitosamente en:
- **INTERPOL:** 95 cKDTree builds/queries → 1 global (190s → 4.8s compute)
- **Roof Fill:** 796 BBox scans + cKDTree builds → 1 global (74s → 8s)
- **Smart Merge:** No aplicable (la lógica per-chunk es necesaria para el quadrant test)

### 2. "Early Abort"
Detener procesamiento cuando el resultado ya es predecible:
- **Smart Merge:** Acumulador + threshold check per-bloque (153 → 6 bloques, 25s → 1.3s)

### 3. "Voxel Downsample"
Reducir puntos manteniendo cobertura espacial:
- **INTERPOL V2:** Per-tile downsample (3.3x mejora)
- **INTERPOL V5.3:** Global downsample 76M → 4.1M (habilita knn global)

### 4. "GPU-accelerated k-NN via torch_cluster"
Usar `torch_cluster.knn()` (ya compilado en el Docker) en lugar de CPU cKDTree:
- **INTERPOL V5.3:** 1 llamada knn en GPU (4.8s vs ~30s CPU estimado)
- **Clave:** `torch_cluster` ya estaba en el Dockerfile, zero dependencias nuevas

### 5. "Adaptive GPU Tier System"
Parámetros adaptativos según VRAM disponible (`_detect_gpu_tier` + `_TIER_PARAMS`):
- HIGH (≥20GB): max_suelo=5M, gpu_loc_limit=120K
- MEDIUM (8-20GB): max_suelo=3M, gpu_loc_limit=60K
- LOW (<8GB): max_suelo=1.5M, gpu_loc_limit=30K
- CPU fallback: misma lógica global con cKDTree

### 6. "Fail-safe con ratio cap"
Prevenir cascadas de error cuando un algoritmo produce resultados anómalos:
- **Smart Merge cap:** >5× maq original → abort merge
- **GPU fallback:** Cualquier error GPU → CPU path automático

---

## Errores Encontrados y Resueltos

| Bug | Versión | Síntoma | Causa raíz | Fix |
|-----|---------|---------|------------|-----|
| GPU no detectada | V5 | `INTERPOL modo: CPU` | `props.total_mem` (inexistente) | `props.total_memory` |
| GPU INTERPOL Z corrupto | V4 | dZ=6.523m | `torch.cdist` no es k-NN search | Reemplazar con `torch_cluster.knn` |
| Smart Merge overflow | V3→V4 | DBSCAN OOM/crash | 38.7M puntos merged | Ratio cap 5× |
| Per-tile GPU lento | V5.2 | 247.8s (peor que CPU) | 95× rebuild de estructura knn | Enfoque global (1 llamada) |
| Double-del NameError | V5.3 | Potencial crash | `del suelo_x, suelo_y` ya eliminados | Limpiar del statement |
| Roof Fill 60B ops | V5.4 | 74s roof fill | 796 × 76M BBox scans | Enfoque global (1 tree + 1 query) |

---

## Dependencias Utilizadas

| Librería | Uso | Notas |
|----------|-----|-------|
| `torch_cluster.knn` | GPU k-NN en INTERPOL | Compilado desde source en Dockerfile (CUDA 12.8) |
| `torch.cdist` | GPU distancias en Smart Merge | PyTorch standard |
| `scipy.spatial.cKDTree` | CPU k-NN (fallback + Roof Fill) | Scipy standard |
| `sklearn.cluster.DBSCAN` | Clustering maquinaria | CPU, n_jobs=-1 |
| `numpy` | Arrays, voxel downsample, IDW | Standard |

No se instalaron dependencias nuevas. Todo el stack GPU estaba pre-existente en el Docker.
