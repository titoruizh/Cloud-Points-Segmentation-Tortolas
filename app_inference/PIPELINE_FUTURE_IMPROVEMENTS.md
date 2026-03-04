# Pipeline de Postproceso: Futuras Mejoras

> **Estado actual (V5.6):** Pipeline total ~218s (Normales 10.5s + Inferencia 157.5s + FIX_TECHO 25.5s + INTERPOL 24.2s)
> **Archivo principal:** `app_inference/core/postprocess.py` (1387 líneas)
> **Última optimización:** Marzo 2026

---

## Distribución actual del tiempo

```
Inferencia PointNet++  ██████████████████████████████████████  157.5s  (72.2%)
FIX_TECHO total        ██████                                  25.5s  (11.7%)
  ├─ Smart Merge         █  1.3s
  ├─ DBSCAN              ▏  0.2s
  ├─ Roof Fill           ██  8.1s
  └─ I/O disco           ███  ~16s
INTERPOL total         █████                                   24.2s  (11.1%)
  ├─ Voxel downsample    █  3.8s
  ├─ GPU knn+IDW         █  4.6s
  └─ I/O disco           ███  ~16s
Normales               ██                                      10.5s  (4.8%)
```

**El 72% del pipeline es inferencia.** El postproceso (FIX_TECHO + INTERPOL) ya está en ~50s, de los cuales ~32s es I/O de disco.

---

## Prioridad 1: Reducir I/O de disco (~32s, 15% del pipeline)

### Problema
FIX_TECHO e INTERPOL se ejecutan como subprocesos aislados (`multiprocessing.spawn`). Cada uno:
1. Lee el .laz completo desde disco (~8s para 76M puntos)
2. Procesa
3. Escribe el .laz completo a disco (~8s)

Total: ~32s en I/O puro (2 reads + 2 writes).

### Solución propuesta: Pipeline en memoria con shared memory
```python
# En lugar de spawn + leer .laz:
from multiprocessing import shared_memory

# Proceso padre crea shared memory con xyz + classification
shm = shared_memory.SharedMemory(create=True, size=xyz.nbytes + classification.nbytes)
# Subproceso accede directamente sin leer disco
shm_child = shared_memory.SharedMemory(name=shm.name)
```

### Alternativa: Unificar FIX_TECHO + INTERPOL en un solo subproceso
- Actualmente son 2 subprocesos → 2 lecturas + 2 escrituras = 4 operaciones I/O
- Si se unifican → 1 lectura + 1 escritura = 2 operaciones I/O
- Ahorro estimado: ~16s

### Riesgo
- El aislamiento por subproceso existe para prevenir OOM (heap de inferencia ~40GB)
- Shared memory requiere gestión cuidadosa de lifecycle
- Unificar subprocesos aumenta el pico de RAM del subproceso

### Impacto estimado: 32s → 16s (ahorro ~16s)

---

## Prioridad 2: Inferencia — Batch size dinámico y DataLoader optimizado (~157s)

### Estado actual
- batch_size=256 fijo, workers=12
- 192 batches × ~0.82s/batch
- VRAM usada: ~10GB de 32GB (31% utilización)

### Mejoras posibles

#### 2a. Batch size mayor
- VRAM disponible: ~20GB libres durante inferencia
- batch_size=512 o 1024 podría reducir overhead de DataLoader
- Requiere benchmark cuidadoso (diminishing returns por memory bandwidth)

#### 2b. `torch.compile` con mode='max-autotune'
- Actualmente usa `torch.compile` estándar
- `mode='max-autotune'` prueba más kernels CUDA (primera ejecución lenta, subsiguientes más rápidas)
- Beneficio principal en producción con múltiples inferencias

#### 2c. FP8 / INT8 cuantización (Blackwell architecture)
- RTX 5090 soporta FP8 nativo (Blackwell)
- `torch.quantization` o `bitsandbytes` para cuantizar pesos
- Podría duplicar throughput con pérdida mínima de accuracy
- Requiere validación de que la clasificación suelo/maquinaria no se degrada

#### 2d. Prefetch + pinned memory
```python
DataLoader(..., pin_memory=True, prefetch_factor=4)
```
- `pin_memory=True` permite transferencia DMA (CPU→GPU sin copia intermedia)
- `prefetch_factor=4` mantiene 4 batches pre-cargados en CPU
- Reduce latencia entre batches

### Impacto estimado: 157s → 100-130s (20-35% reducción)

---

## Prioridad 3: Smart Merge — Mejorar algoritmo en vez de velocidad

### Problema actual
Smart Merge se aborta consistentemente porque encuentra ~28M candidatos (38% del suelo) cuando el cap es ~825K. Esto significa que el algoritmo no discrimina bien entre suelo legítimo y suelo bajo maquinaria.

### Causa raíz
- `merge_radius=2.5m` es muy agresivo para nubes de alta densidad (0.25m)
- El quadrant test (3/4 cuadrantes) no es suficientemente selectivo
- En áreas planas, cualquier punto de suelo a 2.5m de maquinaria tiene vecinos en 3+ cuadrantes

### Soluciones propuestas

#### 3a. Parámetros adaptativos por densidad
```python
density = n_points / area  # pts/m²
if density > 10:   # 0.25m nube
    merge_radius = 1.0  # más conservador
    merge_neighbors = 8  # más estricto
elif density > 4:  # 0.5m nube
    merge_radius = 1.5
    merge_neighbors = 6
else:              # nubes sparse
    merge_radius = 2.5
    merge_neighbors = 4
```

#### 3b. Height-aware merge
Solo mergear suelo que está significativamente POR DEBAJO de la maquinaria cercana:
```python
# En lugar de solo XY proximity:
z_diff = maq_z_median - candidate_z
merge_valid = (xy_distance < radius) & (z_diff > min_height_gap)  # e.g., 1.5m
```

#### 3c. Gradient/slope filter
Suelo plano no debería mergearse. Solo mergear puntos donde hay un cambio brusco de pendiente:
```python
local_slope = compute_slope(candidate, neighbors)
merge_valid = merge_valid & (local_slope > slope_threshold)
```

### Impacto: Permitiría que Smart Merge funcione correctamente (no aborte), mejorando dZ de ~0.47m a ~0.16m (como en V3 donde sí funcionó)

---

## Prioridad 4: Roof Fill query GPU (`torch_cluster.radius`)

### Estado actual
Roof query: 76M puntos contra cKDTree(157K maq), k=1 → 7.4s en CPU con workers=-1

### Propuesta
```python
from torch_cluster import radius

# Transfer a GPU
maq_t = torch.tensor(maq_valid[:, :2], dtype=torch.float32, device='cuda')
cand_t = torch.tensor(cand_xy, dtype=torch.float32, device='cuda')

# Radius search: todos los pares dentro de 1.5m
edge_index = radius(maq_t, cand_t, r=1.5)
# edge_index[0] = índices de candidatos, edge_index[1] = índices de maquinaria
```

### Desafío
- 76M puntos × 2 × 4 bytes = 600MB de VRAM solo para coordenadas XY
- `torch_cluster.radius` construye estructura interna, puede necesitar más VRAM
- Necesita batching si VRAM insuficiente

### Alternativa más práctica
Dado que la maquinaria es pequeña (157K pts), la cKDTree es ya eficiente. La mejora sería de ~7s a ~2-3s. No es el bottleneck principal.

### Impacto estimado: 7.4s → 2-3s (ahorro ~5s)

---

## Prioridad 5: DBSCAN GPU (cuML RAPIDS)

### Estado actual
DBSCAN: 0.2s en CPU con 164K puntos. No es bottleneck.

### Para nubes más grandes (0.15m, 213M puntos)
Con Smart Merge funcionando, DBSCAN podría recibir ~500K-1M puntos.
sklearn DBSCAN es O(N²) en peor caso, O(N log N) con tree optimization.

### Propuesta
```python
from cuml.cluster import DBSCAN as cuDBSCAN
clustering = cuDBSCAN(eps=2.5, min_samples=30)
labels = clustering.fit_predict(gpu_array)
```

### Bloqueante
- cuML RAPIDS no está verificado para CUDA 12.8 / Blackwell
- Instalación compleja (requiere conda environment o compilación custom)
- Para 164K puntos, 0.2s no justifica la complejidad

### Impacto estimado: 0.2s → 0.05s (ahorro marginal, solo relevante para nubes >1M maq pts)

---

## Prioridad 6: Normales — Open3D Tensor batch optimization

### Estado actual
30 chunks procesados secuencialmente en GPU. 10.5s total.

### Propuesta
- Procesar múltiples chunks en batch si VRAM lo permite
- Reducir overhead de transferencia CPU↔GPU
- Usar radio adaptativo según densidad local

### Impacto estimado: 10.5s → 7-8s (ahorro ~3s)

---

## Prioridad 7: Multi-GPU support

### Para servidores con múltiples GPUs
```python
# FIX_TECHO en GPU:0, INTERPOL en GPU:1 (en paralelo)
# O pipeline: Normales→GPU:0, Inferencia→GPU:0, FIX_TECHO→GPU:1
```

### Riesgo
- Complejidad de código significativa
- Pocos usuarios tendrán multi-GPU
- Beneficio marginal dado que FIX_TECHO + INTERPOL ya son ~50s

---

## Prioridad 8: Producción — GPUs de menor potencia

### Estado actual del tier system
```python
_TIER_PARAMS = {
    'HIGH':   {'max_local_suelo': 5_000_000, 'gpu_loc_limit': 120_000},  # RTX 5090, A100
    'MEDIUM': {'max_local_suelo': 3_000_000, 'gpu_loc_limit': 60_000},   # RTX 4060-4070
    'LOW':    {'max_local_suelo': 1_500_000, 'gpu_loc_limit': 30_000},   # RTX 3060, GTX 1660
    'CPU':    {'max_local_suelo': 5_000_000, 'gpu_loc_limit': 0},        # Sin GPU
}
```

### Pendiente de validar
- **RTX 4060 (8GB):** `torch_cluster.knn` con 4M suelo × 360K maq debería funcionar (~90MB VRAM). Estimar ~15-20s para knn.
- **RTX 3060 (12GB):** Similar a RTX 4060 pero con más VRAM. Posiblemente ~15s.
- **GTX 1660 (6GB):** Compilar torch_cluster para Turing. Posible que knn funcione con downsample más agresivo (1.5M).
- **Sin GPU:** El path CPU global (1 cKDTree + 1 query) debería tardar ~40-60s para INTERPOL, ~10s para Roof Fill. Aún mucho mejor que el baseline per-tile.

### Acción requerida
1. Compilar `torch_cluster` para CUDA 11.8 / 12.1 (Ampere, Turing)
2. Benchmark en RTX 3060 y RTX 4060
3. Ajustar `_TIER_PARAMS` basado en benchmarks reales
4. Considerar `voxel_size` adaptativo en función de VRAM libre

---

## Tabla resumen de prioridades

| # | Mejora | Ahorro estimado | Esfuerzo | Riesgo |
|---|--------|-----------------|----------|--------|
| 1 | Unificar subprocesos (I/O) | ~16s | Medio | Bajo |
| 2 | Inferencia (batch/FP8/compile) | ~25-55s | Alto | Medio |
| 3 | Smart Merge algoritmo | Calidad (dZ) | Medio | Medio |
| 4 | Roof Fill GPU | ~5s | Bajo | Bajo |
| 5 | DBSCAN GPU (cuML) | ~0.15s | Alto | Alto |
| 6 | Normales batch | ~3s | Bajo | Bajo |
| 7 | Multi-GPU | Variable | Alto | Alto |
| 8 | Validación GPUs menores | Compatibilidad | Medio | Medio |

---

## Notas para implementación futura

### Patrón "Global vs Per-Element" (aplicar siempre)
Cada vez que veas un loop que procesa N elementos independientes con una estructura espacial (cKDTree, BallTree, etc.), preguntate: **¿puedo construir UNA sola estructura global y hacer UN query?**

Ejemplos exitosos en este proyecto:
- INTERPOL: 95 tiles → 1 global knn (52x más rápido)
- Roof Fill: 796 clusters → 1 global cKDTree (9x más rápido)

### Patrón "Early Abort"
Si un loop acumula resultados que se validarán al final contra un threshold, **validar incrementalmente dentro del loop** y cortar temprano si ya se excedió.

### Patrón "GPU-first, CPU-fallback"
```python
result = gpu_function(...)
if result is None:
    result = cpu_function(...)  # Mismo algoritmo, diferente backend
```
Siempre envolver GPU en try/except, hacer `torch.cuda.empty_cache()` en el except, retornar None para trigger CPU fallback.

### Cuidado con `torch.cdist`
`torch.cdist(A, B)` computa TODAS las distancias pairwise. Para k-NN search, usar `torch_cluster.knn()` que internamente usa spatial hashing y es O(N·K) en vez de O(N·M).
