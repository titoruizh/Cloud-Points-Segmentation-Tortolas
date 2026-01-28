# Análisis de Operaciones CPU y Oportunidades para CUDA Python

**Fecha**: 28 de Enero, 2026  
**Objetivo**: Identificar cuellos de botella en CPU y evaluar viabilidad de aceleración con CUDA Python

---

## 📊 Resumen Ejecutivo

He identificado **7 áreas críticas** donde operaciones CPU están limitando el rendimiento:

| Área | Impacto | Prioridad CUDA | Complejidad |
|------|---------|----------------|-------------|
| 🔴 **Postprocesamiento (DBSCAN + KDTree)** | **CRÍTICO** | ⭐⭐⭐⭐⭐ | Media |
| 🟡 **Cálculo de Normales** | Alto | ⭐⭐⭐⭐ | Baja |
| 🟡 **Data Augmentation** | Alto | ⭐⭐⭐⭐ | Baja |
| 🟢 **Transferencias GPU↔CPU** | Medio | ⭐⭐⭐ | Muy Baja |
| 🟢 **Grid Sampling (Inferencia)** | Medio | ⭐⭐⭐ | Media |
| 🔵 **Métrica IoU** | Bajo | ⭐⭐ | Baja |
| 🔵 **Visualizaciones** | Bajo | ⭐ | N/A |

---

## 🔴 PRIORIDAD 1: Postprocesamiento (FIX_TECHO + INTERPOL)

### 📍 Ubicación
- **Archivo**: [`app_inference/core/postprocess.py`](../app_inference/core/postprocess.py)
- **Líneas críticas**:
  - L129: `tree_maq = cKDTree(xyz[idx_maq])`
  - L235: `clustering = DBSCAN(eps=2.5, min_samples=30)`
  - L275: `tree_2d = cKDTree(cluster_points[:, :2])`
  - L375: `tree = cKDTree(xyz[idx_suelo, :2])`

### 🐌 Problema Actual
```python
# DBSCAN en CPU (sklearn)
clustering = DBSCAN(eps=2.5, min_samples=30, n_jobs=-1)
labels = clustering.fit_predict(xyz[idx_maq])  # ⚠️ CPU-bound

# KDTree en CPU (scipy)
tree = cKDTree(xyz[idx_suelo, :2])
dists, neighbors = tree.query(xyz[idx_maq, :2], k=12, workers=-1)  # ⚠️ CPU-bound
```

**Benchmark estimado** (nube de 10M puntos):
- DBSCAN CPU: ~15-30 segundos
- KDTree Query CPU: ~5-10 segundos
- **Total postprocesamiento**: ~20-40 segundos

### ⚡ Solución con CUDA Python

#### Opción A: RAPIDS cuML (RECOMENDADO)
```python
# Reemplazar sklearn con RAPIDS cuML
from cuml.cluster import DBSCAN as cuDBSCAN
import cupy as cp

# Datos en GPU
xyz_gpu = cp.asarray(xyz[idx_maq])

# DBSCAN acelerado (5-10x más rápido)
clustering = cuDBSCAN(eps=2.5, min_samples=30)
labels = clustering.fit_predict(xyz_gpu).get()  # Devuelve a CPU
```

**Speedup esperado**: 5-10x (DBSCAN), 10-20x (KDTree)

#### Opción B: CuPy + Custom Kernels
```python
import cupy as cp
from cupyx.scipy.spatial import KDTree as cuKDTree

# KDTree en GPU
xyz_suelo_gpu = cp.asarray(xyz[idx_suelo, :2])
tree_gpu = cuKDTree(xyz_suelo_gpu)

xyz_maq_gpu = cp.asarray(xyz[idx_maq, :2])
dists, neighbors = tree_gpu.query(xyz_maq_gpu, k=12)

# IDW vectorizado en GPU
weights = 1.0 / cp.where(dists < 0.001, 0.001, dists)
z_neighbors = xyz_suelo_gpu[neighbors, 2]
interpolated_z = cp.sum(weights * z_neighbors, axis=1) / cp.sum(weights, axis=1)
```

### 📦 Dependencias Requeridas
```bash
pip install cupy-cuda12x  # Según versión CUDA
pip install cuml-cu12     # RAPIDS cuML
```

### 🎯 Ganancia Proyectada
- **Speedup Total**: 8-15x en postprocesamiento completo
- **Tiempo ahorrado**: 15-35 segundos por nube grande
- **Escalabilidad**: Lineal con tamaño de nube (GPU mantiene rendimiento)

---

## 🟡 PRIORIDAD 2: Cálculo de Normales

### 📍 Ubicación
- **Archivo**: [`src/utils/geometry.py`](../src/utils/geometry.py)
- **Función**: `compute_normals_fast()`, `compute_normals_open3d()`
- **Usado en**: [`app_inference/core/inference_engine.py`](../app_inference/core/inference_engine.py) L213-223

### 🐌 Problema Actual
```python
# Open3D CPU (actualmente usado)
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3.5, max_nn=30)
)  # ⚠️ KDTree en CPU, ~5-15s para 5M puntos
```

**YA TIENES** una función GPU preparada pero no se usa:
```python
def compute_normals_gpu(points, k=20):
    """🚀 Usa Open3D Tensor con CUDA - YA IMPLEMENTADA"""
    device = o3c.Device('CUDA:0')
    # ... código ya existe en geometry.py L21-40
```

### ⚡ Solución Inmediata
**No requiere CUDA Python adicional**, solo activar la función existente:

```python
# En inference_engine.py, reemplazar:
# normals = compute_normals_open3d(xyz)  # Viejo (CPU)
normals = compute_normals_gpu(xyz, k=30)  # Nuevo (GPU)
```

### 🎯 Ganancia Proyectada
- **Speedup**: 3-5x (Open3D CPU → Open3D GPU)
- **Tiempo ahorrado**: 3-10 segundos por nube
- **Sin dependencias nuevas** (Open3D ya soporta CUDA)

### ⚠️ Verificación Necesaria
Comprobar que Open3D se instaló con soporte CUDA:
```bash
python3 -c "import open3d.core as o3c; print(o3c.Device('CUDA:0'))"
```

---

## 🟡 PRIORIDAD 3: Data Augmentation (Entrenamiento)

### 📍 Ubicación
- **Archivo**: [`src/data/dataset_v6.py`](../src/data/dataset_v6.py) L45-62
- **Función**: `augment_data()`
- **Frecuencia**: Cada batch en entrenamiento (miles de veces)

### 🐌 Problema Actual
```python
def augment_data(self, xyz, normals):
    # TODOS estos pasos en CPU con NumPy:
    angle = np.random.uniform(0, 2 * np.pi)  # ⚠️
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])  # ⚠️
    xyz = np.dot(xyz, R.T)  # ⚠️
    jitter = np.clip(sigma * np.random.randn(*xyz.shape), -0.05, 0.05)  # ⚠️
    xyz = xyz + jitter  # ⚠️
```

**Problema**: Datos van GPU → CPU (augment) → GPU en cada batch.

### ⚡ Solución con CUDA Python

#### Estrategia 1: Torchvision Transforms (Recomendado)
```python
import torch
import torch.nn.functional as F

class GpuAugmentation:
    """Augmentation completamente en GPU"""
    
    def __init__(self, rotate=True, flip=True, scale_range=(0.9, 1.1), jitter_sigma=0.01):
        self.rotate = rotate
        self.flip = flip
        self.scale_range = scale_range
        self.jitter_sigma = jitter_sigma
    
    def __call__(self, xyz_tensor, normals_tensor):
        """
        Args:
            xyz_tensor: Torch tensor [N, 3] en GPU
            normals_tensor: Torch tensor [N, 3] en GPU
        Returns:
            xyz_aug, normals_aug: Tensores aumentados (en GPU)
        """
        device = xyz_tensor.device
        
        if self.rotate:
            angle = torch.rand(1, device=device) * 2 * np.pi
            c, s = torch.cos(angle), torch.sin(angle)
            R = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], 
                           device=device, dtype=xyz_tensor.dtype)
            xyz_tensor = xyz_tensor @ R.T
            normals_tensor = normals_tensor @ R.T
        
        if self.flip:
            if torch.rand(1) > 0.5:
                xyz_tensor[:, 0] *= -1
                normals_tensor[:, 0] *= -1
            if torch.rand(1) > 0.5:
                xyz_tensor[:, 1] *= -1
                normals_tensor[:, 1] *= -1
        
        # Scale
        scale = torch.empty(1, device=device).uniform_(*self.scale_range)
        xyz_tensor = xyz_tensor * scale
        
        # Jitter
        jitter = torch.randn_like(xyz_tensor) * self.jitter_sigma
        jitter = torch.clamp(jitter, -0.05, 0.05)
        xyz_tensor = xyz_tensor + jitter
        
        return xyz_tensor, normals_tensor
```

**Integración en Dataset**:
```python
class MiningDataset(Dataset):
    def __init__(self, ..., device='cuda'):
        self.device = device
        self.gpu_aug = GpuAugmentation(...) if device == 'cuda' else None
    
    def __getitem__(self, idx):
        # Cargar datos
        data = np.load(file_path).astype(np.float32)[choices, :]
        
        # Convertir DIRECTAMENTE a tensores GPU
        xyz = torch.from_numpy(data[:, 0:3]).to(self.device)
        normals = torch.from_numpy(data[:, 6:9]).to(self.device)
        
        # Augmentation EN GPU
        if self.split == "train" and self.gpu_aug:
            xyz, normals = self.gpu_aug(xyz, normals)
        
        # Resto del pipeline...
        return xyz, features, labels  # Todo ya en GPU
```

#### Estrategia 2: CuPy (Alternativa)
```python
import cupy as cp

def augment_data_gpu(xyz_cp, normals_cp):
    """Augmentation con CuPy (NumPy-like en GPU)"""
    angle = cp.random.uniform(0, 2 * cp.pi)
    c, s = cp.cos(angle), cp.sin(angle)
    R = cp.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    xyz_cp = cp.dot(xyz_cp, R.T)
    jitter = cp.clip(0.01 * cp.random.randn(*xyz_cp.shape), -0.05, 0.05)
    xyz_cp += jitter
    return xyz_cp, normals_cp
```

### 🎯 Ganancia Proyectada
- **Speedup**: 2-3x (elimina transferencias CPU↔GPU)
- **Throughput training**: +15-25% (menos overhead por batch)
- **Escalabilidad**: Crítico para batches grandes (>64)

---

## 🟢 PRIORIDAD 4: Transferencias GPU↔CPU

### 📍 Ubicación (múltiples archivos)
```python
# TRAIN_V6.py L35-37 (cada época)
xyz_np = xyz_sample.cpu().numpy()
labels_np = labels_sample.cpu().numpy()
preds_np = preds_sample.cpu().numpy()

# Todos los scripts de inferencia
probs_np = probs_batch.cpu().numpy()  # ⚠️ Sincronización forzada
```

### 🐌 Problema
Transferencias innecesarias CPU↔GPU para:
1. **Visualizaciones**: Podrían hacerse con WandB/Matplotlib en GPU
2. **Métricas temporales**: IoU puede calcularse en GPU

### ⚡ Solución

#### Para Visualizaciones (WandB)
```python
# Opción 1: Lazy transfer (solo cuando se necesita)
@torch.no_grad()
def create_visualization_lazy(xyz_gpu, labels_gpu, preds_gpu):
    # Mantener en GPU hasta último momento
    # Solo transferir puntos muestreados
    sample_idx = torch.randperm(len(xyz_gpu), device='cuda')[:15000]
    
    xyz_sample = xyz_gpu[sample_idx].cpu().numpy()  # Transfer mínimo
    labels_sample = labels_gpu[sample_idx].cpu().numpy()
    preds_sample = preds_gpu[sample_idx].cpu().numpy()
    
    return xyz_sample, labels_sample, preds_sample

# Opción 2: CuPy para NumPy-like (sin transferir)
import cupy as cp
xyz_cp = cp.asarray(xyz_gpu)  # Copia interna GPU (sin CPU)
# Procesar en GPU...
xyz_np = cp.asnumpy(xyz_cp)  # Solo transferir al final
```

#### Para Métricas
Ver siguiente sección (IoU en GPU).

### 🎯 Ganancia Proyectada
- **Latencia reducida**: -50ms por transferencia evitada
- **Memory bandwidth**: Libera PCIe para datos reales
- **Acumulativo**: En 1000 épocas = 50 segundos ahorrados

---

## 🟢 PRIORIDAD 5: Grid Sampling (Inferencia)

### 📍 Ubicación
- **Archivo**: [`app_inference/core/inference_engine.py`](../app_inference/core/inference_engine.py) L67-101
- **Clase**: `GridDatasetNitro.__getitem__()`

### 🐌 Problema Actual
```python
def __getitem__(self, idx):
    # NumPy operations en CPU
    if n_idx >= self.num_points:
        sel = np.random.choice(n_idx, self.num_points, replace=False)  # ⚠️
    
    block_data = self.full_data[selected_indices].copy()  # ⚠️ CPU array
    
    # Normalización en CPU
    block_data[:, 0] -= (tile_origin_x + self.block_size / 2.0)  # ⚠️
    block_data[:, 1] -= (tile_origin_y + self.block_size / 2.0)
    block_data[:, 2] -= np.min(xyz[:, 2])
    
    return torch.from_numpy(block_data).float(), selected_indices  # CPU→GPU
```

### ⚡ Solución con CUDA Python
```python
import cupy as cp

class GridDatasetCuda(Dataset):
    def __init__(self, full_data, grid_dict, num_points, ...):
        # Mover datos completos a GPU de una vez
        self.full_data_gpu = cp.asarray(full_data)  # ⚡ GPU
        self.grid_dict = grid_dict
        self.num_points = num_points
        # ...
    
    def __getitem__(self, idx):
        key = self.grid_keys[idx]
        indices = self.grid_dict[key]
        
        # Sampling en GPU con CuPy
        n_idx = len(indices)
        if n_idx >= self.num_points:
            sel = cp.random.choice(n_idx, self.num_points, replace=False)  # ⚡
        else:
            sel = cp.random.choice(n_idx, self.num_points, replace=True)
        
        selected_indices = indices[sel]
        
        # Indexing en GPU
        block_data = self.full_data_gpu[selected_indices].copy()  # ⚡ GPU→GPU
        
        # Normalización vectorizada en GPU
        ix, iy = key // 100000, key % 100000
        tile_origin_x = self.min_coord[0] + ix * self.block_size
        tile_origin_y = self.min_coord[1] + iy * self.block_size
        
        block_data[:, 0] -= (tile_origin_x + self.block_size / 2.0)
        block_data[:, 1] -= (tile_origin_y + self.block_size / 2.0)
        block_data[:, 2] -= cp.min(block_data[:, 2])
        
        # Convertir CuPy → PyTorch (GPU→GPU, sin pasar por CPU)
        block_tensor = torch.as_tensor(block_data, device='cuda')
        
        return block_tensor, selected_indices.get()  # Solo índices a CPU
```

**Key Point**: `torch.as_tensor(cupy_array)` hace zero-copy GPU→GPU (sin CPU).

### 🎯 Ganancia Proyectada
- **Speedup por batch**: 1.5-2x (elimina staging CPU)
- **Throughput inferencia**: +20-30%
- **Memory efficiency**: Reduce picos de RAM

---

## 🔵 PRIORIDAD 6: Cálculo de IoU

### 📍 Ubicación
- **Archivo**: [`src/utils/metrics.py`](../src/utils/metrics.py) L17-26

### 🐌 Problema Actual
```python
def add_batch(self, preds, labels):
    # Transferencia forzada a CPU
    preds = preds.detach().cpu().numpy().flatten()  # ⚠️
    labels = labels.detach().cpu().numpy().flatten()
    
    # NumPy bincount en CPU
    conteos = np.bincount(...)  # ⚠️
```

### ⚡ Solución con CUDA Python
```python
import torch

class IoUCalculatorGPU:
    """Cálculo de IoU completamente en GPU"""
    
    def __init__(self, num_classes, device='cuda'):
        self.num_classes = num_classes
        self.device = device
        self.reset()
    
    def reset(self):
        self.confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes),
            dtype=torch.long,
            device=self.device
        )
    
    def add_batch(self, preds, labels):
        """
        Args:
            preds: Tensor [B, N] en GPU
            labels: Tensor [B, N] en GPU
        """
        preds = preds.flatten()
        labels = labels.flatten()
        
        # Máscara de puntos válidos
        mask = (labels >= 0) & (labels < self.num_classes)
        
        # Bincount en GPU (PyTorch nativo)
        indices = self.num_classes * labels[mask] + preds[mask]
        conteos = torch.bincount(
            indices,
            minlength=self.num_classes ** 2
        )
        
        self.confusion_matrix += conteos.reshape(self.num_classes, self.num_classes)
    
    def compute_iou(self):
        """Retorna IoU por clase y mIoU (en CPU)"""
        intersection = torch.diag(self.confusion_matrix)
        union = (self.confusion_matrix.sum(dim=1) +
                self.confusion_matrix.sum(dim=0) - intersection)
        
        iou = intersection.float() / (union.float() + 1e-10)
        miou = iou.mean()
        
        return iou.cpu().numpy(), miou.item()
```

### 🎯 Ganancia Proyectada
- **Speedup**: 2-3x (GPU bincount vs NumPy)
- **Memoria**: -50% (evita copias CPU)
- **Impacto total**: Bajo (no es cuello de botella crítico)

---

## 📋 Plan de Implementación Recomendado

### Fase 1: Quick Wins (1-2 días) 🚀
**Sin instalar CUDA Python, solo activar código existente**

1. ✅ **Activar cálculo de normales GPU** (ya existe en `geometry.py`)
   - Cambiar `compute_normals_open3d()` → `compute_normals_gpu()`
   - Ganancia: 3-5x en esa etapa
   
2. ✅ **Reducir transferencias CPU↔GPU innecesarias**
   - Lazy transfer en visualizaciones (solo muestrear puntos necesarios)
   - Ganancia: -50ms por época

### Fase 2: RAPIDS Integration (3-5 días) ⚡
**Instalar RAPIDS para aceleración máxima**

```bash
# Instalación RAPIDS
conda install -c rapidsai -c conda-forge -c nvidia \
    cuml=24.12 cupy cudatoolkit=12.5
```

1. ✅ **Reemplazar DBSCAN y KDTree con RAPIDS cuML**
   - Modificar `app_inference/core/postprocess.py`
   - Ganancia: 8-15x en postprocesamiento
   
2. ✅ **Grid Sampling con CuPy**
   - Modificar `GridDatasetNitro` en `inference_engine.py`
   - Ganancia: 1.5-2x en inferencia

### Fase 3: Training Optimization (5-7 días) 🔥
**Maximizar throughput de entrenamiento**

1. ✅ **Data Augmentation en GPU** (Torch-based)
   - Modificar `src/data/dataset_v6.py`
   - Ganancia: +15-25% throughput
   
2. ✅ **IoU Calculator en GPU**
   - Modificar `src/utils/metrics.py`
   - Ganancia: Menor impacto, pero libera CPU

### Fase 4: Advanced (Opcional, 10-15 días) 🎯
**Custom CUDA kernels para operaciones específicas**

1. 🔬 **Fused KDTree + IDW Kernel**
   - Fusionar búsqueda de vecinos + interpolación
   - Ganancia: 2-3x adicional sobre cuML
   
2. 🔬 **Multi-GPU Scaling**
   - Distribuir inferencia en múltiples GPUs
   - Ganancia: Lineal con # GPUs

---

## 💰 Análisis Costo-Beneficio

### Inversión de Tiempo
| Fase | Días Dev | Complejidad | ROI |
|------|----------|-------------|-----|
| Fase 1 | 1-2 | Muy Baja | Alto |
| Fase 2 | 3-5 | Media | Muy Alto |
| Fase 3 | 5-7 | Media | Alto |
| Fase 4 | 10-15 | Alta | Medio |

### Ganancia Total Estimada
**Para una nube típica de 10M puntos**:

| Pipeline | Tiempo Actual | Con CUDA (Fase 2) | Speedup |
|----------|---------------|-------------------|---------|
| Inferencia | 45s | 30s | 1.5x |
| Postproceso | 35s | 4s | 8.75x |
| **Total** | **80s** | **34s** | **2.35x** |

**En entrenamiento (1000 épocas)**:
- Actual: ~8-12 horas
- Con CUDA (Fase 3): ~5-7 horas
- **Ahorro**: 3-5 horas por sweep

---

## 🔧 Verificación de Hardware

Antes de implementar, verificar capacidades CUDA:

```bash
# 1. Verificar versión CUDA
nvidia-smi

# 2. Verificar PyTorch CUDA
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, v{torch.version.cuda}')"

# 3. Verificar Open3D CUDA
python3 -c "import open3d.core as o3c; print(o3c.Device('CUDA:0'))"

# 4. Test CuPy (después de instalar)
python3 -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount())"

# 5. Test RAPIDS (después de instalar)
python3 -c "from cuml.cluster import DBSCAN; print('cuML OK')"
```

---

## 📚 Recursos y Documentación

### Librerías CUDA Python
- **CuPy**: https://docs.cupy.dev/ (NumPy-like en GPU)
- **RAPIDS cuML**: https://docs.rapids.ai/api/cuml/stable/ (Scikit-learn en GPU)
- **PyTorch CUDA**: https://pytorch.org/docs/stable/cuda.html
- **Open3D GPU**: http://www.open3d.org/docs/latest/tutorial/core/tensor.html

### Tutoriales Relevantes
1. [CuPy Interoperability with PyTorch](https://docs.cupy.dev/en/stable/user_guide/interoperability.html)
2. [RAPIDS cuML DBSCAN](https://docs.rapids.ai/api/cuml/stable/api/#dbscan)
3. [PyTorch Custom Datasets on GPU](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)

---

## ⚠️ Consideraciones Importantes

### Memoria GPU
- **RTX 5090**: 32GB VRAM (suficiente para todo)
- Postproceso RAPIDS: ~2-4GB adicionales
- CuPy Grid Dataset: ~1-2GB (datos en GPU)
- **Margen seguro**: >10GB libre

### Compatibilidad
- RAPIDS requiere CUDA 11.x o 12.x
- CuPy debe coincidir con versión CUDA instalada
- Open3D GPU requiere compilación especial (verificar instalación)

### Debugging
CUDA puede fallar silenciosamente. Siempre:
```python
import cupy as cp
cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)  # Mejor gestión memoria
cp.cuda.Stream.null.synchronize()  # Forzar ejecución antes de benchmark
```

---

## 🎯 Conclusión y Recomendación

### TL;DR
**SÍ vale la pena incorporar CUDA Python**, especialmente:

1. **RAPIDS cuML** para postprocesamiento (8-15x speedup)
2. **CuPy** para Grid Dataset y augmentation (1.5-3x speedup)
3. **Open3D GPU** para normales (3-5x speedup, ya disponible)

### Next Steps
1. **Hoy**: Activar `compute_normals_gpu()` existente
2. **Esta semana**: Instalar RAPIDS y probar DBSCAN/KDTree
3. **Próxima semana**: Implementar augmentation en GPU
4. **Mes que viene**: Evaluar custom kernels si es necesario

**Speedup total realista**: 2-3x end-to-end (inferencia + postproceso)

---

**Autor**: GitHub Copilot  
**Contacto**: Para dudas sobre implementación específica, consultar código en archivos mencionados.
