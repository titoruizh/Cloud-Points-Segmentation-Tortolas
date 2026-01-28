"""
Ejemplos Prácticos de CUDA Python para Cloud Point Research
=============================================================
Código listo para copiar/pegar en tu proyecto.

Instalación requerida:
    pip install cupy-cuda12x  # Ajustar según versión CUDA
    conda install -c rapidsai -c conda-forge cuml cudatoolkit=12.5
"""

import numpy as np
import torch
import cupy as cp
from typing import Tuple

# ============================================================================
# EJEMPLO 1: Postprocesamiento con RAPIDS (DBSCAN + KDTree)
# ============================================================================

def ejemplo_postprocess_rapids():
    """
    Reemplazo drop-in para postprocess.py con RAPIDS cuML.
    Speedup esperado: 8-15x
    """
    from cuml.cluster import DBSCAN as cuDBSCAN
    from cuml.neighbors import NearestNeighbors
    
    # Datos de ejemplo (simula maquinaria detectada)
    xyz_maq = np.random.rand(50000, 3).astype(np.float32)
    xyz_suelo = np.random.rand(5000000, 3).astype(np.float32)
    
    # ========== DBSCAN en GPU ==========
    print("🔥 Ejecutando DBSCAN en GPU...")
    xyz_maq_gpu = cp.asarray(xyz_maq)
    
    clustering = cuDBSCAN(eps=2.5, min_samples=30, verbose=0)
    labels = clustering.fit_predict(xyz_maq_gpu)
    
    # Convertir resultado a CPU si necesario
    labels_cpu = labels.get() if hasattr(labels, 'get') else cp.asnumpy(labels)
    
    n_clusters = len(set(labels_cpu)) - (1 if -1 in labels_cpu else 0)
    print(f"   ✅ Clusters encontrados: {n_clusters}")
    
    # ========== KDTree 2D en GPU ==========
    print("🌳 Construyendo KDTree en GPU...")
    xyz_suelo_2d_gpu = cp.asarray(xyz_suelo[:, :2])
    xyz_maq_2d_gpu = cp.asarray(xyz_maq[:, :2])
    
    # NearestNeighbors de cuML (equivalente a KDTree)
    nn_model = NearestNeighbors(n_neighbors=12, metric='euclidean')
    nn_model.fit(xyz_suelo_2d_gpu)
    
    distances, indices = nn_model.kneighbors(xyz_maq_2d_gpu)
    
    # Filtrar por distancia
    proximity_radius = 1.5
    valid_mask = (distances[:, 0] <= proximity_radius).get()
    
    print(f"   ✅ Puntos válidos: {valid_mask.sum()}")
    
    return labels_cpu, valid_mask


# ============================================================================
# EJEMPLO 2: IDW Interpolation en GPU (INTERPOL)
# ============================================================================

def interpol_idw_gpu(xyz_maq: np.ndarray, 
                     xyz_suelo: np.ndarray,
                     k_neighbors: int = 12,
                     max_dist: float = 50.0) -> np.ndarray:
    """
    Interpolación IDW (Inverse Distance Weighting) en GPU.
    
    Args:
        xyz_maq: Puntos de maquinaria (N, 3)
        xyz_suelo: Puntos de suelo (M, 3)
        k_neighbors: Número de vecinos
        max_dist: Distancia máxima de búsqueda
    
    Returns:
        interpolated_z: Alturas interpoladas para maquinaria
    
    Speedup esperado: 10-20x sobre scipy.spatial.cKDTree
    """
    from cuml.neighbors import NearestNeighbors
    
    # Mover a GPU
    xyz_suelo_2d_gpu = cp.asarray(xyz_suelo[:, :2], dtype=cp.float32)
    xyz_maq_2d_gpu = cp.asarray(xyz_maq[:, :2], dtype=cp.float32)
    z_suelo_gpu = cp.asarray(xyz_suelo[:, 2], dtype=cp.float32)
    
    # KNN Search en GPU
    nn_model = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
    nn_model.fit(xyz_suelo_2d_gpu)
    
    distances, indices = nn_model.kneighbors(xyz_maq_2d_gpu)
    
    # Máscara de vecinos válidos (dentro de max_dist)
    valid_mask = distances <= max_dist
    
    # IDW (vectorizado en GPU)
    distances = cp.where(distances < 0.001, 0.001, distances)  # Evitar div/0
    weights = 1.0 / distances
    
    # Aplicar máscara de distancia
    weights = cp.where(valid_mask, weights, 0.0)
    
    # Obtener alturas de vecinos
    z_neighbors = z_suelo_gpu[indices]
    
    # IDW formula: sum(w_i * z_i) / sum(w_i)
    numerator = cp.sum(weights * z_neighbors, axis=1)
    denominator = cp.sum(weights, axis=1)
    
    # Evitar división por cero (puntos sin vecinos válidos)
    denominator = cp.where(denominator > 0, denominator, 1.0)
    interpolated_z = numerator / denominator
    
    # Devolver a CPU como NumPy
    return cp.asnumpy(interpolated_z)


# ============================================================================
# EJEMPLO 3: Data Augmentation en GPU (Entrenamiento)
# ============================================================================

class GpuAugmentation:
    """
    Data Augmentation completamente en GPU usando PyTorch.
    Elimina transferencias CPU↔GPU.
    
    Uso:
        augment = GpuAugmentation()
        xyz_aug, normals_aug = augment(xyz_gpu, normals_gpu)
    """
    
    def __init__(self, 
                 rotate: bool = True,
                 flip: bool = True,
                 scale_range: Tuple[float, float] = (0.9, 1.1),
                 jitter_sigma: float = 0.01,
                 jitter_clip: float = 0.05):
        self.rotate = rotate
        self.flip = flip
        self.scale_range = scale_range
        self.jitter_sigma = jitter_sigma
        self.jitter_clip = jitter_clip
    
    def __call__(self, xyz: torch.Tensor, normals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz: Tensor [N, 3] en GPU
            normals: Tensor [N, 3] en GPU
        
        Returns:
            xyz_aug, normals_aug (ambos en GPU)
        """
        device = xyz.device
        dtype = xyz.dtype
        
        # ===== ROTACIÓN Z =====
        if self.rotate:
            angle = torch.rand(1, device=device, dtype=dtype) * 2 * np.pi
            c, s = torch.cos(angle), torch.sin(angle)
            
            # Matriz de rotación en Z
            R = torch.tensor([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]
            ], device=device, dtype=dtype)
            
            xyz = xyz @ R.T
            normals = normals @ R.T
        
        # ===== FLIP ALEATORIO =====
        if self.flip:
            if torch.rand(1, device=device) > 0.5:
                xyz[:, 0] *= -1
                normals[:, 0] *= -1
            if torch.rand(1, device=device) > 0.5:
                xyz[:, 1] *= -1
                normals[:, 1] *= -1
        
        # ===== SCALE ALEATORIO =====
        scale = torch.empty(1, device=device, dtype=dtype).uniform_(*self.scale_range)
        xyz = xyz * scale
        
        # ===== JITTER =====
        jitter = torch.randn_like(xyz) * self.jitter_sigma
        jitter = torch.clamp(jitter, -self.jitter_clip, self.jitter_clip)
        xyz = xyz + jitter
        
        return xyz, normals


# ============================================================================
# EJEMPLO 4: Grid Sampling en GPU (Inferencia)
# ============================================================================

class GridSamplerGPU:
    """
    Sampling de bloques grid en GPU usando CuPy.
    Elimina staging en CPU.
    """
    
    def __init__(self, full_data: np.ndarray, num_points: int):
        """
        Args:
            full_data: Nube completa [N, C] en CPU
            num_points: Puntos por bloque
        """
        # Mover datos COMPLETOS a GPU (una sola vez)
        self.full_data_gpu = cp.asarray(full_data, dtype=cp.float32)
        self.num_points = num_points
    
    def sample_block(self, indices: np.ndarray) -> torch.Tensor:
        """
        Samplea un bloque y devuelve tensor PyTorch en GPU.
        
        Args:
            indices: Índices del bloque (en CPU como NumPy)
        
        Returns:
            block_tensor: PyTorch tensor en GPU [num_points, C]
        """
        # Convertir índices a CuPy
        indices_gpu = cp.asarray(indices)
        
        # Sampling en GPU
        n_idx = len(indices_gpu)
        if n_idx >= self.num_points:
            sel = cp.random.choice(n_idx, self.num_points, replace=False)
        else:
            sel = cp.random.choice(n_idx, self.num_points, replace=True)
        
        selected_indices = indices_gpu[sel]
        
        # Indexing en GPU (GPU→GPU)
        block_data = self.full_data_gpu[selected_indices].copy()
        
        # Normalización en GPU (ejemplo: centrar en 0)
        block_data[:, :3] -= cp.mean(block_data[:, :3], axis=0, keepdims=True)
        
        # Convertir CuPy → PyTorch (GPU→GPU, zero-copy)
        block_tensor = torch.as_tensor(block_data, device='cuda')
        
        return block_tensor


# ============================================================================
# EJEMPLO 5: IoU Calculator en GPU
# ============================================================================

class IoUCalculatorGPU:
    """
    Cálculo de IoU completamente en GPU usando PyTorch.
    Compatible con API existente.
    """
    
    def __init__(self, num_classes: int, device: str = 'cuda'):
        self.num_classes = num_classes
        self.device = device
        self.reset()
    
    def reset(self):
        """Resetea la matriz de confusión."""
        self.confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes),
            dtype=torch.long,
            device=self.device
        )
    
    def add_batch(self, preds: torch.Tensor, labels: torch.Tensor):
        """
        Añade un batch a la matriz de confusión.
        
        Args:
            preds: Predicciones [B, N] en GPU
            labels: Ground truth [B, N] en GPU
        """
        preds = preds.flatten()
        labels = labels.flatten()
        
        # Máscara de puntos válidos
        mask = (labels >= 0) & (labels < self.num_classes)
        
        # Calcular índices de la matriz
        indices = self.num_classes * labels[mask] + preds[mask]
        
        # Bincount en GPU
        conteos = torch.bincount(
            indices,
            minlength=self.num_classes ** 2
        )
        
        # Actualizar matriz
        self.confusion_matrix += conteos.reshape(self.num_classes, self.num_classes)
    
    def compute_iou(self) -> Tuple[np.ndarray, float]:
        """
        Calcula IoU por clase y mIoU.
        
        Returns:
            iou_per_class: Array NumPy [num_classes]
            miou: Float (media)
        """
        # Intersección (diagonal)
        intersection = torch.diag(self.confusion_matrix)
        
        # Unión (fila + columna - intersección)
        union = (self.confusion_matrix.sum(dim=1) +
                self.confusion_matrix.sum(dim=0) - intersection)
        
        # IoU = Intersección / Unión
        iou = intersection.float() / (union.float() + 1e-10)
        miou = iou.mean()
        
        # Devolver en CPU (compatible con código existente)
        return iou.cpu().numpy(), miou.item()
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Devuelve matriz de confusión en CPU."""
        return self.confusion_matrix.cpu().numpy()


# ============================================================================
# EJEMPLO 6: Cálculo de Normales GPU (Open3D Tensor)
# ============================================================================

def compute_normals_open3d_gpu(points: np.ndarray, 
                               search_radius: float = 1.0,
                               max_nn: int = 30) -> np.ndarray:
    """
    Calcula normales usando Open3D con aceleración GPU.
    
    Args:
        points: NumPy array [N, 3]
        search_radius: Radio de búsqueda
        max_nn: Máximo de vecinos
    
    Returns:
        normals: NumPy array [N, 3]
    
    Speedup esperado: 3-5x sobre CPU
    """
    import open3d as o3d
    import open3d.core as o3c
    
    # Intentar GPU
    try:
        device = o3c.Device('CUDA:0')
        # Test rápido
        test = o3c.Tensor([1.0], device=device)
        del test
        print("   🚀 Usando GPU para normales")
    except:
        device = o3c.Device('CPU:0')
        print("   ⚠️ GPU no disponible, fallback a CPU")
    
    # Crear PointCloud tensor
    pcd = o3d.t.geometry.PointCloud(device)
    pcd.point.positions = o3c.Tensor(points.astype(np.float32), device=device)
    
    # Estimar normales en GPU
    pcd.estimate_normals(max_nn=max_nn, radius=search_radius)
    
    # Orientar hacia arriba (Z+)
    normals = pcd.point.normals.cpu().numpy()
    flip_mask = normals[:, 2] < 0
    normals[flip_mask] *= -1
    
    return normals


# ============================================================================
# EJEMPLO 7: Benchmark Comparativo CPU vs GPU
# ============================================================================

def benchmark_cpu_vs_gpu():
    """
    Compara rendimiento CPU vs GPU en operaciones clave.
    """
    import time
    
    print("\n" + "="*60)
    print("🏁 BENCHMARK: CPU vs GPU")
    print("="*60)
    
    # Dataset de prueba
    n_points = 1_000_000
    xyz = np.random.rand(n_points, 3).astype(np.float32)
    
    # ===== TEST 1: Random Sampling =====
    print("\n1️⃣ Random Sampling (100k de 1M puntos)")
    
    # CPU (NumPy)
    t0 = time.time()
    indices_cpu = np.random.choice(n_points, 100000, replace=False)
    sample_cpu = xyz[indices_cpu]
    t_cpu = time.time() - t0
    print(f"   CPU (NumPy):  {t_cpu*1000:.2f} ms")
    
    # GPU (CuPy)
    xyz_gpu = cp.asarray(xyz)
    t0 = time.time()
    indices_gpu = cp.random.choice(n_points, 100000, replace=False)
    sample_gpu = xyz_gpu[indices_gpu]
    cp.cuda.Stream.null.synchronize()  # Forzar ejecución
    t_gpu = time.time() - t0
    print(f"   GPU (CuPy):   {t_gpu*1000:.2f} ms")
    print(f"   ⚡ Speedup:    {t_cpu/t_gpu:.2f}x")
    
    # ===== TEST 2: Normalización =====
    print("\n2️⃣ Normalización (centrar + escalar)")
    
    # CPU
    t0 = time.time()
    mean_cpu = np.mean(xyz, axis=0)
    std_cpu = np.std(xyz, axis=0)
    normalized_cpu = (xyz - mean_cpu) / (std_cpu + 1e-8)
    t_cpu = time.time() - t0
    print(f"   CPU (NumPy):  {t_cpu*1000:.2f} ms")
    
    # GPU
    t0 = time.time()
    mean_gpu = cp.mean(xyz_gpu, axis=0)
    std_gpu = cp.std(xyz_gpu, axis=0)
    normalized_gpu = (xyz_gpu - mean_gpu) / (std_gpu + 1e-8)
    cp.cuda.Stream.null.synchronize()
    t_gpu = time.time() - t0
    print(f"   GPU (CuPy):   {t_gpu*1000:.2f} ms")
    print(f"   ⚡ Speedup:    {t_cpu/t_gpu:.2f}x")
    
    # ===== TEST 3: KDTree Query =====
    print("\n3️⃣ KDTree Query (12 vecinos más cercanos)")
    
    # CPU (scipy)
    from scipy.spatial import cKDTree
    query_points = xyz[:10000]  # 10k queries
    
    t0 = time.time()
    tree_cpu = cKDTree(xyz)
    dists_cpu, idx_cpu = tree_cpu.query(query_points, k=12, workers=-1)
    t_cpu = time.time() - t0
    print(f"   CPU (scipy):  {t_cpu*1000:.2f} ms")
    
    # GPU (cuML)
    try:
        from cuml.neighbors import NearestNeighbors
        query_gpu = cp.asarray(query_points)
        
        t0 = time.time()
        nn_model = NearestNeighbors(n_neighbors=12)
        nn_model.fit(xyz_gpu)
        dists_gpu, idx_gpu = nn_model.kneighbors(query_gpu)
        cp.cuda.Stream.null.synchronize()
        t_gpu = time.time() - t0
        print(f"   GPU (cuML):   {t_gpu*1000:.2f} ms")
        print(f"   ⚡ Speedup:    {t_cpu/t_gpu:.2f}x")
    except ImportError:
        print("   GPU (cuML):   ⚠️ No instalado")
    
    print("\n" + "="*60)


# ============================================================================
# MAIN: Ejecutar ejemplos
# ============================================================================

if __name__ == "__main__":
    print("\n🚀 CUDA Python Examples - Cloud Point Research\n")
    
    # Verificar disponibilidad
    print("📋 Verificando disponibilidad de librerías...")
    
    try:
        import cupy as cp
        print("   ✅ CuPy instalado")
        print(f"      CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
        print(f"      GPU count: {cp.cuda.runtime.getDeviceCount()}")
    except ImportError:
        print("   ❌ CuPy NO instalado")
        print("      Instalar: pip install cupy-cuda12x")
    
    try:
        from cuml.cluster import DBSCAN
        print("   ✅ RAPIDS cuML instalado")
    except ImportError:
        print("   ❌ RAPIDS cuML NO instalado")
        print("      Instalar: conda install -c rapidsai cuml")
    
    try:
        import open3d.core as o3c
        device = o3c.Device('CUDA:0')
        print("   ✅ Open3D GPU disponible")
    except:
        print("   ⚠️ Open3D GPU NO disponible (funciona pero en CPU)")
    
    # Ejecutar ejemplos si están disponibles las librerías
    try:
        print("\n" + "="*60)
        print("EJEMPLO: Postprocesamiento con RAPIDS")
        print("="*60)
        ejemplo_postprocess_rapids()
    except ImportError as e:
        print(f"⚠️ Saltando ejemplo: {e}")
    
    # Benchmark
    try:
        benchmark_cpu_vs_gpu()
    except ImportError as e:
        print(f"⚠️ Saltando benchmark: {e}")
