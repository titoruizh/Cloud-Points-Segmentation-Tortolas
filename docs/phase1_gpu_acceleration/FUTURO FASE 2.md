📑 Plan de Optimización Fase 2: Aceleración de Post-Procesamiento
Objetivo: Eliminar el último gran cuello de botella del pipeline (Clustering DBSCAN) que actualmente corre en CPU (Scikit-Learn) y tarda entre 20-40 segundos por nube grande.

Meta de Rendimiento: Reducir el tiempo de inferencia total de ~80s a ~35s.

1. Estrategia A: Implementación Nativa en PyTorch (La Recomendada)
Esta es la opción preferida porque no requiere instalar nada nuevo. Utiliza la potencia bruta de tu RTX 5090 mediante álgebra matricial en PyTorch para simular el algoritmo DBSCAN.

Ventaja: Mantiene el Docker limpio (sin dependency hell). 100% GPU.

Desventaja: Requiere mantener una función personalizada compleja.

Snippet de Implementación (dbscan_pytorch):
Guardar esto en src/utils/clustering.py en el futuro.

Python
import torch

def dbscan_pytorch(points_gpu, eps=2.5, min_samples=30):
    """
    DBSCAN simplificado en PyTorch puro. Optimizado para GPU RTX 5090.
    """
    N = len(points_gpu)
    device = points_gpu.device
    
    # Inicialización
    labels = torch.full((N,), -1, dtype=torch.long, device=device)
    visited = torch.zeros(N, dtype=torch.bool, device=device)
    cluster_id = 0
    
    # Procesamiento por chunks para no saturar VRAM
    chunk_size = 10000 
    
    for start_idx in range(0, N, chunk_size):
        end_idx = min(start_idx + chunk_size, N)
        chunk = points_gpu[start_idx:end_idx]
        
        # 1. Calcular distancias (La parte pesada, hecha en GPU)
        dists = torch.cdist(chunk, points_gpu, p=2)
        
        # 2. Encontrar vecinos
        neighbors_mask = dists <= eps
        neighbor_counts = neighbors_mask.sum(dim=1)
        
        # 3. Identificar puntos núcleo (Core Points)
        core_mask = neighbor_counts >= min_samples
        
        # 4. Expansión de clusters (BFS)
        for i, is_core in enumerate(core_mask):
            global_idx = start_idx + i
            if visited[global_idx] or not is_core:
                continue
            
            # Iniciar nuevo cluster
            queue = [global_idx]
            visited[global_idx] = True
            labels[global_idx] = cluster_id
            
            while queue:
                current = queue.pop(0)
                # Obtener vecinos del punto actual
                # Nota: Esto puede optimizarse más con kernels custom si fuera necesario
                current_neighbors = torch.where(neighbors_mask[current - start_idx])[0]
                
                for neighbor_idx in current_neighbors:
                    if not visited[neighbor_idx]:
                        visited[neighbor_idx] = True
                        labels[neighbor_idx] = cluster_id
                        # Si el vecino también es core, expandir desde él
                        if neighbor_counts[neighbor_idx - start_idx] >= min_samples:
                            queue.append(neighbor_idx.item())
            
            cluster_id += 1
            
    return labels.cpu().numpy()
2. Estrategia B: Open3D C++ (El "Plan B" Seguro)
Si la implementación de PyTorch te da problemas de memoria o lógica, Open3D tiene un DBSCAN escrito en C++ que es mucho más rápido que el de Python (Scikit-Learn), aunque corre en CPU.

Ventaja: Código muy simple (una línea). Muy estable.

Desventaja: Sigue usando CPU, por lo que hay una pequeña transferencia de datos.

Implementación:

Python
import open3d as o3d
import numpy as np

def cluster_open3d_cpu(xyz_numpy, eps=2.5, min_points=30):
    # Crear objeto PointCloud de Open3D (Legacy CPU)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_numpy)

    # Ejecutar clustering optimizado en C++
    labels = np.array(pcd.cluster_dbscan(
        eps=eps, 
        min_points=min_points, 
        print_progress=False
    ))
    return labels
3. Estrategia C: Integración de RAPIDS cuML (Alto Riesgo / Alta Recompensa)
Esta opción implica instalar las librerías oficiales de NVIDIA para Data Science. Es la más rápida (DBSCAN nativo de GPU), pero es la que descartamos hoy por riesgo de romper tu Docker con dependencias de CUDA.

Cuándo activar esto: Solo si NVIDIA lanza soporte oficial de RAPIDS para CUDA 12.8 (probablemente a mediados/finales de 2025).

Cómo se vería el comando (Futuro):

Dockerfile
# ADVERTENCIA: Solo intentar en el futuro si las versiones coinciden
RUN pip install --no-cache-dir \
    --extra-index-url https://pypi.nvidia.com \
    cudf-cu12==25.* \
    cuml-cu12==25.* ```

---

### 📊 Resumen de Impacto Estimado

| Métrica | Fase 1 (Actual) | Fase 2 (Con Estrategia A/C) |
| :--- | :--- | :--- |
| **Tiempo Inferencia** | ~80 seg | **~35 seg** |
| **Uso de CPU** | Alto (DBSCAN) | **Bajo (Todo en GPU)** |
| **Complejidad** | Media | **Alta (Lógica custom)** |

Guarda esto en tu documentación del proyecto como `FUTURE_ROADMAP.md`. ¡Ahora enfócate en entrenar con la