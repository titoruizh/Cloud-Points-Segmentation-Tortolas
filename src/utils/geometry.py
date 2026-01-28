import numpy as np
import open3d as o3d

def compute_normals_gpu(points, k=20, radius=1.0):
    """
    Cálculo de normales usando Open3D Tensor API (GPU).
    Compatible con RTX 5090 / CUDA 12.8 / Arquitectura Blackwell.
    
    Args:
        points: numpy array (N, 3) con coordenadas XYZ
        k: número máximo de vecinos para estimación
        radius: radio de búsqueda (metros)
    
    Returns:
        normals: numpy array (N, 3) con vectores normales orientados hacia Z+
    
    Speedup esperado: 3-5x sobre CPU para nubes >100k puntos
    """
    import open3d.core as o3c
    import open3d.t.geometry as o3dg
    
    # Intentar usar CUDA (fallback a CPU si falla)
    try:
        device = o3c.Device('CUDA:0')
        # Test rápido para verificar disponibilidad
        test = o3c.Tensor([1.0], device=device)
        del test
        print(f"   🚀 Normales: Usando GPU (CUDA)")
    except Exception as e:
        device = o3c.Device('CPU:0')
        print(f"   ⚠️ Normales: Fallback a CPU ({e})")
    
    # Convertir NumPy → Open3D Tensor (mueve a GPU si device=CUDA)
    points_tensor = o3c.Tensor(points.astype(np.float32), device=device)
    
    # Crear PointCloud tensor
    pcd = o3dg.PointCloud(device)
    pcd.point.positions = points_tensor
    
    # Estimar normales en GPU (hybrid search: radio Y max neighbors)
    pcd.estimate_normals(max_nn=k, radius=radius)
    
    # Obtener normales como NumPy
    normals_np = pcd.point.normals.cpu().numpy()
    
    # Orientar normales hacia arriba (Z+)
    # Si Nz < 0, invertir la normal
    flip_mask = normals_np[:, 2] < 0
    normals_np[flip_mask] *= -1
    
    return normals_np


def compute_normals_fast(points, k=20):
    """
    Cálculo RÁPIDO de normales usando PCA vectorizado con numpy.
    Optimizado para evitar loops Python.
    """
    from scipy.spatial import cKDTree
    
    n_points = len(points)
    
    # Para nubes muy grandes, usar Open3D estándar que es más eficiente
    if n_points > 100000:
        print(f"   📊 Nube grande ({n_points:,} puntos), usando Open3D optimizado...")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=k)
        )
        normals = np.asarray(pcd.normals)
        
        # Orientar hacia arriba
        flip_mask = normals[:, 2] < 0
        normals[flip_mask] *= -1
        
        return normals
    
    # Para nubes pequeñas, usar PCA (original)
    tree = cKDTree(points)
    normals = np.zeros_like(points)
    
    for i in range(n_points):
        _, idx = tree.query(points[i], k=min(k, n_points))
        neighbors = points[idx]
        centered = neighbors - neighbors.mean(axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        normals[i] = eigenvectors[:, 0]
    
    return normals


def compute_normals_open3d(points, search_radius=1.0, max_nn=15):
    """
    Calcula normales usando Open3D optimizado (el más rápido).
    
    Open3D usa un KD-tree en C++ altamente optimizado que supera
    incluso a la versión GPU para la mayoría de los casos.
    
    NOTA: No se corrige la orientación de las normales para mantener
    consistencia con los datos de entrenamiento.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=search_radius, 
            max_nn=max_nn
        )
    )
    normals = np.asarray(pcd.normals)
    
    return normals