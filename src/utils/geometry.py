import numpy as np
import open3d as o3d

def compute_normals_gpu(points, k=20):
    """
    Cálculo de normales usando Open3D con aceleración GPU (CUDA).
    Mucho más rápido que CPU para nubes grandes.
    
    Args:
        points: numpy array (N, 3) con coordenadas XYZ
        k: número de vecinos para estimar normales
    
    Returns:
        normals: numpy array (N, 3) con vectores normales
    """
    import open3d.core as o3c
    
    # Intentar usar CUDA
    try:
        device = o3c.Device('CUDA:0')
        # Verificar que funciona creando un tensor pequeño
        test = o3c.Tensor([1.0], device=device)
        del test
        print(f"   🚀 Usando GPU para cálculo de normales")
    except Exception:
        device = o3c.Device('CPU:0')
        print(f"   ⚠️ GPU no disponible, usando CPU")
    
    # Convertir a tensor Open3D
    points_tensor = o3c.Tensor(points.astype(np.float32), device=device)
    
    # Crear PointCloud tensor
    pcd = o3d.t.geometry.PointCloud(device)
    pcd.point.positions = points_tensor
    
    # Estimar normales en GPU
    pcd.estimate_normals(max_nn=k, radius=1.0)
    
    # Orientar normales hacia arriba (Z+)
    # Open3D tensor no tiene orient_normals, lo hacemos manualmente
    normals_tensor = pcd.point.normals.cpu().numpy()
    
    # Orientar: si Nz < 0, invertir la normal
    flip_mask = normals_tensor[:, 2] < 0
    normals_tensor[flip_mask] *= -1
    
    return normals_tensor


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