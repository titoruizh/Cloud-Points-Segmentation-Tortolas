import numpy as np
import open3d as o3d
from datetime import datetime


def _estimate_normals_on_device(chunk_pts: np.ndarray, k: int, radius: float,
                                 o3d_device) -> np.ndarray:
    """Estima normales en el device dado (GPU o CPU) usando Open3D Tensor API."""
    import open3d.core as o3c
    import open3d.t.geometry as o3dg

    pcd = o3dg.PointCloud(o3d_device)
    pcd.point.positions = o3c.Tensor(
        chunk_pts.astype(np.float32), device=o3d_device
    )
    pcd.estimate_normals(max_nn=k, radius=radius)
    return pcd.point.normals.cpu().numpy().astype(np.float32)


def compute_normals_gpu(points, k=20, radius=1.0, chunk_size_m=500.0, overlap_m=5.0):
    """
    Cálculo de normales en CHUNKS ESPACIALES usando GPU (Open3D Tensor API).

    Cada chunk se procesa en GPU independientemente, manteniendo el uso de
    VRAM bajo (~100-500 MB por chunk) sin importar el tamaño total de la nube.

    Args:
        points:        numpy array (N, 3) con coordenadas XYZ
        k:             número máximo de vecinos
        radius:        radio de búsqueda (metros)
        chunk_size_m:  tamaño de celda espacial en metros (default 200m)
        overlap_m:     margen de solape para normales correctas en bordes

    Returns:
        normals: numpy array (N, 3) orientadas hacia Z+
    """
    import open3d.core as o3c

    n_total = len(points)
    normals_out = np.zeros((n_total, 3), dtype=np.float32)

    # ── Detectar GPU una sola vez ──────────────────────────────────────────
    gpu_available = False
    try:
        dev_gpu = o3c.Device('CUDA:0')
        test = o3c.Tensor([1.0, 2.0, 3.0], device=dev_gpu)
        del test
        gpu_available = True
        print(f"   🔥 Normales: usando GPU (Open3D Tensor CUDA)")
    except Exception as e:
        dev_gpu = None
        print(f"   ⚠️  GPU no disponible para Open3D ({e}), usando CPU")

    dev_cpu = o3c.Device('CPU:0')
    active_device = dev_gpu if gpu_available else dev_cpu

    # ── Grilla espacial ────────────────────────────────────────────────────
    min_x, min_y = points[:, 0].min(), points[:, 1].min()
    max_x, max_y = points[:, 0].max(), points[:, 1].max()

    cols = int(np.ceil((max_x - min_x) / chunk_size_m)) or 1
    rows = int(np.ceil((max_y - min_y) / chunk_size_m)) or 1
    total_chunks = cols * rows

    print(f"   📐 Nube: {n_total:,} puntos → ~{total_chunks} chunks "
          f"({cols}×{rows}) de {chunk_size_m:.0f}m")

    t0 = datetime.now()
    idx_all = np.arange(n_total)
    processed = 0

    for ci in range(cols):
        for ri in range(rows):
            # Zona con overlap para contexto del KDTree
            cx0 = min_x + ci * chunk_size_m - overlap_m
            cx1 = min_x + (ci + 1) * chunk_size_m + overlap_m
            cy0 = min_y + ri * chunk_size_m - overlap_m
            cy1 = min_y + (ri + 1) * chunk_size_m + overlap_m

            mask = (
                (points[:, 0] >= cx0) & (points[:, 0] < cx1) &
                (points[:, 1] >= cy0) & (points[:, 1] < cy1)
            )
            chunk_global_idx = idx_all[mask]
            chunk_pts = points[chunk_global_idx]

            if len(chunk_pts) < 10:
                continue

            # Zona "core" (sin overlap) — única región que escribimos
            core_mask_local = (
                (chunk_pts[:, 0] >= min_x + ci * chunk_size_m) &
                (chunk_pts[:, 0] <  min_x + (ci + 1) * chunk_size_m) &
                (chunk_pts[:, 1] >= min_y + ri * chunk_size_m) &
                (chunk_pts[:, 1] <  min_y + (ri + 1) * chunk_size_m)
            )
            core_local_idx  = np.where(core_mask_local)[0]
            core_global_idx = chunk_global_idx[core_local_idx]

            if len(core_global_idx) == 0:
                continue

            # ── Calcular normales (GPU first, fallback CPU) ───────────────
            try:
                chunk_normals = _estimate_normals_on_device(
                    chunk_pts, k, radius, active_device
                )
            except Exception as e:
                if gpu_available:
                    # Una sola vez: avisar y degradar a CPU para el resto
                    print(f"   ⚠️  Chunk GPU falló ({e}), degradando a CPU...")
                    gpu_available = False
                    active_device = dev_cpu
                    chunk_normals = _estimate_normals_on_device(
                        chunk_pts, k, radius, active_device
                    )
                else:
                    raise

            # Orientar hacia Z+
            flip = chunk_normals[:, 2] < 0
            chunk_normals[flip] *= -1

            # Escribir sólo los puntos core
            normals_out[core_global_idx] = chunk_normals[core_local_idx]

            processed += 1
            elapsed = (datetime.now() - t0).total_seconds()
            rate   = processed / elapsed if elapsed > 0 else 0
            eta    = (total_chunks - processed) / rate if rate > 0 else 0
            pts_s  = int(sum(
                np.sum(mask) for _ in [None]  # puntos por segundo approx
            ) / elapsed * processed) if elapsed > 0 else 0

            print(f"   ⚡ Chunk {processed}/{total_chunks} "
                  f"| core={len(core_global_idx):,} pts "
                  f"| {elapsed:.0f}s elapsed  ETA {eta:.0f}s")

    total_elapsed = (datetime.now() - t0).total_seconds()
    pts_per_s = int(n_total / total_elapsed) if total_elapsed > 0 else 0
    print(f"   ✅ Normales completadas: {total_elapsed:.1f}s  "
          f"({pts_per_s:,} pts/s)")

    return normals_out


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