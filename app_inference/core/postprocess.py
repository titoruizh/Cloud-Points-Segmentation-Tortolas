"""
Pipeline de Postprocesamiento
==============================
Incluye PRE_CLEAN_SURFACE, FIX_TECHO e INTERPOL (Bulldozer Digital).
Wrappers de los scripts originales para uso desde la UI.
"""

import os
import shutil
import numpy as np
import laspy
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
from typing import Callable, Optional
from dataclasses import dataclass
from datetime import datetime
import concurrent.futures


def _detect_gpu_tier():
    """
    Detecta VRAM libre y devuelve tier adaptativo.

    Returns:
        (tier_name, free_vram_gb, torch.device | None)
        tier_name: 'HIGH' (>=20GB), 'MEDIUM' (8-20GB), 'LOW' (<8GB), 'CPU'
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return ('CPU', 0.0, None)
        props = torch.cuda.get_device_properties(0)
        total = props.total_memory / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
        free = total - reserved
        dev = torch.device('cuda')
        if free >= 20:
            return ('HIGH', free, dev)
        elif free >= 8:
            return ('MEDIUM', free, dev)
        else:
            return ('LOW', free, dev)
    except Exception:
        return ('CPU', 0.0, None)


_TIER_PARAMS = {
    'HIGH':   {'max_local_suelo': 5_000_000, 'gpu_loc_limit': 120_000},
    'MEDIUM': {'max_local_suelo': 3_000_000, 'gpu_loc_limit': 60_000},
    'LOW':    {'max_local_suelo': 1_500_000, 'gpu_loc_limit': 30_000},
    'CPU':    {'max_local_suelo': 5_000_000, 'gpu_loc_limit': 0},
}


@dataclass
class FixTechoConfig:
    """Configuración para FIX_TECHO."""
    eps: float = 2.5           # Radio DBSCAN (metros)
    min_samples: int = 30      # Mínimo puntos para cluster
    z_buffer: float = 1.5      # Altura mínima desde suelo
    max_height: float = 8.0    # Altura máxima de maquinaria
    padding: float = 1.5       # Margen XY para búsqueda
    proximity_radius: float = 1.5  # Radio 2D para confirmar relleno de techos
    ground_class: int = 2      # Clase del suelo
    
    # Smart Gap Filling
    smart_merge: bool = False  # Si activar el pre-procesado de unión
    merge_radius: float = 2.5  # Radio de búsqueda de vecinos
    merge_neighbors: int = 4   # Mínimo de vecinos maquinaria para convertir


@dataclass
class InterpolConfig:
    """Configuración para INTERPOL (Bulldozer)."""
    k_neighbors: int = 12      # Vecinos para IDW
    max_dist: float = 50.0     # Distancia máxima de búsqueda
    max_local_suelo: int = 5_000_000  # Cap de suelo por chunk (voxel-downsampling si excede)


@dataclass
class PreCleanConfig:
    """
    Configuración de limpieza previa a FIX_TECHO.

    Esta etapa busca:
    - Reducir ruido de clase 1 (islas pequeñas y puntos espurios)
    - Mantener estructuras altas/legítimas (protección por altura)
    - Rellenar micro-huecos internos antes de la expansión volumétrica
    """
    enabled: bool = False

    # Detección de ruido en maquinaria por clustering
    cluster_eps: float = 1.2
    cluster_min_samples: int = 6
    remove_dbscan_noise: bool = True

    # Regla para "cluster chico" (aislado)
    small_cluster_max_points: int = 35

    # Protección para no borrar estructuras delgadas pero altas (ej. mástiles)
    protect_tall_structures: bool = True
    protected_min_height: float = 1.8

    # Relleno de micro-huecos (suelo rodeado por maquinaria)
    hole_fill_enabled: bool = True
    hole_fill_radius: float = 1.2
    hole_fill_k: int = 12
    hole_fill_min_class1_ratio: float = 0.70

    ground_class: int = 2


@dataclass
class PostprocessResult:
    """Resultado del postprocesamiento."""
    success: bool
    input_file: str
    output_file: str
    step_name: str
    points_modified: int = 0
    processing_time: float = 0.0
    error_message: str = ""


def _smart_merge_gpu_chunk(
    cands_xyz: np.ndarray,
    maq_xyz: np.ndarray,
    merge_radius: float,
    merge_neighbors: int,
    device,
    gpu_loc_limit: int = 120_000,
    maq_q_chunk: int = 8_000,
):
    """
    GPU Smart Merge para un bloque de candidatos (PyTorch).

    Aplica pre-filtro espacial XY para limitar la maquinaria local y luego
    usa torch.cdist en sub-batches para contar vecinos dentro de merge_radius,
    verificando el test de rodeado por al menos 3 cuadrantes.

    Args:
        cands_xyz:     (N_cand, 3) float64 – candidatos suelo para este chunk
        maq_xyz:       (N_maq,  3) float64 – toda la maquinaria
        merge_radius:  radio de búsqueda (metros)
        merge_neighbors: mínimo vecinos maquinaria para pasar primer filtro
        device:        torch.device('cuda')
        gpu_loc_limit: si N_loc > este umbral, retorna None (señal para CPU)
        maq_q_chunk:   chunk de maquinaria para el check de cuadrantes

    Returns:
        np.ndarray (int64) con índices locales (0..N_cand-1) que superan
        el test de surroundedness, o array vacío.
        Retorna None si N_loc > gpu_loc_limit o si ocurre CUDA OOM.
    """
    import torch

    r = float(merge_radius)

    # 1. Pre-filtro espacial XY: maq dentro del bbox de este chunk + r
    bmin = cands_xyz[:, :2].min(axis=0)   # (2,)
    bmax = cands_xyz[:, :2].max(axis=0)   # (2,)
    lmask = (
        (maq_xyz[:, 0] >= bmin[0] - r) & (maq_xyz[:, 0] <= bmax[0] + r) &
        (maq_xyz[:, 1] >= bmin[1] - r) & (maq_xyz[:, 1] <= bmax[1] + r)
    )
    lmaq = maq_xyz[lmask].astype(np.float32)
    N_loc = len(lmaq)

    if N_loc == 0:
        return np.array([], dtype=np.int64)

    if N_loc > gpu_loc_limit:
        return None   # Demasiada maquinaria local → delegar a cKDTree CPU

    # 2. Sub-batch de candidatos: dists (N_sub × N_loc × 4B) < 2 GB
    sub_sz = max(200, min(50_000, int(2 * 1024 ** 3 // (N_loc * 4))))

    maq_t  = torch.from_numpy(lmaq).to(device)   # (N_loc, 3)
    maq_xy = maq_t[:, :2]                         # (N_loc, 2) – para cuadrantes
    N_cand = len(cands_xyz)
    all_valid: list = []

    try:
        for s in range(0, N_cand, sub_sz):
            e       = min(s + sub_sz, N_cand)
            cand_t  = torch.from_numpy(
                cands_xyz[s:e].astype(np.float32)
            ).to(device)                           # (N_sub, 3)

            with torch.no_grad():
                dists  = torch.cdist(cand_t, maq_t)   # (N_sub, N_loc) float32
                within = dists <= r                    # bool  (N_sub, N_loc)
                del dists

                ok_idx = (
                    within.sum(dim=1) >= merge_neighbors
                ).nonzero(as_tuple=True)[0]            # (N_ok,)

                if len(ok_idx) > 0:
                    wok    = within[ok_idx]            # (N_ok, N_loc) bool
                    del within
                    cok_xy = cand_t[ok_idx, :2]        # (N_ok, 2)
                    del cand_t

                    N_ok    = len(ok_idx)
                    q_flags = torch.zeros(N_ok, dtype=torch.uint8, device=device)

                    # Iterar sobre maquinaria en sub-chunks para no OOM en cuadrantes
                    for mq in range(0, N_loc, maq_q_chunk):
                        mq_e  = min(mq + maq_q_chunk, N_loc)
                        mxy_q = maq_xy[mq:mq_e]         # (mc, 2)
                        wq    = wok[:, mq:mq_e]          # (N_ok, mc) bool
                        # dx/dy = maq - cand  →  (N_ok, mc)
                        dx = mxy_q[:, 0].unsqueeze(0) - cok_xy[:, 0].unsqueeze(1)
                        dy = mxy_q[:, 1].unsqueeze(0) - cok_xy[:, 1].unsqueeze(1)
                        q_flags |=  ((dx > 0) & (dy > 0) & wq).any(dim=1).to(torch.uint8)
                        q_flags |= (((dx < 0) & (dy > 0) & wq).any(dim=1).to(torch.uint8) << 1)
                        q_flags |= (((dx < 0) & (dy < 0) & wq).any(dim=1).to(torch.uint8) << 2)
                        q_flags |= (((dx > 0) & (dy < 0) & wq).any(dim=1).to(torch.uint8) << 3)
                        del dx, dy, wq, mxy_q

                    pop  = ((q_flags & 1) + ((q_flags >> 1) & 1) +
                            ((q_flags >> 2) & 1) + ((q_flags >> 3) & 1))
                    surr = pop >= 3
                    valid = ok_idx[surr].cpu().numpy() + s
                    if len(valid) > 0:
                        all_valid.append(valid)
                    del wok, cok_xy, q_flags, pop, surr, ok_idx

                else:
                    del within, cand_t

    except Exception as _gpu_exc:
        # Captura CUDA OOM u otros errores GPU → delegar a CPU
        try:
            import torch as _t
            _t.cuda.empty_cache()
        except Exception:
            pass
        del maq_t
        return None

    del maq_t
    try:
        import torch as _t2
        _t2.cuda.empty_cache()
    except Exception:
        pass

    return (
        np.concatenate(all_valid).astype(np.int64)
        if all_valid
        else np.array([], dtype=np.int64)
    )


def _interpol_idw_gpu_knn(
    lm_xy: np.ndarray,
    ls_xy: np.ndarray,
    ls_z: np.ndarray,
    k_neighbors: int,
    max_dist: float,
    device,
):
    """
    IDW interpolation en GPU usando torch_cluster.knn.

    Reemplaza cKDTree build + query por búsqueda knn nativa en GPU.
    VRAM ~90MB por tile (2.4M suelo × 67K maq), cabe en cualquier GPU.

    Args:
        lm_xy:  (N_maq, 2) float32 - XY maquinaria
        ls_xy:  (N_suelo, 2) float32 - XY suelo
        ls_z:   (N_suelo,) float32 - Z suelo
        k_neighbors: vecinos IDW (12)
        max_dist: radio máximo (50m)
        device: torch.device('cuda')

    Returns:
        (interpolated_z, valid_mask) numpy arrays, o None si falla GPU.
    """
    import torch
    from torch_cluster import knn

    N_maq = len(lm_xy)
    N_suelo = len(ls_xy)

    if N_maq == 0 or N_suelo == 0:
        return np.zeros(N_maq, dtype=np.float32), np.zeros(N_maq, dtype=bool)

    k = min(k_neighbors, N_suelo)

    try:
        ground_t = torch.from_numpy(ls_xy.astype(np.float32)).to(device)
        maq_t = torch.from_numpy(lm_xy.astype(np.float32)).to(device)
        ground_z_t = torch.from_numpy(ls_z.astype(np.float32)).to(device)

        with torch.no_grad():
            # knn(x, y, k): busca en x los k vecinos de cada punto de y
            # Retorna edge_index [2, N_maq * k]
            #   [0] = índices en y (maq), [1] = índices en x (suelo)
            edge_index = knn(ground_t, maq_t, k)

            nn_idx = edge_index[1].reshape(N_maq, k)

            # Distancias manuales (knn no las retorna)
            neighbor_xy = ground_t[nn_idx]
            diff = neighbor_xy - maq_t.unsqueeze(1)
            dists = torch.sqrt((diff ** 2).sum(dim=2))

            del edge_index, neighbor_xy, diff

            # Filtro max_dist
            too_far = dists > max_dist
            dists[too_far] = float('inf')

            has_valid = torch.isfinite(dists[:, 0])

            # IDW: w = 1/d, Z = sum(w*z) / sum(w)
            d_clamped = dists.clamp(min=0.001)
            fin_mask = torch.isfinite(dists)
            weights = torch.where(fin_mask, 1.0 / d_clamped, torch.zeros_like(d_clamped))

            z_neighbors = ground_z_t[nn_idx]
            num = (weights * z_neighbors).sum(dim=1)
            den = weights.sum(dim=1) + 1e-9
            interp_z = num / den

            result_z = interp_z.cpu().numpy().astype(np.float32)
            valid_mask = has_valid.cpu().numpy()

        del ground_t, maq_t, ground_z_t, nn_idx, dists, too_far
        del d_clamped, fin_mask, weights, z_neighbors, num, den, interp_z, has_valid
        torch.cuda.empty_cache()

        return (result_z, valid_mask)

    except Exception:
        try:
            import torch as _t
            _t.cuda.empty_cache()
        except Exception:
            pass
        return None


def _postprocess_worker(step: str, input_file: str, output_file: str,
                        config_dict: dict, queue):
    """
    Worker para ejecutar postprocesamiento en proceso hijo SEPARADO.

    Se invoca via multiprocessing.get_context('spawn').Process() para que
    arranque con memoria limpia (~200 MB) en vez de heredar los 40+ GB
    del heap de inferencia.  Comunica progreso via Queue.

    Args:
        step: 'fix_techo' o 'interpol'
        input_file: Ruta al archivo de entrada
        output_file: Ruta al archivo de salida
        config_dict: Dict con parámetros del dataclass de config
        queue: multiprocessing.Queue para enviar progreso y resultado
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    def progress(msg):
        queue.put(('progress', msg))
        print(msg, flush=True)

    try:
        if step == 'fix_techo':
            cfg = FixTechoConfig(**config_dict)
            pp = PostProcessor(fix_techo_config=cfg)
            result = pp.run_fix_techo(input_file, output_file, progress)
        elif step == 'interpol':
            cfg = InterpolConfig(**config_dict)
            pp = PostProcessor(interpol_config=cfg)
            result = pp.run_interpol(input_file, output_file, progress)
        else:
            raise ValueError(f"Step desconocido: {step}")

        # Enviar resultado como dict (dataclass no siempre es picklable entre procesos)
        queue.put(('result', {
            'success': result.success,
            'points_modified': result.points_modified,
            'processing_time': result.processing_time,
            'error_message': result.error_message,
        }))
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"\n❌ ERROR en worker {step}:\n{tb}", flush=True)
        queue.put(('result', {
            'success': False,
            'points_modified': 0,
            'processing_time': 0.0,
            'error_message': str(e),
        }))


class PostProcessor:
    """
    Pipeline de postprocesamiento para las predicciones.

    Pasos:
    1. PRE_CLEAN_SURFACE: limpieza de ruido y relleno de micro-huecos
    2. FIX_TECHO: relleno volumétrico de techos de maquinaria
    3. INTERPOL: Bulldozer digital (IDW) para generar DTM limpio
    """
    
    def __init__(self, 
                 fix_techo_config: Optional[FixTechoConfig] = None,
                 interpol_config: Optional[InterpolConfig] = None,
                 pre_clean_config: Optional[PreCleanConfig] = None):
        """
        Args:
            fix_techo_config: Configuración para FIX_TECHO
            interpol_config: Configuración para INTERPOL
            pre_clean_config: Configuración para PRE_CLEAN_SURFACE
        """
        self.fix_techo_config = fix_techo_config or FixTechoConfig()
        self.interpol_config = interpol_config or InterpolConfig()
        self.pre_clean_config = pre_clean_config or PreCleanConfig()

    def _copy_without_changes(self, input_file: str, output_file: str):
        """Copia binaria del archivo LAZ/LAS cuando no se requiere modificación."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        shutil.copy2(input_file, output_file)

    def _write_las_with_classification(self, las, output_file: str,
                                       classification: np.ndarray,
                                       z_values: Optional[np.ndarray] = None):
        """Escribe un archivo LAS con nueva clasificación y opcionalmente nuevos valores Z"""
        # Crear la estructura base respetando versión y formato
        new_las = laspy.create(
            point_format=las.header.point_format,
            file_version=las.header.version
        )

        # Copiar offsets y scales originales para no dañar coordenadas
        new_las.header.offsets = las.header.offsets
        new_las.header.scales = las.header.scales

        # Opcionalmente reescribir z
        new_las.x = las.x
        new_las.y = las.y
        if z_values is not None:
            new_las.z = z_values
        else:
            new_las.z = las.z

        new_las.classification = classification

        if hasattr(las, 'red'):
            new_las.red = las.red
            new_las.green = las.green
            new_las.blue = las.blue

        out_dir = os.path.dirname(output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        new_las.write(output_file)

    def _write_las_from_disk(self, input_file: str, output_file: str,
                             classification: np.ndarray,
                             z_values: Optional[np.ndarray] = None):
        """
        Re-lee el archivo original desde disco para escribir la salida.
        Permite liberar el objeto las durante el procesamiento y solo
        recargar al momento de guardar, evitando mantener ~7 GB en RAM.
        """
        import gc
        las = laspy.read(input_file)
        new_las = laspy.create(
            point_format=las.header.point_format,
            file_version=las.header.version
        )
        new_las.header.offsets = las.header.offsets
        new_las.header.scales = las.header.scales
        new_las.x = las.x
        new_las.y = las.y
        if z_values is not None:
            new_las.z = z_values
        else:
            new_las.z = las.z
        new_las.classification = classification
        if hasattr(las, 'red'):
            new_las.red = las.red
            new_las.green = las.green
            new_las.blue = las.blue
        del las
        gc.collect()
        out_dir = os.path.dirname(output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        new_las.write(output_file)
        del new_las
        gc.collect()

    def run_pre_clean(self, input_file: str, output_file: str,
                      progress_callback: Optional[Callable] = None) -> PostprocessResult:
        """
        Limpieza previa de superficie antes de FIX_TECHO.

        Flujo:
        1) DBSCAN sobre clase 1 para aislar ruido (islas/fragmentos pequeños)
        2) Regla geométrica de clusters chicos (N, altura, área XY)
        3) Filtro local de soporte vecinal (elimina outliers de clase 1)
        4) Relleno de micro-huecos en suelo rodeado por maquinaria
        """
        start_time = datetime.now()
        result = PostprocessResult(
            success=False,
            input_file=input_file,
            output_file=output_file,
            step_name="PRE_CLEAN_SURFACE"
        )

        try:
            cfg = self.pre_clean_config

            if not cfg.enabled:
                if progress_callback:
                    progress_callback("🧹 PRE_CLEAN: desactivado, copiando archivo...")
                self._copy_without_changes(input_file, output_file)
                result.success = True
                result.processing_time = (datetime.now() - start_time).total_seconds()
                return result

            if progress_callback:
                progress_callback(f"🧹 PRE_CLEAN: Cargando {os.path.basename(input_file)}...")

            las = laspy.read(input_file)
            xyz = np.vstack((las.x, las.y, las.z)).transpose()
            classification = np.array(las.classification)
            original_classification = classification.copy()

            idx_maq = np.where(classification == 1)[0]
            if len(idx_maq) == 0:
                if progress_callback:
                    progress_callback("⚠️ PRE_CLEAN: no hay maquinaria. Copiando archivo.")
                self._copy_without_changes(input_file, output_file)
                result.success = True
                result.processing_time = (datetime.now() - start_time).total_seconds()
                return result

            if progress_callback:
                progress_callback(f"   🚜 Maquinaria inicial: {len(idx_maq):,} puntos")
                progress_callback("   🧩 PRE_CLEAN: clustering de ruido (DBSCAN)...")

            # Paso A: clustering de maquinaria para aislar ruido fino.
            clustering = DBSCAN(eps=cfg.cluster_eps, min_samples=cfg.cluster_min_samples, n_jobs=-1)
            labels = clustering.fit_predict(xyz[idx_maq])

            removed_dbscan_noise = 0
            removed_small_clusters = 0

            if cfg.remove_dbscan_noise:
                noise_mask = labels == -1
                if np.any(noise_mask):
                    noise_indices = idx_maq[noise_mask]
                    classification[noise_indices] = cfg.ground_class
                    removed_dbscan_noise = len(noise_indices)

            # Paso B: remover clusters chicos.
            # Paso B: remover clusters chicos en PARALELO
            unique_labels = [lbl for lbl in set(labels) if lbl != -1]
            
            def _eval_preclean_cluster(lbl):
                local_indices = np.where(labels == lbl)[0]
                cluster_indices = idx_maq[local_indices]
                cluster_points = xyz[cluster_indices]

                n_points = len(cluster_indices)
                height = float(np.max(cluster_points[:, 2]) - np.min(cluster_points[:, 2]))

                is_small_cluster = n_points <= cfg.small_cluster_max_points
                protected_by_height = cfg.protect_tall_structures and height >= cfg.protected_min_height

                if is_small_cluster and not protected_by_height:
                    return cluster_indices, n_points
                return np.array([], dtype=int), 0

            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(executor.map(_eval_preclean_cluster, unique_labels))

            indices_to_remove = []
            for cluster_indices, n_points in results:
                if len(cluster_indices) > 0:
                    indices_to_remove.append(cluster_indices)
                    removed_small_clusters += n_points

            if indices_to_remove:
                all_indices_to_remove = np.concatenate(indices_to_remove)
                classification[all_indices_to_remove] = cfg.ground_class

            # (Paso C eliminado: filtro de soporte local)

            # Paso D: relleno de micro-huecos de suelo rodeado por maquinaria.
            added_hole_fill = 0
            if cfg.hole_fill_enabled:
                idx_maq = np.where(classification == 1)[0]
                if len(idx_maq) > 0:
                    if progress_callback:
                        progress_callback("   🩹 PRE_CLEAN: relleno de micro-huecos...")

                    min_m = np.min(xyz[idx_maq], axis=0)
                    max_m = np.max(xyz[idx_maq], axis=0)
                    margin = max(cfg.hole_fill_radius * 2.0, 0.5)

                    candidate_mask = (
                        (classification == cfg.ground_class) &
                        (xyz[:, 0] >= min_m[0] - margin) & (xyz[:, 0] <= max_m[0] + margin) &
                        (xyz[:, 1] >= min_m[1] - margin) & (xyz[:, 1] <= max_m[1] + margin) &
                        (xyz[:, 2] >= min_m[2] - margin) & (xyz[:, 2] <= max_m[2] + margin)
                    )
                    candidate_indices = np.where(candidate_mask)[0]

                    if len(candidate_indices) > 0:
                        tree_maq_xy = cKDTree(xyz[idx_maq, :2])
                        k_hole = max(1, int(cfg.hole_fill_k))

                        dists, _ = tree_maq_xy.query(
                            xyz[candidate_indices, :2],
                            k=k_hole,
                            distance_upper_bound=cfg.hole_fill_radius,
                            workers=-1
                        )

                        if dists.ndim == 1:
                            dists = dists[:, None]

                        neighbor_presence_ratio = np.mean(np.isfinite(dists), axis=1)
                        fill_mask = neighbor_presence_ratio >= cfg.hole_fill_min_class1_ratio
                        fill_indices = candidate_indices[fill_mask]

                        if len(fill_indices) > 0:
                            classification[fill_indices] = 1
                            added_hole_fill = len(fill_indices)

            changed_points = int(np.sum(classification != original_classification))
            result.points_modified = changed_points

            if progress_callback:
                progress_callback(
                    "   ✅ PRE_CLEAN resumen: "
                    f"ruido_dbscan={removed_dbscan_noise:,}, "
                    f"clusters_chicos={removed_small_clusters:,}, "
                    f"huecos_rellenados={added_hole_fill:,}"
                )

            self._write_las_with_classification(las, output_file, classification)

            result.success = True
            result.processing_time = (datetime.now() - start_time).total_seconds()
            if progress_callback:
                progress_callback(f"💾 PRE_CLEAN guardado: {os.path.basename(output_file)}")

        except Exception as e:
            result.error_message = str(e)
            if progress_callback:
                progress_callback(f"❌ Error en PRE_CLEAN: {str(e)}")

        return result
        
    def run_fix_techo(self, input_file: str, output_file: str,
                       progress_callback: Optional[Callable] = None) -> PostprocessResult:
        """
        Ejecuta el relleno volumétrico de techos.
        
        Identifica clusters de maquinaria y rellena puntos de suelo
        que estén dentro del bounding box de cada cluster.
        
        Args:
            input_file: Archivo LAZ con clasificación
            output_file: Archivo de salida
            progress_callback: Función para reportar progreso
            
        Returns:
            PostprocessResult con el resultado
        """
        start_time = datetime.now()
        result = PostprocessResult(
            success=False,
            input_file=input_file,
            output_file=output_file,
            step_name="FIX_TECHO"
        )
        
        try:
            cfg = self.fix_techo_config
            
            if progress_callback:
                progress_callback(f"🏗️ FIX_TECHO: Cargando {os.path.basename(input_file)}...")
            print(f"\n🏗️ FIX_TECHO iniciando: {os.path.basename(input_file)}", flush=True)

            # RSS para diagnóstico de OOM
            try:
                import resource
                rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                print(f"   📊 RSS al iniciar FIX_TECHO: {rss_mb:.0f} MB", flush=True)
            except Exception:
                pass

            las = laspy.read(input_file)
            xyz = np.vstack((
                np.asarray(las.x, dtype=np.float32),
                np.asarray(las.y, dtype=np.float32),
                np.asarray(las.z, dtype=np.float32)
            )).transpose()
            classification = np.asarray(las.classification, dtype=np.uint8)
            # Liberar objeto las (~7 GB para nubes grandes) - re-leeremos al guardar
            del las
            import gc
            gc.collect()

            # Buscar maquinaria
            idx_maq = np.where(classification == 1)[0]

            if len(idx_maq) == 0:
                if progress_callback:
                    progress_callback("⚠️ No hay maquinaria detectada. Copiando archivo sin cambios.")
                # Copiar archivo sin cambios
                import shutil
                shutil.copy2(input_file, output_file)
                result.success = True
                result.processing_time = (datetime.now() - start_time).total_seconds()
                return result
                
            if progress_callback:
                progress_callback(f"   🚜 Maquinaria: {len(idx_maq):,} puntos")
                progress_callback("   🧩 Clusterizando con DBSCAN...")
                
            # --- SMART MERGE / GAP FILLING ---
            if cfg.smart_merge:
                if progress_callback:
                    progress_callback("   🧠 Ejecutando Smart Merge (Gap Filling)...")

                maq_xyz_np = xyz[idx_maq]   # referencia numpy (pre-filtro GPU y fallback CPU)

                # Detectar GPU disponible
                _use_gpu = False
                _gpu_dev = None
                try:
                    import torch as _torch
                    if _torch.cuda.is_available():
                        _use_gpu = True
                        _gpu_dev = _torch.device('cuda')
                        if progress_callback:
                            progress_callback(
                                f"   ⚡ Smart Merge GPU: {_torch.cuda.get_device_name(0)}"
                            )
                except Exception:
                    pass

                # cKDTree CPU – construcción lazy, solo si algún chunk lo requiere
                _tree_maq_cpu = None

                # BBox de maquinaria para filtrar candidatos suelo
                min_m, max_m = np.min(maq_xyz_np, axis=0), np.max(maq_xyz_np, axis=0)
                margin = cfg.merge_radius * 2

                # Filtro rápido (Box)
                mask_ground_candidates = (
                    (classification == cfg.ground_class) &
                    (xyz[:, 0] >= min_m[0] - margin) & (xyz[:, 0] <= max_m[0] + margin) &
                    (xyz[:, 1] >= min_m[1] - margin) & (xyz[:, 1] <= max_m[1] + margin) &
                    (xyz[:, 2] >= min_m[2] - margin) & (xyz[:, 2] <= max_m[2] + margin)
                )
                idx_ground_candidates = np.where(mask_ground_candidates)[0]

                if len(idx_ground_candidates) > 0:
                    t_sm = datetime.now()
                    CHUNK_SZ = 500_000
                    n_cands  = len(idx_ground_candidates)
                    all_valid_merges = []
                    accumulated_count = 0
                    _max_merge = max(500_000, len(idx_maq) * 5)
                    num_blocks = (n_cands + CHUNK_SZ - 1) // CHUNK_SZ

                    mode_str = "GPU+CPU fallback" if _use_gpu else "CPU"
                    if progress_callback:
                        progress_callback(
                            f"   🔍 Smart Merge [{mode_str}]: {n_cands:,} candidatos "
                            f"en {num_blocks} bloques (abort cap: {_max_merge:,})"
                        )
                    print(f"   🔍 Smart Merge: {n_cands:,} candidatos, cap={_max_merge:,}", flush=True)

                    early_aborted = False
                    blocks_processed = 0
                    for chunk_start in range(0, n_cands, CHUNK_SZ):
                        # Early abort: stop scanning if already over threshold
                        if accumulated_count > _max_merge:
                            early_aborted = True
                            ea_msg = (f"   ⚡ Early abort: {accumulated_count:,} > {_max_merge:,} "
                                      f"tras {blocks_processed}/{num_blocks} bloques")
                            if progress_callback:
                                progress_callback(ea_msg)
                            print(ea_msg, flush=True)
                            break

                        chunk_end    = min(chunk_start + CHUNK_SZ, n_cands)
                        chunk_idx_gc = idx_ground_candidates[chunk_start:chunk_end]
                        chunk_xyz    = xyz[chunk_idx_gc]
                        blocks_processed += 1

                        used_gpu = False
                        if _use_gpu:
                            gpu_result = _smart_merge_gpu_chunk(
                                chunk_xyz, maq_xyz_np,
                                cfg.merge_radius, cfg.merge_neighbors,
                                _gpu_dev,
                            )
                            if gpu_result is not None:
                                used_gpu = True
                                if len(gpu_result) > 0:
                                    all_valid_merges.append(chunk_idx_gc[gpu_result])
                                    accumulated_count += len(gpu_result)

                        if not used_gpu:
                            # Construir cKDTree la primera vez que hace falta
                            if _tree_maq_cpu is None:
                                if progress_callback:
                                    progress_callback("   🌳 Construyendo cKDTree CPU (fallback)...")
                                _tree_maq_cpu = cKDTree(maq_xyz_np)

                            neighbor_list = _tree_maq_cpu.query_ball_point(
                                chunk_xyz, r=cfg.merge_radius
                            )
                            lens = np.array([len(x) for x in neighbor_list])
                            mask_count = lens >= cfg.merge_neighbors
                            candidates_to_check = chunk_idx_gc[mask_count]
                            neighbor_list_filtered = [
                                neighbor_list[i]
                                for i in range(len(neighbor_list)) if mask_count[i]
                            ]
                            del neighbor_list
                            import gc as _gc
                            _gc.collect()

                            if len(candidates_to_check) == 0:
                                continue

                            all_neighbors = np.concatenate(neighbor_list_filtered)
                            lens_filtered = lens[mask_count]
                            candidate_indices_repeated = np.repeat(candidates_to_check, lens_filtered)
                            del neighbor_list_filtered

                            P_query = xyz[candidate_indices_repeated, :2]
                            P_neigh = maq_xyz_np[all_neighbors][:, :2]

                            diff = P_neigh - P_query
                            dx   = diff[:, 0]
                            dy   = diff[:, 1]

                            q_mask = np.zeros(len(dx), dtype=np.uint8)
                            q_mask[(dx > 0) & (dy > 0)] = 1
                            q_mask[(dx < 0) & (dy > 0)] = 2
                            q_mask[(dx < 0) & (dy < 0)] = 4
                            q_mask[(dx > 0) & (dy < 0)] = 8

                            reduce_indices = np.repeat(
                                np.arange(len(candidates_to_check)), lens_filtered
                            )
                            accumulated_quadrants = np.zeros(
                                len(candidates_to_check), dtype=np.uint8
                            )
                            np.bitwise_or.at(accumulated_quadrants, reduce_indices, q_mask)

                            pop_count = (
                                (accumulated_quadrants & 1) +
                                ((accumulated_quadrants >> 1) & 1) +
                                ((accumulated_quadrants >> 2) & 1) +
                                ((accumulated_quadrants >> 3) & 1)
                            )
                            final_mask   = pop_count >= 3
                            valid_merges = candidates_to_check[final_mask]
                            if len(valid_merges) > 0:
                                all_valid_merges.append(valid_merges)
                                accumulated_count += len(valid_merges)

                    sm_time = (datetime.now() - t_sm).total_seconds()

                    # Aplicar todos los merges de una vez
                    del maq_xyz_np
                    if _tree_maq_cpu is not None:
                        del _tree_maq_cpu
                    if all_valid_merges:
                        all_valid = np.concatenate(all_valid_merges)
                        if len(all_valid) > _max_merge or early_aborted:
                            abort_msg = (
                                f"   ⚠️ Smart Merge abortado: {len(all_valid):,} pts "
                                f"exceden umbral ({_max_merge:,} = 5× maq original) "
                                f"en {sm_time:.1f}s ({blocks_processed}/{num_blocks} bloques)"
                            )
                            if progress_callback:
                                progress_callback(abort_msg)
                            print(abort_msg, flush=True)
                        else:
                            classification[all_valid] = 1
                            idx_maq = np.concatenate((idx_maq, all_valid))
                            ok_msg = f"   ✨ Smart Merge: {len(all_valid):,} puntos unidos en {sm_time:.1f}s"
                            if progress_callback:
                                progress_callback(ok_msg)
                            print(ok_msg, flush=True)
                    else:
                        sm_msg = f"   📊 Smart Merge: 0 candidatos válidos en {sm_time:.1f}s"
                        if progress_callback:
                            progress_callback(sm_msg)
                        print(sm_msg, flush=True)
            
            # Clustering
            t_dbscan = datetime.now()
            clustering = DBSCAN(eps=cfg.eps, min_samples=cfg.min_samples, n_jobs=-1)
            labels = clustering.fit_predict(xyz[idx_maq])
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            dbscan_time = (datetime.now() - t_dbscan).total_seconds()

            db_msg = f"   🧩 DBSCAN: {n_clusters} objetos en {dbscan_time:.1f}s ({len(idx_maq):,} pts)"
            if progress_callback:
                progress_callback(db_msg)
            print(db_msg, flush=True)
                
            # ── Roof Fill: Global approach (ONE tree, ONE query) ──────────
            t_roof = datetime.now()
            count_flipped = 0
            unique_labels = np.array([lbl for lbl in set(labels) if lbl != -1])

            maq_points = xyz[idx_maq]
            valid_cluster_mask = labels >= 0
            maq_valid = maq_points[valid_cluster_mask]
            maq_valid_labels = labels[valid_cluster_mask]
            n_valid_maq = len(maq_valid)

            if n_valid_maq > 0 and len(unique_labels) > 0:
                # Step 1: Per-cluster Z ranges (vectorized)
                t_zr = datetime.now()
                sorted_ulabels = np.sort(unique_labels)
                n_cl = len(sorted_ulabels)
                cluster_z_lo = np.empty(n_cl, dtype=np.float32)
                cluster_z_hi = np.empty(n_cl, dtype=np.float32)
                for i, lbl in enumerate(sorted_ulabels):
                    cz = maq_valid[maq_valid_labels == lbl, 2]
                    min_z = float(np.percentile(cz, 5))
                    cluster_z_lo[i] = min_z + cfg.z_buffer
                    cluster_z_hi[i] = min_z + cfg.max_height

                # Map each valid maq point → cluster index (0..n_cl-1)
                maq_cluster_idx = np.searchsorted(sorted_ulabels, maq_valid_labels)
                zr_time = (datetime.now() - t_zr).total_seconds()
                print(f"   📊 Roof Z-ranges: {n_cl} clusters en {zr_time:.1f}s", flush=True)

                # Step 2: Global XY+Z pre-filter for ground candidates (ONE pass over 76M)
                t_filt = datetime.now()
                global_z_lo = float(cluster_z_lo.min())
                global_z_hi = float(cluster_z_hi.max())
                # XY BBox of all machinery + proximity_radius margin
                maq_xy_min = maq_valid[:, :2].min(axis=0)
                maq_xy_max = maq_valid[:, :2].max(axis=0)
                xy_margin = cfg.proximity_radius + cfg.padding
                cand_mask = (
                    (classification == cfg.ground_class) &
                    (xyz[:, 0] >= maq_xy_min[0] - xy_margin) &
                    (xyz[:, 0] <= maq_xy_max[0] + xy_margin) &
                    (xyz[:, 1] >= maq_xy_min[1] - xy_margin) &
                    (xyz[:, 1] <= maq_xy_max[1] + xy_margin) &
                    (xyz[:, 2] >= global_z_lo) &
                    (xyz[:, 2] <= global_z_hi)
                )
                cand_indices = np.where(cand_mask)[0]
                filt_time = (datetime.now() - t_filt).total_seconds()
                filt_pct = 100.0 * len(cand_indices) / max(1, int((classification == cfg.ground_class).sum()))
                print(f"   📊 Roof filtro XY+Z: {len(cand_indices):,} candidatos ({filt_pct:.1f}% del suelo) en {filt_time:.1f}s", flush=True)

                if len(cand_indices) > 0:
                    # Step 3: ONE cKDTree for all valid machinery XY
                    t_tree = datetime.now()
                    tree_maq_2d = cKDTree(maq_valid[:, :2])
                    tree_time = (datetime.now() - t_tree).total_seconds()
                    print(f"   📊 Roof cKDTree: {n_valid_maq:,} maq en {tree_time:.1f}s", flush=True)

                    # Step 4: ONE query for ALL candidates
                    t_query = datetime.now()
                    cand_xy = xyz[cand_indices, :2]
                    dists_r, nn_idx_r = tree_maq_2d.query(cand_xy, k=1, workers=-1)
                    query_time = (datetime.now() - t_query).total_seconds()
                    print(f"   📊 Roof query: {len(cand_indices):,} pts en {query_time:.1f}s", flush=True)

                    # Step 5: Filter by proximity_radius (2D distance ≤ 1.5m)
                    prox_mask = dists_r <= cfg.proximity_radius
                    del dists_r

                    if np.any(prox_mask):
                        close_cand_idx = cand_indices[prox_mask]
                        close_nn = nn_idx_r[prox_mask]

                        # Step 6: Per-cluster Z range verification
                        matched_cluster = maq_cluster_idx[close_nn]
                        cand_z = xyz[close_cand_idx, 2]
                        z_lo_match = cluster_z_lo[matched_cluster]
                        z_hi_match = cluster_z_hi[matched_cluster]
                        z_valid = (cand_z >= z_lo_match) & (cand_z <= z_hi_match)

                        final_flip = close_cand_idx[z_valid]
                        count_flipped = len(final_flip)
                        if count_flipped > 0:
                            classification[final_flip] = 1

                    del nn_idx_r, tree_maq_2d

            roof_time = (datetime.now() - t_roof).total_seconds()
            roof_msg = f"   ✅ Rellenados {count_flipped:,} puntos de techo en {roof_time:.1f}s"
            if progress_callback:
                progress_callback(roof_msg)
            print(roof_msg, flush=True)
                
            result.points_modified = count_flipped
            
            # Liberar arrays grandes antes de guardar
            del xyz, maq_points
            try:
                del clustering, labels
            except (NameError, UnboundLocalError):
                pass
            gc.collect()

            # Guardar (re-lee desde disco para no mantener las en RAM)
            self._write_las_from_disk(input_file, output_file, classification)

            result.success = True
            result.processing_time = (datetime.now() - start_time).total_seconds()

            ft_msg = f"💾 FIX_TECHO completado en {result.processing_time:.1f}s: {os.path.basename(output_file)}"
            if progress_callback:
                progress_callback(ft_msg)
            print(ft_msg, flush=True)

        except Exception as e:
            import traceback
            print(f"\n❌ ERROR FIX_TECHO:\n{traceback.format_exc()}")
            result.error_message = str(e)
            if progress_callback:
                progress_callback(f"❌ Error en FIX_TECHO: {str(e)}")
                progress_callback("📋 Ver terminal para traceback completo")

        return result
    
    def run_interpol(self, input_file: str, output_file: str,
                     progress_callback: Optional[Callable] = None) -> PostprocessResult:
        """
        Bulldozer Digital (IDW) procesado en chunks espaciales.

        Procesa la nube en celdas de chunk_size_m x chunk_size_m para evitar
        construir un KDTree global que en nubes de millones de puntos consume
        decenas de GB de RAM y mata el proceso.

        Cada chunk:
          1. Extrae maquinaria del tile (core).
          2. Extrae suelo dentro del tile + buffer max_dist.
          3. Construye KDTree LOCAL (pequeño) y hace IDW.
          4. Escribe los Z interpolados en el array global.
        """
        start_time = datetime.now()
        result = PostprocessResult(
            success=False,
            input_file=input_file,
            output_file=output_file,
            step_name="INTERPOL"
        )

        try:
            cfg = self.interpol_config

            if progress_callback:
                progress_callback(f"🚜 INTERPOL: Cargando {os.path.basename(input_file)}...")
            print(f"\n🚜 INTERPOL iniciando: {os.path.basename(input_file)}", flush=True)

            # Monitorear RSS para diagnóstico de OOM
            import resource
            rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            print(f"   📊 RSS al iniciar INTERPOL: {rss_mb:.0f} MB", flush=True)

            las = laspy.read(input_file)

            # ── Arrays 1D float32 separados — NUNCA crear un xyz global (OOM) ──
            x = np.asarray(las.x, dtype=np.float32)
            y = np.asarray(las.y, dtype=np.float32)
            z = np.asarray(las.z, dtype=np.float32)
            classification = np.asarray(las.classification, dtype=np.uint8)
            # Liberar objeto las (~7 GB para nubes grandes) - re-leeremos al guardar
            del las
            import gc
            gc.collect()

            idx_maq   = np.where(classification == 1)[0]
            idx_suelo = np.where(classification == 2)[0]

            mem_mb = (x.nbytes + y.nbytes + z.nbytes + classification.nbytes) / 1024**2
            if progress_callback:
                progress_callback(f"   📉 Maquinaria: {len(idx_maq):,} pts | Suelo: {len(idx_suelo):,} pts | RAM arrays: {mem_mb:.0f} MB")
            print(f"   📉 Maquinaria: {len(idx_maq):,} | Suelo: {len(idx_suelo):,} | Total: {len(x):,} | RAM: {mem_mb:.0f} MB", flush=True)

            if len(idx_maq) == 0:
                if progress_callback:
                    progress_callback("⚠️ No hay maquinaria para aplanar. Copiando archivo.")
                import shutil
                shutil.copy2(input_file, output_file)
                result.success = True
                result.processing_time = (datetime.now() - start_time).total_seconds()
                return result

            # Diagnóstico previo: diferencia de altura maquinaria vs suelo
            if len(idx_suelo) > 0:
                z_maq_med = float(np.median(z[idx_maq]))
                z_sue_med = float(np.median(z[idx_suelo]))
                z_gap = z_maq_med - z_sue_med
                msg_pre = (f"   📐 Altura: mediana_maq={z_maq_med:.2f}m, mediana_suelo={z_sue_med:.2f}m"
                           f", gap={z_gap:.2f}m")
                if progress_callback:
                    progress_callback(msg_pre)
                print(msg_pre, flush=True)
                if abs(z_gap) < 0.5:
                    msg_warn = "   ⚠️ Gap maquinaria-suelo < 0.5m — IDW producirá cambios mínimos en Z"
                    if progress_callback:
                        progress_callback(msg_warn)
                    print(msg_warn, flush=True)

            # ── GPU Tier Detection ─────────────────────────────
            tier_name, free_vram, gpu_device = _detect_gpu_tier()
            tier_params = _TIER_PARAMS[tier_name]
            use_gpu = gpu_device is not None
            if use_gpu:
                try:
                    from torch_cluster import knn as _tc_knn
                    del _tc_knn
                except ImportError:
                    use_gpu = False
            mode_label = f"GPU ({tier_name}, {free_vram:.1f}GB)" if use_gpu else "CPU"
            if progress_callback:
                progress_callback(f"   ⚡ INTERPOL modo: {mode_label}")
            print(f"   ⚡ INTERPOL modo: {mode_label}", flush=True)

            # ── Paso 1: Voxel downsample GLOBAL del suelo ─────────
            t_ds = datetime.now()
            suelo_x = x[idx_suelo]
            suelo_y = y[idx_suelo]
            suelo_z = z[idx_suelo]
            n_suelo_raw = len(idx_suelo)

            # Target: ~5M puntos para HIGH/CPU, 3M MEDIUM, 1.5M LOW
            max_suelo = tier_params['max_local_suelo']
            if n_suelo_raw > max_suelo:
                # Voxel 2D global — resolución adaptativa
                area_total = (suelo_x.max() - suelo_x.min()) * (suelo_y.max() - suelo_y.min())
                voxel_size = float(np.sqrt(max(area_total, 1.0) / max_suelo))
                vx_min, vy_min = float(suelo_x.min()), float(suelo_y.min())
                vx_idx = np.floor((suelo_x - vx_min) / voxel_size).astype(np.int32)
                vy_idx = np.floor((suelo_y - vy_min) / voxel_size).astype(np.int32)
                voxel_key = vx_idx.astype(np.int64) * 1_000_000 + vy_idx.astype(np.int64)
                sort_order = np.argsort(voxel_key, kind='stable')
                sorted_keys = voxel_key[sort_order]
                _, first_in_voxel = np.unique(sorted_keys, return_index=True)
                keep = sort_order[first_in_voxel]
                suelo_x = suelo_x[keep]
                suelo_y = suelo_y[keep]
                suelo_z = suelo_z[keep]
                del vx_idx, vy_idx, voxel_key, sort_order, sorted_keys, first_in_voxel, keep
                import gc as _gc2; _gc2.collect()
                ds_msg = (f"   📊 Voxel downsample global: {n_suelo_raw:,} → {len(suelo_x):,} "
                          f"(voxel={voxel_size:.2f}m) en {(datetime.now()-t_ds).total_seconds():.1f}s")
            else:
                ds_msg = f"   📊 Suelo global: {n_suelo_raw:,} pts (sin downsample necesario)"
            if progress_callback:
                progress_callback(ds_msg)
            print(ds_msg, flush=True)

            n_suelo_ds = len(suelo_x)
            suelo_xy = np.column_stack([suelo_x, suelo_y]).astype(np.float32)

            # Arrays maquinaria
            maq_xy_all = np.column_stack([x[idx_maq], y[idx_maq]]).astype(np.float32)
            n_maq = len(idx_maq)

            final_z = z.copy()
            total_modified = n_maq

            # ── Paso 2: KNN global (GPU o CPU) ────────────────────
            t_knn = datetime.now()

            gpu_ok = False
            if use_gpu:
                if progress_callback:
                    progress_callback(f"   🔍 GPU knn: {n_suelo_ds:,} suelo × {n_maq:,} maq, k={cfg.k_neighbors}...")
                print(f"   🔍 GPU knn: {n_suelo_ds:,} suelo × {n_maq:,} maq, k={cfg.k_neighbors}...", flush=True)

                gpu_result = _interpol_idw_gpu_knn(
                    maq_xy_all, suelo_xy, suelo_z,
                    cfg.k_neighbors, cfg.max_dist,
                    gpu_device,
                )
                if gpu_result is not None:
                    gpu_ok = True
                    result_z, valid_mask = gpu_result
                    knn_time = (datetime.now() - t_knn).total_seconds()
                    knn_msg = f"   ⚡ GPU knn+IDW completado en {knn_time:.1f}s"
                    if progress_callback:
                        progress_callback(knn_msg)
                    print(knn_msg, flush=True)

                    # Aplicar resultados
                    valid_global = idx_maq[valid_mask]
                    if len(valid_global) > 0:
                        final_z[valid_global] = result_z[valid_mask]
                        classification[valid_global] = 2

                    # Fallback para maq sin vecinos dentro de max_dist
                    no_valid_mask = ~valid_mask
                    if np.any(no_valid_mask):
                        no_valid_global = idx_maq[no_valid_mask]
                        tree_fb = cKDTree(suelo_xy)
                        fb_xy = np.column_stack([x[no_valid_global], y[no_valid_global]])
                        _, nn_fb = tree_fb.query(fb_xy, k=1, workers=-1)
                        final_z[no_valid_global] = suelo_z[nn_fb]
                        classification[no_valid_global] = 2
                        del tree_fb
                        fb_msg = f"   📊 {int(no_valid_mask.sum()):,} maq sin vecinos en {cfg.max_dist}m → nearest neighbor"
                        if progress_callback:
                            progress_callback(fb_msg)
                        print(fb_msg, flush=True)
                else:
                    warn_msg = "   ⚠️ GPU knn falló (OOM?), usando CPU cKDTree"
                    if progress_callback:
                        progress_callback(warn_msg)
                    print(warn_msg, flush=True)

            if not gpu_ok:
                # CPU path — UN solo cKDTree global
                if progress_callback:
                    progress_callback(f"   🔍 CPU cKDTree: {n_suelo_ds:,} suelo × {n_maq:,} maq, k={cfg.k_neighbors}...")
                print(f"   🔍 CPU cKDTree: {n_suelo_ds:,} suelo × {n_maq:,} maq...", flush=True)

                t_tree = datetime.now()
                tree_global = cKDTree(suelo_xy)
                tree_time = (datetime.now() - t_tree).total_seconds()
                print(f"   📊 cKDTree construido en {tree_time:.1f}s", flush=True)

                t_query = datetime.now()
                dists, nn_idx = tree_global.query(
                    maq_xy_all,
                    k=min(cfg.k_neighbors, n_suelo_ds),
                    distance_upper_bound=cfg.max_dist,
                    workers=-1
                )
                query_time = (datetime.now() - t_query).total_seconds()
                print(f"   📊 cKDTree query en {query_time:.1f}s", flush=True)

                if dists.ndim == 1:
                    dists = dists[:, np.newaxis]
                    nn_idx = nn_idx[:, np.newaxis]

                has_valid = np.isfinite(dists[:, 0])
                valid_global = idx_maq[has_valid]
                if len(valid_global) > 0:
                    d_v = dists[has_valid]
                    n_v = nn_idx[has_valid]
                    fin = np.isfinite(d_v)
                    d_v = np.where(d_v < 0.001, 0.001, d_v)
                    w = np.where(fin, 1.0 / d_v, 0.0)
                    safe = np.where(n_v < n_suelo_ds, n_v, 0)
                    z_n = suelo_z[safe]
                    num = np.sum(w * z_n, axis=1)
                    den = np.sum(w, axis=1) + 1e-9
                    final_z[valid_global] = (num / den).astype(np.float32)
                    classification[valid_global] = 2

                no_valid_global = idx_maq[~has_valid]
                if len(no_valid_global) > 0:
                    fb_xy = np.column_stack([x[no_valid_global], y[no_valid_global]])
                    _, nn_fb = tree_global.query(fb_xy, k=1, workers=-1)
                    final_z[no_valid_global] = suelo_z[nn_fb]
                    classification[no_valid_global] = 2

                del tree_global, dists, nn_idx
                knn_time = (datetime.now() - t_knn).total_seconds()
                knn_msg = f"   ⚡ CPU cKDTree+IDW completado en {knn_time:.1f}s"
                if progress_callback:
                    progress_callback(knn_msg)
                print(knn_msg, flush=True)

            backend_used = "GPU" if gpu_ok else "CPU"
            result.points_modified = total_modified
            del suelo_x, suelo_y, suelo_xy

            # Diagnóstico: verificar que Z realmente cambió
            z_orig = z[idx_maq]
            z_new = final_z[idx_maq]
            z_diff = np.abs(z_orig - z_new)
            n_changed = int(np.sum(z_diff > 0.01))
            msg_diag = (f"   📊 Z diagnostico: {n_changed:,}/{len(idx_maq):,} puntos con dZ>1cm"
                        f" | dZ medio={np.mean(z_diff):.3f}m"
                        f" | dZ max={np.max(z_diff):.3f}m")
            if progress_callback:
                progress_callback(msg_diag)
            print(msg_diag, flush=True)

            if progress_callback:
                progress_callback(f"   ✅ Aplanados {total_modified:,} puntos")
            print(f"   ✅ INTERPOL: {total_modified:,} puntos aplanados", flush=True)

            # Convertir a float64 para compatibilidad con laspy
            final_z_f64 = final_z.astype(np.float64)

            # Liberar arrays intermedios antes de guardar
            del x, y, z, suelo_z, maq_xy_all, final_z
            del idx_maq, idx_suelo
            gc.collect()

            # Guardar (re-lee desde disco para no mantener las en RAM)
            self._write_las_from_disk(input_file, output_file, classification, z_values=final_z_f64)

            result.success = True
            result.processing_time = (datetime.now() - start_time).total_seconds()

            if progress_callback:
                progress_callback(f"💾 DTM guardado en {result.processing_time:.1f}s: {os.path.basename(output_file)}")
            print(f"💾 DTM guardado: {result.processing_time:.1f}s")

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            result.error_message = str(e)
            print(f"\n❌ ERROR INTERPOL:\n{tb}")
            if progress_callback:
                progress_callback(f"❌ Error en INTERPOL: {str(e)}")
                progress_callback("📋 Ver terminal para traceback completo")

        return result
    
    def run_full_pipeline(self, input_file: str, output_dir: str,
                          base_name: str,
                          run_pre_clean: bool = False,
                          run_fix_techo: bool = True,
                          run_interpol: bool = True,
                          progress_callback: Optional[Callable] = None) -> dict:
        """
        Ejecuta el pipeline completo de postprocesamiento.
        
        Args:
            input_file: Archivo de entrada (salida de inferencia)
            output_dir: Directorio de salida
            base_name: Nombre base para archivos
            run_fix_techo: Si ejecutar FIX_TECHO
            run_interpol: Si ejecutar INTERPOL
            progress_callback: Función de progreso
            
        Returns:
            Dict con resultados de cada paso
        """
        results = {}
        current_file = input_file

        if run_pre_clean and self.pre_clean_config.enabled:
            preclean_file = os.path.join(output_dir, f"{base_name}_preclean.laz")
            result = self.run_pre_clean(current_file, preclean_file, progress_callback)
            results['pre_clean'] = result

            if result.success:
                current_file = preclean_file
            else:
                return results
        
        if run_fix_techo:
            techos_file = os.path.join(output_dir, f"{base_name}_techos.laz")
            result = self.run_fix_techo(current_file, techos_file, progress_callback)
            results['fix_techo'] = result
            
            if result.success:
                current_file = techos_file
            else:
                return results
                
        if run_interpol:
            dtm_file = os.path.join(output_dir, f"{base_name}_DTM.laz")
            result = self.run_interpol(current_file, dtm_file, progress_callback)
            results['interpol'] = result
            
        return results
