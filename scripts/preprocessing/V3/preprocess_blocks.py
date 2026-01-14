#!/usr/bin/env python3
"""
Script de Preprocesamiento de Bloques para Point Cloud Research
Genera bloques .npy con estrategia de muestreo inteligente para balance de clases.

Formato de salida: [x, y, z, nx, ny, nz, verticalidad, label]
"""

import os
import glob
import argparse
import numpy as np
import laspy
import open3d as o3d
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from collections import defaultdict

# --- ⚙️ CONFIGURACIÓN POR DEFECTO ---
DEFAULT_RAW_DIR = "data/raw"
DEFAULT_BLOCK_SIZE = 10.0
DEFAULT_NORMAL_RADIUS = 2.0
DEFAULT_NORMAL_MAX_NN = 50
DEFAULT_MIN_POINTS = 1000
DEFAULT_MAX_POINTS = 20000
DEFAULT_HARD_NEGATIVE_RATIO = 0.8  # Aumentado: más hard negatives
DEFAULT_EASY_NEGATIVE_RATIO = 0.5  # Activado: vital para que el modelo conozca el suelo
DEFAULT_MIN_MACHINERY_RATIO = 0.03  # Nuevo: mínimo 3% de maquinaria en bloques MACHINERY
DEFAULT_HARD_VERTICALITY_THRESHOLD = 0.20  # Nuevo: umbral más estricto para hard negatives

def compute_robust_features(xyz, normal_radius, normal_max_nn):
    """
    Calcula características geométricas robustas: normales + verticalidad.
    
    Args:
        xyz: Array (N, 3) de coordenadas
        normal_radius: Radio de búsqueda para normales
        normal_max_nn: Máximo de vecinos para normales
    
    Returns:
        Array (N, 4): [nx, ny, nz, verticalidad]
    """
    # 1. Crear Nube Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    # 2. Calcular Normales Robustas (Suavizadas)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius, 
            max_nn=normal_max_nn
        )
    )
    
    # 3. Orientar hacia arriba (+Z) - Asumimos minería a cielo abierto
    pcd.orient_normals_to_align_with_direction(
        orientation_reference=np.array([0., 0., 1.])
    )
    normals = np.asarray(pcd.normals)
    
    # Asegurar que apunten arriba (Fix para superficies planas invertidas)
    normals[normals[:, 2] < 0] *= -1

    # 4. Feature de Verticalidad (0.0 = Plano, 1.0 = Pared Vertical)
    # Ayuda a diferenciar rocas/ruedas del suelo
    verticality = 1.0 - np.abs(normals[:, 2])
    verticality = verticality.reshape(-1, 1)
    
    return np.hstack((normals, verticality))


def crop_block(xyz, features, labels, cx, cy, block_size):
    """
    Recorta y normaliza un bloque centrado en (cx, cy).
    
    Args:
        xyz: Coordenadas completas (N, 3)
        features: Features completas (N, 4) [nx, ny, nz, vert]
        labels: Labels completas (N,)
        cx, cy: Centro del bloque
        block_size: Tamaño del bloque en metros
    
    Returns:
        data: Array (M, 7) [x_rel, y_rel, z_rel, nx, ny, nz, vert]
        labels: Array (M,)
    """
    half = block_size / 2.0
    
    mask = (
        (xyz[:, 0] >= cx - half) & (xyz[:, 0] < cx + half) &
        (xyz[:, 1] >= cy - half) & (xyz[:, 1] < cy + half)
    )
    
    if np.sum(mask) == 0:
        return None, None
        
    xyz_crop = xyz[mask].copy()
    feat_crop = features[mask].copy()
    lbl_crop = labels[mask].copy()
    
    # NORMALIZACIÓN RELATIVA (CRUCIAL PARA POINTNET)
    xyz_crop[:, 0] -= cx  # Centrado en X
    xyz_crop[:, 1] -= cy  # Centrado en Y
    xyz_crop[:, 2] -= np.min(xyz_crop[:, 2])  # Z relativo al suelo del bloque
    
    # Input final: [x, y, z, nx, ny, nz, vert]
    final_data = np.hstack((xyz_crop, feat_crop))
    
    return final_data, lbl_crop


def get_machinery_crops(xyz, features, labels, block_size, min_points, min_machinery_ratio=0.03):
    """
    Estrategia: CENTRADO EN OBJETO.
    Encuentra cada máquina con DBSCAN y extrae un bloque centrado en ella.
    Filtra bloques con muy poca maquinaria.
    
    Args:
        min_machinery_ratio: Mínimo ratio de puntos de maquinaria (default 3% para bloques de 10m)
                            Se ajusta automáticamente para bloques más grandes
    
    Returns:
        List of (data, labels, "MACHINERY", machinery_ratio)
    """
    crops = []
    
    # Filtrar solo puntos de maquinaria
    mach_mask = labels == 1
    mach_xyz = xyz[mach_mask]
    
    if len(mach_xyz) < 50:
        return []

    # FILTRO ADAPTATIVO: Escalar ratio con tamaño de bloque
    # Bloques más grandes necesitan ratio más bajo
    # 10m → 3%, 20m → 1.5%, 50m → 0.5%
    adaptive_ratio = min_machinery_ratio * (10.0 / block_size) ** 2
    adaptive_ratio = max(adaptive_ratio, 0.003)  # Mínimo 0.3%
    
    # Usar DBSCAN para encontrar el CENTRO de cada vehículo individual
    # eps=3.0m conecta puntos de un mismo camión
    clustering = DBSCAN(eps=3.0, min_samples=20).fit(mach_xyz)
    unique_clusters = set(clustering.labels_)
    
    for cluster_id in unique_clusters:
        if cluster_id == -1:  # Ruido
            continue
        
        # Obtener centroide del vehículo
        cluster_points = mach_xyz[clustering.labels_ == cluster_id]
        center = np.mean(cluster_points, axis=0)  # [x, y, z] center
        
        # Cortar bloque centrado en este centro
        crop, crop_labels = crop_block(
            xyz, features, labels, center[0], center[1], block_size
        )
        
        if crop is not None and len(crop) >= min_points:
            # Calcular ratio de maquinaria
            machinery_ratio = np.sum(crop_labels == 1) / len(crop_labels)
            
            # Filtrar bloques con muy poca maquinaria (usando ratio adaptativo)
            if machinery_ratio >= adaptive_ratio:
                crops.append((crop, crop_labels, "MACHINERY", machinery_ratio))
            
    return crops


def get_hard_negative_crops(xyz, features, labels, n_needed, block_size, min_points, vert_threshold=0.20):
    """
    Busca zonas SIN maquinaria pero con geometría compleja (taludes, rocas).
    Usa verticalidad promedio como proxy de complejidad.
    
    Args:
        vert_threshold: Umbral de verticalidad para considerar "complejo"
    
    Returns:
        List of (data, labels, "HARD_NEGATIVE")
    """
    crops = []
    attempts = 0
    max_attempts = n_needed * 100
    
    # Límites
    min_x, max_x = np.min(xyz[:, 0]), np.max(xyz[:, 0])
    min_y, max_y = np.min(xyz[:, 1]), np.max(xyz[:, 1])
    
    while len(crops) < n_needed and attempts < max_attempts:
        attempts += 1
        # Random center
        cx = np.random.uniform(min_x, max_x)
        cy = np.random.uniform(min_y, max_y)
        
        crop, crop_labels = crop_block(xyz, features, labels, cx, cy, block_size)
        
        if crop is None or len(crop) < min_points:
            continue
            
        # Validar:
        # 1. NO tiene maquinaria (Label 1)
        # 2. Es "complejo" (Verticalidad promedio > vert_threshold)
        has_machinery = np.sum(crop_labels == 1) > 0
        avg_verticality = np.mean(crop[:, 6])  # Columna de verticalidad
        
        if not has_machinery and avg_verticality > vert_threshold:
            crops.append((crop, crop_labels, "HARD_NEGATIVE"))
            
    return crops


def get_easy_negative_crops(xyz, features, labels, n_needed, block_size, min_points):
    """
    Busca zonas de suelo plano simple para balance general.
    
    Returns:
        List of (data, labels, "EASY_NEGATIVE")
    """
    crops = []
    attempts = 0
    max_attempts = n_needed * 50
    
    # Límites
    min_x, max_x = np.min(xyz[:, 0]), np.max(xyz[:, 0])
    min_y, max_y = np.min(xyz[:, 1]), np.max(xyz[:, 1])
    
    while len(crops) < n_needed and attempts < max_attempts:
        attempts += 1
        # Random center
        cx = np.random.uniform(min_x, max_x)
        cy = np.random.uniform(min_y, max_y)
        
        crop, crop_labels = crop_block(xyz, features, labels, cx, cy, block_size)
        
        if crop is None or len(crop) < min_points:
            continue
            
        # Validar:
        # 1. NO tiene maquinaria
        # 2. Es "simple" (Verticalidad promedio < 0.10)
        has_machinery = np.sum(crop_labels == 1) > 0
        avg_verticality = np.mean(crop[:, 6])
        
        if not has_machinery and avg_verticality < 0.10:
            crops.append((crop, crop_labels, "EASY_NEGATIVE"))
            
    return crops


def process_file(filepath, output_dir, config):
    """
    Procesa un archivo .laz y genera bloques .npy.
    
    Returns:
        dict: Estadísticas del procesamiento
    """
    filename = os.path.basename(filepath).replace('.laz', '').replace('.las', '')
    
    # 1. Cargar
    try:
        las = laspy.read(filepath)
        xyz = np.vstack((las.x, las.y, las.z)).transpose()
        labels = np.array(las.classification)
    except Exception as e:
        print(f"❌ Error cargando {filepath}: {e}")
        return None
    
    # Verificar que tenga labels
    if not hasattr(las, 'classification'):
        print(f"⚠️ {filename}: Sin clasificación, saltando...")
        return None
    
    # Remapear labels: 1=Maquinaria, 2=Suelo -> 1=Maquinaria, 0=Suelo
    labels_remapped = np.zeros_like(labels)
    labels_remapped[labels == 1] = 1
    labels_remapped[labels == 2] = 0
    
    # Contar puntos por clase
    n_machinery = np.sum(labels_remapped == 1)
    n_ground = np.sum(labels_remapped == 0)
    
    if n_machinery < 100:
        print(f"⚠️ {filename}: Solo {n_machinery} puntos de maquinaria, saltando...")
        return None
    
    print(f"\n📂 Procesando: {filename}")
    print(f"   Total: {len(xyz):,} puntos | Maquinaria: {n_machinery:,} | Suelo: {n_ground:,}")
    
    # 2. Calcular Features Globales (una sola vez para eficiencia)
    print(f"   🔍 Calculando normales (radio={config['normal_radius']}m)...")
    geo_features = compute_robust_features(
        xyz, config['normal_radius'], config['normal_max_nn']
    )
    
    # 3. Extraer Maquinaria (Prioridad 1)
    base_ratio = config.get('min_machinery_ratio', 0.03)
    adaptive_ratio = base_ratio * (10.0 / config['block_size']) ** 2
    adaptive_ratio = max(adaptive_ratio, 0.003)
    print(f"   🚜 Extrayendo bloques MACHINERY (min ratio: {adaptive_ratio*100:.1f}% adaptativo)...")
    mach_crops = get_machinery_crops(
        xyz, geo_features, labels_remapped, 
        config['block_size'], config['min_points'],
        min_machinery_ratio=base_ratio
    )
    
    # 4. Extraer Hard Negatives (Prioridad 2)
    n_hard = int(len(mach_crops) * config['hard_negative_ratio'])
    n_hard = max(n_hard, 3)  # Mínimo 3
    print(f"   ⛰️  Extrayendo {n_hard} bloques HARD_NEGATIVE (vert > {config.get('hard_vert_threshold', 0.20)})...")
    hard_crops = get_hard_negative_crops(
        xyz, geo_features, labels_remapped, n_hard,
        config['block_size'], config['min_points'],
        vert_threshold=config.get('hard_vert_threshold', 0.20)
    )
    
    # 5. Extraer Easy Negatives (Prioridad 3) - Opcional
    n_easy = int(len(mach_crops) * config['easy_negative_ratio'])
    if n_easy > 0:
        print(f"   🟤 Extrayendo {n_easy} bloques EASY_NEGATIVE...")
        easy_crops = get_easy_negative_crops(
            xyz, geo_features, labels_remapped, n_easy,
            config['block_size'], config['min_points']
        )
    else:
        easy_crops = []
    
    all_crops = mach_crops + hard_crops + easy_crops
    
    # 6. Guardar con estadísticas
    os.makedirs(output_dir, exist_ok=True)
    saved_count = {'MACHINERY': 0, 'HARD_NEGATIVE': 0, 'EASY_NEGATIVE': 0}
    machinery_ratios = []
    
    for i, crop_data in enumerate(all_crops):
        if len(crop_data) == 4:  # MACHINERY con ratio
            data, lbl, tipo, mach_ratio = crop_data
            machinery_ratios.append(mach_ratio)
        else:  # HARD/EASY sin ratio
            data, lbl, tipo = crop_data
        
        # Guardamos todo junto: [x, y, z, nx, ny, nz, v, label]
        save_array = np.hstack((data, lbl.reshape(-1, 1)))
        
        out_name = f"{tipo}_{filename}_{i:04d}.npy"
        np.save(os.path.join(output_dir, out_name), save_array.astype(np.float32))
        saved_count[tipo] += 1
    
    # Estadísticas mejoradas
    if machinery_ratios:
        avg_mach_ratio = np.mean(machinery_ratios) * 100
        print(f"   ✅ Guardados: {saved_count['MACHINERY']} MACH (avg {avg_mach_ratio:.1f}% maq) | "
              f"{saved_count['HARD_NEGATIVE']} HARD | {saved_count['EASY_NEGATIVE']} EASY")
    else:
        print(f"   ✅ Guardados: {saved_count['MACHINERY']} MACH | "
              f"{saved_count['HARD_NEGATIVE']} HARD | {saved_count['EASY_NEGATIVE']} EASY")
    
    return saved_count


def main():
    parser = argparse.ArgumentParser(
        description='Preprocesamiento de bloques para Point Cloud Research'
    )
    parser.add_argument('--raw-dir', type=str, default=DEFAULT_RAW_DIR,
                        help='Directorio con archivos .laz')
    parser.add_argument('--output', type=str, required=True,
                        help='Nombre de carpeta de salida (ej: blocks_10m)')
    parser.add_argument('--block-size', type=float, default=DEFAULT_BLOCK_SIZE,
                        help='Tamaño de bloque en metros')
    parser.add_argument('--normal-radius', type=float, default=DEFAULT_NORMAL_RADIUS,
                        help='Radio para cálculo de normales')
    parser.add_argument('--normal-max-nn', type=int, default=DEFAULT_NORMAL_MAX_NN,
                        help='Máximo de vecinos para normales')
    parser.add_argument('--min-points', type=int, default=DEFAULT_MIN_POINTS,
                        help='Mínimo de puntos por bloque')
    parser.add_argument('--max-points', type=int, default=DEFAULT_MAX_POINTS,
                        help='Máximo de puntos por bloque (para downsampling futuro)')
    parser.add_argument('--hard-negative-ratio', type=float, default=DEFAULT_HARD_NEGATIVE_RATIO,
                        help='Ratio de bloques HARD_NEGATIVE respecto a MACHINERY')
    parser.add_argument('--easy-negative-ratio', type=float, default=DEFAULT_EASY_NEGATIVE_RATIO,
                        help='Ratio de bloques EASY_NEGATIVE respecto a MACHINERY')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Máximo de archivos a procesar (para testing)')
    
    args = parser.parse_args()
    
    # Configuración
    config = {
        'block_size': args.block_size,
        'normal_radius': args.normal_radius,
        'normal_max_nn': args.normal_max_nn,
        'min_points': args.min_points,
        'max_points': args.max_points,
        'hard_negative_ratio': args.hard_negative_ratio,
        'easy_negative_ratio': args.easy_negative_ratio,
        'min_machinery_ratio': DEFAULT_MIN_MACHINERY_RATIO,
        'hard_vert_threshold': DEFAULT_HARD_VERTICALITY_THRESHOLD,
    }
    
    output_dir = os.path.join("data/processed", args.output)
    
    print("=" * 80)
    print("🚀 PREPROCESSING - Point Cloud Research")
    print("=" * 80)
    print(f"📁 Input:  {args.raw_dir}")
    print(f"📁 Output: {output_dir}")
    print(f"📐 Block Size: {args.block_size}m")
    print(f"🌊 Normal Radius: {args.normal_radius}m (max_nn={args.normal_max_nn})")
    print(f"📊 Ratios: HARD={args.hard_negative_ratio}, EASY={args.easy_negative_ratio}")
    print("=" * 80)
    
    # Buscar archivos
    files = glob.glob(os.path.join(args.raw_dir, "*.laz"))
    files += glob.glob(os.path.join(args.raw_dir, "*.las"))
    
    if args.max_files:
        files = files[:args.max_files]
        print(f"⚠️ Modo TEST: Procesando solo {args.max_files} archivos")
    
    print(f"\n📂 Archivos encontrados: {len(files)}")
    
    # Procesar
    total_stats = defaultdict(int)
    
    for filepath in files:
        stats = process_file(filepath, output_dir, config)
        if stats:
            for k, v in stats.items():
                total_stats[k] += v
    
    # Resumen final
    print("\n" + "=" * 80)
    print("✅ PROCESAMIENTO COMPLETADO")
    print("=" * 80)
    print(f"📊 Total de bloques generados:")
    print(f"   🚜 MACHINERY:      {total_stats['MACHINERY']:4d}")
    print(f"   ⛰️  HARD_NEGATIVE:  {total_stats['HARD_NEGATIVE']:4d}")
    print(f"   🟤 EASY_NEGATIVE:  {total_stats['EASY_NEGATIVE']:4d}")
    print(f"   📦 TOTAL:          {sum(total_stats.values()):4d}")
    print(f"\n💾 Guardados en: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
