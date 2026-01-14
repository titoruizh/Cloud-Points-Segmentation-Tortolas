#!/usr/bin/env python3
"""
Script de Preprocesamiento de Bloques V5 (Ablation Study: No Verticality)
Genera bloques .npy con RGB + Normales ( SIN Verticalidad) + Balance de Clases.

Formato de salida V5 (10 canales): 
[x, y, z, red, green, blue, nx, ny, nz, label]

Cambios vs V4:
- Se MANTIENE el cálculo de verticalidad interno para poder hacer Hard Negative Mining (detectar paredes/pretiles).
- Se ELIMINA el canal de verticalidad del archivo final .npy.
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
DEFAULT_RAW_DIR = "data/raw RGB"
DEFAULT_BLOCK_SIZE = 10.0
DEFAULT_NORMAL_RADIUS = 2.0
DEFAULT_NORMAL_MAX_NN = 50
DEFAULT_MIN_POINTS = 1000
DEFAULT_MAX_POINTS = 20000
DEFAULT_HARD_NEGATIVE_RATIO = 0.8
DEFAULT_EASY_NEGATIVE_RATIO = 0.5
DEFAULT_MIN_MACHINERY_RATIO = 0.03
DEFAULT_HARD_VERTICALITY_THRESHOLD = 0.20

def compute_robust_features(xyz, normal_radius, normal_max_nn):
    """
    Calcula normal y verticalidad internas.
    Returns: [nx, ny, nz, verticalidad]
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius, 
            max_nn=normal_max_nn
        )
    )
    pcd.orient_normals_to_align_with_direction(
        orientation_reference=np.array([0., 0., 1.])
    )
    normals = np.asarray(pcd.normals)
    normals[normals[:, 2] < 0] *= -1

    verticality = 1.0 - np.abs(normals[:, 2])
    verticality = verticality.reshape(-1, 1)
    
    return np.hstack((normals, verticality))


def crop_block(xyz, rgb, features, labels, cx, cy, block_size):
    """
    Recorta bloque centrado en (cx, cy).
    Retorna los 10 canales intermedios (con vert) para poder filtrar,
    pero luego limpiaremos antes de guardar.
    """
    half = block_size / 2.0
    
    mask = (
        (xyz[:, 0] >= cx - half) & (xyz[:, 0] < cx + half) &
        (xyz[:, 1] >= cy - half) & (xyz[:, 1] < cy + half)
    )
    
    if np.sum(mask) == 0:
        return None, None
        
    xyz_crop = xyz[mask].copy()
    rgb_crop = rgb[mask].copy()
    feat_crop = features[mask].copy()
    lbl_crop = labels[mask].copy()
    
    # NORMALIZACIÓN RELATIVA DE COORDENADAS
    xyz_crop[:, 0] -= cx
    xyz_crop[:, 1] -= cy
    xyz_crop[:, 2] -= np.min(xyz_crop[:, 2])
    
    # Concatenar todo: XYZ(3) + RGB(3) + Feat(4) = 10 canales
    # Feat contiene [nx, ny, nz, v]
    final_data = np.hstack((xyz_crop, rgb_crop, feat_crop))
    
    return final_data, lbl_crop


def get_machinery_crops(xyz, rgb, features, labels, block_size, min_points, min_machinery_ratio=0.03):
    crops = []
    mach_mask = labels == 1
    mach_xyz = xyz[mach_mask]
    
    if len(mach_xyz) < 50:
        return []

    adaptive_ratio = min_machinery_ratio * (10.0 / block_size) ** 2
    adaptive_ratio = max(adaptive_ratio, 0.003)
    
    clustering = DBSCAN(eps=3.0, min_samples=20).fit(mach_xyz)
    unique_clusters = set(clustering.labels_)
    
    for cluster_id in unique_clusters:
        if cluster_id == -1: continue
        
        cluster_points = mach_xyz[clustering.labels_ == cluster_id]
        center = np.mean(cluster_points, axis=0)
        
        crop, crop_labels = crop_block(
            xyz, rgb, features, labels, center[0], center[1], block_size
        )
        
        if crop is not None and len(crop) >= min_points:
            machinery_ratio = np.sum(crop_labels == 1) / len(crop_labels)
            if machinery_ratio >= adaptive_ratio:
                crops.append((crop, crop_labels, "MACHINERY", machinery_ratio))
            
    return crops


def get_hard_negative_crops(xyz, rgb, features, labels, n_needed, block_size, min_points, vert_threshold=0.20):
    crops = []
    attempts = 0
    max_attempts = n_needed * 100
    
    min_x, max_x = np.min(xyz[:, 0]), np.max(xyz[:, 0])
    min_y, max_y = np.min(xyz[:, 1]), np.max(xyz[:, 1])
    
    while len(crops) < n_needed and attempts < max_attempts:
        attempts += 1
        cx = np.random.uniform(min_x, max_x)
        cy = np.random.uniform(min_y, max_y)
        
        crop, crop_labels = crop_block(xyz, rgb, features, labels, cx, cy, block_size)
        
        if crop is None or len(crop) < min_points:
            continue
            
        has_machinery = np.sum(crop_labels == 1) > 0
        avg_verticality = np.mean(crop[:, 9]) # Canal 9 sigue siendo verticalidad AQUI
        
        if not has_machinery and avg_verticality > vert_threshold:
            crops.append((crop, crop_labels, "HARD_NEGATIVE"))
            
    return crops


def get_easy_negative_crops(xyz, rgb, features, labels, n_needed, block_size, min_points):
    crops = []
    attempts = 0
    max_attempts = n_needed * 50
    
    min_x, max_x = np.min(xyz[:, 0]), np.max(xyz[:, 0])
    min_y, max_y = np.min(xyz[:, 1]), np.max(xyz[:, 1])
    
    while len(crops) < n_needed and attempts < max_attempts:
        attempts += 1
        cx = np.random.uniform(min_x, max_x)
        cy = np.random.uniform(min_y, max_y)
        
        crop, crop_labels = crop_block(xyz, rgb, features, labels, cx, cy, block_size)
        
        if crop is None or len(crop) < min_points:
            continue
            
        has_machinery = np.sum(crop_labels == 1) > 0
        avg_verticality = np.mean(crop[:, 9])
        
        if not has_machinery and avg_verticality < 0.10:
            crops.append((crop, crop_labels, "EASY_NEGATIVE"))
            
    return crops


def process_file(filepath, output_dir, config):
    filename = os.path.basename(filepath).replace('.laz', '').replace('.las', '')
    
    try:
        las = laspy.read(filepath)
        xyz = np.vstack((las.x, las.y, las.z)).transpose()
        labels = np.array(las.classification)
        
        if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
            red = np.array(las.red)
            green = np.array(las.green)
            blue = np.array(las.blue)
            
            max_val = max(np.max(red), np.max(green), np.max(blue))
            if max_val > 255:
                scale_factor = 65535.0
            else:
                scale_factor = 255.0
                
            rgb = np.vstack((red, green, blue)).transpose() / scale_factor
        else:
            print(f"⚠️ {filename}: No se encontró canal RGB. Rellenando con gris (0.5).")
            rgb = np.full_like(xyz, 0.5)

    except Exception as e:
        print(f"❌ Error cargando {filepath}: {e}")
        return None
    
    if not hasattr(las, 'classification'):
        return None
    
    labels_remapped = np.zeros_like(labels)
    labels_remapped[labels == 1] = 1
    labels_remapped[labels == 2] = 0
    
    n_machinery = np.sum(labels_remapped == 1)
    if n_machinery < 100:
        return None
    
    print(f"\n📂 Procesando V5: {filename}")
    
    geo_features = compute_robust_features(
        xyz, config['normal_radius'], config['normal_max_nn']
    )
    
    base_ratio = config.get('min_machinery_ratio', 0.03)
    
    print(f"   🚜 Extrayendo bloques MACHINERY...")
    mach_crops = get_machinery_crops(
        xyz, rgb, geo_features, labels_remapped, 
        config['block_size'], config['min_points'],
        min_machinery_ratio=base_ratio
    )
    
    n_hard = int(len(mach_crops) * config['hard_negative_ratio'])
    n_hard = max(n_hard, 3)
    print(f"   ⛰️  Extrayendo {n_hard} bloques HARD_NEGATIVE...")
    hard_crops = get_hard_negative_crops(
        xyz, rgb, geo_features, labels_remapped, n_hard,
        config['block_size'], config['min_points'],
        vert_threshold=config.get('hard_vert_threshold', 0.20)
    )
    
    n_easy = int(len(mach_crops) * config['easy_negative_ratio'])
    if n_easy > 0:
        print(f"   🟤 Extrayendo {n_easy} bloques EASY_NEGATIVE...")
        easy_crops = get_easy_negative_crops(
            xyz, rgb, geo_features, labels_remapped, n_easy,
            config['block_size'], config['min_points']
        )
    else:
        easy_crops = []
    
    all_crops = mach_crops + hard_crops + easy_crops
    
    os.makedirs(output_dir, exist_ok=True)
    saved_count = {'MACHINERY': 0, 'HARD_NEGATIVE': 0, 'EASY_NEGATIVE': 0}
    
    for i, crop_data in enumerate(all_crops):
        if len(crop_data) == 4:
            data, lbl, tipo, _ = crop_data
        else:
            data, lbl, tipo = crop_data
        
        # --- ABLATION V5: REMOVER VERTICALIDAD ---
        # Data tiene 10 columnas:
        # [X Y Z R G B Nx Ny Nz V] -> Indices 0..9
        # Queremos guardar solo: [X Y Z R G B Nx Ny Nz] -> Indices 0..8
        
        data_v5 = data[:, :9]  # Excluye columna 9 (Vert)
        
        save_array = np.hstack((data_v5, lbl.reshape(-1, 1)))
        
        out_name = f"{tipo}_{filename}_{i:04d}.npy"
        np.save(os.path.join(output_dir, out_name), save_array.astype(np.float32))
        saved_count[tipo] += 1
    
    print(f"   ✅ Guardados (No-Vert): {saved_count['MACHINERY']} MACH | {saved_count['HARD_NEGATIVE']} HARD | {saved_count['EASY_NEGATIVE']} EASY")
    return saved_count


def main():
    parser = argparse.ArgumentParser(description='Preprocesamiento V5 (No Vert)')
    parser.add_argument('--raw-dir', type=str, default=DEFAULT_RAW_DIR)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--block-size', type=float, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument('--normal-radius', type=float, default=DEFAULT_NORMAL_RADIUS)
    parser.add_argument('--normal-max-nn', type=int, default=DEFAULT_NORMAL_MAX_NN)
    parser.add_argument('--min-points', type=int, default=DEFAULT_MIN_POINTS)
    parser.add_argument('--max-points', type=int, default=DEFAULT_MAX_POINTS)
    parser.add_argument('--hard-negative-ratio', type=float, default=DEFAULT_HARD_NEGATIVE_RATIO)
    parser.add_argument('--easy-negative-ratio', type=float, default=DEFAULT_EASY_NEGATIVE_RATIO)
    parser.add_argument('--max-files', type=int, default=None)
    
    args = parser.parse_args()
    
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
    
    files = glob.glob(os.path.join(args.raw_dir, "*.laz"))
    files += glob.glob(os.path.join(args.raw_dir, "*.las"))
    
    if args.max_files:
        files = files[:args.max_files]
    
    print(f"🚀 Iniciando Preprocesamiento V5 (RGB No-Vert) en {args.raw_dir}...")
    
    total_stats = defaultdict(int)
    for filepath in files:
        stats = process_file(filepath, output_dir, config)
        if stats:
            for k, v in stats.items():
                total_stats[k] += v
                
    print("\n✅ FINALIZADO V5")
    print(f"💾 Output: {output_dir}")

if __name__ == "__main__":
    main()
