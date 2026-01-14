import os
import torch
import numpy as np
import laspy
import open3d as o3d 
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader, Dataset
import yaml
import sys
import re

# Fix import path
sys.path.append(os.getcwd())

# Importar modelo RandLANet
from src.models.randlanet import RandLANet

def compute_features_global_v4(xyz, rgb_raw, radius=2.5):
    """
    Computa normales y verticalidad. Integra RGB normalizado.
    """
    print("🧹 Limpiando nube de puntos (Duplicados, NaNs)...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    # Si tenemos color, lo asociamos para mantener la correspondencia tras limpieza
    if rgb_raw is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb_raw)
    
    # Limpieza crítica
    # handle open3d return type variability
    res = pcd.remove_duplicated_points()
    if isinstance(res, tuple):
        pcd, ind = res
    else:
        pcd = res
    
    pcd = pcd.remove_non_finite_points()
    
    # Actualizar xyz limpio
    xyz_clean = np.asarray(pcd.points)
    rgb_clean = np.asarray(pcd.colors) if rgb_raw is not None else None
    
    print(f"   Puntos tras limpieza: {len(xyz_clean):,}")
    
    print(f"🔍 Calculando normales globales (r={radius}m)...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    
    # Orientar a +Z (Minería cielo abierto)
    pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0., 0., 1.]))
    
    normals = np.asarray(pcd.normals)
    # Fix matemático extra por si alguna normal quedó invertida
    normals[normals[:, 2] < 0] *= -1
    
    # Verticalidad
    verticality = 1.0 - np.abs(normals[:, 2]) 
    
    return xyz_clean, rgb_clean, normals, verticality

class GridDatasetV4(Dataset):
    def __init__(self, full_data, grid_dict, num_points, min_coord, block_size):
        """
        full_data: [N, 10] -> [x, y, z, r, g, b, nx, ny, nz, v]
        """
        self.full_data = full_data
        self.grid_keys = list(grid_dict.keys())
        self.grid_dict = grid_dict
        self.num_points = num_points
        self.min_coord = min_coord 
        self.block_size = block_size
        
    def __len__(self):
        return len(self.grid_keys)
    
    def __getitem__(self, idx):
        key = self.grid_keys[idx]
        indices = self.grid_dict[key]
        
        # Resampling strategy (RandLANet requiere fixed size 65536)
        if len(indices) >= self.num_points:
            selected_indices = np.random.choice(indices, self.num_points, replace=False)
        else:
            selected_indices = np.random.choice(indices, self.num_points, replace=True)
            
        block_data = self.full_data[selected_indices]
        
        # --- NORMALIZACIÓN GEOMÉTRICA ---
        ix = key // 100000
        iy = key % 100000
        
        tile_origin_x = self.min_coord[0] + ix * self.block_size
        tile_origin_y = self.min_coord[1] + iy * self.block_size
        
        tile_center_x = tile_origin_x + (self.block_size / 2.0)
        tile_center_y = tile_origin_y + (self.block_size / 2.0)
        
        xyz = block_data[:, :3]
        xyz_norm = xyz.copy()
        
        # XY Relativo al centro
        xyz_norm[:, 0] = xyz[:, 0] - tile_center_x
        xyz_norm[:, 1] = xyz[:, 1] - tile_center_y
        
        # Z Relativo al suelo local
        z_min = np.min(xyz[:, 2])
        xyz_norm[:, 2] = xyz[:, 2] - z_min
        
        # Reconstruir tensor de features
        # [x, y, z, r, g, b, nx, ny, nz, v]
        features = block_data.copy()
        features[:, :3] = xyz_norm
        
        return torch.FloatTensor(features), selected_indices

def run_inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Iniciando Inferencia RandLANet V4 (RGB) Grid-Based: {args.input_file}")
    print(f"   ⚙️ Config: Block={args.block_size}m | Points={args.num_points} | Threshold={args.conf_threshold}")
    
    # 1. Cargar datos
    print("📂 Cargando nube de puntos...")
    las = laspy.read(args.input_file)
    xyz_raw = np.vstack((las.x, las.y, las.z)).transpose()
    
    # --- RGB EXTRACTION ---
    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        print("🌈 Color RGB detectado. Normalizando...")
        # Normalizar 16-bit a [0-1]
        red = las.red.astype(np.float32) / 65535.0
        green = las.green.astype(np.float32) / 65535.0
        blue = las.blue.astype(np.float32) / 65535.0
        rgb_raw = np.vstack((red, green, blue)).transpose()
    else:
        print("⚠️ NO se detectó canal RGB. Rellenando con gris (0.5)...")
        rgb_raw = np.full_like(xyz_raw, 0.5)

    # 2. Features Globales
    # Intentar leer radio del checkpoint
    match = re.search(r'_R(\d+\.\d+)_', args.checkpoint)
    if match:
        radius = float(match.group(1))
        print(f"📡 Radio detectado en checkpoint: {radius}m")
    else:
        radius = 2.5 # Default safest
        print(f"⚠️ No se detectó radio en nombre. Usando default: {radius}m")
        
    xyz, rgb, normals, verticality = compute_features_global_v4(xyz_raw, rgb_raw, radius=radius)
    
    # Concatenar: [x, y, z, r, g, b, nx, ny, nz, verticality] (10 canales)
    full_data = np.hstack([xyz, rgb, normals, verticality.reshape(-1, 1)]) 
    num_total_points = len(xyz)
    
    # 3. Grid Voxelization
    print(f"📦 Dividiendo en Grids de {args.block_size}x{args.block_size}m...")
    min_coord = np.min(xyz, axis=0)
    grid_indices = np.floor((xyz[:, :2] - min_coord[:2]) / args.block_size).astype(int)
    
    print("⚡ Indexando puntos a Grids...")
    grid_hashes = grid_indices[:, 0] * 100000 + grid_indices[:, 1]
    sort_ord = np.argsort(grid_hashes)
    sorted_hashes = grid_hashes[sort_ord]
    sorted_indices = sort_ord
    
    unique_hashes, split_indices = np.unique(sorted_hashes, return_index=True)
    grouped_indices = np.split(sorted_indices, split_indices[1:])
    
    from collections import defaultdict
    grid_dict = defaultdict(list)
    valid_grids = 0
    for h, indices in zip(unique_hashes, grouped_indices):
        if len(indices) > 50: 
            grid_dict[h] = indices
            valid_grids += 1
            
    print(f"   -> {valid_grids} Grids válidos identificados.")
    
    # 4. DataLoader
    dataset = GridDatasetV4(full_data, grid_dict, args.num_points, min_coord, args.block_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
    
    # 5. Inferencia
    print(f"🏗️ Cargando modelo V4 (d_in=10): {args.checkpoint}")
    model = RandLANet(d_in=10, num_classes=2).to(device)
    
    checkpoint = torch.load(args.checkpoint)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    global_preds = np.zeros(num_total_points, dtype=np.uint8)
    global_probs = np.zeros(num_total_points, dtype=np.float32)
    hit_counts = np.zeros(num_total_points, dtype=np.int8)
    
    print("🧠 Ejecutando RandLANet V4...")
    with torch.no_grad():
        for features, batch_indices_list in tqdm(dataloader):
            # features: [B, N, 10]
            features = features.to(device)
            
            # XYZ (index 0-2)
            xyz_tensor = features[:, :, :3]
            
            outputs = model(xyz_tensor, features) 
            probs_batch = torch.softmax(outputs, dim=1)[:, 1, :]
            
            probs_np = probs_batch.cpu().numpy()
            batch_indices = batch_indices_list.numpy()
            
            flat_indices = batch_indices.flatten()
            flat_probs = probs_np.flatten()
            
            # --- FIX HEURÍSTICO V4 ---
            # Features index 9 es Verticalidad en V4 (antes era 6)
            # [x,y,z, r,g,b, nx,ny,nz, v] -> indices 0..9
            batch_vert = features[:, :, 9].cpu().numpy().flatten()
            mask_flat = batch_vert < 0.05
            flat_probs[mask_flat] = 0.0
            
            global_probs[flat_indices] = flat_probs
            hit_counts[flat_indices] = 1

    # 6. Post-Procesamiento (DBSCAN)
    print("🧹 Post-Procesamiento: Filtrando ruido con DBSCAN...")
    global_preds[global_probs > args.conf_threshold] = 1
    pred_indices = np.where(global_preds == 1)[0]
    
    if len(pred_indices) > 0:
        if len(pred_indices) < 5000000:
            pcd_pred = o3d.geometry.PointCloud()
            pcd_pred.points = o3d.utility.Vector3dVector(xyz[pred_indices])
            
            labels_clust = np.array(pcd_pred.cluster_dbscan(eps=1.5, min_points=30, print_progress=False))
            
            if len(labels_clust) > 0:
                unique_lbs, counts = np.unique(labels_clust, return_counts=True)
                noise_labels = unique_lbs[counts < 50] 
                noise_labels = np.append(noise_labels, [-1]) 
                
                is_noise = np.isin(labels_clust, noise_labels)
                noise_indices = pred_indices[is_noise]
                global_preds[noise_indices] = 0
                print(f"   ⬇️ Ruido eliminado (Clusters pequeños): {len(noise_indices)} puntos.")
        else:
            print("⚠️ Saltando DBSCAN por exceso de puntos.")

    missed = np.sum(hit_counts == 0)
    if missed > 0:
        print(f"⚠️ {missed} puntos no inferidos. Asumidos Suelo.")

    print(f"   Puntos maquinaria finales: {np.sum(global_preds == 1):,}")
    print(f"💾 Guardando LAS V4: {args.output_file}")
    
    las_out_classes = np.zeros(num_total_points, dtype=np.uint8)
    las_out_classes[global_preds == 0] = 2 
    las_out_classes[global_preds == 1] = 1 
    
    new_las = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
    new_las.header.offsets = las.header.offsets
    new_las.header.scales = las.header.scales
    
    new_las.x = xyz[:, 0]
    new_las.y = xyz[:, 1]
    new_las.z = xyz[:, 2]
    
    # Guardar color si existe
    if rgb is not None:
        new_las.red = (rgb[:, 0] * 65535).astype(np.uint16)
        new_las.green = (rgb[:, 1] * 65535).astype(np.uint16)
        new_las.blue = (rgb[:, 2] * 65535).astype(np.uint16)

    new_las.classification = las_out_classes
    new_las.intensity = (global_probs * 65535).astype(np.uint16)
    
    new_las.write(args.output_file)
    print("✅ Inferencia RandLANet V4 Terminada.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_file', type=str, default=None)
    
    # Defaults V4 RandLANet
    parser.add_argument('--block_size', type=float, default=30.0) 
    parser.add_argument('--num_points', type=int, default=65536)
    parser.add_argument('--batch_size', type=int, default=12) 
    parser.add_argument('--conf_threshold', type=float, default=0.50)
    
    args = parser.parse_args()
    
    if args.output_file is None:
        input_basename = os.path.basename(args.input_file)
        filename_no_ext = os.path.splitext(input_basename)[0]
        output_dir = "data/predictions_v4"
        os.makedirs(output_dir, exist_ok=True)
        args.output_file = os.path.join(output_dir, f"{filename_no_ext}_RNET_V4.laz")
        
    run_inference(args)
