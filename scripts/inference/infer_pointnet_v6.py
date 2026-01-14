import os
import torch
import numpy as np
import laspy
import open3d as o3d 
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader, Dataset
import sys
from torch.amp import autocast

sys.path.append(os.getcwd())
from src.models.pointnet2 import PointNet2

def compute_features_fast(las, radius=2.5):
    """
    Versión NITRO V6 (0.25m): 
    1. No limpia duplicados (Asumimos fotogrametría limpia).
    2. Intenta leer normales del LAS. Si no existen, solo ahí usa Open3D.
    """
    print("🚀 [V6 NITRO] Extrayendo puntos...")
    xyz = np.vstack((las.x, las.y, las.z)).transpose()
    
    if len(xyz) == 0:
        raise ValueError(f"⚠️ El archivo LAS está vacío (0 puntos).")

    # 1. RGB
    if hasattr(las, 'red'):
        print("   🎨 Usando RGB nativo...")
        scale = 65535.0 if np.max(las.red) > 255 else 255.0
        rgb = np.vstack((las.red, las.green, las.blue)).transpose() / scale
    else:
        rgb = np.full_like(xyz, 0.5)

    # 2. NORMALES
    has_normals = False
    normals = None

    if hasattr(las, 'normal_x'):
        print("   ⚡ Usando Normales Nativas del LAS (Ultra Rápido)...")
        nx = np.array(las.normal_x)
        ny = np.array(las.normal_y)
        nz = np.array(las.normal_z)
        normals = np.vstack((nx, ny, nz)).transpose()
        has_normals = True
    
    if not has_normals:
        print(f"   ⚠️ No detectadas normales. Calculando con Open3D (r={radius}m)...")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
        pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0., 0., 1.]))
        normals = np.asarray(pcd.normals)
        normals[normals[:, 2] < 0] *= -1 

    return xyz, rgb, normals

class GridDatasetNitro(Dataset):
    def __init__(self, full_data, grid_dict, num_points, min_coord, block_size):
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
        
        n_idx = len(indices)
        if n_idx >= self.num_points:
            sel = np.random.choice(n_idx, self.num_points, replace=False)
            selected_indices = indices[sel]
        else:
            sel = np.random.choice(n_idx, self.num_points, replace=True)
            selected_indices = indices[sel]
            
        block_data = self.full_data[selected_indices] 
        
        # Normalización Vectorizada
        xyz = block_data[:, :3]
        
        ix = key // 100000
        iy = key % 100000
        tile_origin_x = self.min_coord[0] + ix * self.block_size
        tile_origin_y = self.min_coord[1] + iy * self.block_size
        
        block_data[:, 0] -= (tile_origin_x + self.block_size/2.0)
        block_data[:, 1] -= (tile_origin_y + self.block_size/2.0)
        block_data[:, 2] -= np.min(xyz[:, 2]) 
        
        return torch.from_numpy(block_data).float(), selected_indices

def run_inference(args):
    device = torch.device('cuda')
    print(f"🚀 Iniciando Inferencia V6 NITRO (Sync 0.25m): {args.input_file}")
    
    las = laspy.read(args.input_file)
    
    # 2. Extraer features
    import re
    match = re.search(r'_R(\d+\.\d+)_', args.checkpoint)
    radius = float(match.group(1)) if match else 3.5
    
    xyz, rgb, normals = compute_features_fast(las, radius=radius)
    
    # [XYZ, RGB, Normals] -> 9 canales
    full_data = np.hstack([xyz, rgb, normals]).astype(np.float32)
    del las, xyz, rgb, normals
    
    # 3. Grid Voxelization
    print(f"📦 Gridding ({args.block_size}m)...")
    min_coord = np.min(full_data[:, :3], axis=0)
    
    grid_x = ((full_data[:, 0] - min_coord[0]) // args.block_size).astype(np.int32)
    grid_y = ((full_data[:, 1] - min_coord[1]) // args.block_size).astype(np.int32)
    
    grid_hashes = grid_x * 100000 + grid_y
    
    sort_idx = np.argsort(grid_hashes)
    sorted_hashes = grid_hashes[sort_idx]
    
    unique_hashes, split_indices = np.unique(sorted_hashes, return_index=True)
    grouped_indices = np.split(sort_idx, split_indices[1:])
    
    grid_dict = {h: idx for h, idx in zip(unique_hashes, grouped_indices) if len(idx) > 20} # Umbral bajo para 0.25m
    
    print(f"   -> {len(grid_dict)} bloques activos.")

    # 4. Loader
    dataset = GridDatasetNitro(full_data, grid_dict, args.num_points, min_coord, args.block_size)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=12, 
        shuffle=False, 
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2 
    )
    
    # 5. Modelo
    print("🏗️ Cargando Modelo V6...")
    model = PointNet2(d_in=9, num_classes=2, base_radius=radius).to(device)
    ckpt = torch.load(args.checkpoint)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model.eval()
    
    if not args.no_compile:
        print("🔥 Compilando Graph...")
        try:
            model = torch.compile(model) 
        except Exception as e:
            print(f"⚠️ Falló torch.compile: {e}. Ejecutando en modo Eager.")
    else:
        print("🛑 torch.compile desactivado.") 

    global_probs = np.zeros(len(full_data), dtype=np.float16)
    
    print("🧠 Ejecutando...")
    with torch.no_grad():
        for batch_data, batch_indices in tqdm(dataloader):
            batch_data = batch_data.to(device, non_blocking=True)
            xyz_tensor = batch_data[:, :, :3]

            with autocast(device_type='cuda'):
                logits = model(xyz_tensor, batch_data)
                probs = torch.softmax(logits, dim=1)[:, 1, :]
            
            probs_np = probs.cpu().numpy().flatten()
            indices_np = batch_indices.numpy().flatten()
            global_probs[indices_np] = probs_np

    # Guardar
    print("💾 Guardando...")
    preds = (global_probs > 0.5).astype(np.uint8)
    
    las_in = laspy.read(args.input_file)
    new_las = laspy.create(point_format=las_in.header.point_format, file_version=las_in.header.version)
    new_las.header = las_in.header
    new_las.x = las_in.x
    new_las.y = las_in.y
    new_las.z = las_in.z
    
    if hasattr(las_in, 'red'):
        new_las.red = las_in.red
        new_las.green = las_in.green
        new_las.blue = las_in.blue
        
    new_las.classification = np.where(preds==1, 1, 2).astype(np.uint8)
    new_las.write(args.output_file)
    print("✅ Fin V6.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_file', type=str, default="data/predictions_v6/fast_out.laz")
    parser.add_argument('--block_size', type=float, default=10.0)
    # AJUSTE CRÍTICO V6: 2048 Puntos
    parser.add_argument('--num_points', type=int, default=2048) 
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--no_compile', action='store_true')
    args = parser.parse_args()
    
    torch.set_float32_matmul_precision('high')
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    
    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        
    run_inference(args)
