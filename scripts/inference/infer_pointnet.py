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
import os

# Fix import path
sys.path.append(os.getcwd())

# Importar modelo
from src.models.pointnet2 import PointNet2

def compute_features_global(xyz, radius=2.5):
    """
    Computa normales y verticalidad sobre TODA la nube para evitar bordes.
    Limpieza de geometría incluida.
    """
    print("🧹 Limpiando nube de puntos (Duplicados, NaNs)...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    # Limpieza crítica
    pcd = pcd.remove_duplicated_points()
    pcd = pcd.remove_non_finite_points()
    
    # Actualizar xyz limpio
    xyz_clean = np.asarray(pcd.points)
    print(f"   Puntos tras limpieza: {len(xyz_clean):,}")
    
    print("🔍 Calculando normales globales (r=2.5m)...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    
    # Orientar a +Z (Minería cielo abierto)
    pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0., 0., 1.]))
    
    normals = np.asarray(pcd.normals)
    # Fix matemático extra por si alguna normal quedó invertida
    normals[normals[:, 2] < 0] *= -1
    
    # --- FIX CRÍTICO: Verticalidad Invertida ---
    # Preprocessing (Training): verticality = 1.0 - np.abs(normals[:, 2])
    # (1.0 = Muro/Vertical, 0.0 = Suelo/Plano)
    # Antes teníamos: verticality = np.abs(normals[:, 2]) (Inverso)
    
    verticality = 1.0 - np.abs(normals[:, 2]) # Nz absoluto
    
    return xyz_clean, normals, verticality

class GridDataset(Dataset):
    def __init__(self, full_data, grid_dict, num_points, min_coord, block_size):
        """
        full_data: [N_total, C]
        grid_dict: dict {grid_id: [point_indices]}
        num_points: fixed points per block (resampling)
        min_coord: [x, y, z] global offset used for grid calculation
        block_size: size of the grid in meters
        """
        self.full_data = full_data
        self.grid_keys = list(grid_dict.keys())
        self.grid_dict = grid_dict
        self.num_points = num_points
        self.min_coord = min_coord # Numpy array
        self.block_size = block_size
        
    def __len__(self):
        return len(self.grid_keys)
    
    def __getitem__(self, idx):
        key = self.grid_keys[idx]
        indices = self.grid_dict[key]
        
        # Resampling strategy
        if len(indices) >= self.num_points:
            # Downsample (Random Choice)
            selected_indices = np.random.choice(indices, self.num_points, replace=False)
        else:
            # Upsample (Repeat)
            selected_indices = np.random.choice(indices, self.num_points, replace=True)
            
        block_data = self.full_data[selected_indices]
        
        # --- NORMALIZACIÓN GEOMÉTRICA CORRECTA (V2 Match) ---
        # Training data usa coordenadas relativas al CENTRO DEL TILE (Teórico).
        # Análisis: Mean != 0 en training implica que no es centroide de datos, sino centro de caja.
        
        # 1. Recuperar índices del Grid (ix, iy) desde el Hash
        # Hash = ix * 100000 + iy
        ix = key // 100000
        iy = key % 100000
        
        # 2. Calcular Centro Teórico del Tile en coordenadas Globales
        # min_coord es el origen del tiling (0,0 relativo)
        tile_origin_x = self.min_coord[0] + ix * self.block_size
        tile_origin_y = self.min_coord[1] + iy * self.block_size
        
        tile_center_x = tile_origin_x + (self.block_size / 2.0)
        tile_center_y = tile_origin_y + (self.block_size / 2.0)
        
        # 3. Normalizar XYZ
        xyz = block_data[:, :3]
        xyz_norm = xyz.copy()
        
        # XY: Relativo al centro del Tile -> Rango aprox [-5, 5]
        xyz_norm[:, 0] = xyz[:, 0] - tile_center_x
        xyz_norm[:, 1] = xyz[:, 1] - tile_center_y
        
        # Z: Relativo al suelo local (Min Z del bloque) -> Rango [0, H]
        z_min = np.min(xyz[:, 2])
        xyz_norm[:, 2] = xyz[:, 2] - z_min
        
        # Reconstruir tensor de features
        # [x_norm, y_norm, z_norm, nx, ny, nz, vert]
        features = block_data.copy()
        features[:, :3] = xyz_norm
        
        return torch.FloatTensor(features), selected_indices

def run_inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Iniciando Inferencia Grid-Based: {args.input_file}")
    
    # 1. Cargar datos
    print("📂 Cargando nube de puntos...")
    las = laspy.read(args.input_file)
    xyz_raw = np.vstack((las.x, las.y, las.z)).transpose()
    
    # 2. Features Globales (Radio Adaptativo)
    # Intentar leer radio del checkpoint: "LR..._R3.5_..."
    import re
    match = re.search(r'_R(\d+\.\d+)_', args.checkpoint)
    if match:
        radius = float(match.group(1))
        print(f"📡 Radio detectado en checkpoint: {radius}m")
    else:
        radius = 2.5
        print(f"⚠️ No se detectó radio en nombre. Usando default: {radius}m")

    xyz, normals, verticality = compute_features_global(xyz_raw, radius=radius)
    
    # Concatenar: [x, y, z, nx, ny, nz, verticality]
    full_data = np.hstack([xyz, normals, verticality.reshape(-1, 1)]) # [N_total, 7]
    num_total_points = len(xyz)
    
    # 3. Grid Voxelization (O(N))
    print(f"📦 Dividiendo en Grids de {args.block_size}x{args.block_size}m...")
    
    # Min coord para offset
    min_coord = np.min(xyz, axis=0)
    
    # Calcular Grid Indices
    # ix = floor((x - min_x) / block_size)
    grid_indices = np.floor((xyz[:, :2] - min_coord[:2]) / args.block_size).astype(int)
    
    # Hash map para agrupar
    # Usamos un dict para mapear (ix, iy) -> [lista de indices reales]
    from collections import defaultdict
    grid_dict = defaultdict(list)
    
    print("⚡ Indexando puntos a Grids...")
    # Optimizacion: iterar sobre unique rows es lento en python puro si son millones
    # Usaremos un truco de hashing 1D para numpy
    # hash = ix * 100000 + iy
    grid_hashes = grid_indices[:, 0] * 100000 + grid_indices[:, 1]
    unique_hashes = np.unique(grid_hashes)
    
    # Iterar y agrupar (esto puede tardar un poco en CPU single thread, pero es seguro)
    # Para 12M de puntos, un loop puro es lento. Mejor ordenar.
    sort_ord = np.argsort(grid_hashes)
    sorted_hashes = grid_hashes[sort_ord]
    sorted_indices = sort_ord
    
    # Split por cambios de hash
    unique_hashes, split_indices = np.unique(sorted_hashes, return_index=True)
    grouped_indices = np.split(sorted_indices, split_indices[1:])
    
    # Rellenar diccionario
    valid_grids = 0
    for h, indices in zip(unique_hashes, grouped_indices):
        if len(indices) > 50: # Descartar ruido/bordes vacíos
            grid_dict[h] = indices
            valid_grids += 1
            
    print(f"   -> {valid_grids} Grids válidos identificados.")
    
    # 4. DataLoader y Batching
    # 4. DataLoader y Batching
    dataset = GridDataset(full_data, grid_dict, args.num_points, min_coord, args.block_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
    
    # 5. Inferencia
    print(f"🏗️ Cargando modelo: {args.checkpoint} (Radius={radius}m)")
    model = PointNet2(d_in=7, num_classes=2, base_radius=radius).to(device)
    checkpoint = torch.load(args.checkpoint)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Arrays de resultados globales
    # 0 = Suelo, 1 = Maquinaria
    # Usaremos votos si hay overlap, pero en Grid NO hay overlap (son particiones).
    # Sin embargo, GridDataset hace resampling.
    # Si num_points < len(indices), perdemos puntos.
    # Si num_points > len(indices), duplicamos.
    # ESTRATEGIA: Asignar predicción a los índices seleccionados.
    # Para cubrir todos los puntos, idealmente stride < block. 
    # Pero "Grid" implica partición estricta.
    # Los puntos no seleccionados en el downsampling se quedarán sin predicción (0).
    # SOLUCIÓN: PointNet Grid Inference suele requerir procesar TODO.
    # Si el grid es muy denso, deberíamos dividirlo en sub-grids o chunks. 
    # Por ahora, asumimos que 10x10 tiene razonablemente <= 10k puntos o que el sampling es aceptable.
    
    global_preds = np.zeros(num_total_points, dtype=np.uint8)
    global_probs = np.zeros(num_total_points, dtype=np.float32) # Guardar prob de clase 1
    hit_counts = np.zeros(num_total_points, dtype=np.int8) # Para saber si fue visitado
    
    print("🧠 Ejecutando red neuronal...")
    with torch.no_grad():
        for features, batch_indices_list in tqdm(dataloader):
            # features: [B, N, 7]
            features = features.to(device)
            # El modelo (PointNet2.py) espera [B, N, 7] y hace el permute internamente.
            # NO permutar aquí.
            
            # --- SURGICAL FIX (USER REQUEST) ---
            # Separar XYZ y Features para cumplir con la "firma" esperada, 
            # aunque PointNet2 internamente use features slicing.
            # Pasamos XYZ explícitamente en el primer argumento.
            
            xyz_tensor = features[:, :, :3] # [B, N, 3]
            
            outputs = model(xyz_tensor, features) # [B, 2, N]
            probs_batch = torch.softmax(outputs, dim=1)[:, 1, :] # [B, N] Prob clase 1
            
            probs_np = probs_batch.cpu().numpy()
            
            # batch_indices = batch_indices_list.numpy() # [B, N]
            batch_indices = batch_indices_list.numpy()
            
            # --- DEBUG PROBABILIDADES ---
            max_prob_maq = np.max(probs_np)
            if max_prob_maq > 0.05:
                print(f"   [Debug] Batch con posible Maquinaria. Prob Max: {max_prob_maq:.4f}")

            # Asignar resultados            
            # Flatten para operación vectorizada rápida
            flat_indices = batch_indices.flatten()
            flat_probs = probs_np.flatten()
            
            # Asignación directa (Last write wins o Max wins)
            # Como es resampling aleatorio, si un punto sale 2 veces recibe la ultima.
            # Mejor usar atomic max o simplemente asignar.
            # Asignación directa
            # global_probs[flat_indices] = flat_probs
            
            # --- FIX HEURÍSTICO (REALITY CHECK) ---
            # El modelo fue entrenado SIN "Easy Negatives" (Suelo Plano), por lo que 
            # al ver suelo plano (V~0) alucina y predice Maquinaria.
            # Imponemos una restricción física: Si es plano, NO ES MAQUINARIA.
            
            # features es [B, N, 7]. Canal 6 es verticalidad.
            batch_vert = features[:, :, 6].cpu().numpy().flatten()
            mask_flat = batch_vert < 0.05
            
            # Aplicar filtro: Si V < 0.05 -> Prob = 0.0
            flat_probs[mask_flat] = 0.0
            
            global_probs[flat_indices] = flat_probs
            hit_counts[flat_indices] = 1

    # Asignación directa
    global_probs[flat_indices] = flat_probs
    hit_counts[flat_indices] = 1

    # --- POST-PROCESAMIENTO: FILTRADO DE RUIDO (DBSCAN) ---
    print("🧹 Post-Procesamiento: Filtrando ruido con DBSCAN...")
    
    # 1. Umbralizado inicial
    global_preds[global_probs > args.conf_threshold] = 1
    
    # 2. Extract Class 1 points
    pred_indices = np.where(global_preds == 1)[0]
    
    if len(pred_indices) > 0:
        print(f"   Iniciando filtrado de {len(pred_indices)} puntos candidatos...")
        pcd_pred = o3d.geometry.PointCloud()
        pcd_pred.points = o3d.utility.Vector3dVector(xyz[pred_indices])
        
        # DBSCAN: eps=1.5m (distancia entre puntos de una máquina), min_points=20
        # Esto elimina "líneas" sueltas o puntos aislados.
        labels_clust = np.array(pcd_pred.cluster_dbscan(eps=1.5, min_points=30, print_progress=False))
        
        if len(labels_clust) > 0:
            max_label = labels_clust.max()
            print(f"   Clusters encontrados: {max_label + 1}")
            
            # Filtrar clusters muy pequeños ( Ruido )
            # Una máquina debe tener al menos X puntos (depende de densidad).
            # Asumimos que una máquina real > 50 puntos.
            
            # Estrategia rápida: contar ocurrencias
            unique_lbs, counts = np.unique(labels_clust, return_counts=True)
            noise_labels = unique_lbs[counts < 50] # Menos de 50 puntos es ruido
            noise_labels = np.append(noise_labels, [-1]) # -1 es ruido DBSCAN
            
            # Crear máscara de ruido
            is_noise = np.isin(labels_clust, noise_labels)
            
            # Los índices que son ruido vuelven a clase 0 (Suelo)
            noise_indices = pred_indices[is_noise]
            global_preds[noise_indices] = 0
            
            print(f"   ⬇️ Ruido eliminado: {len(noise_indices)} puntos.")
        
    # puntos no visitados (hit_counts == 0) quedan como Suelo (0) por defecto.
    missed = np.sum(hit_counts == 0)
    if missed > 0:
        print(f"⚠️ {missed} puntos no fueron inferidos (perdidos en downsampling). Se asumen Suelo.")

    print(f"   Puntos maquinaria detectados: {np.sum(global_preds == 1):,}")
    
    print(f"💾 Guardando LAS final: {args.output_file}")
    
    # Mapping a estándar LAS: 0->2(Suelo), 1->1(Maq)
    las_out_classes = np.zeros(num_total_points, dtype=np.uint8)
    las_out_classes[global_preds == 0] = 2
    las_out_classes[global_preds == 1] = 1
    
    new_las = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
    new_las.header.offsets = las.header.offsets
    new_las.header.scales = las.header.scales
    
    new_las.x = xyz[:, 0]
    new_las.y = xyz[:, 1]
    new_las.z = xyz[:, 2]
    new_las.classification = las_out_classes
    new_las.intensity = (global_probs * 65535).astype(np.uint16)
    
    new_las.write(args.output_file)
    print("✅ Inferencia Grid-Based Terminada.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--block_size', type=float, default=10.0)
    parser.add_argument('--num_points', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=64) # Batch robusto para 5090
    parser.add_argument('--conf_threshold', type=float, default=0.50) # Umbral balanceado recuperado
    
    args = parser.parse_args()
    
    if args.output_file is None:
        input_basename = os.path.basename(args.input_file)
        filename_no_ext = os.path.splitext(input_basename)[0]
        output_dir = "data/predictions"
        os.makedirs(output_dir, exist_ok=True)
        args.output_file = os.path.join(output_dir, f"{filename_no_ext}_POINTNET_GRID.laz")
        
    run_inference(args)
