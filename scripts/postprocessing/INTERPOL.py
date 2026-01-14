import laspy
import numpy as np
from scipy.spatial import cKDTree
import argparse
from tqdm import tqdm

def flatten_machinery(input_file, output_file, k_neighbors=8, max_dist=50.0):
    print(f"🚜 Encendiendo el Bulldozer Digital en: {input_file}")
    
    # 1. Cargar LAS
    las = laspy.read(input_file)
    xyz = np.vstack((las.x, las.y, las.z)).transpose()
    classification = np.array(las.classification)
    
    # Índices
    idx_maq = np.where(classification == 1)[0]   # Lo que vamos a aplanar
    idx_suelo = np.where(classification == 2)[0] # La referencia (Suelo real)
    
    if len(idx_maq) == 0:
        print("⚠️ No hay maquinaria para aplanar.")
        return

    print(f"   📉 Puntos a aplanar: {len(idx_maq)}")
    print(f"   🏔️ Puntos de referencia (Suelo): {len(idx_suelo)}")
    
    # 2. Construir KDTree con el SUELO REAL
    # Esto nos permite preguntar rápido: "¿Qué altura tiene el suelo cerca de aquí?"
    print("   🌳 Construyendo índice espacial del suelo...")
    tree = cKDTree(xyz[idx_suelo, :2]) # Solo indexamos X, Y (2D)
    
    # 3. Buscar vecinos para cada punto de maquinaria
    # Buscamos los k vecinos de suelo más cercanos en XY
    print("   🔍 Interpolando alturas (IDW)...")
    
    # Query devuelve distancias e índices de los vecinos en idx_suelo
    dists, neighbors_indices = tree.query(xyz[idx_maq, :2], k=k_neighbors, distance_upper_bound=max_dist, workers=-1)
    
    # 4. Interpolación Inversa a la Distancia (IDW)
    # Z_new = sum(Z_i / d_i) / sum(1 / d_i)
    
    new_z_values = np.zeros(len(idx_maq))
    
    # Iteramos por bloques para no explotar la RAM si son millones
    # (Aunque vectorizado con numpy suele aguantar bien)
    
    # Manejo de infinitos (si no encuentra vecinos dentro de max_dist)
    valid_mask = np.isfinite(dists).all(axis=1)
    
    # Para los puntos válidos
    valid_dists = dists[valid_mask]
    valid_neighbors = neighbors_indices[valid_mask]
    
    # Evitar división por cero (si un punto de maq está EXACTAMENTE sobre uno de suelo)
    valid_dists[valid_dists < 0.001] = 0.001
    
    weights = 1.0 / valid_dists
    
    # Obtener Z de los vecinos
    # neighbors_indices apunta a la lista idx_suelo, y esa apunta a xyz
    z_neighbors = xyz[idx_suelo[valid_neighbors], 2]
    
    # Fórmula IDW Vectorizada
    numerator = np.sum(weights * z_neighbors, axis=1)
    denominator = np.sum(weights, axis=1)
    
    interpolated_z = numerator / denominator
    
    # 5. Aplicar cambios
    # Actualizamos Z solo en los puntos de maquinaria
    # Los que no tuvieron vecinos (muy lejos del suelo) no se tocan o se borran
    
    # Asignar Z interpolada
    final_z = xyz[:, 2].copy()
    
    # Mapeo de índices globales
    indices_to_update = idx_maq[valid_mask]
    final_z[indices_to_update] = interpolated_z
    
    # CAMBIAR CLASE: Maquinaria (1) -> Suelo (2)
    # Opcional: Podríamos crear una clase "Suelo Artificial" (ej: 8) para distinguirlo.
    # Pero si quieres "DTMaster style", todo vuelve a ser suelo (2).
    classification[indices_to_update] = 2 
    
    print(f"   ✅ Se aplanaron {len(indices_to_update)} puntos exitosamente.")
    
    # 6. Guardar
    new_las = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
    new_las.header = las.header
    new_las.x = las.x
    new_las.y = las.y
    new_las.z = final_z # Z modificada!
    new_las.classification = classification # Clases modificadas!
    
    if hasattr(las, 'red'):
        new_las.red = las.red
        new_las.green = las.green
        new_las.blue = las.blue
        
    new_las.write(output_file)
    print(f"💾 DTM limpio guardado en: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--k', type=int, default=8, help="Número de vecinos de suelo para interpolar")
    parser.add_argument('--max_dist', type=float, default=100.0, help="Distancia máxima para buscar suelo")
    
    args = parser.parse_args()
    
    flatten_machinery(args.input, args.output, args.k, args.max_dist)