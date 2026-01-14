import laspy
import numpy as np
from sklearn.cluster import DBSCAN
import argparse
from tqdm import tqdm
from scipy.spatial import cKDTree

def fix_roofs_volumetric(input_file, output_file, eps=2.0, min_samples=30, z_buffer=1.0, max_height=8.0, args=None):
    print(f"📦 Iniciando Relleno Volumétrico en: {input_file}")
    
    # 1. Cargar LAS
    las = laspy.read(input_file)
    xyz = np.vstack((las.x, las.y, las.z)).transpose()
    classification = np.array(las.classification) # 0=Suelo, 1=Maq
    
    # 2. Separar Maquinaria (Puntos Rojos)
    idx_maq = np.where(classification == 1)[0]
    
    if len(idx_maq) == 0:
        print("❌ Error: No hay maquinaria detectada en el archivo.")
        return

    print(f"   🚜 Maquinaria detectada: {len(idx_maq)} puntos")
    print("   🧩 Clusterizando objetos (DBSCAN)... esto puede tomar un momento.")
    
    # Clustering para separar cada camión individualmente
    # eps=2.0m es generoso para unir partes separadas del mismo camión
    clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(xyz[idx_maq])
    labels = clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    print(f"   🔢 Objetos encontrados: {n_clusters}")
    
    count_flipped = 0
    
    # 3. Iterar por cada objeto (Camión)
    unique_labels = set(labels)
    for lbl in tqdm(unique_labels, desc="Rellenando Cajas"):
        if lbl == -1: continue # Ruido
        
        # Obtener puntos de ESTE camión
        cluster_mask = (labels == lbl)
        cluster_points = xyz[idx_maq][cluster_mask]
        
        # Calcular Bounding Box (Caja) del camión
        # FIX: Usar percentil 5 en lugar de min absoluto para evitar outliers bajos que rompan el z_buffer
        min_x, min_y = np.min(cluster_points[:, :2], axis=0)
        min_z = np.percentile(cluster_points[:, 2], 5) 
        
        max_x, max_y, max_z = np.max(cluster_points, axis=0)
        
        # Optimización: Pre-filtro rápido con numpy (Bounding Box Grande)
        pad = args.padding if args else 1.0
        
        max_search_z = min_z + max_height
        ground_z_threshold = min_z + z_buffer
        
        # Máscara inicial (Caja AABB)
        # Identificamos candidatos potenciales rapidamente
        candidate_mask = (
            (classification == (args.ground_class if args else 2)) & 
            (xyz[:, 0] >= min_x - pad) & (xyz[:, 0] <= max_x + pad) & 
            (xyz[:, 1] >= min_y - pad) & (xyz[:, 1] <= max_y + pad) & 
            (xyz[:, 2] >= ground_z_threshold) & (xyz[:, 2] <= max_search_z)
        )
        
        candidate_indices = np.where(candidate_mask)[0]
        
        if len(candidate_indices) > 0:
            # --- REFINAMIENTO: PROYECCIÓN CILÍNDRICA 2D ---
            # El AABB captura esquinas de suelo si el camión es diagonal.
            # Verificamos que cada candidato esté cerca (XY) de algún punto REAL del camión.
            
            # Construir KDTree solo con XY del cluster
            tree_2d = cKDTree(cluster_points[:, :2])
            
            # Consultar XY de los candidatos
            candidates_xy = xyz[candidate_indices][:, :2]
            
            # Radio de "gordura" del camión (cuánto puede sobresalir el techo del chasis)
            # 1.5m es razonable para cubrir la cabina que sobresale.
            proximity_radius = 1.5 
            
            # Distancias al punto de camión más cercano en 2D
            distances, _ = tree_2d.query(candidates_xy, k=1, workers=-1)
            
            # Quedarse solo con los que están "dentro" de la silueta 2D del camión
            valid_mask = distances <= proximity_radius
            
            final_indices = candidate_indices[valid_mask]
            
            if len(final_indices) > 0:
                classification[final_indices] = 1
                count_flipped += len(final_indices)

    print(f"✅ ¡Listo! Se rellenaron {count_flipped} puntos de techo/interior.")
    
    # 5. Guardar
    new_las = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
    new_las.header = las.header
    new_las.x = las.x
    new_las.y = las.y
    new_las.z = las.z
    new_las.classification = classification
    
    if hasattr(las, 'red'):
        new_las.red = las.red
        new_las.green = las.green
        new_las.blue = las.blue
        
    new_las.write(output_file)
    print(f"💾 Guardado en: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--eps', type=float, default=2.5, help="Radio DBSCAN para agrupar camión (m)")
    parser.add_argument('--z_buffer', type=float, default=1.5, help="Altura desde el suelo para NO pintar (protege el piso debajo)")
    parser.add_argument('--max_height', type=float, default=8.0, help="Altura máxima esperada del camión (m)")
    
    parser.add_argument('--ground_class', type=int, default=2, help="Clase del suelo a corregir (Default 2)")
    parser.add_argument('--padding', type=float, default=1.0, help="Margen extra en XY para buscar techo (m)")
    
    args = parser.parse_args()
    
    fix_roofs_volumetric(args.input, args.output, eps=args.eps, z_buffer=args.z_buffer, max_height=args.max_height, args=args)