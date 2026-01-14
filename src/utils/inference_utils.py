import torch
import numpy as np
import laspy
import open3d as o3d
from typing import Tuple, List, Dict
import os


def compute_normals_open3d(points: np.ndarray, search_radius: float = 2.0, max_nn: int = 30) -> np.ndarray:
    """
    Calcula las normales de una nube de puntos usando Open3D.
    
    Args:
        points: Array de puntos [N, 3]
        search_radius: Radio de búsqueda para estimación de normales
        max_nn: Número máximo de vecinos
        
    Returns:
        Array de normales [N, 3]
    """
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_centered)
    
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=max_nn)
    )
    pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0., 0., 1.]))
    
    return np.asarray(pcd.normals)


def create_blocks(xyz: np.ndarray, features: np.ndarray, block_size: float = 50.0, 
                  stride: float = 50.0, min_points: int = 1000) -> List[Dict]:
    """
    Divide una nube de puntos en bloques espaciales.
    
    Args:
        xyz: Coordenadas XYZ [N, 3]
        features: Features incluyendo normales [N, 6] (XYZ + Normales)
        block_size: Tamaño del bloque en metros
        stride: Paso entre bloques (sin overlap si stride == block_size)
        min_points: Mínimo de puntos por bloque
        
    Returns:
        Lista de diccionarios con información de cada bloque
    """
    min_x, min_y = np.min(xyz[:, 0]), np.min(xyz[:, 1])
    max_x, max_y = np.max(xyz[:, 0]), np.max(xyz[:, 1])
    
    rango_x = np.arange(min_x, max_x, stride)
    rango_y = np.arange(min_y, max_y, stride)
    
    blocks = []
    
    for x_curr in rango_x:
        for y_curr in rango_y:
            x_end, y_end = x_curr + block_size, y_curr + block_size
            
            mask_box = (xyz[:, 0] >= x_curr) & (xyz[:, 0] < x_end) & \
                       (xyz[:, 1] >= y_curr) & (xyz[:, 1] < y_end)
            
            if np.sum(mask_box) < min_points:
                continue
            
            block_xyz = xyz[mask_box]
            block_features = features[mask_box]
            
            blocks.append({
                'xyz': block_xyz,
                'features': block_features,
                'mask': mask_box,
                'bounds': (x_curr, y_curr, x_end, y_end),
                'num_points': np.sum(mask_box)
            })
    
    return blocks


def prepare_block_for_inference(xyz: np.ndarray, features: np.ndarray, 
                                num_points: int = 4096) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepara un bloque para inferencia: sampling y normalización.
    
    Args:
        xyz: Coordenadas del bloque [N, 3]
        features: Features del bloque [N, 6]
        num_points: Número de puntos objetivo
        
    Returns:
        Tupla (xyz_tensor, features_tensor) listos para el modelo
    """
    # Sampling
    if len(xyz) >= num_points:
        choice_idx = np.random.choice(len(xyz), num_points, replace=False)
    else:
        choice_idx = np.random.choice(len(xyz), num_points, replace=True)
    
    curr_xyz = xyz[choice_idx, :]
    curr_feat = features[choice_idx, :]
    
    # Normalización (igual que en entrenamiento)
    centroid = np.mean(curr_xyz, axis=0)
    curr_xyz[:, :2] -= centroid[:2]  # Centrar XY
    curr_xyz[:, 2] -= np.min(curr_xyz[:, 2])  # Z desde 0
    
    # Convertir a tensores [1, N, C] (batch size = 1)
    xyz_tensor = torch.from_numpy(curr_xyz).float().unsqueeze(0)
    features_tensor = torch.from_numpy(curr_feat).float().unsqueeze(0)
    
    return xyz_tensor, features_tensor


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, device: str = 'cuda') -> torch.nn.Module:
    """
    Carga un checkpoint en el modelo.
    
    Args:
        checkpoint_path: Ruta al archivo .pth
        model: Modelo a cargar
        device: Dispositivo (cuda/cpu)
        
    Returns:
        Modelo con pesos cargados
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint no encontrado: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    return model


def preprocess_las_file(las_path: str, block_size: float = 50.0, stride: float = 50.0,
                        num_points: int = 4096, search_radius: float = 2.0,
                        max_nn: int = 30) -> Tuple[laspy.LasData, List[Dict]]:
    """
    Preprocesa un archivo .las completo para inferencia.
    
    Args:
        las_path: Ruta al archivo .las/.laz
        block_size: Tamaño de bloque en metros
        stride: Paso entre bloques
        num_points: Puntos por bloque
        search_radius: Radio para cálculo de normales
        max_nn: Vecinos máximos para normales
        
    Returns:
        Tupla (las_data_original, lista_de_bloques_procesados)
    """
    print(f"📂 Cargando archivo: {las_path}")
    las = laspy.read(las_path)
    
    # Extraer coordenadas
    xyz = np.vstack((las.x, las.y, las.z)).transpose()
    print(f"   Total de puntos: {len(xyz):,}")
    
    # Calcular normales
    print("   🔄 Calculando normales...")
    normals = compute_normals_open3d(xyz, search_radius=search_radius, max_nn=max_nn)
    
    # Combinar XYZ + Normales
    features = np.hstack((xyz, normals))
    
    # Crear bloques
    print(f"   📦 Dividiendo en bloques de {block_size}m x {block_size}m...")
    blocks = create_blocks(xyz, features, block_size=block_size, stride=stride)
    print(f"   ✅ {len(blocks)} bloques creados")
    
    return las, blocks


def assemble_predictions(original_las: laspy.LasData, blocks: List[Dict], 
                        predictions_list: List[np.ndarray]) -> np.ndarray:
    """
    Ensambla las predicciones de todos los bloques en un array completo.
    
    Args:
        original_las: Archivo LAS original
        blocks: Lista de bloques procesados
        predictions_list: Lista de predicciones por bloque
        
    Returns:
        Array de clasificaciones [N] para todos los puntos
    """
    total_points = len(original_las.points)
    
    # Inicializar con clase por defecto (2 = Suelo)
    final_predictions = np.full(total_points, 2, dtype=np.uint8)
    
    # Contador de votos por punto (para manejar overlaps)
    vote_count = np.zeros(total_points, dtype=np.int32)
    vote_sum = np.zeros(total_points, dtype=np.int32)
    
    for block, predictions in zip(blocks, predictions_list):
        mask = block['mask']
        
        # Acumular votos
        vote_sum[mask] += predictions
        vote_count[mask] += 1
    
    # Promediar votos donde hubo overlap
    mask_voted = vote_count > 0
    final_predictions[mask_voted] = np.round(vote_sum[mask_voted] / vote_count[mask_voted]).astype(np.uint8)
    
    return final_predictions


def save_classified_las(original_las: laspy.LasData, predictions: np.ndarray, 
                       output_path: str, class_mapping: Dict[int, int] = None):
    """
    Guarda un archivo .las con las clasificaciones.
    
    Args:
        original_las: Archivo LAS original
        predictions: Array de predicciones [N]
        output_path: Ruta de salida
        class_mapping: Mapeo de clases modelo -> LAS (ej: {0: 2, 1: 1})
    """
    # Crear copia del LAS original
    las_out = laspy.LasData(original_las.header)
    las_out.points = original_las.points
    
    # Aplicar mapeo de clases si se proporciona
    if class_mapping is not None:
        mapped_predictions = np.copy(predictions)
        for model_class, las_class in class_mapping.items():
            mapped_predictions[predictions == model_class] = las_class
        predictions = mapped_predictions
    
    # Asignar clasificaciones
    try:
        las_out.classification = predictions
    except:
        las_out.raw_classification = predictions
    
    # Guardar
    las_out.write(output_path)
    print(f"💾 Archivo guardado: {output_path}")


def generate_classification_report(predictions: np.ndarray, 
                                   class_names: Dict[int, str] = None) -> Dict:
    """
    Genera un reporte estadístico de las clasificaciones.
    
    Args:
        predictions: Array de predicciones
        class_names: Diccionario {clase: nombre}
        
    Returns:
        Diccionario con estadísticas
    """
    if class_names is None:
        class_names = {1: "Maquinaria", 2: "Suelo"}
    
    unique, counts = np.unique(predictions, return_counts=True)
    total = len(predictions)
    
    report = {
        'total_points': total,
        'classes': {}
    }
    
    for cls, count in zip(unique, counts):
        percentage = (count / total) * 100
        class_name = class_names.get(cls, f"Clase_{cls}")
        
        report['classes'][int(cls)] = {
            'name': class_name,
            'count': int(count),
            'percentage': float(percentage)
        }
    
    return report


def print_report(report: Dict):
    """
    Imprime un reporte de clasificación de forma legible.
    
    Args:
        report: Diccionario de reporte generado por generate_classification_report
    """
    print("\n" + "="*60)
    print("📊 REPORTE DE CLASIFICACIÓN")
    print("="*60)
    print(f"Total de puntos: {report['total_points']:,}")
    print("\nDistribución por clase:")
    print("-"*60)
    
    for cls_id, cls_info in sorted(report['classes'].items()):
        print(f"  {cls_info['name']:15} (Clase {cls_id}): {cls_info['count']:>10,} puntos ({cls_info['percentage']:>6.2f}%)")
    
    print("="*60 + "\n")
