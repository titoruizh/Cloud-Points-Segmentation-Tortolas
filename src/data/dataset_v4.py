import torch
import numpy as np
import glob
import os
from torch.utils.data import Dataset

class MiningDataset(Dataset):
    def __init__(self, data_dir, num_points=4096, split='train', aug_config=None, 
                 use_filtered=False, filtered_file=None, oversample_machinery=0):
        """
        Dataset para nubes de puntos de minería con Runtime Oversampling.
        
        Args:
            data_dir: Directorio con bloques .npy
            num_points: Número de puntos por muestra
            split: 'train' o 'val'
            aug_config: Configuración de augmentación
            use_filtered: Si True, usa solo bloques de filtered_file
            filtered_file: Nombre del archivo con lista de bloques filtrados
            oversample_machinery: Factor de oversampling para MACHINERY (V3 feature)
                                 0 = desactivado
                                 2 = machinery aparece 3x (1 original + 2 copias)
        """
        self.data_dir = data_dir
        self.num_points = num_points
        self.split = split
        self.aug_config = aug_config
        self.oversample_machinery = oversample_machinery
        
        # Cargar lista de archivos
        if use_filtered and filtered_file:
            # Usar lista filtrada
            filtered_path = os.path.join(data_dir, filtered_file)
            if os.path.exists(filtered_path):
                with open(filtered_path, 'r') as f:
                    self.file_list = [line.strip() for line in f if line.strip()]
                print(f"✅ Dataset {split}: {len(self.file_list)} bloques FILTRADOS (con maquinaria)")
            else:
                print(f"⚠️ Archivo filtrado no encontrado: {filtered_path}")
                print(f"   Usando todos los bloques...")
                self.file_list = glob.glob(os.path.join(data_dir, "*.npy"))
        else:
            # Usar todos los bloques
            self.file_list = glob.glob(os.path.join(data_dir, "*.npy"))
            print(f"✅ Dataset {split}: {len(self.file_list)} bloques (sin filtro)")
        
        if len(self.file_list) == 0:
            print(f"⚠️ ADVERTENCIA: No se encontraron archivos en {data_dir}")
        
        # Aplicar runtime oversampling si configurado
        if oversample_machinery > 0 and split == 'train':
            self._apply_runtime_oversampling()

    def _apply_runtime_oversampling(self):
        """
        Oversamplear bloques MACHINERY en runtime sin duplicación física.
        Estrategia V3: Aumenta frecuencia de muestreo de maquinaria.
        """
        # Identificar bloques MACHINERY por nombre de archivo
        machinery_indices = [
            i for i, filepath in enumerate(self.file_list)
            if 'MACHINERY' in os.path.basename(filepath)
        ]
        
        if len(machinery_indices) == 0:
            print(f"⚠️ No se encontraron bloques MACHINERY para oversamplear")
            return
        
        # Crear copias virtuales de índices de maquinaria
        extra_samples = [self.file_list[i] for i in machinery_indices] * self.oversample_machinery
        
        # Guardar lista original para referencia
        self.original_file_list = self.file_list.copy()
        
        # Agregar samples extra a la lista
        self.file_list = self.file_list + extra_samples
        
        # Mezclar para distribuir uniformemente
        import random
        random.shuffle(self.file_list)
        
        print(f"🔁 Runtime Oversampling V3 aplicado:")
        print(f"   Bloques MACHINERY: {len(machinery_indices)}")
        print(f"   Factor: {self.oversample_machinery + 1}x (1 original + {self.oversample_machinery} copias)")
        print(f"   Bloques totales: {len(self.original_file_list)} → {len(self.file_list)}")

    def __len__(self):
        return len(self.file_list)

    def augment_data(self, xyz, normals):
        if self.aug_config is None: return xyz, normals
        
        # 1. Rotación
        if self.aug_config.get('rotate', False):
            angle = np.random.uniform(0, 2 * np.pi)
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            xyz = np.dot(xyz, R.T)
            normals = np.dot(normals, R.T)

        # 2. Flip
        if self.aug_config.get('flip', False):
            if np.random.random() > 0.5:
                xyz[:, 0] = -xyz[:, 0]
                normals[:, 0] = -normals[:, 0]
            if np.random.random() > 0.5:
                xyz[:, 1] = -xyz[:, 1]
                normals[:, 1] = -normals[:, 1]

        # 3. Escalado
        s_min = self.aug_config.get('scale_min', 0.9)
        s_max = self.aug_config.get('scale_max', 1.1)
        scale = np.random.uniform(s_min, s_max)
        xyz = xyz * scale

        # 4. Jitter
        sigma = self.aug_config.get('jitter_sigma', 0.01)
        clip = 0.05
        jitter = np.clip(sigma * np.random.randn(*xyz.shape), -1*clip, clip)
        xyz = xyz + jitter

        # 5. Input Dropout (Nuevo en V2)
        dropout_ratio = self.aug_config.get('input_dropout_ratio', 0.0)
        if dropout_ratio > 0:
            num_drop = int(len(xyz) * dropout_ratio)
            drop_indices = np.random.choice(len(xyz), num_drop, replace=False)
            # Estrategia: Reemplazar puntos eliminados por el primer punto
            # (Mantiene el tamaño del tensor constante sin introducir ceros artificiales)
            xyz[drop_indices] = xyz[0]
            normals[drop_indices] = normals[0]

        return xyz, normals

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        
        try:
            data = np.load(file_path).astype(np.float32)
            
            # Detect columns
            num_cols = data.shape[1]
            
            # Sampling
            N = len(data)
            if N >= self.num_points:
                choices = np.random.choice(N, self.num_points, replace=False)
            else:
                choices = np.random.choice(N, self.num_points, replace=True)
            
            data = data[choices, :]
            
            if num_cols == 11: # V4 RGB
                xyz = data[:, 0:3]
                rgb = data[:, 3:6]
                normals = data[:, 6:9]
                verticality = data[:, 9:10]
                labels = data[:, 10].astype(np.int64)
            else: # V3 Legacy
                xyz = data[:, 0:3]
                rgb = None
                normals = data[:, 3:6]
                verticality = data[:, 6:7]
                labels = data[:, 7].astype(np.int64)
            
            if self.split == "train" and self.aug_config is not None:
                xyz, normals = self.augment_data(xyz, normals)
            
            xyz = xyz.astype(np.float32)
            normals = normals.astype(np.float32)
            verticality = verticality.astype(np.float32)
            
            if rgb is not None:
                rgb = rgb.astype(np.float32)
                features = np.hstack([xyz, rgb, normals, verticality])
            else:
                features = np.hstack([xyz, normals, verticality])
            
            return (
                torch.from_numpy(xyz).float(),  # torch.float32 [N, 3]
                torch.from_numpy(features).float(),  # torch.float32 [N, 7]
                torch.from_numpy(labels).long()  # torch.int64 [N]
            )
            
        except Exception as e:
            print(f"Error cargando {file_path}: {e}")
            # Retornar datos dummy en caso de error
            return (
                torch.zeros((self.num_points, 3), dtype=torch.float32),
                torch.zeros((self.num_points, 6), dtype=torch.float32),
                torch.zeros(self.num_points, dtype=torch.int64)
            )
