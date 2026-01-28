import torch
import numpy as np
import glob
import os
from torch.utils.data import Dataset

class MiningDataset(Dataset):
    def __init__(self, data_dir, num_points=4096, split='train', aug_config=None, 
                 use_filtered=False, filtered_file=None, oversample_machinery=0,
                 device='cuda'):
        """
        Mining Dataset V6 con Data Augmentation en GPU.
        
        Args:
            device: 'cuda' para GPU augmentation, 'cpu' para legacy mode
        """
        self.data_dir = data_dir
        self.num_points = num_points
        self.split = split
        self.aug_config = aug_config
        self.oversample_machinery = oversample_machinery
        self.device = device  # GPU por defecto
        
        if use_filtered and filtered_file:
            filtered_path = os.path.join(data_dir, filtered_file)
            if os.path.exists(filtered_path):
                with open(filtered_path, 'r') as f:
                    self.file_list = [line.strip() for line in f if line.strip()]
                print(f"✅ Dataset V6 {split}: {len(self.file_list)} bloques FILTRADOS")
            else:
                self.file_list = glob.glob(os.path.join(data_dir, "*.npy"))
        else:
            self.file_list = glob.glob(os.path.join(data_dir, "*.npy"))
            print(f"✅ Dataset V6 {split}: {len(self.file_list)} bloques (sin filtro)")
        
        if oversample_machinery > 0 and split == 'train':
            self._apply_runtime_oversampling()

    def _apply_runtime_oversampling(self):
        machinery_indices = [
            i for i, filepath in enumerate(self.file_list)
            if 'MACHINERY' in os.path.basename(filepath)
        ]
        if len(machinery_indices) == 0: return
        extra_samples = [self.file_list[i] for i in machinery_indices] * self.oversample_machinery
        self.file_list = self.file_list + extra_samples
        import random
        random.shuffle(self.file_list)
        print(f"🔁 Runtime Oversampling V6: {len(self.file_list)} bloques total (Factor {self.oversample_machinery + 1}x)")

    def __len__(self): return len(self.file_list)

    def augment_data_gpu(self, xyz_tensor, normals_tensor):
        """
        Data Augmentation COMPLETAMENTE en GPU usando PyTorch.
        Compatible con RTX 5090 / CUDA 12.8.
        Elimina transferencias CPU↔GPU durante entrenamiento.
        
        Args:
            xyz_tensor: torch.Tensor [N, 3] en GPU
            normals_tensor: torch.Tensor [N, 3] en GPU
        
        Returns:
            xyz_aug, normals_aug: Tensores aumentados (permanecen en GPU)
        """
        if self.aug_config is None:
            return xyz_tensor, normals_tensor
        
        device = xyz_tensor.device
        dtype = xyz_tensor.dtype
        
        # ===== ROTACIÓN EN Z =====
        if self.aug_config.get('rotate', False):
            angle = torch.rand(1, device=device, dtype=dtype) * 2 * np.pi
            c, s = torch.cos(angle), torch.sin(angle)
            
            # Matriz de rotación 3x3
            R = torch.zeros(3, 3, device=device, dtype=dtype)
            R[0, 0] = c
            R[0, 1] = -s
            R[1, 0] = s
            R[1, 1] = c
            R[2, 2] = 1.0
            
            # Aplicar rotación (matmul en GPU)
            xyz_tensor = xyz_tensor @ R.T
            normals_tensor = normals_tensor @ R.T
        
        # ===== FLIP ALEATORIO =====
        if self.aug_config.get('flip', False):
            if torch.rand(1, device=device) > 0.5:
                xyz_tensor[:, 0] *= -1
                normals_tensor[:, 0] *= -1
            if torch.rand(1, device=device) > 0.5:
                xyz_tensor[:, 1] *= -1
                normals_tensor[:, 1] *= -1
        
        # ===== SCALE ALEATORIO =====
        s_min = self.aug_config.get('scale_min', 0.9)
        s_max = self.aug_config.get('scale_max', 1.1)
        scale = torch.empty(1, device=device, dtype=dtype).uniform_(s_min, s_max)
        xyz_tensor = xyz_tensor * scale
        
        # ===== JITTER =====
        sigma = self.aug_config.get('jitter_sigma', 0.01)
        clip = 0.05
        
        jitter = torch.randn_like(xyz_tensor, device=device) * sigma
        jitter = torch.clamp(jitter, -clip, clip)
        xyz_tensor = xyz_tensor + jitter
        
        return xyz_tensor, normals_tensor

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        try:
            data = np.load(file_path).astype(np.float32)
            num_cols = data.shape[1]
            N = len(data)
            
            # Sampling Strategy for V6 (0.25m density)
            if N >= self.num_points: 
                choices = np.random.choice(N, self.num_points, replace=False)
            else: 
                choices = np.random.choice(N, self.num_points, replace=True)
            
            data = data[choices, :]
            
            # V6 Format: [X Y Z R G B Nx Ny Nz Label] (10 cols) same as V5
            if num_cols == 10:
                xyz = data[:, 0:3]
                rgb = data[:, 3:6]
                normals = data[:, 6:9]
                labels = data[:, 9].astype(np.int64)
            else:
                # Fallback genérico
                xyz = data[:, 0:3]
                rgb = data[:, 3:6]
                normals = data[:, 6:9]
                labels = data[:, -1].astype(np.int64)

            # Convertir a tensores de PyTorch EN GPU directamente
            xyz_tensor = torch.from_numpy(xyz.astype(np.float32)).to(self.device)
            normals_tensor = torch.from_numpy(normals.astype(np.float32)).to(self.device)
            rgb_tensor = torch.from_numpy(rgb.astype(np.float32)).to(self.device)
            labels_tensor = torch.from_numpy(labels).long().to(self.device)
            
            # ===== AUGMENTATION EN GPU =====
            if self.split == "train" and self.aug_config is not None:
                xyz_tensor, normals_tensor = self.augment_data_gpu(xyz_tensor, normals_tensor)
            
            # Concatenar features [xyz, rgb, normals] = 9 canales
            features = torch.cat([xyz_tensor, rgb_tensor, normals_tensor], dim=1)
            
            # Devolver todo en GPU (el DataLoader ya NO necesita pin_memory)
            return xyz_tensor, features, labels_tensor
            
        except Exception as e:
            print(f"Error {file_path}: {e}")
            # Fallback en GPU
            return (
                torch.zeros((self.num_points, 3), device=self.device),
                torch.zeros((self.num_points, 9), device=self.device),
                torch.zeros(self.num_points, dtype=torch.long, device=self.device)
            )
