import torch
import numpy as np
import glob
import os
from torch.utils.data import Dataset

class MiningDataset(Dataset):
    def __init__(self, data_dir, num_points=4096, split='train', aug_config=None, 
                 use_filtered=False, filtered_file=None, oversample_machinery=0):
        self.data_dir = data_dir
        self.num_points = num_points
        self.split = split
        self.aug_config = aug_config
        self.oversample_machinery = oversample_machinery
        
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

    def augment_data(self, xyz, normals):
        if self.aug_config is None: return xyz, normals
        if self.aug_config.get('rotate', False):
            angle = np.random.uniform(0, 2 * np.pi)
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            xyz = np.dot(xyz, R.T); normals = np.dot(normals, R.T)
        if self.aug_config.get('flip', False):
            if np.random.random() > 0.5: xyz[:, 0] = -xyz[:, 0]; normals[:, 0] = -normals[:, 0]
            if np.random.random() > 0.5: xyz[:, 1] = -xyz[:, 1]; normals[:, 1] = -normals[:, 1]
        s_min = self.aug_config.get('scale_min', 0.9)
        s_max = self.aug_config.get('scale_max', 1.1)
        scale = np.random.uniform(s_min, s_max)
        xyz = xyz * scale
        sigma = self.aug_config.get('jitter_sigma', 0.01)
        clip = 0.05
        jitter = np.clip(sigma * np.random.randn(*xyz.shape), -1*clip, clip)
        xyz = xyz + jitter
        return xyz, normals

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

            if self.split == "train" and self.aug_config is not None:
                xyz, normals = self.augment_data(xyz, normals)
            
            xyz = xyz.astype(np.float32)
            normals = normals.astype(np.float32)
            
            if rgb is not None:
                features = np.hstack([xyz, rgb, normals]) # 9 Channels
            else:
                features = np.hstack([xyz, normals, np.zeros((len(xyz), 1))])

            return torch.from_numpy(xyz).float(), torch.from_numpy(features).float(), torch.from_numpy(labels).long()
        except Exception as e:
            print(f"Error {file_path}: {e}")
            return torch.zeros((self.num_points, 3)), torch.zeros((self.num_points, 9)), torch.zeros(self.num_points, dtype=torch.long)
