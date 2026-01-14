import torch
import torch.nn as nn
import torch.nn.functional as F

# Intentamos importar la extensión C++ optimizada
try:
    from torch_cluster import knn
    KNN_AVAILABLE = True
    print("✅ ACELERACIÓN C++ (torch_cluster) ACTIVADA para RandLA-Net")
except ImportError:
    KNN_AVAILABLE = False
    print("⚠️ ADVERTENCIA: torch_cluster no encontrado. Usando implementación LENTA (PyTorch nativo).")

class SharedMLP(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, transpose=False, padding_mode='zeros', bn=True, activation_fn=None):
        super().__init__()
        layers = []
        if transpose:
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=0))
        else:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=False))
        
        if bn:
            layers.append(nn.BatchNorm2d(out_channels))
        
        if activation_fn is not None:
            layers.append(activation_fn)
        
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

def square_distance(src, dst):
    """
    Calcula la distancia euclidiana al cuadrado (Fallback Lento).
    Solo se usa si falla la carga de C++ o si estamos en CPU.
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Recoge los puntos dados sus índices.
    points: [B, N, C]
    idx: [B, S] o [B, S, K]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

class LocalFeatureAggregation(nn.Module):
    def __init__(self, d_in, d_out, num_neighbors=16, k=16):
        # NOTE: k argument added for compatibility if called as k=... or num_neighbors=...
        # The user's code uses k, but the main model uses num_neighbors.
        # We will map num_neighbors to k to support both or prefer one.
        super(LocalFeatureAggregation, self).__init__()
        self.k = k if k is not None else num_neighbors
        # If the caller sends num_neighbors (like RandLANet does), make sure self.k uses it if k was default.
        if num_neighbors != 16 and k == 16:
             self.k = num_neighbors
        
        self.mlp1 = nn.Conv2d(10, d_out//2, 1)
        self.lse1 = nn.Conv2d(d_out//2, d_out//2, 1)
        self.lse2 = nn.Conv2d(d_out//2, d_out//2, 1)
        self.shortcut = nn.Conv1d(d_in, d_out, 1)

    def forward(self, xyz, feature):
        """
        xyz: [B, 3, N] or [B, N, 3] (Coordenadas)
        feature: [B, d_in, N] (Features previos)
        """
        # Auto-detect and fix xyz shape
        if xyz.shape[1] != 3 and xyz.shape[2] == 3:
             # Input is [B, N, 3], convert to [B, 3, N] expected by this block
             xyz = xyz.permute(0, 2, 1)
        
        B, N = xyz.shape[0], xyz.shape[2]
        
        # --- BLOQUE OPTIMIZADO KNN (C++ vs Python) ---
        if KNN_AVAILABLE and xyz.is_cuda:
            # 🚀 MODO TURBO (RTX 5090)
            # xyz viene como [B, 3, N], necesitamos [B, N, 3] para KNN
            xyz_trans = xyz.permute(0, 2, 1).contiguous()
            
            batch_idx_list = []
            # Iteramos por batch (B es pequeño, ~4-8, el loop es despreciable)
            # Esto evita gestionar offsets complejos para batching plano
            for i in range(B):
                # knn(x, y, k) -> busca en x los vecinos de y
                # x=y para auto-búsqueda
                # Retorna [2, N*k]: fila 0 source, fila 1 target
                # NOTA: torch_cluster.knn retorna indices en orden
                edge_index = knn(xyz_trans[i], xyz_trans[i], self.k)
                
                # edge_index[1] son los vecinos. Reshape a [N, K]
                # Aseguramos que sea LongTensor
                neighbors = edge_index[1].reshape(N, self.k).long()
                batch_idx_list.append(neighbors)
            
            idx = torch.stack(batch_idx_list, dim=0) # [B, N, K]
            
        else:
            # 🐢 MODO TORTUGA (OOM RISK)
            # [B, 3, N] -> [B, N, 3]
            xyz_trans = xyz.permute(0, 2, 1)
            dist = square_distance(xyz_trans, xyz_trans)
            idx = dist.topk(k=self.k, dim=-1, largest=False)[1] # [B, N, K]
        # ----------------------------------------------

        # 2. Local Spatial Encoding
        # xyz: [B, 3, N] -> permute -> [B, N, 3]
        xyz_trans = xyz.permute(0, 2, 1).contiguous()
        
        knn_xyz = index_points(xyz_trans, idx) # [B, N, K, 3]
        
        # Centrar vecinos respecto al punto central
        center_xyz = xyz_trans.unsqueeze(2).repeat(1, 1, self.k, 1) # [B, N, K, 3]
        
        diff_xyz = knn_xyz - center_xyz # [B, N, K, 3]
        dist_xyz = torch.sum(diff_xyz**2, dim=-1, keepdim=True) # [B, N, K, 1]
        
        # Concatenar todo para el MLP
        # [center(3), neighbor(3), diff(3), dist(1)] = 10 canales
        pos_enc = torch.cat((center_xyz, knn_xyz, diff_xyz, dist_xyz), dim=-1) # [B, N, K, 10]
        
        # Ajustar dimensiones para Conv2d: [B, C, H, W] -> [B, 10, N, K]
        pos_enc = pos_enc.permute(0, 3, 1, 2)
        
        # MLP Encoding
        pos_enc = self.mlp1(pos_enc)
        pos_enc = self.lse1(pos_enc)
        pos_enc = self.lse2(pos_enc) # [B, d_out/2, N, K]
        
        # 3. Attentive Pooling
        # Max pooling sobre los K vecinos
        feature_encoded = torch.max(pos_enc, dim=3)[0] # [B, d_out/2, N]
        
        # Shortcut connection
        shortcut = self.shortcut(feature) # [B, d_out, N]
        
        # Combinar
        # duplicamos canales para sumar con shortcut d_out
        # (feature_encoded es d_out/2)
        return torch.cat((feature_encoded, feature_encoded), dim=1) + shortcut

class DilatedResidualBlock(nn.Module):
    def __init__(self, d_in, d_out, k=16):
        super(DilatedResidualBlock, self).__init__()
        self.k = k
        self.lfa1 = LocalFeatureAggregation(d_in, d_out//2, k=k)
        self.lfa2 = LocalFeatureAggregation(d_out//2, d_out, k=k)
        
        self.shortcut = nn.Conv1d(d_in, d_out, 1) if d_in != d_out else nn.Identity()

    def forward(self, xyz, feature):
        x1 = self.lfa1(xyz, feature) # [B, d_out/2, N]
        x2 = self.lfa2(xyz, x1)      # [B, d_out, N]
        
        if isinstance(self.shortcut, nn.Identity):
            return x2 + feature
        else:
            return x2 + self.shortcut(feature)