import torch
import torch.nn as nn
from src.models.randlanet_blocks import LocalFeatureAggregation, SharedMLP

class RandLANet(nn.Module):
    def __init__(self, d_in, num_classes):
        super().__init__()
        
        # --- ENCODER ---
        # Capa de entrada: 6 canales (XYZ + Normales) -> 8 canales
        self.fc0 = nn.Conv1d(d_in, 8, kernel_size=1)
        
        # Bloques LFA (Bajada)
        # Reducimos puntos a la mitad (aprox) en cada paso o simplemente extraemos features
        # En RandLA puro se hace Random Sampling explícito.
        # Aquí usaremos MaxPool para simular la reducción de dimensionalidad espacial si fuera imagen,
        # pero en nubes de puntos el sampling se hace externamente o por índices.
        # Para simplificar tu implementación inicial y que CORRA RÁPIDO:
        # Mantendremos N puntos pero aumentaremos la profundidad de features.
        
        self.lfa1 = LocalFeatureAggregation(8, 32, num_neighbors=16)   # 8 -> 32
        self.lfa2 = LocalFeatureAggregation(32, 64, num_neighbors=16)  # 32 -> 64
        self.lfa3 = LocalFeatureAggregation(64, 128, num_neighbors=16) # 64 -> 128
        self.lfa4 = LocalFeatureAggregation(128, 256, num_neighbors=16)# 128 -> 256
        
        # --- DECODER ---
        # Upsampling y concatenación (Skip connections simuladas)
        self.up4 = SharedMLP(256, 128)
        self.up3 = SharedMLP(256, 64)
        self.up2 = SharedMLP(128, 32)
        self.up1 = SharedMLP(64, 8)
        
        # Capas finales
        self.final_mlp = nn.Sequential(
            nn.Conv1d(16, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv1d(64, 32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, num_classes, kernel_size=1)
        )

    def forward(self, xyz, features):
        """
        xyz: [B, N, 3]
        features: [B, N, C]
        """
        # Ajustar dimensiones para Conv1d: [B, C, N]
        features = features.permute(0, 2, 1)
        
        # Entrada
        x0 = self.fc0(features) # [B, 8, N]
        
        # Encoder
        # Nota: En RandLA real se reduce N. Aquí mantenemos N constante por simplicidad
        # y confiamos en el LFA para capturar contexto.
        # Esto es más parecido a una PointNet++ densa pero con el bloque LFA eficiente.
        
        x1 = self.lfa1(xyz, x0) # [B, 32, N]
        x2 = self.lfa2(xyz, x1) # [B, 64, N]
        x3 = self.lfa3(xyz, x2) # [B, 128, N]
        x4 = self.lfa4(xyz, x3) # [B, 256, N]
        
        # Decoder (Simulado para N constante)
        # Si hiciéramos sampling, aquí habría que interpolar.
        # Al mantener N, simplemente pasamos por MLPs y concatenamos (U-Net style).
        
        d4 = self.up4(x4.unsqueeze(-1)).squeeze(-1) # [B, 128, N]
        d4 = torch.cat([d4, x3], dim=1) # Skip connection
        
        d3 = self.up3(d4.unsqueeze(-1)).squeeze(-1) # [B, 64, N]
        d3 = torch.cat([d3, x2], dim=1)
        
        d2 = self.up2(d3.unsqueeze(-1)).squeeze(-1) # [B, 32, N]
        d2 = torch.cat([d2, x1], dim=1)
        
        d1 = self.up1(d2.unsqueeze(-1)).squeeze(-1) # [B, 8, N]
        d1 = torch.cat([d1, x0], dim=1)
        
        # Clasificador
        out = self.final_mlp(d1) # [B, num_classes, N]
        
        # Devolver a [B, N, Puntos]
        return out

if __name__ == '__main__':
    # Test rápido de dimensiones
    sim_xyz = torch.rand(2, 4096, 3)
    sim_feat = torch.rand(2, 4096, 6) # XYZ + Normales
    model = RandLANet(d_in=6, num_classes=2)
    output = model(sim_xyz, sim_feat)
    print("Output shape:", output.shape) # Debería ser [2, 4096, 2]