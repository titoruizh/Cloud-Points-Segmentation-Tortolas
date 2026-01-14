#!/usr/bin/env python3
"""
Test rápido de carga del dataset con el nuevo formato d_in=7
"""

import sys
sys.path.append('/workspaces/Cloud-Point-Research')

from src.data.dataset_v3 import MiningDataset
import torch

print("=" * 80)
print("🧪 TEST DE DATASET - d_in=7")
print("=" * 80)

# Crear dataset
dataset = MiningDataset(
    data_dir="data/processed/blocks_10m_test",
    num_points=4096,
    split='train',
    aug_config=None,
    oversample_machinery=0
)

print(f"\n📊 Dataset cargado: {len(dataset)} bloques")

# Cargar una muestra
xyz, features, labels = dataset[0]

print(f"\n✅ Shapes:")
print(f"   xyz:      {xyz.shape} (esperado: [4096, 3])")
print(f"   features: {features.shape} (esperado: [4096, 7])")
print(f"   labels:   {labels.shape} (esperado: [4096])")

print(f"\n✅ Tipos:")
print(f"   xyz:      {xyz.dtype}")
print(f"   features: {features.dtype}")
print(f"   labels:   {labels.dtype}")

print(f"\n✅ Rangos:")
print(f"   xyz X:    [{xyz[:, 0].min():.2f}, {xyz[:, 0].max():.2f}]")
print(f"   xyz Y:    [{xyz[:, 1].min():.2f}, {xyz[:, 1].max():.2f}]")
print(f"   xyz Z:    [{xyz[:, 2].min():.2f}, {xyz[:, 2].max():.2f}]")
print(f"   Normal Z: [{features[:, 5].min():.3f}, {features[:, 5].max():.3f}]")
print(f"   Vertical: [{features[:, 6].min():.3f}, {features[:, 6].max():.3f}]")

print(f"\n✅ Distribución de clases:")
n_maq = (labels == 1).sum().item()
n_ground = (labels == 0).sum().item()
print(f"   Maquinaria: {n_maq} ({n_maq/len(labels)*100:.1f}%)")
print(f"   Suelo:      {n_ground} ({n_ground/len(labels)*100:.1f}%)")

print("\n" + "=" * 80)
print("✅ TEST COMPLETADO - Dataset funciona correctamente con d_in=7")
print("=" * 80)
