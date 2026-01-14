import os
import subprocess

# Configuración V6 (Resolution Sync: 0.25m)
BLOCK_SIZE = 10.0
# Aumentamos ligeramente el radio normal porque la densidad baja (0.25m vs 0.10m)
NORMAL_RADIUS = 2.0 
OUTPUT_NAME = "blocks_10m V6" # Output destination

# Balance V3/V4 (Mantener consistencia)
EASY_RATIO = 1.5 
HARD_RATIO = 1.0

cmd = [
    "python3", "scripts/preprocessing/V6/preprocess_blocks_v6.py",
    "--raw-dir", "data/raw RGB/0.25m",  # <--- RUTA CLAVE 0.25m
    "--output", OUTPUT_NAME,
    "--block-size", str(BLOCK_SIZE),
    "--normal-radius", str(NORMAL_RADIUS),
    "--hard-negative-ratio", str(HARD_RATIO),
    "--easy-negative-ratio", str(EASY_RATIO),
    "--min-points", "500", # Menor densidad = menos puntos min
    "--max-files", "9999"
]

print(f"🚀 Lanzando Preprocesamiento V6: {OUTPUT_NAME} (0.25m Source)")
subprocess.run(cmd)
