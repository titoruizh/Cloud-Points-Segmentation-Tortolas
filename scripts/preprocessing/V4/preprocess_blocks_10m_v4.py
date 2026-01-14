import os
import subprocess

# Configuración V4 (PointNet++ RGB)
BLOCK_SIZE = 10.0
NORMAL_RADIUS = 1.0 # 1.0 para detalle fino
OUTPUT_NAME = "blocks_10m V4" # Nombre carpeta output

# Balance V3/V4 (Ratio 1.5 para Easy Negatives)
EASY_RATIO = 1.5 
HARD_RATIO = 1.0

cmd = [
    "python3", "scripts/preprocessing/V4/preprocess_blocks_v4.py",
    "--raw-dir", "data/raw RGB",
    "--output", OUTPUT_NAME,
    "--block-size", str(BLOCK_SIZE),
    "--normal-radius", str(NORMAL_RADIUS),
    "--hard-negative-ratio", str(HARD_RATIO),
    "--easy-negative-ratio", str(EASY_RATIO),
    "--min-points", "1000",
    "--max-files", "9999"
]

print(f"🚀 Lanzando Preprocesamiento V4: {OUTPUT_NAME} (RGB Mode)")
subprocess.run(cmd)
