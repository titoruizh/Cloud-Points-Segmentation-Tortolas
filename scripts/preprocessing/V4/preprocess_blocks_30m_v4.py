import os
import subprocess

# Configuración V4 (RandLANet RGB)
BLOCK_SIZE = 30.0
NORMAL_RADIUS = 2.0 
OUTPUT_NAME = "blocks_30m V4" 

# Balance Standard
EASY_RATIO = 0.5 
HARD_RATIO = 0.8

cmd = [
    "python3", "scripts/preprocessing/V4/preprocess_blocks_v4.py",
    "--raw-dir", "data/raw RGB",
    "--output", OUTPUT_NAME,
    "--block-size", str(BLOCK_SIZE),
    "--normal-radius", str(NORMAL_RADIUS),
    "--hard-negative-ratio", str(HARD_RATIO),
    "--easy-negative-ratio", str(EASY_RATIO),
    "--min-points", "5000",
    "--max-files", "9999"
]

print(f"🚀 Lanzando Preprocesamiento V4: {OUTPUT_NAME} (RGB Mode)")
subprocess.run(cmd)
