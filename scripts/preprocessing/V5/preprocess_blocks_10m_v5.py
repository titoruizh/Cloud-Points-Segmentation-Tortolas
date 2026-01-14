import os
import subprocess

# Configuración V5 (Ablation: No Verticality)
BLOCK_SIZE = 10.0
NORMAL_RADIUS = 1.0 
OUTPUT_NAME = "blocks_10m V5" # Output destination

# Balance V3/V4 (Mantener consistencia)
EASY_RATIO = 1.5 
HARD_RATIO = 1.0

cmd = [
    "python3", "scripts/preprocessing/V5/preprocess_blocks_v5.py",
    "--raw-dir", "data/raw RGB",
    "--output", OUTPUT_NAME,
    "--block-size", str(BLOCK_SIZE),
    "--normal-radius", str(NORMAL_RADIUS),
    "--hard-negative-ratio", str(HARD_RATIO),
    "--easy-negative-ratio", str(EASY_RATIO),
    "--min-points", "1000",
    "--max-files", "9999"
]

print(f"🚀 Lanzando Preprocesamiento V5: {OUTPUT_NAME} (No Verticality)")
subprocess.run(cmd)
