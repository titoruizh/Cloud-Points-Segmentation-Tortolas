#!/usr/bin/env python3
"""
Wrapper para Preprocesamiento V3 (10m) para PointNet++
Llama a preprocess_blocks.py con los parámetros corregidos (Easy Negatives activados).
"""

import os
import subprocess

def main():
    # Configuración PointNet++ (V2 Balanced)
    BLOCK_SIZE = 10.0  
    OUTPUT_NAME = "blocks_10m_v3_balanced" # Nombre nuevo para diferenciar
    
    # MEJORA 1: Ratios más agresivos para combatir Falsos Positivos
    # Antes: 0.5. Ahora: 1.5 (Más suelo que máquinas = Realismo)
    EASY_RATIO = 1.5  
    HARD_RATIO = 1.0  # Igualar cantidad de rocas y máquinas
    
    # MEJORA 2: Radio más fino para bloques pequeños
    # Antes: 2.0m (Muy borroso). Ahora: 1.0m (Alta definición gracias a densidad 95pts/m2)
    NORMAL_RADIUS = 1.0 

    cmd = [
        "python3", "scripts/preprocessing/V3/preprocess_blocks.py",
        "--raw-dir", "data/raw",
        "--output", OUTPUT_NAME,
        "--block-size", str(BLOCK_SIZE),
        "--normal-radius", str(NORMAL_RADIUS), 
        "--min-points", "1000",   # Mínimo PointNet
        "--hard-negative-ratio", str(HARD_RATIO),
        "--easy-negative-ratio", str(EASY_RATIO)
    ]
    
    print(f"🚀 Lanzando Preprocesamiento V3 PRO (10m)...")
    print(f"   📐 Config: Radio={NORMAL_RADIUS}m (Definición Alta)")
    print(f"   ⚖️  Balance: EasyRatio={EASY_RATIO} (Combate Alucinaciones)")
    
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
