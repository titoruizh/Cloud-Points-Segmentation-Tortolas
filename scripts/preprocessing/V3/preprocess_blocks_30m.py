#!/usr/bin/env python3
"""
Wrapper para Preprocesamiento V3 (30m) para RandLANet
Llama a preprocess_blocks.py con los parámetros óptimos para densidad y cobertura.
"""

import os
import subprocess
import glob

def main():
    # Configuración V3
    BLOCK_SIZE = 30.0  # 30m x 30m (Densidad 72 pts/m2 con 65k)
    OUTPUT_NAME = "blocks_30m"
    
    # Ratios V3 (Vital para balance)
    EASY_RATIO = 0.5  # 50% de bloques fáciles (suelo) para enseñar al modelo
    HARD_RATIO = 0.8  # 80% de bloques difíciles
    
    # Comandos
    cmd = [
        "python3", "scripts/preprocessing/V3/preprocess_blocks.py",
        "--raw-dir", "data/raw",
        "--output", OUTPUT_NAME,
        "--block-size", str(BLOCK_SIZE),
        "--normal-radius", "0.6",  # Reducido proporcionalmente (era 2.0 para 50m?) No, para 10m era 2.0. 
                                   # Espera, si bajamos a 30m para más detalle, el radio normal debería ser fino.
                                   # 30m bloque. Radio 1.0m parece razonable para capturar curvatura de maquinaria.
        "--min-points", "5000",    # Mínimo puntos para considerar bloque útil
        "--hard-negative-ratio", str(HARD_RATIO),
        "--easy-negative-ratio", str(EASY_RATIO)
    ]
    
    print(f"🚀 Lanzando Preprocesamiento V3 (30m)...")
    print(f"   Estrategia: Block={BLOCK_SIZE}m | EasyNeg={EASY_RATIO} (Suelo Activado!)")
    
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
