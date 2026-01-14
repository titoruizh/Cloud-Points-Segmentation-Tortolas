import os
import glob
import numpy as np
from tqdm import tqdm

DATA_DIR = "data/processed/blocks_10m V5"

def analyze_dataset():
    files = glob.glob(os.path.join(DATA_DIR, "*.npy"))
    
    if len(files) == 0:
        print(f"❌ No se encontraron archivos en {DATA_DIR}")
        return

    print(f"📊 Analizando {len(files)} bloques en {DATA_DIR}...")
    
    total_points = 0
    class_counts = {0: 0, 1: 0}
    block_types = {'MACHINERY': 0, 'HARD_NEGATIVE': 0, 'EASY_NEGATIVE': 0}
    
    for f in tqdm(files):
        # Count block types based on filename
        basename = os.path.basename(f)
        if "MACHINERY" in basename:
            block_types['MACHINERY'] += 1
        elif "HARD_NEGATIVE" in basename:
            block_types['HARD_NEGATIVE'] += 1
        elif "EASY_NEGATIVE" in basename:
            block_types['EASY_NEGATIVE'] += 1
            
        # Load data for point stats
        data = np.load(f)
        # Assuming label is the last column (index 9 in V5)
        labels = data[:, -1].astype(int)
        
        unique, counts = np.unique(labels, return_counts=True)
        for u, c in zip(unique, counts):
            if u in class_counts:
                class_counts[u] += c
                
        total_points += len(labels)

    print("\n--- 📈 ESTADÍSTICAS DEL DATASET V5 ---")
    print(f"Total Bloques: {len(files)}")
    print(f"  🚜 Machinery Blocks: {block_types['MACHINERY']}")
    print(f"  ⛰️  Hard Negative:    {block_types['HARD_NEGATIVE']}")
    print(f"  🟤 Easy Negative:    {block_types['EASY_NEGATIVE']}")
    
    print("\n--- 👥 BALANCE DE CLASES (PUNTOS) ---")
    n_suelo = class_counts[0]
    n_maq = class_counts[1]
    
    print(f"Suelo (0):      {n_suelo:,} pts ({(n_suelo/total_points)*100:.2f}%)")
    print(f"Maquinaria (1): {n_maq:,} pts ({(n_maq/total_points)*100:.2f}%)")
    
    if n_maq > 0:
        ratio = n_suelo / n_maq
        print(f"\n⚖️ Ratio Suelo:Maquinaria = {ratio:.2f} : 1")
        print(f"💡 Sugerencia Class Weight: [1.0, {min(ratio/2, 50.0):.1f}] - [1.0, {min(ratio, 80.0):.1f}]")
    else:
        print("\n⚠️ NO SE DETECTÓ MAQUINARIA (Posible error en generación)")

if __name__ == "__main__":
    analyze_dataset()
