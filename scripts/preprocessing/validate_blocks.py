#!/usr/bin/env python3
"""
Script de Validación de Bloques Preprocesados
Verifica la calidad y balance de los bloques generados.
"""

import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

def validate_blocks(blocks_dir):
    """
    Valida bloques .npy y genera reporte.
    
    Returns:
        dict: Estadísticas de validación
    """
    files = glob.glob(os.path.join(blocks_dir, "*.npy"))
    
    if len(files) == 0:
        print(f"❌ No se encontraron archivos .npy en {blocks_dir}")
        return None
    
    print(f"📂 Validando {len(files)} bloques...")
    
    stats = {
        'total_blocks': len(files),
        'machinery_blocks': 0,
        'hard_negative_blocks': 0,
        'easy_negative_blocks': 0,
        'total_points': 0,
        'machinery_points': 0,
        'ground_points': 0,
        'invalid_blocks': 0,
        'shape_errors': [],
        'nan_blocks': [],
        'coord_ranges': {'x': [], 'y': [], 'z': []},
        'normal_ranges': {'nx': [], 'ny': [], 'nz': []},
        'verticality_values': [],
        'points_per_block': [],
    }
    
    for filepath in tqdm(files):
        basename = os.path.basename(filepath)
        
        # Contar por tipo
        if basename.startswith('MACHINERY'):
            stats['machinery_blocks'] += 1
        elif basename.startswith('HARD_NEGATIVE'):
            stats['hard_negative_blocks'] += 1
        elif basename.startswith('EASY_NEGATIVE'):
            stats['easy_negative_blocks'] += 1
        
        try:
            data = np.load(filepath)
            
            # Verificar shape
            if data.ndim != 2 or data.shape[1] != 8:
                stats['shape_errors'].append(
                    f"{basename}: shape={data.shape} (esperado: (N, 8))"
                )
                stats['invalid_blocks'] += 1
                continue
            
            # Verificar NaN/Inf
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                stats['nan_blocks'].append(basename)
                stats['invalid_blocks'] += 1
                continue
            
            # Extraer componentes
            xyz = data[:, :3]
            normals = data[:, 3:6]
            verticality = data[:, 6]
            labels = data[:, 7].astype(int)
            
            # Estadísticas de puntos
            n_points = len(data)
            n_machinery = np.sum(labels == 1)
            n_ground = np.sum(labels == 0)
            
            stats['total_points'] += n_points
            stats['machinery_points'] += n_machinery
            stats['ground_points'] += n_ground
            stats['points_per_block'].append(n_points)
            
            # Rangos de coordenadas (deben estar normalizadas)
            stats['coord_ranges']['x'].append((xyz[:, 0].min(), xyz[:, 0].max()))
            stats['coord_ranges']['y'].append((xyz[:, 1].min(), xyz[:, 1].max()))
            stats['coord_ranges']['z'].append((xyz[:, 2].min(), xyz[:, 2].max()))
            
            # Rangos de normales (deben estar en [-1, 1])
            stats['normal_ranges']['nx'].append((normals[:, 0].min(), normals[:, 0].max()))
            stats['normal_ranges']['ny'].append((normals[:, 1].min(), normals[:, 1].max()))
            stats['normal_ranges']['nz'].append((normals[:, 2].min(), normals[:, 2].max()))
            
            # Verticalidad (debe estar en [0, 1])
            stats['verticality_values'].extend(verticality)
            
        except Exception as e:
            stats['invalid_blocks'] += 1
            print(f"❌ Error en {basename}: {e}")
    
    return stats


def print_report(stats):
    """Imprime reporte de validación."""
    print("\n" + "=" * 80)
    print("📊 REPORTE DE VALIDACIÓN")
    print("=" * 80)
    
    # Resumen de bloques
    print(f"\n📦 Bloques Totales: {stats['total_blocks']}")
    print(f"   🚜 MACHINERY:      {stats['machinery_blocks']:4d} ({stats['machinery_blocks']/stats['total_blocks']*100:.1f}%)")
    print(f"   ⛰️  HARD_NEGATIVE:  {stats['hard_negative_blocks']:4d} ({stats['hard_negative_blocks']/stats['total_blocks']*100:.1f}%)")
    print(f"   🟤 EASY_NEGATIVE:  {stats['easy_negative_blocks']:4d} ({stats['easy_negative_blocks']/stats['total_blocks']*100:.1f}%)")
    
    # Resumen de puntos
    print(f"\n📊 Puntos Totales: {stats['total_points']:,}")
    print(f"   🚜 Maquinaria: {stats['machinery_points']:,} ({stats['machinery_points']/stats['total_points']*100:.2f}%)")
    print(f"   🟤 Suelo:      {stats['ground_points']:,} ({stats['ground_points']/stats['total_points']*100:.2f}%)")
    
    # Estadísticas de puntos por bloque
    ppb = np.array(stats['points_per_block'])
    print(f"\n📏 Puntos por Bloque:")
    print(f"   Min:    {ppb.min():,}")
    print(f"   Max:    {ppb.max():,}")
    print(f"   Media:  {ppb.mean():.0f}")
    print(f"   Mediana: {np.median(ppb):.0f}")
    
    # Validación de rangos
    print(f"\n✅ Validación de Rangos:")
    
    # Coordenadas
    x_ranges = np.array(stats['coord_ranges']['x'])
    y_ranges = np.array(stats['coord_ranges']['y'])
    z_ranges = np.array(stats['coord_ranges']['z'])
    
    print(f"   X: [{x_ranges[:, 0].min():.2f}, {x_ranges[:, 1].max():.2f}]")
    print(f"   Y: [{y_ranges[:, 0].min():.2f}, {y_ranges[:, 1].max():.2f}]")
    print(f"   Z: [{z_ranges[:, 0].min():.2f}, {z_ranges[:, 1].max():.2f}]")
    
    # Normales
    nx_ranges = np.array(stats['normal_ranges']['nx'])
    ny_ranges = np.array(stats['normal_ranges']['ny'])
    nz_ranges = np.array(stats['normal_ranges']['nz'])
    
    print(f"   Nx: [{nx_ranges[:, 0].min():.3f}, {nx_ranges[:, 1].max():.3f}]")
    print(f"   Ny: [{ny_ranges[:, 0].min():.3f}, {ny_ranges[:, 1].max():.3f}]")
    print(f"   Nz: [{nz_ranges[:, 0].min():.3f}, {nz_ranges[:, 1].max():.3f}]")
    
    # Verticalidad
    vert = np.array(stats['verticality_values'])
    print(f"   Verticalidad: [{vert.min():.3f}, {vert.max():.3f}] (media: {vert.mean():.3f})")
    
    # Errores
    if stats['invalid_blocks'] > 0:
        print(f"\n⚠️ Bloques Inválidos: {stats['invalid_blocks']}")
        if stats['shape_errors']:
            print(f"   Errores de shape: {len(stats['shape_errors'])}")
            for err in stats['shape_errors'][:5]:
                print(f"      {err}")
        if stats['nan_blocks']:
            print(f"   Bloques con NaN/Inf: {len(stats['nan_blocks'])}")
            for name in stats['nan_blocks'][:5]:
                print(f"      {name}")
    else:
        print(f"\n✅ Todos los bloques son válidos")
    
    print("=" * 80)


def create_visualizations(stats, output_path):
    """Crea visualizaciones del dataset."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Distribución de tipos de bloques
    ax = axes[0, 0]
    types = ['MACHINERY', 'HARD_NEG', 'EASY_NEG']
    counts = [
        stats['machinery_blocks'],
        stats['hard_negative_blocks'],
        stats['easy_negative_blocks']
    ]
    colors = ['#FF3030', '#FFA500', '#A0A0A0']
    ax.bar(types, counts, color=colors, alpha=0.7)
    ax.set_title('Distribución de Tipos de Bloques', fontweight='bold')
    ax.set_ylabel('Cantidad')
    ax.grid(axis='y', alpha=0.3)
    
    # 2. Distribución de puntos por clase
    ax = axes[0, 1]
    classes = ['Maquinaria', 'Suelo']
    points = [stats['machinery_points'], stats['ground_points']]
    ax.bar(classes, points, color=['#FF3030', '#A0A0A0'], alpha=0.7)
    ax.set_title('Distribución de Puntos por Clase', fontweight='bold')
    ax.set_ylabel('Cantidad de Puntos')
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Histograma de puntos por bloque
    ax = axes[1, 0]
    ax.hist(stats['points_per_block'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_title('Distribución de Puntos por Bloque', fontweight='bold')
    ax.set_xlabel('Número de Puntos')
    ax.set_ylabel('Frecuencia')
    ax.grid(axis='y', alpha=0.3)
    
    # 4. Histograma de verticalidad
    ax = axes[1, 1]
    ax.hist(stats['verticality_values'], bins=50, color='green', alpha=0.7, edgecolor='black')
    ax.set_title('Distribución de Verticalidad', fontweight='bold')
    ax.set_xlabel('Verticalidad (0=Plano, 1=Vertical)')
    ax.set_ylabel('Frecuencia')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualización guardada en: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Validación de bloques preprocesados'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Directorio con bloques .npy (ej: data/processed/blocks_10m)')
    parser.add_argument('--output', type=str, default=None,
                        help='Ruta para guardar visualización (opcional)')
    
    args = parser.parse_args()
    
    # Validar
    stats = validate_blocks(args.input)
    
    if stats is None:
        return
    
    # Reporte
    print_report(stats)
    
    # Visualización
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(args.input, "validation_report.png")
    
    create_visualizations(stats, output_path)


if __name__ == "__main__":
    main()
