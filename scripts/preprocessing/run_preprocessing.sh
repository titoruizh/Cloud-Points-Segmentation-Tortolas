#!/bin/bash
# Script para ejecutar preprocessing en diferentes escalas

set -e  # Exit on error

echo "========================================="
echo "🚀 Preprocessing Multi-Escala"
echo "========================================="

# Configuración
RAW_DIR="data/raw"
MAX_FILES=${1:-3}  # Default: 3 archivos para testing

echo "📁 Directorio raw: $RAW_DIR"
echo "📊 Archivos a procesar: $MAX_FILES"
echo ""

# 1. Bloques de 10m (PointNet2 / MiniPointNet)
echo "========================================="
echo "📐 Generando bloques de 10m..."
echo "========================================="
python3 scripts/preprocessing/preprocess_blocks.py \
    --raw-dir "$RAW_DIR" \
    --output blocks_10m \
    --block-size 10.0 \
    --normal-radius 2.0 \
    --max-files "$MAX_FILES"

echo ""
echo "✅ Bloques de 10m completados"
echo ""

# 2. Bloques de 20m (RandLANet)
echo "========================================="
echo "📐 Generando bloques de 20m..."
echo "========================================="
python3 scripts/preprocessing/preprocess_blocks.py \
    --raw-dir "$RAW_DIR" \
    --output blocks_20m \
    --block-size 20.0 \
    --normal-radius 2.5 \
    --max-files "$MAX_FILES"

echo ""
echo "✅ Bloques de 20m completados"
echo ""

# 3. Validación
echo "========================================="
echo "🔍 Validando bloques generados..."
echo "========================================="

echo ""
echo "Validando blocks_10m..."
python3 scripts/preprocessing/validate_blocks.py \
    --input data/processed/blocks_10m

echo ""
echo "Validando blocks_20m..."
python3 scripts/preprocessing/validate_blocks.py \
    --input data/processed/blocks_20m

echo ""
echo "========================================="
echo "✅ PREPROCESSING COMPLETADO"
echo "========================================="
echo "📁 Bloques generados en:"
echo "   - data/processed/blocks_10m/"
echo "   - data/processed/blocks_20m/"
echo "========================================="
