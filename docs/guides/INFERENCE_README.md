# 🎯 Inferencia - Guía Rápida

## Scripts Disponibles

### 🔷 MiniPointNet (Rápido)
```bash
./run_inference_mini.sh
```
- Sin solapamiento
- Más rápido
- Salida: `data/test_results/MiniPointNet/`

### 🔶 PointNet++ (Preciso)
```bash
./run_inference_pointnet2.sh
```
- Con solapamiento 50%
- Sistema de votación
- Más preciso en bordes
- Salida: `data/test_results/PointNet2/`

### ⚙️ Comando Manual
```bash
python3 inference.py \
  --input data/raw_test/tu_archivo.las \
  --model checkpoints/tu_modelo.pth \
  --architecture [MiniPointNet|PointNet2|RandLANet] \
  --block-size 10.0 \
  --stride 5.0
```

## 📁 Organización de Resultados

```
data/test_results/
├── MiniPointNet/
│   └── archivo_CLASIFICADO_IA.las
├── PointNet2/
│   └── archivo_CLASIFICADO_IA.las
└── RandLANet/
    └── archivo_CLASIFICADO_IA.las
```

## 🔍 Ver Guía Completa
```bash
cat INFERENCE_GUIDE.md
```

## ⚡ Diferencia Entre Modos

| Modo | Stride | Características |
|------|--------|----------------|
| **Rápido** | stride = block-size | Sin solapamiento, más rápido |
| **Votación** | stride < block-size | Con solapamiento, más preciso |

PointNet++ funciona mejor con **votación** (stride < block-size).
