# Guía de Uso - Script de Inferencia

## 📋 Descripción
Script para clasificar nubes de puntos .las usando modelos entrenados en el proyecto Cloud-Point-Research.

Soporta 3 arquitecturas: **MiniPointNet**, **PointNet++** (PointNet2), y **RandLANet**.

## 🚀 Ejemplos por Arquitectura

### 🔷 MiniPointNet (Rápido y Preciso)

**Uso básico - Sin solapamiento (más rápido):**
```bash
python3 inference.py \
  --input data/raw_test/MP_acotado.las \
  --model checkpoints/RTX5090_MiniPointNet_D3_R1_BEST_IOU.pth \
  --architecture MiniPointNet \
  --block-size 10.0 \
  --stride 10.0
```

**Uso con solapamiento (más preciso en bordes):**
```bash
python3 inference.py \
  --input data/raw_test/MP_acotado.las \
  --model checkpoints/RTX5090_MiniPointNet_D3_R1_BEST_IOU.pth \
  --architecture MiniPointNet \
  --block-size 10.0 \
  --stride 5.0
```

**Salida:** `data/test_results/MiniPointNet/MP_acotado_CLASIFICADO_IA.las`

---

### 🔶 PointNet++ (Mayor Precisión con Votación)

**Configuración recomendada con solapamiento:**
```bash
python3 inference.py \
  --input data/raw_test/MP_acotado.las \
  --model checkpoints/PointNet2_Dataset3_BEST_IOU.pth \
  --architecture PointNet2 \
  --block-size 10.0 \
  --stride 5.0
```

**Configuración ultra-precisa (más lento):**
```bash
python3 inference.py \
  --input data/raw_test/MP_acotado.las \
  --model checkpoints/PointNet2_Dataset3_BEST_IOU.pth \
  --architecture PointNet2 \
  --block-size 15.0 \
  --stride 7.5
```

**Salida:** `data/test_results/PointNet2/MP_acotado_CLASIFICADO_IA.las`

> ⚠️ **Nota PointNet++**: Esta arquitectura se beneficia del solapamiento (stride < block-size). 
> El sistema de **votación** consolidará las predicciones múltiples en cada punto.

---

### 🔴 RandLANet (Para Nubes Grandes)

**Uso estándar:**
```bash
python3 inference.py \
  --input data/raw_test/nube_grande.las \
  --model checkpoints/RandLANet_Best.pth \
  --architecture RandLANet \
  --block-size 20.0 \
  --stride 20.0
```

**Salida:** `data/test_results/RandLANet/nube_grande_CLASIFICADO_IA.las`

---

## 📝 Parámetros Completos

| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `--input` | Archivo .las a clasificar | **Requerido** |
| `--model` | Checkpoint .pth del modelo | **Requerido** |
| `--architecture` | Tipo de modelo (MiniPointNet/RandLANet/PointNet2) | MiniPointNet |
| `--block-size` | Tamaño del bloque en metros | 10.0 |
| `--stride` | Distancia entre bloques | 10.0 |
| `--output-dir` | Carpeta de salida | data/test_results |
| `--num-classes` | Número de clases | 2 |
| `--d-in` | Dimensión entrada (XYZ+Normales) | 6 |
| `--label-offset` | Offset de etiquetas del entrenamiento | 1 |

## 📁 Estructura de Archivos

Las salidas se organizan automáticamente por arquitectura:

```
Cloud-Point-Research/
├── data/
│   ├── raw_test/                    # Archivos .las a clasificar
│   │   └── MP_acotado.las
│   └── test_results/                # Resultados organizados
│       ├── MiniPointNet/            # Salidas de MiniPointNet
│       │   └── MP_acotado_CLASIFICADO_IA.las
│       ├── PointNet2/               # Salidas de PointNet++
│       │   └── MP_acotado_CLASIFICADO_IA.las
│       └── RandLANet/               # Salidas de RandLANet
│           └── MP_acotado_CLASIFICADO_IA.las
├── checkpoints/                     # Modelos entrenados .pth
│   ├── RTX5090_MiniPointNet_D3_R1_BEST_IOU.pth
│   └── PointNet2_Dataset3_BEST_IOU.pth
└── inference.py                     # Script de inferencia
```

## ⚡ Sistema de Votación vs Modo Rápido

El script detecta automáticamente el modo según tus parámetros:

### 🗳️ Modo VOTACIÓN (stride < block-size)
- Cada punto recibe múltiples predicciones
- Se consolida por mayoría de votos
- **Más preciso en bordes**
- Más lento (procesa más bloques)
- **Recomendado para PointNet++**

Ejemplo: `--block-size 10.0 --stride 5.0` → 50% solapamiento

### ⚡ Modo RÁPIDO (stride = block-size)
- Cada punto se clasifica una sola vez
- Sin solapamiento
- **Más rápido**
- Suficiente para MiniPointNet en nubes simples

Ejemplo: `--block-size 10.0 --stride 10.0` → Sin solapamiento

## 🎯 Mapeo de Clases

Durante la inferencia:
- **Clase 1** (LAS) = Maquinaria/Objeto
- **Clase 2** (LAS) = Suelo

Esto es automático basándose en el `label_offset` del entrenamiento.

## 💡 Consejos por Arquitectura

### MiniPointNet
- ✅ Usa `--stride 10.0` (sin solapamiento) para velocidad
- ✅ Usa `--stride 5.0` si hay mucho detalle fino
- ✅ Prefiere checkpoints `*_BEST_IOU.pth` para maquinaria

### PointNet++ (PointNet2)
- 🎯 **Siempre usa solapamiento**: `--stride 5.0` con `--block-size 10.0`
- 🎯 Aumenta `min_points` a 100 para bloques más robustos
- 🎯 Ideal para nubes con geometría compleja
- 🎯 El sistema de votación mejora la precisión final

### RandLANet
- 🔥 Puede manejar `--block-size 20.0` o más
- 🔥 Más eficiente con bloques grandes
- 🔥 Mejor para nubes masivas (millones de puntos)

## 🎛️ Tabla de Configuraciones Recomendadas

| Arquitectura | Block Size | Stride | Velocidad | Precisión |
|--------------|-----------|--------|-----------|-----------|
| MiniPointNet | 10m | 10m | ⚡⚡⚡ Rápido | ✓✓ Buena |
| MiniPointNet | 10m | 5m | ⚡⚡ Normal | ✓✓✓ Excelente |
| PointNet++ | 10m | 5m | ⚡⚡ Normal | ✓✓✓✓ Superior |
| PointNet++ | 15m | 7.5m | ⚡ Lento | ✓✓✓✓✓ Máxima |
| RandLANet | 20m | 20m | ⚡⚡⚡ Muy Rápido | ✓✓✓ Excelente |

## 🐛 Troubleshooting

**Error: "command not found python"**
```bash
# Usa python3 en Linux
python3 inference.py --help
```

**Error: "No module named 'open3d'"**
```bash
# Instala las dependencias
pip install open3d laspy tqdm
```

**Error: CUDA out of memory**
```bash
# Reduce el block-size
python3 inference.py --input ... --model ... --block-size 5.0
```

## 📊 Salida Esperada

### Ejemplo: MiniPointNet (modo rápido)
```
💻 Usando dispositivo: cuda
📁 Carpeta de salida: data/test_results/MiniPointNet
🏗️  Cargando arquitectura: MiniPointNet
📦 Cargando checkpoint: checkpoints/RTX5090_MiniPointNet_D3_R1_BEST_IOU.pth
📂 Leyendo nube: data/raw_test/MP_acotado.las
   Total de puntos: 1,234,567
🔍 Calculando normales de toda la nube...
⚡ Modo RÁPIDO sin solapamiento
🔲 Procesando grid de 225 bloques (15x15)...
Clasificando: 100%|████████████████| 225/225 [02:15<00:00,  1.66it/s]
✅ Bloques procesados: 225/225
💾 Guardando resultado: data/test_results/MiniPointNet/MP_acotado_CLASIFICADO_IA.las

📊 Resultados:
   🚜 Maquinaria: 45,678 puntos (3.70%)
   🟤 Suelo:      1,188,889 puntos (96.30%)

🎉 ¡Terminado! Abre el archivo en CloudCompare para verificar.
```

### Ejemplo: PointNet++ (con votación)
```
💻 Usando dispositivo: cuda
📁 Carpeta de salida: data/test_results/PointNet2
🏗️  Cargando arquitectura: PointNet2
📦 Cargando checkpoint: checkpoints/PointNet2_Dataset3_BEST_IOU.pth
📂 Leyendo nube: data/raw_test/MP_acotado.las
   Total de puntos: 1,234,567
🔍 Calculando normales de toda la nube...
🗳️  Modo VOTACIÓN activado (stride 5.0m < block 10.0m)
🔲 Procesando grid de 841 bloques (29x29)...
Clasificando: 100%|████████████████| 841/841 [08:45<00:00,  1.60it/s]
✅ Bloques procesados: 841/841
🗳️  Consolidando votos...
💾 Guardando resultado: data/test_results/PointNet2/MP_acotado_CLASIFICADO_IA.las

📊 Resultados:
   🚜 Maquinaria: 47,234 puntos (3.83%)
   🟤 Suelo:      1,187,333 puntos (96.17%)

🎉 ¡Terminado! Abre el archivo en CloudCompare para verificar.
```
