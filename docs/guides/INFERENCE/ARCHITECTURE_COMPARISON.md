# 📊 Comparativa de Arquitecturas - Inferencia

## Características por Modelo

### 🔷 MiniPointNet
**Mejor para:** Clasificación rápida de nubes medianas

| Aspecto | Detalles |
|---------|----------|
| **Velocidad** | ⚡⚡⚡ Muy Rápido |
| **Precisión** | ✓✓ Buena - ✓✓✓ Excelente (con solapamiento) |
| **Memoria GPU** | Baja (~2-4 GB) |
| **Configuración Recomendada** | Block: 10m, Stride: 10m (rápido) o 5m (preciso) |
| **Solapamiento** | Opcional |
| **Script** | `./run_inference_mini.sh` |
| **Salida** | `data/test_results/MiniPointNet/` |

**Cuándo usar:**
- Nubes de tamaño medio (< 5 millones de puntos)
- Necesitas resultados rápidos
- Geometría simple o moderada
- Primera prueba de un modelo nuevo

---

### 🔶 PointNet++ (PointNet2)
**Mejor para:** Máxima precisión con geometría compleja

| Aspecto | Detalles |
|---------|----------|
| **Velocidad** | ⚡⚡ Normal - ⚡ Lento |
| **Precisión** | ✓✓✓✓ Superior - ✓✓✓✓✓ Máxima |
| **Memoria GPU** | Media-Alta (~4-8 GB) |
| **Configuración Recomendada** | Block: 10m, Stride: 5m (siempre con solapamiento) |
| **Solapamiento** | **OBLIGATORIO** para mejores resultados |
| **Sistema de Votación** | ✅ Activado automáticamente |
| **Script** | `./run_inference_pointnet2.sh` |
| **Salida** | `data/test_results/PointNet2/` |

**Cuándo usar:**
- Geometría compleja (maquinaria con muchos detalles)
- Necesitas la máxima precisión posible
- Tienes tiempo para procesar
- Dataset crítico (producción final)
- Detectar objetos pequeños o bordes finos

**⚠️ Nota Importante:** PointNet++ **SIEMPRE** debe usar stride < block-size para aprovechar el sistema de votación.

---

### 🔴 RandLANet
**Mejor para:** Nubes masivas y escenas completas

| Aspecto | Detalles |
|---------|----------|
| **Velocidad** | ⚡⚡⚡ Muy Rápido (con bloques grandes) |
| **Precisión** | ✓✓✓ Excelente |
| **Memoria GPU** | Alta (~6-12 GB, pero procesa más puntos) |
| **Configuración Recomendada** | Block: 20m, Stride: 20m |
| **Solapamiento** | No necesario |
| **Salida** | `data/test_results/RandLANet/` |

**Cuándo usar:**
- Nubes enormes (> 10 millones de puntos)
- Escenas completas de minería
- Necesitas procesar múltiples archivos grandes
- Balance entre velocidad y precisión

---

## 🔄 Sistema de Votación Explicado

### Sin Solapamiento (Stride = Block Size)
```
[Bloque 1][Bloque 2][Bloque 3]
```
- Cada punto se clasifica **1 vez**
- Más rápido
- Posibles errores en bordes

### Con Solapamiento (Stride < Block Size)
```
[Bloque 1  ]
   [Bloque 2  ]
      [Bloque 3  ]
```
- Cada punto se clasifica **múltiples veces**
- Se consolida por **mayoría de votos**
- Más lento pero **mucho más preciso en bordes**
- Ideal para PointNet++

**Ejemplo con stride 5m y block 10m:**
- Solapamiento: 50%
- Un punto en el centro puede recibir 4 votos
- Un punto en el borde recibe 1-2 votos
- Se elige la clase más votada

---

## 📐 Tabla de Configuraciones por Caso de Uso

| Caso de Uso | Arquitectura | Block | Stride | Tiempo Estimado* |
|-------------|-------------|-------|--------|------------------|
| Prueba rápida | MiniPointNet | 10m | 10m | 2-3 min |
| Clasificación estándar | MiniPointNet | 10m | 5m | 5-7 min |
| Máxima precisión | PointNet2 | 10m | 5m | 8-12 min |
| Precisión extrema | PointNet2 | 15m | 7.5m | 15-20 min |
| Nube masiva | RandLANet | 20m | 20m | 10-15 min |

*Para ~1.5M puntos en RTX 5090

---

## 🎯 Mapeo de Clases (Todos los Modelos)

Durante inferencia con `--label-offset 1`:

| Predicción Red | Clase LAS | Significado |
|----------------|-----------|-------------|
| 0 | 1 | 🚜 Maquinaria/Objeto |
| 1 | 2 | 🟤 Suelo |

Esto se aplica automáticamente en el script de inferencia.

---

## 💡 Tips Avanzados

### Optimizar Velocidad
```bash
# MiniPointNet sin solapamiento
python3 inference.py --architecture MiniPointNet --stride 10.0 --block-size 10.0
```

### Optimizar Precisión
```bash
# PointNet++ con máximo solapamiento
python3 inference.py --architecture PointNet2 --stride 5.0 --block-size 15.0
```

### Balance Óptimo
```bash
# MiniPointNet con solapamiento moderado
python3 inference.py --architecture MiniPointNet --stride 7.5 --block-size 10.0
```

---

## 🔍 Interpretar Resultados

### Estadísticas en Terminal
```
📊 Resultados:
   🚜 Maquinaria: 45,678 puntos (3.70%)
   🟤 Suelo:      1,188,889 puntos (96.30%)
```

**Interpretación:**
- **< 5% maquinaria**: Típico en zonas de trabajo ya limpiadas
- **5-15% maquinaria**: Zona activa con equipo presente
- **> 15% maquinaria**: Múltiples equipos o estructuras

### Validación en CloudCompare
1. Abrir el archivo `*_CLASIFICADO_IA.las`
2. Color por Clasificación
3. Revisar visualmente:
   - Clase 1 (Maquinaria) debe resaltar equipos
   - Clase 2 (Suelo) debe cubrir el terreno
   - Bordes deben estar limpios (especialmente con PointNet++)

---

## 📞 Troubleshooting Rápido

**"Modo RÁPIDO" cuando esperabas votación:**
- Verifica que `stride < block-size`

**Resultados con mucho ruido:**
- Usa PointNet++ con solapamiento
- Aumenta block-size a 15m

**Muy lento:**
- Reduce block-size
- Usa MiniPointNet sin solapamiento
- Aumenta stride (menos solapamiento)

**Bordes mal clasificados:**
- Usa stride < block-size (50% solapamiento)
- Cambia a PointNet++
