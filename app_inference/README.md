# 🚀 Point Cloud Inference App V5

**Aplicación de Inferencia de Nubes de Puntos con PointNet++ V5**

---

## 📋 Descripción

Esta aplicación proporciona una interfaz web moderna y elegante para realizar inferencia de nubes de puntos LiDAR/Fotogrametría utilizando el modelo PointNet++ V5 "Geometric Purification" entrenado para segmentación de maquinaria minera.

### ✨ Características Principales

- 🎨 **Interfaz Web Moderna**: UI intuitiva construida con Gradio
- ⚡ **Alto Rendimiento**: Optimizada para RTX 5090 con FP16 y torch.compile
- 🔄 **Pipeline Completo**: Inferencia + FIX_TECHO + INTERPOL (Bulldozer DTM)
- 📁 **Procesamiento por Lotes**: Soporte para archivos individuales o carpetas completas
- ✅ **Validación RGB**: Verificación automática de nubes con datos RGB
- 📊 **Trazabilidad**: Logs detallados de todo el proceso

---

## 🛠️ Instalación

### Requisitos Previos

- Python 3.10+
- CUDA 12.0+ con GPU NVIDIA (Probado en RTX 5090)
- Docker (Opcional, ya configurado en el proyecto)

### Dependencias

```bash
pip install gradio laspy numpy torch open3d scikit-learn scipy tqdm
```

O usar el requirements.txt de la raíz del proyecto.

---

## 🚀 Uso Rápido

### Desde la raíz del proyecto:

```bash
python3 main_inference_app.py
```

La aplicación abrirá automáticamente en: `http://localhost:7860`

---

## 📁 Estructura del Proyecto

```
app_inference/
├── README.md                 # Esta documentación
├── config/
│   └── default_config.yaml   # Configuración por defecto
├── core/
│   ├── __init__.py
│   ├── inference_engine.py   # Motor principal de inferencia
│   ├── postprocess.py        # Pipeline de postprocesamiento
│   └── validators.py         # Validadores de archivos
├── ui/
│   ├── __init__.py
│   ├── app.py                # Aplicación Gradio principal
│   ├── components.py         # Componentes UI reutilizables
│   └── styles.py             # Estilos CSS personalizados
├── utils/
│   ├── __init__.py
│   ├── file_utils.py         # Utilidades de archivos
│   └── logging_utils.py      # Sistema de logging
└── outputs/                  # Carpeta de salidas temporales
```

---

## ⚙️ Configuración

### Parámetros de Inferencia

| Parámetro | Valor por Defecto | Descripción |
|-----------|-------------------|-------------|
| `batch_size` | 64 | Tamaño de batch para RTX 5090 |
| `block_size` | 10.0 | Tamaño de bloque en metros |
| `num_points` | 10000 | Puntos por bloque |

### Parámetros de FIX_TECHO

| Parámetro | Valor por Defecto | Descripción |
|-----------|-------------------|-------------|
| `eps` | 2.5 | Radio DBSCAN para clustering |
| `z_buffer` | 1.5 | Altura mínima desde el suelo |
| `max_height` | 8.0 | Altura máxima de maquinaria |
| `padding` | 1.5 | Margen XY para búsqueda de techo |

### Parámetros de INTERPOL (Bulldozer)

| Parámetro | Valor por Defecto | Descripción |
|-----------|-------------------|-------------|
| `k` | 12 | Vecinos para interpolación IDW |
| `max_dist` | 50 | Distancia máxima de búsqueda |

---

## 📊 Pipeline de Procesamiento

```
┌─────────────────┐
│ Nube LAS/LAZ    │ (Entrada con RGB)
│   (Original)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 1. INFERENCIA   │ (PointNet++ V5 - RTX 5090)
│    Nitro 🏎️     │
└────────┬────────┘
         │ (Clasificación Maquinaria vs Suelo)
         ▼
┌─────────────────┐
│ 2. FIX_TECHO    │ (Relleno volumétrico de techos)
│    🏗️           │
└────────┬────────┘
         │ (Corrige techos perdidos)
         ▼
┌─────────────────┐
│ 3. INTERPOL     │ (Bulldozer Digital - IDW)
│    🚜           │
└────────┬────────┘
         │ (DTM limpio)
         ▼
┌─────────────────┐
│ Nube DTM Final  │
│   (Salida)      │
└─────────────────┘
```

---

## 🔍 Validación de Archivos

La aplicación verifica automáticamente:

1. ✅ Formato LAZ/LAS válido
2. ✅ Presencia de canales RGB (obligatorio para V5)
3. ✅ Cantidad mínima de puntos
4. ⚠️ Advertencia si no tiene normales pre-calculadas (se calcularán con Open3D)

---

## 📝 Logs y Trazabilidad

Los logs se guardan en:
- `app_inference/outputs/logs/` - Logs de cada ejecución
- Formato: `inference_YYYYMMDD_HHMMSS.log`

---

## 🔧 Mejoras Futuras

- [ ] Visualización 3D integrada de resultados
- [ ] Exportación a formatos adicionales (PLY, PCD)
- [ ] Generación automática de reportes PDF
- [ ] Métricas de calidad post-inferencia
- [ ] Procesamiento distribuido multi-GPU
- [ ] API REST para integración con otros sistemas

---

## 📄 Licencia

Proyecto interno - Antigravity AI © 2026

---

## 👥 Autores

- **Antigravity AI Team**
- Basado en PointNet++ V5 "Geometric Purification"
