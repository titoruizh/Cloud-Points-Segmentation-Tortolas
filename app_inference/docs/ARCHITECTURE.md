# Arquitectura Técnica - App Inference V5

## 📐 Visión General

```
┌─────────────────────────────────────────────────────────────────────┐
│                     main_inference_app.py                           │
│                    (Punto de entrada CLI)                           │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        app_inference/                               │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────────────┐   │
│  │    core/    │   │     ui/     │   │        utils/           │   │
│  │             │   │             │   │                         │   │
│  │ • Engine    │   │ • app.py    │   │ • file_utils.py         │   │
│  │ • Postproc  │   │ • styles    │   │ • logging_utils.py      │   │
│  │ • Validators│   │ • components│   │                         │   │
│  └──────┬──────┘   └──────┬──────┘   └────────────┬────────────┘   │
│         │                 │                        │                │
└─────────┼─────────────────┼────────────────────────┼────────────────┘
          │                 │                        │
          ▼                 ▼                        ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────────────┐
│  Scripts Base   │ │     Gradio      │ │      Librerías Estándar     │
│  (src/models/)  │ │     (UI Web)    │ │  (laspy, numpy, sklearn)    │
└─────────────────┘ └─────────────────┘ └─────────────────────────────┘
```

---

## 🔧 Módulos

### 1. Core (`app_inference/core/`)

#### `inference_engine.py`
Motor principal de inferencia. Wrapper del script original `infer_pointnet_v5.2.py`.

**Clases:**
- `InferenceConfig`: Dataclass con configuración
- `InferenceResult`: Dataclass con resultados
- `GridDatasetNitro`: Dataset PyTorch optimizado para acceso por bloques
- `InferenceEngine`: Motor de inferencia

**Optimizaciones:**
- FP16 con `torch.amp.autocast`
- `torch.compile` para optimización del grafo
- Lectura directa de normales del LAS
- Gridding vectorizado con NumPy

#### `postprocess.py`
Pipeline de postprocesamiento.

**Clases:**
- `FixTechoConfig`: Configuración FIX_TECHO
- `InterpolConfig`: Configuración INTERPOL
- `PostprocessResult`: Resultado de postproceso
- `PostProcessor`: Ejecutor del pipeline

**Algoritmos:**
- **FIX_TECHO**: DBSCAN clustering + relleno volumétrico
- **INTERPOL**: IDW (Inverse Distance Weighting) para bulldozer digital

#### `validators.py`
Validación de archivos de entrada.

**Verificaciones:**
- Formato LAZ/LAS válido
- Presencia de canales RGB (obligatorio)
- Cantidad mínima de puntos
- Detección de normales pre-calculadas

---

### 2. UI (`app_inference/ui/`)

#### `app.py`
Aplicación Gradio principal.

**Clase `InferenceApp`:**
- Maneja estado de la aplicación
- Coordina validación, inferencia y postproceso
- Gestiona el log de mensajes

**Funciones:**
- `create_app()`: Construye la interfaz Gradio
- `launch_app()`: Lanza el servidor web

#### `styles.py`
CSS personalizado para la interfaz.

**Temas:**
- Colores: Indigo/Purple gradient
- Cards con bordes y sombras
- Status badges animados
- Scrollbar personalizada

#### `components.py`
Componentes UI reutilizables.

**Funciones:**
- `create_header()`: HTML del header
- `create_stats_html()`: Estadísticas de procesamiento
- `create_pipeline_status()`: Estado del pipeline
- `create_validation_report_html()`: Reporte de validación

---

### 3. Utils (`app_inference/utils/`)

#### `file_utils.py`
Utilidades de manejo de archivos.

**Funciones:**
- `find_las_files()`: Buscar archivos LAS/LAZ
- `ensure_dir()`: Crear directorios
- `get_file_info()`: Información de archivo
- `format_file_size()`: Formateo de tamaños

#### `logging_utils.py`
Sistema de logging.

**Clases:**
- `LogCollector`: Buffer de mensajes para UI

**Funciones:**
- `setup_logger()`: Configurar logger estándar
- `get_log_path()`: Generar ruta de log

---

## 🔄 Flujo de Datos

```
┌─────────────────┐
│ Archivo LAZ/LAS │
│   (Entrada)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Validación    │ ← PointCloudValidator
│  (RGB, puntos)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Extracción de   │ ← compute_features_fast()
│    Features     │
│ XYZ+RGB+Normals │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Gridding      │ ← División en bloques 10x10m
│  (Vectorizado)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  DataLoader     │ ← GridDatasetNitro
│  (PyTorch)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  PointNet++ V5  │ ← GPU (FP16 + compile)
│   Inferencia    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   FIX_TECHO     │ ← DBSCAN + Bounding Box
│ (Relleno techos)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    INTERPOL     │ ← IDW + KDTree
│  (Bulldozer)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Archivo LAZ DTM │
│    (Salida)     │
└─────────────────┘
```

---

## 🧠 Modelo PointNet++ V5

### Arquitectura
- **Entrada**: 9 canales (XYZ + RGB + Normals)
- **Sin verticalidad**: Eliminada en V5 ("Geometric Purification")
- **Salida**: 2 clases (Suelo vs Maquinaria)

### Set Abstraction Layers
```
SA1: 1024 puntos, r=0.5*base, [32,32,64]
SA2: 256 puntos, r=1.0*base, [64,64,128]
SA3: 64 puntos, r=2.0*base, [128,128,256]
SA4: 16 puntos, r=4.0*base, [256,256,512]
```

### Feature Propagation
```
FP4: 768 → [256,256]
FP3: 384 → [256,256]
FP2: 320 → [256,128]
FP1: 128+6 → [128,128,128]
```

---

## 📊 Configuración por Defecto

Ver: `app_inference/config/default_config.yaml`

### Inferencia
| Parámetro | Valor | Notas |
|-----------|-------|-------|
| batch_size | 64 | Óptimo RTX 5090 |
| block_size | 10.0m | Igual que entrenamiento |
| num_points | 10000 | Por bloque |
| use_compile | true | +30s inicio, +30% velocidad |

### Postproceso
| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| eps (FIX) | 2.5m | Radio DBSCAN |
| z_buffer | 1.5m | Protección suelo |
| k (INTERPOL) | 12 | Vecinos IDW |

---

## 🔌 Extensibilidad

### Agregar nuevo postproceso

1. Crear método en `PostProcessor`:
```python
def run_mi_postproceso(self, input_file, output_file, ...):
    # Lógica aquí
    pass
```

2. Agregar al pipeline en `run_full_pipeline()`

3. Agregar controles en `ui/app.py`

### Agregar nuevo modelo

1. Colocar checkpoint en `checkpoints/`
2. Modificar `InferenceEngine` si cambia `d_in`
3. El dropdown lo detectará automáticamente

---

## 🧪 Tests

*Pendiente de implementación*

```bash
# Ejecutar tests
pytest app_inference/tests/

# Con coverage
pytest --cov=app_inference app_inference/tests/
```

---

## 📝 Convenciones de Código

- **Docstrings**: Google style
- **Type hints**: En todos los métodos públicos
- **Logging**: Usar LogCollector para UI, logging estándar para archivos
- **Errores**: Capturar y retornar en dataclasses Result
