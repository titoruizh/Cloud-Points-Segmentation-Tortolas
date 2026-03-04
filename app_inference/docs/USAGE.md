# Guía de Uso - Point Cloud Inference App V5

## 🚀 Inicio Rápido

### 1. Ejecutar la Aplicación

```bash
# Desde la raíz del proyecto
cd /workspaces/Cloud-Point-Research\ V2\ Docker\ C\ /

# Iniciar la aplicación
python3 main_inference_app.py
```

La aplicación se abrirá en: **http://localhost:7860**

### 2. Interfaz Principal

La interfaz está dividida en dos paneles:

#### Panel Izquierdo (Configuración)
- **📁 Archivos de Entrada**: Arrastra o selecciona archivos LAZ/LAS
- **🧠 Modelo**: Selecciona el checkpoint a usar
- **💾 Salida**: Define el directorio de salida
- **🔧 Parámetros**: Ajusta batch size, postprocesamiento, etc.

#### Panel Derecho (Resultados)
- **Log en tiempo real**: Muestra el progreso del procesamiento
- **Estadísticas**: Resumen de archivos procesados

---

## 📋 Flujo de Trabajo Típico

### Paso 1: Cargar Archivos

1. Haz clic en "📁 Archivos de Entrada"
2. Selecciona uno o más archivos `.laz` o `.las`
3. **Importante**: Los archivos DEBEN tener canales RGB

### Paso 2: Validar (Opcional pero Recomendado)

1. Haz clic en "🔍 Validar Archivos"
2. El log mostrará:
   - ✅ Archivos válidos (con RGB)
   - ❌ Archivos inválidos (sin RGB, muy pocos puntos, etc.)
   - ⚠️ Advertencias (sin normales pre-calculadas)

### Paso 3: Configurar

#### Modelo
- Por defecto usa: `SWEEP_RTX 5090 PointNet2 V5 NoVert/LR0.0010_W20_J0.005_R3.5_BEST_IOU.pth`
- Este es el modelo campeón del V5

#### Parámetros de Inferencia
- **Batch Size**: 64 (óptimo para RTX 5090)
- **torch.compile**: Activado para máxima velocidad

#### Postprocesamiento
- **FIX_TECHO**: Rellena techos de maquinaria omitidos
- **INTERPOL**: Genera DTM limpio (bulldozer digital)

### Paso 4: Ejecutar

1. Haz clic en "🚀 Ejecutar Pipeline"
2. Observa el progreso en el log
3. Al finalizar, los archivos estarán en `data/predictions/app_v5/`

---

## 📊 Archivos de Salida

Por cada archivo de entrada `NOMBRE.laz`, se generan:

| Archivo | Descripción |
|---------|-------------|
| `NOMBRE_inferido.laz` | Clasificación directa del modelo |
| `NOMBRE_techos.laz` | Con techos rellenados (FIX_TECHO) |
| `NOMBRE_DTM.laz` | DTM final limpio (INTERPOL) |

### Clasificaciones

| Código | Clase | Descripción |
|--------|-------|-------------|
| 1 | Maquinaria | Camiones, excavadoras, etc. |
| 2 | Suelo | Terreno natural/modificado |

---

## ⚙️ Parámetros Avanzados

### FIX_TECHO

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| EPS | 2.5m | Radio de clustering DBSCAN |
| Z Buffer | 1.5m | Altura mínima sobre el suelo |
| Max Height | 8.0m | Altura máxima de maquinaria |
| Padding | 1.5m | Margen XY para búsqueda |

### INTERPOL

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| K Vecinos | 12 | Puntos para interpolación IDW |
| Max Dist | 50m | Radio máximo de búsqueda |

---

## 🔧 Solución de Problemas

### "El archivo NO tiene canales RGB"
- El modelo V5 requiere RGB para funcionar
- Usa datos de fotogrametría o LiDAR con color
- Los archivos sin RGB serán omitidos automáticamente

### "CUDA out of memory"
- Reduce el Batch Size (ej: 32 en lugar de 64)
- Desactiva torch.compile
- Cierra otras aplicaciones que usen GPU

### "El modelo tarda en iniciar"
- `torch.compile` añade 30-60 segundos en el primer batch
- Esto es normal y mejora la velocidad total
- Para evitarlo, desactiva "Usar torch.compile"

### "No se detectaron normales"
- Es solo una advertencia, no un error
- Las normales se calcularán con Open3D
- Esto toma más tiempo pero funciona correctamente

---

## 📈 Rendimiento Esperado

### RTX 5090 (Batch 64)

| Puntos | Tiempo Aprox. |
|--------|---------------|
| 1M | ~10s |
| 10M | ~60s |
| 50M | ~5min |
| 100M | ~10min |

*Tiempos aproximados incluyendo postprocesamiento*

---

## 🛠️ Modo Desarrollo

### Ejecutar sin verificaciones

```bash
python3 main_inference_app.py --no-check
```

### Puerto personalizado

```bash
python3 main_inference_app.py --port 8080
```

### Link público (para demos)

```bash
python3 main_inference_app.py --share
```

---

## 📝 Logs

Los logs de cada ejecución se guardan en:
```
app_inference/outputs/logs/inference_YYYYMMDD_HHMMSS.log
```

Esto permite trazabilidad completa de cada procesamiento.
