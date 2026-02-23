# 🔬 Guía de Sweep W&B - MiniPointNet RTX 5090

## 📋 ¿Qué es un Sweep?

Un **Sweep** de Weights & Biases automatiza la búsqueda de los mejores hiperparámetros para tu modelo. 

En lugar de entrenar manualmente con diferentes configuraciones, el sweep:
- ✅ Prueba múltiples combinaciones automáticamente
- ✅ Usa optimización Bayesiana (aprende de intentos anteriores)
- ✅ Guarda solo los mejores modelos
- ✅ Nombra cada modelo según sus hiperparámetros: `LR0.0027_W28_J0.016`

## 🚀 Inicio Rápido

### Paso 1: Iniciar el Sweep
```bash
./start_sweep_minipointnet.sh
```

Esto creará el sweep en W&B y te dará un ID como:
```
tito-ruiz-haros/Point-Cloud-Research/abc123de
```

### Paso 2: Ejecutar Agentes

El script te preguntará si quieres iniciar el agente automáticamente. Si dices que sí, comenzará a entrenar modelos.

**O manualmente:**
```bash
wandb agent tito-ruiz-haros/Point-Cloud-Research/abc123de
```

### Paso 3: Monitorear en W&B

Abre tu navegador en:
```
https://wandb.ai/tito-ruiz-haros/Point-Cloud-Research/sweeps
```

Verás en tiempo real:
- 📊 Gráficos de IoU vs hiperparámetros
- 🏆 El mejor modelo encontrado
- 📈 Evolución del sweep

## ⚙️ Configuración del Sweep

Archivo: `configs/sweeps/sweep_minipointnet_rtx5090.yaml`

### Hiperparámetros que se optimizan:

| Parámetro | Rango | Descripción |
|-----------|-------|-------------|
| `learning_rate` | 0.0008 - 0.005 | Tasa de aprendizaje |
| `weight_maq` | 20 - 40 | Peso de clase Maquinaria |
| `jitter_sigma` | 0.005 - 0.015 | Ruido de augmentación |

### Parámetros fijos:
- **Épocas**: 50 (más rápido para sweep)
- **Arquitectura**: MiniPointNet
- **Dataset**: blocks_10m (Dataset 3)

## 📁 Organización de Resultados

Los modelos se guardan automáticamente en:
```
checkpoints/
└── SWEEP_RTX5090_MiniPointNet_D3_R1/
    ├── LR0.0027_W28_J0.016_BEST_IOU.pth
    ├── LR0.0027_W28_J0.016_BEST_LOSS.pth
    ├── LR0.0012_W35_J0.008_BEST_IOU.pth
    ├── LR0.0012_W35_J0.008_BEST_LOSS.pth
    └── ...
```

**Nomenclatura:**
- `LR0.0027` = Learning Rate de 0.0027
- `W28` = Weight de Maquinaria = 28
- `J0.016` = Jitter Sigma de 0.016

Cada configuración guarda 2 modelos:
- `*_BEST_IOU.pth` → Mejor IoU de Maquinaria (🎯 tu prioridad)
- `*_BEST_LOSS.pth` → Mejor pérdida de validación

## 🎛️ Ejecutar Múltiples Agentes en Paralelo

Tu RTX 5090 es un misil, pero con sweeps es mejor ir de uno en uno para no saturar:

```bash
# Terminal 1
wandb agent tito-ruiz-haros/Point-Cloud-Research/abc123de
```

Si tienes suficiente RAM y quieres acelerar (opcional):
```bash
# Terminal 2 (solo si tienes >100GB RAM libres)
wandb agent tito-ruiz-haros/Point-Cloud-Research/abc123de
```

## 🛑 Detener el Sweep

**Detener agente actual:**
```bash
Ctrl + C
```

**Detener el sweep completo en W&B:**
1. Ve a la página del sweep
2. Click en "Stop Sweep"

## 📊 Interpretar Resultados

### En W&B verás:

**1. Parallel Coordinates Plot**
- Líneas de colores mostrando cada run
- Las líneas que llegan más arriba en `IoU_Maquinaria` son las mejores

**2. Importance Plot**
- Qué hiperparámetro tiene más impacto
- Ayuda a entender qué optimizar primero

**3. Table View**
- Tabla con todos los runs ordenados
- Ordena por `IoU_Maquinaria` para ver el mejor

### Mejor Modelo

El sweep guardará el mejor encontrado. Para usarlo en inferencia:

```bash
# Encuentra el mejor modelo en la carpeta
ls -lh checkpoints/SWEEP_RTX5090_MiniPointNet_D3_R1/*_BEST_IOU.pth

# Usa el que tenga mejor IoU (verás en W&B cuál fue)
python3 inference.py \
  --input data/raw_test/MP_acotado.las \
  --model checkpoints/SWEEP_RTX5090_MiniPointNet_D3_R1/LR0.0027_W28_J0.016_BEST_IOU.pth \
  --architecture MiniPointNet
```

## 🔧 Modificar Configuración del Sweep

Edita: `configs/sweeps/sweep_minipointnet_rtx5090.yaml`

### Cambiar rangos de búsqueda:
```yaml
learning_rate:
  min: 0.001  # Mínimo
  max: 0.01   # Máximo
```

### Agregar nuevo hiperparámetro:
```yaml
batch_size:
  values: [128, 256, 512]  # Prueba estos valores
```

Después de editar, vuelve a ejecutar:
```bash
./start_sweep_minipointnet.sh
```

## 💡 Tips Avanzados

### 1. Early Termination
El sweep ya tiene configurado Hyperband que detiene runs malos después de 10 épocas. Esto ahorra tiempo.

### 2. Cambiar Método de Optimización
```yaml
method: random  # Búsqueda aleatoria (más simple)
method: grid    # Búsqueda exhaustiva (más lento)
method: bayes   # Bayesiano (más inteligente) ← Actual
```

### 3. Más Épocas para el Mejor
Una vez encuentres el mejor, entrénalo manualmente con 100 épocas:

```bash
# Edita rtx5090_beast.yaml con los mejores hiperparámetros
# Luego entrena normal
python3 train_2.py --config configs/minipointnet/rtx5090_beast.yaml
```

## 🎯 Estrategia Recomendada

1. **Fase 1: Exploración Rápida** (Este sweep)
   - 50 épocas por run
   - Optimización Bayesiana
   - Encuentra región prometedora

2. **Fase 2: Refinamiento** (Opcional)
   - Crea nuevo sweep con rangos más estrechos
   - Alrededor de los mejores valores encontrados

3. **Fase 3: Entrenamiento Final**
   - Toma el MEJOR hiperparámetro set
   - Entrena con 100-150 épocas
   - Usa ese modelo en producción

## 📞 Troubleshooting

**"wandb: command not found"**
```bash
pip install wandb
wandb login
```

**"Sweep keeps failing"**
- Revisa que el config YAML esté bien
- Verifica que `train_2.py` exista
- Checa los logs en W&B

**"Out of memory"**
- Reduce `batch_size` en el config base
- No ejecutes múltiples agentes en paralelo

**"Too many runs"**
- Detén el sweep en W&B
- Borra runs malos desde la interfaz web

## 📈 Métricas que se Trackean

Para cada run el sweep guarda:
- ✅ `IoU_Maquinaria` (objetivo principal)
- ✅ `IoU_Suelo`
- ✅ `mIoU` (promedio)
- ✅ `val_loss`
- ✅ `accuracy`
- ✅ `learning_rate` actual
- ✅ Curvas de entrenamiento por época

## 🏆 Ejemplo de Resultados Esperados

Después de ~10-15 runs podrías ver algo como:

| Run | LR | Weight | Jitter | IoU_Maq | mIoU |
|-----|-----|--------|--------|---------|------|
| 🥇 LR0.0027_W28_J0.012 | 0.0027 | 28 | 0.012 | **88.5%** | 93.2% |
| 🥈 LR0.0015_W32_J0.008 | 0.0015 | 32 | 0.008 | 87.3% | 92.8% |
| 🥉 LR0.0031_W25_J0.014 | 0.0031 | 25 | 0.014 | 86.8% | 92.5% |

El 🥇 es tu modelo ganador!
