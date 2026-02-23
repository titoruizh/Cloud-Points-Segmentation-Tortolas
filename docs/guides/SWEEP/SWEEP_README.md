# 🎯 Sistema de Sweep W&B - Resumen Ejecutivo

## ✅ Archivos Creados

### Configuración
- **`configs/sweeps/sweep_minipointnet_rtx5090.yaml`** - Configuración del sweep
  - Optimiza: Learning Rate, Weight Maquinaria, Jitter Sigma
  - Método: Optimización Bayesiana
  - Objetivo: Maximizar IoU de Maquinaria

### Scripts
- **`start_sweep_minipointnet.sh`** - Inicia el sweep automáticamente
- **`check_sweep_setup.sh`** - Verifica que todo esté listo

### Documentación
- **`SWEEP_GUIDE.md`** - Guía completa de uso del sweep

### Modificaciones
- **`train_2.py`** actualizado para:
  - Guardar checkpoints en carpetas organizadas: `checkpoints/SWEEP_[nombre]/`
  - Nombrar modelos automáticamente: `LR0.0027_W28_J0.016`

---

## 🚀 Inicio Rápido (3 Pasos)

### 1. Verificar Setup
```bash
./check_sweep_setup.sh
```

### 2. Login en W&B (si es necesario)
```bash
wandb login
```
Pega tu API key de: https://wandb.ai/authorize

### 3. Iniciar Sweep
```bash
./start_sweep_minipointnet.sh
```

---

## 📁 Organización de Resultados

```
checkpoints/
├── SWEEP_RTX5090_MiniPointNet_D3_R1/
│   ├── LR0.0027_W28_J0.012_BEST_IOU.pth
│   ├── LR0.0027_W28_J0.012_BEST_LOSS.pth
│   ├── LR0.0015_W32_J0.008_BEST_IOU.pth
│   ├── LR0.0015_W32_J0.008_BEST_LOSS.pth
│   └── ... (un par de archivos por cada configuración probada)
```

**Nomenclatura:**
- `LR` = Learning Rate
- `W` = Weight de clase Maquinaria
- `J` = Jitter Sigma

Cada run genera 2 modelos:
- `*_BEST_IOU.pth` → Mejor para detectar maquinaria
- `*_BEST_LOSS.pth` → Mejor pérdida general

---

## 🎛️ Qué hace el Sweep

1. **Prueba automáticamente** diferentes combinaciones de:
   - Learning Rate: 0.0008 a 0.005
   - Peso Maquinaria: 20 a 40
   - Jitter Sigma: 0.005 a 0.015

2. **Aprende** de intentos anteriores (Bayesiano)

3. **Detiene** runs malos temprano (Hyperband)

4. **Guarda** solo los mejores modelos

5. **Nombra** cada modelo según sus hiperparámetros

---

## 📊 Monitoreo

Abre en tu navegador:
```
https://wandb.ai/tito-ruiz-haros/Point-Cloud-Research/sweeps
```

Verás:
- 📈 Gráficos de IoU vs hiperparámetros
- 🏆 Ranking de mejores modelos
- 📉 Evolución del entrenamiento
- 🎯 Importancia de cada hiperparámetro

---

## 🛑 Control del Sweep

### Detener un agente
```bash
Ctrl + C
```

### Detener el sweep completo
1. Ve a la página del sweep en W&B
2. Click en "Stop Sweep"

### Ejecutar múltiples agentes (paralelo)
```bash
# Terminal 1
wandb agent [tu-sweep-id]

# Terminal 2 (opcional, si tienes RAM suficiente)
wandb agent [tu-sweep-id]
```

---

## 🎯 Después del Sweep

### 1. Ver Mejor Modelo
En W&B, ordena por `IoU_Maquinaria` descendente.

### 2. Usar el Mejor para Inferencia
```bash
python3 inference.py \
  --input data/raw_test/MP_acotado.las \
  --model checkpoints/SWEEP_RTX5090_MiniPointNet_D3_R1/LR0.0027_W28_J0.012_BEST_IOU.pth \
  --architecture MiniPointNet
```

### 3. Re-entrenar con Más Épocas (Opcional)
Si quieres exprimir más el mejor:

1. Edita `configs/minipointnet/rtx5090_beast.yaml` con los mejores hiperparámetros
2. Entrena normal:
```bash
python3 train_2.py --config configs/minipointnet/rtx5090_beast.yaml
```

---

## 💡 Tips

✅ **Deja que corra toda la noche** - El sweep encuentra patrones con ~15-20 runs

✅ **Revisa W&B frecuentemente** - Puedes detener el sweep cuando encuentres un gran modelo

✅ **Los primeros runs son exploratorios** - No te desanimes si empiezan mal

✅ **Importance Plot** - Te dice qué hiperparámetro tiene más impacto

✅ **Parallel Coordinates** - Visualiza qué combinaciones funcionan mejor

---

## 📚 Documentación Completa

Para detalles exhaustivos lee:
```bash
cat SWEEP_GUIDE.md
```

---

## 🎉 Resultado Esperado

Después del sweep tendrás:
- 🏆 El mejor set de hiperparámetros encontrado automáticamente
- 📊 Datos de cuál hiperparámetro importa más
- 💾 10-20 modelos organizados por configuración
- 📈 Gráficos completos de rendimiento

**Tu modelo actual:** 86.3% IoU Maquinaria
**Objetivo del sweep:** Superar el 88-90% IoU Maquinaria 🚀
