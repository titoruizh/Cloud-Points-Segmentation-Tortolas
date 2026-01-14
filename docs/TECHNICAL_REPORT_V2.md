# Informe Técnico V2: Segmentación High Density & Robustness 🧬
**Versión:** 2.1 (Evolución V2)
**Fecha:** 07 Enero 2026
**Autor:** Antigravity AI & Usuario
**Referencia V1:** `docs/TECHNICAL_REPORT_V1.md` (Base del pipeline)

---

## 1. Motivación del Upgrade V2 🎯
Tras analizar los resultados de la V1, detectamos dos comportamientos opuestos en los modelos derivados de una incorrecta gestión de la densidad de puntos:

1.  **RandLANet "Miope" (Undersampling Severo):**
    - En V1, veía solo 20k puntos de los 250k reales en un bloque de 50m.
    - Perdía el 92% de la información, generando predicciones "fantasmas" y baja confianza en bordes.

2.  **PointNet++ "Obsesivo" (Overfitting):**
    - En V1, veía 8k puntos en 10m (casi resolución nativa).
    - Memorizaba detalles irrelevantes (piedras específicas), fallando al generalizar en nuevas nubes.

---

## 2. Nueva Arquitectura V2 🛠️

### 2.1 RandLANet V2 (Efficiency Spot Strategy) ⚡
*Objetivo: Maximizar densidad sin sacrificar operatividad.*

- **Puntos de Entrada:** Ajustado a **25,000**.
    - *Nota Evolutiva:* 
        - 65k (High Density): Fallo por complejidad $O(N^2)$.
        - 40k (Theoretical Sweet Spot): Lento (>1h por época).
        - **25k (Final):** +25% que V1, balanceando velocidad y detalle.
- **Compensación de Hardware:**
    - `Batch Size`: 4.
    - `Accumulations`: 6.
    - *Resultado:* Entrenamiento fluido y gestión de memoria estable.

### 2.2 PointNet++ V2 (Robust)
*Objetivo: Forzar aprendizaje conceptual.*

- **Puntos de Entrada:** Ajustado a **10,000** (Densidad real 10x10m).
- **Augmentation Agresiva:**
    - `Scale`: **0.80 - 1.20** (vs 0.95-1.05 en V1).
    - `Input Dropout`: **0.20** (Se eliminan 20% de puntos al azar en entrenamiento).
    - *Efecto:* El modelo aprende a reconstruir camiones incompletos.

---

## 3. Configuración de Operaciones V2 ⚙️

### 3.1 Naming Convention
Para evitar mezclar experimentos, todo el pipeline V2 usa sufijos estrictos:
- **Project:** `Tortolas-segmentation`
- **W&B Groups:** `RandLANet_V2`, `Pointnet_V2`
- **Agent:** `RTX 5090 Agent V2`
- **Inferencia Output:** `data/predictions/*_MODEL_V2.laz`

### 3.2 Instrucciones de Entrenamiento (Nightly)
El Agente V2 ya tiene cargadas las configuraciones de High Density.

```bash
# Iniciar Agente V2 High Density
wandb sweep configs/randlanet/agent_sweep.yaml
# Copiar ID y ejecutar:
wandb agent tito-ruiz-haros/Tortolas-segmentation/<SWEEP_ID>
```

### 3.3 Instrucciones de Inferencia V2
Los scripts tienen nuevos defaults (65k para RandLA, 10k para PointNet).

```bash
# Inferencia High Density (Automática)
PYTHONPATH=. python3 scripts/inference/infer_randlanet.py \
  --input_file "data/raw_test/MINA_NUEVA.laz" \
  --checkpoint "checkpoints/SWEEP_RTX 5090 RandLANet V2 HighDensity/BEST_IOU.pth"
```
*Salida:* `data/predictions/MINA_NUEVA_RANDLANET_V2.laz`

---

## 4. Fase 2.2: Hyperparameter Tuning (V2.2) 🎛️
*Objetivo: Maximizar IoU manteniendo la robustez ganada.*

Una vez eliminado el overfitting (Train~Val), buscamos el límite de rendimiento mediante Búsqueda Bayesiana.

**Estrategia de Sweep:**
- **Método:** Bayesiano (Optimizar `iou_maq`).
- **Iteraciones:** Continua (Agente).
- **Parámetros:**
    - `learning_rate`: `0.0001 - 0.005` (Buscando convergencia fina).
    - `class_weights`: `[1.0, 10.0], [1.0, 15.0], [1.0, 20.0]` (Penalización variable).
    - `base_radius`: `2.5m, 3.5m, 5.0m` (Contexto local vs medio).

**Cambios en Entrenamiento:**
- **Épocas:** Aumentadas a **300** (vs 60).
- **Razón:** El modelo "atontado" (Dropout 20%) aprende más lento pero más seguro. Necesita tiempo para capturar patrones sutiles.

**Comando de Lanzamiento (Hyperparam Sweep):**
wandb sweep configs/pointnet2/sweep_hyperparam.yaml
wandb agent tito-ruiz-haros/Tortolas-segmentation/<SWEEP_ID>
```

### 4.2 RandLANet Sweep (V2.2)
*Objetivo: Equilibrar la densidad "Efficiency Spot" (25k).*

- **Parámetros:**
    - `learning_rate`: `0.001 - 0.01`.
    - `class_weights`: `[50.0, 100.0, 150.0]` (Penalización severa por desbalance).
- **Comando:**
```bash
wandb sweep configs/randlanet/sweep_hyperparam.yaml
wandb agent tito-ruiz-haros/Tortolas-segmentation/<SWEEP_ID>
```

---
**Conclusión:** V2 no es solo un re-entrenamiento... (continúa)
