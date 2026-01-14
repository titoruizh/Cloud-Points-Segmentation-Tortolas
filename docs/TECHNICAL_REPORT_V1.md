# Informe Técnico Maestro: Segmentación de Maquinaria Minera 🏗️
**Versión:** 2.0 (Final Producción)
**Fecha:** 06 Enero 2026
**Autor:** Antigravity & Equipo de Desarrollo

---

## 1. Resumen Ejecutivo 🚀
Este documento detalla la arquitectura, ingeniería de datos y estrategias de Deep Learning implementadas para la segmentación semántica de maquinaria en entornos mineros a cielo abierto.

**Logros Clave:**
- **Precisión:** PointNet++ v4 alcanzó un **96.8% IoU** en la clase Maquinaria.
- **Velocidad:** Inferencia optimizada (KDTree) procesa **12 Millones de puntos en 6 minutos** (RTX 5090).
- **Escalabilidad:** Pipeline dual para detalle fino (10m) y contexto global (50m).

---

## 2. Ingeniería de Datos (Preprocessing) 🛠️

### 2.1 Pipeline de Transformación
El flujo convierte nubes de puntos crudas (`.laz`) en tensores listos para entrenamiento (`.npy`).

1.  **Limpieza:** Eliminación de duplicados y puntos no finitos (`NaN`/`Inf`).
2.  **Cálculo de Features:**
    *   **Normales:** Estimación con radio `r=2.5m`.
    *   **Orientación:** Forzada estrictamente hacia `+Z` `[0,0,1]` para consistencia en taludes.
    *   **Verticalidad:** Feature sintético `abs(Nz)` para distinguir muros de suelos planos.
3.  **Formato de Tensor:** `[X, Y, Z, R, G, B, Verticality]` -> `d_in=7`.

### 2.2 Estrategia de Bloques y Balance de Clases ⚖️
El desafío principal es el desbalance extremo de clases. Se diseñaron dos datasets específicos:

#### A. Dataset "Detail" (PointNet++)
Diseñado para capturar la geometría fina de la maquinaria.
- **Tamaño de Bloque:** 10m x 10m.
- **Balance Nativo:** Maquinaria ~1% del área total.
- **Estrategia de Filtrado:**
    - **MACHINERY:** Se guardan todos los bloques con >3% de maquinaria.
    - **HARD NEGATIVES (Ratio 0.8):** Bloques de suelo complejos (taludes verticales) para reducir falsos positivos.
    - **EASY NEGATIVES:** Suelo plano descartado masivamente.
- **Balance Final (Training):** **~5.8% Maquinaria** / 94.2% Suelo.

#### B. Dataset "Context" (RandLANet)
Diseñado para entender el entorno amplio y reducir falsos positivos globales.
- **Tamaño de Bloque:** 50m x 50m.
- **Balance Nativo:** Maquinaria **~0.6%** (Extremadamente desbalanceado).
- **Estrategia de Filtrado:** Similar al anterior, pero incluye más contexto de suelo.
- **Oversampling en Runtime:** Se inyectan **5 copias** de cada bloque de maquinaria por época para equilibrar artificialmente.

---

## 3. Modelos y Entrenamiento (Deep Learning) 🧠

### 3.1 PointNet++ (V4 Optimized)
Modelo de extracción de características locales mediante `Set Abstraction` (MSG).
- **Objetivo:** Precisión geométrica en bordes de maquinaria.
- **Configuración:**
    - `Batch Size`: 32.
    - `Class Weights`: `[1.0, 15.0]` (Penalización moderada).
    - `Learning Rate`: 0.0005 (Fino).
- **Métricas:** ~96% IoU Maquinaria.

### 3.2 RandLANet (V4 Optimized) - *Entrenamiento Activo*
Modelo eficiente en memoria para grandes nubes de puntos (Random Sampling + Local Feature Aggregation).
- **Objetivo:** Contexto global.
- **Configuración:**
    - `Batch Size`: 24 (Efectivo con Accumulation Steps).
    - `Class Weights`: `[1.0, 100.0]` (Penalización severa por "aguja en pajar").
    - `Oversampling`: x5 dinámico.
    - `Learning Rate`: 0.005 (Agresivo).

### 3.3 Operaciones: W&B Agent (Nightly) 🌙
Para robustez en entrenamientos largos, se utiliza un Agente W&B.
- **Beneficio:** Recuperación ante fallos, monitoreo remoto y gestión de colas.
- **Comando:** `wandb agent <SWEEP_ID>`

---

## 4. Pipeline de Inferencia (Production Grade) ⚡

Se desarrolló un motor de inferencia unificado y altamente optimizado.

### 4.1 Tecnologías Clave
1.  **KDTree Segmentation ($O(1)$):** Reemplazo de filtrado booleano por búsqueda espacial indexada.
    - *Impacto:* Segmentación de 12M puntos bajó de **25 min** a **<20 seg**.
2.  **Sliding Window Robusta:**
    - Superposición del **50-75%** (Stride 2.5m/25m).
    - **Voting System:** Acumulación probabilística (Softmax) para eliminar bordes de bloque.
3.  **Fresh LAS Headers:** Reconstrucción total del encabezado LAS para evitar corrupciones de `laspy`.

### 4.2 Scripts Disponibles

#### Inferencia PointNet++ (Rápida y Precisa)
```bash
PYTHONPATH=. python3 scripts/inference/infer_pointnet.py \
  --input_file "data/raw/ARCHIVO.laz" \
  --checkpoint "checkpoints/RTX 5090 PointNet2 V4 Optimized_BEST_IOU.pth" \
  --batch_size 32
```

#### Inferencia RandLANet (Contexto Masivo)
```bash
PYTHONPATH=. python3 scripts/inference/infer_randlanet.py \
  --input_file "data/raw/ARCHIVO.laz" \
  --checkpoint "checkpoints/RTX 5090 RandLANet V4 Optimized_BEST_IOU.pth" \
  --batch_size 12 --conf_threshold 0.60
```

---

## 5. Próximos Pasos 🔮
1.  **Validación Cruzada:** Ejecutar inferencia RandLANet sobre el set de validación (Epoch 100).
2.  **Ensemble (Fusión):** Crear un script `ensemble.py` que combine:
    - *Geometría* de PointNet++.
    - *Contexto* de RandLANet.
    - `Final_Prob = 0.7 * P_PointNet + 0.3 * P_RandLA`.

---
**Antigravity AI - Google Deepmind**
