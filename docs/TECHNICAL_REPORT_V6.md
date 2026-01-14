# Informe Técnico V6: "Resolution Sync" (0.25m) 📐

**Versión:** 6.0 (Resolution Match)
**Fecha:** 13 Enero 2026
**Autor:** Antigravity AI & Usuario
**Estado:** 🛠️ Preparando Pipeline

---

## 1. El Problema de la Resolución y la Inferencia V5 ⚠️

El modelo V5 (PointNet++ sin verticalidad) demostró ser muy robusto teóricamente. Sin embargo, en pruebas de producción (datasets mensuales) tuvo un desempeño inferior al esperado.

**Diagnóstico:**
*   **Entrenamiento:** Se usó data a **0.10m** (Sub-sampling agresivo o nubes de alta densidad).
*   **Producción:** Las nubes reales de fotogrametría mensual llegan a **0.25m** (menor densidad).
*   **Consecuencia:** El modelo aprendió patrones de "micro-textura" que no existen en la nube de 0.25m, o la escala de las features geométricas (radios de búsqueda) no es compatible con la densidad real.

---

## 2. Definición Técnica V6 🛠️

V6 no es un cambio de arquitectura del modelo (seguiremos usando PointNet++ MSG sin verticalidad, ya que esa hipótesis fue validada). V6 es una **corrección de datos**.

### 2.1 Estrategia de Datos
*   **Input:** `data/raw RGB/0.25m` (Nubes clasificadas a resolución nativa).
*   **Output:** `data/processed/blocks_10m V6`
*   **Resolución:** 0.25m (Consistente con producción).

### 2.2 Pipeline
Reutilizamos el pipeline robusto de V5:
1.  **Preprocesamiento:** Generación de bloques de 10x10.
2.  **Entrenamiento:** PointNet++ (XYZ + RGB + Normals).

---

## 3. Plan de Acción
1.  Generar Dataset V6 (`blocks_10m V6`).
2.  Entrenar modelo V6 desde cero con los mismos parámetros que el mejor V5.
3.  Validar contra nube mensual (que ahora sí tendrá la misma densidad que el train set).

---

## 4. Configuración de Entrenamiento V6 🔥

Para adaptar el entrenamiento a la nueva densidad (0.25m), hemos realizado un ajuste crítico:

*   **`num_points`: 2048** (Antes 10,000).
    *   *Razón:* Un bloque de 10x10m a 0.25m de resolución tiene teóricamente $40 \times 40 = 1600$ puntos en un plano perfecto.
    *   Mantener 10,000 forzaría un oversampling masivo (repetir cada punto ~6 veces), ralentizando el entrenamiento sin ganar información.
    *   2048 ofrece un margen seguro para zonas con estructuras verticales (máquinas) sin desperdiciar cómputo.

### 4.1 Archivos Creados
*   `src/data/dataset_v6.py`: Loader específico para V6 (routing automático en `TRAIN.py`).
*   `configs/pointnet2/pointnet2_v6_0.25m.yaml`: Configuración base.
*   `configs/pointnet2/sweep_v6_0.25m.yaml`: Sweep para buscar LR y Pesos óptimos.

### 4.2 Hipótesis de Entrenamiento
Esperamos que al entrenar con la **misma densidad** que la inferencia real, el modelo aprenda features geométricas (radios de vecindad) que sean generalizables a los datos de producción, eliminando el "Domain Gap" de resolución.

### 4.3 Confirmación de Arquitectura
*   **Modelo:** PointNet++ MSG (Multi-Scale Grouping).
*   **Canales de Entrada (`d_in`): 9**
    *   3 Coordenadas (Available via XYZ)
    *   3 Colores (RGB)
    *   3 Normales (Nx, Ny, Nz)
    *   *Nota:* Al igual que en V5, la **verticalidad** se usa internamente para minería de negativos difíciles, pero se excluye del input del modelo.
*   **Hiperparámetros de Búsqueda (Sweep):**
    *   `learning_rate`: [0.0005, 0.001]
    *   `class_weights`: [[1.0, 15.0], [1.0, 20.0]]
    *   `base_radius`: 3.5m (Fijo, validado previamente)
