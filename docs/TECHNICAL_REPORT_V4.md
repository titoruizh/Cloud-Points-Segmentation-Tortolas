# Informe Técnico V4: "Colors of the Earth" (RGB Integration) 🎨
**Versión:** 4.0 (Data-Centric AI - Color)
**Fecha:** 10 Enero 2026
**Autor:** Antigravity AI & Usuario
**Estado:** 🏗️ Generación de Datos V4 (En curso)

---

## 1. Motivación V4: El Poder del Color 🌈
Hasta la V3, nuestros modelos (PointNet++ y RandLANet) eran **daltónicos**. Solo veían geometría (XYZ, Normales, Verticalidad).
Sin embargo, en minería, el color es un discriminador fundamental:
*   **Maquinaria:** Amarillo/Naranja/Blanco brillante (Caterpillar/Komatsu).
*   **Terreno:** Marrón/Gris (Tierra, Rocas).

La V4 integra esta "dimensión perdida" para facilitar la segmentación donde la geometría es ambigua (ej: una roca con forma de camión, pero de color marrón).

---

## 2. Definición Técnica V4 🛠️

### 2.1 Nueva Dimensionalidad de Entrada
Aumentamos los canales de entrada (`d_in`) de 7 a 10.

| Canal | Descripción | Notas |
| :--- | :--- | :--- |
| **0-2** | X, Y, Z | Coordenadas geométricas (Normalizadas por bloque) |
| **3-5** | **R, G, B** | **Colores** (Normalizados [0.0 - 1.0]) |
| **6-8** | Nx, Ny, Nz | Normales de superficie |
| **9** | Verticalidad | 1.0 = Muro, 0.0 = Plano |

**Configuración de Modelos:**
*   **PointNet++:** `d_in: 10`
*   **RandLANet:** `d_in: 10`

### 2.2 Estrategia de Preprocesamiento (`scripts/preprocessing/V4/`)
Hemos actualizado el pipeline ETL para ingerir archivos `.las` con información de color.

#### A. Normalización de Color
Los archivos LAS suelen guardar color en 16-bit (0-65535).
*   **Transformación:** `RGB_norm = RGB_raw / 65535.0`
*   **Fallback:** Si un archivo no tiene color, se rellena con gris neutro (0.5) para no romper el modelo.

#### B. Datasets Generados
1.  **PointNet++ V4 (10m):**
    *   **Folder:** `data/processed/blocks_10m V4`
    *   **Config:** Balanced Ratio (1.5 Easy Negatives), Radius 1.0m.
    *   **Uso:** Inferencia de alta precisión local.

2.  **RandLANet V4 (30m):**
    *   **Folder:** `data/processed/blocks_30m V4`
    *   **Config:** High Density (65k pts), Radius 2.0m.
    *   **Uso:** Inferencia masiva eficiente.

---

## 3. Hoja de Ruta V4 🗺️

1.  **Generación de Datos:** Ejecutar scripts V4 en `data/raw RGB`. ⏳
2.  **Entrenamiento PointNet++ V4:** 
    *   Nuevo config `d_in: 10`.
    *   Validar si el color reduce los falsos positivos en rocas.
3.  **Entrenamiento RandLANet V4:**
    *   Nuevo config `d_in: 10`.
    *   Probar si el peso de 35.0 sigue siendo el límite o si el color estabiliza el gradiente.

---

## 4. Estado de Preparación PointNet++ V4 (Ready to Train) ✅
*   **Separación de Arquitectura (Clean Separation):**
    *   **V3 Loader (`dataset_v3.py`):** REVERTIDO a su estado original (8 comunas hardcoded). Se mantiene estricto para trazabilidad.
    *   **V4 Loader (`dataset_v4.py`):** NUEVO módulo que soporta `d_in: 10` (RGB).
    *   **TRAIN.py:** Actualizado con "Dynamic Import". Si el config pide `d_in: 10`, carga automáticamente V4. Si no, usa V3.
*   **Configuración:** `configs/pointnet2/pointnet2_v4_rgb.yaml` creada con:
    *   `d_in: 10`
    *   `base_radius`: 3.5m (Baseline V3)
    *   `path`: `data/processed/blocks_10m V4`
*   **Sweep:** `configs/pointnet2/sweep_v4_rgb.yaml` listo para explorar si el color afecta el radio óptimo (`[2.5, 3.5, 4.5]`).

El sistema está listo para iniciar el entrenamiento de PointNet++ V4 en cuanto termine el preprocesamiento.

## 5. Resultados PointNet++ V4 (RGB) 🏆
**Estado:** Completado (Sweep `8808q860`).

### 5.1. Impacto del RGB
La integración del color ha sido **transformadora**. Hemos pasado de un techo de ~64% IoU en V3 a **>83% IoU** en V4.

| Métrica | V3 (Geometría Pura) | V4 (RGB + Geometría) | Mejora |
| :--- | :---: | :---: | :---: |
| **IoU Maquinaria** | ~64.0% | **83.64%** | **+19.6%** 🚀 |
| **mIoU Global** | ~80.0% | **88.52%** | +8.5% |
| **Accuracy** | ~96.0% | **98.93%** | +2.9% |

### 5.2. Hiperparámetros Ganadores (`LR0.0010_W20..._BEST_IOU.pth`)
El sweep confirmó que los parámetros de V3 seguían siendo sólidos, pero el peso de clase `20.0` funcionó mejor con la información extra del color.

*   **Learning Rate:** `0.001`
*   **Class Weights:** `[1.0, 20.0]`
*   **Base Radius:** `3.5m` (El color no cambió la escala geométrica óptima)

> [!NOTE]
> El modelo es capaz de distinguir maquinaria oxidada/amarilla del entorno rocoso con una precisión sin precedentes en este proyecto.

---

## 6. Preparación RandLANet V4 (RGB) 🚧
*   **Objetivo:** Replicar el éxito del RGB en la arquitectura densa (RandLANet).
*   **Configuración:** `d_in: 10`.
*   **Estrategia:** Usar los pesos `35.0` (Límite estable V3) como punto de partida, pero explorar si el RGB permite ser más agresivo sin "paranoia".

### 6.1 Resultados Preliminares & Desafíos (RandLANet V4) 📉
A diferencia de PointNet++, la arquitectura RandLANet V4 (RGB + High Density) ha presentado dificultades significativas en los primeros experimentos (`SWEEP_RTX 5090 RandLANet V4 RGB`).

**Tabla de Resultados:**
| Configuración | Best IoU Maq | Accuracy | Estado | Notas |
| :--- | :---: | :---: | :---: | :--- |
| **LR 0.0002 / W25** | **38.95%** | 90.1% | Crashed | Mejor resultado, pero muy lejos de PointNet++ (83%). |
| **LR 0.0010 / W50** | **11.92%** | 85.8% | Crashed | **Colapso del Modelo.** Pesos altos desestabilizan el gradiente. |

**Conclusiones V4 (RandLANet):**
1.  **Brecha de Rendimiento:** Existe un gap masivo entre PointNet++ (83% IoU) y RandLANet (39% IoU) en esta versión.
2.  **Sensibilidad a Pesos:** Confirmamos la hipótesis de V3: RandLANet es extremadamente sensible a `class_weights > 20`. El intento de subir a 50 provocó una degradación total (11% IoU).
3.  **Próximos Pasos:** Se requiere una revisión profunda de la arquitectura o estrategia de muestreo para RandLANet antes de continuar. Por ahora, **PointNet++ V4 es el campeón indiscutible.**

---

## 7. Post-Procesamiento y Generación de DTM (Workflow V4) 🚜➡️🏔️

Para convertir la segmentación en un producto topográfico final (Curvas de Nivel), hemos implementado un pipeline de post-procesamiento robusto que elimina la maquinaria y restaura el terreno original.

### 7.1 Reparación de Techos (`FIX_TECHO.py`)
Modelos como PointNet++ a veces detectan la base del camión pero fallan en el techo debido a la similitud geométrica con el suelo plano, creando "camiones descapotables".

*   **Solución:** Un script de **Releno Volumétrico** que detecta la base de la maquinaria y proyecta una búsqueda hacia arriba (hasta 8m).
*   **Innovación V4:**
    *   **Proyección Cilíndrica (2D Shape):** En lugar de una caja rectangular (que falla en diagonales), usamos `cKDTree` para verificar que los puntos a rellenar estén dentro de la silueta 2D real del camión.
    *   **Base Robusta:** Usa el percentil 5 de altura para ignorar ruido subterráneo.

```bash
python3 scripts/postprocessing/FIX_TECHO.py \
  --input "data/predictions/V4/2DEM_MP_251230_PINTADA_V4_2_LR20_W25.laz" \
  --output "data/predictions/V4/2DEM_MP_251230_PINTADA_V4_2_LR20_W25_fixed_v4.laz" \
  --eps 2.5 --z_buffer 1.5 --max_height 8.0 --padding 2.0
```

> [!WARNING]
> **Limitación Actual (Work in Progress):** 
> Aunque el script recupera la mayoría de los techos, la limpieza depende de la calidad del clustering inicial (DBSCAN). 
> Si el modelo deja puntos dispersos de maquinaria fuera del cluster principal, `FIX_TECHO` no los "atrapará", y por lo tanto `INTERPOL` no los borrará, dejando pequeños "bultos" o artefactos en el DTM final. 
> **Próximo Paso:** Refinar la agresividad del DBSCAN o implementar un filtro de limpieza por densidad antes de interpolar.

### 7.2 Generación de DTM (`INTERPOL.py`)
Una vez segmentada y reparada la maquinaria, el objetivo es eliminarla para obtener el terreno limpio. Si solo borramos los puntos, quedan "agujeros negros".

#### El Problema
Al borrar un camión (Clase 1), queda un vacío en la nube de puntos que rompe la generación de curvas de nivel, creando artefactos visuales.

#### La Solución (Interpolación IDW)
El script `scripts/postprocessing/INTERPOL.py` realiza una "cirugía" digital:
1.  **Identifica** los agujeros dejados por la maquinaria (Clase 1).
2.  **Busca** vecinos de Suelo (Clase 2) alrededor del perímetro del agujero.
3.  **Calcula** la altura estimada ($Z_{suelo\_estimado}$) usando un promedio ponderado por distancia (IDW) de los vecinos.
4.  **Rellena** el agujero bajando los puntos del techo a la nueva altura del suelo y cambiando su clase a Suelo.

#### El Resultado
El camión desaparece y es reemplazado por una "sábana" de tierra continua que conecta suavemente el terreno de un lado al otro. **Las curvas de nivel pasan rectas y limpias por donde antes había una máquina de 100 toneladas.**

**Comando V4 Ejecutado:**
```bash
python3 scripts/postprocessing/INTERPOL.py \
  --input "data/predictions/V4/2DEM_MP_251230_PINTADA_V4_2_LR20_W25_fixed_v4.laz" \
  --output "data/predictions/V4/DTM_FINAL_CLEAN.laz" \
  --k 12 \
  --max_dist 50
```
> **Nota:** Se usó `k=12` para suavizar la transición. Si el terreno queda rugoso, aumentar `k`.

