# 🌩️ Cloud Point Research: Segmentation AI for Large Scale Photogrammetry

> **Estado:** Producción (V6 Resolution Sync)
> **Stack:** PyTorch | PointNet++ | CUDA Optimized
> **Hardware:** NVIDIA RTX 5090

Proyecto de investigación y desarrollo enfocado en la segmentación semántica automática de nubes de puntos fotogramétricas a gran escala (Minería/Obra Civil). El objetivo principal es la clasificación precisa de **Maquinaria Pesada** vs **Terreno** en entornos complejos.

## 🏆 Selección del Modelo: ¿Por qué PointNet++?
Inicialmente exploramos arquitecturas como **RandLa-Net** por su eficiencia en "Large Scale". Sin embargo, nuestras pruebas comparativas demostraron que **PointNet++ (MSG - Multi-Scale Grouping)** ofrece una capacidad superior para capturar detalles geométricos finos en maquinaria compleja (suspensiones, cabinas, brazos hidráulicos) que RandLa-Net tendía a suavizar excesivamente.

Aunque RandLa-Net presentaba ventajas de velocidad teórica, la prioridad de este proyecto es la **precisión crítica en bordes y formas complejas**, donde PointNet++ demostró ser inigualable para nuestra casuística.

---

## 🔬 Evolución Técnica e Innovación

El núcleo del éxito del proyecto reside en dos iteraciones críticas de ingeniería de datos, donde refinamos la interacción entre la geometría 3D y el aprendizaje profundo:

### 📉 V5: Geometric Purification (Verticality Ablation)
En la versión 5, desafiamos la intuición común de usar la "Verticalidad" (Normal Z) como feature explícita.

* **El Problema:** El modelo aprendía atajos falsos ("Pared = Máquina", "Plano = Suelo"), fallando en **techos de contenedores** (planos pero máquinas) o **muros de contención** (verticales pero suelo).
* **La Solución:** Eliminamos la feature de verticalidad del input (`d_in=9` → `XYZ + RGB + Normals`). Forzamos a la red a aprender la **morfología 3D pura** y el contexto geométrico local en lugar de depender de la orientación simple de la normal.
* **Resultado:** Drástica reducción de falsos positivos en estructuras ambiguas (pretiles y contenedores).

### 📐 V6: Resolution Sync (0.25m Production Match)
Detectamos un "Domain Gap" silencioso: entrenábamos con nubes densas (sub-sampling a 0.10m) pero inferíamos en nubes mensuales de producción más ligeras (0.25m).

* **Ajuste:** Recalibramos el pipeline de entrenamiento para operar nativamente a **0.25m**, alineando la distribución de datos de *train* con la realidad de *producción*.
* **Optimización:** Redujimos los puntos de muestreo por bloque de 10,000 a **2,048**. Esto aceleró el entrenamiento y eliminó el ruido generado por buscar "micro-texturas" inexistentes en fotogrametría mensual.

---

### 📸 Comparativa Visual

<p align="center">
  <img src="https://github.com/user-attachments/assets/fe48377a-706e-4325-b9b9-61fedec29dfb" width="32%" alt="Visual 1" />
  <img src="https://github.com/user-attachments/assets/8b470eea-c02c-4e34-8a1a-461d20a1f672" width="32%" alt="Visual 2" />
  <img src="https://github.com/user-attachments/assets/a3bd5f06-ee67-4057-9d37-b31bc674509c" width="32%" alt="Visual 3" />
</p>
<p align="center">
  <em>Evolución de la detección: Visualización de las mejoras en segmentación tras aplicar V5 y V6.</em>
</p>


---

## 🏎️ Rendimiento y Hardware (Nitro Engine) 🚀

El pipeline ha sido optimizado especificamente para hardware de última generación (**NVIDIA RTX 5090**), implementando un motor de inferencia personalizado denominado "Nitro":

*   **Mixed Precision (FP16):** Implementación de `torch.amp.autocast` para maximizar el uso de Tensor Cores y reducir el consumo de VRAM, permitiendo batch sizes más grandes.
*   **Vectorización Masiva:** El preprocesamiento de bloques (grid splitting) se reescribió utilizando operaciones vectorizadas de Numpy, eliminando cuellos de botella de CPU.
*   **Direct I/O:** Lectura nativa de normales pre-calculadas desde archivos LAZ, reduciendo el tiempo de pre-carga en un **~80%**.

| Métrica | Detalle |
| :--- | :--- |
| **GPU Target** | NVIDIA RTX 5090 (24GB+ VRAM) |
| **Batch Size** | 64 - 96 (Optimizado) |
| **Capacidad** | Inferencia continua en nubes de >100 Millones de puntos |
| **Velocidad** | 5x más rápido que la implementación base V1 |

> <img width="1397" height="75" alt="image" src="https://github.com/user-attachments/assets/ac5f9fa1-14ff-4a6c-93aa-a5f714399614" />


---

## 📊 Galería de Resultados

El modelo actual demuestra una robustez excepcional, capaz de ignorar "Hard Negatives" (rocas grandes, taludes verticales naturales) y capturar maquinaria completa sin fragmentación.

### Clasificación de Precisión
> <img width="1269" height="584" alt="image" src="https://github.com/user-attachments/assets/5d128d28-6e47-4f20-b6fb-91a126d0c464" />
<img width="1328" height="620" alt="image" src="https://github.com/user-attachments/assets/55eed393-e1b1-4da2-bca8-01e6df7d830e" />


### Robustez en Terreno Complejo
> <img width="1268" height="426" alt="image" src="https://github.com/user-attachments/assets/7ae3afa1-2cce-4400-9385-eaad264debc5" />
<img width="1228" height="419" alt="image" src="https://github.com/user-attachments/assets/e4b68bc7-60f0-4031-92dd-38244e113758" />



---

## 🛠️ Estructura del repositorio

*   `src/models`: Implementación de la arquitectura **PointNet++ MSG**.
*   `src/data`: Data Loaders personalizados para estrategias de sampling V5 (Oversampling) y V6 (Resolution Sync).
*   `scripts/inference`: Scripts de inferencia de producción optimizados (`infer_pointnet_v5.2.py`).
*   `configs`: Archivos de configuración YAML para reproducibilidad de experimentos.
*   `docs`: Reportes técnicos detallados (V1-V6).
