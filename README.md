# 🌩️ GeoAI: Large Scale Point Cloud Segmentation
### Automated Mining Asset Detection via Deep Learning

![Status](https://img.shields.io/badge/Status-Production_V6-success?style=for-the-badge)
![Tech](https://img.shields.io/badge/Stack-PyTorch_%7C_CUDA-orange?style=for-the-badge&logo=pytorch)
![Hardware](https://img.shields.io/badge/Hardware-NVIDIA_RTX_5090-black?style=for-the-badge&logo=nvidia)

---

Proyecto de investigación y desarrollo enfocado en la segmentación semántica automática de nubes de puntos fotogramétricas a gran escala en minería. El objetivo principal es la clasificación precisa de **Maquinaria Pesada** vs **Terreno** en entornos complejos para posteriormente en postprocesso realizar un 'Bulldozer' de los puntos clasificados y obtener terreno limpio DTM sin objetos de manera automatizada.

## 📐 Arquitectura y Metodología Técnica

Tras exhaustivas pruebas comparativas (Benchmark V1-V4), la solución final cristalizó en una variante optimizada de **PointNet++ MSG (Multi-Scale Grouping)**. Esta arquitectura demostró una superioridad crítica en la captura de **topologías finas** (brazos hidráulicos, suspensiones) frente a alternativas como RandLa-Net.

| Versión | Configuración Ganadora (LR / Weights / Radius) | IoU Maquinaria (Val) | mIoU (Val) | Accuracy Global | Notas Técnicas |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **V2 (Robust)** | `LR: 0.0047` / `W: 10` / `R: 2.5m` | 73.77% | 81.87% | 94.46% | Configuración base robusta. Inicio de experimentación. |
| **V3** | `LR: 0.0040` / `W: 10` / `R: 5.0m` | 69.81% | 79.69% | 95.66% | Variación de radios. Rendimiento inferior a V2. |
| **V4 (RGB)** | `LR: 0.0010` / `W: 20` / `R: 3.5m` | 83.64% | 88.51% | 98.63% | **Salto Cuántico.** Inclusión de canales RGB (d_in=10). Gran mejora. |
| **V5 (No-Vert)** | `LR: 0.0010` / `W: 20` / `R: 3.5m` | **87.75%** | **92.95%** | **98.67%** | **Optimización.** Eliminación de canal verticalidad explícito (d_in=9). Menor ruido. |
| **V6 (0.25m)** | `LR: 0.0010` / `W: 15` / `R: 3.5m` | **88.85%** | **93.06%** | **98.61%** | **Definitiva (Resolution Sync).** Entrenamiento a densidad real (2048 pts) igualando producción. Máxima generalización. |

### Definición del Modelo
La red neuronal opera bajo principios de **"Purificación Geométrica"**, eliminando sesgos inductivos clásicos para maximizar la generalización:
*   **Input Tensors:** `(N, 9)` → Coordenadas (XYZ) + Color (RGB) + Normales de Superficie (Nx, Ny, Nz).
    *   *Nota Técnica:* Se eliminó explícitamente la feature de "Verticalidad" para evitar el overfitting en techos planos y estructuras de contención.
*   **Densidad Sincronizada:** Entrenamiento nativo a **0.25m/voxel** con **2048 puntos por bloque**, garantizando congruencia matemática entre el *Training Manifold* y la data de inferencia real. incluso probado a 10000 puntos por bloque por resoluciones a 0.10m con los mismos buenos resultados.

### MLOps & Pipeline
El proyecto sigue una estructura estricta de **Data-Centric AI**:
1.  **Ingesta:** `data/raw` (LAS/LAZ) → Validación de integridad y normalización.
2.  **Tracking:** Integración profunda con **Weights & Biases (WandB)** para monitoreo en tiempo real de gradientes, LR scheduling (Cosine Annealing) y versionado de artefactos.
3.  **Checkpoints:** Serialización de modelos basada en métricas de validación (`best_iou_machinery.pth`) almacenados en `checkpoints/`.

<img width="1328" height="620" alt="image" src="https://github.com/user-attachments/assets/96a2daaf-4c4f-4f54-a225-06c0c747a78b" />


---

### 📸 Comparativa Visual

<p align="center">
  <img src="https://github.com/user-attachments/assets/fe48377a-706e-4325-b9b9-61fedec29dfb" width="32%" alt="Visual 1" />
  <img src="https://github.com/user-attachments/assets/8b470eea-c02c-4e34-8a1a-461d20a1f672" width="32%" alt="Visual 2" />
  <img src="https://github.com/user-attachments/assets/a3bd5f06-ee67-4057-9d37-b31bc674509c" width="32%" alt="Visual 3" />
</p>
<p align="center">
  <em>Vista Planta Ortomosaico | Vista Planta Nube de puntos RGB | Vista 3D Nube de puntos Clasificada</em>
</p>


---

## ⚡ Nitro Inference Engine & Hardware

El despliegue productivo se ejecuta sobre una estación de trabajo **NVIDIA RTX 5090**, utilizando un motor de inferencia propietario ("Nitro Engine") diseñado para throughput masivo.

### Optimizaciones de Bajo Nivel
*   **FP16 Mixed Precision:** Rutinas de inferencia reescritas con `torch.amp` para duplicar el *effective memory bandwidth* y maximizar la utilización de los Tensor Cores.
*   **Vectorización Numpy:** El algoritmo de *Grid Tiling* (corte de la nube en bloques) fue portado de bucles Python a operaciones matriciales vectorizadas, reduciendo el overhead de CPU en un **90%**.
*   **IO-Bound Optimization:** Lectura directa de atributos binarios (Normales/Color) desde cabeceras LAZ, evitando el reprocesamiento redundante con Open3D.

| Recurso | Especificación |
| :--- | :--- |
| **Compute** | NVIDIA RTX 5090 (32GB GDDR6X) |
| **Throughput** | ~120 M points x min |
| **Batch Strategy** | Dynamic Batching (64-96 bloques) |
| **Precision** | FP16/FP32 Mixed Mode |

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
