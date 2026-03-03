---
description: Guía de arquitectura y despliegue del motor de Point Cloud Inference (app_frontend, Docker, Electron).
---

# 🏭 Point Cloud Engine - Production Skill

Esta skill define el conocimiento crítico sobre cómo funciona el entorno de producción (`app_frontend`) frente al entorno de desarrollo (raíz del proyecto). **NUNCA modifiques el entorno de producción sin entender estos conceptos.**

## 1. División de Entornos (DEV vs PROD)

*   **Ruta Raíz (`/workspaces/...`)**: Es un entorno puramente **DEV**. Contiene scripts de entrenamiento (`TRAIN.py`), preparación de datos (`scripts/`), y pruebas directas (`main_inference_app.py`). *Altera estos archivos bajo tu propio riesgo.*
*   **Carpeta `app_frontend/`**: Es el entorno de **PRODUCCIÓN**. Contiene la aplicación Electron (GUI) y los manifiestos exactos para crear la imagen Docker que se entrega al cliente final.

## 2. Arquitectura de Inferencia en Producción

El modelo de distribución para el cliente final consta de dos artefactos:
1.  **El Ejecutable Portable (`Point Cloud Engine.exe`)**: Generado desde `app_frontend` usando `npm run dist`. Contiene una app React/Electron.
2.  **La Imagen Docker IA (`pointcloud-engine:prod`)**: Generada desde `app_frontend/production/Dockerfile.production`. Contiene PyTorch Nightly (`cu128`) para soportar RTX 5090 (Lovelace/Blackwell) y un backend en Python mínimo. Se exporta al cliente como un `.tar.gz` gigante usando `docker save`.

## 3. Flujo Crítico de Ejecución (Electron a Docker)

Cuando el usuario hace clic en "Iniciar" en la interfaz gráfica (`App.jsx`), ocurre lo siguiente en `main.js`:

1.  **Identificación de Discos**: Lee de dónde vienen las nubes (ej. `C:\Users\...` o `D:\...`).
2.  **Mounts Automáticos**: Arranca el contenedor `pointcloud-engine:prod` en segundo plano montando esos discos específicos (`-v C:\:/host/C`).
    *   *Bug Conocido*: Si Docker Desktop en Windows no tiene permisos de **File Sharing** para esos discos, la inferencia funciona pero no puede guardar el output (salen carpetas vacías o error de permisos).
3.  **Path Translation**: Las rutas de Windows se deben convertir a formato Linux-Docker. Una variable como `C:\Data\nube.laz` en Electron se envía al backend Python de Docker como `/host/C/Data/nube.laz`.
4.  **Ejecución Headless**: Llama a `python desktop_headless_engine.py` dentro de Docker con los parámetros seleccionados en la UI.

## 4. OOM (Out Of Memory) y Perfiles de Hardware

El motor usa Open3D (Tensor API en GPU) y PointNet++ optimizado con `fp16` + `torch.compile`. 
Esto es devastadoramente rápido en una RTX 5090 (32GB VRAM) pero lanza errores OOM (`OPEN3D_GET_LAST_CUDA_ERROR` o "ParallelFor failed") en GPUs menores como la RTX 4060 (8GB VRAM).

Se implementaron dos salvavidas críticos:
1.  **CPU Fallback para Normales (`geometry.py`)**: Si `pcd.estimate_normals()` falla en Open3D-CUDA por falta de VRAM, el código atrapa el error (`Exception`) y transfiere **esa nube puntual a la memoria RAM normal (CPU)** para calcular las normales lentamente pero sin crashear toda la pipeline.
2.  **Hardware Profiler (`App.jsx` + CLI `--batch`)**: El usuario selecciona su "Perfil de Tarjeta":
    *   `--batch 64`: Ultra (Para RTX 5090).
    *   `--batch 16`: Balanceado (Para RTX 3060/4060).
    *   `--batch 8`: Seguro (Para notebooks / bajo VRAM).

## 5. REGLAS PARA AGENTES IA

1.  **NO USES `app_frontend/situacion_XXX.txt`** para evaluar bugs actuales de DEV. Esos son logs viejos de pruebas de rendimiento de los clientes.
2.  Si un bug ocurre en `python main_inference_app.py` (ejecutado en la terminal raíz), es un problema de DEV. **NO apliques lógicas exclusivas de Producción/Docker a DEV** sin consultarlo.
3.  La documentación maestra que el cliente final y Tito deben seguir para crear la versión final está exclusivamente en: `app_frontend/production/MANUAL_PRODUCCION_FINAL.md`.
