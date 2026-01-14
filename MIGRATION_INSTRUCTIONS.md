# 🔄 Instrucciones de Migración y Contexto para Nueva Sesión

Copia y pega el siguiente bloque (Prompt) al iniciar tu nueva sesión con el AI en el nuevo Docker. Esto garantizará que entienda inmediatamente el estado "V2" del proyecto.

---

## 📋 PROMPT DE INICIO (Copiar y Pegar al AI)

```text
Hola, estamos continuando el proyecto "Tortolas-segmentation" (Minería a Cielo Abierto) en su FASE V2.
Este es un entorno migrado (Nueva carpeta/Docker), pero el código y los datos son los mismos.

🛑 CONTEXTO CRÍTICO (LEE ESTO PRIMERO):
1.  **Estado Actual:** Estamos en la fase "V2 High Density & Robustness".
    - El documento maestro es `docs/TECHNICAL_REPORT_V2.md`. Léelo para entender el pivot de V1 a V2.
    - `TECHNICAL_REPORT_V1.md` es solo histórico.

2.  **Configuraciones Activas (V2):**
    - **RandLANet:** Configuración "Efficiency Spot" (25,000 puntos, Batch 4). 40k era muy lento, 65k rompía la matriz de complejidad.
    - **PointNet++:** Configuración "Robust" (10,000 puntos, Input Dropout 0.2, Augmentation 0.8-1.2). Diseñado para evitar overfitting exagerado.

3.  **Objetivo Inmediato (Fase 2.2):**
    - Estamos ejecutando Hyperparameter Sweeps de 300 épocas.
    - Los archivos clave de configuración de sweep son:
        - `configs/pointnet2/sweep_hyperparam.yaml`
        - `configs/randlanet/sweep_hyperparam.yaml`
    - Queremos maximizar `iou_maq`.

4.  **Tu Misión Ahora:**
    - Verifica que el entorno tenga las dependencias (`requirements.txt` instalado).
    - Ayúdame a loguearme en W&B (`wandb login`).
    - Verifica que los datos (`data/processed`) estén visibles.
    - Ayúdame a lanzar los agentes de Sweep nuevamente para continuar el entrenamiento nocturno.

Por favor, confirma que has leído `docs/TECHNICAL_REPORT_V2.md` y revisado los archivos YAML de configuración V2 antes de darme instrucciones.
```

---

## ✅ Checklist de Migración (Para ti, Usuario)

Antes de pegar ese prompt, asegúrate de haber hecho esto en el nuevo Docker:

1.  **Copiar Datos:** Asegúrate de que la carpeta `data/processed` (con los bloques `.npy`) se copió correctamente a la nueva ubicación.
2.  **Instalar Librerías:** Posiblemente necesites correr:
    ```bash
    pip install -r requirements.txt
    pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
    ```
    *(Ajusta la versión de CUDA/Torch según el nuevo Docker)*.
3.  **W&B Key:** Ten a mano tu API Key de Weights & Biases.
