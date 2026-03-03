# Integración Multimodal Futura: PointNet++ (3D) + YOLO (2D)

Este documento establece la arquitectura base y los enfoques estratégicos para el próximo gran proyecto de GeoAI: **Fusión de Detección 2D (Ortomosaicos) con Segmentación 3D (Nubes de Puntos)**.

Dado que las nubes de puntos (`.laz`) y los ortomosaicos (`.tif`/`.png`) comparten exactamente el mismo sistema de coordenadas geográficas, esta integración es matemáticamente viable y altamente poderosa.

---

## 1. Los Dos Enfoques Estratégicos

### Enfoque A: "Top-Down Focus" (YOLO guía a PointNet) - *Velocidad Extrema*
En este enfoque, se utiliza el ortomosaico como un "mapa de calor" económico para no desperdiciar poder de cómputo geométrico en zonas vacías.

1. **Inferencia 2D**: Pasa el ortomosaico por YOLO.
2. **Detección**: YOLO arroja *Bounding Boxes* [X_min, Y_min, X_max, Y_max] con coordenadas reales.
3. **Recorte Espacial**: Un script toma la Nube de Puntos MASIVA y **corta** físicamente un bloque espacial (ej. usando PDAL o Laspy) exclusivo de esas coordenadas dictadas por YOLO.
4. **Verificación 3D**: PointNet++ procesa *solo* esos recortes diminutos. Si encuentra geometría de máquina, extrae los puntos.
* **Ventaja**: El procesamiento global pasa de horas a minutos, ya que PointNet no procesa el 99% vacío de la mina.
* **Riesgo**: Si un camión está oculto bajo un techo/sombra densa que YOLO no detecta en la foto, PointNet nunca examinará esa zona.

### Enfoque B: "Bottom-Up Verification" (PointNet guía a YOLO) - *Precisión Absoluta (Doble Check)*
En este enfoque, la geometría volumétrica es la ley, y la visión 2D se usa para eliminar los infames "Falsos Positivos" (Ej: rocas con forma de bulldozer).

1. **Inferencia 3D Normal**: PointNet procesa toda la mina como lo hace hoy y extrae los clusters de "Maquinaria".
2. **Generación de Tiles 2D**: Por cada cluster extraído, se calcula su centro coordenado y su radio. Con esos datos, se recorta automáticamente un "Tile" 2D (ej. 10x10 metros) directamente del Ortomosaico gigante.
3. **Inferencia 2D**: El Tile se pasa a YOLO. 
4. **Decisión Conjunta**: Si YOLO confirma "Máquina" (Confianza cruzada), se guarda permanentemente. Si YOLO no ve nada, se marca como Falso Positivo.
* **Ventaja**: Genera automáticamente un Dataset Perfecto y Pareado. Podrás guardar el Tile 2D (`.png`) junto con su Nube 3D (`.laz`) en una carpeta para reentrenar ambos modelos en el futuro.
* **Riesgo**: Sigue tomando el tiempo de inferencia total de PointNet.

---

## 2. Arquitectura de Infraestructura (El entorno `geoai-rtx5090`)

Es vital entender que el contenedor actual lab/dev que usas (`geoai-rtx5090` compilado desde el `Dockerfile` raíz) es un **Entorno de Alta Precisión**. 
Tiene forzadas instrucciones de hiper-compilación C++ (como `torch-cluster`, `torch-scatter`) específicas para la arquitectura de tu RTX 5090 (`sm_120`, Blackwell).

### ¿Cómo lidiar con YOLO en este ecosistema?
**Librerías "Choconas" de YOLO**: 
Modelos como Ultralytics (YOLOv8/10/11) tienen dependencias fuertes que tradicionalmente rompen entornos matemáticos puros. Traen consigo librerías como `opencv-python` (que muchas veces instala versiones raras de `ffmpeg` o `libGL` que chocan con Open3D) y versiones específicas de `torchvision`.

#### Opciones de Despliegue:

**Opción Recomendada (Dos Contenedores, un Pipe):**
Crear un nuevo Dockerfile `Dockerfile.yolo` basado **SÓLO** en `nvidia/cuda:12.8.0-cudnn-runtime` + Ultralytics.
- **Por qué**: Mantienes tu templo de Pointcloud prístino. 
- **Cómo hablan**: Un pequeño script en tu PC anfitriona "Orquestador" le dice a YOLO: *"Toma este TIF y dame el JSON"*. Luego le dice al geoai-rtx5090: *"Usa el geoai-rtx5090 para recortar estos X,Y y procesar"*.

**Opción Monolítica (Apostarlo todo en `geoai-rtx5090`):**
Modificar el `Dockerfile` raíz para incluir YOLO.
Si deseas ir por este camino (tener 1 solo super-contenedor), **debes ser quirúrgico**:
Instalar Ultralytics SIN sus versiones atadas de PyTorch (que destruirían la compilación de `sm_120` que hicimos), usando flag especiales:
`pip install ultralytics --no-deps` seguido de instalaciones cuidadosas manuales de `opencv-python-headless`.

---

## 3. Próximo Paso Experimental
Para el próximo inicio del sub-proyecto YOLO, abre el editor en la carpeta donde tienes tu modelo YOLO viejo y usa un Agente con un Prompt como el siguiente para empezar bien:

> *"AI, voy a integrar mi modelo YOLO de ortomosaicos mineros con mi actual motor geométrico PointNet++. Mi motor actual corre en un entorno Docker muy delicado compilado en C++ exclusivo para mi RTX 5090 ('geoai-rtx5090'). Revisa todos los `.py` y `requirements.txt` de esta carpeta de YOLO y hazme un reporte de qué librerías gráficas, CV2 u otras dependencias pesadas usan. Dime si recomiendas intentar fusionar YOLO dentro de mi 'Dockerfile' masivo de PointNet, o si arquitectónicamente me recomiendas hacer un 'Dockerfile.yolo' independiente y comunicarlos mediante archivos JSON."*
