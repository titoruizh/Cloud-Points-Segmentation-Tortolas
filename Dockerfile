# -----------------------------------------------------------------------------
# Dockerfile Optimizado para NVIDIA RTX 5090 (Blackwell Architecture)
# GeoAI Engineering - Point Cloud Deep Learning
# -----------------------------------------------------------------------------

# 1. Imagen Base: Usamos devel para tener nvcc (compilador CUDA) completo
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

LABEL maintainer="Tito GeoAI"

# Evitar interacciones durante la instalación
ENV DEBIAN_FRONTEND=noninteractive

# -----------------------------------------------------------------------------
# VARIABLES DE ENTORNO CRÍTICAS PARA RTX 5090
# -----------------------------------------------------------------------------
# TORCH_CUDA_ARCH_LIST="10.0+PTX":
# - 10.0 es la Compute Capability nativa de Blackwell (RTX 5090).
# - +PTX asegura compatibilidad futura si hay revisiones menores.
ENV TORCH_CUDA_ARCH_LIST="10.0+PTX" 

# FORCE_CUDA="1": Obliga a las extensiones a compilarse con soporte GPU
ENV FORCE_CUDA="1"

# MAX_JOBS: Controla hilos de compilación para no saturar la RAM si compilas local
ENV MAX_JOBS=8
ENV PIP_BREAK_SYSTEM_PACKAGES=1
# -----------------------------------------------------------------------------
# 2. DEPENDENCIAS DE SISTEMA
# -----------------------------------------------------------------------------
# Incluimos librerías gráficas (libgl1, libx*) obligatorias para Open3D
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    ninja-build \
    python3-dev \
    python3-pip \
    libgl1 \
    libgles2 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Crear symlink para asegurar que 'python' apunte a python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# -----------------------------------------------------------------------------
# 3. PYTORCH NIGHTLY (PREVIEW)
# -----------------------------------------------------------------------------
# Instalamos la versión preliminar que soporta oficialmente CUDA 12.8
RUN pip3 install --no-cache-dir --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# -----------------------------------------------------------------------------
# 4. COMPILACIÓN MANUAL DE EXTENSIONES (EL "GAME CHANGER")
# -----------------------------------------------------------------------------
# Esto es lo que soluciona la lentitud. Compilamos el código C++ localmente
# para tu tarjeta gráfica.
# NOTA: Este paso tardará entre 10 a 20 minutos durante el 'docker build'.
RUN pip3 install --no-cache-dir --verbose \
    --no-binary=torch-cluster,torch-scatter,torch-sparse,torch-spline-conv \
    torch-cluster \
    torch-scatter \
    torch-sparse \
    torch-spline-conv

# -----------------------------------------------------------------------------
# 5. LIBRERÍAS GEOESPACIALES Y DE NUBES DE PUNTOS
# -----------------------------------------------------------------------------
RUN pip3 install --no-cache-dir \
    torch-geometric \
    open3d \
    laspy[lazrs] \
    trimesh \
    pandas \
    scikit-learn \
    matplotlib \
    tqdm \
    h5py

# Configuración final
WORKDIR /workspace
CMD ["/bin/bash"]