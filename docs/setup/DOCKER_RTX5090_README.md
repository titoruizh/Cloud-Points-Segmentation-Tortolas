# 🚀 Docker Setup para RTX 5090 - Point Cloud Deep Learning

Este Docker está optimizado para entrenar modelos de Deep Learning con nubes de puntos en **RTX 5090** (arquitectura Blackwell).

## 🎯 Características

- ✅ CUDA 12.8 para RTX 5090 (Compute Capability 9.0)
- ✅ PyTorch Nightly con soporte Blackwell
- ✅ Open3D para procesamiento de point clouds
- ✅ RandLA-Net y PointNet2 listos para usar
- ✅ Bibliotecas especializadas (laspy, torch-geometric, etc.)

## 📋 Pre-requisitos en el Host (Windows)

1. **Drivers NVIDIA actualizados** (566.03 o superior para RTX 5090)
2. **Docker Desktop** con WSL2
3. **NVIDIA Container Toolkit** (para GPU en Docker)

### Instalar NVIDIA Container Toolkit en WSL2

```bash
# En WSL2 Ubuntu
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## 🏗️ Construcción del Contenedor

```bash
# Opción 1: Dev Container (Recomendado en VS Code)
# Abre el proyecto en VS Code y selecciona "Reopen in Container"

# Opción 2: Docker manual
docker build -t pointcloud-rtx5090 .
```

## 🎮 Ejecutar el Contenedor

```bash
# Con GPU
docker run --gpus all -it --rm \
  -v ${PWD}:/workspace \
  -v ${PWD}/data:/workspace/data \
  -v ${PWD}/checkpoints:/workspace/checkpoints \
  pointcloud-rtx5090

# Verificar GPU
python3 verify_gpu.py
```

## ✅ Verificación de la RTX 5090

Una vez dentro del contenedor:

```python
import torch
print(torch.cuda.is_available())  # Debe ser True
print(torch.cuda.get_device_name(0))  # NVIDIA GeForce RTX 5090
print(torch.cuda.get_device_capability(0))  # (9, 0)
```

O ejecuta el script de verificación completo:

```bash
python3 verify_gpu.py
```

## 📊 Entrenar Modelos

```bash
# RandLA-Net
python train.py --config configs/randlanet/config.yaml

# PointNet2
python train.py --config configs/pointnet2/config.yaml
```

## 🐛 Troubleshooting

### GPU no detectada en Docker

```bash
# Verificar drivers en host
nvidia-smi

# Verificar runtime de Docker
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

### Memoria insuficiente

La RTX 5090 tiene 32GB, pero si tienes OOM:

```python
# En tu script de entrenamiento
torch.cuda.empty_cache()
# O reduce batch_size en configs
```

### PyTorch no reconoce RTX 5090

Asegúrate de estar usando PyTorch nightly:

```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

## 📝 Notas Importantes

1. **Open3D sin CUDA**: Open3D usa CPU para operaciones de I/O y visualización. PyTorch maneja todas las operaciones en GPU durante el entrenamiento.

2. **Arquitectura 9.0+PTX**: El flag `TORCH_CUDA_ARCH_LIST="9.0+PTX"` permite compatibilidad forward con futuras GPUs.

3. **Torch Geometric**: Las operaciones de grafos (KNN, FPS, etc.) están optimizadas para CUDA y funcionan perfectamente en RTX 5090.

## 🔗 Referencias

- [Open3D Documentation](http://www.open3d.org/docs/)
- [PyTorch CUDA Support](https://pytorch.org/get-started/locally/)
- [RandLA-Net Paper](https://arxiv.org/abs/1911.11236)
- [NVIDIA RTX 5090](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/)
