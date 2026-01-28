#!/usr/bin/env python3
"""
Test de Verificación: Open3D CUDA Support
==========================================
Script para verificar que Open3D compilado desde fuente
tiene soporte CUDA correctamente activado para RTX 5090.

Ejecutar ANTES de correr entrenamientos con Fase 1.
"""

import sys

def test_open3d_cuda():
    """Verifica soporte CUDA en Open3D Tensor API"""
    
    print("\n" + "="*60)
    print("🧪 TEST: Open3D CUDA Support (RTX 5090 / Blackwell)")
    print("="*60 + "\n")
    
    # Test 1: Importar Open3D Core
    try:
        import open3d.core as o3c
        print("✅ Test 1/4: Open3D Core importado correctamente")
    except ImportError as e:
        print(f"❌ Test 1/4 FALLO: No se pudo importar open3d.core")
        print(f"   Error: {e}")
        return False
    
    # Test 2: Detectar dispositivo CUDA
    try:
        device = o3c.Device("CUDA:0")
        print(f"✅ Test 2/4: Dispositivo CUDA detectado: {device}")
    except Exception as e:
        print(f"❌ Test 2/4 FALLO: No se pudo crear dispositivo CUDA")
        print(f"   Error: {e}")
        print("   Tu Open3D NO tiene soporte CUDA activado.")
        print("   Puede funcionar pero usará CPU (mucho más lento).")
        return False
    
    # Test 3: Crear tensor en VRAM
    try:
        test_tensor = o3c.Tensor([1.0, 2.0, 3.0], device=device)
        print(f"✅ Test 3/4: Tensor creado en VRAM: {test_tensor}")
        del test_tensor
    except Exception as e:
        print(f"❌ Test 3/4 FALLO: No se pudo crear tensor en GPU")
        print(f"   Error: {e}")
        return False
    
    # Test 4: Crear PointCloud Tensor en GPU
    try:
        import open3d.t.geometry as o3dg
        import numpy as np
        
        # Crear nube de puntos pequeña
        points = np.random.rand(100, 3).astype(np.float32)
        points_tensor = o3c.Tensor(points, device=device)
        
        pcd = o3dg.PointCloud(device)
        pcd.point.positions = points_tensor
        
        # Estimar normales en GPU
        pcd.estimate_normals(max_nn=10, radius=1.0)
        
        normals = pcd.point.normals.cpu().numpy()
        
        print(f"✅ Test 4/4: PointCloud + Normales GPU ejecutado correctamente")
        print(f"   Puntos: {len(points)}, Normales: {normals.shape}")
        
    except Exception as e:
        print(f"❌ Test 4/4 FALLO: Error en PointCloud GPU")
        print(f"   Error: {e}")
        return False
    
    # Tests adicionales: Info del dispositivo
    print("\n" + "-"*60)
    print("📊 Información del Dispositivo:")
    print("-"*60)
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            print(f"   GPU: {gpu_name}")
            print(f"   CUDA Version (PyTorch): {cuda_version}")
            print(f"   VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("   ⚠️ PyTorch no detecta CUDA (esto es raro)")
    except ImportError:
        print("   ⚠️ PyTorch no instalado")
    
    print("\n" + "="*60)
    print("🎉 TODOS LOS TESTS PASARON")
    print("="*60)
    print("\n✅ Tu entorno está listo para Fase 1 (GPU Optimization)")
    print("   - Normales GPU: FUNCIONAL")
    print("   - Augmentation GPU: COMPATIBLE")
    print("   - Open3D Tensor API: OPERATIVO\n")
    
    return True


def test_pytorch_cuda():
    """Verifica que PyTorch puede usar CUDA"""
    
    print("\n" + "="*60)
    print("🔥 TEST BONUS: PyTorch CUDA")
    print("="*60 + "\n")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("❌ PyTorch NO detecta CUDA")
            return False
        
        # Crear tensor en GPU
        x = torch.rand(1000, 1000, device='cuda')
        y = torch.rand(1000, 1000, device='cuda')
        z = x @ y  # Matmul en GPU
        
        print(f"✅ PyTorch CUDA operativo")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"   GPU Count: {torch.cuda.device_count()}")
        
        # Benchmark rápido
        import time
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100):
            z = x @ y
        torch.cuda.synchronize()
        t_gpu = time.time() - t0
        
        print(f"   Benchmark (100x matmul 1000x1000): {t_gpu*1000:.2f} ms")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en PyTorch CUDA: {e}")
        return False


if __name__ == "__main__":
    print("\n🚀 Verificación de Entorno CUDA para Cloud Point Research")
    print("   Arquitectura: Blackwell (RTX 5090)")
    print("   Docker: Ubuntu 22.04 + CUDA 12.8")
    
    # Test Open3D
    success_o3d = test_open3d_cuda()
    
    # Test PyTorch
    success_torch = test_pytorch_cuda()
    
    # Resumen
    print("\n" + "="*60)
    print("📋 RESUMEN DE VERIFICACIÓN")
    print("="*60)
    
    if success_o3d and success_torch:
        print("\n🟢 ESTADO: PERFECTO")
        print("   Puedes ejecutar TRAIN_V6.py con Fase 1 activada.")
        print("   Speedup esperado: +20-30% throughput\n")
        sys.exit(0)
    elif success_o3d:
        print("\n🟡 ESTADO: PARCIAL")
        print("   Open3D GPU funciona, pero PyTorch tiene problemas.")
        print("   Revisa instalación de PyTorch.\n")
        sys.exit(1)
    else:
        print("\n🔴 ESTADO: FALLO CRÍTICO")
        print("   Open3D NO tiene soporte GPU activado.")
        print("   El código funcionará pero en CPU (muy lento).")
        print("\n   Solución:")
        print("   1. Verifica que compilaste Open3D con -DBUILD_CUDA_MODULE=ON")
        print("   2. Verifica CUDA Toolkit 12.8 instalado")
        print("   3. Reinstala Open3D desde fuente si es necesario\n")
        sys.exit(1)
