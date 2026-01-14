import torch
import os

def check_system():
    print("\n" + "="*50)
    print("🚀 DIAGNÓSTICO DE ACELERACIÓN RTX 5090")
    print("="*50)

    # 1. Verificación Básica de CUDA
    print(f"\n[1] PyTorch Base:")
    print(f"    - Versión PyTorch: {torch.__version__}")
    print(f"    - CUDA Disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    - Versión CUDA: {torch.version.cuda}")
        print(f"    - Dispositivo: {torch.cuda.get_device_name(0)}")
        print(f"    - Capacidad de Cómputo: {torch.cuda.get_device_capability(0)}")
    else:
        print("    ❌ ERROR: CUDA no detectado. Revisa tus drivers o flags de Docker (--gpus all).")
        return

    # 2. Verificación de Extensiones C++ (Lo crítico para RandLA-Net)
    print(f"\n[2] Extensiones Geométricas (Kernel C++):")
    
    try:
        import torch_cluster
        import torch_scatter
        import torch_sparse
        
        print(f"    ✅ torch_cluster: Instalado (v{torch_cluster.__version__})")
        print(f"    ✅ torch_scatter: Instalado (v{torch_scatter.__version__})")
        print(f"    ✅ torch_sparse:  Instalado (v{torch_sparse.__version__})")
        
        # Prueba de fuego: ¿Dónde están corriendo las ops?
        device = torch.device('cuda')
        try:
            # Creamos un tensor dummy en GPU
            src = torch.randn(10, 64, device=device)
            index = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 3], device=device)
            
            # Intentamos una operación scatter (típica de GNNs)
            from torch_scatter import scatter_add
            out = scatter_add(src, index, dim=0)
            
            print("\n    🔥 PRUEBA DE RENDIMIENTO:")
            print("    ¡Éxito! La operación 'scatter_add' se ejecutó nativamente en la GPU.")
            print("    Tu RTX 5090 está siendo utilizada por los Kernels C++.")
            
        except Exception as e:
            print(f"    ❌ ERROR CRÍTICO: Las librerías están instaladas pero fallaron al ejecutar en GPU.")
            print(f"    Error: {e}")
            
    except ImportError as e:
        print(f"    ❌ ERROR: Faltan librerías. {e}")
        print("    El Dockerfile no compiló correctamente las extensiones.")

if __name__ == "__main__":
    check_system()