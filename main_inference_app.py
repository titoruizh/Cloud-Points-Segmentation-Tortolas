#!/usr/bin/env python3
"""
Point Cloud Inference App V5
============================
Aplicación de inferencia con interfaz web Gradio.

Uso:
    python3 main_inference_app.py              # Inicia en localhost:7860
    python3 main_inference_app.py --port 8080  # Puerto personalizado
    python3 main_inference_app.py --share      # Link público de Gradio
"""

import os
import sys
import argparse

# IMPORTANTE: Agregar el path del proyecto PRIMERO
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

def print_banner():
    print("""

                                                                  ║
   🚀 Point Cloud Inference App V5                               ║
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   ║
                                                                  ║
   PointNet++ "Geometric Purification"                           ║
   Optimizado para RTX 5090 | FP16 + torch.compile               ║
                                                                  ║
   Pipeline: Inferencia → FIX_TECHO → INTERPOL (DTM)             ║
                                                                  ║
""")

def check_dependencies():
    missing = []
    try:
        import torch
    except ImportError:
        missing.append("torch")
    try:
        import gradio
    except ImportError:
        missing.append("gradio")
    try:
        import laspy
    except ImportError:
        missing.append("laspy")
    try:
        import open3d
    except ImportError:
        missing.append("open3d")
    try:
        from src.models.pointnet2 import PointNet2
    except ImportError as e:
        missing.append(f"src.models.pointnet2 ({e})")
    if missing:
        print(f"❌ Dependencias faltantes: {', '.join(missing)}")
        print("   Instala con: pip install torch gradio laspy open3d")
        return False
    return True

def print_system_info():
    """Imprime información detallada del sistema para debugging"""
    print("\n" + "="*60)
    print("📊 INFORMACIÓN DEL SISTEMA")
    print("="*60)
    
    # Sistema básico
    import platform
    print(f"Sistema Operativo: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    
    # GPU
    try:
        import torch
        print(f"\n🔥 GPU:")
        if torch.cuda.is_available():
            print(f"   GPU Detectada: {torch.cuda.get_device_name(0)}")
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   VRAM Total: {total_mem:.2f} GB")
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            print(f"   VRAM Reservada: {reserved:.2f} GB")
            print(f"   VRAM Usada: {allocated:.2f} GB")
            print(f"   VRAM Libre: {total_mem - reserved:.2f} GB")
        else:
            print("   ⚠️ No hay GPU disponible (usando CPU)")
    except:
        print("   ⚠️ No se pudo obtener info de GPU")
    
    # CPU y RAM
    try:
        import psutil
        cpu_count = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        mem = psutil.virtual_memory()
        print(f"\n💻 CPU:")
        print(f"   Núcleos: {cpu_count}")
        if cpu_freq:
            print(f"   Frecuencia: {cpu_freq.current:.0f} MHz")
        print(f"   RAM Total: {mem.total / 1024**3:.2f} GB")
        print(f"   RAM Disponible: {mem.available / 1024**3:.2f} GB")
        print(f"   RAM Usada: {mem.percent}%")
    except:
        print("   ⚠️ No se pudo obtener info de CPU/RAM (psutil no disponible)")
    
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Point Cloud Inference App V5')
    parser.add_argument('--port', type=int, default=7860, help='Puerto del servidor (default: 7860)')
    parser.add_argument('--share', action='store_true', help='Generar link público')
    parser.add_argument('--no-check', action='store_true', help='Omitir verificación de dependencias')
    parser.add_argument('--prod', action='store_true', help='Lanzar en Modo Producción (oculta opciones avanzadas para el operador)')
    args = parser.parse_args()
    
    print_banner()
    print_system_info()
    
    if not args.no_check:
        print("🔍 Verificando dependencias...")
        if not check_dependencies():
            sys.exit(1)
        print("✅ Dependencias OK\n")
    
    output_dir = os.path.join(PROJECT_ROOT, "data/predictions/app_output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Directorio de salida: {output_dir}\n")
    
    print(f"🌐 Iniciando servidor en puerto {args.port}...")
    print(f"   Abre en tu navegador: http://localhost:{args.port}")
    print("\n" + "="*60 + "\n")
    
    # Import aquí para asegurar que sys.path está configurado
    from app_inference.ui.app import launch_app
    launch_app(port=args.port, share=args.share, prod_mode=args.prod)

if __name__ == "__main__":
    main()
