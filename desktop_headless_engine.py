#!/usr/bin/env python3
"""
Desktop Headless Engine
=======================
Entrypoint exclusivo para la aplicación de escritorio Electron.
Se comunica vía CLI y stdout. No requiere Gradio ni inicializa servidores web.
"""

import os
import sys
import argparse
from datetime import datetime

# Agregar el path del proyecto PRIMERO para evitar errores de importación
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

def print_banner():
    print("""
============================================================
   🚀 Point Cloud Inference App V5 - DESKTOP BATCH ENGINE   
------------------------------------------------------------
   Inferencia en Segundo Plano (Headless Process)
   Optimizaciones activas: FP16 + torch.compile
============================================================
""")

def main():
    parser = argparse.ArgumentParser(description='Point Cloud Desktop Headless Engine')
    parser.add_argument('--input', action='append', required=True, help='Archivos de entrada (.laz/.las)')
    parser.add_argument('--output', type=str, required=True, help='Carpeta de salida')
    parser.add_argument('--bypass', action='store_true', help='Activar Bypass AI (solo Postprocesamiento)')
    parser.add_argument('--export_dtm', action='store_true', help='Exportar DTM')
    parser.add_argument('--export_clasificado', action='store_true', help='Exportar nube clasificada (Techos)')
    parser.add_argument('--preset', type=str, default='Las Tórtolas (Default - Suave)', help='Preset de Faena')
    args = parser.parse_args()

    print_banner()
    sys.stdout.flush()

    # V6 Champion: El mismo checkpoint que usa la app Gradio en producción
    # Configuración por versión (match main_inference_app.py / app.py)
    VERSION_CONFIG = {
        "V5": {"num_points": 10000, "suffix": "_PointnetV5", "checkpoint": "LR0.0010_W20_J0.005_R3.5_BEST_IOU.pth"},
        "V6": {"num_points": 2048,  "suffix": "_PointnetV6", "checkpoint": "LR0.0010_W15_J0.005_R3.5_BEST_IOU.pth"},
    }
    
    # Usar V6 por defecto (producción)
    active_version = "V6"
    v_config = VERSION_CONFIG[active_version]
    
    checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints")
    checkpoint_path = None
    
    # Buscar el checkpoint ganador V6 en las carpetas de SWEEP PointNet2
    for root, dirs, files in os.walk(checkpoint_dir):
        # Solo buscar en carpetas de PointNet2 (ignorar RandLANet)
        if "RandLA" in root or "randla" in root:
            continue
        if v_config["checkpoint"] in files:
            checkpoint_path = os.path.join(root, v_config["checkpoint"])
            break

    if not checkpoint_path and not args.bypass:
        print(f"[ERROR] No se encontró ningún checkpoint V6 PointNet++ en {checkpoint_dir}")
        print("  Buscando archivo:", v_config["checkpoint"])
        sys.exit(1)
    
    # Importaciones pesadas diferidas (para que el banner aparezca rápido)
    from app_inference.core.inference_engine import InferenceEngine, InferenceConfig
    from app_inference.core.postprocess import (
        PostProcessor, FixTechoConfig, InterpolConfig, PreCleanConfig
    )
    from app_inference.core.validators import PointCloudValidator

    if not args.bypass:
        print(f"[OK] Checkpoint: {os.path.basename(checkpoint_path)}")
        print(f"[OK] Version: PointNet++ {active_version} ({v_config['num_points']} pts/bloque)")
    print(f"[OK] Preset de faena activo: {args.preset}")
    
    output_dir_final = args.output
    os.makedirs(output_dir_final, exist_ok=True)
    print(f"[OK] Directorio de salida: {output_dir_final}")
    sys.stdout.flush()

    # Configurar Parámetros de Postprocesamiento según Preset
    if args.preset == "Spence (Agresivo)":
        pre_clean_cfg = PreCleanConfig(enabled=True, cluster_eps=1.6, cluster_min_samples=4,
            small_cluster_max_points=60, protect_tall_structures=True,
            protected_min_height=1.8, hole_fill_enabled=True,
            hole_fill_radius=1.8, hole_fill_k=14, hole_fill_min_class1_ratio=0.60)
        fix_cfg = FixTechoConfig(eps=4.0, min_samples=30, z_buffer=0.5,
            max_height=12.0, padding=2.5, proximity_radius=2.0,
            smart_merge=True, merge_radius=4.0, merge_neighbors=2)
        interpol_cfg = InterpolConfig(k_neighbors=6, max_dist=100.0)
    else:  # Las Tortolas (Suave)
        pre_clean_cfg = PreCleanConfig(enabled=True, cluster_eps=1.2, cluster_min_samples=6,
            small_cluster_max_points=35, protect_tall_structures=True,
            protected_min_height=1.8, hole_fill_enabled=True,
            hole_fill_radius=1.2, hole_fill_k=12, hole_fill_min_class1_ratio=0.70)
        fix_cfg = FixTechoConfig(eps=2.5, min_samples=30, z_buffer=1.5,
            max_height=8.0, padding=1.5, proximity_radius=1.5,
            smart_merge=True, merge_radius=2.5, merge_neighbors=4)
        interpol_cfg = InterpolConfig(k_neighbors=12, max_dist=50.0)
    postprocessor = PostProcessor(fix_cfg, interpol_cfg, pre_clean_cfg)
    validator = PointCloudValidator()

    # Iniciar Motor de Inferencia (solo si no estamos en Bypass)
    inference_engine = None
    if not args.bypass:
        inf_config = InferenceConfig(batch_size=64, num_points=v_config["num_points"], use_compile=True)
        inference_engine = InferenceEngine(inf_config)
        print(f"\n[>>] Cargando pesos neuronales en GPU...")
        sys.stdout.flush()
        
        ok = inference_engine.load_model(checkpoint_path, lambda msg: (print(f"    {msg}"), sys.stdout.flush()))
        if not ok:
            print("[ERROR] Falló la inicialización del modelo neuronal.")
            sys.exit(1)
        print("[OK] Motor Neuronal Online")
        sys.stdout.flush()

    start_total = datetime.now()
    total_ok = 0

    # Iterar Archivos
    for file_path in args.input:
        file_path = file_path.strip('"').strip("'")
        if not os.path.exists(file_path):
            print(f"\n[ERROR] Archivo no accesible: {file_path}")
            sys.stdout.flush()
            continue

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"\n{'='*60}")
        print(f"[>>] Procesando Archivo: {base_name}")
        sys.stdout.flush()

        val_result = validator.validate_file(file_path)
        if not val_result.is_valid:
            print(f"[WARN] Saltando archivo {val_result.file_name} por validación fallida: {val_result.error_message}")
            sys.stdout.flush()
            continue

        current_file = file_path

        if not args.bypass:
            # 1. Inferencia
            print("[>>] Ejecutando Inferencia de Red Neuronal Profunda...")
            sys.stdout.flush()
            inference_out = os.path.join(output_dir_final, f"{base_name}_inf.laz")
            res = inference_engine.run_inference(file_path, inference_out,
                lambda msg: (print(f"    {msg}"), sys.stdout.flush()), confidence=0.5)
            
            if not res.success:
                print(f"[ERROR] Inferencia falló: {res.error_message}")
                sys.stdout.flush()
                continue
            print(f"[OK] Inferencia exitosa: {res.total_points} puntos procesados.")
            sys.stdout.flush()
            current_file = inference_out

            # 2. Pre-Clean
            print("[>>] Ejecutando Fase PRE_CLEAN (Limpieza Geométrica Temprana)...")
            sys.stdout.flush()
            preclean_out = os.path.join(output_dir_final, f"{base_name}_preclean.laz")
            res = postprocessor.run_pre_clean(current_file, preclean_out,
                lambda msg: (print(f"    {msg}"), sys.stdout.flush()))
            if res.success:
                current_file = preclean_out
            sys.stdout.flush()

            # 3. Fix-Techos
            print("[>>] Ejecutando Fase FIX_TECHO (Sub-Clustering Espacial)...")
            sys.stdout.flush()
            clasificado_out = os.path.join(output_dir_final, f"{base_name}_clasificado.laz")
            res = postprocessor.run_fix_techo(current_file, clasificado_out,
                lambda msg: (print(f"    {msg}"), sys.stdout.flush()))
            if res.success:
                # Limpieza de temporales pesados
                if os.path.exists(inference_out): os.remove(inference_out)
                if os.path.exists(preclean_out): os.remove(preclean_out)
                
                current_file = clasificado_out
                if args.export_clasificado:
                    print(f"[OUT] Guardado: {base_name}_clasificado.laz")
                elif not args.export_dtm and not args.export_clasificado:
                    # En raras ocasiones si no seleccionan output pero no bypass, borramos el clasificado extra
                    os.remove(clasificado_out)
            sys.stdout.flush()

        # 4. Interpolación DTM
        if args.export_dtm:
            print("[>>] Ejecutando Fase INTERPOL DTM (Generación de Terreno)...")
            sys.stdout.flush()
            dtm_out = os.path.join(output_dir_final, f"{base_name}_DTM.laz")
            res = postprocessor.run_interpol(current_file, dtm_out,
                lambda msg: (print(f"    {msg}"), sys.stdout.flush()))
            if res.success:
                print(f"[OUT] Guardado: {base_name}_DTM.laz")
            else:
                print(f"[ERROR] Falló generación DTM.")
            sys.stdout.flush()

        # Limpieza Final del temporal Clasificado si no lo solicitó el operador
        if not args.bypass and not args.export_clasificado:
            clasificado_out_temp = os.path.join(output_dir_final, f"{base_name}_clasificado.laz")
            if os.path.exists(clasificado_out_temp):
                os.remove(clasificado_out_temp)

        total_ok += 1

    elapsed = (datetime.now() - start_total).total_seconds()
    print(f"\n{'='*60}")
    print(f"[DONE] Tarea Finalizada: {total_ok}/{len(args.input)} archivos correctos")
    print(f"[DONE] Tiempo Total: {elapsed:.1f} segundos")
    print(f"[DONE] Directorio Exportación: {output_dir_final}")
    sys.stdout.flush()
    sys.exit(0)

if __name__ == "__main__":
    main()
