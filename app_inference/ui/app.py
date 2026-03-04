"""
Aplicación Principal Gradio - V5/V6
====================================
Con filtrado de checkpoints por versión, selección de salidas y ganadores marcados.
"""
import os
import sys
import re
import gradio as gr
from datetime import datetime
from typing import List, Optional, Tuple, Dict

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(APP_DIR)
sys.path.insert(0, PROJECT_ROOT)

from app_inference.core.validators import PointCloudValidator
from app_inference.core.inference_engine import InferenceEngine, InferenceConfig
from app_inference.core.postprocess import PostProcessor, FixTechoConfig, InterpolConfig

# Configuración por versión
VERSION_CONFIG = {
    "V5 (0.10m)": {"num_points": 10000, "suffix": "_PointnetV5", "filter": "V5"},
    "V6 (0.25m)": {"num_points": 2048, "suffix": "_PointnetV6", "filter": "V6"}
}

# Checkpoints GANADORES (campeones validados en producción)
WINNER_CHECKPOINTS = {
    "V5": "LR0.0010_W20_J0.005_R3.5_BEST_IOU.pth",   # V5 Champion
    "V6": "LR0.0010_W15_J0.005_R3.5_BEST_IOU.pth",   # V6 Champion
}

# Presets de parámetros por faena
FAENA_PRESETS = {
    "Las Tórtolas": {
        "batch_size": 256,
        "confidence": 0.5,
        "eps": 2.5,
        "z_buffer": 1.5,
        "max_height": 8.0,
        "padding": 1.5,
        "smart_merge": True,
        "merge_radius": 2.5,
        "merge_neighbors": 4,
        "k_neighbors": 12,
        "max_dist": 50.0,
    },
    "Spence": {
        "batch_size": 256,
        "confidence": 0.2,
        "eps": 4.0,
        "z_buffer": 0.5,
        "max_height": 12.0,
        "padding": 2.5,
        "smart_merge": True,
        "merge_radius": 4.0,
        "merge_neighbors": 2,
        "k_neighbors": 6,
        "max_dist": 1000.0,
    },
}


class InferenceApp:
    def __init__(self):
        self.validator = PointCloudValidator()
        self.inference_engine = None
        self.postprocessor = None
        self.log_messages = []
        self.all_checkpoints = self._scan_checkpoints()
    
    def log(self, message: str) -> str:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_messages.append(f"[{timestamp}] {message}")
        return "\n".join(self.log_messages[-50:])
    
    def clear_log(self):
        self.log_messages = []

    def _run_in_subprocess(self, step: str, input_file: str, output_file: str,
                           config_dict: dict) -> dict:
        """
        Ejecuta un paso de postprocesamiento en un proceso hijo con memoria limpia.

        Usa multiprocessing spawn para que el hijo NO herede los ~41 GB de
        heap fragmentado de la inferencia. El hijo arranca con ~200 MB,
        carga solo lo que necesita, y al terminar el OS recupera 100% de su RAM.
        """
        import multiprocessing as mp
        from queue import Empty
        from app_inference.core.postprocess import _postprocess_worker

        ctx = mp.get_context('spawn')
        queue = ctx.Queue()

        proc = ctx.Process(
            target=_postprocess_worker,
            args=(step, input_file, output_file, config_dict, queue)
        )
        proc.start()

        result = None
        while proc.is_alive() or not queue.empty():
            try:
                msg_type, msg = queue.get(timeout=2)
                if msg_type == 'progress':
                    self.log(msg)
                elif msg_type == 'result':
                    result = msg
            except Empty:
                continue

        proc.join(timeout=30)

        # Drenar mensajes residuales
        while not queue.empty():
            try:
                msg_type, msg = queue.get_nowait()
                if msg_type == 'progress':
                    self.log(msg)
                elif msg_type == 'result':
                    result = msg
            except Empty:
                break

        if result is None:
            exit_code = proc.exitcode
            self.log(f"   ❌ Proceso {step} terminó sin resultado (exit code: {exit_code})")
            result = {
                'success': False,
                'points_modified': 0,
                'processing_time': 0.0,
                'error_message': f"Proceso terminado con código {exit_code}",
            }
        elif result['success']:
            self.log(f"   ✅ {step}: {result['points_modified']:,} puntos en {result['processing_time']:.1f}s")
        else:
            self.log(f"   ❌ {step}: {result['error_message']}")

        return result

    def _scan_checkpoints(self) -> List[Tuple[str, str, str, bool]]:
        """Escanea todos los checkpoints: (display, path, version, is_winner)"""
        checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints")
        checkpoints = []
        if not os.path.exists(checkpoint_dir):
            return checkpoints
        for root, dirs, files in os.walk(checkpoint_dir):
            for f in files:
                if f.endswith('.pth'):
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, PROJECT_ROOT)
                    # Detectar versión
                    version = "V5" if "V5" in rel_path else ("V6" if "V6" in rel_path else "")
                    # Verificar si es ganador
                    is_winner = f == WINNER_CHECKPOINTS.get(version, "")
                    display = self._format_display(rel_path, is_winner)
                    checkpoints.append((display, rel_path, version, is_winner))
        # Ordenar: ganadores primero, luego BEST_IOU, luego el resto
        return sorted(checkpoints, key=lambda x: (not x[3], "BEST_IOU" not in x[0], x[1]))
    
    def _format_display(self, rel_path: str, is_winner: bool = False) -> str:
        """Formatea el checkpoint para display con símbolo de ganador."""
        params = {}
        for key, pattern in [('LR', r'LR([\d.]+)'), ('W', r'W(\d+)'), ('R', r'R([\d.]+)')]:
            match = re.search(pattern, rel_path)
            if match:
                params[key] = match.group(1)
        
        parts = []
        
        # Símbolo de ganador o modelo normal
        if is_winner:
            parts.append("🏆")
        else:
            parts.append("🧠")
        
        parts.append("PointNet++")
        
        version_match = re.search(r'V(\d+)', rel_path)
        if version_match:
            parts.append(f"V{version_match.group(1)}")
        
        if 'NoVert' in rel_path:
            parts.append("[NoVert]")
        
        if params:
            parts.append(f"({', '.join(f'{k}={v}' for k,v in params.items())})")
        
        if 'BEST_IOU' in rel_path:
            if is_winner:
                parts.append("⭐GANADOR")
            else:
                parts.append("★BEST")
        elif 'BEST_LOSS' in rel_path:
            parts.append("◆LOSS")
        
        return " ".join(parts) if parts else os.path.basename(rel_path)
    
    def get_checkpoints_for_version(self, version: str) -> List[Tuple[str, str]]:
        """Filtra checkpoints por versión."""
        v_filter = VERSION_CONFIG.get(version, {}).get("filter", "")
        filtered = [(d, p) for d, p, v, is_win in self.all_checkpoints if v == v_filter]
        if not filtered:
            # Si no hay de esa versión, mostrar todos
            filtered = [(d, p) for d, p, v, is_win in self.all_checkpoints]
        return filtered
    
    def get_default_checkpoint(self, version: str) -> Optional[str]:
        """Retorna el checkpoint ganador por defecto para la versión."""
        v_filter = VERSION_CONFIG.get(version, {}).get("filter", "")
        # Buscar el ganador primero
        for d, p, v, is_win in self.all_checkpoints:
            if v == v_filter and is_win:
                return p
        # Si no hay ganador, buscar cualquier BEST_IOU
        for d, p, v, is_win in self.all_checkpoints:
            if v == v_filter and "BEST_IOU" in p:
                return p
        # Si no hay nada, retornar el primero
        for d, p, v, is_win in self.all_checkpoints:
            if v == v_filter:
                return p
        return None
    
    def validate_files(self, files: List) -> str:
        self.clear_log()
        if not files:
            return self.log("⚠️ No se han seleccionado archivos.")
        self.log(f"📋 Validando {len(files)} archivo(s)...")
        valid_count = 0
        for file in files:
            file_path = file.name if hasattr(file, 'name') else file
            result = self.validator.validate_file(file_path)
            if result.is_valid:
                self.log(f"   ✅ {result.file_name} ({result.point_count:,} puntos)")
                valid_count += 1
            else:
                self.log(f"   ❌ {result.file_name}: {', '.join(result.errors)}")
        self.log(f"\n📊 Resumen: {valid_count}/{len(files)} válidos")
        return "\n".join(self.log_messages)
    
    def run_pipeline(self, files, checkpoint_path, output_dir, model_version,
                     output_types, batch_size, use_compile,
                     eps, z_buffer, max_height, padding,
                     smart_merge, merge_radius, merge_neighbors, # Argumentos nuevos
                     k_neighbors, max_dist, confidence,
                     progress=gr.Progress()) -> Tuple[str, str]:
        self.clear_log()
        start_time = datetime.now()
        
        if not files:
            return self.log("❌ No hay archivos para procesar."), ""
        
        if not output_types:
            return self.log("❌ Selecciona al menos un tipo de salida."), ""
        
        # Configuración según versión
        v_config = VERSION_CONFIG.get(model_version, VERSION_CONFIG["V5 (0.10m)"])
        num_points = v_config["num_points"]
        output_suffix = v_config["suffix"]
        
        # Determinar qué postprocesamiento ejecutar
        # "Clasificado" = salida de FIX_TECHO  |  "DTM" = salida de INTERPOL
        export_clasificado = "📊 Clasificado" in output_types
        export_dtm = "🌍 DTM" in output_types
        
        self.log(f"📐 Versión: {model_version} | num_points={num_points}")
        self.log(f"💾 Salidas: {', '.join(output_types)}")
        self.log(f"🧠 Confianza: {confidence}")
        
        if smart_merge:
            self.log(f"🧩 Smart Merge: ACTIVADO (R={merge_radius}m, N={merge_neighbors})")
        
        # Validar archivos
        valid_files = []
        for file in files:
            file_path = file.name if hasattr(file, 'name') else file
            result = self.validator.validate_file(file_path)
            if result.is_valid:
                valid_files.append(file_path)
            else:
                self.log(f"⚠️ Omitiendo {result.file_name}: Sin RGB")
        
        if not valid_files:
            return self.log("❌ Ningún archivo válido."), ""
        
        self.log(f"🚀 Procesando {len(valid_files)} archivo(s)...")
        
        # Cargar modelo
        checkpoint_full_path = os.path.join(PROJECT_ROOT, checkpoint_path)
        if not os.path.exists(checkpoint_full_path):
            return self.log(f"❌ Checkpoint no encontrado: {checkpoint_path}"), ""
        
        config = InferenceConfig(batch_size=int(batch_size), num_points=num_points, use_compile=use_compile)
        self.inference_engine = InferenceEngine(config)
        self.log("🏗️ Cargando modelo...")
        progress(0.05, desc="Cargando modelo...")
        
        if not self.inference_engine.load_model(checkpoint_full_path, lambda msg: self.log(msg)):
            return "\n".join(self.log_messages), ""
        
        # Configurar postprocesamiento
        fix_config = FixTechoConfig(
            eps=eps, z_buffer=z_buffer, max_height=max_height, padding=padding,
            smart_merge=smart_merge, merge_radius=merge_radius, merge_neighbors=int(merge_neighbors)
        )
        interpol_config = InterpolConfig(k_neighbors=int(k_neighbors), max_dist=max_dist)
        self.postprocessor = PostProcessor(fix_config, interpol_config)
        
        output_dir_full = os.path.join(PROJECT_ROOT, output_dir)
        os.makedirs(output_dir_full, exist_ok=True)
        
        # Procesar archivos
        results_detail = []
        for i, file_path in enumerate(valid_files):
            file_name = os.path.basename(file_path)
            base_name = os.path.splitext(file_name)[0]
            
            progress((i + 0.5) / len(valid_files), desc=f"Procesando {file_name}...")
            self.log(f"\n{'='*50}")
            self.log(f"📂 [{i+1}/{len(valid_files)}] {file_name}")
            
            file_result = {'input': file_name, 'outputs': [], 'success': False, 'time': 0,
                          'points': {'total': 0, 'machinery': 0, 'ground': 0}}
            file_start = datetime.now()
            
            # PASO 1: Inferencia (siempre necesaria)
            inference_output = os.path.join(output_dir_full, f"{base_name}{output_suffix}.laz")
            result = self.inference_engine.run_inference(
                file_path, inference_output, 
                lambda msg: self.log(msg), 
                confidence=confidence
            )
            
            if not result.success:
                self.log(f"❌ Error: {result.error_message}")
                results_detail.append(file_result)
                continue
            
            file_result['success'] = True
            file_result['points'] = {'total': result.total_points, 'machinery': result.machinery_points, 'ground': result.ground_points}
            
            # La inferencia base es un archivo intermedio — se elimina al final
            
            # ═══════════════════════════════════════════════════════════
            # LIMPIEZA PROFUNDA: GPU + RAM + torch.compile cache
            # Sin esto, ~25 GB de RSS quedan retenidos y OOM en postproc
            # ═══════════════════════════════════════════════════════════
            try:
                import torch
                import gc
                import ctypes

                # RSS antes de limpiar
                try:
                    import resource
                    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                    self.log(f"   📊 RAM RSS antes de limpiar: {rss_before:.0f} MB")
                except Exception:
                    pass

                if torch.cuda.is_available():
                    # Destruir motor de inferencia completo
                    self.inference_engine.model = None
                    self.inference_engine = None

                    # Limpiar cache de torch.compile (retiene grafos compilados en VRAM)
                    torch.compiler.reset()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

                    reserved = torch.cuda.memory_reserved(0) / 1024**3
                    self.log(f"   🧹 GPU liberada → VRAM reservada: {reserved:.2f} GB")
                    print(f"   🧹 GPU liberada → VRAM reservada: {reserved:.2f} GB", flush=True)

                # Forzar GC completo (3 generaciones)
                gc.collect(0)
                gc.collect(1)
                gc.collect(2)

                # Forzar que glibc devuelva memoria al kernel
                try:
                    libc = ctypes.CDLL("libc.so.6")
                    libc.malloc_trim(0)
                except Exception:
                    pass

                # RSS despues de limpiar
                try:
                    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                    self.log(f"   📊 RAM RSS despues de limpiar: {rss_after:.0f} MB")
                except Exception:
                    pass

                self.log(f"   🧹 Limpieza completa (GPU + RAM + torch.compile)")
                print(f"   🧹 Limpieza completa", flush=True)
            except Exception as e:
                self.log(f"   ⚠️ Error en limpieza: {e}")
                import traceback
                print(f"   ⚠️ Limpieza error: {traceback.format_exc()}", flush=True)
            
            current_file = inference_output

            # ═══════════════════════════════════════════════════════════
            # POSTPROCESAMIENTO EN PROCESO HIJO (spawn)
            # El heap de Python queda con ~41 GB RSS después de inferencia
            # de 213M puntos. gc.collect + malloc_trim NO pueden recuperar
            # esa memoria fragmentada. La única solución es ejecutar
            # FIX_TECHO e INTERPOL en un proceso NUEVO con memoria limpia.
            # ═══════════════════════════════════════════════════════════
            from dataclasses import asdict

            clasificado_output = os.path.join(output_dir_full, f"{base_name}{output_suffix}_Clasificado.laz")

            # PASO 2: FIX_TECHO (si se pide Clasificado o DTM)
            if export_clasificado or export_dtm:
                self.log("   🔄 Lanzando FIX_TECHO en proceso limpio...")
                print("   🔄 Lanzando FIX_TECHO en proceso limpio...", flush=True)

                pp_result = self._run_in_subprocess(
                    'fix_techo', current_file, clasificado_output,
                    asdict(self.postprocessor.fix_techo_config)
                )
                if pp_result['success']:
                    current_file = clasificado_output
                    if export_clasificado:
                        file_result['outputs'].append(f"{base_name}{output_suffix}_Clasificado.laz")
                        self.log(f"   💾 {base_name}{output_suffix}_Clasificado.laz")

            # PASO 3: INTERPOL DTM
            if export_dtm:
                dtm_output = os.path.join(output_dir_full, f"{base_name}{output_suffix}_DTM.laz")
                self.log("   🔄 Lanzando INTERPOL en proceso limpio...")
                print("   🔄 Lanzando INTERPOL en proceso limpio...", flush=True)

                pp_result = self._run_in_subprocess(
                    'interpol', current_file, dtm_output,
                    asdict(self.postprocessor.interpol_config)
                )
                if pp_result['success']:
                    file_result['outputs'].append(f"{base_name}{output_suffix}_DTM.laz")
                    self.log(f"   💾 {base_name}{output_suffix}_DTM.laz")

            # Limpiar intermedios:
            # 1. Inferencia base (siempre intermedio)
            if os.path.exists(inference_output):
                os.remove(inference_output)
            # 2. Clasificado si solo se usó como paso previo al DTM
            if not export_clasificado and os.path.exists(clasificado_output):
                os.remove(clasificado_output)
            
            file_result['time'] = (datetime.now() - file_start).total_seconds()
            self.log(f"   ⏱️ {file_result['time']:.1f}s")
            results_detail.append(file_result)
        
        total_time = (datetime.now() - start_time).total_seconds()
        successful = sum(1 for r in results_detail if r['success'])
        
        self.log(f"\n{'='*50}")
        self.log(f"✅ Completado en {total_time:.1f}s | {successful}/{len(valid_files)} archivos")
        
        # Generar HTML de resultados
        results_html = self._generate_results_html(results_detail, total_time, output_dir)
        return "\n".join(self.log_messages), results_html
    
    def _generate_results_html(self, results: List[Dict], total_time: float, output_dir: str) -> str:
        successful = sum(1 for r in results if r['success'])
        total_outputs = sum(len(r['outputs']) for r in results)
        
        html = f"""<div style="font-family: system-ui; padding: 15px;">
        <h2 style="color: #10b981;">✅ Completado</h2>
        <div style="display: flex; gap: 10px; margin: 15px 0;">
            <div style="background: #1e293b; padding: 12px 20px; border-radius: 8px; text-align: center;">
                <div style="font-size: 20px; font-weight: bold; color: #60a5fa;">{successful}/{len(results)}</div>
                <div style="color: #94a3b8; font-size: 11px;">Archivos</div>
            </div>
            <div style="background: #1e293b; padding: 12px 20px; border-radius: 8px; text-align: center;">
                <div style="font-size: 20px; font-weight: bold; color: #34d399;">{total_time:.1f}s</div>
                <div style="color: #94a3b8; font-size: 11px;">Tiempo</div>
            </div>
            <div style="background: #1e293b; padding: 12px 20px; border-radius: 8px; text-align: center;">
                <div style="font-size: 20px; font-weight: bold; color: #fbbf24;">{total_outputs}</div>
                <div style="color: #94a3b8; font-size: 11px;">Salidas</div>
            </div>
        </div>
        <div style="max-height: 250px; overflow-y: auto;">"""
        
        for r in results:
            icon = "✅" if r['success'] else "❌"
            color = "#10b981" if r['success'] else "#ef4444"
            html += f"""<div style="background: #334155; padding: 10px; border-radius: 6px; margin: 5px 0; border-left: 3px solid {color};">
                <b style="color: #f1f5f9;">{icon} {r['input']}</b>"""
            if r['success'] and r['outputs']:
                html += f"""<div style="font-size: 11px; color: #94a3b8; margin-top: 5px;">
                    💾 {', '.join(r['outputs'])}</div>"""
            html += "</div>"
        
        html += f"""</div>
        <div style="margin-top: 10px; padding: 8px; background: #1e3a5f; border-radius: 6px; color: #93c5fd; font-size: 12px;">
            📁 {output_dir}</div></div>"""
        return html


def create_app() -> gr.Blocks:
    app_instance = InferenceApp()

    def update_faena(faena):
        """Carga presets de parámetros para la faena seleccionada."""
        p = FAENA_PRESETS.get(faena, FAENA_PRESETS["Las Tórtolas"])
        return (
            gr.update(value=p["batch_size"]),
            gr.update(value=p["confidence"]),
            gr.update(value=p["eps"]),
            gr.update(value=p["z_buffer"]),
            gr.update(value=p["max_height"]),
            gr.update(value=p["padding"]),
            gr.update(value=p["smart_merge"]),
            gr.update(value=p["merge_radius"]),
            gr.update(value=p["merge_neighbors"]),
            gr.update(value=p["k_neighbors"]),
            gr.update(value=p["max_dist"]),
        )

    def update_checkpoints(version):
        """Actualiza el dropdown de checkpoints según la versión."""
        checkpoints = app_instance.get_checkpoints_for_version(version)
        default = app_instance.get_default_checkpoint(version)
        if checkpoints:
            return gr.update(choices=checkpoints, value=default)
        return gr.update(choices=[], value=None)
    
    with gr.Blocks(title="Point Cloud Inference V5/V6") as app:
        gr.HTML('''<div style="text-align: center; padding: 15px; background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%); border-radius: 10px; margin-bottom: 15px;">
            <h1 style="color: #60a5fa; margin: 0; font-size: 24px;">🚀 Point Cloud Inference</h1>
            <p style="color: #94a3b8; margin: 5px 0 0 0; font-size: 13px;">PointNet++ | V5 (0.10m) · V6 (0.25m) | 🏆 = Ganador</p>
        </div>''')
        
        with gr.Row():
            with gr.Column(scale=1):
                # Archivos
                gr.Markdown("#### 📁 Entrada")
                input_files = gr.File(label="Archivos LAZ/LAS (con RGB)", file_count="multiple", file_types=[".las", ".laz"])
                validate_btn = gr.Button("🔍 Validar", variant="secondary", size="sm")
                
                # Modelo
                gr.Markdown("#### 🧠 Modelo")
                model_version = gr.Radio(
                    choices=["V5 (0.10m)", "V6 (0.25m)"],
                    value="V5 (0.10m)",
                    label="Versión",
                    info="V5=10k pts | V6=2k pts"
                )
                
                initial_checkpoints = app_instance.get_checkpoints_for_version("V5 (0.10m)")
                default_ckpt = app_instance.get_default_checkpoint("V5 (0.10m)")
                
                checkpoint_dropdown = gr.Dropdown(
                    choices=initial_checkpoints,
                    value=default_ckpt,
                    label="Checkpoint",
                    info="🏆 = Ganador validado en producción"
                )
                
                # Salidas
                gr.Markdown("#### 💾 Salidas")
                output_types = gr.CheckboxGroup(
                    choices=["📊 Clasificado", "🌍 DTM"],
                    value=["📊 Clasificado", "🌍 DTM"],
                    label="Exportar"
                )
                output_dir = gr.Textbox(value="data/predictions/app_output", label="Directorio")

                # Selector de faena (presets rápidos)
                gr.Markdown("#### ⛏️ Faena")
                faena_selector = gr.Radio(
                    choices=["Las Tórtolas", "Spence"],
                    value="Las Tórtolas",
                    label="Seleccionar faena",
                    info="Carga parámetros optimizados para la faena"
                )

                # Parámetros
                with gr.Accordion("⚙️ Parámetros", open=False):
                    batch_size = gr.Slider(16, 256, value=256, step=16, label="Batch Size", info="V6 0.25m: hasta 256 | V5 0.10m: max 128")
                    use_compile = gr.Checkbox(value=True, label="torch.compile")
                    confidence = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Umbral de Confianza", info="Default: 0.5")
                    gr.Markdown("**FIX_TECHO:**")
                    eps = gr.Slider(1.0, 8.0, value=2.5, step=0.5, label="EPS", info="Radio clustering")
                    z_buffer = gr.Slider(0.1, 3.0, value=1.5, step=0.1, label="Z Buffer")
                    max_height = gr.Slider(5.0, 20.0, value=8.0, step=0.5, label="Max Height")
                    padding = gr.Slider(0.5, 5.0, value=1.5, step=0.5, label="Padding")
                    gr.Markdown("**SMART MERGE (Gap Filling):**")
                    smart_merge = gr.Checkbox(value=True, label="Activar Smart Merge", info="Rellena huecos entre fragmentos")
                    merge_radius = gr.Slider(1.0, 8.0, value=2.5, step=0.5, label="Radio Búsqueda (m)")
                    merge_neighbors = gr.Slider(1, 10, value=4, step=1, label="Min Vecinos Maquinaria")
                    gr.Markdown("**INTERPOL:**")
                    k_neighbors = gr.Slider(4, 24, value=12, step=2, label="K Vecinos")
                    max_dist = gr.Slider(10, 2000, value=50, step=10, label="Max Dist (m)")
                
                run_btn = gr.Button("🚀 Ejecutar", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                gr.Markdown("#### 📋 Log")
                log_output = gr.Textbox(label="", lines=18, max_lines=25, interactive=False)
                gr.Markdown("#### 📊 Resultados")
                results_output = gr.HTML("")
        
        # Eventos
        faena_selector.change(
            fn=update_faena,
            inputs=[faena_selector],
            outputs=[batch_size, confidence, eps, z_buffer, max_height, padding,
                     smart_merge, merge_radius, merge_neighbors, k_neighbors, max_dist]
        )
        model_version.change(fn=update_checkpoints, inputs=[model_version], outputs=[checkpoint_dropdown])
        validate_btn.click(fn=app_instance.validate_files, inputs=[input_files], outputs=[log_output])
        run_btn.click(
            fn=app_instance.run_pipeline,
            inputs=[input_files, checkpoint_dropdown, output_dir, model_version,
                    output_types, batch_size, use_compile,
                    eps, z_buffer, max_height, padding, 
                    smart_merge, merge_radius, merge_neighbors, # Inputs nuevos
                    k_neighbors, max_dist, confidence],
            outputs=[log_output, results_output]
        )
    
    return app


def launch_app(port: int = 7860, share: bool = False, prod_mode: bool = False):
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=port, share=share)


if __name__ == "__main__":
    launch_app()
