"""
Motor de Inferencia V5 - PointNet++ Nitro
==========================================
Wrapper del script de inferencia original para uso desde la UI.
Optimizado para RTX 5090 con FP16 y torch.compile.
"""

import os
import sys
import re
import torch
import numpy as np
import laspy
import open3d as o3d
from tqdm import tqdm
from typing import Callable, Optional, Dict, Any
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast
from dataclasses import dataclass
from datetime import datetime

# Agregar path del proyecto para importar modelos
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.pointnet2 import PointNet2


@dataclass
class InferenceConfig:
    """Configuración para la inferencia."""
    batch_size: int = 64
    block_size: float = 10.0
    num_points: int = 10000
    use_compile: bool = True
    device: str = "cuda"
    num_workers: int = 12


@dataclass
class InferenceResult:
    """Resultado de la inferencia."""
    success: bool
    input_file: str
    output_file: str
    total_points: int = 0
    machinery_points: int = 0
    ground_points: int = 0
    processing_time: float = 0.0
    error_message: str = ""


class GridDatasetNitro(Dataset):
    """
    Dataset optimizado para acceso por bloques de grid.
    Diseñado para máximo rendimiento en GPU.
    """
    
    def __init__(self, full_data: np.ndarray, grid_dict: Dict, 
                 num_points: int, min_coord: np.ndarray, block_size: float):
        self.full_data = full_data
        self.grid_keys = list(grid_dict.keys())
        self.grid_dict = grid_dict
        self.num_points = num_points
        self.min_coord = min_coord
        self.block_size = block_size
        
    def __len__(self) -> int:
        return len(self.grid_keys)
    
    def __getitem__(self, idx: int):
        key = self.grid_keys[idx]
        indices = self.grid_dict[key]
        
        n_idx = len(indices)
        if n_idx >= self.num_points:
            sel = np.random.choice(n_idx, self.num_points, replace=False)
        else:
            sel = np.random.choice(n_idx, self.num_points, replace=True)
        selected_indices = indices[sel]
        
        block_data = self.full_data[selected_indices].copy()
        
        # Normalización
        xyz = block_data[:, :3]
        
        # Centro del tile
        ix = key // 100000
        iy = key % 100000
        tile_origin_x = self.min_coord[0] + ix * self.block_size
        tile_origin_y = self.min_coord[1] + iy * self.block_size
        
        block_data[:, 0] -= (tile_origin_x + self.block_size / 2.0)
        block_data[:, 1] -= (tile_origin_y + self.block_size / 2.0)
        block_data[:, 2] -= np.min(xyz[:, 2])
        
        return torch.from_numpy(block_data).float(), selected_indices


class InferenceEngine:
    """
    Motor de inferencia principal para PointNet++ V5.
    
    Optimizaciones:
    - Lectura directa de normales del LAS si existen
    - Gridding vectorizado con NumPy
    - FP16 con torch.amp
    - torch.compile opcional
    """
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        """
        Args:
            config: Configuración de inferencia
        """
        self.config = config or InferenceConfig()
        self.model = None
        self.device = torch.device(self.config.device)
        
        # Optimizaciones para RTX
        torch.set_float32_matmul_precision('high')
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        
    def _log_gpu_memory(self, stage: str, callback: Optional[Callable] = None):
        """Log del uso de memoria GPU para debugging de OOM"""
        if not torch.cuda.is_available():
            return

        try:
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            free = total - reserved

            msg = (f"   🔍 [{stage}] GPU Memory: "
                   f"Usada={allocated:.2f}GB, Reservada={reserved:.2f}GB, "
                   f"Libre={free:.2f}GB, Total={total:.2f}GB")

            # Detectar VRAM spillover a RAM del sistema (unified memory)
            if reserved > total * 1.1:
                spillover = reserved - total
                msg += (f"\n   🚨 PELIGRO: VRAM spillover detectado! "
                        f"{spillover:.1f} GB derramados a RAM del sistema. "
                        f"Esto causa OOM en postprocesamiento.")

            if callback:
                callback(msg)
            print(msg, flush=True)
        except Exception as e:
            if callback:
                callback(f"   ⚠️ Error leyendo memoria GPU: {e}")
    
    def load_model(self, checkpoint_path: str, 
                   progress_callback: Optional[Callable] = None) -> bool:
        """
        Carga el modelo desde un checkpoint.
        
        Args:
            checkpoint_path: Ruta al archivo .pth
            progress_callback: Función para reportar progreso
            
        Returns:
            True si se cargó correctamente
        """
        try:
            if progress_callback:
                progress_callback(f"🏗️ Cargando pesos neuronas (Config: {self.config.num_points} pts/bloque)...")
            
            self._log_gpu_memory("Antes de cargar modelo", progress_callback)
                
            # Extraer radio del nombre del checkpoint
            match = re.search(r'_R(\d+\.\d+)_', checkpoint_path)
            self.radius = float(match.group(1)) if match else 3.5
            
            # Crear modelo
            self.model = PointNet2(d_in=9, num_classes=2, base_radius=self.radius)
            self.model = self.model.to(self.device)
            
            # Cargar pesos
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state_dict = ckpt.get('model_state_dict', ckpt)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            # Compilar si está habilitado
            if self.config.use_compile:
                if progress_callback:
                    progress_callback("🔥 Compilando modelo (30-60s en primer uso)...")
                try:
                    self.model = torch.compile(self.model)
                except Exception as e:
                    if progress_callback:
                        progress_callback(f"⚠️ torch.compile falló: {e}. Usando modo eager.")
            
            self._log_gpu_memory("Después de cargar modelo", progress_callback)
            
            if progress_callback:
                progress_callback("✅ Modelo cargado correctamente")
                
            return True
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            if progress_callback:
                progress_callback(f"❌ Error cargando modelo: {str(e)}")
                progress_callback(f"📋 Traceback completo:\n{error_detail}")
            print(f"\n❌ ERROR CARGANDO MODELO:\n{error_detail}")
            return False
    
    def _compute_features(self, las, 
                          progress_callback: Optional[Callable] = None) -> tuple:
        """
        Extrae features de la nube de puntos.
        Optimizado para leer normales directamente del LAS si existen.
        """
        if progress_callback:
            progress_callback("🚀 Extrayendo puntos y features...")
        
        self._log_gpu_memory("Antes de extraer features", progress_callback)
            
        xyz = np.vstack((las.x, las.y, las.z)).transpose()
        
        if len(xyz) == 0:
            raise ValueError("El archivo LAS está vacío (0 puntos)")
        
        if progress_callback:
            progress_callback(f"   📊 Total de puntos: {len(xyz):,}")
            
        # RGB
        if hasattr(las, 'red'):
            if progress_callback:
                progress_callback("   🎨 Procesando RGB...")
            scale = 65535.0 if np.max(las.red) > 255 else 255.0
            rgb = np.vstack((las.red, las.green, las.blue)).transpose() / scale
        else:
            rgb = np.full_like(xyz, 0.5)
            
        # Normales
        has_normals = False
        normals = None
        
        if hasattr(las, 'normal_x'):
            if progress_callback:
                progress_callback("   ⚡ Usando normales nativas (ultra rápido)...")
            nx = np.array(las.normal_x)
            ny = np.array(las.normal_y)
            nz = np.array(las.normal_z)
            normals = np.vstack((nx, ny, nz)).transpose()
            has_normals = True
            
        if not has_normals:
            msg = (f"   🧮 Calculando normales en chunks espaciales "
                   f"(r={self.radius}m, ~50m x 50m por chunk)...")
            if progress_callback:
                progress_callback(msg)
            print(msg)
            
            from src.utils.geometry import compute_normals_gpu
            normals = compute_normals_gpu(xyz, k=30, radius=self.radius)
            
        return xyz, rgb, normals
    
    def run_inference(self, input_file: str, output_file: str,
                      progress_callback: Optional[Callable] = None,
                      confidence: float = 0.5) -> InferenceResult:
        """
        Ejecuta la inferencia completa sobre un archivo.
        
        Args:
            input_file: Ruta al archivo LAS/LAZ de entrada
            output_file: Ruta donde guardar el resultado
            progress_callback: Función para reportar progreso
            confidence: Umbral de confianza para clasificación (0.0 - 1.0)
            
        Returns:
            InferenceResult con el resultado
        """
        start_time = datetime.now()
        result = InferenceResult(
            success=False,
            input_file=input_file,
            output_file=output_file
        )
        
        try:
            if self.model is None:
                raise ValueError("Modelo no cargado. Llama a load_model() primero.")
            
            print(f"\n{'='*70}")
            print(f"🎯 INICIANDO INFERENCIA: {os.path.basename(input_file)}")
            print(f"{'='*70}")
            
            self._log_gpu_memory("Inicio de inferencia", progress_callback)
                
            if progress_callback:
                progress_callback(f"📂 Leyendo: {os.path.basename(input_file)}...")
            
            # Verificar tamaño del archivo
            file_size_mb = os.path.getsize(input_file) / 1024**2
            if progress_callback:
                progress_callback(f"   📦 Tamaño del archivo: {file_size_mb:.2f} MB")
            print(f"   📦 Tamaño del archivo: {file_size_mb:.2f} MB")
                
            # Leer archivo
            las = laspy.read(input_file)
            
            # Extraer features
            xyz, rgb, normals = self._compute_features(las, progress_callback)
            
            # Concatenar en memoria continua
            full_data = np.hstack([xyz, rgb, normals]).astype(np.float32)
            result.total_points = len(full_data)
            data_size_mb = full_data.nbytes / 1024**2
            
            if progress_callback:
                progress_callback(f"   💾 Array de features: {data_size_mb:.2f} MB en RAM")
            print(f"   💾 Array de features: {data_size_mb:.2f} MB en RAM")
            
            # Liberar memoria
            del las, xyz, rgb, normals
            
            self._log_gpu_memory("Después de extraer features", progress_callback)
            
            # Gridding
            if progress_callback:
                progress_callback(f"📦 Dividiendo en bloques ({self.config.block_size}m)...")
                
            min_coord = np.min(full_data[:, :3], axis=0)
            grid_x = ((full_data[:, 0] - min_coord[0]) // self.config.block_size).astype(np.int32)
            grid_y = ((full_data[:, 1] - min_coord[1]) // self.config.block_size).astype(np.int32)
            grid_hashes = grid_x * 100000 + grid_y
            
            sort_idx = np.argsort(grid_hashes)
            sorted_hashes = grid_hashes[sort_idx]
            unique_hashes, split_indices = np.unique(sorted_hashes, return_index=True)
            grouped_indices = np.split(sort_idx, split_indices[1:])
            grid_dict = {h: idx for h, idx in zip(unique_hashes, grouped_indices) if len(idx) > 50}
            
            if progress_callback:
                progress_callback(f"   → {len(grid_dict)} bloques activos")
            print(f"   → {len(grid_dict)} bloques activos")
            
            # DataLoader
            if progress_callback:
                progress_callback(f"⚙️ Configurando DataLoader (batch_size={self.config.batch_size}, workers={self.config.num_workers})...")
            print(f"⚙️ Configurando DataLoader (batch_size={self.config.batch_size}, workers={self.config.num_workers})...", flush=True)

            dataset = GridDatasetNitro(
                full_data, grid_dict, self.config.num_points,
                min_coord, self.config.block_size
            )
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                shuffle=False,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2
            )
            
            self._log_gpu_memory("Antes de inferencia GPU", progress_callback)
            
            # Inferencia
            global_probs = np.zeros(len(full_data), dtype=np.float16)
            
            if progress_callback:
                progress_callback("🧠 Ejecutando inferencia en GPU...")
            print("🧠 Ejecutando inferencia en GPU...")
                
            total_batches = len(dataloader)
            print(f"   Total de batches: {total_batches}")
            
            with torch.no_grad():
                for batch_idx, (batch_data, batch_indices) in enumerate(dataloader):
                    batch_data = batch_data.to(self.device, non_blocking=True)
                    xyz_tensor = batch_data[:, :, :3]
                    
                    with autocast(device_type='cuda'):
                        logits = self.model(xyz_tensor, batch_data)
                        probs = torch.softmax(logits, dim=1)[:, 1, :]
                        
                    probs_np = probs.cpu().numpy().flatten()
                    indices_np = batch_indices.numpy().flatten()
                    global_probs[indices_np] = probs_np
                    
                    # Log cada 10 batches o en el primer batch
                    if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                        pct = (batch_idx + 1) / total_batches * 100
                        msg = f"   → Batch {batch_idx + 1}/{total_batches} ({pct:.1f}%)"
                        if progress_callback:
                            progress_callback(msg)
                        print(msg)
                        
                        # Log de memoria cada 50 batches
                        if (batch_idx + 1) % 50 == 0:
                            self._log_gpu_memory(f"Batch {batch_idx + 1}", progress_callback)
            
            self._log_gpu_memory("Después de inferencia GPU", progress_callback)

            # ═══════════════════════════════════════════════════════════════
            # LIMPIEZA AGRESIVA DE MEMORIA — Crítico para evitar OOM en
            # postprocesamiento. persistent_workers + pin_memory retienen
            # GB de RAM compartida que glibc NO devuelve al OS.
            # ═══════════════════════════════════════════════════════════════
            import gc
            import ctypes

            # 1) Matar DataLoader workers EXPLICITAMENTE
            #    del dataloader no basta con persistent_workers=True
            if hasattr(dataloader, '_iterator') and dataloader._iterator is not None:
                dataloader._iterator._shutdown_workers()
            del dataloader

            # 2) Romper referencia circular Dataset → full_data
            dataset.full_data = None
            dataset.grid_dict = None
            del dataset, grid_dict, sort_idx, sorted_hashes
            del unique_hashes, split_indices, grouped_indices
            del grid_x, grid_y, grid_hashes, min_coord
            gc.collect()

            # Guardar
            if progress_callback:
                progress_callback(f"💾 Guardando resultado (Confianza: {confidence})...")

            preds = (global_probs > confidence).astype(np.uint8)
            result.machinery_points = int(np.sum(preds == 1))
            result.ground_points = int(np.sum(preds == 0))

            # 3) Liberar full_data y global_probs (~9 GB combinados)
            del full_data, global_probs
            gc.collect()

            # Re-leer para guardar
            las_in = laspy.read(input_file)
            new_las = laspy.create(
                point_format=las_in.header.point_format,
                file_version=las_in.header.version
            )
            new_las.header = las_in.header
            new_las.x = las_in.x
            new_las.y = las_in.y
            new_las.z = las_in.z

            if hasattr(las_in, 'red'):
                new_las.red = las_in.red
                new_las.green = las_in.green
                new_las.blue = las_in.blue

            # Clase 1 = Maquinaria, Clase 2 = Suelo
            new_las.classification = np.where(preds == 1, 1, 2).astype(np.uint8)

            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            new_las.write(output_file)

            # 4) Liberar todo y forzar devolución al OS
            del las_in, new_las, preds
            gc.collect()
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass

            result.success = True
            result.processing_time = (datetime.now() - start_time).total_seconds()

            self._log_gpu_memory("Final de inferencia", progress_callback)

            success_msg = (
                f"✅ Inferencia completada en {result.processing_time:.1f}s - "
                f"Maquinaria: {result.machinery_points:,} puntos ({result.machinery_points/result.total_points*100:.1f}%)"
            )

            if progress_callback:
                progress_callback(success_msg)
            print(success_msg)
            print(f"{'='*70}\n")
                
        except Exception as e:
            import traceback
            import sys
            
            result.error_message = str(e)
            error_detail = traceback.format_exc()
            
            # Log detallado del error
            error_msg = f"\n{'='*70}\n❌ ERROR EN INFERENCIA\n{'='*70}\n"
            error_msg += f"Archivo: {input_file}\n"
            error_msg += f"Error: {str(e)}\n"
            error_msg += f"\n📋 Traceback completo:\n{error_detail}"
            error_msg += f"{'='*70}\n"
            
            print(error_msg, file=sys.stderr)
            
            if progress_callback:
                progress_callback(f"❌ Error: {str(e)}")
                progress_callback(f"📋 Ver terminal para traceback completo")
            
            # Intentar liberar memoria GPU
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    self._log_gpu_memory("Después de error (post-limpieza)", progress_callback)
            except:
                pass
                
        return result
