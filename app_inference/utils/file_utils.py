"""
Utilidades de Archivos
=======================
Funciones para manejo de archivos LAS/LAZ.
"""

import os
from typing import List, Dict, Optional
from datetime import datetime


def find_las_files(path: str, recursive: bool = False) -> List[str]:
    """
    Busca archivos LAS/LAZ en una ruta.
    
    Args:
        path: Ruta a archivo o carpeta
        recursive: Si buscar recursivamente en subcarpetas
        
    Returns:
        Lista de rutas absolutas a archivos LAS/LAZ
    """
    files = []
    
    if os.path.isfile(path):
        if path.lower().endswith(('.las', '.laz')):
            files.append(os.path.abspath(path))
    elif os.path.isdir(path):
        if recursive:
            for root, dirs, filenames in os.walk(path):
                for f in filenames:
                    if f.lower().endswith(('.las', '.laz')):
                        files.append(os.path.join(root, f))
        else:
            for f in os.listdir(path):
                if f.lower().endswith(('.las', '.laz')):
                    files.append(os.path.join(path, f))
                    
    return sorted(files)


def ensure_dir(path: str) -> str:
    """
    Crea un directorio si no existe.
    
    Args:
        path: Ruta del directorio
        
    Returns:
        Ruta absoluta del directorio
    """
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path


def get_file_info(file_path: str) -> Dict:
    """
    Obtiene información básica de un archivo.
    
    Args:
        file_path: Ruta al archivo
        
    Returns:
        Dict con información del archivo
    """
    if not os.path.exists(file_path):
        return {
            'exists': False,
            'path': file_path
        }
        
    stat = os.stat(file_path)
    
    return {
        'exists': True,
        'path': os.path.abspath(file_path),
        'name': os.path.basename(file_path),
        'extension': os.path.splitext(file_path)[1].lower(),
        'size_bytes': stat.st_size,
        'size_mb': stat.st_size / (1024 * 1024),
        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
        'created': datetime.fromtimestamp(stat.st_ctime).isoformat()
    }


def generate_output_path(input_path: str, output_dir: str, suffix: str = "") -> str:
    """
    Genera la ruta de salida basada en la entrada.
    
    Args:
        input_path: Ruta del archivo de entrada
        output_dir: Directorio de salida
        suffix: Sufijo a agregar al nombre
        
    Returns:
        Ruta completa del archivo de salida
    """
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    ext = os.path.splitext(input_path)[1]
    
    output_name = f"{base_name}{suffix}{ext}"
    return os.path.join(output_dir, output_name)


def get_relative_path(file_path: str, base_path: str) -> str:
    """
    Obtiene la ruta relativa respecto a una base.
    
    Args:
        file_path: Ruta del archivo
        base_path: Ruta base
        
    Returns:
        Ruta relativa
    """
    try:
        return os.path.relpath(file_path, base_path)
    except ValueError:
        return file_path


def format_file_size(size_bytes: int) -> str:
    """
    Formatea un tamaño en bytes a formato legible.
    
    Args:
        size_bytes: Tamaño en bytes
        
    Returns:
        String formateado (ej: "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"
