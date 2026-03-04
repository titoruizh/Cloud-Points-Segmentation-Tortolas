"""
Validadores de Nubes de Puntos
==============================
Verifica que los archivos LAS/LAZ cumplan con los requisitos
para el modelo PointNet++ V5.
"""

import os
import laspy
import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Resultado de la validación de un archivo."""
    is_valid: bool
    file_path: str
    file_name: str
    point_count: int = 0
    has_rgb: bool = False
    has_normals: bool = False
    rgb_range: str = ""
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class PointCloudValidator:
    """
    Validador de nubes de puntos para inferencia con PointNet++ V5.
    
    Verifica:
    - Formato LAZ/LAS válido
    - Presencia de RGB (obligatorio para V5)
    - Cantidad mínima de puntos
    - Presencia de normales (opcional pero recomendado)
    """
    
    def __init__(self, 
                 require_rgb: bool = True,
                 min_points: int = 1000,
                 warn_no_normals: bool = True):
        """
        Args:
            require_rgb: Si True, rechaza archivos sin RGB
            min_points: Cantidad mínima de puntos requeridos
            warn_no_normals: Si True, advierte cuando no hay normales pre-calculadas
        """
        self.require_rgb = require_rgb
        self.min_points = min_points
        self.warn_no_normals = warn_no_normals
        
    def validate_file(self, file_path: str) -> ValidationResult:
        """
        Valida un archivo LAS/LAZ individual.
        
        Args:
            file_path: Ruta al archivo a validar
            
        Returns:
            ValidationResult con el resultado de la validación
        """
        file_name = os.path.basename(file_path)
        result = ValidationResult(
            is_valid=False,
            file_path=file_path,
            file_name=file_name
        )
        
        # 1. Verificar que el archivo existe
        if not os.path.exists(file_path):
            result.errors.append(f"❌ El archivo no existe: {file_path}")
            return result
            
        # 2. Verificar extensión
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in ['.las', '.laz']:
            result.errors.append(f"❌ Formato no soportado: {ext}. Use .las o .laz")
            return result
            
        # 3. Intentar leer el archivo
        try:
            las = laspy.read(file_path)
        except Exception as e:
            result.errors.append(f"❌ Error leyendo archivo: {str(e)}")
            return result
            
        # 4. Verificar cantidad de puntos
        result.point_count = len(las.x)
        if result.point_count < self.min_points:
            result.errors.append(
                f"❌ Muy pocos puntos: {result.point_count:,} (mínimo: {self.min_points:,})"
            )
            return result
            
        # 5. Verificar RGB
        result.has_rgb = hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue')
        
        if result.has_rgb:
            # Determinar rango de valores RGB
            max_val = max(np.max(las.red), np.max(las.green), np.max(las.blue))
            result.rgb_range = "16-bit" if max_val > 255 else "8-bit"
        else:
            if self.require_rgb:
                result.errors.append(
                    "❌ El archivo NO tiene canales RGB. "
                    "El modelo V5 requiere RGB para la inferencia. "
                    "No se procesará este archivo."
                )
                return result
                
        # 6. Verificar normales (opcional)
        result.has_normals = hasattr(las, 'normal_x') or hasattr(las, 'vl_x')
        
        if not result.has_normals and self.warn_no_normals:
            result.warnings.append(
                "⚠️ El archivo no tiene normales pre-calculadas. "
                "Se calcularán con Open3D (más lento pero funciona)."
            )
            
        # Si llegamos aquí, el archivo es válido
        result.is_valid = True
        
        return result
    
    def validate_folder(self, folder_path: str) -> Tuple[List[ValidationResult], Dict]:
        """
        Valida todos los archivos LAS/LAZ en una carpeta.
        
        Args:
            folder_path: Ruta a la carpeta
            
        Returns:
            Tupla con lista de resultados y resumen estadístico
        """
        results = []
        
        if not os.path.isdir(folder_path):
            empty_result = ValidationResult(
                is_valid=False,
                file_path=folder_path,
                file_name="",
                errors=[f"❌ No es una carpeta válida: {folder_path}"]
            )
            return [empty_result], {"total": 0, "valid": 0, "invalid": 0}
            
        # Buscar archivos LAS/LAZ
        files = []
        for f in os.listdir(folder_path):
            if f.lower().endswith(('.las', '.laz')):
                files.append(os.path.join(folder_path, f))
                
        if not files:
            empty_result = ValidationResult(
                is_valid=False,
                file_path=folder_path,
                file_name="",
                errors=["❌ No se encontraron archivos .las o .laz en la carpeta"]
            )
            return [empty_result], {"total": 0, "valid": 0, "invalid": 0}
            
        # Validar cada archivo
        for file_path in sorted(files):
            result = self.validate_file(file_path)
            results.append(result)
            
        # Resumen
        summary = {
            "total": len(results),
            "valid": sum(1 for r in results if r.is_valid),
            "invalid": sum(1 for r in results if not r.is_valid),
            "with_normals": sum(1 for r in results if r.has_normals),
            "total_points": sum(r.point_count for r in results if r.is_valid)
        }
        
        return results, summary
    
    def format_validation_report(self, results: List[ValidationResult], 
                                  summary: Optional[Dict] = None) -> str:
        """
        Genera un reporte formateado de la validación.
        
        Args:
            results: Lista de resultados de validación
            summary: Diccionario con resumen (opcional)
            
        Returns:
            String con el reporte formateado
        """
        lines = ["# 📋 Reporte de Validación\n"]
        
        if summary:
            lines.append(f"## Resumen")
            lines.append(f"- **Total archivos:** {summary['total']}")
            lines.append(f"- **Válidos:** {summary['valid']} ✅")
            lines.append(f"- **Inválidos:** {summary['invalid']} ❌")
            if 'total_points' in summary:
                lines.append(f"- **Puntos totales:** {summary['total_points']:,}")
            lines.append("")
            
        lines.append("## Detalle por Archivo\n")
        
        for result in results:
            status = "✅" if result.is_valid else "❌"
            lines.append(f"### {status} {result.file_name}")
            lines.append(f"- **Ruta:** `{result.file_path}`")
            lines.append(f"- **Puntos:** {result.point_count:,}")
            lines.append(f"- **RGB:** {'Sí' if result.has_rgb else 'No'} ({result.rgb_range})")
            lines.append(f"- **Normales:** {'Sí (rápido)' if result.has_normals else 'No (se calcularán)'}")
            
            if result.errors:
                lines.append("\n**Errores:**")
                for err in result.errors:
                    lines.append(f"  - {err}")
                    
            if result.warnings:
                lines.append("\n**Advertencias:**")
                for warn in result.warnings:
                    lines.append(f"  - {warn}")
                    
            lines.append("")
            
        return "\n".join(lines)
