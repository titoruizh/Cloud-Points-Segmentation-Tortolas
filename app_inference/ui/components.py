"""
Componentes UI Reutilizables
=============================
Componentes modulares para la interfaz de Gradio.
"""

import gradio as gr
from typing import List, Optional


def create_header() -> str:
    """
    Crea el HTML del header de la aplicación.
    
    Returns:
        HTML string
    """
    return """
    <div class="app-header">
        <h1>🚀 Point Cloud Inference V5</h1>
        <p>PointNet++ "Geometric Purification" | Optimizado para RTX 5090</p>
    </div>
    """


def create_file_input() -> gr.File:
    """
    Crea el componente de subida de archivos.
    
    Returns:
        Componente gr.File configurado
    """
    return gr.File(
        label="📁 Archivos de Entrada",
        file_count="multiple",
        file_types=[".las", ".laz"],
        elem_classes=["file-upload-area"]
    )


def create_output_panel() -> gr.Markdown:
    """
    Crea el panel de salida/log.
    
    Returns:
        Componente gr.Markdown para logs
    """
    return gr.Markdown(
        value="*Esperando archivos...*",
        elem_classes=["progress-log"]
    )


def create_stats_html(stats: dict) -> str:
    """
    Genera HTML para las estadísticas de procesamiento.
    
    Args:
        stats: Diccionario con estadísticas
        
    Returns:
        HTML string
    """
    return f"""
    <div class="stats-grid">
        <div class="stat-card">
            <div class="value">{stats.get('total_files', 0)}</div>
            <div class="label">Archivos</div>
        </div>
        <div class="stat-card">
            <div class="value">{stats.get('total_points', 0):,}</div>
            <div class="label">Puntos Totales</div>
        </div>
        <div class="stat-card">
            <div class="value">{stats.get('machinery_points', 0):,}</div>
            <div class="label">🚜 Maquinaria</div>
        </div>
        <div class="stat-card">
            <div class="value">{stats.get('processing_time', 0):.1f}s</div>
            <div class="label">⏱️ Tiempo</div>
        </div>
    </div>
    """


def create_pipeline_status(steps: List[dict]) -> str:
    """
    Genera HTML para el estado del pipeline.
    
    Args:
        steps: Lista de pasos con estado
        
    Returns:
        HTML string
    """
    html = ""
    for step in steps:
        status_class = step.get('status', '')  # active, completed, error
        icon = step.get('icon', '⚪')
        name = step.get('name', '')
        
        html += f"""
        <div class="pipeline-step {status_class}">
            <span>{icon}</span>
            <span>{name}</span>
        </div>
        """
    return html


def create_validation_report_html(results: List, summary: dict) -> str:
    """
    Genera HTML para el reporte de validación.
    
    Args:
        results: Lista de resultados de validación
        summary: Resumen estadístico
        
    Returns:
        HTML string
    """
    valid_count = summary.get('valid', 0)
    invalid_count = summary.get('invalid', 0)
    
    html = f"""
    <div style="margin-bottom: 16px;">
        <span class="status-badge status-success">✅ {valid_count} válidos</span>
        <span class="status-badge status-error" style="margin-left: 8px;">❌ {invalid_count} inválidos</span>
    </div>
    """
    
    for result in results:
        status_class = "status-success" if result.is_valid else "status-error"
        status_icon = "✅" if result.is_valid else "❌"
        
        html += f"""
        <div class="config-card" style="margin-bottom: 12px;">
            <h3>{status_icon} {result.file_name}</h3>
            <p style="color: var(--text-secondary); margin: 8px 0;">
                📊 {result.point_count:,} puntos | 
                🎨 RGB: {'Sí' if result.has_rgb else 'No'} |
                📐 Normales: {'Sí' if result.has_normals else 'No'}
            </p>
        """
        
        if result.errors:
            html += "<div style='color: var(--error-color);'>"
            for err in result.errors:
                html += f"<p>{err}</p>"
            html += "</div>"
            
        if result.warnings:
            html += "<div style='color: var(--warning-color);'>"
            for warn in result.warnings:
                html += f"<p>{warn}</p>"
            html += "</div>"
            
        html += "</div>"
        
    return html


def create_results_summary(results: dict) -> str:
    """
    Genera HTML con el resumen final de resultados.
    
    Args:
        results: Diccionario con resultados del procesamiento
        
    Returns:
        HTML string
    """
    success = results.get('success', False)
    status_class = "status-success" if success else "status-error"
    status_text = "Completado" if success else "Error"
    
    html = f"""
    <div class="config-card">
        <h3>📋 Resumen del Procesamiento</h3>
        <p><span class="status-badge {status_class}">{status_text}</span></p>
        
        <div class="stats-grid" style="margin-top: 16px;">
            <div class="stat-card">
                <div class="value">{results.get('files_processed', 0)}</div>
                <div class="label">Archivos Procesados</div>
            </div>
            <div class="stat-card">
                <div class="value">{results.get('total_time', 0):.1f}s</div>
                <div class="label">Tiempo Total</div>
            </div>
        </div>
    """
    
    if results.get('output_files'):
        html += "<h4 style='margin-top: 16px;'>📁 Archivos Generados:</h4><ul>"
        for f in results['output_files']:
            html += f"<li><code>{f}</code></li>"
        html += "</ul>"
        
    html += "</div>"
    return html
