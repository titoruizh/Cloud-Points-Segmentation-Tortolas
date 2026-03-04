"""
Sistema de Logging
===================
Configuración de logs con rotación y formateo.
"""

import os
import logging
from datetime import datetime
from typing import Optional


# Directorio de logs
LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs", "logs"
)


def get_log_path(prefix: str = "inference") -> str:
    """
    Genera la ruta para un nuevo archivo de log.
    
    Args:
        prefix: Prefijo para el nombre del archivo
        
    Returns:
        Ruta completa del archivo de log
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(LOG_DIR, f"{prefix}_{timestamp}.log")


def setup_logger(
    name: str = "inference_app",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    """
    Configura y retorna un logger.
    
    Args:
        name: Nombre del logger
        log_file: Ruta al archivo de log (opcional)
        level: Nivel de logging
        console: Si mostrar logs en consola
        
    Returns:
        Logger configurado
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Evitar duplicados
    if logger.handlers:
        return logger
        
    # Formato
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler de consola
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    # Handler de archivo
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger


class LogCollector:
    """
    Clase para recolectar mensajes de log para la UI.
    Mantiene un buffer de mensajes recientes.
    """
    
    def __init__(self, max_messages: int = 100):
        """
        Args:
            max_messages: Máximo de mensajes a mantener en buffer
        """
        self.messages = []
        self.max_messages = max_messages
        
    def log(self, message: str, level: str = "INFO") -> str:
        """
        Agrega un mensaje al buffer.
        
        Args:
            message: Mensaje a agregar
            level: Nivel (INFO, WARNING, ERROR, etc.)
            
        Returns:
            El log completo formateado
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Iconos por nivel
        icons = {
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "SUCCESS": "✅",
            "PROGRESS": "🔄"
        }
        icon = icons.get(level.upper(), "•")
        
        formatted = f"[{timestamp}] {icon} {message}"
        self.messages.append(formatted)
        
        # Limitar buffer
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
            
        return self.get_log()
    
    def get_log(self) -> str:
        """
        Retorna el log completo.
        
        Returns:
            String con todos los mensajes
        """
        return "\n".join(self.messages)
    
    def clear(self):
        """Limpia el buffer de mensajes."""
        self.messages = []
        
    def info(self, message: str) -> str:
        """Atajo para log INFO."""
        return self.log(message, "INFO")
    
    def warning(self, message: str) -> str:
        """Atajo para log WARNING."""
        return self.log(message, "WARNING")
    
    def error(self, message: str) -> str:
        """Atajo para log ERROR."""
        return self.log(message, "ERROR")
    
    def success(self, message: str) -> str:
        """Atajo para log SUCCESS."""
        return self.log(message, "SUCCESS")
    
    def progress(self, message: str) -> str:
        """Atajo para log PROGRESS."""
        return self.log(message, "PROGRESS")
