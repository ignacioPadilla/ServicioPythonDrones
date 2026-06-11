"""
Configuración de logging
"""
import logging
import sys
from typing import Optional


_logger: Optional[logging.Logger] = None


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Obtener logger singleton con formato configurado
    
    Args:
        name: Nombre del logger (típicamente __name__)
        level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Logger configurado
    """
    global _logger
    
    if _logger is None:
        _logger = logging.getLogger(name)
        _logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        
        # Handler para consola
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        
        # Formato
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        _logger.addHandler(handler)
    
    return _logger
