"""
Servicio MQTT de Procesamiento de Imágenes con YOLOE
"""
__version__ = "1.0.0"

from .main import MQTTProcessingService, main

__all__ = ["MQTTProcessingService", "main"]
