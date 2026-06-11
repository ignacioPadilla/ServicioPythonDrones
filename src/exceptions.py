"""
Excepciones custom para el servicio MQTT
"""


class MQTTConnectionError(Exception):
    """Error al conectar/desconectar del broker MQTT"""
    pass


class MQTTPublishError(Exception):
    """Error al publicar mensaje en MQTT"""
    pass


class ModelLoadError(Exception):
    """Error al descargar o cargar modelos YOLOE"""
    pass


class ImageProcessingError(Exception):
    """Error al procesar imagen (decodificar, inferencia, etc)"""
    pass


class ValidationError(Exception):
    """Error al validar mensaje con Pydantic"""
    pass


class ConfigurationError(Exception):
    """Error en configuración"""
    pass


class DeviceError(Exception):
    """Error al determinar device (GPU/CPU)"""
    pass
