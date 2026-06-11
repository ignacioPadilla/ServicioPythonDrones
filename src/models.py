"""
Modelos Pydantic para validación de mensajes MQTT
"""
from pydantic import BaseModel
from typing import List


class BBox(BaseModel):
    """Bounding box con coordenadas y dimensiones"""
    left: float
    top: float
    width: float
    height: float


class Detection(BaseModel):
    """Detección de objeto único"""
    id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: BBox


class Frame(BaseModel):
    """Información de imagen en base64"""
    image_type: str
    width: int
    height: int
    data: str


class DroneEvent(BaseModel):
    """Evento completo recibido desde drone/sensor"""
    timestamp: float
    event: str
    type: str
    detections: List[Detection]
    frame: Frame

    model_config = {"extra": "ignore"}
