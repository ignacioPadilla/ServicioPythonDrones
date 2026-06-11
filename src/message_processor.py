"""
Orquestación del procesamiento de mensajes MQTT
"""
from datetime import datetime
from pathlib import Path
import json
from typing import Tuple, Optional

from .models import DroneEvent
from .config import get_config
from .image_processor import get_image_processor
from .iou_calculator import calculate_iou
from .logger_config import get_logger
from .exceptions import ImageProcessingError, ValidationError

logger = get_logger(__name__)


class MessageProcessor:
    """Procesa mensajes MQTT: validación, procesamiento, modificación"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MessageProcessor, cls).__new__(cls)
        return cls._instance
    
    def process_message(self, payload: str) -> Tuple[Optional[DroneEvent], bool]:
        """
        Procesa un mensaje MQTT completo
        
        Args:
            payload: String JSON del mensaje MQTT
        
        Returns:
            Tupla (evento_modificado, fue_modificado)
            - evento_modificado: Evento procesado o None si hay error
            - fue_modificado: True si se modificó el class_name
        
        Raises:
            ValidationError si el mensaje es inválido
        """
        try:
            # 1. Validar mensaje con Pydantic
            logger.info("🔍 Validando mensaje con Pydantic...")
            evento = self._validate_message(payload)
            logger.info(f"✅ Mensaje validado")
            logger.info(f"   Evento: {evento.event} | Detecciones: {len(evento.detections)}")
            
            # 2. Si no hay detecciones, retornar sin procesar
            if len(evento.detections) == 0:
                logger.warning("   ⚠️ No hay detecciones en el mensaje, retornando sin cambios")
                return evento, False
            
            # 3. Extraer imagen y procesar
            logger.info("🖼️  Extrayendo imagen base64...")
            imagen = self._extract_image(evento)
            
            # 4. Realizar inferencia
            logger.info("🤖 Procesando imagen con YOLOE...")
            image_processor = get_image_processor()
            annotated_image, detections = image_processor.detect_objects(imagen)
            
            # 5. Si no hay detecciones del modelo, retornar original
            if len(detections) == 0:
                logger.warning("   ⚠️ Modelo no detectó objetos, retornando sin cambios")
                return evento, False
            
            # 6. Calcular IoU entre bbox del mensaje y bbox del modelo
            logger.info("📐 Calculando IoU...")
            bbox_mensaje = self._extract_bbox_from_event(evento)
            bbox_modelo = image_processor.get_first_bbox(detections)
            
            if bbox_mensaje is None or bbox_modelo is None:
                logger.warning("   ⚠️ No se pudo obtener bboxes, retornando sin cambios")
                return evento, False
            
            iou_value = calculate_iou(bbox_mensaje, bbox_modelo)
            logger.info(f"   IoU: {iou_value:.4f}")
            
            # 7. Aplicar lógica condicional
            config = get_config()
            iou_threshold = config.get("processing", "iou_threshold")
            
            fue_modificado = False
            if iou_value > iou_threshold:
                logger.info(f"✅ IoU > {iou_threshold} → Modificando class_name...")
                new_class_name = config.get("processing", "class_name_on_match")
                
                for deteccion in evento.detections:
                    if deteccion.class_name != new_class_name:
                        logger.info(f"   '{deteccion.class_name}' → '{new_class_name}'")
                        deteccion.class_name = new_class_name
                        fue_modificado = True
            else:
                logger.info(f"⚠️ IoU ≤ {iou_threshold} → Sin cambios")
            
            # 8. Guardar en disco si está configurado
            if config.get("output", "save_processed"):
                self._save_processed_message(evento, iou_value)
            
            return evento, fue_modificado
        
        except ValidationError as e:
            logger.error(f"❌ Error de validación: {e}")
            return None, False
        except ImageProcessingError as e:
            logger.error(f"❌ Error procesando imagen: {e}")
            return None, False
        except Exception as e:
            logger.error(f"❌ Error inesperado: {e}")
            return None, False
    
    def _validate_message(self, payload: str) -> DroneEvent:
        """
        Valida mensaje JSON con Pydantic
        
        Args:
            payload: String JSON
        
        Returns:
            Evento validado
        
        Raises:
            ValidationError si falla validación
        """
        try:
            data = json.loads(payload)
            evento = DroneEvent.model_validate(data)
            return evento
        except json.JSONDecodeError as e:
            raise ValidationError(f"JSON inválido: {e}")
        except Exception as e:
            raise ValidationError(f"Pydantic validation error: {e}")
    
    def _extract_image(self, evento: DroneEvent):
        """Extrae y decodifica imagen del evento"""
        try:
            image_processor = get_image_processor()
            base64_str = evento.frame.data
            imagen = image_processor.decode_base64_image(base64_str)
            return imagen
        except Exception as e:
            raise ImageProcessingError(f"Error extrayendo imagen: {e}")
    
    def _extract_bbox_from_event(self, evento: DroneEvent) -> Optional[list]:
        """
        Extrae bbox de la primera detección del evento
        
        Args:
            evento: Evento MQTT
        
        Returns:
            bbox en formato [x1, y1, x2, y2] o None
        """
        try:
            if len(evento.detections) == 0:
                return None
            
            det = evento.detections[0]
            bbox = det.bbox
            
            # Convertir de (left, top, width, height) a (x1, y1, x2, y2)
            x1 = bbox.left
            y1 = bbox.top
            x2 = x1 + bbox.width
            y2 = y1 + bbox.height
            
            return [x1, y1, x2, y2]
        
        except Exception as e:
            raise ImageProcessingError(f"Error extrayendo bbox: {e}")
    
    def _save_processed_message(self, evento: DroneEvent, iou_value: float) -> None:
        """
        Guarda mensaje procesado en disco
        
        Args:
            evento: Evento procesado
            iou_value: Valor de IoU calculado
        """
        try:
            config = get_config()
            output_dir = Path(config.get("output", "output_dir"))
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generar nombre de archivo con timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{timestamp}_deteccion_procesada.json"
            filepath = output_dir / filename
            
            # Agregar metadata
            data = evento.model_dump()
            data["_iou"] = iou_value
            data["_saved_at"] = datetime.now().isoformat()
            
            # Guardar
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"   Mensaje guardado: {filepath}")
        
        except Exception as e:
            logger.warning(f"   ⚠️ Error guardando mensaje: {e}")


def get_message_processor() -> MessageProcessor:
    """Obtener instancia singleton de MessageProcessor"""
    return MessageProcessor()
