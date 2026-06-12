"""
Procesamiento de imágenes con YOLOE
"""
import base64
import io
from PIL import Image
from typing import Tuple, Optional
import supervision as sv
from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor
import numpy as np

from .config import get_config
from .model_loader import get_model_loader
from .logger_config import get_logger
from .exceptions import ImageProcessingError

logger = get_logger(__name__)


class ImageProcessor:
    """Procesa imágenes: decodificación, inferencia, anotación"""
    
    _instance = None
    _model_configured = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ImageProcessor, cls).__new__(cls)
        return cls._instance
    
    def setup(self) -> None:
        """
        Configura el modelo YOLOE con la imagen de referencia usando Visual Prompt.
        Debe llamarse UNA sola vez al iniciar el servicio.
        """
        try:
            if self._model_configured:
                logger.debug("   Modelo ya configurado, saltando setup")
                return
            
            config = get_config()
            reference_image_path = config.get("models", "reference_image")
            reference_bbox = config.get("models", "reference_bbox") #[x1, y1, x2, y2]
            class_name = config.get("processing","class_name_on_match") #ej: "Camioneta"

            logger.info("Configurando modelo YOLOE con imagen de referencia...")
            logger.info(f"  Imagen de referencia: {reference_image_path}")
            logger.info(f" BBox de referencia: {reference_bbox}")
            logger.info(f" Clase objetivo: {class_name}")
            
            # Cargar imagen de referencia
            ref_image = Image.open(reference_image_path)
            logger.info(f"   ✅ Imagen de referencia cargada: {ref_image.size}")

            #Obtener el modelo ya cargado desde ModelLoader
            model_loader = get_model_loader()
            model = model_loader.load_model()
            logger.info("   ✅ Modelo YOLOE cargado desde ModelLoader")


            #Construir prompt visual con bbox y clase
            NAMES = [class_name]  
            bboxes = np.array([reference_bbox], dtype=np.float64)
            cls = np.array([0], dtype=np.int32)
            prompts = dict(bboxes=bboxes, cls=cls)
            logger.info(f"   ✅ Prompt visual construido ")

            #Paso 1: Inferencia en imagen de referencia. Extraer el VPE
            logger.info(" Extrayendo Visual Prompt Embedding (VPE)...")
            model.predict(
                ref_image,
                prompts=prompts,
                predictor=YOLOEVPSegPredictor(),
                return_vpe=True,

            )
            logger.info("   ✅ VPE extraído y modelo configurado con imagen de referencia") 

            #Paso 2: configurar el modelo con las con las clase y el embedding extraído
            model.set_classes(NAMES,model.predictor.vpe)
            model.predictor= None #Liberar predictor para que se cree uno nuevo en cada inferencia con el VPE ya configurado

            self._model_configured = True
            logger.info("✅ Modelo YOLOE configurado correctamente con Visual Prompt")

        except Exception as e:
            raise ImageProcessingError(f"Error configurando modelo YOLOE: {e}")



            






    
    def decode_base64_image(self, base64_str: str) -> Image.Image:
        """
        Decodifica imagen en base64 a PIL Image
        
        Args:
            base64_str: String en base64 (con o sin prefijo data:image/...)
        
        Returns:
            PIL Image decodificada
        
        Raises:
            ImageProcessingError si hay error
        """
        try:
            # Limpiar prefijo si existe
            if ',' in base64_str:
                base64_str = base64_str.split(',')[-1]
            
            # Decodificar base64 a bytes
            img_bytes = base64.b64decode(base64_str)
            
            # Convertir a PIL Image
            image = Image.open(io.BytesIO(img_bytes))
            
            logger.debug(f"   Imagen decodificada: {image.size}, formato {image.format}")
            return image
        
        except Exception as e:
            raise ImageProcessingError(f"Error decodificando imagen base64: {e}")
    
    def detect_objects(self, image: Image.Image) -> Tuple[Image.Image, sv.Detections]:
        """
        Realiza inferencia en imagen con YOLOE
        
        Args:
            image: PIL Image a procesar
        
        Returns:
            Tupla (imagen_anotada, detecciones)
        
        Raises:
            ImageProcessingError si hay error
        """
        try:
            logger.info("🔍 Realizando inferencia con YOLOE...")
            
            # Cargar modelo
            model_loader = get_model_loader()
            model = model_loader.load_model()
            
            # Inferencia
            results = model.predict(image)
            
            # Extraer detecciones
            detections = sv.Detections.from_ultralytics(results[0])
            
            logger.info(f"✅ Inferencia completada: {len(detections)} detecciones")
            
            # Anotar imagen
            annotated_image = image.copy()
            annotated_image = sv.BoxAnnotator().annotate(
                scene=annotated_image,
                detections=detections
            )
            annotated_image = sv.LabelAnnotator().annotate(
                scene=annotated_image,
                detections=detections
            )
            
            return annotated_image, detections
        
        except Exception as e:
            raise ImageProcessingError(f"Error en inferencia YOLOE: {e}")
    
    def extract_bboxes(self, detections: sv.Detections) -> list:
        """
        Extrae bounding boxes de detecciones
        
        Args:
            detections: Detecciones de supervision
        
        Returns:
            Lista de bboxes en formato [x1, y1, x2, y2]
        """
        try:
            if len(detections) == 0:
                logger.warning("   ⚠️ No hay detecciones")
                return []
            
            bboxes = detections.xyxy.tolist()
            logger.debug(f"   {len(bboxes)} bboxes extraídas")
            
            return bboxes
        
        except Exception as e:
            raise ImageProcessingError(f"Error extrayendo bboxes: {e}")
    
    def get_first_bbox(self, detections: sv.Detections) -> Optional[list]:
        """
        Obtiene la primera detección como bbox
        
        Args:
            detections: Detecciones de supervision
        
        Returns:
            Primera bbox en formato [x1, y1, x2, y2] o None
        """
        try:
            if len(detections) == 0:
                return None
            
            bbox = detections.xyxy[0].tolist()
            logger.debug(f"   Primera bbox: {bbox}")
            
            return bbox
        
        except Exception as e:
            raise ImageProcessingError(f"Error obteniendo primera bbox: {e}")


def get_image_processor() -> ImageProcessor:
    """Obtener instancia singleton de ImageProcessor"""
    return ImageProcessor()
