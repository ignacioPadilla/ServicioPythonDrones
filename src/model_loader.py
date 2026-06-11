"""
Descarga y carga de modelos YOLOE
"""
from pathlib import Path
from typing import Optional
import torch
from huggingface_hub import hf_hub_download
from ultralytics import YOLOE
from .config import get_config
from .device_manager import get_device_manager
from .logger_config import get_logger
from .exceptions import ModelLoadError

logger = get_logger(__name__)


class ModelLoader:
    """Singleton para cargar y cachear modelos YOLOE"""
    
    _instance = None
    _model = None
    _model_name = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance
    
    def load_model(self, force_reload: bool = False) -> YOLOE:
        """
        Carga el modelo YOLOE, descargando si es necesario
        
        Args:
            force_reload: Forzar recarga aunque esté en caché
        
        Returns:
            Modelo YOLOE cargado
        
        Raises:
            ModelLoadError si hay error en la carga
        """
        try:
            if self._model is not None and not force_reload:
                logger.debug("   Modelo cacheado, usando instancia existente")
                return self._model
            
            config = get_config()
            model_name = config.get("models", "yoloe_model")
            model_dir = config.get("models", "model_dir")
            
            # Crear directorio si no existe
            model_path = Path(model_dir)
            model_path.mkdir(parents=True, exist_ok=True)
            
            # Ruta completa del modelo
            full_model_path = model_path / model_name
            
            # Descargar si no existe
            if not full_model_path.exists():
                logger.info(f"📥 Descargando modelo {model_name}...")
                self._download_model(model_name, str(model_path))
                logger.info(f"✅ Modelo descargado en {full_model_path}")
            else:
                logger.info(f"✅ Modelo encontrado: {full_model_path}")
            
            # Cargar modelo
            logger.info("🔄 Cargando modelo YOLOE...")
            device_manager = get_device_manager()
            device = device_manager.get_device(config.get("processing", "device"))
            
            self._model = YOLOE(str(full_model_path)).to(device)
            self._model_name = model_name
            
            logger.info(f"✅ Modelo YOLOE cargado correctamente")
            return self._model
        
        except Exception as e:
            raise ModelLoadError(f"Error al cargar modelo YOLOE: {e}")
    
    def _download_model(self, model_name: str, local_dir: str) -> None:
        """
        Descarga modelo de HuggingFace Hub
        
        Args:
            model_name: Nombre del modelo (ej: yoloe-v8l-seg.pt)
            local_dir: Directorio local donde guardar
        
        Raises:
            ModelLoadError si hay error en descarga
        """
        try:
            logger.info(f"   Descargando desde HuggingFace: jameslahm/yoloe/{model_name}")
            path = hf_hub_download(
                repo_id="jameslahm/yoloe",
                filename=model_name,
                local_dir=local_dir
            )
            logger.info(f"   ✅ Descargado en: {path}")
        except Exception as e:
            raise ModelLoadError(f"Error descargando modelo desde HuggingFace: {e}")
    
    def get_model(self) -> Optional[YOLOE]:
        """Retorna el modelo cacheado o None si no ha sido cargado"""
        return self._model
    
    def clear_cache(self) -> None:
        """Limpia el modelo de la memoria"""
        if self._model is not None:
            del self._model
            torch.cuda.empty_cache()
            self._model = None
            self._model_name = None
            logger.info("✅ Modelo descargado de memoria")


def get_model_loader() -> ModelLoader:
    """Obtener instancia singleton de ModelLoader"""
    return ModelLoader()
