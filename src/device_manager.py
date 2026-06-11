"""
Gestión de device (GPU/CPU)
"""
import torch
from .logger_config import get_logger
from .exceptions import DeviceError

logger = get_logger(__name__)


class DeviceManager:
    """Gestiona la selección de device (GPU o CPU)"""
    
    _instance = None
    _device = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DeviceManager, cls).__new__(cls)
        return cls._instance
    
    def get_device(self, device_preference: str = "auto") -> torch.device:
        """
        Obtener device óptimo
        
        Args:
            device_preference: "auto" (GPU si disponible), "cuda", "cpu"
        
        Returns:
            torch.device configurado
        
        Raises:
            DeviceError si la configuración es inválida
        """
        if self._device is not None:
            return self._device
        
        try:
            if device_preference.lower() == "auto":
                if torch.cuda.is_available():
                    self._device = torch.device("cuda")
                    gpu_name = torch.cuda.get_device_name(0)
                    logger.info(f"✅ GPU detectada: {gpu_name}")
                else:
                    self._device = torch.device("cpu")
                    logger.info("⚠️  GPU no disponible, usando CPU")
            
            elif device_preference.lower() == "cuda":
                if torch.cuda.is_available():
                    self._device = torch.device("cuda")
                    gpu_name = torch.cuda.get_device_name(0)
                    logger.info(f"✅ GPU seleccionada: {gpu_name}")
                else:
                    raise DeviceError("Se solicitó CUDA pero GPU no está disponible")
            
            elif device_preference.lower() == "cpu":
                self._device = torch.device("cpu")
                logger.info("✅ CPU seleccionada")
            
            else:
                raise DeviceError(f"Device inválido: {device_preference}. Usa 'auto', 'cuda' o 'cpu'")
            
            return self._device
        
        except Exception as e:
            raise DeviceError(f"Error al seleccionar device: {e}")
    
    def info(self) -> dict:
        """Retorna información del device"""
        if self._device is None:
            return {"status": "Device no inicializado"}
        
        info = {
            "device": str(self._device),
            "device_type": self._device.type,
        }
        
        if self._device.type == "cuda":
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
            info["gpu_memory_mb"] = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        
        return info


def get_device_manager() -> DeviceManager:
    """Obtener instancia singleton de DeviceManager"""
    return DeviceManager()
