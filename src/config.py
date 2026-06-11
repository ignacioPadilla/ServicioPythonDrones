"""
Gestión centralizada de configuración
Carga config.json y sobrescribe con variables de entorno
"""
import json
import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv
from .exceptions import ConfigurationError
from .logger_config import get_logger

logger = get_logger(__name__)


class MqttConfig(BaseModel):
    """Configuración MQTT"""
    broker: str = "localhost"
    port: int = 1883
    username: str = ""
    password: str = ""
    topic_subscribe: str = "vision/camera/+"
    topic_publish: str = "vision/result/+"
    keepalive: int = 60


class ModelsConfig(BaseModel):
    """Configuración de modelos"""
    model_dir: str = "models"
    reference_image: str = "reference.jpg"
    yoloe_model: str = "yoloe8n.onnx"
    conf_threshold: float = 0.25
    reference_bbox: tuple[int, int, int, int] = Field(
        description="Bounding box de la imagen de referencia (x1, y1, x2, y2)"
    )


class ProcessingConfig(BaseModel):
    """Configuración de procesamiento"""
    device: str = "cpu"
    iou_threshold: float = 0.45
    max_detections: int = 100

    @field_validator('device')
    @classmethod
    def validate_device(cls, v: str) -> str:
        if v not in ('cpu', 'cuda', 'mps'):
            raise ValueError(f"Device '{v}' no válido. Debe ser: cpu, cuda, o mps")
        return v


class OutputConfig(BaseModel):
    """Configuración de salida"""
    output_dir: str = "output"
    save_processed: bool = True
    save_debug: bool = False


class LoggingConfig(BaseModel):
    """Configuración de logging"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    @field_validator('level')
    @classmethod
    def validate_level(cls, v: str) -> str:
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        if v.upper() not in valid_levels:
            raise ValueError(f"Level '{v}' no válido. Debe ser: {valid_levels}")
        return v.upper()


class Config(BaseModel):
    """Configuración principal con validación Pydantic"""
    mqtt: MqttConfig = Field(default_factory=MqttConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    _instance: Optional['Config'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self) -> None:
        """Carga config.json y sobrescribe con .env"""
        try:
            load_dotenv()
            logger.info("✅ Archivo .env cargado")

            config_path = Path(__file__).parent.parent / "config" / "config.json"

            if not config_path.exists():
                raise ConfigurationError(f"config.json no encontrado en {config_path}")

            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            logger.info(f"✅ config.json cargado desde {config_path}")

            self._apply_env_overrides(config_data)

            # Actualizar el modelo con los datos
            self.model_validate(config_data)
            logger.info("✅ Variables de entorno aplicadas")

        except Exception as e:
            raise ConfigurationError(f"Error al cargar configuración: {e}")

    def _apply_env_overrides(self, config_data: dict) -> None:
        """Sobrescribe valores de config con variables de entorno"""
        env_mappings = {
            "MQTT_BROKER": ("mqtt", "broker"),
            "MQTT_PORT": ("mqtt", "port"),
            "MQTT_USERNAME": ("mqtt", "username"),
            "MQTT_PASSWORD": ("mqtt", "password"),
            "MQTT_TOPIC_SUBSCRIBE": ("mqtt", "topic_subscribe"),
            "MQTT_TOPIC_PUBLISH": ("mqtt", "topic_publish"),
            "MODEL_DIR": ("models", "model_dir"),
            "REFERENCE_IMAGE": ("models", "reference_image"),
            "REFERENCE_BBOX": ("models", "reference_bbox"),
            "DEVICE": ("processing", "device"),
            "IOU_THRESHOLD": ("processing", "iou_threshold"),
            "OUTPUT_DIR": ("output", "output_dir"),
            "SAVE_OUTPUT": ("output", "save_processed"),
            "LOG_LEVEL": ("logging", "level"),
        }

        for env_key, (section, key) in env_mappings.items():
            if env_value := os.getenv(env_key):
                if section not in config_data:
                    config_data[section] = {}

                if env_key == "MQTT_PORT":
                    env_value = int(env_value)
                elif env_key == "IOU_THRESHOLD":
                    env_value = float(env_value)
                elif env_key == "SAVE_OUTPUT":
                    env_value = env_value.lower() in ('true', '1', 'yes')
                elif env_key == "REFERENCE_BBOX":
                    # Parsear "x1,y1,x2,y2" -> (x1, y1, x2, y2)
                    parts = env_value.split(',')
                    if len(parts) == 4:
                        env_value = tuple(int(p.strip()) for p in parts)
                    else:
                        logger.warning(f"REFERENCE_BBOX formato inválido: {env_value}. Usar: x1,y1,x2,y2")
                        env_value = (0, 0, 0, 0)

                config_data[section][key] = env_value
                logger.debug(f"   {env_key} -> {section}.{key} = {env_value}")

        # Actualizar la instancia con los datos sobrescritos
        self.__dict__.update(Config.model_validate(config_data).__dict__)

    def get(self, section: str, key: str, default=None):
        """Obtener valor de configuración"""
        section_obj = getattr(self, section, None)
        if section_obj is None:
            if default is not None:
                return default
            raise ConfigurationError(f"Config {section}.{key} no encontrada")
        return getattr(section_obj, key, default)

    def get_all(self) -> dict:
        """Retorna la configuración completa como dict"""
        return self.model_dump()

    def validate(self) -> bool:
        """Valida que la configuración sea válida"""
        self.model_validate()
        logger.info("✅ Configuración validada correctamente")
        return True


def get_config() -> Config:
    """Obtener instancia singleton de Config"""
    return Config()
