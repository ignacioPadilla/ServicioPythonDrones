"""
Cliente MQTT para conexión y comunicación
"""
import paho.mqtt.client as mqtt
import time
from typing import Callable, Optional
from .config import get_config
from .logger_config import get_logger
from .exceptions import MQTTConnectionError, MQTTPublishError

logger = get_logger(__name__)


class MQTTService:
    """Gestiona conexión MQTT y publicación de mensajes"""
    
    _instance = None
    _client = None
    _connected = False
    _on_message_callback = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MQTTService, cls).__new__(cls)
        return cls._instance
    
    def connect(self, on_message_callback: Callable) -> None:
        """
        Conecta al broker MQTT
        
        Args:
            on_message_callback: Función que se ejecuta al recibir mensaje
                Firma: callback(client, userdata, msg)
        
        Raises:
            MQTTConnectionError si falla la conexión
        """
        try:
            config = get_config()
            
            # Parámetros de conexión
            broker = config.get("mqtt", "broker")
            port = config.get("mqtt", "port")
            username = config.get("mqtt", "username")
            password = config.get("mqtt", "password")
            
            logger.info(f"🔌 Conectando a broker MQTT: {broker}:{port}...")
            
            # Crear cliente
            client_id = f"mqtt-service-{int(time.time())}"
            self._client = mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                client_id=client_id
            )
            
            # Configurar credenciales
            self._client.username_pw_set(username, password)
            
            # Configurar callbacks
            self._client.on_connect = self._on_connect
            self._client.on_message = self._create_message_wrapper(on_message_callback)
            self._client.on_disconnect = self._on_disconnect
            
            self._on_message_callback = on_message_callback
            
            # Conectar
            self._client.connect(broker, port, keepalive=60)
            
            # Iniciar loop
            self._client.loop_start()
            
            # Esperar a que se conecte
            max_wait = 10
            for _ in range(max_wait):
                if self._connected:
                    break
                time.sleep(0.5)
            
            if not self._connected:
                raise MQTTConnectionError("Timeout esperando conexión")
            
            logger.info("✅ MQTT conectado y escuchando")
        
        except Exception as e:
            raise MQTTConnectionError(f"Error conectando a MQTT: {e}")
    
    def _on_connect(self, client, userdata, flags, reason_code, properties):
        """Callback de conexión"""
        try:
            if reason_code == 0:
                logger.info("✅ Conexión exitosa al broker")
                self._connected = True
                
                # Suscribirse al topic
                config = get_config()
                topic = config.get("mqtt", "topic_subscribe")
                client.subscribe(topic)
                logger.info(f"   📡 Suscrito a: {topic}")
            else:
                logger.error(f"❌ Error conexión (código {reason_code})")
                self._connected = False
        
        except Exception as e:
            logger.error(f"❌ Error en on_connect: {e}")
    
    def _on_disconnect(self, client, userdata, flags, reason_code, properties):
        """Callback de desconexión"""
        self._connected = False
        logger.warning(f"⚠️ Desconectado del broker (código {reason_code})")
    
    def _create_message_wrapper(self, callback: Callable) -> Callable:
        """Envuelve el callback del usuario con logging"""
        def wrapper(client, userdata, msg):
            try:
                logger.info(f"\n📥 Mensaje recibido en {msg.topic}")
                callback(client, userdata, msg)
            except Exception as e:
                logger.error(f"❌ Error en on_message: {e}")
        
        return wrapper
    
    def publish(self, topic: str, payload: str, qos: int = 1) -> None:
        """
        Publica un mensaje en un topic
        
        Args:
            topic: Topic MQTT
            payload: Contenido del mensaje (string)
            qos: Quality of Service (0, 1, 2)
        
        Raises:
            MQTTPublishError si falla la publicación
        """
        try:
            if not self._connected:
                raise MQTTPublishError("No conectado al broker MQTT")
            
            result = self._client.publish(topic, payload, qos=qos)
            
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                raise MQTTPublishError(f"Error publicando (rc={result.rc})")
            
            logger.info(f"✅ Mensaje publicado en {topic}")
        
        except Exception as e:
            raise MQTTPublishError(f"Error publicando: {e}")
    
    def publish_processed_message(self, payload: str) -> None:
        """
        Publica mensaje procesado en el topic de salida
        
        Args:
            payload: JSON del mensaje procesado
        
        Raises:
            MQTTPublishError si falla
        """
        try:
            config = get_config()
            topic = config.get("mqtt", "topic_publish")
            self.publish(topic, payload)
        except Exception as e:
            raise MQTTPublishError(f"Error publicando mensaje procesado: {e}")
    
    def disconnect(self) -> None:
        """Desconecta del broker MQTT"""
        try:
            if self._client:
                self._client.loop_stop()
                self._client.disconnect()
                self._connected = False
                logger.info("✅ MQTT desconectado")
        except Exception as e:
            logger.warning(f"⚠️ Error desconectando: {e}")
    
    def is_connected(self) -> bool:
        """Retorna estado de conexión"""
        return self._connected


def get_mqtt_service() -> MQTTService:
    """Obtener instancia singleton de MQTTService"""
    return MQTTService()
