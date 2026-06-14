"""
Servicio MQTT para procesamiento de imágenes con YOLOE
Punto de entrada principal
"""
import signal
import sys
import time
from pathlib import Path

from .config import get_config
from .logger_config import get_logger
from .device_manager import get_device_manager
from .model_loader import get_model_loader
from .image_processor import get_image_processor
from .message_processor import get_message_processor
from .mqtt_service import get_mqtt_service
from .exceptions import ConfigurationError, ModelLoadError, MQTTConnectionError

logger = get_logger(__name__)


class MQTTProcessingService:
    """Servicio principal que orquesta todo"""
    
    def __init__(self):
        self.running = False
        self.mqtt_service = get_mqtt_service()
        self.message_processor = get_message_processor()
    
    def initialize(self) -> None:
        """Inicialización del servicio"""
        try:
            logger.info("=" * 60)
            logger.info("🚀 Iniciando Servicio MQTT de Procesamiento de Imágenes")
            logger.info("=" * 60)
            
            # 1. Cargar y validar configuración
            logger.info("\n📋 Paso 1: Validando configuración...")
            config = get_config()
            #config.validate()
            logger.info("✅ Configuración válida")
            
            # 2. Mostrar device
            logger.info("\n🖥️  Paso 2: Seleccionando device...")
            device_manager = get_device_manager()
            device = device_manager.get_device(config.get("processing", "device"))
            device_info = device_manager.info()
            logger.info(f"✅ Device: {device_info['device_type'].upper()}")
            
            # 3. Descargar y cargar modelos
            logger.info("\n🤖 Paso 3: Cargando modelos...")
            model_loader = get_model_loader()
            model = model_loader.load_model()
            logger.info("✅ Modelos cargados")
            
            # 4. Configurar procesador de imágenes
            logger.info("\n🔧 Paso 4: Configurando procesador...")
            image_processor = get_image_processor()
            image_processor.setup()
            logger.info("✅ Procesador configurado")
            
            # 5. Conectar a MQTT
            logger.info("\n📡 Paso 5: Conectando a MQTT...")
            self.mqtt_service.connect(self._on_mqtt_message)
            logger.info("✅ MQTT conectado")
            
            logger.info("\n" + "=" * 60)
            logger.info("✅ Servicio inicializado correctamente")
            logger.info("   Esperando mensajes MQTT...")
            logger.info("   Presiona Ctrl+C para detener")
            logger.info("=" * 60 + "\n")
            
        except ConfigurationError as e:
            logger.error(f"❌ Error de configuración: {e}")
            sys.exit(1)
        except ModelLoadError as e:
            logger.error(f"❌ Error cargando modelos: {e}")
            sys.exit(1)
        except MQTTConnectionError as e:
            logger.error(f"❌ Error conectando MQTT: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"❌ Error inesperado: {e}")
            sys.exit(1)
    
    def _on_mqtt_message(self, client, userdata, msg) -> None:
        """Callback cuando llega mensaje MQTT"""
        try:
            # Decodificar payload
            payload = msg.payload.decode('utf-8')
            
            # Procesar mensaje
            evento_procesado, fue_modificado = self.message_processor.process_message(payload)
            
            if evento_procesado is None:
                logger.warning("   ⚠️ Mensaje descartado debido a error")
                return
            
            # Convertir a JSON para enviar
            json_salida = evento_procesado.model_dump_json(indent=2)
            
            # Publicar
            try:
                self.mqtt_service.publish_processed_message(json_salida)
                
                if fue_modificado:
                    logger.info("   ✅ Mensaje modificado y reenviado")
                else:
                    logger.info("   ✅ Mensaje reenviado sin cambios")
            
            except Exception as e:
                logger.error(f"   ❌ Error publicando: {e}")
        
        except Exception as e:
            logger.error(f"❌ Error procesando mensaje MQTT: {e}")
    
    def run(self) -> None:
        """Ejecuta el servicio indefinidamente"""
        self.running = True
        
        # Registrar handlers de señal
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            # Mantener el servicio ejecutándose
            while self.running:
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            logger.info("\n⏹️  Parada controlada...")
            self.shutdown()
    
    def _signal_handler(self, signum, frame) -> None:
        """Maneja señales de interrupción"""
        logger.info(f"\n📍 Señal recibida ({signum}), deteniendo servicio...")
        self.running = False
        self.shutdown()
    
    def shutdown(self) -> None:
        """Limpieza al apagar el servicio"""
        logger.info("🛑 Apagando servicio...")
        
        try:
            # Desconectar MQTT
            if self.mqtt_service:
                self.mqtt_service.disconnect()
            
            # Limpiar modelos de memoria
            model_loader = get_model_loader()
            model_loader.clear_cache()
            
            logger.info("✅ Servicio apagado correctamente")
        
        except Exception as e:
            logger.error(f"⚠️ Error durante apagado: {e}")
        
        sys.exit(0)


def main():
    """Punto de entrada principal"""
    try:
        service = MQTTProcessingService()
        service.initialize()
        service.run()
    
    except KeyboardInterrupt:
        logger.info("\n✅ Apagado")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Error fatal: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
