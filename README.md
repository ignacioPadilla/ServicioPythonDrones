# Servicio MQTT de Procesamiento de Imágenes con YOLOE

Servicio Python que procesa mensajes MQTT conteniendo imágenes en base64, realiza detección de objetos con YOLOE, calcula IoU y reenvía mensajes modificados.

## 🎯 Características

- ✅ Escucha continua de mensajes MQTT
- 🤖 Detección de objetos con modelo YOLOE
- 📐 Cálculo de IoU entre detecciones
- 🔄 Modificación condicional de class_name según IoU
- 💾 Persistencia de mensajes procesados en JSON
- 🖥️ Soporte automático GPU/CPU
- ⚙️ Configuración flexible (JSON + .env)
- 📋 Validación robusta con Pydantic

## 📋 Requisitos

- Python 3.9+
- GPU NVIDIA (opcional, fallback a CPU automático)
- CUDA 11.8+ (si se usa GPU)

## 🚀 Instalación

### 1. Clonar/Preparar proyecto

```bash
cd VC_Mqtt
```

### 2. Crear ambiente virtual

```bash
python -m venv venv
# En Windows:
venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar

#### Opción A: Solo config.json (recomendado para inicio)
```bash
# Ya viene preconfigurado en config/config.json
# Solo asegurate de tener la imagen de referencia
```

#### Opción B: Con variables de entorno
```bash
# Copiar template
cp .env.example .env

# Editar .env con tus valores
# (Sobrescribe los valores de config.json)
```

### 5. Agregar imagen de referencia

Coloca la imagen de referencia que el modelo usará para el prompt:
```
config/reference_image.jpg
```

### 6. Primera ejecución (descarga modelos)

```bash
python run.py
```

La primera ejecución descargará automáticamente los modelos YOLOE (~2-5 minutos según conexión).

## 📁 Estructura del Proyecto

```
VC_Mqtt/
├── src/
│   ├── __init__.py
│   ├── main.py                    # Punto de entrada principal
│   ├── config.py                  # Gestión de configuración
│   ├── models.py                  # Modelos Pydantic
│   ├── logger_config.py           # Logging centralizado
│   ├── exceptions.py              # Excepciones custom
│   ├── device_manager.py          # Gestión GPU/CPU
│   ├── model_loader.py            # Carga de YOLOE
│   ├── image_processor.py         # Procesamiento de imágenes
│   ├── iou_calculator.py          # Cálculo de IoU
│   ├── message_processor.py       # Orquestación de procesamiento
│   └── mqtt_service.py            # Cliente MQTT
├── config/
│   ├── config.json                # Configuración (JSON)
│   └── reference_image.jpg        # Imagen de prompt (agregar)
├── models/                        # Auto-descargados en primera ejecución
├── output/                        # JSONs procesados con timestamp
├── .env.example                   # Template de variables de entorno
├── .env                           # Variables de entorno locales (no versionar)
├── requirements.txt               # Dependencias Python
├── run.py                         # Script de ejecución
└── README.md                      # Este archivo
```

## ⚙️ Configuración

### config/config.json

```json
{
  "mqtt": {
    "broker": "mqtt.mediamtx.gcp.mindlabs.cl",
    "port": 1883,
    "username": "test",
    "password": "Javj4UXtlsq2gE",
    "topic_subscribe": "test/ignacio/sensor1",
    "topic_publish": "test/ignacio/sensor1"
  },
  "models": {
    "yoloe_model": "yoloe-v8l-seg.pt",
    "model_dir": "./models",
    "auto_download": true,
    "reference_image": "./config/reference_image.jpg"
  },
  "processing": {
    "iou_threshold": 0.5,           # Umbral de IoU
    "device": "auto",              # "auto", "cuda", "cpu"
    "class_name_on_match": "Camioneta"
  },
  "output": {
    "save_processed": true,
    "output_dir": "./output",
    "filename_pattern": "{timestamp}_deteccion_procesada.json"
  },
  "logging": {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  }
}
```

### .env (variables de entorno que sobrescriben config.json)

```bash
MQTT_BROKER=mqtt.mediamtx.gcp.mindlabs.cl
MQTT_PORT=1883
MQTT_USERNAME=test
MQTT_PASSWORD=Javj4UXtlsq2gE

DEVICE=auto                        # "auto", "cuda", "cpu"
IOU_THRESHOLD=0.5
SAVE_OUTPUT=true
LOG_LEVEL=INFO
```

## 🏃 Ejecución

### Ejecución básica

```bash
python run.py
```

### Ver logs detallados

```bash
# En .env o config.json:
LOG_LEVEL=DEBUG
python run.py
```

### Sin guardar archivos procesados

```bash
SAVE_OUTPUT=false python run.py
```

### Forzar CPU (sin GPU)

```bash
DEVICE=cpu python run.py
```

## 📊 Flujo de Procesamiento

```
1. Recibir mensaje MQTT con imagen en base64
   ↓
2. Validar con Pydantic
   ↓
3. Decodificar imagen base64 → PIL Image
   ↓
4. Inferencia con modelo YOLOE
   ↓
5. Extraer bboxes del mensaje y del modelo
   ↓
6. Calcular IoU entre bboxes
   ↓
7. Si IoU > umbral (0.5):
   └─ Modificar class_name → "Camioneta"
   ↓
8. Guardar JSON en ./output/{timestamp}.json
   ↓
9. Reenviare mensaje por MQTT al topic de salida
```

## 📤 Formato de Mensajes MQTT

### Mensaje de entrada esperado

```json
{
  "timestamp": 1704067200.123,
  "event": "detection",
  "type": "sensor_data",
  "detections": [
    {
      "id": 1,
      "class_id": 0,
      "class_name": "objeto_original",
      "confidence": 0.95,
      "bbox": {
        "left": 100.0,
        "top": 50.0,
        "width": 200.0,
        "height": 150.0
      }
    }
  ],
  "frame": {
    "image_type": "jpeg",
    "width": 1920,
    "height": 1080,
    "data": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
  }
}
```

### Mensaje de salida

Idéntico al entrada pero con:
- `class_name` potencialmente modificado si IoU > 0.5
- Metadata adicional guardada en archivo:
  - `_iou`: Valor de IoU calculado
  - `_saved_at`: Timestamp de procesamiento

## 🔧 Troubleshooting

### Error: "config.json no encontrado"
- Verifica que `config/config.json` exista
- Ejecuta desde el directorio raíz del proyecto

### Error: "GPU no disponible"
- El servicio fallback automático a CPU
- Verifica CUDA con: `python -c "import torch; print(torch.cuda.is_available())"`

### Error: "Descargando modelo..." muy lento
- Es normal en la primera ejecución (2-5 minutos)
- Requiere conexión a HuggingFace Hub
- Las ejecuciones posteriores cargarán desde caché local

### Mensajes no llegan
- Verifica credenciales MQTT en config.json
- Verifica que el broker esté activo
- Comprueba el topic_subscribe

### Error: "reference_image.jpg no encontrado"
- Coloca la imagen en `config/reference_image.jpg`
- Debe ser una imagen JPEG/PNG válida

## 📝 Logs

Los logs se muestran en consola con formato:
```
2024-01-01 12:00:00 - __main__ - INFO - ✅ Servicio inicializado correctamente
2024-01-01 12:00:05 - __main__ - INFO - 📥 Mensaje recibido en test/ignacio/sensor1
```

Para cambiar nivel de logging:
```bash
# En .env
LOG_LEVEL=DEBUG
```

## 🛑 Apagado

```bash
Ctrl+C
```

El servicio se apagará de forma controlada:
1. Desconecta MQTT
2. Limpia modelos de memoria
3. Sale sin errores

## 🐛 Debugging

### Modo debug (logs muy detallados)

```bash
LOG_LEVEL=DEBUG python run.py
```

### Mensaje de prueba (MQTTX)

Desde MQTTX o similar, publica un mensaje en `test/ignacio/sensor1`:
```json
{
  "timestamp": 1704067200.123,
  "event": "test",
  "type": "sensor_data",
  "detections": [{"id": 1, "class_id": 0, "class_name": "test", "confidence": 0.9, "bbox": {"left": 10, "top": 10, "width": 100, "height": 100}}],
  "frame": {"image_type": "jpeg", "width": 640, "height": 480, "data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABA..."}
}
```

## 📦 Dependencias Principales

- **paho-mqtt**: Cliente MQTT
- **pydantic**: Validación de datos
- **torch/torchvision**: PyTorch y utilidades de visión
- **ultralytics**: Framework YOLOE
- **supervision**: Herramientas de detección
- **Pillow**: Procesamiento de imágenes
- **huggingface-hub**: Descarga de modelos

## 📄 Licencia

Proyecto interno de Tesis.

## 👨‍💻 Mantenimiento

- Cambios en configuración: Editar `config/config.json` o `.env`
- Nueva imagen de referencia: Reemplazar `config/reference_image.jpg`
- Modelo diferente: Editar `models.yoloe_model` en `config.json`

---

**Creado**: 2024  
**Última actualización**: Abril 2026
