# 🚀 Guía de Instalación - Servicio MQTT de Procesamiento de Imágenes

## ✅ Estado Actual

- ✅ Estructura de carpetas creada
- ✅ Código fuente implementado (12 módulos)
- ✅ Configuración (config.json y .env)
- ✅ Entorno virtual creado (`venv/`)
- ✅ Dependencias esenciales instaladas

## 📋 Dependencias Instaladas

### Ya instaladas ✅
```
paho-mqtt==2.1.0          ✅ Cliente MQTT
pydantic==2.13.2          ✅ Validación de datos
python-dotenv==1.2.2      ✅ Variables de entorno
Pillow==12.2.0            ✅ Procesamiento de imágenes
numpy==2.4.4              ✅ Cálculos numéricos
```

### Falta instalar (Machine Learning) ⚠️

Torch, torchvision, ultralytics, supervision, etc. necesitan instalarse en un paso separado debido a su tamaño.

## 🎯 Pasos de Instalación

### Paso 1: Instalar Dependencias de ML

**Opción A: Automática (recomendado)**

```bash
cd "c:\Users\Ignacio\Desktop\Tesis\Pega\VC_Mqtt"
.\venv\Scripts\python install_ml_deps.py
```

Esto te permitirá elegir:
- CPU solo
- GPU NVIDIA CUDA 11.8
- GPU NVIDIA CUDA 12.1
- Prueba solo torch

**Opción B: Manual - CPU**

```bash
cd "c:\Users\Ignacio\Desktop\Tesis\Pega\VC_Mqtt"
.\venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\venv\Scripts\pip install ultralytics supervision huggingface-hub opencv-python scipy matplotlib
```

**Opción C: Manual - GPU CUDA 12.1 (más reciente)**

```bash
cd "c:\Users\Ignacio\Desktop\Tesis\Pega\VC_Mqtt"
.\venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
.\venv\Scripts\pip install ultralytics supervision huggingface-hub opencv-python scipy matplotlib
```

### Paso 2: Agregar Imagen de Referencia

Coloca una imagen de ejemplo (camioneta, auto, etc) en:

```
config/reference_image.jpg
```

Esta imagen se usa para que el modelo YOLOE aprenda qué detectar.

### Paso 3: Crear Archivo .env (opcional pero recomendado)

```bash
Copy-Item ".env.example" ".env"
# Edita .env si necesitas cambiar credenciales MQTT
```

### Paso 4: Verificar Instalación

```bash
.\venv\Scripts\python -c "import torch; print(f'Torch: {torch.__version__}'); print(f'GPU: {torch.cuda.is_available()}')"
```

Debe mostrar:
```
Torch: 2.x.x
GPU: True (si tienes GPU) o False (si usas CPU)
```

## 🏃 Ejecutar el Servicio

```bash
cd "c:\Users\Ignacio\Desktop\Tesis\Pega\VC_Mqtt"
.\venv\Scripts\python run.py
```

Debe mostrar:

```
============================================================
🚀 Iniciando Servicio MQTT de Procesamiento de Imágenes
============================================================

📋 Paso 1: Validando configuración...
✅ Configuración válida

🖥️  Paso 2: Seleccionando device...
✅ Device: CPU (o CUDA)

🤖 Paso 3: Cargando modelos...
📥 Descargando modelo yoloe-v8l-seg.pt...
✅ Modelo descargado

🔧 Paso 4: Configurando procesador...
✅ Procesador configurado

📡 Paso 5: Conectando a MQTT...
✅ MQTT conectado

============================================================
✅ Servicio inicializado correctamente
   Esperando mensajes MQTT...
   Presiona Ctrl+C para detener
============================================================
```

## 🐛 Troubleshooting

### Error: "No module named 'torch'"

→ Instala torch: `.\venv\Scripts\pip install torch`

### Error: "Could not find a version that satisfies the requirement torch"

→ Usa índice específico: `.\venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu`

### GPU no detectada

→ Verifica CUDA:
```bash
.\venv\Scripts\python -c "import torch; print(torch.cuda.is_available())"
```

Si es False, instala con CPU o actualiza drivers NVIDIA.

### Conexión MQTT falla

→ Verifica credenciales en `config/config.json`:
- broker
- puerto (1883)
- usuario/contraseña

### Primera ejecución es lenta

→ Normal: descarga modelos YOLOE (~2-5 minutos)

## 📊 Tamaños Esperados de Descarga

- torch CPU: ~140 MB
- torch GPU CUDA 12.1: ~90 MB  
- torchvision: ~4-5 MB
- ultralytics: ~1 MB
- Modelos YOLOE: ~700 MB (descargado en primera ejecución)

**Total estimado**: 1-2 GB (depende de GPU/CPU)

## 📝 Archivos Importantes

```
config/config.json           ← Configuración principal
config/reference_image.jpg   ← Imagen de referencia (AGREGAR)
.env                         ← Variables de entorno
requirements-essential.txt   ← Dependencias base (instaladas)
requirements-ml.txt          ← Dependencias ML (pendiente)
install_ml_deps.py          ← Script de instalación automática
run.py                       ← Punto de entrada
src/main.py                  ← Lógica principal
```

## ✅ Verificación Final

Antes de ejecutar, verifica:

```bash
# 1. Venv activado
.\venv\Scripts\python --version

# 2. Pydantic instalado
.\venv\Scripts\python -c "import pydantic; print('✅ Pydantic OK')"

# 3. MQTT instalado
.\venv\Scripts\python -c "import paho.mqtt; print('✅ MQTT OK')"

# 4. config.json existe
if (Test-Path "config\config.json") { echo "✅ config.json OK" }

# 5. Directorio src existe
if (Test-Path "src") { echo "✅ src OK" }
```

## 🎓 Próximo: Instalar ML Deps

```bash
.\venv\Scripts\python install_ml_deps.py
```

Luego ejecuta:

```bash
.\venv\Scripts\python run.py
```

---

**¿Problemas?** Revisa `README.md` para más detalles o los logs de la consola.
