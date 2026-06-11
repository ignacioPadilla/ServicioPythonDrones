#!/usr/bin/env python
"""
Script de ejecución para el Servicio MQTT de Procesamiento de Imágenes
"""
import sys
from pathlib import Path

# Agregar directorio src al path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.main import main

if __name__ == "__main__":
    main()
