#!/usr/bin/env python
"""
Script de instalación interactiva para dependencias de ML
Ejecutar con: python install_ml_deps.py
"""
import subprocess
import sys

def run_command(cmd):
    """Ejecutar comando en terminal"""
    print(f"\n{'='*60}")
    print(f"Ejecutando: {cmd}")
    print('='*60)
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0

def main():
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║  INSTALADOR DE DEPENDENCIAS ML - Servicio MQTT            ║
    ║  (torch, torchvision, ultralytics, supervision, etc)       ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    print("\n¿Qué dispositivo usarás? (responde 1, 2, 3 o 4)")
    print("1. CPU solo (más lento, pero funciona sin GPU)")
    print("2. GPU NVIDIA CUDA 11.8 (recomendado para RTX 30/40 series)")
    print("3. GPU NVIDIA CUDA 12.1 (para GPUs más nuevas)")
    print("4. Instalar solo torch CPU como prueba")
    
    choice = input("\nSelecciona opción (1-4): ").strip()
    
    venv_pip = r".\venv\Scripts\pip.exe"
    
    if choice == "1":
        print("\n📦 Instalando torch para CPU...")
        run_command(f"{venv_pip} install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
        
    elif choice == "2":
        print("\n📦 Instalando torch para GPU CUDA 11.8...")
        run_command(f"{venv_pip} install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        
    elif choice == "3":
        print("\n📦 Instalando torch para GPU CUDA 12.1...")
        run_command(f"{venv_pip} install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        
    elif choice == "4":
        print("\n📦 Instalando torch CPU solo como prueba...")
        run_command(f"{venv_pip} install torch")
    else:
        print("❌ Opción inválida")
        return 1
    
    # Instalar resto de dependencias ML
    print("\n📦 Instalando dependencias de ML adicionales...")
    ml_packages = [
        "ultralytics>=8.0.0",
        "opencv-python",
        "scipy",
        "matplotlib",
        "supervision",
        "huggingface-hub",
    ]
    
    for package in ml_packages:
        print(f"\n  ➤ Instalando {package}...")
        run_command(f"{venv_pip} install {package}")
    
    print("""
    
    ✅ INSTALACIÓN COMPLETADA
    
    Próximos pasos:
    1. Coloca imagen de referencia en: config/reference_image.jpg
    2. Ejecuta el servicio: python run.py
    3. Envía mensaje de prueba desde MQTTX al broker
    
    """)
    return 0

if __name__ == "__main__":
    sys.exit(main())
