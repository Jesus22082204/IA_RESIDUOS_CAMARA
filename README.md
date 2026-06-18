# 🤖 Sistema Inteligente de Clasificación de Residuos

> Proyecto de Inteligencia Artificial y Visión Computacional para la identificación automática de materiales y control de compartimentos de reciclaje.

<div align="center">
  <!-- Reemplaza las URLs de abajo con tus capturas reales -->
  <img src="img/demo_clasificacion.png" alt="Demo de IA" width="800"/>
  <br/>
  <i>Demo del sistema identificando residuos en tiempo real</i>
</div>

## 🎯 Logros Técnicos Destacados
* **Visión por Computadora (Computer Vision):** Entrenamiento y despliegue de un modelo de IA capaz de clasificar múltiples tipos de residuos (plástico, papel, vidrio, orgánicos) a partir de imágenes en tiempo real.
* **Integración Hardware-Software:** El software no solo detecta el tipo de residuo, sino que envía señales de control a un microcontrolador (Arduino/ESP32) para abrir automáticamente el compartimento de basura correspondiente.
* **Pre-procesamiento de Datos:** Manejo de grandes volúmenes de datos e imágenes estructuradas mediante scripts de Python para entrenamiento, validación y pruebas.

---

## 📸 Galería de Funciones (Screenshots)

<div align="center">
  <img src="img/entrenamiento_modelo.png" alt="Métricas de Entrenamiento" width="400"/>
  <img src="img/hardware_setup.png" alt="Prototipo Físico" width="400"/>
</div>

*(Agrega aquí capturas de pantalla de la terminal entrenando el modelo, y fotos del prototipo físico que abre los compartimentos)*

---

## 🛠️ Stack Tecnológico
- **Lenguaje Principal:** Python.
- **Machine Learning / AI:** TensorFlow / Keras / OpenCV (Librerías de procesamiento y visión artificial).
- **Control Hardware:** PySerial (para comunicación con microcontroladores Arduino).
- **Datos:** Dataset de clasificación de basura (formateado en `.txt` y `.zip`).

---

## 🗂️ Estructura del Proyecto
```text
IA/
├── Garbage classification/     # Imágenes clasificadas por categorías para el dataset
├── proyecto_residuos/          # Scripts principales de Python para predicción y control
├── archive/                    # Recursos adicionales
├── venv_residuos/              # Entorno virtual de Python
├── zero-indexed-files.txt      # Metadatos para el set de entrenamiento
└── one-indexed-files-*.txt     # Metadatos para el set de validación y testeo
```

---

## 🚀 Instalación y Uso Local

### 1. Preparar Entorno Virtual
Se recomienda usar el entorno virtual proveído o crear uno nuevo para instalar las dependencias:
```bash
# En Windows
python -m venv venv_residuos
.\venv_residuos\Scripts\activate
```

### 2. Instalar Dependencias
```bash
# (Asegúrate de exportar un requirements.txt desde tu proyecto)
pip install -r requirements.txt
```

### 3. Ejecutar el Modelo
Navega a la carpeta de ejecución e inicia el script principal (ejemplo genérico):
```bash
cd proyecto_residuos
python main.py
```
*(El script abrirá la cámara conectada o procesará una imagen predeterminada, y enviará la señal serial al microcontrolador conectado).*
