# PDI • Ejecución por partes con IPython 

## 1) Requisitos

- **Python 3.12+** (sirve 3.13). Verificá con:
  ```bash
  python --version
  ```
- **Git** (opcional, para clonar el repo)
- **Windows / Linux / macOS**

> En Windows, usá **PowerShell** o **CMD**. En Linux/macOS, la terminal del sistema.

---

## 2) Clonar el repo (o ubicarse en tu carpeta)

### Opción A – Clonar el repo
```bash
# con HTTPS
git clone https://github.com/Lauaf/facu_cosas.git
cd facu_cosas
```

### Opción B – Ya tenés la carpeta local
```bash
cd "C:/ruta/a/tu/carpeta/facu"   # Windows (PowerShell)
# o
cd /ruta/a/tu/carpeta/facu        # Linux/macOS
```

> **Siempre** corré los comandos desde **la carpeta del proyecto** (donde están tus .py e imágenes).

---

## 3) Crear y activar el entorno virtual (venv)

```bash
# crear venv en la carpeta del proyecto
python -m venv .venv

# activar
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Windows (CMD)
.venv\Scripts\activate.bat
# Windows (Git Bash)
source .venv/Scripts/activate
# Linux/macOS
source .venv/bin/activate
```

Actualizá pip e instalá dependencias:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Si no existe `requirements.txt`, crealo así
Crea `requirements.txt` con este contenido:
```
numpy
matplotlib
opencv-contrib-python
Pillow
```
Luego:
```bash
pip install -r requirements.txt
```

> `opencv-contrib-python` incluye extras que se usan en la cátedra. `Pillow` se usa para guardar/leer imágenes en varios formatos.


## 4) **IPython** para ejecutar por partes (Shift+Enter)

Instalá y abrí **IPython** dentro del **venv activado** y **parado en la carpeta del proyecto** (la carpeta con el codigo a ejecutar ej.: C:\Users\...\Escritorio\facu\src\Código-U1) :
```bash
pip install ipython     # una sola vez
ipython
```

Confirmá dónde estás y qué archivos hay:
```python
import os
os.getcwd()      # carpeta actual
os.listdir()     # archivos visibles (deberías ver xray-chest.png / cameraman.tif)
```

ahora solo queda abrir el archivo a probar, copias las lineas a ejecutar y las pegas en la consola corriendo iPython

En mi caso:
```python
PS C:\Users\lflorenza\...\Escritorio\facu> .\.venv\Scripts\activate
(.venv) PS C:\Users\lflorenza\...Escritorio\facu> cd .\src\
(.venv) PS C:\Users\lflorenza\...\Escritorio\facu\src> cd .\Código-U1\
(.venv) PS C:\Users\lflorenza\...\Escritorio\facu\src\Código-U1> ipython
Python 3.13.5 (tags/v3.13.5:6cb20a2, Jun 11 2025, 16:15:46) [MSC v.1943 64 bit (AMD64)]
Type 'copyright', 'credits' or 'license' for more information
IPython 9.6.0 -- An enhanced Interactive Python. Type '?' for help.
Tip: You can use `%hist` to view history, see the options with `%history?`
Ctrl click to launch VS Code Native REPL

In [1]: import cv2
   ...: import matplotlib.pyplot as plt
   ...: import numpy as np

In [2]: img1 = cv2.imread('xray-chest.png',cv2.IMREAD_GRAYSCALE)
   ...: img2 = cv2.imread('cameraman.tif',cv2.IMREAD_GRAYSCALE)

In [3]: h = plt.imshow(img1, cmap='gray')
   ...: plt.colorbar(h)
   ...:
   ...: plt.figure(2)
   ...: h = plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
   ...: plt.colorbar(h)
   ...:
   ...: plt.show()

In [4]:
```