# OCR independiente para matrículas

El módulo recibe un recorte BGR de OpenCV. No depende del detector ni modifica
ningún módulo externo. EasyOCR se carga al primer reconocimiento; conviene crear
el lector una sola vez y reutilizarlo durante el procesamiento del vídeo.

## Dependencias

```powershell
py -m pip install easyocr opencv-python numpy
```

## Uso desde otro módulo

```python
import cv2
from P4.OCR.matricula_ocr import extraer_texto

recorte = cv2.imread("matricula.png")  # imagen BGR, ya recortada
texto = extraer_texto(recorte)
print(texto)  # por ejemplo: 1234BCD
```

`extraer_texto` devuelve una cadena en mayúsculas, sin separadores. Para obtener
también la confianza OCR y la validación de formato:

```python
from P4.OCR.matricula_ocr import leer_matricula

texto, confianza, formato_valido = leer_matricula(recorte)
```

## Evaluación

Desde la raíz del proyecto:

```powershell
py P4/OCR/evaluar_dataset.py
```

El script lee todas las filas de `P4/datos/matriculas_recortadas/ocr_dataset.csv`
y muestra en consola la etiqueta real y la predicción. No guarda resultados.

El extractor amplía recortes pequeños, aplica gris y CLAHE, ejecuta EasyOCR una
vez y une las detecciones principales. El lector se carga una vez y se reutiliza.

El rendimiento depende del equipo y del hardware disponible.

