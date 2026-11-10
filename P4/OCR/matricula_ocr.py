"""OCR sencillo para recortes de matrículas."""

from __future__ import annotations

import re

import cv2
import numpy as np
import matplotlib.pyplot as plt

ALLOW = "AEIOUBCDFGHJKLMNPRSTVWXYZ0123456789"
_reader = None


def _get_reader():
    """Carga EasyOCR una vez y reutiliza el modelo en llamadas posteriores."""
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


def _preprocesar(img: np.ndarray) -> np.ndarray:
    """Amplía y mejora el contraste para las pruebas visuales del OCR."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, d=11, sigmaColor=17, sigmaSpace=17)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    if np.mean(thresh) < 127:
        thresh = 255 - thresh
    return thresh

def _es_formato_matricula(texto: str) -> bool:
    return len(texto) > 4

def leer_matricula(img: np.ndarray) -> tuple[str, float, bool]:
    if img is None or img.size == 0:
        return "", 0.0, False

    resultado = _get_reader().readtext(
        img,
        allowlist=ALLOW,
        paragraph=False,
        decoder="greedy",
        mag_ratio=1.5,
        width_ths=0.7,
        height_ths=0.7,
    )

    # Cada caja: (centro_x, centro_y, alto, texto, confianza)
    cajas = []
    for puntos, texto, conf in resultado:
        if conf < 0.15:
            continue
        p = np.asarray(puntos)
        cajas.append((
            p[:, 0].mean(),
            p[:, 1].mean(),
            p[:, 1].max() - p[:, 1].min(),
            texto,
            conf,
        ))

    if not cajas:
        return "", 0.0, False

    # Descarta cajas mucho más pequeñas que la mayor (ruido, tornillos, textos auxiliares)
    alto_max = max(c[2] for c in cajas)
    cajas = [c for c in cajas if c[2] >= 0.55 * alto_max]

    # Orden de lectura: de izquierda a derecha
    cajas.sort(key=lambda c: (c[0], c[1]))

    texto = re.sub(r"[^A-Z0-9]", "", "".join(c[3] for c in cajas).upper())
    confianza = float(np.mean([c[4] for c in cajas]))

    return texto, confianza, _es_formato_matricula(texto)

def extraer_texto(img: np.ndarray) -> str:
    """Devuelve solo el texto reconocido y normalizado."""
    texto, _, _ = leer_matricula(img)
    return texto


def leer_matricula_easyocr(img: np.ndarray) -> tuple[str, float, bool]:
    """Alias conservado para código que usaba el nombre anterior."""
    return leer_matricula(img)


def leer_matricula_para_prueba(img: np.ndarray):
    """Reconoce sobre la imagen preprocesada y devuelve texto e imagen usada."""
    if img is None or img.size == 0:
        return "", None
    procesada = _preprocesar(img)
    resultado = _get_reader().readtext(
        procesada, allowlist=ALLOW, detail=1, paragraph=False,
        decoder="greedy", mag_ratio=1.0,
    )
    if not resultado:
        return "", procesada
    cajas = []
    for puntos, texto, confianza in resultado:
        puntos = np.asarray(puntos)
        cajas.append((
            float(puntos[:, 1].max() - puntos[:, 1].min()),
            float(puntos[:, 0].mean()), float(puntos[:, 1].mean()), texto,
        ))
    alto_max = max(caja[0] for caja in cajas)
    cajas = [caja for caja in cajas if caja[0] >= 0.55 * alto_max]
    h, w = img.shape[:2]
    cajas.sort(key=(lambda caja: (caja[2], caja[1])) if h > w * 1.12 else (lambda caja: (caja[1], caja[2])))
    texto = re.sub(r"[^A-Z0-9]", "", "".join(caja[3] for caja in cajas).upper())
    return texto, procesada


def mostrar_comparativa(img: np.ndarray, procesada: np.ndarray) -> None:
    """Muestra el recorte y la versión preprocesada en una ventana Matplotlib."""
    import matplotlib.pyplot as plt

    fig, ejes = plt.subplots(1, 2, figsize=(10, 5))
    ejes[0].imshow(procesada, cmap="gray")
    ejes[0].set_title("Preprocesada")
    ejes[0].axis("off")
    ejes[1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ejes[1].set_title("Original")
    ejes[1].axis("off")
    fig.tight_layout()
    plt.show()


def comprobar_evaluacion(img: np.ndarray, etiqueta: str):
    """Muestra una comprobación manual y devuelve el texto reconocido."""
    texto, procesada = leer_matricula_para_prueba(img)
    print(f"| real: {etiqueta} | predicha: {texto} | {texto == etiqueta}")
    if procesada is not None:
        mostrar_comparativa(img, procesada)
    return texto


def evaluar(img: np.ndarray, etiqueta: str):
    """Devuelve predicción, acierto exacto y confianza EasyOCR."""
    texto, confianza, _ = leer_matricula(img)
    return texto, texto == etiqueta, confianza
