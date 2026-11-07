"""OCR sencillo para recortes de matrículas."""

from __future__ import annotations

import re

import cv2
import numpy as np
import matplotlib.pyplot as plt

ALLOW = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
_reader = None


def _get_reader():
    """Carga EasyOCR una vez y reutiliza el modelo en llamadas posteriores."""
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


def _preprocesar(img: np.ndarray) -> np.ndarray:
    """Convierte el recorte a binario; no intenta abrir ventanas de OpenCV."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(
        gray,
        20,    # umbral
        255,  # valor para los píxeles > umbral
        cv2.THRESH_BINARY
    )
    # OpenCV instalado sin GUI no permite cv2.imshow; Matplotlib muestra la imagen.
    

    plt.imshow(thresh, cmap="gray")
    plt.title("Recorte preprocesado")
    plt.axis("off")
    plt.show()

    return thresh


def _es_formato_matricula(texto: str) -> bool:
    return bool(
        re.fullmatch(r"\d{4}[A-Z]{3}", texto)
        or re.fullmatch(r"[A-Z]{1,3}\d{3,4}[A-Z]{1,2}", texto)
    )


def leer_matricula(img: np.ndarray) -> tuple[str, float, bool]:
    if img is None or img.size == 0:
        return "", 0.0, False

    resultado = _get_reader().readtext(
        _preprocesar(img), allowlist=ALLOW, detail=1,
        paragraph=False, decoder="greedy", mag_ratio=1.0,
    )
    if not resultado:
        return "", 0.0, False

    # Conserva las cajas altas (matrícula) y descarta texto pequeño del marco.
    cajas = []
    for puntos, texto, confianza in resultado:
        puntos = np.asarray(puntos)
        alto = float(puntos[:, 1].max() - puntos[:, 1].min())
        centro_x = float(puntos[:, 0].mean())
        centro_y = float(puntos[:, 1].mean())
        cajas.append((alto, centro_x, centro_y, texto, float(confianza)))

    alto_max = max(caja[0] for caja in cajas)
    cajas = [caja for caja in cajas if caja[0] >= 0.55 * alto_max]

    # Las placas en dos filas se leen de arriba abajo; las demás, de izquierda a derecha.
    h, w = img.shape[:2]
    cajas.sort(key=(lambda c: (c[2], c[1])) if h > w * 1.12 else (lambda c: (c[1], c[2])))
    texto = re.sub(r"[^A-Z0-9]", "", "".join(caja[3] for caja in cajas).upper())
    confianza = sum(caja[4] for caja in cajas) / len(cajas)
    return texto, confianza, _es_formato_matricula(texto)


def extraer_texto(img: np.ndarray) -> str:
    """Devuelve solo el texto reconocido y normalizado."""
    texto, _, _ = leer_matricula(img)
    return texto


def leer_matricula_easyocr(img: np.ndarray) -> tuple[str, float, bool]:
    """Alias conservado para código que usaba el nombre anterior."""
    return leer_matricula(img)
