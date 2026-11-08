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
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Ampliar la imagen para facilitar el OCR
    gray = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    # Reducir ruido preservando bordes
    gray = cv2.bilateralFilter(
        gray,
        d=11,
        sigmaColor=17,
        sigmaSpace=17
    )

    # Mejorar contraste local
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    gray = clahe.apply(gray)

    # Binarización automática
    thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    # Limpiar pequeños ruidos
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel
    )

    # Texto negro sobre fondo blanco
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

#PARA EVALUAR--------------------------------------------------
def mostrar_comparativa(img, thresh):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(thresh, cmap="gray")
    ax[0].set_title("Thresh")
    ax[0].axis("off")

    ax[1].imshow(img_rgb)
    ax[1].set_title("Original")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()


def comprobar_evaluacion(img: np.ndarray, etiqueta: str):
    """Devuelve True si la predicción coincide con la matrícula real."""
    texto, imagen = leer_matricula_para_prueba(img)
    print(f"| real: {etiqueta} | predicha: {texto} | {texto == etiqueta}")
    mostrar_comparativa(img, imagen)
    return texto



def leer_matricula_para_prueba(img: np.ndarray):
    if img is None or img.size == 0:
        return "", 0.0, False

    img_procesada = _preprocesar(img)
    resultado = _get_reader().readtext(
        img_procesada, allowlist=ALLOW, detail=1,
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
    return texto, img_procesada


def evaluar(img, etiqueta):
    """Devuelve True si la predicción coincide con la matrícula real."""
    texto, confiaza, _= leer_matricula(img)
    return texto, texto == etiqueta, confiaza
