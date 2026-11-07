import os
import cv2
import easyocr

reader = easyocr.Reader(["en"], gpu=False)   # gpu=True si tienes CUDA
ALLOW = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

import re
import numpy as np

LETRAS = "BCDFGHJKLMNPRSTVWXYZ"          # matrícula española actual: sin vocales ni Ñ/Q
A_NUM = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2",
         "S": "5", "G": "6", "B": "8"}
A_LET = {"0": "D", "1": "L", "2": "Z", "5": "S", "6": "G", "8": "B", "4": "H"}


def variantes(img, recortar_banda_azul=True):
    """Genera varias versiones preprocesadas de la imagen de la matrícula."""
    if recortar_banda_azul:
        h, w = img.shape[:2]
        img = img[:, int(w * 0.10):]      # quita la banda azul "E" de la izquierda

    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gris = cv2.resize(gris, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gris)
    _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adapt = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 31, 10)

    yield gris
    yield clahe
    yield otsu
    yield cv2.bitwise_not(otsu)
    yield adapt


def normalizar(bruto):
    """Limpia el texto y corrige confusiones típicas según la posición (formato 0000BBB)."""
    t = re.sub(r"[^A-Z0-9]", "", bruto.upper())
    if len(t) == 7:
        corregido = "".join(
            A_NUM.get(c, c) if i < 4 else A_LET.get(c, c)
            for i, c in enumerate(t)
        )
        if re.fullmatch(rf"\d{{4}}[{LETRAS}]{{3}}", corregido):
            return corregido, True
        return corregido, False
    return t, False

def leer_matricula_easyocr(img):
    """Prueba varias variantes y devuelve (texto, confianza, válida)."""
    mejor = ("", 0.0, False)
    for var in variantes(img, recortar_banda_azul=True):
        res = reader.readtext(var, allowlist=ALLOW, detail=1,
                              paragraph=False, decoder="greedy")
        if not res:
            continue
        bruto = "".join(t for _, t, _ in res)
        conf = sum(c for _, _, c in res) / len(res)
        texto, valida = normalizar(bruto)
        if (valida, conf) > (mejor[2], mejor[1]):
            mejor = (texto, conf, valida)
    return mejor
