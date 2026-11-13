
from __future__ import annotations

import cv2
import numpy as np


def _limitar_caja(caja, ancho: int, alto: int):

    if caja is None or len(caja) != 4:
        return None

    x1, y1, x2, y2 = (int(round(float(valor))) for valor in caja)
    x1 = max(0, min(x1, ancho))
    x2 = max(0, min(x2, ancho))
    y1 = max(0, min(y1, alto))
    y2 = max(0, min(y2, alto))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def pixelar_region(imagen: np.ndarray, caja, bloque: int | None = None) -> np.ndarray:
    if imagen is None or imagen.size == 0:
        return imagen

    alto, ancho = imagen.shape[:2]
    limites = _limitar_caja(caja, ancho, alto)
    if limites is None:
        return imagen

    x1, y1, x2, y2 = limites
    region = imagen[y1:y2, x1:x2]
    if region.size == 0:
        return imagen

    alto_region, ancho_region = region.shape[:2]
    if bloque is None:
        bloque = max(8, min(alto_region, ancho_region) // 6)
    else:
        bloque = max(1, int(bloque))

    columnas = max(1, ancho_region // bloque)
    filas = max(1, alto_region // bloque)

    reducida = cv2.resize(
        region, (columnas, filas), interpolation=cv2.INTER_AREA
    )
    pixelada = cv2.resize(
        reducida, (ancho_region, alto_region), interpolation=cv2.INTER_NEAREST
    )
    imagen[y1:y2, x1:x2] = pixelada
    return imagen


def pixelar_cajas(imagen: np.ndarray, cajas, bloque: int | None = None) -> np.ndarray:
    if imagen is None or imagen.size == 0:
        return imagen

    for caja in cajas or []:
        imagen = pixelar_region(imagen, caja, bloque=bloque)
    return imagen
