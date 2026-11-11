"""Evalúa EasyOCR sobre el dataset común de recortes de matrículas."""

from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2

from comparativa.evaluacion_comun import DATA_DIR, calcular_analisis, cargar_dataset, crear_prediccion, guardar_csv
from OCR.matricula_ocr import leer_matricula


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTADOS_DIR = SCRIPT_DIR / "resultados"


def evaluar_ocr():
    filas = cargar_dataset()
    # Evita dejar resultados antiguos si la evaluación actual falla.
    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
    for nombre in ("predicciones.csv", "analisis.csv"):
        (RESULTADOS_DIR / nombre).unlink(missing_ok=True)

    ruta_prueba = DATA_DIR / filas[0]["ruta_recorte"]
    imagen_prueba = cv2.imread(str(ruta_prueba))
    if imagen_prueba is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen de calentamiento: {ruta_prueba}")
    leer_matricula(imagen_prueba)  # warm-up fuera del tiempo medido

    predicciones = []
    for indice, fila in enumerate(filas, start=1):
        ruta = DATA_DIR / fila["ruta_recorte"]
        imagen = cv2.imread(str(ruta))
        if imagen is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {ruta}")
        inicio = time.perf_counter()
        predicha, confianza, _ = leer_matricula(imagen)
        tiempo = time.perf_counter() - inicio
        predicciones.append(crear_prediccion(fila, predicha, tiempo, confianza))
        print(f"EasyOCR {indice}/{len(filas)}: {ruta.name}")

    analisis = calcular_analisis(predicciones, "EasyOCR", "cpu")
    guardar_csv(RESULTADOS_DIR / "predicciones.csv", predicciones)
    guardar_csv(RESULTADOS_DIR / "analisis.csv", [analisis])
    print(f"Accuracy exacta: {analisis['accuracy_exacta_pct']:.2f}% | CER global: {analisis['cer_global']:.4f}")
    print(f"Resultados guardados en {RESULTADOS_DIR}")
    return predicciones, analisis


if __name__ == "__main__":
    evaluar_ocr()
