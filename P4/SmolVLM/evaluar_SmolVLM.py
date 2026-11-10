"""Evalúa SmolVLM sobre el mismo dataset que el evaluador EasyOCR."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "OCR"))

from comparativa.evaluacion_comun import (  # noqa: E402
    DATA_DIR,
    calcular_analisis,
    cargar_dataset,
    crear_prediccion,
    guardar_csv,
    sincronizar_gpu,
)

from matricula_SmolVLM import DISPOSITIVO, leer_matricula  # noqa: E402


RESULTADOS_DIR = SCRIPT_DIR / "resultados"


def evaluar_smolvlm():
    filas = cargar_dataset()
    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
    for nombre in ("predicciones.csv", "analisis.csv"):
        (RESULTADOS_DIR / nombre).unlink(missing_ok=True)

    ruta_prueba = DATA_DIR / filas[0]["ruta_recorte"]
    imagen_prueba = cv2.imread(str(ruta_prueba))
    if imagen_prueba is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen de calentamiento: {ruta_prueba}")
    leer_matricula(imagen_prueba)  # carga el modelo y calienta la inferencia
    sincronizar_gpu(DISPOSITIVO)

    predicciones = []
    for indice, fila in enumerate(filas, start=1):
        ruta = DATA_DIR / fila["ruta_recorte"]
        imagen = cv2.imread(str(ruta))
        if imagen is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {ruta}")
        sincronizar_gpu(DISPOSITIVO)
        inicio = time.perf_counter()
        predicha, _, _ = leer_matricula(imagen)
        sincronizar_gpu(DISPOSITIVO)
        tiempo = time.perf_counter() - inicio
        predicciones.append(crear_prediccion(fila, predicha, tiempo))
        print(f"SmolVLM {indice}/{len(filas)}: {ruta.name}")

    analisis = calcular_analisis(predicciones, "SmolVLM", DISPOSITIVO)
    guardar_csv(RESULTADOS_DIR / "predicciones.csv", predicciones)
    guardar_csv(RESULTADOS_DIR / "analisis.csv", [analisis])
    print(f"Accuracy exacta: {analisis['accuracy_exacta_pct']:.2f}% | CER global: {analisis['cer_global']:.4f}")
    print(f"Resultados guardados en {RESULTADOS_DIR}")
    return predicciones, analisis


if __name__ == "__main__":
    evaluar_smolvlm()
