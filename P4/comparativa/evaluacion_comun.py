"""Funciones pequeñas compartidas por los dos evaluadores."""

from __future__ import annotations

import csv
import re
from pathlib import Path


P4_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = P4_DIR / "datos" / "matriculas_recortadas"
DATASET_CSV = DATA_DIR / "ocr_dataset.csv"


def cargar_dataset():
    with DATASET_CSV.open("r", encoding="utf-8-sig", newline="") as archivo:
        filas = list(csv.DictReader(archivo, delimiter=";"))
    if not filas:
        raise ValueError(f"El dataset está vacío: {DATASET_CSV}")
    return filas


def normalizar(texto):
    return re.sub(r"[^A-Z0-9]", "", str(texto or "").upper())


def distancia_caracteres(real, predicha):
    anterior = list(range(len(predicha) + 1))
    for i, caracter_real in enumerate(real, start=1):
        actual = [i]
        for j, caracter_predicho in enumerate(predicha, start=1):
            actual.append(min(anterior[j] + 1, actual[j - 1] + 1,
                              anterior[j - 1] + (caracter_real != caracter_predicho)))
        anterior = actual
    return anterior[-1]


def crear_prediccion(fila, predicha, tiempo, confianza=""):
    real = normalizar(fila["matricula"])
    predicha = normalizar(predicha)
    errores = distancia_caracteres(real, predicha)
    return {
        "imagen": fila["ruta_recorte"],
        "matricula_real": real,
        "matricula_predicha": predicha,
        "acierto_exacto": "sí" if real == predicha else "no",
        "errores_caracteres": errores,
        "cer_individual": errores / len(real) if real else (0.0 if not predicha else 1.0),
        "tiempo_segundos": tiempo,
        "confianza": confianza,
    }


def calcular_analisis(predicciones, modelo, dispositivo):
    cantidad = len(predicciones)
    caracteres_reales = sum(len(fila["matricula_real"]) for fila in predicciones)
    errores = sum(int(fila["errores_caracteres"]) for fila in predicciones)
    aciertos = sum(fila["acierto_exacto"] == "sí" for fila in predicciones)
    vacios = sum(fila["matricula_predicha"] == "" for fila in predicciones)
    tiempo_total = sum(float(fila["tiempo_segundos"]) for fila in predicciones)
    cer = errores / caracteres_reales if caracteres_reales else 0.0
    return {
        "modelo": modelo,
        "dispositivo": dispositivo,
        "imagenes_procesadas": cantidad,
        "aciertos_exactos": aciertos,
        "accuracy_exacta_pct": 100 * aciertos / cantidad if cantidad else 0.0,
        "lecturas_vacias": vacios,
        "lecturas_vacias_pct": 100 * vacios / cantidad if cantidad else 0.0,
        "errores_caracteres": errores,
        "caracteres_reales": caracteres_reales,
        "cer_global": cer,
        "accuracy_caracteres": 1 - cer,
        "tiempo_total_segundos": tiempo_total,
        "tiempo_medio_segundos": tiempo_total / cantidad if cantidad else 0.0,
    }


def guardar_csv(ruta, filas):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8-sig", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)


def sincronizar_gpu(dispositivo):
    if str(dispositivo).startswith("cuda"):
        import torch
        torch.cuda.synchronize()
