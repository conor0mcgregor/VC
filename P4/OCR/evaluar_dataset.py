"""Imprime la etiqueta real y la predicción para cada fila del dataset."""

import csv
from pathlib import Path

import cv2

from matricula_ocr import extraer_texto


OCR_DIR = Path(__file__).resolve().parent
DATA_DIR = OCR_DIR.parent / "datos" / "matriculas_recortadas"
CSV_PATH = DATA_DIR / "ocr_dataset.csv"


def main() -> None:
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as archivo:
        filas = list(csv.DictReader(archivo, delimiter=";"))

    if False:
        for indice, fila in enumerate(filas, start=1):
            ruta = DATA_DIR / fila["ruta_recorte"]
            imagen = cv2.imread(str(ruta))
            predicha = extraer_texto(imagen) if imagen is not None else ""
            real = fila["matricula"].strip().upper()
            print(f"{indice}/{len(filas)} | real: {real} | predicha: {predicha}")

    fila = filas[7]
    ruta = DATA_DIR / fila["ruta_recorte"]
    imagen = cv2.imread(str(ruta))
    predicha = extraer_texto(imagen) if imagen is not None else ""
    real = fila["matricula"].strip().upper()
    print(f"{0}/{len(filas)} | real: {real} | predicha: {predicha}")


if __name__ == "__main__":
    main()
