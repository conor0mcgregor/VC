"""Imprime la etiqueta real y la predicción para cada fila del dataset."""

import csv
from pathlib import Path

import cv2

from matricula_ocr import extraer_texto
from matricula_ocr import comprobar_evaluacion
from matricula_ocr import evaluar
from matricula_ocr import evaluacion_rapida


OCR_DIR = Path(__file__).resolve().parent
DATA_DIR = OCR_DIR.parent / "datos" / "matriculas_recortadas"
CSV_PATH = DATA_DIR / "ocr_dataset.csv"

def apuntar_fallos(real, predicha, confianza):
    """Guarda en un archivo de texto los fallos de predicción."""
    with open(OCR_DIR / "fallos.txt", "a", encoding="utf-8") as f:
        f.write(f"real: {real} | predicha: {predicha} | confianza: {confianza}\n")

def main() -> None:
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as archivo:
        filas = list(csv.DictReader(archivo, delimiter=";"))

    test = 1
    if test:
        asiertos = 0
        for indice, fila in enumerate(filas, start=1):
            ruta = DATA_DIR / fila["ruta_recorte"]
            imagen = cv2.imread(str(ruta))
            real = fila["matricula"].strip().upper()

            predicha, asierto, confianza = evaluacion_rapida(imagen, real)
            if asierto: 
                asiertos += 1 
            else:
                apuntar_fallos(real, predicha, confianza)
            print(f"{indice}/{len(filas)} | real: {real} | predicha: {predicha} | {asierto} | confianza: {confianza} | asiertos: {asiertos}/{indice} |")
    else:
        i = 0
        while True:
            fila = filas[i]
            ruta = DATA_DIR / fila["ruta_recorte"]
            imagen = cv2.imread(str(ruta))
            real = fila["matricula"].strip().upper()
            comprobar_evaluacion(imagen, real)
            i += 1

if __name__ == "__main__":
    main()
