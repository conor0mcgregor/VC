import argparse
import csv
import math
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SALIDA_DIR = os.path.join(BASE_DIR, "salida")

model = YOLO(os.path.join(BASE_DIR, "best.pt"))

class Matricula:
    def __init__(self, bbox, texto, confianza):
        self.bbox = bbox
        self.texto = texto
        self.confianza = confianza

    def __str__(self):
        return f"Matricula(bbox={self.bbox}, texto='{self.texto}', confianza={self.confianza})"


def detectar_matricula(imagen):
    resultados = model.predict(imagen, verbose=False)

    matriculas = []

    for resultado in resultados:
        for det in resultado.boxes.data.numpy():
            x1, y1, x2, y2, conf, _ = det

            bbox = (int(x1), int(y1), int(x2), int(y2))
            confianza = float(conf)
            texto = "MATRICULA_DETECTADA"

            matriculas.append(Matricula(bbox, texto, confianza))

    return matriculas

def dibujar_matriculas(imagen, matriculas):
    for matricula in matriculas:
        x1, y1, x2, y2 = matricula.bbox
        cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(imagen, f"{matricula.texto} ({matricula.confianza:.2f})", 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return imagen


if __name__ == "__main__":
    ruta_imagen = os.path.join(BASE_DIR, "images/matriculas/matriculas7.jpeg")

    print(f"Procesando imagen: {ruta_imagen}")

    img_original = cv2.imread(ruta_imagen)

    if img_original is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen en: {ruta_imagen}")

    matriculas_detectadas = detectar_matricula(img_original)
    img_resultado = dibujar_matriculas(img_original, matriculas_detectadas)

    for matricula in matriculas_detectadas:
        print(matricula)

    nombre_ventana = "det matriculas"

    # Ventana redimensionable arrastrando los bordes
    cv2.namedWindow(nombre_ventana, cv2.WINDOW_NORMAL)

    # Tamaño inicial (opcional)
    cv2.resizeWindow(nombre_ventana, 1200, 700)

    cv2.imshow(nombre_ventana, img_resultado)

    cv2.waitKey(0)
    cv2.destroyAllWindows()