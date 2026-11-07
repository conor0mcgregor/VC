
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


CLASES = ("FRA", "PEL", "TAR")
IMAGENES_ENTRENAMIENTO = {"FRA": "FRA.png", "PEL": "PEL.png", "TAR": "TAR.png"}
UMBRALES_SEGMENTACION = {"FRA": 10, "PEL": 12, "TAR": 15}


def _estadisticas_color(pixeles, fondo):
    """Incluye intensidad, saturación y diferencia de color respecto al fondo."""
    if len(pixeles) == 0:
        pixeles = np.zeros((1, 3), dtype=np.float32)
    fondo = np.asarray(fondo, dtype=np.float32)
    pixeles = np.asarray(pixeles, dtype=np.float32)
    diferencia = pixeles - fondo
    return [
        float(np.mean(pixeles)),
        float(np.std(pixeles)),
        float(np.mean(diferencia)),
        float(np.std(diferencia)),
    ]


def extraer_caracteristicas(roi, mascara):
    """Calcula características de apariencia y geometría de una partícula."""
    alto, ancho = mascara.shape[:2]
    seleccion = mascara.astype(bool)
    if int(seleccion.sum()) < 4:
        seleccion = np.ones((alto, ancho), dtype=bool)

    borde = np.concatenate(
        (roi[0], roi[-1], roi[:, 0], roi[:, -1]), axis=0
    ).astype(np.float32)
    fondo = np.median(borde, axis=0)
    pixeles = roi[seleccion].astype(np.float32)
    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    caracteristicas = []
    for canal in (gris, hsv[:, :, 1], hsv[:, :, 2]):
        caracteristicas.extend(_estadisticas_color(canal[seleccion, None], [0])[0:2])
    # Diferencias por canal BGR: robustas frente a la iluminación del fondo.
    for canal in range(3):
        caracteristicas.extend(_estadisticas_color(pixeles[:, canal], [fondo[canal]])[2:])

    contornos, _ = cv2.findContours(
        (seleccion.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contornos:
        contorno = max(contornos, key=cv2.contourArea)
        area = cv2.contourArea(contorno)
        perimetro = cv2.arcLength(contorno, True)
        _, _, ancho_caja, alto_caja = cv2.boundingRect(contorno)
        area_hull = cv2.contourArea(cv2.convexHull(contorno))
        caracteristicas.extend(
            [
                area / max(ancho * alto, 1),
                perimetro * perimetro / max(area, 1e-6),
                ancho_caja / max(alto_caja, 1),
                area / max(ancho_caja * alto_caja, 1),
                area / max(area_hull, 1e-6),
            ]
        )
    else:
        caracteristicas.extend([0.0] * 5)

    distancia = np.linalg.norm(pixeles - fondo, axis=1)
    caracteristicas.extend(
        [
            ancho / max(alto, 1),
            float(seleccion.mean()),
            float(np.mean(distancia)),
            float(np.std(distancia)),
        ]
    )
    return np.asarray(caracteristicas, dtype=np.float32)


def segmentar_entrenamiento(imagen, umbral):
    """Obtiene una muestra por contorno de la imagen de una clase."""
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    fondo = cv2.GaussianBlur(gris, (0, 0), 20)
    contraste = cv2.absdiff(gris, fondo)
    _, mascara = cv2.threshold(contraste, umbral, 255, cv2.THRESH_BINARY)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contornos, _ = cv2.findContours(
        mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    muestras = []
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area < 70 or area > 30000:
            continue
        x, y, ancho, alto = cv2.boundingRect(contorno)
        margen = 5
        x0, y0 = max(0, x - margen), max(0, y - margen)
        x1 = min(imagen.shape[1], x + ancho + margen)
        y1 = min(imagen.shape[0], y + alto + margen)
        recorte = imagen[y0:y1, x0:x1]
        mascara_local = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        contorno_local = contorno.copy()
        contorno_local[:, :, 0] -= x0
        contorno_local[:, :, 1] -= y0
        cv2.drawContours(mascara_local, [contorno_local], -1, 1, -1)
        muestras.append(extraer_caracteristicas(recorte, mascara_local))
    return muestras


def mascara_de_recorte(recorte):
    """Segmenta una caja de test respecto a su fondo local."""
    borde = np.concatenate(
        (recorte[0], recorte[-1], recorte[:, 0], recorte[:, -1]), axis=0
    ).astype(np.float32)
    fondo = np.median(borde, axis=0)
    distancia = np.linalg.norm(recorte.astype(np.float32) - fondo, axis=2)
    _, mascara = cv2.threshold(
        distancia.astype(np.uint8), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mascara


def cargar_test(ruta_imagen, ruta_csv):
    imagen = cv2.imread(str(ruta_imagen))
    caracteristicas, etiquetas, cajas = [], [], []
    with open(ruta_csv, newline="", encoding="utf-8") as archivo:
        for fila in csv.DictReader(archivo):
            caja = tuple(
                int(fila[nombre])
                for nombre in ("x_min", "y_min", "x_max", "y_max")
            )
            x0, y0, x1, y1 = caja
            recorte = imagen[y0 : y1 + 1, x0 : x1 + 1]
            caracteristicas.append(extraer_caracteristicas(recorte, mascara_de_recorte(recorte)))
            etiquetas.append(fila["label"])
            cajas.append(caja)
    return imagen, np.asarray(caracteristicas), np.asarray(etiquetas), cajas


def entrenar_y_evaluar(carpeta=Path(".")):
    carpeta = Path(carpeta)
    caracteristicas_entrenamiento, etiquetas_entrenamiento = [], []
    for clase in CLASES:
        imagen = cv2.imread(str(carpeta / IMAGENES_ENTRENAMIENTO[clase]))
        if imagen is None:
            raise FileNotFoundError(IMAGENES_ENTRENAMIENTO[clase])
        muestras = segmentar_entrenamiento(imagen, UMBRALES_SEGMENTACION[clase])
        caracteristicas_entrenamiento.extend(muestras)
        etiquetas_entrenamiento.extend([clase] * len(muestras))
        print(f"{clase}: {len(muestras)} partículas de entrenamiento")

    imagen_test, x_test, y_test, cajas = cargar_test(
        carpeta / "MPs_test.jpg", carpeta / "MPs_test_bbs.csv"
    )
    # Las primeras componentes de apariencia absoluta dependen mucho de la
    # iluminación de cada fotografía. Se usan las características relativas
    # y geométricas (compacidad, aspecto, extensión, solidez y contraste).
    indice_modelo = slice(13, None)
    x_entrenamiento = np.asarray(caracteristicas_entrenamiento)[:, indice_modelo]
    x_test_modelo = x_test[:, indice_modelo]
    clasificador = ExtraTreesClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42,
        max_features=1.0,
    )
    clasificador.fit(x_entrenamiento, etiquetas_entrenamiento)
    predicciones = clasificador.predict(x_test_modelo)

    print(f"Accuracy: {accuracy_score(y_test, predicciones):.4f}")
    print(f"Precision weighted: {precision_score(y_test, predicciones, average='weighted', zero_division=0):.4f}")
    print(f"Recall weighted: {recall_score(y_test, predicciones, average='weighted', zero_division=0):.4f}")
    print(f"F1 weighted: {f1_score(y_test, predicciones, average='weighted', zero_division=0):.4f}")
    print(classification_report(y_test, predicciones, labels=CLASES, zero_division=0))

    matriz = confusion_matrix(y_test, predicciones, labels=CLASES)
    print("Matriz de confusión (filas reales, columnas predichas):")
    print(matriz)
    fig, ejes = plt.subplots(1, 2, figsize=(13, 7))
    sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ejes[0], xticklabels=CLASES, yticklabels=CLASES)
    ejes[0].set_xlabel("Predicha")
    ejes[0].set_ylabel("Real / anotada")
    ejes[0].set_title("Matriz de confusión")

    visual = imagen_test.copy()
    colores = {"FRA": (0, 0, 255), "PEL": (0, 255, 0), "TAR": (255, 0, 0)}
    for (x0, y0, x1, y1), real, predicha in zip(cajas, y_test, predicciones):
        color = colores[predicha]
        cv2.rectangle(visual, (x0, y0), (x1, y1), color, 2)
        cv2.putText(visual, f"{predicha}/{real}", (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    ejes[1].imshow(cv2.cvtColor(visual, cv2.COLOR_BGR2RGB))
    ejes[1].set_title("Predicción / anotación")
    ejes[1].axis("off")
    fig.tight_layout()
    salida = carpeta / "tarea2_resultado.png"
    fig.savefig(salida, dpi=160, bbox_inches="tight")
    plt.show()
    print(f"Resultado guardado en: {salida}")
    return matriz, y_test, predicciones


entrenar_y_evaluar()