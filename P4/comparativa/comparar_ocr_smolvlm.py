"""Ejecuta los dos evaluadores en orden y compara sus CSV de resultados."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

from P4.OCR.evaluar_ocr import evaluar_ocr
from P4.SmolVLM.evaluar_SmolVLM import evaluar_smolvlm


P4_DIR = Path(__file__).resolve().parents[1]
OCR_RESULTADOS = P4_DIR / "OCR" / "resultados"
SMOL_RESULTADOS = P4_DIR / "SmolVLM" / "resultados"
RESULTADOS_DIR = Path(__file__).resolve().parent / "resultados"


def leer_csv(ruta):
    with ruta.open("r", encoding="utf-8-sig", newline="") as archivo:
        return list(csv.DictReader(archivo))


def guardar_csv(ruta, filas):
    with ruta.open("w", encoding="utf-8-sig", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)


def ejecutar_evaluadores():
    print("Ejecutando evaluador de SmolVLM...")
    evaluar_ocr()
    print("=========================================================")
    print("=========================================================")
    print("Ejecutando evaluador de EasyOCR...")
    evaluar_smolvlm()


def validar_mismas_imagenes(predicciones_ocr, predicciones_smol):
    ocr = {fila["imagen"]: fila for fila in predicciones_ocr}
    smol = {fila["imagen"]: fila for fila in predicciones_smol}
    if len(ocr) != len(predicciones_ocr) or len(smol) != len(predicciones_smol):
        raise ValueError("Hay nombres de imagen duplicados en uno de los CSV.")
    if ocr.keys() != smol.keys():
        faltan_smol = sorted(ocr.keys() - smol.keys())
        faltan_ocr = sorted(smol.keys() - ocr.keys())
        raise ValueError(f"Los modelos no procesaron las mismas imágenes. Solo EasyOCR: {faltan_smol}; solo SmolVLM: {faltan_ocr}")
    for imagen in ocr:
        if ocr[imagen]["matricula_real"] != smol[imagen]["matricula_real"]:
            raise ValueError(f"La etiqueta real no coincide para {imagen}.")
    return ocr, smol


def crear_graficas(resumenes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    graficas = (
        ("accuracy_exacta_pct", "Comparativa de accuracy exacta", "Accuracy exacta (%)", "comparativa_accuracy.png"),
        ("cer_global", "Comparativa de CER global", "CER (proporción de errores)", "comparativa_cer.png"),
        ("tiempo_medio_segundos", "Comparativa de tiempo medio", "Tiempo medio por imagen (s)", "comparativa_tiempos.png"),
        ("lecturas_vacias_pct", "Comparativa de lecturas vacías", "Lecturas vacías (%)", "comparativa_vacios.png"),
    )
    nombres = [fila["modelo"] for fila in resumenes]
    for clave, titulo, eje_y, archivo in graficas:
        valores = [float(fila[clave]) for fila in resumenes]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(nombres, valores, color=("#3978a8", "#e18b43"))
        ax.set_title(titulo)
        ax.set_xlabel("Modelo")
        ax.set_ylabel(eje_y)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(RESULTADOS_DIR / archivo, dpi=160)
        plt.close(fig)


def comparar_resultados():
    analisis_ocr = leer_csv(OCR_RESULTADOS / "analisis.csv")
    analisis_smol = leer_csv(SMOL_RESULTADOS / "analisis.csv")
    predicciones_ocr = leer_csv(OCR_RESULTADOS / "predicciones.csv")
    predicciones_smol = leer_csv(SMOL_RESULTADOS / "predicciones.csv")
    
    print("Comparando resultados...")
    if len(analisis_ocr) != 1 or len(analisis_smol) != 1:
        raise ValueError("Cada analisis.csv debe contener exactamente una fila.")

    ocr, smol = validar_mismas_imagenes(predicciones_ocr, predicciones_smol)
    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
    resumen = [analisis_ocr[0], analisis_smol[0]]
    guardar_csv(RESULTADOS_DIR / "comparativa.csv", resumen)

    detalle = []
    for imagen in ocr:
        a, b = ocr[imagen], smol[imagen]
        detalle.append({
            "imagen": imagen,
            "matricula_real": a["matricula_real"],
            "prediccion_easyocr": a["matricula_predicha"],
            "acierto_easyocr": a["acierto_exacto"],
            "errores_easyocr": a["errores_caracteres"],
            "cer_easyocr": a["cer_individual"],
            "tiempo_easyocr_segundos": a["tiempo_segundos"],
            "confianza_easyocr": a.get("confianza", ""),
            "prediccion_smolvlm": b["matricula_predicha"],
            "acierto_smolvlm": b["acierto_exacto"],
            "errores_smolvlm": b["errores_caracteres"],
            "cer_smolvlm": b["cer_individual"],
            "tiempo_smolvlm_segundos": b["tiempo_segundos"],
        })
    guardar_csv(RESULTADOS_DIR / "detalle_comparativa.csv", detalle)
    crear_graficas(resumen)
    print(f"\nComparativa y gráficas guardadas en {RESULTADOS_DIR}")


def main():
    ejecutar_evaluadores()
    comparar_resultados()


if __name__ == "__main__":
    main()
