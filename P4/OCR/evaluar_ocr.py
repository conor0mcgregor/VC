import os
import cv2
import pandas as pd
from P4.OCR.prueba_ocr import leer_matricula_easyocr

BASE = "datos/matriculas_recortadas"          # carpeta que contiene ocr_dataset.csv y todas/


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def evaluar(df):
    filas = []
    for r in df.itertuples():
        img = cv2.imread(os.path.join(BASE, r.ruta_recorte))
        if img is None:
            print("No se pudo cargar", r.ruta_recorte)
            continue
        texto, conf, valida = leer_matricula_easyocr(img)
        filas.append(dict(ruta=r.ruta_recorte, split=r.split, formato=r.formato,
                          dificultad=r.dificultad, real=r.matricula, pred=texto,
                          conf=conf, valida=valida, ok=texto == r.matricula,
                          cer=levenshtein(texto, r.matricula) / len(r.matricula)))
    return pd.DataFrame(filas)


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(BASE, "ocr_dataset.csv"), sep=";")
    df = df[df.seguridad_texto == "alta"]          # fuera las dudosas
    res = evaluar(df[df.split.isin(["train", "valid"])])   # test NO se toca todavía

    print(res.groupby("dificultad")[["ok", "cer"]].mean())
    print(res.groupby("formato")[["ok", "cer"]].mean())
    print(f"\nExactas: {res.ok.sum()}/{len(res)}  CER medio: {res.cer.mean():.3f}")
    res[~res.ok].sort_values("cer", ascending=False).to_csv("errores.csv", index=False)