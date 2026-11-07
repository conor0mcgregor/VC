"""
matricula_ocr.py
================
Extractor de texto de matrículas españolas optimizado para imágenes recortadas.

Estrategia:
  1. Genera varias variantes preprocesadas de la imagen (escala, contraste, umbral...).
  2. Para cada variante pasa por EasyOCR con allowlist estricta.
  3. Aplica corrección posicional (confusiones O/0, I/1, S/5…) según el formato.
  4. Valida contra los formatos legales españoles conocidos.
  5. Escoge el resultado con mayor confianza + validez mediante votación ponderada.

Uso como módulo:
    from P4.OCR.matricula_ocr import leer_matricula
    texto, confianza, valida = leer_matricula(img_bgr)

Uso como script de evaluación:
    py -3 P4/OCR/matricula_ocr.py [--csv ruta_dataset.csv] [--base ruta_carpeta]
"""

from __future__ import annotations

import os
import re
import sys
from collections import Counter
from typing import Generator

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# EasyOCR (carga perezosa para no ralentizar imports en el vídeo)
# ---------------------------------------------------------------------------
_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr  # noqa: PLC0415
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


# ---------------------------------------------------------------------------
# Constantes del formato matrícula española
# ---------------------------------------------------------------------------
ALLOW = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Letras válidas en posición de letra de matrícula actual (sin vocales AEIOU ni Ñ ni Q)
_LETRAS_ACTUAL = "BCDFGHJKLMNPRSTVWXYZ"

# Sustituciones visuales dígito → letra y letra → dígito
_A_NUM: dict[str, str] = {
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "L": "1",
    "Z": "2",
    "S": "5",
    "G": "6",
    "B": "8",
}
_A_LET: dict[str, str] = {
    "0": "O",   # en posición de letra se prefiere O→ pero O no está en LETRAS_ACTUAL;
                # en la práctica 0→B o 0→D resultan menos frecuentes; se trata caso a caso.
    "1": "L",
    "2": "Z",
    "4": "H",
    "5": "S",
    "6": "G",
    "8": "B",
}

# Patrones de matrícula
_PAT_ACTUAL    = re.compile(rf"^\d{{4}}[{_LETRAS_ACTUAL}]{{3}}$")          # 1234BBB
_PAT_CON_VOCAL = re.compile(r"^\d{4}[A-Z]{3}$")                           # 1234AXP (con vocal)
_PAT_PROVINCIAL = re.compile(r"^[A-Z]{1,2}\d{4}[A-Z]{2}$")               # GC4370BV / M1234AB
_PAT_ESPECIAL  = re.compile(r"^[A-Z]{2,3}\d{3,4}[A-Z]{1,2}$")            # CNP2309AW u otras


def _valida(texto: str) -> bool:
    return bool(
        _PAT_ACTUAL.match(texto)
        or _PAT_CON_VOCAL.match(texto)
        or _PAT_PROVINCIAL.match(texto)
        or _PAT_ESPECIAL.match(texto)
    )


# ---------------------------------------------------------------------------
# Corrección posicional según formato detectado
# ---------------------------------------------------------------------------

def _corregir_actual(t: str) -> str:
    """Corrige confusiones en formato 4D+3L (matrícula actual española)."""
    if len(t) != 7:
        return t
    return "".join(
        _A_NUM.get(c, c) if i < 4 else _A_LET.get(c, c)
        for i, c in enumerate(t)
    )


def _corregir_provincial(t: str) -> str:
    """Corrige confusiones en formato prefijo+4D+2L (provincial/especial)."""
    # Detecta el límite entre prefijo alfabético y bloque numérico
    m = re.match(r"^([A-Z]{1,3})(\d{3,4})([A-Z]{1,2})$", t)
    if not m:
        return t
    pref, nums, suf = m.group(1), m.group(2), m.group(3)
    nums_c = "".join(_A_NUM.get(c, c) for c in nums)
    suf_c  = "".join(_A_LET.get(c, c) for c in suf)
    return pref + nums_c + suf_c


def normalizar(bruto: str) -> tuple[str, bool]:
    """Limpia y corrige el texto OCR; devuelve (texto_corregido, es_válida)."""
    t = re.sub(r"[^A-Z0-9]", "", bruto.upper())

    if len(t) == 7:
        c = _corregir_actual(t)
        return c, _valida(c)

    if 7 <= len(t) <= 9:
        c = _corregir_provincial(t)
        return c, _valida(c)

    return t, False


# ---------------------------------------------------------------------------
# Preprocesado de imagen
# ---------------------------------------------------------------------------

def _quitar_banda_azul(img: np.ndarray) -> np.ndarray:
    """
    Elimina la franja azul europea del lado izquierdo de la matrícula.
    Busca el pico de densidad azul (máximo ratio) dentro del primer 25% del ancho
    y recorta hasta el final de esa zona. Si no hay zona azul clara, devuelve
    un recorte mínimo del 3% para eliminar el borde de píxeles oscuros.
    """
    h, w = img.shape[:2]
    max_buscar = int(w * 0.25)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mascara_azul = cv2.inRange(hsv, (95, 60, 50), (135, 255, 255))

    ratios = np.array([mascara_azul[:, col].mean() / 255.0
                       for col in range(max_buscar)])

    # Si hay zona azul significativa (pico > 25%)
    if ratios.max() > 0.25:
        # Recortar hasta la última columna con ratio > 0.10
        col_fin = int(w * 0.03)
        for col in range(max_buscar - 1, 0, -1):
            if ratios[col] > 0.10:
                col_fin = col + 2  # +2 margen de seguridad
                break
        col_fin = min(col_fin, int(w * 0.22))  # nunca más del 22%
    else:
        col_fin = int(w * 0.03)  # sin banda azul clara: sólo borde mínimo

    return img[:, col_fin:]


def _deblur(gris: np.ndarray) -> np.ndarray:
    """Intenta reducir el desenfoque mediante deconvolución de Wiener simplificada."""
    # Usamos un sharpen fuerte iterativo como aproximación ligera
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]], dtype=np.float32)
    afilado = cv2.filter2D(gris, -1, kernel)
    return np.clip(afilado, 0, 255).astype(np.uint8)


def _generar_variantes_rapidas(img: np.ndarray) -> Generator[tuple[np.ndarray, str], None, None]:
    """
    Genera un conjunto priorizado y reducido de variantes para máxima velocidad.
    1ª opción (la más probable): sin banda azul + CLAHE con escalado óptimo.
    2ª opción (fallback si falla): sin banda azul + Otsu.
    3ª opción (fallback final): imagen completa (sin recortar banda) + CLAHE.
    """
    h, w = img.shape[:2]

    # Redimensionar si es muy pequeña para que el detector reconozca los caracteres
    factor = max(1.0, 120.0 / max(h, 1))
    if factor > 1.0:
        img_redim = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    else:
        img_redim = img

    # --- Variante 1: Recorte azul + CLAHE (resuelve >80% de los casos en 1 segundo) ---
    sin_banda = _quitar_banda_azul(img_redim)
    gris_sb = cv2.cvtColor(sin_banda, cv2.COLOR_BGR2GRAY)
    clahe_sb = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gris_sb)
    yield (clahe_sb, "recorte_clahe")

    # --- Variante 2: Recorte azul + Otsu (para placas con poco contraste o sombras) ---
    _, otsu_sb = cv2.threshold(clahe_sb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield (otsu_sb, "recorte_otsu")

    # --- Variante 3: Sin recortar banda + CLAHE (por si el recorte azul eliminó el 1er carácter) ---
    gris_orig = cv2.cvtColor(img_redim, cv2.COLOR_BGR2GRAY)
    clahe_orig = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gris_orig)
    yield (clahe_orig, "orig_clahe")


# ---------------------------------------------------------------------------
# Lectura principal
# ---------------------------------------------------------------------------

def leer_matricula(img: np.ndarray, early_exit: bool = True) -> tuple[str, float, bool]:
    """
    Lee el texto de una matrícula española a partir de un recorte BGR.

    Parámetros:
        img         - Recorte BGR de la placa.
        early_exit  - Si True (por defecto), termina en cuanto encuentra
                      una matrícula formalmente válida. Reduce el tiempo a ~1 segundo.

    Devuelve:
        (texto, confianza, valida)
    """
    if img is None or img.size == 0:
        return "", 0.0, False

    reader = _get_reader()
    candidatos: list[tuple[str, float, bool]] = []

    for variante, _ in _generar_variantes_rapidas(img):
        try:
            resultados = reader.readtext(
                variante,
                allowlist=ALLOW,
                detail=1,
                paragraph=False,
                decoder="greedy",
                batch_size=1,
                mag_ratio=1.0,  # 1.0 es el doble de rápido que 1.5 en CPU
            )
        except Exception:
            continue

        if not resultados:
            continue

        bruto = "".join(t for _, t, _ in resultados)
        conf = sum(c for _, _, c in resultados) / len(resultados)
        texto, valida = normalizar(bruto)

        if texto:
            candidatos.append((texto, conf, valida))
            # EARLY EXIT: si encontramos una matrícula válida con buena confianza, no seguir probando variantes
            if early_exit and valida and conf >= 0.70:
                return texto, conf, True

    if not candidatos:
        return "", 0.0, False

    return _seleccionar_mejor(candidatos)


def _seleccionar_mejor(
    candidatos: list[tuple[str, float, bool]]
) -> tuple[str, float, bool]:
    """
    Estrategia de selección:
      1. Si hay candidatos válidos, coge el de mayor confianza entre ellos.
      2. Si hay empate de confianza, se prefiere el que aparece más veces (votación).
      3. Si ninguno es válido, devuelve el de mayor confianza.
    """
    validos   = [(t, c, v) for t, c, v in candidatos if v]
    invalidos = [(t, c, v) for t, c, v in candidatos if not v]

    pool = validos if validos else invalidos

    # Votación: peso = confianza × (2 si válido, 1 si no)
    pesos: dict[str, float] = {}
    for t, c, v in pool:
        pesos[t] = pesos.get(t, 0.0) + c * (2.0 if v else 1.0)

    mejor_texto = max(pesos, key=pesos.__getitem__)
    # Confianza media de las ocurrencias del texto ganador
    ocurrencias = [(c, v) for t, c, v in candidatos if t == mejor_texto]
    conf_media  = sum(c for c, _ in ocurrencias) / len(ocurrencias)
    es_valida   = any(v for _, v in ocurrencias)

    return mejor_texto, conf_media, es_valida


# ---------------------------------------------------------------------------
# Alias de compatibilidad con el código anterior (prueba_ocr.py)
# ---------------------------------------------------------------------------

def leer_matricula_easyocr(img: np.ndarray) -> tuple[str, float, bool]:
    """Alias de leer_matricula para compatibilidad con el evaluador."""
    return leer_matricula(img)


# ---------------------------------------------------------------------------
# Script de evaluación autónomo
# ---------------------------------------------------------------------------

def _levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def evaluar(csv_path: str, base_dir: str, splits=("train", "valid")):
    import pandas as pd  # noqa: PLC0415

    df = pd.read_csv(csv_path, sep=";")
    df = df[df["seguridad_texto"] == "alta"]
    df = df[df["split"].isin(splits)]

    filas = []
    total = len(df)
    for i, r in enumerate(df.itertuples(), 1):
        ruta = os.path.join(base_dir, r.ruta_recorte)
        img  = cv2.imread(ruta)
        if img is None:
            print(f"[WARN] No se pudo cargar: {ruta}")
            continue

        texto, conf, valida = leer_matricula(img)
        cer = _levenshtein(texto, r.matricula) / max(len(r.matricula), 1)
        ok  = texto == r.matricula

        filas.append(dict(
            ruta=r.ruta_recorte, split=r.split,
            formato=r.formato, dificultad=r.dificultad,
            real=r.matricula, pred=texto,
            conf=round(conf, 4), valida=valida, ok=ok,
            cer=round(cer, 4),
        ))

        estado = "✓" if ok else "✗"
        print(f"[{i:>3}/{total}] {estado} real={r.matricula:<10} pred={texto:<10} "
              f"conf={conf:.2f} cer={cer:.2f}")

    res = pd.DataFrame(filas)
    print("\n── Resultados por dificultad ──")
    print(res.groupby("dificultad")[["ok", "cer"]].mean().to_string())
    print("\n── Resultados por formato ──")
    print(res.groupby("formato")[["ok", "cer"]].mean().to_string())
    print(f"\nExactas: {res['ok'].sum()}/{len(res)}  "
          f"CER medio: {res['cer'].mean():.4f}  "
          f"Válidas: {res['valida'].sum()}/{len(res)}")

    errores_path = os.path.join(os.path.dirname(csv_path), "errores_ocr.csv")
    res[~res["ok"]].sort_values("cer", ascending=False).to_csv(errores_path, index=False)
    print(f"Errores guardados en: {errores_path}")
    return res


if __name__ == "__main__":
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="Evalúa el OCR de matrículas")
    parser.add_argument(
        "--csv",
        default=os.path.join(
            os.path.dirname(__file__), "..", "datos",
            "matriculas_recortadas", "ocr_dataset.csv"
        ),
        help="Ruta al CSV de dataset (sep=';')",
    )
    parser.add_argument(
        "--base",
        default=os.path.join(
            os.path.dirname(__file__), "..", "datos", "matriculas_recortadas"
        ),
        help="Directorio raíz donde están las imágenes",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "valid"],
        help="Splits a evaluar (train, valid, test)",
    )
    args = parser.parse_args()

    evaluar(
        csv_path=os.path.abspath(args.csv),
        base_dir=os.path.abspath(args.base),
        splits=args.splits,
    )
