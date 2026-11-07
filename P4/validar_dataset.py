"""Valida un dataset en formato YOLO antes de entrenar.

Comprueba:
  - que existen las imágenes de cada split (train/val/test)
  - que cada imagen tiene su .txt de etiquetas
  - que cada línea tiene 5 valores: clase x y w h
  - que la clase está dentro de [0, nc-1]
  - que las coordenadas están normalizadas en [0, 1]

Uso:
  python validar_dataset.py --data P4/matriculas/data.yaml
"""

import argparse
import os
import sys
from collections import Counter

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def resolver(base, ruta):
    if os.path.isabs(ruta):
        return ruta
    return os.path.normpath(os.path.join(base, ruta))


def imagenes_de(ruta):
    """Devuelve la lista de imágenes de una carpeta o de un .txt con rutas."""
    if os.path.isfile(ruta) and ruta.lower().endswith(".txt"):
        with open(ruta, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    if not os.path.isdir(ruta):
        return []
    return [
        os.path.join(ruta, f)
        for f in sorted(os.listdir(ruta))
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    ]


def etiqueta_de(imagen):
    """Ruta del .txt correspondiente, soportando las dos layouts habituales.

    split/images/x.jpg   -> split/labels/x.txt
    images/train/x.jpg   -> labels/train/x.txt
    """
    partes = os.path.normpath(imagen).split(os.sep)
    for i in range(len(partes) - 1, -1, -1):
        if partes[i].lower() == "images":
            partes[i] = "labels"
            return os.path.splitext(os.sep.join(partes))[0] + ".txt"
    # Sin carpeta "images": se asume una carpeta hermana "labels"
    carpeta, nombre = os.path.split(imagen)
    return os.path.join(carpeta, "labels", os.path.splitext(nombre)[0] + ".txt")


def validar_split(nombre_split, ruta_imagenes, nc, errores, avisos):
    imagenes = imagenes_de(ruta_imagenes)
    print(f"\n[{nombre_split}] {ruta_imagenes}")
    print(f"  imágenes encontradas: {len(imagenes)}")

    if not imagenes:
        errores.append(f"[{nombre_split}] no se encontraron imágenes en {ruta_imagenes}")
        return Counter()

    conteo_clases = Counter()
    sin_etiqueta = 0
    vacias = 0
    lineas_malas = 0

    for img in imagenes:
        lbl = etiqueta_de(img)
        if not os.path.isfile(lbl):
            sin_etiqueta += 1
            errores.append(f"falta etiqueta: {lbl}")
            continue

        with open(lbl, encoding="utf-8") as f:
            lineas = [l.strip() for l in f if l.strip()]

        if not lineas:
            vacias += 1
            avisos.append(f"etiqueta vacía (imagen sin objetos): {lbl}")
            continue

        for num_linea, linea in enumerate(lineas, 1):
            partes = linea.split()
            if len(partes) != 5:
                lineas_malas += 1
                errores.append(f"{lbl}:{num_linea} tiene {len(partes)} valores (deben ser 5)")
                continue
            try:
                cls = int(partes[0])
                x, y, w, h = (float(v) for v in partes[1:])
            except ValueError:
                lineas_malas += 1
                errores.append(f"{lbl}:{num_linea} no es numérica: '{linea}'")
                continue

            if not (0 <= cls < nc):
                lineas_malas += 1
                errores.append(f"{lbl}:{num_linea} clase {cls} fuera de rango [0, {nc - 1}]")
            if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)):
                lineas_malas += 1
                errores.append(f"{lbl}:{num_linea} coordenadas fuera de [0,1]: {linea}")
            if w <= 0 or h <= 0:
                lineas_malas += 1
                errores.append(f"{lbl}:{num_linea} ancho/alto no positivo: {linea}")

            conteo_clases[cls] += 1

    print(f"  sin etiqueta: {sin_etiqueta}   vacías: {vacias}   líneas inválidas: {lineas_malas}")
    return conteo_clases


def main():
    parser = argparse.ArgumentParser(description="Valida un dataset YOLO")
    parser.add_argument("--data", required=True, help="Ruta al data.yaml")
    args = parser.parse_args()

    if not os.path.isfile(args.data):
        print(f"ERROR: no existe el archivo: {args.data}")
        return 1

    with open(args.data, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        print("ERROR: el data.yaml no contiene un diccionario válido")
        return 1

    nc = data.get("nc")
    names = data.get("names")
    if nc is None:
        print("ERROR: falta 'nc' en el data.yaml")
        return 1
    if names is None or len(names) != nc:
        print(f"ERROR: 'names' debe tener {nc} entradas y tiene {len(names) if names else 0}")
        return 1

    # Ultralytics resuelve las rutas relativas respecto a 'path' (o al directorio del yaml)
    base_yaml = os.path.dirname(os.path.abspath(args.data))
    base = resolver(base_yaml, data.get("path", "."))
    print(f"data.yaml: {os.path.abspath(args.data)}")
    print(f"raíz del dataset: {base}")
    print(f"nc: {nc}   names: {names}")

    errores, avisos = [], []
    total_clases = Counter()

    for split in ("train", "val", "test"):
        if split not in data:
            if split in ("train", "val"):
                errores.append(f"falta el split obligatorio '{split}'")
            continue
        ruta = resolver(base, data[split]) if isinstance(data[split], str) else data[split]
        if isinstance(ruta, list):
            ruta = ruta[0] if ruta else ""
        total_clases.update(validar_split(split, ruta, nc, errores, avisos))

    print("\nObjetos por clase:")
    for i in range(nc):
        print(f"  {i} ({names[i]}): {total_clases.get(i, 0)}")

    if avisos:
        print(f"\nAvisos ({len(avisos)}):")
        for a in avisos[:10]:
            print("  -", a)
        if len(avisos) > 10:
            print(f"  ... y {len(avisos) - 10} más")

    if errores:
        print(f"\nERRORES ({len(errores)}):")
        for e in errores[:20]:
            print("  -", e)
        if len(errores) > 20:
            print(f"  ... y {len(errores) - 20} más")
        print("\nRESULTADO: DATASET NO VÁLIDO")
        return 1

    print("\nRESULTADO: DATASET VÁLIDO")
    return 0


if __name__ == "__main__":
    sys.exit(main())
