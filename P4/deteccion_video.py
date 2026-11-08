import argparse
import csv
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

# Windows: NumPy/MKL y PyTorch pueden cargar dos runtimes OpenMP distintos.
# Permite que el proceso continúe en este entorno de trabajo.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from OCR.matricula_ocr import leer_matricula_easyocr

import cv2
from ultralytics import YOLO


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SALIDA_DIR = os.path.join(BASE_DIR, "salida/pruebas")
VENTANA = "Deteccion y seguimiento"

CLASS_NAMES = {
    0: "person", 1: "bicycle", 2: "car",
    3: "motorcycle", 5: "bus", 7: "truck",
}
CLASSES = list(CLASS_NAMES)
VEHICULOS = {2, 3, 5, 7}
CLASE_MATRICULA = 100  # fuera del rango COCO para no colisionar
NOMBRE_MATRICULA = "matricula"

CLASS_COLORS = {  # BGR
    0: (0, 255, 0), 1: (255, 255, 0), 2: (0, 0, 255),
    3: (255, 0, 255), 5: (0, 165, 255), 7: (255, 0, 0),
    CLASE_MATRICULA: (255, 255, 255),
}

CABECERA_CSV = [
    "fotograma", "tipo_objeto", "confianza", "identificador_tracking",
    "x1", "y1", "x2", "y2",
    "matricula_en_su_caso", "confianza_matricula",
    "mx1", "my1", "mx2", "my2", "texto_matricula",
]

LIMITE_CONFIANZA = 0.4


@dataclass
class Matricula:
    bbox: tuple
    confianza: float
    texto: str = ""
    formato_valido: bool = False
    confianza_matricula: float = 0.0


@dataclass
class Deteccion:
    id: Optional[int]
    bbox: tuple
    confianza: float
    clase: int
    matricula: Optional[Matricula] = None

    @property
    def nombre(self):
        return CLASS_NAMES[self.clase]


def parse_args():
    p = argparse.ArgumentParser(description="Detección y seguimiento en vídeo")
    p.add_argument("--video", default=os.path.join(BASE_DIR, "videos/video7.mp4"))
    p.add_argument("--no-show", action="store_true", help="no mostrar ventana")
    p.add_argument("--max-frames", type=int, default=None)
    return p.parse_args()


def detectar_matricula(model_matriculas, frame, bbox):
    x1, y1, x2, y2 = bbox
    recorte = frame[max(y1, 0):y2, max(x1, 0):x2]
    if recorte.size == 0:
        return None

    boxes = model_matriculas.predict(recorte, verbose=False)[0].boxes
    if boxes is None or len(boxes) == 0:
        return None

    i = int(boxes.conf.argmax())
    mx1, my1, mx2, my2 = (int(v) for v in boxes.xyxy[i].cpu().tolist())
    # Volver a coordenadas del fotograma completo
    texto, confianza, es_valida = leer_matricula_easyocr(recorte[my1:my2, mx1:mx2])
    if es_valida: anotar_matricula(texto, confianza)
    return Matricula(
        bbox=(mx1 + max(x1, 0), my1 + max(y1, 0), mx2 + max(x1, 0), my2 + max(y1, 0)),
        confianza=float(boxes.conf[i]),
        texto=texto,
        formato_valido=es_valida,
        confianza_matricula=confianza,
    )


def detectar(model_clases, model_matriculas, frame):
    boxes = model_clases.track(
        frame, persist=True, tracker="botsort.yaml",
        classes=CLASSES, verbose=False,
    )[0].boxes

    detecciones = []
    if boxes is None or len(boxes) == 0:
        return detecciones

    xyxy = boxes.xyxy.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)
    ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(xyxy)

    for bbox, conf, cls, track_id in zip(xyxy, confs, clss, ids):
        if conf < LIMITE_CONFIANZA:
            continue
        bbox = tuple(int(v) for v in bbox)
        det = Deteccion(
            id=None if track_id is None else int(track_id),
            bbox=bbox, confianza=float(conf), clase=int(cls),
        )
        if det.clase in VEHICULOS:
            det.matricula = detectar_matricula(model_matriculas, frame, bbox)
        detecciones.append(det)
    return detecciones


def poner_texto(frame, texto, x, y, color):
    cv2.putText(frame, texto, (x, max(y - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def dibujar_detecciones(frame, detecciones):
    for det in detecciones:
        x1, y1, x2, y2 = det.bbox
        color = CLASS_COLORS[det.clase]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        id_txt = "" if det.id is None else f" {det.id}"
        poner_texto(frame, f"{det.nombre}{id_txt} {det.confianza:.2f}", x1, y1, color)

        if det.matricula:
            mx1, my1, mx2, my2 = det.matricula.bbox
            color_m = CLASS_COLORS[CLASE_MATRICULA]
            cv2.rectangle(frame, (mx1, my1), (mx2, my2), color_m, 2)
            poner_texto(frame, f"{det.matricula.texto} {det.matricula.confianza:.2f}",
                        mx1, my2 + 25, color_m)


def dibujar_leyenda(frame, ids_por_clase):
    y = 25
    for cls, nombre in CLASS_NAMES.items():
        total = len(ids_por_clase[cls])
        if total:
            cv2.putText(frame, f"{nombre}: {total}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, CLASS_COLORS[cls], 2)
            y += 22


def escribir_fila_csv(csv_writer, frame_idx, det):
    x1, y1, x2, y2 = det.bbox
    fila = {
        "fotograma": frame_idx,
        "tipo_objeto": det.nombre,
        "confianza": round(det.confianza, 3),
        "identificador_tracking": "" if det.id is None else det.id,
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
    }
    if det.matricula:
        mx1, my1, mx2, my2 = det.matricula.bbox
        fila.update({
            "matricula_en_su_caso": "si",
            "confianza_matricula": round(det.matricula.confianza, 3),
            "mx1": mx1, "my1": my1, "mx2": mx2, "my2": my2,
            "texto_matricula": det.matricula.texto,
        })
    csv_writer.writerow(fila)

def anotar_matricula(texto, confianza):
    ruta_anotaciones = os.path.join(SALIDA_DIR, "video4_matriculas_detectadas.txt")
    with open(ruta_anotaciones, "a", encoding="utf-8") as f:
        f.write(f"{texto} (confianza: {confianza:.2f})\n")

def escribir_resumen(ruta_resumen, ids_por_clase, detecciones_por_clase):
    print("\nResumen (IDs únicos por clase):")
    with open(ruta_resumen, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tipo_objeto", "ids_unicos", "detecciones_totales"])
        for cls, nombre in CLASS_NAMES.items():
            unicos = len(ids_por_clase[cls])
            total = detecciones_por_clase[cls]
            if unicos or total:
                print(f"  {nombre:<12} IDs únicos: {unicos:<4} detecciones: {total}")
            w.writerow([nombre, unicos, total])


def main():
    args = parse_args()
    mostrar = True

    os.makedirs(SALIDA_DIR, exist_ok=True)
    nombre = os.path.splitext(os.path.basename(args.video))[0]
    ruta_video_out = os.path.join(SALIDA_DIR, f"{nombre}_deteccion.mp4")
    ruta_csv = os.path.join(SALIDA_DIR, f"{nombre}_deteccion.csv")
    ruta_resumen = os.path.join(SALIDA_DIR, f"{nombre}_resumen.csv")

    vid = cv2.VideoCapture(args.video)
    if not vid.isOpened():
        print(f"ERROR: no se puede abrir el vídeo: {args.video}")
        return 1

    ancho = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vid.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Vídeo: {args.video}\n  {ancho}x{alto} @ {fps:.1f} fps, {total_frames} fotogramas")

    writer = cv2.VideoWriter(ruta_video_out, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (ancho, alto))
    if not writer.isOpened():
        print(f"ERROR: no se puede crear el vídeo de salida: {ruta_video_out}")
        vid.release()
        return 1

    model_clases = YOLO(os.path.join(BASE_DIR, "yolo11n.pt"))
    model_matriculas = YOLO(os.path.join(BASE_DIR, "best.pt"))

    ids_por_clase = defaultdict(set)
    detecciones_por_clase = defaultdict(int)
    frame_idx = 0

    if mostrar:
        cv2.namedWindow(VENTANA, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(VENTANA, 1200, 700)

    try:
        with open(ruta_csv, "w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=CABECERA_CSV, restval="")
            csv_writer.writeheader()

            while args.max_frames is None or frame_idx < args.max_frames:
                ret, frame = vid.read()
                if not ret:
                    print("Fin del vídeo o error de lectura.")
                    break

                detecciones = detectar(model_clases, model_matriculas, frame)

                for deteccion in detecciones:
                    detecciones_por_clase[deteccion.clase] += 1
                    if deteccion.id is not None:
                        ids_por_clase[deteccion.clase].add(deteccion.id)
                    escribir_fila_csv(csv_writer, frame_idx, deteccion)

                dibujar_detecciones(frame, detecciones)
                dibujar_leyenda(frame, ids_por_clase)
                writer.write(frame)

                if mostrar:
                    cv2.imshow(VENTANA, frame)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        print("Interrumpido por el usuario.")
                        break

                frame_idx += 1
                if frame_idx % 25 == 0:
                    print(f"  fotograma {frame_idx}/{total_frames or '?'}")
    finally:
        vid.release()
        writer.release()
        if mostrar:
            cv2.destroyAllWindows()

    escribir_resumen(ruta_resumen, ids_por_clase, detecciones_por_clase)
    print(f"\nVídeo de salida: {ruta_video_out}")
    print(f"CSV de detecciones: {ruta_csv}")
    print(f"CSV resumen: {ruta_resumen}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
