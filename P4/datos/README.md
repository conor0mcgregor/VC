---
license: cc-by-4.0
language:
- es
pretty_name: Matrículas españolas (VC Práctica 4)
task_categories:
- object-detection
tags:
- license-plates
- yolo
- ocr
size_categories:
- n<1K
viewer: false
---

# Matrículas españolas · VC Práctica 4

Datos de la Práctica 4 de Visión por Computador: detección de matrículas con YOLO y lectura con OCR.

> Las imágenes contienen matrículas legibles (dato personal). Repositorio privado: no lo hagas público.

## Contenido

| Carpeta | Qué es | Tamaño |
|---|---|---|
| `dataset_limpio/` | Dataset YOLO de detección, listo para entrenar | 210 imágenes, ~153 MB |
| `matriculas_recortadas/` | Recortes de cada matrícula y CSV con el texto, para evaluar OCR (P4b) | 217 recortes, ~40 MB |
| `originales_roboflow/` | Los dos exports de Roboflow sin tocar, de los que sale `dataset_limpio` | 219 imágenes, ~156 MB |
| `videos/C0142.MP4` | Vídeo de test del prototipo | ~193 MB |

## `dataset_limpio`

Formato YOLO (una clase, `0 = License Plate`), una etiqueta `.txt` por imagen con cajas `clase cx cy w h` normalizadas.

| Split | Imágenes | Cajas |
|---|---|---|
| train | 151 | 154 |
| valid | 28 | 31 |
| test | 31 | 32 |

`data.yaml` usa rutas relativas a su propia carpeta (sin `path:`), así que funciona en cualquier equipo:

```python
from ultralytics import YOLO
YOLO("yolo11n.pt").train(data="datos/dataset_limpio/data.yaml", epochs=60, imgsz=640)
```

Cómo se construyó a partir de `originales_roboflow/`:

- Se juntaron los exports *Total License Plates v1* y *License plates 2 v1* (219 imágenes).
- Se descartaron 9 imágenes y 2 etiquetas sin imagen que venían en el export.
- Se movieron 24 imágenes de train a valid/test para equilibrar el reparto (`movimientos_reparto.csv`).
- `5422MSY` venía girada 90°: se enderezó junto con su caja.

Comprobado: todas las imágenes tienen etiqueta y viceversa, coordenadas dentro de `[0, 1]`, sin duplicados exactos ni casi duplicados, y sin la misma foto repetida entre splits (los 5 coches con dos fotos están ambas en train).

## `matriculas_recortadas`

- `todas/`: un PNG por matrícula, recortado de `dataset_limpio` con su caja.
- `ocr_dataset.csv`: texto verdadero de cada recorte, formato, dificultad y nitidez.
- `informe.csv`: datos del recorte (ancho, nitidez, motivo de la dificultad).
- `escribir_matriculas.csv`: textos revisados a mano.

## Descarga

```bash
hf download YoelRuso/MatriculasVC --type dataset --local-dir datos
```

O con el script `descargar_datos.py` del repositorio de código.

## Licencia

CC BY 4.0, la misma que los proyectos de Roboflow de origen.
