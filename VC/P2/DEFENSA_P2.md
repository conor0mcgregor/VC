# Defensa de la Práctica 2

## Objetivo

La práctica completa operaciones básicas de OpenCV sobre la imagen `mandril.jpg`: detección de bordes, conteo de píxeles relevantes por filas y columnas, umbralizado y una propuesta sencilla de demostrador con webcam.

## Tarea 1: conteo por filas en Canny

Primero se usa la imagen en escala de grises y se obtiene la imagen de bordes con `cv2.Canny(gris, 100, 200)`.

Después se cuentan los píxeles blancos por filas con `cv2.reduce(canny, 1, cv2.REDUCE_SUM)`. Como Canny devuelve valores 0 o 255, la suma de cada fila se divide entre 255 para obtener el número real de píxeles blancos.

Con esos valores se calcula `maxfil`, que es el máximo número de píxeles blancos encontrado en una fila. Luego se buscan las filas cuyo valor es mayor o igual que `0.90 * maxfil`.

En la imagen usada:

- `maxfil = 220`
- filas destacadas: `[6, 12, 15, 20, 21, 88, 100]`

Para visualizarlo, la imagen de Canny se convierte a RGB y se dibuja una línea roja sobre cada fila destacada.

## Tarea 2: Sobel umbralizado y comparación con Canny

Para Sobel se suaviza primero la imagen de grises con una gaussiana. Después se calculan las derivadas horizontal y vertical con `cv2.Sobel`, y se combinan con `cv2.add`.

El resultado de Sobel se convierte a 8 bits con `cv2.convertScaleAbs`, porque la salida original puede tener valores fuera del rango normal de una imagen de 8 bits.

Luego se aplica un umbral binario con valor 60:

`cv2.threshold(sobel8, 60, 255, cv2.THRESH_BINARY)`

Sobre esa imagen umbralizada se cuentan píxeles blancos por filas y por columnas. Igual que antes, se buscan las posiciones que están por encima del 90% del máximo.

En la imagen usada:

- máximo por filas en Sobel: `328`
- filas Sobel destacadas: `[1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 15, 18, 19, 20, 24, 52, 80, 82, 83, 84, 85, 87, 88, 93, 96, 99, 100]`
- máximo por columnas en Sobel: `314`
- columnas Sobel destacadas: `[90, 99, 100, 103, 104, 105, 106, 107, 112, 119, 123, 125, 126, 127, 128, 132]`

También se repite el conteo por filas y columnas para Canny, para poder comparar ambos detectores sobre la misma imagen.

En Canny:

- máximo por columnas: `187`
- columnas Canny destacadas: `[67, 86, 92, 96, 99, 100, 103, 104, 105, 112, 115, 119, 123, 379, 380, 383, 392, 396, 403]`

Las filas se dibujan en rojo y las columnas en amarillo sobre la imagen original del mandril.

## Comparación entre Sobel y Canny

Canny da bordes más finos y limpios. Esto ocurre porque no solo calcula gradientes, sino que además aplica etapas como suavizado, supresión de no máximos e histéresis.

Sobel umbralizado detecta muchos cambios de intensidad, por eso aparecen más zonas marcadas. Es útil para ver textura y variaciones locales, pero produce bordes más gruesos y más ruido que Canny.

Para defenderlo de forma sencilla:

- Canny es mejor si queremos contornos claros.
- Sobel es más directo y simple, pero necesita elegir bien el umbral.
- En esta imagen, Sobel marca más textura del pelo y detalles internos del mandril, mientras que Canny selecciona bordes más definidos.

## Tarea 3: demostrador con webcam

La propuesta está inspirada en *My little piece of privacy*.

La idea es crear una cortina de privacidad digital. La webcam se muestra en espejo, se detectan las zonas en movimiento con sustracción de fondo y esas zonas se pixelan. Así se ve la escena general, pero se oculta la parte activa de la persona.

Pasos del demostrador:

1. Capturar vídeo con `cv2.VideoCapture(0)`.
2. Aplicar efecto espejo con `cv2.flip`.
3. Detectar movimiento con `cv2.createBackgroundSubtractorMOG2`.
4. Limpiar la máscara con operaciones morfológicas.
5. Crear una versión pixelada del fotograma reduciendo y ampliando la imagen.
6. Sustituir por mosaico solo las zonas donde hay movimiento.
7. Dibujar los contornos detectados para que se vea qué región se está ocultando.

La tecla `ESC` cierra el demostrador.

## Puntos clave para la defensa

OpenCV representa muchas imágenes binarias con valores 0 y 255. Por eso, cuando se suman píxeles blancos, hay que dividir entre 255 para obtener el número de píxeles y no la suma de intensidades.

`cv2.reduce` permite sumar por dimensión:

- dimensión `0`: suma por columnas.
- dimensión `1`: suma por filas.

Las primitivas gráficas usadas para remarcar resultados son líneas con `cv2.line`.

La diferencia principal entre Canny y Sobel es que Sobel calcula gradientes directamente, mientras que Canny usa más pasos para quedarse con bordes más fiables.
