# Práctica 1: Primeros pasos con OpenCV

**Autoría:** Ossam (o sustituir por los nombres de los miembros del grupo)

## Descripción del trabajo
En este directorio se encuentra el cuaderno Jupyter (`VC_P1.ipynb`) que contiene exclusivamente la resolución de las tareas solicitadas para la Práctica 1:
1. Generación de la textura de un tablero de ajedrez (versión manual y versión IA con su comparativa).
2. Composición de una imagen estilo Mondrian utilizando las funciones de dibujo de OpenCV.
3. Detección en tiempo real (mediante cámara) de los píxeles más claro y más oscuro de forma fluida usando funciones optimizadas de OpenCV.
4. Propuesta propia de un filtro estilo Pop Art sobre la captura de vídeo.

## Fuentes e IA utilizadas
- Documentación oficial de [OpenCV](https://docs.opencv.org/4.x/) y [NumPy](https://numpy.org/doc/).
- Se ha utilizado un asistente de IA (Antigravity/Gemini) como apoyo para:
  - Generar la versión alternativa (optimizada sin bucles) del tablero de ajedrez.
  - Recibir recomendaciones de optimización (uso de `cv2.minMaxLoc`) para la tarea de buscar los píxeles extremos en tiempo real.
  - Inspiración y operaciones de matrices para los filtros aplicados en la tarea de Pop Art.

## Requisitos de ejecución
Para ejecutar el cuaderno se requiere el entorno estándar descrito en la práctica con las siguientes librerías instaladas:
- `opencv-python`
- `numpy`
- `matplotlib`

No es necesaria ninguna configuración ni instalación de librerías extra no vistas en la asignatura.
