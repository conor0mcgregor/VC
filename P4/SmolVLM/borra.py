import cv2
import numpy as np

from matricula_SmolVLM import leer_matricula
imagen = cv2.imread("P4\\images\\matricula-prueba.png")
imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
imagen = np.array(imagen)

predicha, confianza, es_formato = leer_matricula(imagen)
print(f"Predicha: {predicha}")