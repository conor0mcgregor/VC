from ultralytics import YOLO
import cv2

modelo = YOLO("yolo11n.pt")

resultados = modelo("P4/test.jpg")

# Imagen con las cajas dibujadas
img = resultados[0].plot()

cv2.imshow("Detecciones", img)
cv2.waitKey(0)
cv2.destroyAllWindows()