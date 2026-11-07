import cv2
import numpy as np

# ---------------------------------------------------------------- datos euro
DIAMETROS_MM = {2.00: 25.75, 0.50: 24.25, 1.00: 23.25, 0.20: 22.25,
                0.05: 21.25, 0.10: 19.75, 0.02: 18.75, 0.01: 16.25}
GRUPOS = {                      # candidatas según el aspecto de la moneda
    "cobre":   [0.05, 0.02, 0.01],
    "oro":     [0.50, 0.20, 0.10],
    "bimetal": [2.00, 1.00],
}

# --------------------------------------------------------------- parámetros
HUE_COBRE_ORO = 17      # hue (0-180) mediano del centro: < 17 cobre, >= 17 oro
DIF_BIMETAL = 14        # diferencia de color (a,b de Lab) centro-anillo
ESCALA_PX_MM = None     # fíjala (px/mm) si calibras con una moneda conocida
DEBUG = False


# -------------------------------------------------------------- segmentación
def segmentar(frame):
    """Máscara de monedas: oscuro (Otsu) O con color saturado (dorado/cobre)."""
    gris = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    sat = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1], (7, 7), 0)
    _, m_osc = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, m_sat = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mascara = cv2.bitwise_or(m_osc, m_sat)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, k, iterations=2)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, k, iterations=2)
    return mascara



def detectar_monedas(frame):
    """Devuelve lista de (x, y, radio_px) para cada moneda."""
    mascara = segmentar(frame)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_min = 0.002 * frame.shape[0] * frame.shape[1]
    monedas = []
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area < area_min:
            continue
        (x, y), r_env = cv2.minEnclosingCircle(contorno)
        monedas.append((x, y, r_env))

    return monedas


# ------------------------------------------------------------ color / grupo
def analizar_color(frame, x, y, r):
    """Devuelve (grupo, hue_centro, dif_centro_anillo)."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = frame.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    d = np.hypot(xx - x, yy - y) / r
    centro, anillo = d < 0.55, (d > 0.78) & (d < 0.92)
    lab_c, lab_a = np.median(lab[centro], 0), np.median(lab[anillo], 0)
    hue = float(np.median(hsv[centro][:, 0]))
    dif = float(np.hypot(*(lab_c[1:] - lab_a[1:])))
    if dif > DIF_BIMETAL:
        grupo = "bimetal"
    else:
        grupo = "cobre" if hue < HUE_COBRE_ORO else "oro"
    return grupo, hue, dif


# ------------------------------------------------- escala px/mm + valor final
def _error(d_mm, cands):
    return min(abs(d_mm - DIAMETROS_MM[v]) / DIAMETROS_MM[v] for v in cands)


def estimar_escala(diam_px, grupos):
    """px/mm que mejor explica TODAS las monedas a la vez (error recortado
    para que un outlier, p. ej. una sombra, no estropee la estimación)."""
    mejor, mejor_coste = None, np.inf
    for d, g in zip(diam_px, grupos):
        for v in GRUPOS[g]:
            s = d / DIAMETROS_MM[v]
            coste = sum(min(_error(di / s, GRUPOS[gi]), 0.08)
                        for di, gi in zip(diam_px, grupos))
            if coste < mejor_coste:
                mejor, mejor_coste = s, coste
    return mejor


def clasificar(diam_px, grupo, escala):
    d_mm = diam_px / escala
    return min(GRUPOS[grupo], key=lambda v: abs(d_mm - DIAMETROS_MM[v]))


# ----------------------------------------------------------------- pipeline
def procesar_frame(frame):
    monedas = detectar_monedas(frame)
    info = [analizar_color(frame, x, y, r) for x, y, r in monedas]
    grupos = [g for g, _, _ in info]
    diam = [2 * r for _, _, r in monedas]

    escala = ESCALA_PX_MM
    if escala is None and monedas:
        escala = estimar_escala(diam, grupos)

    total, resultados = 0.0, []
    for (x, y, r), (g, hue, dif), d in zip(monedas, info, diam):
        valor = clasificar(d, g, escala)
        total += valor
        resultados.append((x, y, r, valor, g, hue, dif))
    return resultados, total, escala


# ------------------------------------------------------------------ dibujo
def dibujar(frame, resultados, total):
    for x, y, r, valor, grupo, hue, dif in resultados:
        x, y, r = int(x), int(y), int(r)
        cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
        texto = f"{valor:.2f} EUR"
        (tw, _), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.putText(frame, texto, (x - tw // 2, max(y - r - 6, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        if DEBUG:   # ayuda para calibrar HUE_COBRE_ORO y DIF_BIMETAL
            cv2.putText(frame, f"{grupo[:3]} h{hue:.0f} d{dif:.0f}", (x - 35, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    banda = f"Monedas: {len(resultados)} | Total: {total:.2f} EUR"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 32), (255, 255, 255), -1)
    cv2.putText(frame, banda, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    return cv2.resize(frame, (600, 600))


# -------------------------------------------------------------------- main
MODO = "imagen"            # "imagen" para probar con la foto, "webcam" para directo

if MODO == "imagen":
    frame = cv2.imread("moneda4.jpg")
    cv2.imshow("mascara", segmentar(frame))
    resultados, total, escala = procesar_frame(frame)
    print(f"Escala estimada: {escala:.2f} px/mm | Total: {total:.2f} EUR")
    cv2.imshow("Conteo de monedas - pulsa una tecla", dibujar(frame, resultados, total))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not captura.isOpened():
        raise RuntimeError("No se pudo abrir la webcam.")
    try:
        while True:
            ok, frame = captura.read()
            if not ok:
                break
            resultados, total, _ = procesar_frame(frame)
            cv2.imshow("Conteo de monedas - pulsa q para salir",
                       dibujar(frame, resultados, total))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        captura.release()
        cv2.destroyAllWindows()