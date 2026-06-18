import cv2
import time
from ultralytics import YOLO

# Cargar modelo
model = YOLO("runs/classify/residuos_v3/weights/best.pt")
# Clases en español
traducciones = {
    "cardboard": "Carton",
    "glass": "Vidrio",
    "metal": "Metal/Lata",
    "paper": "Papel",
    "plastic": "Plastico",
    "trash": "Basura general"
}

# Colores por clase (BGR)
colores = {
    "cardboard":  (0, 165, 255),   # naranja
    "glass":      (255, 255, 0),   # cyan
    "metal":      (0, 0, 220),     # rojo
    "paper":      (200, 200, 200), # gris claro
    "plastic":    (0, 200, 0),     # verde
    "trash":      (80, 80, 80)     # gris oscuro
}

# Emojis de reciclaje por clase (texto)
iconos = {
    "cardboard": "[CARTON]",
    "glass":     "[VIDRIO]",
    "metal":     "[METAL]",
    "paper":     "[PAPEL]",
    "plastic":   "[PLASTICO]",
    "trash":     "[BASURA]"
}

# Historial para estabilizar predicciones
historial = []
HISTORIAL_MAX = 8

def prediccion_estable(clase):
    historial.append(clase)
    if len(historial) > HISTORIAL_MAX:
        historial.pop(0)
    return max(set(historial), key=historial.count)

# Barra de confianza
def dibujar_barra(frame, x, y, ancho, confianza, color):
    cv2.rectangle(frame, (x, y), (x + ancho, y + 18), (50, 50, 50), -1)
    relleno = int(ancho * confianza / 100)
    cv2.rectangle(frame, (x, y), (x + relleno, y + 18), color, -1)
    cv2.rectangle(frame, (x, y), (x + ancho, y + 18), (200, 200, 200), 1)

# Abrir camara
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

fps_timer = time.time()
fps = 0
frame_count = 0

print("Camara iniciada. Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # FPS
    frame_count += 1
    if time.time() - fps_timer >= 1.0:
        fps = frame_count
        frame_count = 0
        fps_timer = time.time()

    # Prediccion
    results = model(frame, verbose=False)
    probs = results[0].probs
    clase = results[0].names[probs.top1]
    confianza = probs.top1conf.item() * 100

    # Top 3 clases
    top3_idx = probs.top5[:3]
    top3_names = [results[0].names[i] for i in top3_idx]
    top3_confs = [probs.data[i].item() * 100 for i in top3_idx]

    # Estabilizar prediccion
    clase_estable = prediccion_estable(clase)
    nombre = traducciones.get(clase_estable, clase_estable)
    color = colores.get(clase_estable, (255, 255, 255))
    icono = iconos.get(clase_estable, "")

    alto, ancho = frame.shape[:2]

    # Panel superior
    cv2.rectangle(frame, (0, 0), (ancho, 90), (15, 15, 15), -1)
    cv2.putText(frame, "CLASIFICADOR DE RESIDUOS", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
    cv2.putText(frame, f"FPS: {fps}", (ancho - 110, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

    # Clase principal
    cv2.putText(frame, f"{icono} {nombre}", (10, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    # Panel lateral derecho - Top 3
    panel_x = ancho - 280
    cv2.rectangle(frame, (panel_x - 10, 100), (ancho, 310), (15, 15, 15), -1)
    cv2.putText(frame, "Top 3:", (panel_x, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2)

    for i, (n, c) in enumerate(zip(top3_names, top3_confs)):
        y_pos = 160 + i * 50
        nombre_top = traducciones.get(n, n)
        color_top = colores.get(n, (200, 200, 200))
        cv2.putText(frame, f"{i+1}. {nombre_top}", (panel_x, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_top, 2)
        dibujar_barra(frame, panel_x, y_pos + 5, 200, c, color_top)
        cv2.putText(frame, f"{c:.1f}%", (panel_x + 205, y_pos + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_top, 1)

    # Barra de confianza principal
    cv2.putText(frame, f"Confianza:", (10, alto - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)
    dibujar_barra(frame, 10, alto - 40, 400, confianza, color)
    cv2.putText(frame, f"{confianza:.1f}%", (420, alto - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Borde de color segun clase
    cv2.rectangle(frame, (0, 0), (ancho - 1, alto - 1), color, 4)

    cv2.imshow("Clasificador de Residuos - IA Parcial", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()