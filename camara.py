import cv2
from ultralytics import YOLO

# Cargar modelo entrenado
model = YOLO("runs/classify/residuos_v1/weights/best.pt")

# Traducción de clases al español
traducciones = {
    "cardboard": "Carton",
    "glass": "Vidrio",
    "metal": "Lata/Metal",
    "paper": "Papel",
    "plastic": "Plastico",
    "trash": "Basura general"
}

# Colores por clase
colores = {
    "cardboard": (0, 165, 255),
    "glass": (255, 255, 0),
    "metal": (0, 0, 255),
    "paper": (255, 255, 255),
    "plastic": (0, 255, 0),
    "trash": (128, 128, 128)
}

# Abrir cámara
cap = cv2.VideoCapture(0)
print("Cámara iniciada. Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Predecir
    results = model(frame, verbose=False)
    probs = results[0].probs

    clase = results[0].names[probs.top1]
    confianza = probs.top1conf.item() * 100

    nombre = traducciones.get(clase, clase)
    color = colores.get(clase, (255, 255, 255))

    # Mostrar resultado en pantalla
    cv2.rectangle(frame, (0, 0), (640, 80), (0, 0, 0), -1)
    cv2.putText(frame, f"{nombre}", (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
    cv2.putText(frame, f"Confianza: {confianza:.1f}%", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Clasificador de Residuos", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()