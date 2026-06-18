import cv2
import os

clase = input("Que objeto vas a fotografiar? (plastic/metal/glass/paper/cardboard/trash): ")
carpeta = f"train/{clase}"
os.makedirs(carpeta, exist_ok=True)

cap = cv2.VideoCapture(0)
contador = 0

print(f"Capturando fotos para '{clase}'")
print("Presiona ESPACIO para tomar foto, 'q' para salir")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fotos_tomadas = len(os.listdir(carpeta))
    cv2.putText(frame, f"Clase: {clase}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Fotos: {fotos_tomadas}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, "ESPACIO=foto  Q=salir", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Capturar fotos", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        nombre = f"{carpeta}/real_{contador:04d}.jpg"
        cv2.imwrite(nombre, frame)
        contador += 1
        print(f"Foto guardada: {nombre}")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Total fotos tomadas: {contador}")