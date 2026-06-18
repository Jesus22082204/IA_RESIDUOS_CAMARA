from ultralytics import YOLO

model = YOLO("yolov8s-cls.pt")

model.train(
    data=".",
    epochs=50,
    imgsz=224,
    batch=16,
    device="cpu",
    name="residuos_v3"
)

print("Entrenamiento terminado!")


