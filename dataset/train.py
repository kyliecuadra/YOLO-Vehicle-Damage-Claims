from ultralytics import YOLO

# === SETTINGS ===
MODEL = 'vehicle_damages_claims.pt'               # Use a YOLO checkpoint (n/s/m/l/x) or a custom one
DATASET_YAML = 'data.yaml'  # Dataset YAML
EPOCHS = 50
DEVICE = 'cpu'                       # GPU = '0', CPU = 'cpu'
IMG_SIZE = 640                     # Image size during training

# === TRAIN ===
model = YOLO(MODEL)  # Load pre-trained model or empty if you specify

model.train(
    data=DATASET_YAML,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    device=DEVICE
)

print("✅ Training completed!")
