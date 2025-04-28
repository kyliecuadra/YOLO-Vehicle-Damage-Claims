from ultralytics import YOLO

# === SETTINGS ===
MODEL = 'path/to/best.pt'           # Your trained model
SOURCE = 'path/to/image_or_folder'  # Single image or folder
DEVICE = '0'                        # GPU = '0', CPU = 'cpu'
CONFIDENCE = 0.50                 # Confidence threshold

# === PREDICT ===
model = YOLO(MODEL)

results = model.predict(
    source=SOURCE,
    device=DEVICE,
    conf=CONFIDENCE,
    save=True,         # Save output images with boxes
    imgsz=640
)

print("✅ Prediction completed! Results saved.")
