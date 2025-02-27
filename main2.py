from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")

results = model.train(data="config.yaml", epochs=10)