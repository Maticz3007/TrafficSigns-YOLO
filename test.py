from ultralytics import YOLO

model = YOLO("runs/detect/train9/weights/best.pt")

results = model("randomphotos\\13.png", conf=0.5, device=0)
results[0].save("output.jpg")