from ultralytics import YOLO

model = YOLO("runs/detect/87epochsnoflip/weights/best.pt")

results = model("randomphotos\\13.png", conf=0.25, verbose=True, device=0)
results[0].save("output.jpg")

