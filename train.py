from ultralytics import YOLO


if __name__ == "__main__":

    """#mosaic jest dziwnym ustawieniem
    #jest copy-paste - można rozważyć wklejanie z GRSRB
    #76epochs
    model = YOLO("yolo11m.pt")
    results = model.train(data="data.yaml", epochs=250, patience=25, imgsz=640,
                          batch=0.8, device=0, rect=True, save=True,
                          save_period=50, degrees=4.0, shear=5.0, mosaic=0.25 )"""
    #83epochs, 87epochsnoflip
    model = YOLO("yolo11l.pt")
    results = model.train(data="data.yaml", epochs=250, patience=25, imgsz=640,
                          batch=0.95, device=0, rect=True, save=True,
                          save_period=25, degrees=5.0, shear=7.5,
                          fliplr=0.0, flipud=0.0)
    #na przyszłość: usunąć save period, github odrzuci pliki 100+MB



