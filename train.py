from ultralytics import YOLO
from multiprocessing import Process, freeze_support
import torch

def train():
    model = YOLO("yolo11x.pt")

    train_result = model.train(
        data = "C:/Users/Usuario/source/repos/YoloCuento/datasets/Car counnnting.v6i.yolov11/data.yaml",
        epochs = 400,
        imgsz = 640,
        device = 'cuda',
    )


if __name__ == "__main__":
    freeze_support()
    torch.cuda.empty_cache()
    p = Process(target=train)
    p.start()
