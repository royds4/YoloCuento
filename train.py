from ultralytics import YOLO

model = YOLO("yolo11x.pt")

train_result = model.train(
    data = "coco8.yaml",
    epochs = 100,
    imgsz = 640,
    device = 'cuda'
)

metrics = model.val()

results = model("path/to/image/jpg")
results[0].show()

