from ultralytics import YOLO
from multiprocessing import freeze_support
import torch
import wandb
import ray
from ray import tune


DATA_SET_PATH = "C:/Users/Usuario/source/repos/YoloCuento/datasets/Car counnnting.v16i.yolov11/data.yaml"
BASE_MODEL = "best.pt" # Or yolov11l.pt if trying a larger model
# BASE_MODEL = "path/to/your/previous/best.pt" # To resume/continue training
wandb.login(key="954833e5f7792ba11ef0dbfdae8bdaf34c34fe4c") # Replace with your actual WandB API key
# --- Main Execution ---

if __name__ == "__main__":
    freeze_support() # Needed for multiprocessing on Windows

    final_weights_path = BASE_MODEL

    # --- Optional Phase 1: Feature Extraction ---
    # Initialize the YOLO model
    model = YOLO(final_weights_path)

    # Tune hyperparameters on COCO8 for 30 epochs
    model.tune(
        project="Toll_Vehicle_Detection",
        data=DATA_SET_PATH, 
        epochs=50, 
        #iterations=100, 
        #space={"lr0": tune.uniform(1e-5, 1e-1)},
        optimizer="AdamW", 
        plots=False, 
        save=False, 
        val=True,
        )
