from ultralytics import YOLO
from multiprocessing import Process, freeze_support
import torch
import os
import glob

DATA_SET_PATH = "C:/Users/Usuario/source/repos/YoloCuento/datasets/Car counnnting.v17i.yolov11/data.yaml"
BASE_MODEL = "yolo11l.pt" # Or yolov11l.pt if trying a larger model
# BASE_MODEL = "path/to/your/previous/best.pt" # To resume/continue training

# Phase 1 (Feature Extraction / Initial Training) - Optional, try if direct training isn't optimal
EPOCHS_PHASE_1 = 0   # Number of epochs to train with frozen backbone (adjust as needed)
FREEZE_LAYERS = 10    # Typical value for YOLOv8 backbone stages. Verify for yolov11 if possible.
                      # Set to 0 or None to skip phase 1 / freezing.

# Phase 2 (Fine-tuning)
EPOCHS_PHASE_2 = 225  # Remaining epochs (total epochs = EPOCHS_PHASE_1 + EPOCHS_PHASE_2 = 300)
TOTAL_EPOCHS = EPOCHS_PHASE_1 + EPOCHS_PHASE_2 if FREEZE_LAYERS else 300 # Ensure total is consistent

# General Training Params
IMG_SIZE = 1280
BATCH_SIZE = 16 # Reduced batch size as a test, or if memory issues arise with larger models/augs
INITIAL_LR0 = 1e-3 # Default is often 1e-2, maybe start slightly lower?
FINETUNE_LR0 = 1e-4 # Significantly lower LR for fine-tuning
OPTIMIZER = 'AdamW' # Good default
HYP_FILE = 'C:/Users/Usuario/source/repos/YoloCuento/Toll_Vehicle_Detection/train9/args.yaml'    # Path to your custom hyperparameter YAML file (Highly Recommended!)
                      # Example: 'ultralytics/cfg/hyps/hyp.scratch-low.yaml'
                      # Create a copy and modify it, e.g., 'hyp.custom.yaml'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

PROJECT_NAME = "Toll_Vehicle_Detection"
EXPERIMENT_NAME = f"Run_yolo11m_E{TOTAL_EPOCHS}_B{BATCH_SIZE}_F{FREEZE_LAYERS}" # Example naming

# --- Functions ---

def train_model(base_weights, epochs, freeze_n_layers, lr0, hyp_path, phase_name):
    """Helper function to run a training phase."""
    print(f"\n--- Starting Training: {phase_name} ---")
    print(f"Base Weights: {base_weights}")
    print(f"Epochs: {epochs}")
    print(f"Freeze Layers: {freeze_n_layers}")
    print(f"Initial LR: {lr0}")
    print(f"Hyperparameters: {hyp_path if hyp_path else 'Defaults'}")

    model = YOLO(base_weights)

    # Clear cache before training starts
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()

    train_result = model.train(
        data=DATA_SET_PATH,
        epochs=epochs,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        #optimizer=OPTIMIZER,
        #lr0=lr0,
        # cos_lr=True, # Usually enabled by default, confirm if needed
        device=DEVICE,
        freeze=freeze_n_layers, # Set to N (int) or None
        project=PROJECT_NAME,
        name=EXPERIMENT_NAME + f"_{phase_name}",
        workers=8, # Adjust based on your CPU/GPU capabilities
        optimizer= 'AdamW',
        verbose= True,
        seed= 0,
        deterministic= True,
        single_cls= False,
        rect= False,
        cos_lr= False,
        close_mosaic= 10,
        resume= False,
        amp= True,
        fraction= 1.0,
        profile= False,
        multi_scale= False,
        overlap_mask= True,
        mask_ratio= 4,
        dropout= 0.0,
        val= True,
        split= 'val',
        save_json= False,
        save_hybrid= False,
        conf= None,
        iou= 0.7,
        max_det= 300,
        half= False,
        dnn= False,
        plots= False,
        source= None,
        vid_stride= 1,
        stream_buffer= False,
        visualize= False,
        augment= False,
        agnostic_nms= False,
        classes= None,
        retina_masks= False,
        embed= None,
        show= False,
        save_frames= False,
        save_txt= False,
        save_conf= False,
        save_crop= False,
        show_labels= True,
        show_conf= True,
        show_boxes= True,
        line_width= None,
        format= 'torchscript',
        keras= False,
        optimize= False,
        int8= False,
        dynamic= False,
        simplify= True,
        opset= None,
        workspace= None,
        nms= False,
        lr0= 0.00673,
        lrf= 0.0106,
        momentum= 0.79088,
        weight_decay= 0.00047,
        warmup_epochs= 4.01107,
        warmup_momentum= 0.8,
        warmup_bias_lr= 0.1,
        box= 7.09556,
        cls= 0.58686,
        dfl= 1.82722,
        pose= 12.0,
        kobj= 1.0,
        nbs= 64,
        hsv_h= 0.01346,
        hsv_s= 0.81268,
        hsv_v= 0.36574,
        degrees= 0.0,
        translate= 0.09963,
        scale= 0.30358,
        shear= 0.0,
        perspective= 0.0,
        flipud= 0.0,
        fliplr= 0.54117,
        bgr= 0.0,
        mosaic= 0.56685,
        mixup= 0.0,
        copy_paste= 0.0,
        copy_paste_mode= 'flip',
        auto_augment= 'randaugment',
        erasing= 0.4,
        crop_fraction= 1.0,
        cfg= None
    )

    # Find the path to the best weights of THIS training run
    # Note: model.trainer.best should hold the path after training finishes
    best_weights_path = model.trainer.best
    print(f"Best weights from {phase_name}: {best_weights_path}")
    if not os.path.exists(best_weights_path):
         print(f"WARNING: Could not find best weights at expected path: {best_weights_path}")
         # Fallback logic if needed - find last run directory (less reliable)
         try:
            runs_dir = model.trainer.save_dir # Get the save directory
            best_weights_path = os.path.join(runs_dir, "weights", "best.pt")
            print(f"Trying fallback path: {best_weights_path}")
            if not os.path.exists(best_weights_path):
                 raise FileNotFoundError("Fallback path also failed.")
         except Exception as e:
              print(f"Error finding best weights: {e}. Returning base weights path.")
              return base_weights # Return the input weights if finding best failed

    return best_weights_path


def validate_model(weights_path, conf_threshold=0.25, iou_threshold=0.45):
    """Validate the model with specific thresholds."""
    print(f"\n--- Validating Model ---")
    print(f"Weights: {weights_path}")
    print(f"Confidence Threshold: {conf_threshold}")
    print(f"IoU Threshold: {iou_threshold}")

    model = YOLO(weights_path)
    val_results = model.val(
        data=DATA_SET_PATH,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE, # Can often use larger batch for validation
        conf=conf_threshold,
        iou=iou_threshold,
        device=DEVICE,
        project=PROJECT_NAME,
        name=EXPERIMENT_NAME + f"_Validation_Conf{conf_threshold}_IoU{iou_threshold}",
        # split='test' # Use 'test' split if you have one defined in data.yaml
    )
    print("Validation Metrics (mAP50-95, mAP50):")
    print(val_results.box.map)   # mAP50-95
    print(val_results.box.map50) # mAP50


# --- Main Execution ---

if __name__ == "__main__":
    freeze_support() # Needed for multiprocessing on Windows

    final_weights_path = BASE_MODEL

    # --- Optional Phase 1: Feature Extraction ---
    if FREEZE_LAYERS and EPOCHS_PHASE_1 > 0:
        final_weights_path = train_model(
            base_weights=final_weights_path,
            epochs=EPOCHS_PHASE_1,
            freeze_n_layers=FREEZE_LAYERS,
            lr0=INITIAL_LR0, # Use the standard/initial LR
            hyp_path=HYP_FILE,
            phase_name="Phase1_Frozen"
        )
    else:
        print("Skipping Phase 1 (Freezing).")

    final_weights_path = "C:/Users/Usuario/source/repos/YoloCuento/Toll_Vehicle_Detection/Run_yolo11m_E300_B16_F10_Phase1_Frozen2/weights/best.pt"
    # --- Phase 2: Fine-tuning (or Full Training if Phase 1 skipped) ---
    # Determine epochs and LR based on whether Phase 1 ran
    current_epochs = EPOCHS_PHASE_2 if (FREEZE_LAYERS and EPOCHS_PHASE_1 > 0) else TOTAL_EPOCHS
    current_lr0 = FINETUNE_LR0 if (FREEZE_LAYERS and EPOCHS_PHASE_1 > 0) else INITIAL_LR0

    final_weights_path = train_model(
        base_weights=final_weights_path, # Start from BASE_MODEL or Phase 1 result
        epochs=current_epochs,
        freeze_n_layers=None, # Always unfreeze for fine-tuning/full training
        lr0=current_lr0,      # Use potentially lower LR for fine-tuning
        hyp_path=HYP_FILE,
        phase_name="Phase2_Finetune" if (FREEZE_LAYERS and EPOCHS_PHASE_1 > 0) else "Full_Training"
    )

    print(f"\n--- Training Complete ---")
    print(f"Final best weights saved to: {final_weights_path}")

    # --- Validation ---
    if os.path.exists(final_weights_path):
        # Validate with standard thresholds
        validate_model(final_weights_path, conf_threshold=0.25, iou_threshold=0.45)

        # Validate with thresholds potentially better for F1 score (based on your curves)
        validate_model(final_weights_path, conf_threshold=0.493, iou_threshold=0.45) # Near peak F1 conf

        # Validate with higher confidence for potentially higher precision
        validate_model(final_weights_path, conf_threshold=0.70, iou_threshold=0.45)
    else:
        print("Could not find final weights file to run validation.")
