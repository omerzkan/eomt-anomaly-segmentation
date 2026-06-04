
#==================================
# INSERTING PATHS TO THE SYSTEM
#==================================
import sys
REPO_ROOT = "/content/cloned_repo_feature_omer"
sys.path.insert(0, REPO_ROOT)
REPO_EOMT = "/content/cloned_repo_feature_omer/eomt"
sys.path.insert(0, REPO_EOMT)
 
#==================================
# IMPORTS
#==================================
import glob
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
import numpy as np 
 
from utils.eomt_utils import DEVICE, build_model, setup_seed, wandb_setup
from utils.anomaly_utils import MaxLogit, MSP, MaxEntropy, RbA, eomt_anomaly_inference, print_anomaly_results
 
import os
import zipfile

#==================================
# DATASET EXTRACTION (if needed)
#==================================

ZIP_PATH = "/content/drive/MyDrive/FAIMDL/data/Anomaly_Validation_Datasets.zip"
EXTRACT_TO = "/content/data"
DATASET_FOLDER_NAME = "Validation_Dataset"

# Only extract if not already done (idempotent)
if not os.path.exists(f"{EXTRACT_TO}/{DATASET_FOLDER_NAME}"):
    print("Extracting anomaly validation datasets...")
    os.makedirs(EXTRACT_TO, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_TO)
    print(f"Extracted to {EXTRACT_TO}/{DATASET_FOLDER_NAME}\n")
else:
    print(f"Datasets already extracted at {EXTRACT_TO}/{DATASET_FOLDER_NAME}\n")

#==================================
# CONFIGURATION & SETUP
#==================================

CONFIG_PATH_COCO       = f"{REPO_EOMT}/configs/dinov2/coco/panoptic/eomt_base_640_2x.yaml"
CONFIG_PATH_CITYSCAPES = f"{REPO_EOMT}/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml"
 
# Checkpoint paths (model weights on Google Drive)
CHECKPOINT_PATH_COCO       = "/content/drive/MyDrive/FAIMDL/checkpoints/eomt_coco.bin"
CHECKPOINT_PATH_CITYSCAPES = "/content/drive/MyDrive/FAIMDL/checkpoints/eomt_cityscapes.bin"
CHECKPOINT_PATH_FINETUNED  = "/content/drive/MyDrive/FAIMDL/checkpoints/coco_eomt_finetuned_on_cityscapes.bin"
 
RESULTS_DIR = "/content/drive/MyDrive/FAIMDL/results/step8"
DATA_PATH_VALIDATION = f"{EXTRACT_TO}/{DATASET_FOLDER_NAME}"

os.chdir(REPO_ROOT)
wandb_setup(enable=False)
setup_seed(seed=42)

#==================================
# DATASET CONFIGURATION
#==================================
 
DATASET_GLOBS = {
    "RoadAnomaly21":     f"{DATA_PATH_VALIDATION}/RoadAnomaly21/images/*.png",
    "RoadObsticle21":    f"{DATA_PATH_VALIDATION}/RoadObsticle21/images/*.webp",
    "FS_LostFound_full": f"{DATA_PATH_VALIDATION}/FS_LostFound_full/images/*.png",
    "fs_static":         f"{DATA_PATH_VALIDATION}/fs_static/images/*.jpg",
    "RoadAnomaly":       f"{DATA_PATH_VALIDATION}/RoadAnomaly/images/*.jpg",
}

#==================================
# SCORING METHODS
#==================================
# All 4 post-hoc methods at default temperature (T=1.0).

scoring_methods = {
    "MSP":        MSP(),
    "MaxLogit":   MaxLogit(),
    "MaxEntropy": MaxEntropy(),
    "RbA":        RbA(),
}

#==================================
# HELPER FUNCTION: FOR RUN ALL 5 DATASETS FOR ONE MODEL
#==================================

def evaluate_model_on_all_datasets(model, model_name, input_transform, target_transform, is_save_logits=True):
    all_results = {}
    for dataset_name, dataset_glob in DATASET_GLOBS.items():
        print(f"\n***************** {model_name} - {dataset_name} *****************\n")
        image_paths = glob.glob(dataset_glob)
        
        if(len(image_paths) == 0):
            print(f"Warning: No images found for {dataset_name}")
            continue
        print(f"Found {len(image_paths)} images for {dataset_name}")
        
        results = eomt_anomaly_inference(
            model=model,
            image_paths=image_paths,
            scoring_methods=scoring_methods,
            input_transform=input_transform,
            target_transform=target_transform,
            device=DEVICE,
            description=f"{model_name} - {dataset_name}",
            save_logits_path = f"/content/saved_logits/{model_name}/{dataset_name}" if is_save_logits else None
            
        )
        all_results[dataset_name] = results
        torch.cuda.empty_cache()  # Clear GPU memory after each dataset
    
    return all_results


#==================================
# HELPER FUNCTION: FOR CONVERTING PIL IMAGE TO UINT8 TENSOR
#==================================

def pil_to_uint8_tensor(pil_img):
    arr = np.array(pil_img)  # (H, W, C) uint8
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (C, H, W) uint8
    return tensor


"""
***************** SECTION 1: EoMT-COCO-trained *****************
"""
print("\n***************** EoMT-COCO-trained *****************\n")
 
#==================================
# 1-BUILD THE MODEL
#==================================
IMG_SIZE_COCO = [640, 640]
N_COCO_CLASSES = 133
STUFF_CLASSES_COCO = [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
                      96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                      110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122,
                      123, 124, 125, 126, 127, 128, 129, 130, 131, 132]
 
overriders_coco = {
    ('model', 'init_args', 'network', 'init_args', 'encoder', 'init_args', 'img_size'): IMG_SIZE_COCO,
    ('model', 'init_args', 'network', 'init_args', 'num_classes'): N_COCO_CLASSES,
    ('model', 'init_args', 'network', 'init_args', 'masked_attn_enabled'): False,
    ('model', 'init_args', 'img_size'): IMG_SIZE_COCO,
    ('model', 'init_args', 'num_classes'): N_COCO_CLASSES,
    ('model', 'init_args', 'stuff_classes'): STUFF_CLASSES_COCO,
}
 
model_coco = build_model(
    config_path=CONFIG_PATH_COCO,
    eval_mode=True,
    config_overriders=overriders_coco,
    sanity_check=False,
    checkpoint_path=CHECKPOINT_PATH_COCO,
    device=DEVICE,
)
 
# Image preprocessing transforms for COCO model
input_transform_coco = Compose([
    Resize((640, 640), Image.BILINEAR),
    pil_to_uint8_tensor
])
target_transform_coco = Compose([
    Resize((640, 640), Image.NEAREST),
])
 
#==================================
# 2-INFERENCE LOOP OVER ALL DATASETS
#==================================
results_coco = evaluate_model_on_all_datasets(
    model=model_coco,
    model_name="eomt_coco",
    input_transform=input_transform_coco,
    target_transform=target_transform_coco,
    is_save_logits=False
)

#==================================
# 3-PRINT & SAVE RESULTS
#==================================
df_auprc_coco, df_fpr95_coco = print_anomaly_results(
    model_name="eomt_coco",
    all_results=results_coco,
    save_json_path=f"{RESULTS_DIR}/eomt_coco_anomaly_results.json",
)
print("\n=== AuPRC (%) ===\n")
print(df_auprc_coco)
print("\n=== FPR95 (%) ===\n")
print(df_fpr95_coco)
 
# Free GPU memory before the next model
del model_coco
torch.cuda.empty_cache()
 
"""
***************** SECTION 2: EoMT-Cityscapes-trained *****************
"""
print("\n***************** EoMT-Cityscapes-trained *****************\n")
 
#==================================
# 1-BUILD THE MODEL
#==================================
IMG_SIZE_CITYSCAPES = [1024, 1024]
N_CITYSCAPES_CLASSES = 19
 
overriders_cityscapes = {
    ('model', 'init_args', 'network', 'init_args', 'encoder', 'init_args', 'img_size'): IMG_SIZE_CITYSCAPES,
    ('model', 'init_args', 'network', 'init_args', 'num_classes'): N_CITYSCAPES_CLASSES,
    ('model', 'init_args', 'network', 'init_args', 'masked_attn_enabled'): False,
    ('model', 'init_args', 'img_size'): IMG_SIZE_CITYSCAPES,
    ('model', 'init_args', 'num_classes'): N_CITYSCAPES_CLASSES,
}
 
model_cityscapes = build_model(
    config_path=CONFIG_PATH_CITYSCAPES,
    eval_mode=True,
    config_overriders=overriders_cityscapes,
    sanity_check=False,
    checkpoint_path=CHECKPOINT_PATH_CITYSCAPES,
    device=DEVICE,
)
 
input_transform_cityscapes = Compose([
    Resize((1024, 1024), Image.BILINEAR),
    pil_to_uint8_tensor
])
target_transform_cityscapes = Compose([
    Resize((1024, 1024), Image.NEAREST),
])
 
#==================================
# 2-INFERENCE LOOP OVER ALL DATASETS
#==================================
results_cityscapes = evaluate_model_on_all_datasets(
    model=model_cityscapes,
    model_name="eomt_cityscapes",
    input_transform=input_transform_cityscapes,
    target_transform=target_transform_cityscapes,
    is_save_logits=True
)
 
#==================================
# 3-PRINT & SAVE RESULTS
#==================================
df_auprc_city, df_fpr95_city = print_anomaly_results(
    model_name="eomt_cityscapes",
    all_results=results_cityscapes,
    save_json_path=f"{RESULTS_DIR}/eomt_cityscapes_anomaly_results.json",
)
print("\n=== AuPRC (%) ===\n")
print(df_auprc_city)
print("\n=== FPR95 (%) ===\n")
print(df_fpr95_city)
 
# Free GPU memory before next model
del model_cityscapes
torch.cuda.empty_cache()
 
"""
***************** SECTION 3: EoMT-Fine-tuned *****************
"""
print("\n***************** EoMT-Fine-tuned *****************\n")
 
#==================================
# 1-BUILD THE MODEL
#==================================
# Fine-tuned uses Cityscapes config but with COCO image size (640) and num_q=200
IMG_SIZE_FINETUNED = [640, 640]
 
overriders_finetuned = {
    ('model', 'init_args', 'network', 'init_args', 'encoder', 'init_args', 'img_size'): IMG_SIZE_FINETUNED,
    ('model', 'init_args', 'network', 'init_args', 'num_classes'): N_CITYSCAPES_CLASSES,
    ('model', 'init_args', 'network', 'init_args', 'num_q'): 200,
    ('model', 'init_args', 'network', 'init_args', 'masked_attn_enabled'): False,
    ('model', 'init_args', 'img_size'): IMG_SIZE_FINETUNED,
    ('model', 'init_args', 'num_classes'): N_CITYSCAPES_CLASSES,
}
 
model_finetuned = build_model(
    config_path=CONFIG_PATH_CITYSCAPES,
    eval_mode=True,
    config_overriders=overriders_finetuned,
    sanity_check=False,
    checkpoint_path=CHECKPOINT_PATH_FINETUNED,
    device=DEVICE,
)
 
input_transform_finetuned = Compose([
    Resize((640, 640), Image.BILINEAR),
    pil_to_uint8_tensor
])
target_transform_finetuned = Compose([
    Resize((640, 640), Image.NEAREST),
])
 
#==================================
# 2-INFERENCE LOOP OVER ALL DATASETS
#==================================
results_finetuned = evaluate_model_on_all_datasets(
    model=model_finetuned,
    model_name="eomt_finetuned",
    input_transform=input_transform_finetuned,
    target_transform=target_transform_finetuned,
    is_save_logits=True
)
 
#==================================
# 3-PRINT & SAVE RESULTS
#==================================
df_auprc_finetuned, df_fpr95_finetuned = print_anomaly_results(
    model_name="eomt_finetuned",
    all_results=results_finetuned,
    save_json_path=f"{RESULTS_DIR}/eomt_finetuned_anomaly_results.json",
)
print("\n=== AuPRC (%) ===\n")
print(df_auprc_finetuned)
print("\n=== FPR95 (%) ===\n")
print(df_fpr95_finetuned)
 