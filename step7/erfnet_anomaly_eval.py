
#==================================
# INSERTING PATHS TO THE SYSTEM
#==================================
import sys
REPO_ROOT = "/content/cloned_repo_feature_omer"
sys.path.insert(0, REPO_ROOT)
REPO_EVAL = "/content/cloned_repo_feature_omer/eval"
sys.path.insert(0, REPO_EVAL)
 
#==================================
# IMPORTS
#==================================
import os
import glob
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
 
from utils.eomt_utils import DEVICE, setup_seed, wandb_setup
from utils.anomaly_utils import MaxLogit, MSP, MaxEntropy, erfnet_anomaly_inference, print_anomaly_results
 
from erfnet import ERFNet

import os
import zipfile

#==================================
# DATASET EXTRACTION (if needed)
#==================================

ZIP_PATH = "/content/drive/MyDrive/FAIMDL/data/Anomaly_Validation_Datasets.zip"
EXTRACT_TO = "/content/data"
DATASET_FOLDER_NAME = "Validation_Dataset"   # the folder name INSIDE the zip

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
 
NUM_CLASSES = 20
# ERFNet outputs 20 channels: 19 Cityscapes classes + 1 background class
 
WEIGHTS_PATH_ERFNET = f"{REPO_ROOT}/trained_models/erfnet_pretrained.pth"
# Path to the pretrained ERFNet weights. This is the original ERFNet pretrained
# on Cityscapes. We use it as-is without any retraining.
 
DATA_PATH_VALIDATION = f"{EXTRACT_TO}/{DATASET_FOLDER_NAME}"
# Root folder containing the 5 anomaly validation datasets.
 
RESULTS_DIR = "/content/drive/MyDrive/FAIMDL/results/step7"
 
os.makedirs(RESULTS_DIR, exist_ok=True)
os.chdir(REPO_ROOT)
 
wandb_setup(enable=False)
setup_seed(seed=42)

#==================================
# DATASET CONFIGURATION
#==================================
# Each entry maps the dataset name to its glob pattern for input images.
# Note: RoadObsticle21 uses .webp, fs_static and RoadAnomaly use .jpg, others .png.
# The "Obsticle" typo is intentional — that's how the dataset folder is named.
 
DATASET_GLOBS = {
    "RoadAnomaly21":     f"{DATA_PATH_VALIDATION}/RoadAnomaly21/images/*.png",
    "RoadObsticle21":    f"{DATA_PATH_VALIDATION}/RoadObsticle21/images/*.webp",
    "FS_LostFound_full": f"{DATA_PATH_VALIDATION}/FS_LostFound_full/images/*.png",
    "fs_static":         f"{DATA_PATH_VALIDATION}/fs_static/images/*.jpg",
    "RoadAnomaly":       f"{DATA_PATH_VALIDATION}/RoadAnomaly/images/*.jpg",
}
 
 
#==================================
# IMAGE PREPROCESSING TRANSFORMS
#==================================
# Same as evalAnomaly.py — keeps results comparable with the TA baseline.
 
input_transform = Compose([
    Resize((512, 1024), Image.BILINEAR),
    ToTensor(),
])
 
target_transform = Compose([
    Resize((512, 1024), Image.NEAREST),
])
 
#==================================
# 1-BUILD THE MODEL
#==================================
print("==================================")
print("LOADING ERFNet MODEL")
print("==================================\n")
 
model = ERFNet(NUM_CLASSES)
model = torch.nn.DataParallel(model).to(DEVICE)
 
 
def load_my_state_dict(model, state_dict):
    """Custom loader from existing evalAnomaly.py that handles
    weight files that don't perfectly match the model's state_dict.
    """
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            if name.startswith("module."):
                own_state[name.split("module.")[-1]].copy_(param)
            else:
                print(name, " not loaded")
                continue
        else:
            own_state[name].copy_(param)
    return model
 
 
model = load_my_state_dict(
    model,
    torch.load(WEIGHTS_PATH_ERFNET, map_location=lambda storage, loc: storage)
)
print("Model and weights LOADED successfully\n")
model.eval()

#==================================
# 2-DEFINE THE SCORING METHODS
#==================================
# All three methods at default temperature (T=1.0).
# Temperature sweep is a separate experiment (run after this main eval).
 
scoring_methods = {
    "MSP":        MSP(),
    "MaxLogit":   MaxLogit(),
    "MaxEntropy": MaxEntropy(),
}


#==================================
# 3-INFERENCE LOOP OVER ALL DATASETS
#==================================
all_results = {}
 
for dataset_name, dataset_glob in DATASET_GLOBS.items():
 
    print(f"\n***************** Evaluating on {dataset_name} *****************\n")
 
    image_paths = sorted(glob.glob(dataset_glob))
 
    if len(image_paths) == 0:
        print(f"WARNING: No images found for {dataset_name} at {dataset_glob}")
        continue
 
    print(f"Found {len(image_paths)} images")
 
    # Run model once, apply all 3 scoring methods
    results = erfnet_anomaly_inference(
        model=model,
        image_paths=image_paths,
        scoring_methods=scoring_methods,
        input_transform=input_transform,
        target_transform=target_transform,
        device=DEVICE,
        description=f"ERFNet on {dataset_name}",
    )
 
    all_results[dataset_name] = results
 
    # Clean GPU cache between datasets to avoid OOM
    torch.cuda.empty_cache()
 
#==================================
# 4-PRINT & SAVE RESULTS
#==================================
result_json_path = f"{RESULTS_DIR}/erfnet_anomaly_results.json"
 
df_auprc, df_fpr95 = print_anomaly_results(
    model_name="ERFNet",
    all_results=all_results,
    save_json_path=result_json_path,
)
 
print("\n=== AuPRC Percentage ===\n")
print(df_auprc)
print("\n=== FPR95 Percentage ===\n")
print(df_fpr95)
