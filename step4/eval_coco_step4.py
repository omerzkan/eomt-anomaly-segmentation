
"""
Step4 Evaluation of COCO trained EoMT on Cityscapes validation set.
Using coco_to_cityscapes.py to convert COCO annotations to Cityscapes format, and then using the converted annotations for evaluation.

Produces:
- mIoU on Cityscapes 19 classes
- Per-class IoU on Cityscapes 19 classes
"""

# Imports and Setup

import os, sys, json
os.environ["WANDB_MODE"] = "disabled"

REPO = "/content/cloned_repo_feature_omer/eomt"
sys.path.insert(0, REPO)
sys.path.insert(0, "/content/cloned_repo_feature_omer/step4")

import yaml, torch, numpy as np
from torch.nn import functional as F
from torch.amp.autocast_mode import autocast
from tqdm import tqdm
from lightning import seed_everything
seed_everything(0, verbose=False)
os.chdir(REPO)

# Configuration 

CONFIG_PATH = f"{REPO}/configs/dinov2/coco/panoptic/eomt_base_640_2x.yaml"
CKPT_PATH = "/content/drive/MyDrive/FAIMDL/checkpoints/eomt_coco.bin"
DATA_PATH = "/content/drive/MyDrive/FAIMDL/data"
DEVICE = "cuda"
N_CITYSCAPES_CLASSES = 19
IGNORE=255

# Load the COCO-EoMT model from config and checkpoint
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)
    # with safe_load, it automatically converts the config to a dictionary, so we can access the values using keys

for key, value in config.items():
    print(f"{key}: {value}")









