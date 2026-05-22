

"""
This is the evaluation script for 
the Fine-tuned on Cityscapes Dataset version of the COCO-Trained-EoMT model vs the original COCO-Trained-EoMT model. 
It evaluates the performance of both models on the Cityscapes validation set and compares their results.
"""

#==================================
# IMPORTS
#==================================

import sys
import os
from torchmetrics.classification import MulticlassJaccardIndex
from utils.eomt_utils import CITYSCAPES_CLASS_NAMES, DEVICE, IGNORE_INDEX, IMG_SIZE, N_CITYSCAPES_CLASSES
from utils.eomt_utils import build_model, semantic_inference, insert_path, setup_seed, wandb_setup, compare_result_iou, print_results
import json
import numpy as np

#==================================
# INSERTING PATHS TO THE SYSTEM
#==================================

REPO_EOMT = "/content/cloned_repo_feature_omer/eomt"
REPO_ROOT = "/content/cloned_repo_feature_omer"

sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, REPO_EOMT)

#==================================
# CONFIGURATION & SETUP
#==================================

REPO = "/content/cloned_repo_feature_omer/eomt"
# This is the path of the copied github repo in the google colab. 

CONFIG_PATH_CITYSCAPES_TRAINED = f"{REPO_EOMT}/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml"
# Path to the YAML config that describes EoMT's architecture and training settings.

DATA_PATH_CITYSCAPES_VALIDATION = "/content/drive/MyDrive/FAIMDL/data"
# Root folder containing the dataset (images + annotations).

CHECKPOINT_PATH_COCO_FINETUNED_ON_CITYSCAPES = "/content/drive/MyDrive/FAIMDL/checkpoints/coco_eomt_finetuned_on_cityscapes.bin"

os.chdir(REPO_ROOT)
# We change the directory to the REPO

wandb_setup(enable=True)
setup_seed(seed=42)

#==================================
# LOAD CITYSCAPES VALIDATION DATA
#==================================

from eomt.datasets.cityscapes_semantic import CityscapesSemantic

data = CityscapesSemantic(
    path=DATA_PATH_CITYSCAPES_VALIDATION,
    batch_size=1,
    num_workers=2,
    img_size=(640, 640)
)

data.setup("validate")
val_loader = data.val_dataloader()

print(f"Val set size: {len(val_loader.dataset)} images")
# prepare the dataset

#==================================
# IOU EVALUATOR
#==================================

evaluator = MulticlassJaccardIndex(
    num_classes=N_CITYSCAPES_CLASSES,
    ignore_index=IGNORE_INDEX,
    average=None,   # return per-class IoU; we'll mean it ourselves
).to(DEVICE)


"""
***************** COCO TRAINED EOMT FINE-TUNED ON CITYSCAPES DATASET EVALUATION ON CITYSCAPES DATASET *****************
"""

#==================================
# 1-BUILD THE MODEL
#==================================

overriders_coco_finetuned_on_cityscapes = {
    ('model', 'init_args', 'network', 'init_args', 'encoder', 'init_args', 'img_size'): IMG_SIZE,
    ('model', 'init_args', 'network', 'init_args', 'num_classes'): N_CITYSCAPES_CLASSES,  # 19, not 133
    ('model', 'init_args', 'network', 'init_args', 'masked_attn_enabled'): False,
    ('model', 'init_args', 'img_size'): IMG_SIZE,
    ('model', 'init_args', 'num_classes'): N_CITYSCAPES_CLASSES,  # 19
}

model_coco_finetuned_on_cityscapes = build_model(
    config_path=CONFIG_PATH_CITYSCAPES_TRAINED,
    eval_mode=True,
    config_overriders=overriders_coco_finetuned_on_cityscapes,
    sanity_check=True,
    checkpoint_path=CHECKPOINT_PATH_COCO_FINETUNED_ON_CITYSCAPES,
    device=DEVICE
)


#==================================
# 2-INFERENCE LOOP
#==================================

evaluator_coco_finetuned_on_cityscapes = semantic_inference(
    model=model_coco_finetuned_on_cityscapes, 
    dataloader=val_loader, 
    remap_function=None,  # no remapping, this model trained on Cityscapes and outputs Cityscapes
    evaluator=evaluator, 
    device=DEVICE, 
    description="Evaluation Fine tuned on Cityscapes dataset version of COCO Trained EoMT on CityScapes DataSet"
)

#==================================
# 3-PRINT & SAVE RESULT
#==================================
result_json_path_coco_finetuned_on_cityscapes = f"{REPO_ROOT}/results/step5/coco_finetuned_on_cityscapes_results.json"
per_class_iou_coco_finetuned_on_cityscapes = evaluator_coco_finetuned_on_cityscapes.compute().cpu().numpy()
print_results(model_name="COCO-trained EoMT Fine-tuned on CityScapes", per_class_iou=per_class_iou_coco_finetuned_on_cityscapes, class_names=CITYSCAPES_CLASS_NAMES, save_json_path=result_json_path_coco_finetuned_on_cityscapes)


"""
***************** COMPARE & SAVE RESULTS FOR CITYSCAPES-TRAINED-EOMT vs COCO-TRAINED-EOMT EVALUATION ON CITYSCAPES DATASET *****************
"""

#==================================
# 1-READ THE COCO-EoMT IOU
#==================================


with open(f"{REPO_ROOT}/results/step4/coco_trained_eomt_on_cityscapes_results.json", 'r') as f:
    coco_results = json.load(f)

per_class_iou_coco_trained = np.array(list(coco_results['per_class'].values()))


#==================================
# 2-COMPARE THE RESULTS
#==================================

compare_result_iou(
    "COCO-trained EoMT Fine Tuned on CityScapes",
    "COCO-trained EoMT", 
    per_class_iou1=per_class_iou_coco_finetuned_on_cityscapes,
    per_class_iou2=per_class_iou_coco_trained,
    class_names=CITYSCAPES_CLASS_NAMES,
    save_json_path=f"{REPO_ROOT}/results/step5/compare_results_coco_trained_vs_coco_finetuned_on_cityscapes.json"
)
