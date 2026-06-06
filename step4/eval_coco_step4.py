
"""
This is the evaluation script for 
the COCO-Trained-EoMT model vs the Cityscapes-Trained-EoMT model on Cityscapes validation set.
It evaluates the performance of both models on the Cityscapes validation set and compares their results.

Using coco_to_cityscapes.py to convert COCO annotations to Cityscapes format, 
and then using the converted annotations for evaluation.
"""

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

import os
import torch
import numpy as np
from torchmetrics.classification import MulticlassJaccardIndex
from utils.eomt_utils import CITYSCAPES_CLASS_NAMES, DEVICE, IGNORE_INDEX, N_CITYSCAPES_CLASSES
from utils.eomt_utils import build_model, compare_result_iou, print_results, semantic_inference, setup_seed, wandb_setup, single_semantic_inference, create_mapping, apply_colormap
import matplotlib.pyplot as plt

#==================================
# CONFIGURATION & SETUP
#==================================

CONFIG_PATH_COCO_TRAINED = f"{REPO_EOMT}/configs/dinov2/coco/panoptic/eomt_base_640_2x.yaml"
CONFIG_PATH_CITYSCAPES_TRAINED = f"{REPO_EOMT}/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml"
# Path to the YAML config that describes EoMT's architecture and training settings.
# This particular config is for a ViT-Base backbone, 640px input, trained on COCO panoptic.

CHECKPOINT_PATH_COCO_EOMT = "/content/drive/MyDrive/FAIMDL/checkpoints/eomt_coco.bin"
CHECKPOINT_PATH_CITYSCAPES_EOMT = "/content/drive/MyDrive/FAIMDL/checkpoints/eomt_cityscapes.bin"
# This is the path for the checkpoints for the COCO-trained-EoMT and Cityscapes-trained-EoMT model weights
# This model weights saved in the google drive. So, we gave the path in the google drive.

DATA_PATH_CITYSCAPES_VALIDATION = "/content/drive/MyDrive/FAIMDL/data"
# Root folder containing the dataset (images + annotations).

SAMPLE_INDICES = np.linspace(0, 499, 3, dtype=int)

os.chdir(REPO_ROOT)
# We change the directory to the REPO

wandb_setup(enable=False)
setup_seed(seed=42)

#==================================
# LOAD CITYSCAPES VALIDATION DATA
#==================================

from eomt.datasets.cityscapes_semantic import CityscapesSemantic

data = CityscapesSemantic(
    path=DATA_PATH_CITYSCAPES_VALIDATION,
    batch_size=1,
    num_workers=2,
    img_size=[896, 896]
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
***************** COCO TRAINED EOMT EVALUATION ON CITYSCAPES DATASET *****************
"""
print("\n***************** COCO TRAINED EOMT EVALUATION ON CITYSCAPES DATASET *****************\n")

#==================================
# 1-BUILD THE MODEL
#==================================

IMG_SIZE_COCO = [640, 640]
N_COCO_CLASSES = 133
STUFF_CLASSES_COCO = [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132]
overriders_coco_trained= {
    ('model', 'init_args', "network", "init_args", "encoder", "init_args", 'img_size'): IMG_SIZE_COCO,
    ('model', 'init_args', "network", "init_args", "num_classes"): N_COCO_CLASSES,
    ('model', 'init_args', "network", "init_args", "masked_attn_enabled"): False,
    ('model', 'init_args', "img_size"): IMG_SIZE_COCO,
    ('model', 'init_args', "num_classes"): N_COCO_CLASSES,
    ('model', 'init_args', "stuff_classes"): STUFF_CLASSES_COCO
}

model_coco_trained = build_model(
    config_path=CONFIG_PATH_COCO_TRAINED,
    eval_mode=True,
    config_overriders=overriders_coco_trained,
    sanity_check=False,
    checkpoint_path=CHECKPOINT_PATH_COCO_EOMT,
    device=DEVICE
)


#==================================
# 2-ID MAPPING & INFERENCE LOOP
#==================================

from coco_to_cityscapes import coco_to_city_id_map
lookup = torch.full((133,), IGNORE_INDEX, dtype=torch.long, device=DEVICE)

for coco_id, city_id in coco_to_city_id_map.items():
    lookup[coco_id] = city_id

print("==================================")
print("SANITY CHECK: REMAPPING RESULT")
print("==================================\n")
print(f"Mapped Classes: {(lookup != IGNORE_INDEX).sum().item()}/133 COCO classes mapped to Cityscapes")


print("\n==================================")
print("SANITY CHECK: EVALUATION START")
print("==================================\n")
evaluator_coco_trained = semantic_inference(
    model=model_coco_trained, 
    dataloader=val_loader, 
    remap_function=lambda pred: lookup[pred],  # ← this applies the COCO→CS mapping
    evaluator=evaluator, 
    device="cuda", 
    description="Evaluation COCO Trained EoMT on CityScapes DataSet"
)

#==================================
# 3-PRINT & SAVE RESULT
#==================================
coco_trained_result_json_path = f"/content/drive/MyDrive/FAIMDL/results/step4/coco_trained_eomt_on_cityscapes_results.json"
per_class_iou_coco_trained = evaluator_coco_trained.compute().cpu().numpy()

df_miou_coco, df_iou_per_class_coco = print_results(model_name="COCO-trained EoMT", per_class_iou=per_class_iou_coco_trained, class_names=CITYSCAPES_CLASS_NAMES, save_json_path=coco_trained_result_json_path)

print("\n")
print(df_miou_coco)
print("\n")
print(df_iou_per_class_coco)

#==================================
# 4-SAVE 3 COCO PREDICTIONS FOR FIGURE
#==================================

# Save 3 COCO predictions for qualitative figure
coco_preds_for_fig = []
for idx in SAMPLE_INDICES:
    img, target = val_loader.dataset[idx]
    pred = single_semantic_inference(model_coco_trained, img, remap_fn=lambda p: lookup[p])
    coco_preds_for_fig.append(pred)
print(f"Saved {len(coco_preds_for_fig)} COCO predictions for figure.")

#==================================
# 5-RESET EVALUATOR & GPU CACHE
#==================================

del model_coco_trained
torch.cuda.empty_cache()

evaluator.reset()

"""
***************** CITYSCAPES TRAINED EOMT EVALUATION ON CITYSCAPES DATASET *****************
"""
print("\n***************** CITYSCAPES TRAINED EOMT EVALUATION ON CITYSCAPES DATASET *****************\n")


#==================================
# 1-BUILD THE MODEL
#==================================
IMG_SIZE_CITYSCAPE = [1024,1024]
overriders_cityscapes_trained = {
    ('model', 'init_args', 'network', 'init_args', 'encoder', 'init_args', 'img_size'): IMG_SIZE_CITYSCAPE,
    ('model', 'init_args', 'network', 'init_args', 'num_classes'): N_CITYSCAPES_CLASSES,  # 19, not 133
    ('model', 'init_args', 'network', 'init_args', 'masked_attn_enabled'): False,
    ('model', 'init_args', 'img_size'): IMG_SIZE_CITYSCAPE,
    ('model', 'init_args', 'num_classes'): N_CITYSCAPES_CLASSES,  # 19
}

model_cityscapes_trained = build_model(
    config_path=CONFIG_PATH_CITYSCAPES_TRAINED,
    eval_mode=True,
    config_overriders=overriders_cityscapes_trained,
    sanity_check=False,
    checkpoint_path=CHECKPOINT_PATH_CITYSCAPES_EOMT,
    device=DEVICE
)


#==================================
# 2-INFERENCE LOOP
#==================================

print("\n==================================")
print("SANITY CHECK: EVALUATION START")
print("==================================\n")
evaluator_cityscapes_trained = semantic_inference(
    model=model_cityscapes_trained, 
    dataloader=val_loader, 
    remap_function=None,  # ← no remapping needed since this model is trained on Cityscapes and outputs Cityscapes IDs directly
    evaluator=evaluator, 
    device="cuda", 
    description="Evaluation CITYSCAPES-Trained-EoMT on CityScapes DataSet"
)

#==================================
# 3-PRINT & SAVE RESULT
#==================================
cityscapes_trained_result_json_path = f"/content/drive/MyDrive/FAIMDL/results/step4/cityscapes_trained_eomt_on_cityscapes_results.json"
per_class_iou_cityscapes_trained = evaluator_cityscapes_trained.compute().cpu().numpy()

df_miou_cityscapes, df_iou_per_class_cityscapes = print_results(model_name="CITYSCAPES-trained EoMT", per_class_iou=per_class_iou_cityscapes_trained, class_names=CITYSCAPES_CLASS_NAMES, save_json_path=cityscapes_trained_result_json_path)
print("\n")
print(df_miou_cityscapes)
print("\n")
print(df_iou_per_class_cityscapes)

#==================================
# 4-SAVE 3 CITYSCAPES PREDICTIONS FOR FIGURE
#==================================

city_preds_for_fig, targets_for_fig, imgs_for_fig = [], [], []
for idx in SAMPLE_INDICES:
    img, target = val_loader.dataset[idx]
    city_preds_for_fig.append(single_semantic_inference(model_cityscapes_trained, img))
    targets_for_fig.append(model_cityscapes_trained.to_per_pixel_targets_semantic([target], IGNORE_INDEX)[0].numpy())
    imgs_for_fig.append(img)

"""
***************** COMPARE & SAVE RESULTS FOR CITYSCAPES-TRAINED-EOMT vs COCO-TRAINED-EOMT EVALUATION ON CITYSCAPES DATASET *****************
"""
df_miou_compare, df_iou_per_class_compare = compare_result_iou(
    "COCO-trained EoMT", 
    "CITYSCAPES-trained EoMT",
    per_class_iou1=per_class_iou_coco_trained,
    per_class_iou2=per_class_iou_cityscapes_trained,
    class_names=CITYSCAPES_CLASS_NAMES,
    save_json_path=f"/content/drive/MyDrive/FAIMDL/results/step4/compare_results_coco_trained_vs_cityscapes_trained_on_cityscapes_dataset.json"
)

print("\n")
print(df_miou_compare)
print("\n")
print(df_iou_per_class_compare)


"""
***************** BUILD FIGURE & SAVE FOR CITYSCAPES-TRAINED-EOMT vs COCO-TRAINED-EOMT EVALUATION ON CITYSCAPES DATASET *****************
"""

FIGURE_DIR = "/content/drive/MyDrive/FAIMDL/results/figures/step4"
os.makedirs(FIGURE_DIR, exist_ok=True)

fig, axes = plt.subplots(3, 4, figsize=(20, 12))
titles = ['Input', 'Ground Truth', 'EoMT-COCO', 'EoMT-Cityscapes']

for row in range(3):
    mapping = create_mapping([targets_for_fig[row], coco_preds_for_fig[row], city_preds_for_fig[row]], IGNORE_INDEX)
    axes[row,0].imshow(imgs_for_fig[row].permute(1,2,0).cpu().numpy())
    axes[row,1].imshow(apply_colormap(targets_for_fig[row], mapping))
    axes[row,2].imshow(apply_colormap(coco_preds_for_fig[row], mapping))
    axes[row,3].imshow(apply_colormap(city_preds_for_fig[row], mapping))
    for col in range(4):
        axes[row,col].axis('off')
        if row == 0: axes[row,col].set_title(titles[col], fontsize=14, fontweight='bold')

plt.tight_layout()
fig.savefig(f'{FIGURE_DIR}/step4_comparison.pdf', dpi=300, bbox_inches='tight')
fig.savefig(f'{FIGURE_DIR}/step4_comparison.png', dpi=200, bbox_inches='tight')
plt.show()
print(f'Saved to {FIGURE_DIR}/')
