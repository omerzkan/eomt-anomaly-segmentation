
import sys
REPO_ROOT = "/content/cloned_repo_feature_omer"
sys.path.insert(0, REPO_ROOT)
REPO_EOMT = f"{REPO_ROOT}/eomt"
sys.path.insert(0, REPO_EOMT)

import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torchvision.transforms import Compose, Resize, ToTensor

from utils.eomt_utils import DEVICE, build_model, setup_seed, wandb_setup, CITYSCAPES_CLASS_NAMES
from utils.anomaly_utils import MSP, MaxLogit, MaxEntropy, RbA, load_gt_mask

# ==================================
# SETUP
# ==================================

wandb_setup(enable=False)
setup_seed(seed=42)

SAVE_DIR = "/content/drive/MyDrive/FAIMDL/results/figures"
os.makedirs(SAVE_DIR, exist_ok=True)

# Cityscapes color palette (19 classes)
CITYSCAPES_COLORS = np.array([
    [128, 64,128],  # road
    [244, 35,232],  # sidewalk
    [ 70, 70, 70],  # building
    [102,102,156],  # wall
    [190,153,153],  # fence
    [153,153,153],  # pole
    [250,170, 30],  # traffic light
    [220,220,  0],  # traffic sign
    [107,142, 35],  # vegetation
    [152,251,152],  # terrain
    [ 70,130,180],  # sky
    [220, 20, 60],  # person
    [255,  0,  0],  # rider
    [  0,  0,142],  # car
    [  0,  0, 70],  # truck
    [  0, 60,100],  # bus
    [  0, 80,100],  # train
    [  0,  0,230],  # motorcycle
    [119, 11, 32],  # bicycle
], dtype=np.uint8)


def pred_to_color(pred_map, palette=CITYSCAPES_COLORS):
    """Convert class index map (H, W) → RGB image (H, W, 3)."""
    h, w = pred_map.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in range(len(palette)):
        mask = pred_map == cls_id
        color[mask] = palette[cls_id]
    return color


def gt_to_color(gt_path, palette=CITYSCAPES_COLORS):
    """Load Cityscapes GT label and colorize."""
    gt = np.array(Image.open(gt_path))
    return pred_to_color(gt, palette)


# ==============================================================
# FIGURE 1: STEP 4 — Semantic Segmentation Comparison
# ==============================================================

def generate_step4_figures(
    image_paths,
    gt_paths,
    models_dict,       # {"COCO": model, "Cityscapes": model, "Fine-tuned": model}
    transforms_dict,   # {"COCO": transform, "Cityscapes": transform, ...}
    n_samples=3,
    save_prefix="step4",
):
    """
    Generate side-by-side semantic segmentation comparison.
    Columns: Input | GT | COCO pred | Cityscapes pred | Fine-tuned pred
    """
    
    model_names = list(models_dict.keys())
    n_cols = 2 + len(model_names)  # input + GT + N models
    
    fig, axes = plt.subplots(n_samples, n_cols, figsize=(4 * n_cols, 4 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]
    
    for row_idx in range(n_samples):
        img_path = image_paths[row_idx]
        gt_path = gt_paths[row_idx]
        
        # Load input image
        img_pil = Image.open(img_path).convert("RGB")
        img_np = np.array(img_pil)
        
        # Load GT
        gt_color = gt_to_color(gt_path)
        
        # Column 0: input
        axes[row_idx, 0].imshow(img_np)
        if row_idx == 0:
            axes[row_idx, 0].set_title("Input", fontsize=12, fontweight='bold')
        axes[row_idx, 0].axis("off")
        
        # Column 1: GT
        axes[row_idx, 1].imshow(gt_color)
        if row_idx == 0:
            axes[row_idx, 1].set_title("Ground Truth", fontsize=12, fontweight='bold')
        axes[row_idx, 1].axis("off")
        
        # Columns 2+: model predictions
        for col_idx, name in enumerate(model_names):
            model = models_dict[name]
            transform = transforms_dict[name]
            
            # Run inference
            img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad(), autocast(dtype=torch.float16, device_type="cuda"):
                imgs = [img_tensor[0]]
                img_sizes = [imgs[0].shape[-2:]]
                
                transformed_imgs, origins, _ = model.resize_and_pad_imgs_semantic(imgs)
                mask_logits_per_layer, class_logits_per_layer = model(transformed_imgs)
                crop_logits = model.to_per_pixel_logits_semantic(
                    mask_logits_per_layer[-1], class_logits_per_layer[-1]
                )
                logits = model.revert_window_logits_semantic(crop_logits, origins, img_sizes)
            
            pred = logits[0].argmax(dim=0).cpu().numpy()
            pred_color = pred_to_color(pred)
            
            axes[row_idx, 2 + col_idx].imshow(pred_color)
            if row_idx == 0:
                axes[row_idx, 2 + col_idx].set_title(name, fontsize=12, fontweight='bold')
            axes[row_idx, 2 + col_idx].axis("off")
    
    plt.tight_layout(pad=0.5)
    save_path = f"{SAVE_DIR}/{save_prefix}_comparison.pdf"
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    save_path_png = f"{SAVE_DIR}/{save_prefix}_comparison.png"
    fig.savefig(save_path_png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")
    print(f"  Saved: {save_path_png}")


# ==============================================================
# FIGURE 2: STEP 7/8 — Anomaly Score Maps
# ==============================================================

def generate_anomaly_figures(
    image_paths,
    scoring_methods,    # {"MSP": MSP(), "MaxLogit": MaxLogit(), ...}
    model,
    model_name,
    input_transform,
    target_transform,
    n_samples=3,
    save_prefix="anomaly",
    is_mask_based=True,
):
    """
    Generate anomaly score map comparisons.
    Columns: Input | GT mask | score map per method
    """
    
    method_names = list(scoring_methods.keys())
    n_cols = 2 + len(method_names)
    
    fig, axes = plt.subplots(n_samples, n_cols, figsize=(3.5 * n_cols, 3.5 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]
    
    sample_count = 0
    
    for path in image_paths:
        if sample_count >= n_samples:
            break
        
        # Load GT mask
        gt_mask = load_gt_mask(path, target_transform)
        if 1 not in np.unique(gt_mask):
            continue  # skip images without anomaly
        
        # Load and show input image
        img_pil = Image.open(path).convert("RGB")
        img_np = np.array(img_pil)
        
        axes[sample_count, 0].imshow(img_np)
        if sample_count == 0:
            axes[sample_count, 0].set_title("Input", fontsize=11, fontweight='bold')
        axes[sample_count, 0].axis("off")
        
        # Show GT mask
        gt_display = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
        gt_display[gt_mask == 0] = [0, 0, 180]     # inlier = blue
        gt_display[gt_mask == 1] = [255, 0, 0]     # anomaly = red
        gt_display[gt_mask == 255] = [128, 128, 128]  # void = gray
        
        axes[sample_count, 1].imshow(gt_display)
        if sample_count == 0:
            axes[sample_count, 1].set_title("GT Mask", fontsize=11, fontweight='bold')
        axes[sample_count, 1].axis("off")
        
        # Run model inference
        img_tensor = input_transform(img_pil).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad(), autocast(dtype=torch.float16, device_type="cuda"):
            imgs = [img_tensor[0]]
            img_sizes = [imgs[0].shape[-2:]]
            
            if is_mask_based:
                transformed_imgs, origins, _ = model.resize_and_pad_imgs_semantic(imgs)
                mask_logits_per_layer, class_logits_per_layer = model(transformed_imgs)
                crop_logits = model.to_per_pixel_logits_semantic(
                    mask_logits_per_layer[-1], class_logits_per_layer[-1]
                )
                logits = model.revert_window_logits_semantic(crop_logits, origins, img_sizes)
            else:
                # ERFNet pixel-based
                logits = model(img_tensor)
        
        logits_np = logits[0].float().cpu().numpy()
        
        # Generate score maps for each method
        for col_idx, (method_name, method) in enumerate(scoring_methods.items()):
            score = method.anomaly_score(logits_np)
            
            # Resize score map to match input image size if needed
            if score.shape != img_np.shape[:2]:
                score_resized = np.array(
                    Image.fromarray(score.astype(np.float32)).resize(
                        (img_np.shape[1], img_np.shape[0]), Image.BILINEAR
                    )
                )
            else:
                score_resized = score
            
            im = axes[sample_count, 2 + col_idx].imshow(
                score_resized, cmap='hot', interpolation='bilinear'
            )
            if sample_count == 0:
                axes[sample_count, 2 + col_idx].set_title(
                    method_name, fontsize=11, fontweight='bold'
                )
            axes[sample_count, 2 + col_idx].axis("off")
        
        sample_count += 1
    
    plt.tight_layout(pad=0.5)
    save_path = f"{SAVE_DIR}/{save_prefix}_{model_name.lower().replace(' ', '_')}.pdf"
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    save_path_png = f"{SAVE_DIR}/{save_prefix}_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(save_path_png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")
    print(f"  Saved: {save_path_png}")


# ==============================================================
# MAIN — Example usage (run this section on Colab)
# ==============================================================

if __name__ == "__main__":
    
    print("=" * 60)
    print("  Qualitative Figure Generation")
    print("=" * 60)
    print()
    print("This script provides the functions. To generate figures,")
    print("call them from your Colab notebook with your loaded models.")
    print()
    print("Example for Step 4:")
    print("  generate_step4_figures(")
    print("      image_paths=val_images[:3],")
    print("      gt_paths=val_gts[:3],")
    print("      models_dict={'COCO': model_coco, 'Cityscapes': model_city, 'Fine-tuned': model_ft},")
    print("      transforms_dict={'COCO': transform_coco, 'Cityscapes': transform_city, 'Fine-tuned': transform_ft},")
    print("  )")
    print()
    print("Example for Step 8 anomaly maps:")
    print("  generate_anomaly_figures(")
    print("      image_paths=sorted(glob.glob('/content/data/Validation_Dataset/RoadAnomaly21/images/*.png')),")
    print("      scoring_methods={'MSP': MSP(), 'MaxLogit': MaxLogit(), 'MaxEntropy': MaxEntropy(), 'RbA': RbA()},")
    print("      model=model_cityscapes,")
    print("      model_name='EoMT-Cityscapes',")
    print("      input_transform=pil_to_uint8_tensor,")
    print("      target_transform=Resize((640, 640), interpolation=Image.NEAREST),")
    print("  )")