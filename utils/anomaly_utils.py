
import json
import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from ood_metrics import fpr_at_95_tpr
from PIL import Image
import torch
from tqdm import tqdm
import torch.nn.functional as F

def calc_softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=0, keepdims=True))
    probs= exp_logits / np.sum(exp_logits, axis=0, keepdims=True)
    
    return probs


def apply_temperature(logits, temperature=1.0):
    
    scaled_logits = logits / temperature
    return scaled_logits

class MaxLogit:
    
    def anomaly_score(self, logits): # this is the raw logits before softmax
        
        anomaly_result = 1.0 - np.max(logits, axis=0)    
        return anomaly_result
    
    
class MSP:
    def __init__(self, temperature=1.0):
        self.temperature = temperature
    
    def anomaly_score(self, logits): 
        
        scaled_logits = apply_temperature(logits, self.temperature)    
        probs = calc_softmax(scaled_logits)
        
        anomaly_result = 1.0 - np.max(probs, axis=0)
        return anomaly_result


class MaxEntropy:
    
    def __init__(self, temperature=1.0):
        self.temperature = temperature
    
    def anomaly_score(self, logits):

        scaled_logits = apply_temperature(logits, self.temperature)
        
        probs = calc_softmax(scaled_logits)
        
        log_probs = np.log(probs + 1e-12) # I added small amount to avoid log(0)
        entropy = -np.sum(probs * log_probs, axis=0)
        
        return entropy

class RbA:
    def __init__(self, temperature=1.0):
        self.temperature = temperature
    
    def anomaly_score(self, logits):
        
        scaled_logits = apply_temperature(logits, self.temperature)    
        return -np.tanh(scaled_logits).sum(axis=0)
    

def load_gt_mask(image_path, target_transform=None):
    
    """
    0 --> inlier
    1 --> anomaly
    255 --> void    
    """
        
    pathGT = image_path.replace("images", "labels_masks")                
    if "RoadObsticle21" in pathGT:
        pathGT = pathGT.replace("webp", "png")
    if "fs_static" in pathGT:
        pathGT = pathGT.replace("jpg", "png")                
    if "RoadAnomaly" in pathGT:
        pathGT = pathGT.replace("jpg", "png")  

    mask = Image.open(pathGT)
    if target_transform is not None:
        mask = target_transform(mask)

    ood_gts = np.array(mask)

    if "RoadAnomaly" in pathGT:
        ood_gts = np.where((ood_gts==2), 1, ood_gts)
        
    if "LostAndFound" in pathGT:
        ood_gts = np.where((ood_gts==0), 255, ood_gts)
        ood_gts = np.where((ood_gts==1), 0, ood_gts)
        ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)

    return ood_gts



def calc_anomaly_metrics(ood_gts_list, anomaly_score_list):
    
    
    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)  

    ood_mask = (ood_gts == 1)
    inl_mask = (ood_gts == 0)              

    ood_out = anomaly_scores[ood_mask]      # flat array of all anomaly-pixel scores
    inl_out = anomaly_scores[inl_mask]      # flat array of all inlier-pixel scores

    ood_label = np.ones(len(ood_out))       # labels: 1 for all anomaly pixels
    inl_label = np.zeros(len(inl_out))      # labels: 0 for all inlier pixels

    scores = np.concatenate((inl_out, ood_out))    # all scores in one flat array
    labels = np.concatenate((inl_label, ood_label))  # corresponding labels

    prc_auc = average_precision_score(labels, scores)
    fpr = fpr_at_95_tpr(scores, labels)
    
    return {'auprc': prc_auc, 'fpr95': fpr}


def erfnet_anomaly_inference(model, image_paths, scoring_methods, input_transform, target_transform, device="cuda", description="ERFNeT Anomaly Inference", save_logits_path=None):
    
    model.eval()
    ood_gts_list = []
    scores_per_method = {name: [] for name in scoring_methods}
    
    saved_logits = []
    saved_gts = []
    
    with torch.no_grad():
        for path in tqdm(image_paths, desc=description):
            
            image = Image.open(path).convert("RGB")
            input_tensor = input_transform(image).unsqueeze(0).float().to(device)
            
            logits = model(input_tensor)
            logits_np = logits.squeeze(0).cpu().numpy()
            
            gt_mask = load_gt_mask(path, target_transform)
            
            if 1 not in np.unique(gt_mask):
                continue
            
            ood_gts_list.append(gt_mask)
            
            for name, method in scoring_methods.items():
                
                anomaly_score = method.anomaly_score(logits_np)
                scores_per_method[name].append(anomaly_score)
                
            if save_logits_path is not None:
                saved_logits.append(logits_np.astype(np.float16))
                saved_gts.append(gt_mask.astype(np.uint8))
                
                if len(saved_logits) >= 50:
                    batch_num = len([f for f in os.listdir(save_logits_path) if f.startswith("batch_")]) if os.path.exists(save_logits_path) else 0
                    save_batch_to_npz(save_logits_path, batch_num, saved_logits, saved_gts)
                    saved_logits = []
                    saved_gts = []
                        
    results = {}
    for name, scores in scores_per_method.items():
        metrics = calc_anomaly_metrics(ood_gts_list, scores)
        results[name] = metrics
        
    if save_logits_path is not None:
        if saved_logits:
            batch_num = len([f for f in os.listdir(save_logits_path) if f.startswith("batch_")]) if os.path.exists(save_logits_path) else 0
            save_batch_to_npz(save_logits_path, batch_num, saved_logits, saved_gts)
        print(f"\nAll batch logits saved to {save_logits_path}/\n") 
    
    return results


def save_batch_to_npz(save_logits_path, batch_num, logits_list, gt_list):
    """Save a single batch of logits to disk."""
    os.makedirs(save_logits_path, exist_ok=True)
    batch_file = os.path.join(save_logits_path, f"batch_{batch_num}.npz")
    
    start = time.time()
    np.savez(batch_file,
                        logits=np.array(logits_list, dtype=object),
                        gt=np.array(gt_list, dtype=object))
    
    print(f"  Save took {time.time()-start:.1f}s")
    print(f"  → Saved batch {batch_num} ({len(logits_list)} images)")

    
def eomt_anomaly_inference(model, image_paths, scoring_methods, input_transform, target_transform, device="cuda", description="EoMT Anomaly Inference", save_logits_path=None):

    model.eval()
    ood_gts_list = []
    scores_per_method = {name: [] for name in scoring_methods}
    
    saved_logits = []
    saved_gts = []
    
    with torch.no_grad():
        for path in tqdm(image_paths, desc=description):
            
            image = Image.open(path).convert("RGB")
            img_tensor = input_transform(image).to(device)
            
            imgs = [img_tensor]
            img_sizes = [img_tensor.shape[-2:]]
            
            with torch.autocast(device_type=device):
                
                crops, origins = model.window_imgs_semantic(imgs)
                mask_logits_list, class_logits_list = model(crops)
                mask_logits = F.interpolate(mask_logits_list[-1], model.img_size, mode='bilinear')
                crop_logits = model.to_per_pixel_logits_semantic(
                    mask_logits, class_logits_list[-1]
                )
                logits = model.revert_window_logits_semantic(
                    crop_logits, origins, img_sizes
                )
            
            logits_np = logits[0].float().cpu().numpy()
            
            gt_mask = load_gt_mask(path, target_transform)
            
            if 1 not in np.unique(gt_mask):
                continue
            
            ood_gts_list.append(gt_mask)
            
            for name, method in scoring_methods.items():
                
                anomaly_score = method.anomaly_score(logits_np)
                scores_per_method[name].append(anomaly_score)
        
            if save_logits_path is not None:
                saved_logits.append(logits_np.astype(np.float16))
                saved_gts.append(gt_mask.astype(np.uint8))
                
                if len(saved_logits) >= 20:
                    batch_num = len([f for f in os.listdir(save_logits_path) if f.startswith("batch_")]) if os.path.exists(save_logits_path) else 0
                    save_batch_to_npz(save_logits_path, batch_num, saved_logits, saved_gts)
                    saved_logits = []
                    saved_gts = []

    results = {}
    for name, scores in scores_per_method.items():
        metrics = calc_anomaly_metrics(ood_gts_list, scores)
        results[name] = metrics
    
    if save_logits_path is not None:
        if saved_logits:
            batch_num = len([f for f in os.listdir(save_logits_path) if f.startswith("batch_")]) if os.path.exists(save_logits_path) else 0
            save_batch_to_npz(save_logits_path, batch_num, saved_logits, saved_gts)

        print(f"\nAll batch logits saved to {save_logits_path}/\n")
    
    return results

def print_anomaly_results(model_name, all_results, save_json_path=None):
    
    print(f"\n============ {model_name} ============\n")
    
    dataset_names = list(all_results.keys())
    
    method_names = []    
    for dataset in dataset_names:
        for method in all_results[dataset].keys():
            if method not in method_names:
                method_names.append(method)
    
    auprc_data = {
        dataset: [all_results[dataset][m]["auprc"] * 100 for m in method_names] for dataset in dataset_names
    }
    df_auprc = pd.DataFrame(auprc_data, index=method_names)
    
    fpr_data = {
        dataset: [all_results[dataset][m]["fpr95"] * 100 for m in method_names] for dataset in dataset_names
        }
    df_fpr = pd.DataFrame(fpr_data, index=method_names)
    
    
    results = {
        'model_name': model_name,
        'datasets': dataset_names,
        'methods': method_names,
        'per_dataset': all_results,
    }

    if save_json_path is not None:
        with open(save_json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {save_json_path}\n")

    return df_auprc, df_fpr

def evaluate_temperature(saved_logits_path, scoring_methods, temperatures):
    import glob
    
    batch_files = sorted(glob.glob(os.path.join(saved_logits_path, "batch_*.npz")))
    if not batch_files:
        raise FileNotFoundError(f"No batch files in {saved_logits_path}")
    
    # Load ALL images once, cast to float32 once
    # This trades RAM for speed — fine for ERFNet (small images)
    all_logits_f32 = []
    all_gts = []
    for bf in batch_files:
        data = np.load(bf, allow_pickle=True)
        for logits, gt in zip(data['logits'], data['gt']):
            all_logits_f32.append(np.array(logits, dtype=np.float32))
            all_gts.append(np.array(gt))
        del data
    
    print(f"  (Loaded {len(all_logits_f32)} images)")
    
    results = {name: {} for name in scoring_methods}
    
    for method_name, method_class in scoring_methods.items():
        for T in temperatures:
            scorer = method_class(temperature=T)
            scores = [scorer.anomaly_score(logits) for logits in all_logits_f32]
            metrics = calc_anomaly_metrics(all_gts, scores)
            results[method_name][T] = metrics
    
    return results