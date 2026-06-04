 
#==================================
# INSERTING PATHS TO THE SYSTEM
#==================================
import glob
import sys
REPO_ROOT = "/content/cloned_repo_feature_omer"
sys.path.insert(0, REPO_ROOT)
 
#==================================
# IMPORTS
#==================================
import os
import json
from utils.anomaly_utils import MSP, MaxEntropy, RbA, evaluate_temperature
 
#==================================
# CONFIGURATION
#==================================
 
TEMPERATURES = [0.5, 0.75, 1.0, 1.1, 1.5, 2.0]
 
SAVED_LOGITS_ROOT = "/content/saved_logits"
 
RESULTS_DIR = "/content/drive/MyDrive/FAIMDL/results/step8"
os.makedirs(RESULTS_DIR, exist_ok=True)
 
DATASET_NAMES = [
    "RoadAnomaly21",
    "RoadObsticle21",
    "FS_LostFound_full",
    "fs_static",
    "RoadAnomaly",
]
 
# EoMT is mask-based → MSP, MaxEntropy, and RbA (all support temperature)
SCORING_METHODS = {
    "MSP": MSP,
    "MaxEntropy": MaxEntropy,
    "RbA": RbA,
}
 
MODELS = ["eomt_cityscapes", "eomt_finetuned"]
 
#==================================
# TEMPERATURE SCALING
#==================================
 
for model_name in MODELS:
    
    print(f"\n{'=' * 50}")
    print(f"Temperature scaling: {model_name}")
    print(f"Temperatures: {TEMPERATURES}")
    print(f"Methods: {list(SCORING_METHODS.keys())}")
    print(f"{'=' * 50}")
    
    all_results = {}
    
    for dataset_name in DATASET_NAMES:
        
        logits_dir = f"{SAVED_LOGITS_ROOT}/{model_name}/{dataset_name}"
        
        has_batches = len(glob.glob(os.path.join(logits_dir, "batch_*.npz"))) > 0
        if not has_batches:
            print(f"  WARNING: No saved logits for {model_name}/{dataset_name}, skipping")
            continue
        
        print(f"  {dataset_name}...", end=" ")
                
        dataset_results = evaluate_temperature(
            saved_logits_path=logits_dir,
            scoring_methods=SCORING_METHODS,
            temperatures=TEMPERATURES,
        )
        
        all_results[dataset_name] = dataset_results
        print("done")
    
    #==================================
    # SAVE RESULTS
    #==================================

    save_path = f"{RESULTS_DIR}/{model_name}_temperature.json"
    
    with open(save_path, 'w') as f:
        json.dump({
            "model_name": model_name,
            "temperatures": TEMPERATURES,
            "methods": list(SCORING_METHODS.keys()),
            "results": all_results,
        }, f, indent=2)
    
    print(f"\nResults saved to {save_path}")
    
    #==================================
    # PRINT SUMMARY — best T per method per dataset
    #==================================
    print(f"\nBest temperature per method per dataset AuPRC & FPR95:")
    
    for dataset_name, dataset_results in all_results.items():
        print(f"\n  {dataset_name}:")
        for method_name, t_results in dataset_results.items():
            print(f"    {method_name}:")
            for T in TEMPERATURES:
                auprc = t_results[T]['auprc'] * 100
                fpr95 = t_results[T]['fpr95'] * 100
                print(f"      T={T}  AuPRC={auprc:.2f}%  FPR95={fpr95:.2f}%")
 
print("\n\nDone! All temperature scaling results saved.")
 