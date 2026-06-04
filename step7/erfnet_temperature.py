
#==================================
# INSERTING PATHS TO THE SYSTEM
#==================================
import sys
REPO_ROOT = "/content/cloned_repo_feature_omer"
sys.path.insert(0, REPO_ROOT)
 
#==================================
# IMPORTS
#==================================
import os
import json
from utils.anomaly_utils import MSP, MaxEntropy, evaluate_temperature

#==================================
# CONFIGURATION
#==================================
 
TEMPERATURES = [0.5, 0.75, 1.0, 1.1, 1.5, 2.0]
 
SAVED_LOGITS_ROOT = "/content/saved_logits/erfnet"
 
RESULTS_DIR = "/content/drive/MyDrive/FAIMDL/results/step7"
 
DATASET_NAMES = [
    "RoadAnomaly21",
    "RoadObsticle21",
    "FS_LostFound_full",
    "fs_static",
    "RoadAnomaly",
]
 
# ERFNet is pixel-based → only MSP and MaxEntropy (no RbA)
SCORING_METHODS = {
    "MSP": MSP,
    "MaxEntropy": MaxEntropy,
}
 
#==================================
# TEMPERATURE SCALING
#==================================
print("=" * 50)
print("Temperature scaling: ERFNet")
print(f"Temperatures: {TEMPERATURES}")
print(f"Methods: {list(SCORING_METHODS.keys())}")
print("=" * 50)
 
all_results = {}
 
for dataset_name in DATASET_NAMES:
    
    logits_dir = f"{SAVED_LOGITS_ROOT}/{dataset_name}"
    
    if not os.path.exists(os.path.join(logits_dir, "logits_and_gt.npz")):
        print(f"WARNING: No saved logits for erfnet/{dataset_name}, skipping")
        continue
    
    print(f"  {dataset_name}...", end=" ")
    
    logits_file = os.path.join(logits_dir, "logits_and_gt.npz")
    dataset_results = evaluate_temperature(
        saved_logits_path=logits_file,
        scoring_methods=SCORING_METHODS,
        temperatures=TEMPERATURES,
    )
    
    all_results[dataset_name] = dataset_results
    print("done")
 
#==================================
# SAVE RESULTS
#==================================
save_path = f"{RESULTS_DIR}/erfnet_temperature.json"
 
with open(save_path, 'w') as f:
    json.dump({
        "model_name": "erfnet",
        "temperatures": TEMPERATURES,
        "methods": list(SCORING_METHODS.keys()),
        "results": all_results,
    }, f, indent=2)
 
print(f"\nResults saved to {save_path}")
 
#==================================
# PRINT SUMMARY — best T per method per dataset
#==================================
print("\n" + "=" * 50)
print("Best temperature per method per dataset AuPRC & FPR95")
print("=" * 50)
 

for dataset_name, dataset_results in all_results.items():
    print(f"\n  {dataset_name}:")
    for method_name, t_results in dataset_results.items():
        print(f"    {method_name}:")
        for T in TEMPERATURES:
            auprc = t_results[T]['auprc'] * 100
            fpr95 = t_results[T]['fpr95'] * 100
            print(f"      T={T}  AuPRC={auprc:.2f}%  FPR95={fpr95:.2f}%")