

import json
import os
import pandas as pd
 
#==================================
# CONFIGURATION
#==================================
 
RESULTS_ROOT = "/content/drive/MyDrive/FAIMDL/results"
 
# All result JSON files
RESULT_FILES = {
    "ERFNet":            f"{RESULTS_ROOT}/step7/erfnet_anomaly_results.json",
    "EoMT-COCO":         f"{RESULTS_ROOT}/step8/eomt_coco_anomaly_results.json",
    "EoMT-Cityscapes":   f"{RESULTS_ROOT}/step8/eomt_cityscapes_anomaly_results.json",
    "EoMT-Fine-tuned":   f"{RESULTS_ROOT}/step8/eomt_finetuned_anomaly_results.json",
}

MIOU_VALUES = {
    "ERFNet":           72.5,     # Took from ERFNet paper
    "EoMT-COCO":        55.17,    # from step4
    "EoMT-Cityscapes":  81.68,    # from step4
    "EoMT-Fine-tuned":  78.85,    # from step5
}
 
# Short dataset names for the table
DATASET_SHORT = {
    "RoadAnomaly21":     "RA-21",
    "RoadObsticle21":    "RO-21",
    "FS_LostFound_full": "FS L&F",
    "fs_static":         "FS Static",
    "RoadAnomaly":       "RoadAnom",
}

#==================================
# LOAD ALL RESULTS
#==================================
 
all_data = {}
for model_name, filepath in RESULT_FILES.items():
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            all_data[model_name] = json.load(f)
        print(f"Loaded: {model_name}")
    else:
        print(f"WARNING: Missing {filepath}")

#==================================
# BUILD THE TABLE
#==================================
 
rows = []
 
for model_name, data in all_data.items():
    miou = MIOU_VALUES.get(model_name, "—")
    methods = data["methods"]
    datasets = data["datasets"]
    
    for method in methods:
        row = {
            "Model": model_name,
            "mIoU": miou,
            "Method": method,
        }
        
        for dataset in datasets:
            short_name = DATASET_SHORT.get(dataset, dataset)
            auprc = data["per_dataset"][dataset][method]["auprc"] * 100
            fpr95 = data["per_dataset"][dataset][method]["fpr95"] * 100
            row[f"{short_name} AuPRC"] = round(auprc, 2)
            row[f"{short_name} FPR95"] = round(fpr95, 2)
        
        rows.append(row)
 
df_table = pd.DataFrame(rows)

#==================================
# PRINT THE TABLES
#==================================
 
print("\n" + "=" * 80)
print("TABLE — AuPRC (%)")
print("=" * 80)
 
auprc_cols = ["Model", "mIoU", "Method"] + [c for c in df_table.columns if "AuPRC" in c]
print(df_table[auprc_cols].to_string(index=False))
 
print("\n" + "=" * 80)
print("TABLE — FPR95 (%)")
print("=" * 80)
 
fpr_cols = ["Model", "mIoU", "Method"] + [c for c in df_table.columns if "FPR95" in c]
print(df_table[fpr_cols].to_string(index=False))

#==================================
# SAVE AS CSV
#==================================
 
save_dir = f"{RESULTS_ROOT}/final"
os.makedirs(save_dir, exist_ok=True)
 
df_table.to_csv(f"{save_dir}/result_table.csv", index=False)
print(f"\nSaved to {save_dir}/result_table.csv")
 