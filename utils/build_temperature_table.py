import json
import os
import pandas as pd

#==================================
# CONFIGURATION
#==================================

RESULTS_ROOT = "/content/drive/MyDrive/FAIMDL/results"

# Temperature scaling result JSON files
TEMPERATURE_FILES = {
    "EoMT-Cityscapes": f"{RESULTS_ROOT}/step8/eomt_cityscapes_temperature.json",
    "EoMT-Fine-tuned": f"{RESULTS_ROOT}/step8/eomt_finetuned_temperature.json",
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
# LOAD ALL TEMPERATURE RESULTS
#==================================

all_temp_data = {}
for model_name, filepath in TEMPERATURE_FILES.items():
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            all_temp_data[model_name] = json.load(f)
        print(f"Loaded: {model_name}")
    else:
        print(f"WARNING: Missing {filepath}")

#==================================
# BUILD TEMPERATURE TABLES
#==================================

for model_name, data in all_temp_data.items():
    
    print(f"\n\n{'='*100}")
    print(f"MODEL: {model_name}")
    print(f"{'='*100}\n")
    
    temperatures = data["temperatures"]
    methods = data["methods"]
    datasets = data["datasets"]
    results = data["results"]
    
    # Build separate tables for AUPRC and FPR95
    
    # ===== AUPRC TABLE =====
    print(f"AuPRC (%) — Impact of Temperature Scaling\n")
    
    auprc_rows = []
    for method in methods:
        row = {"Method": method}
        
        for dataset in datasets:
            short_name = DATASET_SHORT.get(dataset, dataset)
            
            # For each temperature, get AUPRC
            temps_auprc = []
            for T in temperatures:
                T_str = str(T)
                auprc = results[dataset][method][T_str]["auprc"] * 100
                temps_auprc.append(f"{auprc:.2f}")
            
            # Create column name with temperature values
            row[short_name] = " | ".join(temps_auprc)
        
        auprc_rows.append(row)
    
    df_auprc = pd.DataFrame(auprc_rows)
    print(df_auprc.to_string(index=False))
    print(f"\nTemperature values: {temperatures}\n")
    
    # ===== FPR95 TABLE =====
    print(f"\n{'─'*100}\n")
    print(f"FPR95 (%) — Impact of Temperature Scaling\n")
    
    fpr_rows = []
    for method in methods:
        row = {"Method": method}
        
        for dataset in datasets:
            short_name = DATASET_SHORT.get(dataset, dataset)
            
            # For each temperature, get FPR95
            temps_fpr = []
            for T in temperatures:
                T_str = str(T)
                fpr95 = results[dataset][method][T_str]["fpr95"] * 100
                temps_fpr.append(f"{fpr95:.2f}")
            
            # Create column name with temperature values
            row[short_name] = " | ".join(temps_fpr)
        
        fpr_rows.append(row)
    
    df_fpr = pd.DataFrame(fpr_rows)
    print(df_fpr.to_string(index=False))
    print(f"\nTemperature values: {temperatures}\n")
    
    # ===== BEST TEMPERATURE PER METHOD PER DATASET =====
    print(f"\n{'─'*100}\n")
    print(f"BEST TEMPERATURE (highest AUPRC)\n")
    
    best_rows = []
    for method in methods:
        for dataset in datasets:
            short_name = DATASET_SHORT.get(dataset, dataset)
            
            best_T = None
            best_auprc = -1
            best_fpr95 = None
            
            for T in temperatures:
                T_str = str(T)
                auprc = results[dataset][method][T_str]["auprc"]
                if auprc > best_auprc:
                    best_auprc = auprc
                    best_T = T
                    best_fpr95 = results[dataset][method][T_str]["fpr95"]
            
            best_rows.append({
                "Dataset": short_name,
                "Method": method,
                "Best T": best_T,
                "AuPRC (%)": round(best_auprc * 100, 2),
                "FPR95 (%)": round(best_fpr95 * 100, 2),
            })
    
    df_best = pd.DataFrame(best_rows)
    print(df_best.to_string(index=False))

#==================================
# SAVE DETAILED CSV FILES
#==================================

save_dir = f"{RESULTS_ROOT}/final"
os.makedirs(save_dir, exist_ok=True)

for model_name, data in all_temp_data.items():
    
    temperatures = data["temperatures"]
    methods = data["methods"]
    datasets = data["datasets"]
    results = data["results"]
    
    # Build detailed rows for CSV (one row per method-dataset-temperature combo)
    detailed_rows = []
    for method in methods:
        for dataset in datasets:
            for T in temperatures:
                T_str = str(T)
                auprc = results[dataset][method][T_str]["auprc"] * 100
                fpr95 = results[dataset][method][T_str]["fpr95"] * 100
                
                detailed_rows.append({
                    "Model": model_name,
                    "Method": method,
                    "Dataset": dataset,
                    "Temperature": T,
                    "AuPRC (%)": round(auprc, 2),
                    "FPR95 (%)": round(fpr95, 2),
                })
    
    df_detailed = pd.DataFrame(detailed_rows)
    csv_filename = f"{save_dir}/{model_name.lower().replace('-', '_')}_temperature.csv"
    df_detailed.to_csv(csv_filename, index=False)
    print(f"\nSaved detailed results to {csv_filename}")

print(f"\nAll temperature scaling tables and CSVs saved to {save_dir}/")
