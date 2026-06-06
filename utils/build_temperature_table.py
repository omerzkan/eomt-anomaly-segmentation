

import json
import os
import pandas as pd
 
# ==================================
# CONFIGURATION
# ==================================
 
RESULTS_ROOT = "/content/drive/MyDrive/FAIMDL/results"
 
TEMPERATURE_FILES = {
    "ERFNet":            f"{RESULTS_ROOT}/step7/erfnet_temperature.json",
    "EoMT-Cityscapes":   f"{RESULTS_ROOT}/step8/eomt_cityscapes_temperature.json",
    "EoMT-Fine-tuned":   f"{RESULTS_ROOT}/step8/eomt_finetuned_temperature.json",
}
 
MIOU_VALUES = {
    "ERFNet":           72.50,
    "EoMT-Cityscapes":  81.68,
    "EoMT-Fine-tuned":  78.85,
}
 
DATASET_SHORT = {
    "RoadAnomaly21":     "RA-21",
    "RoadObsticle21":    "RO-21",
    "FS_LostFound_full": "FS L&F",
    "fs_static":         "FS Static",
    "RoadAnomaly":       "RoadAnom",
}
 
# Temperatures to show as explicit rows (besides t=1.0 baseline and best)
DISPLAY_TEMPS = [0.5, 0.75, 1.1]
 
# ==================================
# LOAD DATA
# ==================================
 
all_temp_data = {}
for model_name, filepath in TEMPERATURE_FILES.items():
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            all_temp_data[model_name] = json.load(f)
        print(f"  Loaded: {model_name}")
    else:
        print(f"  WARNING: Missing {filepath}")
 
# ==================================
# BUILD TABLE — one row per method-temperature combo
# ==================================
 
def build_model_table(model_name, data):
    """Build the temperature table for one model, matching project guide format."""
    
    temperatures = data["temperatures"]
    methods = data["methods"]
    results = data["results"]
    datasets = list(results.keys())
    miou = MIOU_VALUES.get(model_name, "—")
    
    rows = []
    
    for method in methods:
        
        # --- Row 1: baseline (t=1.0) — labeled just "MSP", "MaxEntropy", etc.
        row_baseline = {"Method": method, "mIoU": miou}
        for dataset in datasets:
            short = DATASET_SHORT.get(dataset, dataset)
            r = results[dataset][method]["1.0"]
            row_baseline[f"{short} AuPRC"] = round(r["auprc"] * 100, 2)
            row_baseline[f"{short} FPR95"] = round(r["fpr95"] * 100, 2)
        rows.append(row_baseline)
        
        # --- Rows for each display temperature (skip 1.0, it's already the baseline)
        for T in DISPLAY_TEMPS:
            T_str = str(T)
            if T_str not in results[datasets[0]][method]:
                continue
            
            row_t = {"Method": f"{method}(t={T})", "mIoU": ""}
            for dataset in datasets:
                short = DATASET_SHORT.get(dataset, dataset)
                r = results[dataset][method][T_str]
                row_t[f"{short} AuPRC"] = round(r["auprc"] * 100, 2)
                row_t[f"{short} FPR95"] = round(r["fpr95"] * 100, 2)
            rows.append(row_t)
        
        # --- Row: best temperature (highest average AuPRC across datasets)
        best_T = None
        best_avg_auprc = -1
        
        for T in temperatures:
            T_str = str(T)
            avg_auprc = sum(
                results[ds][method][T_str]["auprc"] for ds in datasets
            ) / len(datasets)
            if avg_auprc > best_avg_auprc:
                best_avg_auprc = avg_auprc
                best_T = T
        
        best_T_str = str(best_T)
        row_best = {"Method": f"{method}(best t={best_T})", "mIoU": ""}
        for dataset in datasets:
            short = DATASET_SHORT.get(dataset, dataset)
            r = results[dataset][method][best_T_str]
            row_best[f"{short} AuPRC"] = round(r["auprc"] * 100, 2)
            row_best[f"{short} FPR95"] = round(r["fpr95"] * 100, 2)
        rows.append(row_best)
    
    return pd.DataFrame(rows)
 
 
# ==================================
# GENERATE AND PRINT TABLES
# ==================================
 
save_dir = f"{RESULTS_ROOT}/tables"
os.makedirs(save_dir, exist_ok=True)
 
all_tables = {}
 
for model_name, data in all_temp_data.items():
    
    df = build_model_table(model_name, data)
    all_tables[model_name] = df
    
    # --- Print readable table ---
    print(f"\n{'='*120}")
    print(f"  TEMPERATURE SCALING — {model_name}")
    print(f"{'='*120}\n")
    print(df.to_string(index=False))
    
    # --- Save CSV ---
    safe_name = model_name.lower().replace("-", "_").replace(" ", "_")
    csv_path = f"{save_dir}/csv_tables/{safe_name}_temperature_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")
 
 
# ==================================
# GENERATE LATEX TABLE (for report)
# ==================================
 
def to_latex_table(df, model_name, caption=""):
    """Convert DataFrame to a LaTeX table string for the CVPR report."""
    
    # Get dataset columns
    dataset_cols = [c for c in df.columns if "AuPRC" in c]
    datasets_short = [c.replace(" AuPRC", "") for c in dataset_cols]
    n_datasets = len(datasets_short)
    
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{tab:temp_" + model_name.lower().replace("-","_").replace(" ","_") + "}")
    lines.append(r"\resizebox{\textwidth}{!}{")
    
    # Column spec: Method | mIoU | (AuPRC FPR95) per dataset
    col_spec = "l|c|" + "|".join(["cc"] * n_datasets)
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    
    # Header row 1: dataset names spanning 2 cols each
    header1 = r" & & "
    header1 += " & ".join([r"\multicolumn{2}{c|}{" + ds + "}" for ds in datasets_short])
    header1 += r" \\"
    lines.append(header1)
    
    # Header row 2: AuPRC / FPR95 repeated
    header2 = r"Method & mIoU"
    for _ in datasets_short:
        header2 += r" & AuPRC$\uparrow$ & FPR95$\downarrow$"
    header2 += r" \\"
    lines.append(header2)
    lines.append(r"\midrule")
    
    # Data rows
    for _, row in df.iterrows():
        method = row["Method"]
        miou = row["mIoU"] if row["mIoU"] != "" else ""
        
        cells = [method, str(miou)]
        for ds in datasets_short:
            auprc = row.get(f"{ds} AuPRC", "")
            fpr95 = row.get(f"{ds} FPR95", "")
            cells.append(f"{auprc}")
            cells.append(f"{fpr95}")
        
        line = " & ".join(str(c) for c in cells) + r" \\"
        
        # Add a small rule after the "best t" row to separate methods
        if "best t" in method:
            line += "\n" + r"\midrule"
        
        lines.append(line)
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table*}")
    
    return "\n".join(lines)
 
 
for model_name, df in all_tables.items():
    caption = f"Temperature scaling results for {model_name}."
    latex = to_latex_table(df, model_name, caption)
    
    safe_name = model_name.lower().replace("-", "_").replace(" ", "_")
    tex_path = f"{save_dir}/latex_tables/{safe_name}_temperature_table.tex"
    with open(tex_path, 'w') as f:
        f.write(latex)
    print(f"  LaTeX saved: {tex_path}")
 
print("\nDone!")
 