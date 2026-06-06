import json
import os
import pandas as pd
 
# ==================================
# CONFIGURATION
# ==================================
 
RESULTS_ROOT = "/content/drive/MyDrive/FAIMDL/results"
 
RESULT_FILES = {
    "ERFNet":            f"{RESULTS_ROOT}/step7/erfnet_anomaly_results.json",
    "EoMT-COCO":         f"{RESULTS_ROOT}/step8/eomt_coco_anomaly_results.json",
    "EoMT-Cityscapes":   f"{RESULTS_ROOT}/step8/eomt_cityscapes_anomaly_results.json",
    "EoMT-Fine-tuned":   f"{RESULTS_ROOT}/step8/eomt_finetuned_anomaly_results.json",
}
 
MIOU_VALUES = {
    "ERFNet":           72.50,
    "EoMT-COCO":        55.17,
    "EoMT-Cityscapes":  81.68,
    "EoMT-Fine-tuned":  78.85,
}
 
DATASET_SHORT = {
    "RoadAnomaly21":     "RA-21",
    "RoadObsticle21":    "RO-21",
    "FS_LostFound_full": "FS L\\&F",   # LaTeX-safe ampersand
    "fs_static":         "FS Static",
    "RoadAnomaly":       "RoadAnom",
}
 
DATASET_SHORT_CSV = {
    "RoadAnomaly21":     "RA-21",
    "RoadObsticle21":    "RO-21",
    "FS_LostFound_full": "FS L&F",
    "fs_static":         "FS Static",
    "RoadAnomaly":       "RoadAnom",
}
 
# ==================================
# LOAD
# ==================================
 
all_data = {}
for model_name, filepath in RESULT_FILES.items():
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            all_data[model_name] = json.load(f)
        print(f"  Loaded: {model_name}")
    else:
        print(f"  WARNING: Missing {filepath}")
 
# ==================================
# BUILD TABLE
# ==================================
 
rows = []
for model_name, data in all_data.items():
    miou = MIOU_VALUES.get(model_name, "—")
    methods = data["methods"]
    datasets = data["datasets"]
    
    for i, method in enumerate(methods):
        row = {
            "Model": model_name if i == 0 else "",   # Only print model name on first method row
            "mIoU": miou if i == 0 else "",
            "Method": method,
        }
        for dataset in datasets:
            short = DATASET_SHORT_CSV.get(dataset, dataset)
            auprc = data["per_dataset"][dataset][method]["auprc"] * 100
            fpr95 = data["per_dataset"][dataset][method]["fpr95"] * 100
            row[f"{short} AuPRC"] = round(auprc, 2)
            row[f"{short} FPR95"] = round(fpr95, 2)
        rows.append(row)
 
df = pd.DataFrame(rows)
 
# ==================================
# PRINT
# ==================================
 
print(f"\n{'='*140}")
print("  MAIN ANOMALY SEGMENTATION RESULTS")
print(f"{'='*140}\n")
print(df.to_string(index=False))
 
# ==================================
# CSV
# ==================================
 
save_dir = f"{RESULTS_ROOT}/tables"
os.makedirs(save_dir, exist_ok=True)
csv_path = f"{save_dir}/csv_tables/result_table.csv"
df.to_csv(csv_path, index=False)
print(f"\n  CSV saved: {csv_path}")
 
# ==================================
# LATEX
# ==================================
 
datasets_ordered = list(DATASET_SHORT.values())
 
lines = []
lines.append(r"\begin{table*}[t]")
lines.append(r"\centering")
lines.append(r"\caption{Anomaly segmentation results. AuPRC (\%) $\uparrow$ and FPR@95 (\%) $\downarrow$ on five benchmarks.}")
lines.append(r"\label{tab:anomaly_results}")
lines.append(r"\resizebox{\textwidth}{!}{")
 
col_spec = "l|c|l|" + "|".join(["cc"] * len(datasets_ordered))
lines.append(r"\begin{tabular}{" + col_spec + "}")
lines.append(r"\toprule")
 
# Header row 1
h1 = r" & & "
h1 += " & ".join([r"\multicolumn{2}{c" + ("|" if i < len(datasets_ordered)-1 else "") + "}{" + ds + "}" 
                   for i, ds in enumerate(datasets_ordered)])
h1 += r" \\"
lines.append(h1)
 
# Header row 2
h2 = r"Model & mIoU & Method"
for _ in datasets_ordered:
    h2 += r" & AuPRC$\uparrow$ & FPR95$\downarrow$"
h2 += r" \\"
lines.append(h2)
lines.append(r"\midrule")
 
# Data rows
prev_model = None
for _, r in df.iterrows():
    model = r["Model"]
    
    # Add horizontal rule between different models
    if model != "" and prev_model is not None and model != prev_model:
        lines.append(r"\midrule")
    if model != "":
        prev_model = model
    
    miou = r["mIoU"] if r["mIoU"] != "" else ""
    method = r["Method"]
    
    cells = [str(model), str(miou), method]
    for ds_csv, ds_tex in zip(DATASET_SHORT_CSV.values(), DATASET_SHORT.values()):
        auprc = r.get(f"{ds_csv} AuPRC", "")
        fpr95 = r.get(f"{ds_csv} FPR95", "")
        cells.append(f"{auprc}")
        cells.append(f"{fpr95}")
    
    lines.append(" & ".join(str(c) for c in cells) + r" \\")
 
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}}")
lines.append(r"\end{table*}")
 
latex = "\n".join(lines)
tex_path = f"{save_dir}/latex_tables/result_table.tex"
with open(tex_path, 'w') as f:
    f.write(latex)
print(f"  LaTeX saved: {tex_path}")
 
print("\nDone!")