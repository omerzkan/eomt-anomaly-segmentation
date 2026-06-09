# EoMT for Anomaly Segmentation in Road Scenes

Mask-based vs. pixel-based architectures for semantic segmentation and out-of-distribution detection in autonomous driving.

This repository contains the implementation, evaluation, and analysis for the FAIMDL course project at Politecnico di Torino (VANDAL lab): **Comprehensive Road Scene Understanding for Autonomous Driving**. We evaluate the Encoder-only Mask Transformer (EoMT) with a DINOv2 backbone for semantic segmentation on Cityscapes and adapt it for post-hoc anomaly segmentation across five road-scene benchmarks, using ERFNet as a pixel-based baseline.

The project is structured around the Colab notebooks in `colab_notebooks/` — one per step — which are the intended way to run everything end-to-end. All data, checkpoints, and results live on Google Drive.

---

## Research Question

Do mask-based architectures (EoMT) outperform pixel-based ones (ERFNet) at detecting unknown / out-of-distribution objects in road scenes, and what is the trade-off between closed-set semantic segmentation accuracy and open-set anomaly detection?

## Headline Results

| Model            | Cityscapes mIoU | RA-21 AuPRC (best)    | RO-21 AuPRC (best)    | RoadAnomaly AuPRC (best) |
|------------------|:---------------:|:---------------------:|:---------------------:|:------------------------:|
| ERFNet           |     72.50       |   38.31 (MaxLogit)    |    4.63 (MaxLogit)    |     15.58 (MaxLogit)     |
| EoMT-COCO        |     55.17       |   41.07 (MaxEntropy)  |   39.36 (RbA)         |     23.40 (MaxEntropy)   |
| EoMT-Cityscapes  |   **81.68**     | **77.88** (MaxEntropy)| **90.11** (MSP)       |   **78.14** (MaxEntropy) |
| EoMT-Fine-tuned  |     78.85       |   57.15 (MSP)         |   73.61 (MSP)         |     56.22 (MaxEntropy)   |

EoMT-Cityscapes is dominant on anomaly detection (e.g. **74.36 vs 38.31** AuPRC on RoadAnomaly21 against ERFNet with MSP). Fine-tuning EoMT-COCO on Cityscapes recovers most of the closed-set mIoU gap (55.17 → 78.85) but reduces anomaly-detection performance, exposing a trade-off between in-distribution accuracy and OOD sensitivity. See `results/tables/` for the full table.

---

## Repository Structure

```
.
├── colab_notebooks/       # One run notebook per step — the main entry points
│   ├── step4_colab_notebook.ipynb
│   ├── step5_colab_notebook.ipynb
│   ├── step7_colab_notebook.ipynb
│   ├── step8_colab_notebook.ipynb
│   └── tables_colab_notebook.ipynb
├── eomt/                  # EoMT codebase (from the official tue-mps/eomt repo,
│                          # included in the course starter project)
├── eval/                  # ERFNet evaluation tools (from starter repo)
├── trained_models/        # Pretrained ERFNet weights (.pth)
├── utils/
│   ├── eomt_utils.py      # Model building, semantic inference, IoU helpers
│   ├── anomaly_utils.py   # MSP / MaxLogit / MaxEntropy / RbA, metrics, inference
│   ├── build_table.py     # Build the main results table (CSV + LaTeX)
│   └── build_temperature_table.py
├── step4/                 # EoMT-COCO vs EoMT-Cityscapes on Cityscapes val
├── step5/                 # Fine-tune EoMT-COCO on Cityscapes + evaluate
├── step7/                 # ERFNet anomaly baselines (MSP / MaxLogit / MaxEntropy)
├── step8/                 # EoMT anomaly segmentation (adds RbA) + temperature scaling
├── results/               # Per-step JSON results, tables (CSV/LaTeX), figures
└── requirements.txt
```

---

## Setup (Google Colab + Google Drive)

The project is designed to run on Google Colab with Google Drive as persistent storage. Tested on Colab Pro with an NVIDIA L4 GPU.

### 1. Prepare your Google Drive

Create the following structure under `MyDrive/FAIMDL/`:

```
MyDrive/FAIMDL/
├── data/
│   └── Anomaly_Validation_Datasets.zip   # provided by the course
├── checkpoints/
│   ├── eomt_coco.bin                          # EoMT-Base 640, COCO panoptic
│   ├── eomt_cityscapes.bin                    # EoMT-Base 640, Cityscapes semantic
│   └── coco_eomt_finetuned_on_cityscapes.bin  # produced by Step 5
└── results/
    ├── step4/
    ├── step5/
    ├── step7/
    ├── step8/
    ├── tables/
    └── figures/
```

For step 4 (Cityscapes semantic eval) you also need Cityscapes itself under `MyDrive/FAIMDL/data/`.

### 2. Open the relevant notebook on Colab

Each notebook in `colab_notebooks/` is self-contained and handles:

- mounting Google Drive
- cloning this repository
- installing dependencies from `eval/requirements.txt` and `eomt/requirements.txt`
- unzipping the anomaly validation datasets to `/content/data/Validation_Dataset/` (idempotent)
- running the corresponding `stepN/` script
- writing JSON results, tables, and figures back to Drive

Key dependencies (pinned in `eomt/requirements.txt`): `torch==2.7.0`, `torchvision==0.22.0`, `lightning==2.5.1`, `transformers==4.56.1`, `timm==1.0.15`, `torchmetrics==1.7.1`, `pycocotools==2.0.8`, plus `ood-metrics` and `scikit-learn` from `eval/requirements.txt`.

### 3. Run the notebooks in order

| Notebook                          | What it does                                                      |
|-----------------------------------|-------------------------------------------------------------------|
| `step4_colab_notebook.ipynb`      | EoMT-COCO vs EoMT-Cityscapes on Cityscapes val (mIoU + per-class) |
| `step5_colab_notebook.ipynb`      | Fine-tune EoMT-COCO on Cityscapes, then re-evaluate               |
| `step7_colab_notebook.ipynb`      | ERFNet anomaly baselines + temperature scaling                    |
| `step8_colab_notebook.ipynb`      | EoMT anomaly segmentation (all four scores) + temperature scaling |
| `tables_colab_notebook.ipynb`     | Aggregate JSON results into the final CSV / LaTeX tables          |

> All paths in the scripts assume `/content/drive/MyDrive/FAIMDL/...`. Adjust the constants at the top of each `stepN/*.py` file if your Drive layout differs.

---

## Method Summary

### Semantic Segmentation (Steps 4–5)

Three EoMT-Base variants with a DINOv2 backbone and 640×640 input are evaluated on Cityscapes:

1. **EoMT-COCO** — panoptic checkpoint, used zero-shot. The COCO→Cityscapes label-space mismatch produces 0% IoU on `pole` and `rider`, which is the central motivation for fine-tuning.
2. **EoMT-Cityscapes** — semantic checkpoint, the upper bound for in-distribution accuracy.
3. **EoMT-Fine-tuned** — EoMT-COCO fine-tuned on Cityscapes for 20 epochs (L4 GPU, AdamW, lr=1e-4, weight decay=0.05, 200 queries, backbone frozen, only mask head and query embeddings trained).

### Anomaly Segmentation (Steps 7–8)

Four post-hoc scoring functions are applied on top of frozen segmentation models:

- **MSP** — Maximum Softmax Probability (`1 − max softmax`).
- **MaxLogit** — `−max(logits)`; rank-equivalent to `1.0 − max(logits)`.
- **MaxEntropy** — Shannon entropy of the softmax distribution.
- **RbA** (mask architectures only) — `−Σ tanh(class_logits)` across class axis, following the official RbA implementation. Crucially, RbA does **not** apply softmax to mask-architecture logits, since the independence assumption it requires holds at the query level rather than across mutually exclusive classes.

Each method is also evaluated with temperature scaling over `T ∈ {0.5, 0.75, 1.0, 1.1, 1.5, 2.0}`. The main finding is that AuPRC is largely insensitive to T, while FPR95 can swing substantially (especially for RbA), making post-hoc temperature calibration unreliable as a tuning knob on these benchmarks.

### Datasets

- **Cityscapes** for closed-set semantic segmentation evaluation.
- **Anomaly Validation Datasets** (`Anomaly_Validation_Datasets.zip`, provided by the course): RoadAnomaly21, RoadObstacle21, Fishyscapes Lost&Found, Fishyscapes Static, RoadAnomaly.

### Metrics

- **Closed-set:** mean Intersection-over-Union (mIoU), per-class IoU on Cityscapes (19 classes, ignore_index=255).
- **Anomaly:** AuPRC (↑) and FPR@95TPR (↓) on the five SMIYC / Fishyscapes benchmarks.

---

## Key Findings

1. **Mask-based wins decisively on anomaly detection.** EoMT-Cityscapes more than doubles ERFNet's AuPRC on RoadAnomaly21 (74.36 vs 38.31) and RoadAnomaly (75.69 vs 12.42), and reaches 90.11 on RoadObstacle21 vs ERFNet's 4.63. Object-query classifiers separate inliers from outliers far better than per-pixel softmax.
2. **Fine-tuning closes the closed-set gap but hurts OOD.** EoMT-COCO → fine-tuned improves Cityscapes mIoU from 55.17 to 78.85 (nearly closing the gap with EoMT-Cityscapes at 81.68), but MSP AuPRC on RoadAnomaly21 drops from 74.36 to 57.15. Short fine-tuning over-commits the model to inlier classes and erodes the over-confidence signal that anomaly scoring relies on.
3. **The right scoring method depends on the architecture.** MaxLogit is the strongest score on ERFNet (38.31 vs 29.09 MSP on RA-21), but on EoMT-Cityscapes MSP / MaxLogit / MaxEntropy are within a few points of each other. RbA wins on Fishyscapes Static / L&F but at the cost of very high FPR95.
4. **Temperature scaling is not a reliable lever.** AuPRC stays in narrow bands across T, while FPR95 swings widely — gains are not consistent across (model, method, dataset) combinations.

---

## Limitations

- Training was constrained to 20 epochs on a single L4 GPU; the backbone was frozen throughout.
- Evaluation uses the validation splits of the SMIYC and Fishyscapes benchmarks. Full benchmark submission was out of scope.
- Only post-hoc methods are explored; no outlier exposure or auxiliary training.

---

## References

The project builds on the following works:

- Kerssies et al., *Your ViT is Secretly an Image Segmentation Model* (EoMT, CVPR 2025).
- Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*.
- Romera et al., *ERFNet: Efficient Residual Factorized ConvNet for Real-Time Semantic Segmentation*.
- Cheng et al., *MaskFormer / Mask2Former*.
- Nayal et al., *RbA: Segmenting Unknown Regions Rejected by All*.
- Hendrycks et al., *MaxLogit / Scaling Out-of-Distribution Detection*.
- Chan et al., *Maximum Entropy for Anomaly Segmentation*.
- Blum et al., *Fishyscapes*. Chan et al., *SegmentMeIfYouCan*.

See the report for the full bibliography.

---

## Acknowledgments

This project was developed for the *Comprehensive Road Scene Understanding for Autonomous Driving* project of the FAIMDL course. The starter repository provided by the course includes the ERFNet baseline and a copy of the official [EoMT codebase](https://github.com/tue-mps/eomt); we built our experiments on top of that base. The anomaly validation datasets were also provided by the course staff.