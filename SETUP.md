# Setup Guide — EoMT Anomaly Segmentation

PoliTo FAIMDL Course Project  
TAs: Alessandro Marinai, Davide Sferrazza, Stephany Ortuno

---

## 1. Clone the repo

```bash
git clone https://github.com/omerzkan/eomt-anomaly-segmentation.git
cd eomt-anomaly-segmentation
```

## 2. Install dependencies

For the evaluation pipeline (ERFNet & EoMT baselines):
```bash
pip install -r eval/requirements.txt
```

For EoMT training and fine-tuning:
```bash
pip install -r eomt/requirements.txt
```

## 3. Download model weights

Download from the shared Google Drive and place in `trained_models/`:
- `eomt_cityscapes.bin`  ← EoMT trained on Cityscapes (364 MB)
- `eomt_coco.bin`        ← EoMT trained on COCO (357 MB)

Note: `erfnet_pretrained.pth` and `erfnet_encoder_pretrained.pth.tar` 
are already in the repo (committed by the professor).

**Drive link:** https://drive.google.com/drive/folders/1q2vHUzora2nP52fP50zmoQAykWuwoGav

## 4. Download anomaly datasets

Download `Anomaly_Validation_Datasets.zip` from the same Drive link,
unzip it, and place the contents under `data/`:

**Drive link:** https://drive.google.com/drive/folders/1q2vHUzora2nP52fP50zmoQAykWuwoGav

Expected structure after unzipping:
data/
  RoadAnomaly21/
  RoadObstacle21/
  fs_static/
  LostAndFound/
  RoadAnomaly/

## 5. Set up Weights & Biases

```bash
pip install wandb
wandb login
```
Create a free account at https://wandb.ai if you don't have one.

## 6. Verify your setup

```bash
# Quick sanity check — ERFNet baseline on one image
cd eval
python evalAnomaly.py
```

---

## Branch strategy

| Branch | Purpose |
|---|---|
| `main` | Stable, working code only — never push broken code here |
| `dev` | Integration branch — merge features here first |
| `feature/erfnet-baselines` | MSP, MaxEntropy on ERFNet |
| `feature/eomt-eval` | Adapting eval to EoMT output |
| `feature/finetune` | Fine-tuning configs and scripts |
| `feature/extension` | Extension experiment (agreed w/ TA at Session 3) |

## Task ownership

| Task | Owner |
|---|---|
| ERFNet baselines (MaxEntropy, MaxLogit in evalAnomaly.py) | [NAME] |
| EoMT eval adaptation (evalAnomaly_eomt.py) | [NAME] |
| Fine-tuning pipeline + configs | [NAME] |
| Class mapping + eval pipeline (Step 4) | [NAME] |

## Python version

Python 3.10+ recommended. CUDA 11.8+ required for GPU training.
