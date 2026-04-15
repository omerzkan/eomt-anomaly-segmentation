cat > SETUP.md << 'EOF'
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
- `erfnet_pretrained.pth`
- `erfnet_encoder_pretrained.pth.tar`
- `eomt_cityscapes_semantic.pth` ← EoMT trained on Cityscapes
- `eomt_coco_panoptic.pth`       ← EoMT trained on COCO

**Drive link:** https://drive.google.com/drive/folders/1q2vHUzora2nP52fP50zmoQAykWuwoGav

## 4. Download datasets

Place all datasets under `data/` following this exact structure:
data/
cityscapes/
gtFine/
leftImg8bit/
RoadAnomaly21/
RoadObstacle21/
fishyscapes_static/
LostAndFound/
RoadAnomaly/

**Dataset download links:**
- Cityscapes val set: https://www.cityscapes-dataset.com/
- Anomaly datasets (`Anomaly_Validation_Datasets.zip`): https://drive.google.com/drive/folders/1q2vHUzora2nP52fP50zmoQAykWuwoGav

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
