# Setup Guide — EoMT Anomaly Segmentation

PoliTo FAIMDL Course Project  
TAs: Alessandro Marinai, Davide Sferrazza, Stephany Ortuno

---

## Workflow Philosophy

**VS Code = write code** | **Google Colab = run code**

Your local machine is only for reading, writing, and committing code.
All model inference, training, and evaluation requires a GPU — use Google Colab for everything that actually runs. You cannot run `evalAnomaly.py` locally — it requires CUDA/GPU.

---

## 1. Clone the repo

```bash
git clone https://github.com/omerzkan/eomt-anomaly-segmentation.git
cd eomt-anomaly-segmentation
```

---

## 2. Install dependencies (local — for code editing only)

For the evaluation pipeline (ERFNet & EoMT baselines):
```bash
pip install -r eval/requirements.txt
```

For EoMT training and fine-tuning:
```bash
pip install -r eomt/requirements.txt
```

---

## 3. Download model weights

Download from the shared Google Drive and place in `trained_models/`:
- `eomt_cityscapes.bin` ← EoMT trained on Cityscapes (364 MB)
- `eomt_coco.bin`       ← EoMT trained on COCO (357 MB)

Note: `erfnet_pretrained.pth` and `erfnet_encoder_pretrained.pth.tar`
are already in the repo (committed by the professor).

**Drive link:** https://drive.google.com/drive/folders/1q2vHUzora2nP52fP50zmoQAykWuwoGav

---

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

Note: `data/` and `trained_models/*.bin` are gitignored — they live
only on your local machine and on Colab. Never commit them.

---

## 5. Set up Weights & Biases

Create a free account at https://wandb.ai, then:
```bash
wandb login
```
Paste your API key from https://wandb.ai/authorize when prompted.

Why: the EoMT training config requires wandb. It also lets all 4
teammates monitor training runs in real time on a shared dashboard.

---

## 6. Verify your setup (local)

Run a quick import check to confirm packages installed correctly:
```bash
python -c "import torch; print(torch.__version__)"
python -c "import torchvision; print('torchvision ok')"
python -c "import wandb; print('wandb ok')"
```

---

## 7. Running on Google Colab

1. Go to https://colab.research.google.com and enable GPU:
   Runtime → Change runtime type → T4 GPU

2. Mount your Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

3. Clone the repo inside Colab:
```bash
!git clone https://github.com/omerzkan/eomt-anomaly-segmentation.git
%cd eomt-anomaly-segmentation
```

4. Install dependencies:
```bash
!pip install -r eval/requirements.txt
!pip install -r eomt/requirements.txt
```

5. Copy weights and datasets from your Drive into the repo:
```python
!cp /content/drive/MyDrive/CourseProjectAnomaly/eomt_cityscapes.bin trained_models/
!cp /content/drive/MyDrive/CourseProjectAnomaly/eomt_coco.bin trained_models/
!unzip /content/drive/MyDrive/CourseProjectAnomaly/Anomaly_Validation_Datasets.zip -d data/
```

6. Run the ERFNet baseline:
```bash
!cd eval && python evalAnomaly.py
```

---

## 8. Branch strategy

| Branch | Purpose |
|---|---|
| `main` | Stable code only — never push broken code here |
| `dev` | Integration branch — merge features here first |
| `feature/erfnet-baselines` | MSP, MaxEntropy on ERFNet |
| `feature/eomt-eval` | Adapting eval to EoMT output format |
| `feature/finetune` | Fine-tuning configs and scripts |
| `feature/extension` | Extension experiment (agreed with TA at Session 3) |

---

## 9. Task ownership

| Task | Owner |
|---|---|
| ERFNet baselines (MaxEntropy, MaxLogit) | [NAME] |
| EoMT eval adaptation (evalAnomaly_eomt.py) | [NAME] |
| Fine-tuning pipeline + configs | [NAME] |
| Class mapping + eval pipeline (Step 4) | [NAME] |

---

## 10. Environment

- Python 3.10+ recommended
- CUDA required for GPU training — use Google Colab (free T4 GPU)
- W&B account required for training runs


