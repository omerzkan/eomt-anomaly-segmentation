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

Note: `erfnet_pretrained.pth` is committed directly in the repo — no need to download it separately.

**Drive link:** https://drive.google.com/drive/folders/1q2vHUzora2nP52fP50zmoQAykWuwoGav

---

## 4. Download anomaly datasets

Download `Anomaly_Validation_Datasets.zip` from the same Drive link.

**Drive link:** https://drive.google.com/drive/folders/1q2vHUzora2nP52fP50zmoQAykWuwoGav

The zip extracts into a `Validation_Dataset/` subfolder. Expected structure after unzipping into `data/`:

```
data/
  Validation_Dataset/
    RoadAnomaly21/
    RoadObsticle21/      ← note: typo in original zip, not RoadObstacle21
    RoadAnomaly/
    fs_static/
    FS_LostFound_full/   ← note: not LostAndFound
```

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

### 7.1 First-time setup (every new Colab session)

Colab resets completely on every session — repeat steps 3–5 each time.

**Step 1** — Enable GPU:
> Runtime → Change runtime type → T4 GPU

**Step 2** — Mount Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

**Step 3** — Clone the repo:
```python
!git clone https://github.com/omerzkan/eomt-anomaly-segmentation.git
%cd /content/eomt-anomaly-segmentation
```

> ⚠️ Always use `%cd` (not `!cd`) to change directory. `%cd` persists for the session; `!cd` only applies to that one command.

**Step 4** — Install dependencies:
```python
!pip install -r eval/requirements.txt
!pip install -r eomt/requirements.txt
```

> When pip shows a "Restart session" dialog — **click Restart session**. This is normal. You do NOT need to re-run pip after restarting.

**Step 5** — Copy weights from Drive:
```python
!cp /content/drive/MyDrive/CourseProjectAnomaly/eomt_cityscapes.bin /content/eomt-anomaly-segmentation/trained_models/
!cp /content/drive/MyDrive/CourseProjectAnomaly/eomt_coco.bin /content/eomt-anomaly-segmentation/trained_models/
```

**Step 6** — Unzip datasets (only needed once per session):
```python
!unzip /content/drive/MyDrive/CourseProjectAnomaly/Anomaly_Validation_Datasets.zip \
  -d /content/eomt-anomaly-segmentation/data/
```

Verify the structure:
```python
import os
print(os.listdir("/content/eomt-anomaly-segmentation/data/Validation_Dataset/"))
# Expected: ['RoadAnomaly21', 'RoadObsticle21', 'RoadAnomaly', 'fs_static', 'FS_LostFound_full', ...]
```

---

### 7.2 Running the ERFNet baseline (evalAnomaly.py)

Always run from inside the `eval/` folder using `%cd`:

```python
%cd /content/eomt-anomaly-segmentation/eval

# RoadAnomaly21
!python evalAnomaly.py --input "/content/eomt-anomaly-segmentation/data/Validation_Dataset/RoadAnomaly21/images/*"

# RoadObsticle21
!python evalAnomaly.py --input "/content/eomt-anomaly-segmentation/data/Validation_Dataset/RoadObsticle21/images/*"

# RoadAnomaly
!python evalAnomaly.py --input "/content/eomt-anomaly-segmentation/data/Validation_Dataset/RoadAnomaly/images/*"

# fs_static
!python evalAnomaly.py --input "/content/eomt-anomaly-segmentation/data/Validation_Dataset/fs_static/images/*"

# FS_LostFound_full
!python evalAnomaly.py --input "/content/eomt-anomaly-segmentation/data/Validation_Dataset/FS_LostFound_full/images/*"
```

Results are saved to `eval/results.txt`.

---

## 8. ERFNet Baseline Results (MSP — Maximum Softmax Probability)

| Dataset         | AUPRC  | FPR@TPR95 |
|-----------------|--------|-----------|
| RoadAnomaly21   | 38.32% | 59.34%    |

These are the Step 6 baseline numbers to beat in subsequent steps.

---

## 9. Branch strategy

| Branch | Purpose |
|---|---|
| `main` | Stable code only — never push broken code here |
| `dev` | Integration branch — merge features here first |
| `feature/erfnet-baselines` | MSP, MaxEntropy on ERFNet |
| `feature/eomt-eval` | Adapting eval to EoMT output format |
| `feature/finetune` | Fine-tuning configs and scripts |
| `feature/extension` | Extension experiment (agreed with TA at Session 3) |

---

## 10. Task ownership

| Task | Owner |
|---|---|
| ERFNet baselines (MaxEntropy, MaxLogit) | [NAME] |
| EoMT eval adaptation (evalAnomaly_eomt.py) | [NAME] |
| Fine-tuning pipeline + configs | [NAME] |
| Class mapping + eval pipeline (Step 4) | [NAME] |

---

## 11. Environment

- Python 3.10+ recommended
- CUDA required for GPU training — use Google Colab (free T4 GPU)
- W&B account required for training runs

---

## 12. Known Issues & Fixes

| Issue | Cause | Fix |
|---|---|---|
| `cd: eval: No such file or directory` | Using `!cd` instead of `%cd`, or session reset cleared working dir | Always use `%cd /content/eomt-anomaly-segmentation` after session restart |
| `FileNotFoundError: erfnet_pretrained.pth` | Running script from wrong directory | Run from inside `eval/` with `%cd /content/eomt-anomaly-segmentation/eval` |
| `IsADirectoryError: Is a directory: '/'` | No `--input` argument passed, script uses hardcoded author path | Always pass `--input` with the correct dataset path |
| `RuntimeError: expected input to have 3 channels, but got 1024` | Bug in original `evalAnomaly.py`: spurious `.permute(0,3,1,2)` line | Fixed in repo — `git pull` to get the fix |
| `data/` only contains `.gitkeep` | Datasets not unzipped yet, or unzipped to wrong location | Run the unzip command with the full absolute path |
| Zip extracts to `Validation_Dataset/` subfolder | Zip internal structure differs from what SETUP originally said | Use paths under `data/Validation_Dataset/` |
