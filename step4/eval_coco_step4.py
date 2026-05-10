
"""
Step4 Evaluation of COCO trained EoMT on Cityscapes validation set.
Using coco_to_cityscapes.py to convert COCO annotations to Cityscapes format, and then using the converted annotations for evaluation.

Produces:
- mIoU on Cityscapes 19 classes
- Per-class IoU on Cityscapes 19 classes
"""

#==================================
# IMPORTS AND SETUP
#==================================

# 'importlib' is a built-in Python module that lets you import other modules
# *dynamically at runtime* — meaning you can import a module whose name you
# only know as a string (e.g. "torch.nn"), instead of writing 'import torch.nn'
# at the top. We use this inside _build() to turn YAML class paths into real classes.
import importlib

# 'os'  → interact with the operating system: file paths, env variables, cwd
# 'sys' → control the Python interpreter itself: module search path, argv, etc.
# 'json'→ read/write JSON files (not used heavily here but common companion)
import os, sys, json

# Disable Weights & Biases (wandb) experiment tracking.
# wandb is a popular ML logging library. Setting WANDB_MODE="disabled" tells it
# to silently do nothing — we don't want training logs during pure evaluation.
# os.environ sets an environment variable that the wandb library reads on import.
os.environ["WANDB_MODE"] = "disabled"

# Absolute path to the cloned EoMT repository inside the Colab runtime.
REPO = "/content/cloned_repo_feature_omer/eomt"

# sys.path is the list of directories Python searches when you write 'import something'.
# insert(0, ...) puts our paths at the FRONT of that list (highest priority),
# so Python finds the EoMT source files before any similarly-named installed packages.
sys.path.insert(0, REPO)
sys.path.insert(0, "/content/cloned_repo_feature_omer/step4")

# 'yaml'  → parse YAML config files into Python dicts (PyYAML library)
# 'torch' → PyTorch: the deep-learning framework used to build and run the model
# 'numpy as np' → fast numerical arrays; standard alias 'np' is universal convention
import yaml, torch, numpy as np

# torch.nn.functional contains stateless neural-network operations (no learned params):
# e.g. F.interpolate to resize tensors, F.softmax, F.cross_entropy, etc.
from torch.nn import functional as F

# autocast enables Automatic Mixed Precision (AMP): PyTorch automatically uses
# float16 where safe and float32 where needed. This speeds up GPU inference and
# reduces memory without much accuracy loss.
from torch.amp.autocast_mode import autocast

# tqdm wraps any iterable and draws a live progress bar in the terminal/notebook.
# e.g.  for batch in tqdm(dataloader):  → shows % done, elapsed time, ETA
from tqdm import tqdm

# Lightning's seed_everything sets random seeds for Python, NumPy, PyTorch (CPU+GPU)
# all at once.  Fixing the seed makes runs reproducible: same weights init, same
# random augmentations, same results every time you re-run.
from lightning import seed_everything
seed_everything(0, verbose=False)   # seed=0, suppress the "Seed set to 0" print

# Change the current working directory to the repo root.
# Many EoMT scripts use relative paths (e.g. "configs/..."), so we must be
# *inside* the repo for those relative paths to resolve correctly.
os.chdir(REPO)


#==================================
# CONFIGURATION
#==================================

# Path to the YAML config that describes EoMT's architecture and training settings.
# This particular config is for a ViT-Base backbone, 640px input, trained on COCO panoptic.
CONFIG_PATH = f"{REPO}/configs/dinov2/coco/panoptic/eomt_base_640_2x.yaml"

# Path to the saved model weights (.bin file = a PyTorch checkpoint stored on Google Drive).
CHECKPOINT_PATH = "/content/drive/MyDrive/FAIMDL/checkpoints/eomt_coco.bin"

# Root folder containing the dataset (images + annotations).
DATA_PATH = "/content/drive/MyDrive/FAIMDL/data"

# Which hardware device to run inference on.
# "cuda" = the GPU (much faster than CPU for large models).
DEVICE = "cuda"

# Cityscapes has 19 semantic classes (road, sky, person, car, …).
# We evaluate against these 19 classes even though the model was trained on COCO.
N_CITYSCAPES_CLASSES = 19

# The standard "ignore" label value in Cityscapes annotations.
# Pixels labelled 255 should be skipped during IoU calculation — they are
# either void regions or evaluation-ignored classes (e.g. license plates).
IGNORE = 255


#==================================
# LOAD CONFIG FROM YAML
#==================================

# Open the YAML file and parse it into a plain Python dict.
# yaml.safe_load is preferred over yaml.load because safe_load forbids
# arbitrary Python object construction (a security best-practice).
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)
    # After this line, 'config' is a nested dict, e.g.:
    #   config['model']['class_path'] = 'eomt.models.EoMT'
    #   config['model']['init_args']['backbone'] = {'class_path': 'eomt.backbones.DINOv2', ...}

# Print every top-level key-value pair so we can inspect the config structure.
for key, value in config.items():
    print(f"{key}: {value}")


#==================================
# RECURSIVE MODEL BUILDER
#==================================

# The YAML config describes the model as a *blueprint*: nested dicts with
# 'class_path' (dotted Python import path) and 'init_args' (constructor kwargs).
# _build() walks that blueprint recursively and instantiates every real object.
def _build(d):
    """
    Recursively converts a config dictionary (loaded from YAML) into real Python objects.

    The YAML config describes the model architecture using 'class_path' and 'init_args' keys,
    like a blueprint. This function reads that blueprint and actually constructs every object.

    Example YAML entry that triggers Case 1:
        class_path: torch.nn.Linear
        init_args:
            in_features: 256
            out_features: 128

    This function turns that into: torch.nn.Linear(in_features=256, out_features=128)
    """

    # ── CASE 1: dict that describes a Python class (has 'class_path') ────────────────────

    # Check if d is a dict AND has the special key 'class_path'.
    # 'class_path' signals that this dict represents a real Python class to be instantiated.
    # Example: d = {'class_path': 'torch.nn.Linear', 'init_args': {'in_features': 256}}
    if isinstance(d, dict) and 'class_path' in d:

        # Extract the full dotted path to the class, e.g. 'torch.nn.Linear'
        cls_path = d['class_path']

        # Split the path into module and class name at the LAST dot.
        # 'torch.nn.Linear'  →  module_name='torch.nn', class_name='Linear'
        # rsplit('.', 1) means "split from the right, at most 1 time"
        module_name, class_name = cls_path.rsplit('.', 1)

        # Dynamically import the module (e.g. 'torch.nn') at runtime,
        # then grab the class attribute (e.g. 'Linear') from that module.
        # Result: Cls = torch.nn.Linear  (the class itself, not an instance yet)
        Cls = getattr(importlib.import_module(module_name), class_name)

        # Get the 'init_args' sub-dict that holds the constructor arguments.
        # If 'init_args' is missing (no arguments needed), default to an empty dict {}.
        # Example: {'in_features': 256, 'out_features': 128}
        init_args = d.get('init_args', {})

        # Recursively call _build on each argument value.
        # This handles nested classes: an argument itself might be another class_path dict.
        # Result is a flat dict of ready-to-use Python values, e.g. {'in_features': 256, ...}
        kwargs = {k: _build(v) for k, v in init_args.items()}

        # Instantiate the class with the built arguments and return the live object.
        # Equivalent to: torch.nn.Linear(in_features=256, out_features=128)
        return Cls(**kwargs)

    # ── CASE 2: plain dict WITHOUT 'class_path' (just nested config data) ────────────────

    # If d is a dict but has no 'class_path', it is just a container of config values
    # (e.g. a group of settings). Recurse into each value so any nested class_path dicts
    # deeper inside still get built.
    if isinstance(d, dict):
        return {k: _build(v) for k, v in d.items()}

    # ── CASE 3: list → recurse into every element ─────────────────────────────────────────

    # If d is a list, each element might itself be a class_path dict or another list,
    # so we recursively build every item and return a new list of the results.
    if isinstance(d, list):
        return [_build(item) for item in d]

    # ── CASE 4: leaf value (int, float, str, bool, None) → return as-is ──────────────────

    # If d is a plain scalar value (not a dict or list), there is nothing to build.
    # Just return it directly so it can be used as a constructor argument upstream.
    return d


#==================================
# BUILD THE MODEL
#==================================

# Disable masked attention for faster inference (paper recommends this at eval time).
# We mutate the config dict before building so the change propagates into the model.
config['model']['init_args']['network']['init_args']['masked_attn_enabled'] = False

# Recursively build the actual PyTorch model object from the config blueprint.
model = _build(config['model'])
print(f"Model built: {type(model).__name__}")


#==================================
# LOAD MODEL WEIGHTS FROM CHECKPOINT
#==================================

# Notice: ZERO indentation now — these are at module level, not inside _build.
checkpoints = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
state_dict = checkpoints.get("state_dict", checkpoints)

missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"Loaded checkpoint. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

model.eval().to(DEVICE)

#==================================
# LOAD CITYSCAPES VALIDATION DATA
#==================================

from datasets.cityscapes_semantic import CityscapesSemantic

data = CityscapesSemantic(
    path=DATA_PATH,
    batch_size=1,
    num_workers=2,
    img_size=(640, 640)
)

data.setup("validate")
val_loader = data.val_dataloader()

print(f"Val set size: {len(val_loader.dataset)} images")
# prepare the dataset (e.g. build file lists, load annotations, etc.)

#================================
# COCO --> CITYSCAPES LOOKUP TENSOR WITH MAPPING FROM COCO CLASS IDS TO CITYSCAPES CLASS IDS
#================================

from coco_to_cityscapes import coco_to_city_id_map
lookup = torch.full((133,), IGNORE, dtype=torch.long, device=DEVICE)

for coco_id, city_id in coco_to_city_id_map.items():
    lookup[coco_id] = city_id

print(f"Mapped Classes: {(lookup != IGNORE).sum().item()}/133 COCO classes mapped to Cityscapes")

