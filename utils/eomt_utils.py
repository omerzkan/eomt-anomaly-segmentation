
import importlib
import sys, os


IMG_SIZE = [640, 640]
DEVICE = "cuda"
N_CITYSCAPES_CLASSES = 19
IGNORE_INDEX = 255
CITYSCAPES_CLASS_NAMES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
            'traffic light', 'traffic sign', 'vegetation', 'terrain',
            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
            'motorcycle', 'bicycle']


# Function for using or not using wandb according to the script and scenarios
def wandb_setup(enable=False):
    
    if enable:
        os.environ["WANDB_MODE"] = "online"
        import wandb
        wandb.login()
        
    else:
        os.environ["WANDB_MODE"] = "disabled"

# This function adds the specified repository path and optional subdirectories
# to the Python module search path
def insert_path(repo_path, subdirs=None):
    
    if subdirs is None:
        subdirs = []
    
    full_path = os.path.join(repo_path, *subdirs)
    
    sys.path.insert(0, full_path)
    
# This function sets the random seed for everything
# Python, Numpy, Pytorch
def setup_seed(seed=42):

    from lightning import seed_everything
    seed_everything(seed, verbose=False)


# function for reading a YAML config file and returning as a dictionary
def read_yaml_config(config_path, sanity_check=False):
    
    import yaml
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    # Sanity check: print the config dict
    if sanity_check:
        print("Sanity Check: This is the loaded config file")
        for key, value in config_dict.items():
            print(f"{key}:{value}\n")
    
    return config_dict
        
# The recursive function that build the objects from the config dict.
def _build_helper(config_dict):
    
    if isinstance(config_dict, dict) and "class_path" in config_dict:
        
        class_path = config_dict["class_path"]
        module_name, class_name = class_path.rsplit(".", 1)
        
        class_imported = getattr(importlib.import_module(module_name), class_name)
        init_args = config_dict.get("init_args", {})
        
        arguments_values_dict = {key: _build_helper(val) for key, val in init_args.items()}
        
        return class_imported(**arguments_values_dict)
    
    if isinstance(config_dict, dict):
        return {key: _build_helper(val) for key, val in config_dict.items()}
    
    if isinstance(config_dict, list):
        return [_build_helper(item) for item in config_dict]
    
    return config_dict

# main function used for building the model.
# It uses the _build_helper function and read_yaml_config function.
# if sanity_check is True, it prints the all the sanity check inclueded in the _build_helper function and read_yaml_config function.
# if checkpoint_path is not None, it loads the checkpoint and prints the missing and unexpected keys if sanity_check is True.

"""
Config Overriders Example Format:
overrides = {
    ('model', 'init_args', 'img_size'): [640, 640],
    ('model', 'init_args', 'num_classes'): 133,
    # etc...
}
This is the path you want to modify the config dict. If you dont want to modify, you can just pass None.
"""
def build_model(config_path, eval_mode=False, config_overriders=None, sanity_check=False, checkpoint_path=None, device="cuda"):
    
    config_dict = read_yaml_config(config_path, sanity_check)
    
    if config_overriders is not None:
        current_dict = config_dict        
        for key_tuple, value in config_overriders.items():
            for key in key_tuple[:-1]:
                if key in current_dict:
                    current_dict = current_dict[key]
                else:
                    raise KeyError(f"Key {key} not found in the config dict.")
            current_dict[key_tuple[-1]] = value
            current_dict = config_dict 
        
        # Sanity check: print the updated config dict
        if sanity_check:
            print("Sanity Check: This is the updated config dict")
            for key, value in config_dict.items():
                print(f"{key}:{value}\n")
    
    model = _build_helper(config_dict["model"])
    
    if sanity_check:
        print("Sanity Check: This is the built model")
        print(f"{type(model).__name__}")
    
    if checkpoint_path is not None:
        import torch
        checkpoints = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        # we loaded to the cpu, then we move to the cuda
        # We loaded the extra code too, as we told that weights_only=False.
        # if we told weights_only = True, then we will only load the state_dict, and we will not load the extra code.
        # we loaded the extra code because we trust the our TAs drive.
        state_dict = checkpoints.get("state_dict", checkpoints)
        # It tries to get the state dict if it is not availabe returns the checkpoints
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        # strict=False means that we will not raise an error if there are missing keys or unexpected keys in the state dict.
        
        if sanity_check:
            print("Sanity Check: This is the checkpoint loading result")
            print(f"Missing keys: {missing}")
            print(f"Unexpected keys: {unexpected}")
    
    model.to(device)
    
    if eval_mode:
        model.eval()
    
    return model

def semantic_inference(model, dataloader, remap_function=None, evaluator=None, device="cuda", description="SemanticInference"):
    
    import torch
    from tqdm import tqdm
    from torch.nn import functional as F
    from torch.amp.autocast_mode import autocast
    
    model.eval()
    
    with torch.no_grad():
        
        for imgs, targets in tqdm(dataloader, desc=description):
            # tqdm shows the progess bar for the inference process
            imgs = [img.to(device) for img in imgs]
            img_sizes = [img.shape[-2: ] for img in imgs]
            
            ground_truth = model.to_per_pixel_targets_semantic(targets, IGNORE_INDEX=IGNORE_INDEX)[0].to(device)
            
            with autocast(dtype=torch.float16, device_type="cuda"):
                
                crops, origins = model.window_imgs_semantic(imgs)
            
                mask_logits_list, class_logits_list = model(crops)
            
                mask_logits = F.interpolate(
                    mask_logits_list[-1], model.img_size, mode="bilinear"
                )
            
                crop_logits = model.to_per_pixel_logits_semantic(
                    mask_logits, class_logits_list[-1]
                )
                
                logits = model.revert_window_logits_semantic(
                    crop_logits, origins, img_sizes
                )
                
                pred = logits[0].argmax(0)
            
            
            if remap_function is not None:
                pred = remap_function(pred)
                
            if evaluator is not None:
                evaluator.update(pred, ground_truth)
        
                

    return evaluator



def print_results(model_name, per_class_iou=None, class_names=None, save_json_path=None):
    
    import json

    print(f"Results for {model_name}:")
    miou = float(per_class_iou.mean())

    print(f"\n========== {model_name} ==========")
    print(f"mIoU: {miou * 100}\n")
    print(f"{'Class':<20} {'IoU (%)':>8}")
    print("-" * 30)
    for name, iou in zip(class_names, per_class_iou):
        print(f"{name:<20} {iou * 100:>7.2f}")

    # Save results to JSON for the report
    results = {
        'mIoU': miou,
        'per_class': dict(zip(class_names, [float(x) for x in per_class_iou])),
    }

    if save_json_path is not None:
        with open(save_json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {save_json_path}")

def compare_result_iou(model_name1, model_name2, per_class_iou1=None, per_class_iou2=None, class_names=None, save_json_path=None):
    
    import json

    miou1 = float(per_class_iou1.mean())
    miou2 = float(per_class_iou2.mean())

    print(f"\n========== Comparison: {model_name1} vs {model_name2} ==========")
    print(f"{'ModelName':<20} {model_name1:>8} {model_name2:>8}")
    print("-" * 30)
    print(f"{'mIoU':<20}: {miou1 * 100:>8} {miou2 * 100:>8}\n")
    
    print(f"{'Class':<20} {model_name1:>8} {model_name2:>8}")
    print("-" * 30)
    for name, iou1, iou2 in zip(class_names, per_class_iou1, per_class_iou2):
        print(f"{name:<20} {iou1 * 100:>7.2f} {iou2 * 100:>7.2f}")

    # Save results to JSON for the report
    results = {
        f'mIoU {model_name1}': miou1,
        f'mIoU {model_name2}': miou2,
        f'per_class {model_name1}': dict(zip(class_names, [float(x) for x in per_class_iou1])),
        f'per_class {model_name2}': dict(zip(class_names, [float(x) for x in per_class_iou2]))
    }

    if save_json_path is not None:    
        with open(save_json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {save_json_path}")