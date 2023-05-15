import importlib
import os.path as osp
import logging
from ptflops import get_model_complexity_info
import torch
from torch.nn import Module, Sequential

def get_config(config_file: str):
    assert 'configs' in config_file, 'config file setting must start with configs'
    if config_file.startswith("./"):
        config_file = config_file[2:]
    # remove .py
    if config_file.endswith(".py"):
        config_file =  config_file[:-3]
    config_path = ".".join(config_file.split("/"))
    return importlib.import_module(config_path).config

def print_config(cfg):
    for key, value in cfg.items():
        num_space = 25 - len(key)
        if(key == 'model'):
            macs, params = get_model_complexity_info(value, (3, 112, 112), print_per_layer_stat=False)
            logging.info(": " + key + " " * num_space + f'{macs} - {params} params')
        else:
            logging.info(": " + key + " " * num_space + str(value))
            
def inference(model: torch.nn.Module, images: torch.Tensor, ref_images: torch.Tensor = None, ref_embs: torch.Tensor = None) -> torch.Tensor:
    # images must have shape [B, C, 112, 112] or [C, 112, 112]
    assert images.shape[-1] == 112, "Image width must be 112px"
    assert images.shape[-2] == 112, "Image height must be 112px"
    if images.shape == 3:
        # extend to [1, C, H, W]
        images = images[None]
    assert ref_images or ref_embs, "Must provide either ref_images or ref_embs for verification"
    if ref_images and ref_embs:
        logging.warning("Only ref_images will be used for verification")
    embs = model(images)
    if ref_images is not None:
        if ref_images.shape == 3:
            ref_images = ref_images[None]
        ref_embs = model(ref_images)
    cosine = torch.cosine_similarity(embs, ref_embs)
    normalized_score = (cosine + 1) / 2 # normalize to range [0, 1]
    return normalized_score


def get_inference_model(backbone):
    class ImageNorm(Module):
        def __init__(self) -> None:
            super().__init__()
            
        def forward(self, img):
            img = img / 255
            img = (img - 0.5) / 0.5
            return img
    
    return Sequential(ImageNorm(),backbone)