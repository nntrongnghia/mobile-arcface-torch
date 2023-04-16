import importlib
import os.path as osp
import logging
from ptflops import get_model_complexity_info

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