from __future__ import absolute_import, division, print_function

import os
import pdb
import json
import torch

def load_checkpoint(model_path, device='cuda'):
    if device == 'cpu':
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(model_path)
    return checkpoint

def load_config(opt, config_path=None):
    try:
        if not config_path: config_path = opt.config
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        config = dict()
    return config

def to_device(x, device):
    if type(x) != list: # torch.tensor
        x = x.to(device)
    else:               # list of torch.tensor
        for i in range(len(x)):
            x[i] = x[i].to(device)
    return x

def to_numpy(x):
    if type(x) != list: # torch.tensor
        x = x.detach().cpu().numpy()
    else:               # list of torch.tensor
        for i in range(len(x)):
            x[i] = x[i].detach().cpu().numpy()
    return x
