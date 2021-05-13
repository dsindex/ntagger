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

def load_dict(input_path):
    dic = {}
    with open(input_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            toks = line.strip().split()
            _key = toks[0]
            _id = int(toks[1])
            dic[_id] = _key
    return dic

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
